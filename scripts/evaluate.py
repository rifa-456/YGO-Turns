"""Score trained models on the validation and test splits."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from data.schema.constants import MATCH_ID, TARGET
from scripts.train import build_model
from scripts.train_baseline_mlp import build_module
from training.card_lookup import CardLookup
from training.config import Config, PathConfig, ScalarBaselineConfig
from training.dataset import TurnsDataset
from training.module import WinPredictorModule
from training.scalar_dataset import ScalarDataset, build_scalar_array
from training.turns import REVISION, load_split
from training.vocabulary import CardVocabularies

PATHS = PathConfig()
DEFAULT_OUTPUT_PATH = PATHS.artifacts / "results.json"
EVALUATION_SPLITS = ("validation", "test")
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0


def best_checkpoint(folder: Path, prefix: str) -> Path:
    """Returns the checkpoint with the highest validation AUROC in its name.

    Args:
        folder: Folder of one training seed.
        prefix: Start of the model's checkpoint names.

    Raises:
        FileNotFoundError: If the folder has no checkpoint of the model.
    """
    scored = []
    for path in folder.glob(f"{prefix}*.ckpt"):
        match = re.search(r"val_auroc=([0-9.]+[0-9])", path.stem)
        if match:
            scored.append((float(match.group(1)), path.name, path))
    if not scored:
        raise FileNotFoundError(f"no {prefix} checkpoint in {folder}")
    return max(scored)[2]


@torch.inference_mode()
def predict(
    module: torch.nn.Module, dataset: Dataset, batch_size: int = 512
) -> np.ndarray:
    """Returns the predicted P1 win probability of every row.

    Args:
        module: Model that maps a batch dict to logits.
        dataset: Dataset of batch dicts.
        batch_size: Rows per batch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()
    parts = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        moved = {key: value.to(device) for key, value in batch.items()}
        parts.append(torch.sigmoid(module(moved)).cpu().numpy())
    return np.concatenate(parts).reshape(-1)


def score(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Computes AUROC, Brier score and accuracy.

    Args:
        targets: Binary labels.
        probabilities: Predicted P1 win probabilities.
    """
    return {
        "auroc": float(roc_auc_score(targets, probabilities)),
        "brier": float(brier_score_loss(targets, probabilities)),
        "accuracy": float(((probabilities >= 0.5) == targets).mean()),
    }


def auroc_interval(df: pd.DataFrame, predictions: list[np.ndarray]) -> list[float]:
    """Estimates a 95% interval of the mean AUROC by resampling whole matches.

    Args:
        df: Rows the predictions belong to.
        predictions: Predictions of each training seed.

    Returns:
        Lower and upper bound.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    targets = df[TARGET].to_numpy()
    rows = [np.flatnonzero(df[MATCH_ID] == match) for match in df[MATCH_ID].unique()]
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        picked = rng.integers(len(rows), size=len(rows))
        index = np.concatenate([rows[i] for i in picked])
        means.append(
            np.mean([roc_auc_score(targets[index], p[index]) for p in predictions])
        )
    return [float(bound) for bound in np.percentile(means, [2.5, 97.5])]


def summarize(df: pd.DataFrame, predictions: list[np.ndarray]) -> dict:
    """Summarizes the scores of every training seed.

    Args:
        df: Rows the predictions belong to.
        predictions: Predictions of each training seed.
    """
    runs = [score(df[TARGET].to_numpy(), p) for p in predictions]
    summary: dict = {
        metric: {
            "mean": statistics.mean(run[metric] for run in runs),
            "std": statistics.pstdev(run[metric] for run in runs),
        }
        for metric in runs[0]
    }
    summary["auroc"]["interval_95"] = auroc_interval(df, predictions)
    summary["runs"] = runs
    return summary


def logistic_predictions(df: pd.DataFrame) -> list[np.ndarray]:
    """Predicts with the saved logistic regression.

    Args:
        df: Turns DataFrame.
    """
    model = joblib.load(PATHS.logistic_model)
    return [model.predict_proba(build_scalar_array(df))[:, 1]]


def mlp_predictions(df: pd.DataFrame, seeds: int) -> list[np.ndarray]:
    """Predicts with the best scalar MLP checkpoint of every seed.

    Args:
        df: Turns DataFrame.
        seeds: Number of training seeds.
    """
    dataset = ScalarDataset(df)
    predictions = []
    for seed in range(seeds):
        module = build_module(ScalarBaselineConfig())
        path = best_checkpoint(PATHS.checkpoints / f"seed_{seed}", "ygo-scalar-mlp-")
        module.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        predictions.append(predict(module, dataset))
    return predictions


def board_predictions(df: pd.DataFrame, seeds: int) -> list[np.ndarray]:
    """Predicts with the best BoardDSN checkpoint of every seed.

    Args:
        df: Turns DataFrame.
        seeds: Number of training seeds.
    """
    config = Config()
    cards = pd.read_parquet(PATHS.cards_parquet)
    dataset = TurnsDataset(df, CardLookup(cards, CardVocabularies.from_cards_df(cards)))
    predictions = []
    for seed in range(seeds):
        module = WinPredictorModule(build_model(config), config.training)
        path = best_checkpoint(PATHS.checkpoints / f"seed_{seed}", "ygo-epoch")
        module.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        predictions.append(predict(module, dataset))
    return predictions


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "mlp", "board"],
        choices=["logistic", "mlp", "board"],
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    results: dict = {"dataset": REVISION, "models": {}}
    for name in args.models:
        results["models"][name] = {}
        for split in EVALUATION_SPLITS:
            df = load_split(split)
            if name == "logistic":
                predictions = logistic_predictions(df)
            elif name == "mlp":
                predictions = mlp_predictions(df, args.seeds)
            else:
                predictions = board_predictions(df, args.seeds)
            summary = summarize(df, predictions)
            results["models"][name][split] = summary
            auroc = summary["auroc"]
            print(f"{name} {split}: {auroc['mean']:.4f} ± {auroc['std']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
