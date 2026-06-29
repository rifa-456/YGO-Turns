from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from model.card_encoder import CardEncoder
from model.win_predictor import WinPredictor
from training.config import Config
from training.datamodule import TurnsDataModule
from training.module import WinPredictorModule
from training.vocabulary import CardVocabularies

torch.set_float32_matmul_precision("medium")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train(config: Config | None = None) -> None:
    """Runs the full training pipeline.

    Args:
        config: Config instance driving all hyperparameters and paths.
            Defaults are used when None is passed.
    """
    if config is None:
        config = Config()

    seed_everything(config.training.run_seed, workers=True)

    vocab_sizes = CardVocabularies.from_cards_df(
        pd.read_parquet(config.paths.cards_parquet)
    ).sizes

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    card_encoder = CardEncoder(
        effect_embedding_dim=config.model.effect_embedding_dim,
        fine_category_vocab_size=vocab_sizes["fine_category"],
        level_vocab_size=vocab_sizes["level"],
        fine_category_embedding_dim=config.model.fine_category_embedding_dim,
        level_embedding_dim=config.model.level_embedding_dim,
        card_vector_dim=config.model.card_vector_dim,
    )

    model = WinPredictor(
        card_encoder=card_encoder,
        zone_dim=config.model.zone_dim,
        card_vector_dim=config.model.card_vector_dim,
        deck_slots=config.model.deck_slots,
        scalar_dim=config.model.scalar_dim,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        dropout=config.model.dropout,
        card_dropout_p=config.model.card_dropout_p,
        zone_dropout_p=config.model.zone_dropout_p,
    )

    module = WinPredictorModule(model, config.training)
    datamodule = TurnsDataModule(config)

    callbacks: list[Callback] = [
        progress_bar,
        ModelCheckpoint(
            dirpath=PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / f"seed_{config.training.run_seed}",
            filename="ygo-{epoch:02d}-{val_auroc:.4f}",
            monitor="val_auroc",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val_auroc",
            mode="max",
            patience=10,
        ),
    ]

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        log_every_n_steps=10,
        callbacks=callbacks,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    cfg.training.run_seed = args.run_seed
    train(cfg)
