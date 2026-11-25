from __future__ import annotations

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
from torch.utils.data import DataLoader

from model.scalar_baseline import ScalarBaseline
from training.augmentation import swap_perspective
from training.config import PathConfig, ScalarBaselineConfig
from training.scalar_dataset import ScalarDataset
from training.turns import load_split

torch.set_float32_matmul_precision("medium")


def build_module(config: ScalarBaselineConfig):
    import lightning as pl
    import torch.nn as nn
    from torch import Tensor
    from torchmetrics import MeanSquaredError
    from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

    model = ScalarBaseline(
        scalar_dim=config.scalar_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    class _Module(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self._model = model
            self._loss_fn = nn.BCEWithLogitsLoss()
            self._train_auroc = BinaryAUROC()
            self._val_auroc = BinaryAUROC()
            self._val_brier = MeanSquaredError()
            self._val_accuracy = BinaryAccuracy()

        def forward(self, batch: dict[str, Tensor]) -> Tensor:
            """Delegates to the underlying ScalarBaseline.

            Args:
                batch: Dict containing 'scalars' tensor.

            Returns:
                Float tensor of shape (B,) with raw logits.
            """
            return self._model(batch)

        def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
            """Computes label-smoothed BCE loss and logs training AUROC.

            Args:
                batch: Collated batch dict from the training DataLoader.
                batch_idx: Position of this batch within the current epoch.

            Returns:
                Scalar loss tensor.
            """
            logits = self(batch)
            targets = batch["target"]
            smoothed = (
                targets * (1.0 - config.label_smoothing) + config.label_smoothing * 0.5
            )
            loss = self._loss_fn(logits, smoothed)
            probs = torch.sigmoid(logits)
            self._train_auroc.update(probs, targets.int())
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=targets.size(0),
            )
            self.log(
                "train_auroc",
                self._train_auroc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=targets.size(0),
            )
            return loss

        def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
            """Computes and logs validation loss, AUROC, Brier score, and accuracy.

            Args:
                batch: Collated batch dict from the validation DataLoader.
                batch_idx: Position of this batch within the current epoch.
            """
            logits = self(batch)
            targets = batch["target"]
            loss = self._loss_fn(logits, targets)
            probs = torch.sigmoid(logits)
            self._val_auroc.update(probs, targets.int())
            self._val_brier.update(probs, targets)
            self._val_accuracy.update(probs, targets.int())
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=targets.size(0),
            )
            self.log(
                "val_auroc",
                self._val_auroc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=targets.size(0),
            )
            self.log(
                "val_brier",
                self._val_brier,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=targets.size(0),
            )
            self.log(
                "val_accuracy",
                self._val_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=targets.size(0),
            )

        def configure_optimizers(self):
            """Builds the Adam optimiser over all model parameters.

            Returns:
                Configured Adam optimiser.
            """
            return torch.optim.Adam(
                self._model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

    return _Module()


def train(config: ScalarBaselineConfig | None = None) -> None:
    """Runs the scalar MLP baseline training pipeline.

    Args:
        config: ScalarBaselineConfig driving all hyperparameters and paths.
            Defaults are used when None is passed.
    """
    if config is None:
        config = ScalarBaselineConfig()

    seed_everything(config.run_seed, workers=True)

    train_df = load_split("train")
    val_df = load_split("validation")
    train_df = pd.concat([train_df, swap_perspective(train_df)], ignore_index=True)

    num_workers = config.num_workers
    train_loader = DataLoader(
        ScalarDataset(train_df),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        ScalarDataset(val_df),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    module = build_module(config)

    callbacks: list[Callback] = [
        RichProgressBar(
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
        ),
        ModelCheckpoint(
            dirpath=PathConfig().checkpoints / f"seed_{config.run_seed}",
            filename="ygo-scalar-mlp-{epoch:02d}-{val_auroc:.4f}",
            monitor="val_auroc",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_auroc", mode="max", patience=10),
    ]

    trainer = Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=10,
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ScalarBaselineConfig()
    cfg.run_seed = args.run_seed
    train(cfg)
