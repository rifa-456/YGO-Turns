from __future__ import annotations

import lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from model.win_predictor import WinPredictor
from training.config import TrainingConfig


class WinPredictorModule(pl.LightningModule):
    """LightningModule wrapping WinPredictor with loss computation and metrics."""

    def __init__(self, model: WinPredictor, config: TrainingConfig) -> None:
        super().__init__()

        self._model = model
        self._config = config

        self._loss_fn = nn.BCEWithLogitsLoss()

        self._train_auroc = BinaryAUROC()

        self._val_auroc = BinaryAUROC()
        self._val_brier = MeanSquaredError()
        self._val_accuracy = BinaryAccuracy()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Delegates to the underlying WinPredictor.

        Args:
            batch: Dict of tensors from TurnsDataset.

        Returns:
            Float tensor of shape (B,) with raw logits.
        """
        return self._model(batch)

    def training_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Computes training loss and metrics.

        Args:
            batch: Collated batch dict from the training DataLoader.
            batch_idx: Position of this batch within the current epoch.

        Returns:
            Scalar BCE loss tensor computed against label-smoothed targets.
        """
        logits = self(batch)
        targets = batch["target"]
        smoothed = (
            targets * (1.0 - self._config.label_smoothing)
            + self._config.label_smoothing * 0.5
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

    def validation_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Computes validation loss and metrics.

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

    def configure_optimizers(self) -> torch.optim.Adam:
        """Builds the Adam optimiser over all model parameters.

        Returns:
            Configured Adam optimiser with L2 weight decay sourced from TrainingConfig.
        """
        return torch.optim.Adam(
            self._model.parameters(),
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
        )
