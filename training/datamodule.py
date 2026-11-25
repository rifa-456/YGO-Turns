from __future__ import annotations

import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader

from training.augmentation import swap_perspective
from training.card_lookup import CardLookup
from training.config import Config
from training.dataset import SCALAR_DIM, TurnsDataset
from training.turns import load_split
from training.vocabulary import CardVocabularies


class TurnsDataModule(pl.LightningDataModule):
    """LightningDataModule with the train and validation splits."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._train_dataset: TurnsDataset | None = None
        self._val_dataset: TurnsDataset | None = None
        self.vocabs: CardVocabularies | None = None

    def setup(self, stage: str | None = None) -> None:
        """Loads data artifacts, builds vocabularies, and creates train/val datasets.

        Args:
            stage: Lightning stage identifier. Setup is skipped if datasets are
                already initialised.

        Raises:
            AssertionError: If ModelConfig.scalar_dim does not match the number of
                scalar columns defined in TurnsDataset.
        """
        if self._train_dataset is not None:
            return

        assert self._config.model.scalar_dim == SCALAR_DIM, (
            f"ModelConfig.scalar_dim={self._config.model.scalar_dim} does not match "
            f"dataset SCALAR_DIM={SCALAR_DIM}. Update ModelConfig to match _SCALAR_COLS."
        )

        cards_df = pd.read_parquet(self._config.paths.cards_parquet)

        self.vocabs = CardVocabularies.from_cards_df(cards_df)
        card_lookup = CardLookup(cards_df, self.vocabs)

        train_df = load_split("train")
        val_df = load_split("validation")

        if self._config.training.augment_swap:
            train_df = pd.concat(
                [train_df, swap_perspective(train_df)],
                ignore_index=True,
            )

        self._train_dataset = TurnsDataset(train_df, card_lookup)
        self._val_dataset = TurnsDataset(val_df, card_lookup)

    def train_dataloader(self) -> DataLoader:
        """Returns the shuffled training DataLoader.

        Returns:
            DataLoader over the training split.
        """
        num_workers = self._config.training.num_workers
        return DataLoader(
            self._train_dataset,
            batch_size=self._config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the sequential validation DataLoader.

        Returns:
            DataLoader over the validation split.
        """
        num_workers = self._config.training.num_workers
        return DataLoader(
            self._val_dataset,
            batch_size=self._config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
