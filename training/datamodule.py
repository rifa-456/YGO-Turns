from __future__ import annotations

import lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.schema.constants import MATCH_ID
from training.augmentation import swap_perspective
from training.card_lookup import CardLookup
from training.config import Config
from training.dataset import SCALAR_DIM, TurnsDataset
from training.vocabulary import CardVocabularies


class TurnsDataModule(pl.LightningDataModule):
    """LightningDataModule with match-level train/val split."""

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
        turns_df = pd.read_parquet(self._config.paths.turns_parquet)

        self.vocabs = CardVocabularies.from_cards_df(cards_df)
        card_lookup = CardLookup(cards_df, self.vocabs)

        train_df, val_df = _split_by_match_id(
            turns_df,
            val_fraction=self._config.training.val_fraction,
            seed=self._config.training.seed,
        )

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


def _split_by_duel(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a turns DataFrame along duel boundaries with no turn leakage.

    Args:
        df: Full preprocessed turns DataFrame.
        val_fraction: Fraction of unique duels assigned to validation.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_df, val_df), each with a clean integer index.
    """
    duel_keys = df[[MATCH_ID, "duel_number"]].drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    shuffled = duel_keys.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(
        drop=True
    )

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_keys = shuffled.iloc[:n_val]

    val_mask = df.set_index([MATCH_ID, "duel_number"]).index.isin(
        pd.MultiIndex.from_frame(val_keys)
    )

    return (
        df[~val_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
    )


def _split_by_match_id(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a turns DataFrame along match boundaries with no row leakage.

    Args:
        df: Full preprocessed turns DataFrame.
        val_fraction: Fraction of unique match IDs assigned to validation.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_df, val_df), each with a clean integer index.
    """
    match_ids = df[MATCH_ID].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(match_ids)

    n_val = max(1, int(len(match_ids) * val_fraction))
    val_ids = set(match_ids[:n_val])

    is_val = df[MATCH_ID].isin(val_ids)
    return (
        df[~is_val].reset_index(drop=True),
        df[is_val].reset_index(drop=True),
    )
