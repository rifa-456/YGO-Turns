from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.schema import IS_P1_TURN, P1, P2
from data.schema.constants import PAD_TOKEN, TARGET

_LP_NORM: float = 16000.0
_DECK_NORM: float = 60.0
_HAND_NORM: float = 15.0
_SIZE_NORM: float = 15.0
_ZONE_COUNT_NORM: float = 5.0
_TOTAL_ATK_NORM: float = 15000.0
_TOTAL_DEF_NORM: float = 15000.0

BASELINE_SCALAR_DIM: int = 23


def _count_occupied(df: pd.DataFrame, name_cols: list[str]) -> np.ndarray:
    """Counts non-PAD, non-null values across a set of name columns row-wise.

    Args:
        df: Preprocessed turns DataFrame.
        name_cols: List of column names containing card name strings.

    Returns:
        Float32 ndarray of shape (N,) with the count of occupied slots per row.
    """
    result = np.zeros(len(df), dtype=np.float32)
    for col in name_cols:
        result += (df[col].notna() & (df[col] != PAD_TOKEN)).astype(np.float32).values
    return result


def _sum_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Sums numeric columns row-wise, coercing errors and filling NaN with zero.

    Args:
        df: Preprocessed turns DataFrame.
        cols: List of numeric column names to sum.

    Returns:
        Float32 ndarray of shape (N,) with per-row sums.
    """
    result = np.zeros(len(df), dtype=np.float32)
    for col in cols:
        result += (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0.0)
            .values.astype(np.float32)
        )
    return result


def _sum_bool(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Sums boolean columns row-wise, treating NaN as False.

    Args:
        df: Preprocessed turns DataFrame.
        cols: List of boolean column names to sum.

    Returns:
        Float32 ndarray of shape (N,) with per-row sums.
    """
    result = np.zeros(len(df), dtype=np.float32)
    for col in cols:
        result += df[col].fillna(False).astype(np.float32).values
    return result


def build_scalar_array(df: pd.DataFrame) -> np.ndarray:
    """Derives the full 23-dimensional scalar feature matrix from a turns DataFrame.

    No card identity or semantic information is used. All features are normalised
    to approximately [0, 1].

    Feature order (23 total):
        0  p1_life_points     / 16000
        1  p2_life_points     / 16000
        2  is_p1_turn         (binary)
        3  p1_hand_size       / 15
        4  p2_hand_size       / 15
        5  p1_deck_size       / 60
        6  p2_deck_size       / 60
        7  p1_gy_size         / 15
        8  p2_gy_size         / 15
        9  p1_banished_size   / 15
        10 p2_banished_size   / 15
        11 p1_extra_size      / 15
        12 p2_extra_size      / 15
        13 p1_monsters_on_field / 5
        14 p2_monsters_on_field / 5
        15 p1_spells_on_field   / 5
        16 p2_spells_on_field   / 5
        17 p1_total_atk       / 15000
        18 p2_total_atk       / 15000
        19 p1_total_defense   / 15000
        20 p2_total_defense   / 15000
        21 p1_monsters_face_down / 5
        22 p2_monsters_face_down / 5

    Args:
        df: Preprocessed turns DataFrame produced by data.preprocessing.preprocess.

    Returns:
        Float32 ndarray of shape (N, BASELINE_SCALAR_DIM).
    """
    cols: list[np.ndarray] = [
        df[P1.life_points].astype(np.float32).values / _LP_NORM,
        df[P2.life_points].astype(np.float32).values / _LP_NORM,
        df[IS_P1_TURN].astype(np.float32).values,
        df[P1.hand_size].astype(np.float32).values / _HAND_NORM,
        df[P2.hand_size].astype(np.float32).values / _HAND_NORM,
        df[P1.deck_size].astype(np.float32).values / _DECK_NORM,
        df[P2.deck_size].astype(np.float32).values / _DECK_NORM,
        df[P1.gy_size].astype(np.float32).values / _SIZE_NORM,
        df[P2.gy_size].astype(np.float32).values / _SIZE_NORM,
        df[P1.banished_size].astype(np.float32).values / _SIZE_NORM,
        df[P2.banished_size].astype(np.float32).values / _SIZE_NORM,
        df[P1.extra_size].astype(np.float32).values / _SIZE_NORM,
        df[P2.extra_size].astype(np.float32).values / _SIZE_NORM,
        _count_occupied(df, P1.monster_zones.name) / _ZONE_COUNT_NORM,
        _count_occupied(df, P2.monster_zones.name) / _ZONE_COUNT_NORM,
        _count_occupied(df, P1.spell_zones.name) / _ZONE_COUNT_NORM,
        _count_occupied(df, P2.spell_zones.name) / _ZONE_COUNT_NORM,
        _sum_numeric(df, P1.monster_zones.atk) / _TOTAL_ATK_NORM,
        _sum_numeric(df, P2.monster_zones.atk) / _TOTAL_ATK_NORM,
        _sum_numeric(df, P1.monster_zones.defense) / _TOTAL_DEF_NORM,
        _sum_numeric(df, P2.monster_zones.defense) / _TOTAL_DEF_NORM,
        _sum_bool(df, P1.monster_zones.face_down) / _ZONE_COUNT_NORM,
        _sum_bool(df, P2.monster_zones.face_down) / _ZONE_COUNT_NORM,
    ]

    return np.column_stack(cols)


class ScalarDataset(Dataset):
    """PyTorch Dataset emitting only scalar features and the win target."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Precomputes the scalar feature matrix and target vector.

        Args:
            df: Preprocessed turns DataFrame produced by data.preprocessing.preprocess.
        """
        self._scalars = build_scalar_array(df)
        self._targets = df[TARGET].astype(np.float32).values

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Returns the scalar feature dict for a single turn.

        Args:
            idx: Integer row index.

        Returns:
            Dict with keys 'scalars' (float32 tensor of shape (BASELINE_SCALAR_DIM,))
            and 'target' (scalar float32 tensor).
        """
        return {
            "scalars": torch.from_numpy(self._scalars[idx].copy()),
            "target": torch.tensor(self._targets[idx]),
        }
