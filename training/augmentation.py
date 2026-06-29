from __future__ import annotations

import pandas as pd

from data.schema.constants import IS_P1_TURN, TARGET


def swap_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Produces a mirror copy of a turns DataFrame with P1 and P2 swapped.

    Args:
        df: Preprocessed turns DataFrame produced by data.preprocessing.preprocess.

    Returns:
        New DataFrame of the same shape where every p1_* column is renamed to
        its p2_* counterpart and vice versa, IS_P1_TURN is flipped, and TARGET
        is flipped. The index is reset.
    """
    _SKIP: frozenset[str] = frozenset({IS_P1_TURN, TARGET})

    rename: dict[str, str] = {}
    for col in df.columns:
        if col in _SKIP:
            continue
        if col.startswith("p1_"):
            rename[col] = "p2_" + col[3:]
        elif col.startswith("p2_"):
            rename[col] = "p1_" + col[3:]

    swapped = df.rename(columns=rename).copy()
    swapped[IS_P1_TURN] = 1 - swapped[IS_P1_TURN]
    swapped[TARGET] = 1 - swapped[TARGET]
    return swapped.reset_index(drop=True)
