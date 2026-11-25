from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import P1, P2
from data.schema.constants import PAD_TOKEN


def _card_name_columns() -> list[str]:
    """Returns all flat column names that hold card name strings for both players.

    Returns:
        Ordered list of column name strings covering hand, deck top, GY,
        banished, extra deck, monster zone names, spell zone names, and
        all equip slots for both players.
    """
    cols: list[str] = []
    for player in (P1, P2):
        cols += player.hand
        cols += player.deck_top
        cols += player.gy
        cols += player.banished
        cols += player.extra
        cols += player.monster_zones.name
        cols += player.spell_zones.name
        cols += [name for slot in player.monster_zones.equips for name in slot]
    return cols


def collect_card_names(turns_paths: list[Path]) -> set[str]:
    """Derives the set of all card names that appear in turns parquet files.

    Reads only the card name columns, flattens them into a single array, and
    returns the unique non-padding values. This is used to restrict the cards
    parquet to only cards that actually appear in the training data.

    Args:
        turns_paths: Paths to the turns parquet files.

    Returns:
        Set of card name strings, excluding PAD_TOKEN and null values.
    """
    cols = _card_name_columns()
    df = pd.concat([pd.read_parquet(path, columns=cols) for path in turns_paths])
    flat = df.values.ravel()
    return {
        str(v)
        for v in pd.unique(flat)
        if v is not None
        and str(v) != PAD_TOKEN
        and not (isinstance(v, float) and np.isnan(v))
    }
