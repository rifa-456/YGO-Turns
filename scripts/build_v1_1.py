"""Build YGO-Turns v1.1 from the v1.0 file and the corrections list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema.constants import (
    DUEL_NUMBER,
    IS_P1_TURN,
    MATCH_ID,
    TARGET,
    TOKEN_NAME,
    TURN_NUMBER,
)
from training.turns import SPLITS, download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORRECTIONS_PATH = PROJECT_ROOT / "corrections" / "v1.1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "v1.1"
SPLIT_SEED = 42
HELD_OUT_FRACTION = 0.2
DUEL_KEY = [MATCH_ID, DUEL_NUMBER]


def _in_duels(df: pd.DataFrame, duels: list[list[int]]) -> pd.Series:
    """Marks the rows of the given duels.

    Args:
        df: Turns DataFrame.
        duels: Pairs of match id and duel number.

    Raises:
        ValueError: If a duel is not in the data.
    """
    keys = {tuple(duel) for duel in duels}
    rows = list(zip(df[MATCH_ID], df[DUEL_NUMBER], strict=True))
    missing = keys - set(rows)
    if missing:
        raise ValueError(f"duels not in the data: {sorted(missing)}")
    return pd.Series([row in keys for row in rows], index=df.index)


def remove_rows(
    df: pd.DataFrame, corrections: dict
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Removes repeated matches, merged duels and duels without a known winner.

    Args:
        df: Turns DataFrame.
        corrections: Contents of the corrections file.

    Returns:
        The remaining rows, and the number of rows removed for each reason.
    """
    repeated = [entry["remove"] for entry in corrections["duplicate_matches"]]
    masks = {
        "duplicate_matches": df[MATCH_ID].isin(repeated),
        "merged_duels": _in_duels(df, corrections["merged_duels"]),
        "unknown_winner": _in_duels(df, corrections["unknown_winner"]),
    }
    removed: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    for reason, mask in masks.items():
        removed[reason] = int((mask & keep).sum())
        keep &= ~mask
    return df[keep].reset_index(drop=True), removed


def fix_winners(df: pd.DataFrame, corrections: dict) -> tuple[pd.DataFrame, int]:
    """Sets the winner of the duels that ended by a player leaving.

    Args:
        df: Turns DataFrame.
        corrections: Contents of the corrections file.

    Returns:
        The corrected rows, and the number of rows whose label changed.
    """
    df = df.copy()
    changed = 0
    for entry in corrections["recovered_winners"]:
        mask = _in_duels(df, [[entry[MATCH_ID], entry[DUEL_NUMBER]]])
        changed += int((df.loc[mask, TARGET] != entry[TARGET]).sum())
        df.loc[mask, TARGET] = entry[TARGET]
    return df, changed


def rename_tokens(df: pd.DataFrame, corrections: dict) -> tuple[pd.DataFrame, int]:
    """Replaces token picture codes with one token name.

    Args:
        df: Turns DataFrame.
        corrections: Contents of the corrections file.

    Returns:
        The corrected rows, and the number of cells that changed.
    """
    df = df.copy()
    codes = corrections["token_names"]
    changed = 0
    for column in df.columns:
        if "_zone_" in column and column.endswith("_name"):
            mask = df[column].isin(codes)
            changed += int(mask.sum())
            df.loc[mask, column] = TOKEN_NAME
    return df, changed


def split_matches(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Assigns v1.0 match ids to splits, keeping the validation matches of v1.0.

    Args:
        df: v1.0 turns DataFrame.

    Returns:
        Match ids of each split.
    """
    matches = df[MATCH_ID].unique().copy()
    n_held_out = int(len(matches) * HELD_OUT_FRACTION)
    np.random.default_rng(SPLIT_SEED).shuffle(matches)
    rest = np.sort(matches[n_held_out:])
    np.random.default_rng(SPLIT_SEED).shuffle(rest)
    return {
        "train": rest[n_held_out:],
        "validation": matches[:n_held_out],
        "test": rest[:n_held_out],
    }


def check(df: pd.DataFrame) -> None:
    """Checks the rules the data must keep.

    Args:
        df: Corrected turns DataFrame.

    Raises:
        ValueError: If a rule is broken.
    """
    if df.duplicated([*DUEL_KEY, TURN_NUMBER]).any():
        raise ValueError("a turn appears twice")
    duels = df.groupby(DUEL_KEY)
    if (duels[TARGET].nunique() != 1).any():
        raise ValueError("a duel has two winners")
    if (duels[TURN_NUMBER].max() + 1 != duels.size()).any():
        raise ValueError("a duel misses a turn")
    if not (df.loc[df[TURN_NUMBER] == 0, IS_P1_TURN] == 1).all():
        raise ValueError("p1 does not take the first turn of a duel")


def build(source: Path, output_dir: Path) -> dict:
    """Writes one Parquet file per split.

    Args:
        source: v1.0 Parquet file.
        output_dir: Folder for the split files.

    Returns:
        Counts of the rows and cells changed, and the size of each split.
    """
    schema = pq.read_schema(source).remove_metadata()
    original = pd.read_parquet(source)
    corrections = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))

    splits = split_matches(original)
    df, removed = remove_rows(original, corrections)
    df, relabelled = fix_winners(df, corrections)
    df, renamed = rename_tokens(df, corrections)
    check(df)

    report: dict = {
        "rows": {"v1.0": len(original), "v1.1": len(df)},
        "removed_rows": removed,
        "relabelled_rows": relabelled,
        "renamed_token_cells": renamed,
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = df[df[MATCH_ID].isin(splits[name])].reset_index(drop=True)
        table = pa.Table.from_pandas(part, schema=schema, preserve_index=False)
        pq.write_table(
            table.replace_schema_metadata(None),
            output_dir / f"{name}.parquet",
            compression="zstd",
        )
        report["splits"][name] = {
            "matches": int(part[MATCH_ID].nunique()),
            "duels": int(part.groupby(DUEL_KEY).ngroups),
            "rows": len(part),
        }
    return report


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="v1.0 file; downloaded by default")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    source = args.source or download("ygo-turns.parquet", revision="v1.0")
    print(json.dumps(build(source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
