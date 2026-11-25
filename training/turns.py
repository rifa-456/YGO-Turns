from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "Rifa456/YGO-Turns"
REVISION = "v1.1"
SPLITS = ("train", "validation", "test")
DATA_DIR_VARIABLE = "YGO_TURNS_DIR"


def download(filename: str, revision: str = REVISION) -> Path:
    """Returns the local path of a dataset file, downloading it when needed.

    Args:
        filename: Path of the file inside the Hugging Face repository.
        revision: Dataset version to read.
    """
    return Path(
        hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=revision)
    )


def split_path(split: str) -> Path:
    """Returns the path of one split's Parquet file.

    Files are read from the folder named by the YGO_TURNS_DIR environment
    variable when it is set, and from Hugging Face otherwise.

    Args:
        split: One of SPLITS.

    Raises:
        ValueError: If the split is unknown.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    local = os.environ.get(DATA_DIR_VARIABLE)
    if local:
        return Path(local) / f"{split}.parquet"
    return download(f"data/{split}.parquet")


def load_split(split: str) -> pd.DataFrame:
    """Loads one split as a DataFrame.

    Args:
        split: One of SPLITS.
    """
    return pd.read_parquet(split_path(split))
