from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor

from training.vocabulary import CardVocabularies

_EFFECT_DIM: int = 32


@dataclass(frozen=True)
class CardTensors:
    effect_embedding: Tensor
    categorical_indices: Tensor


class CardLookup:
    """Pre-built mapping from card name to its feature tensors."""

    def __init__(self, df: pd.DataFrame, vocabs: CardVocabularies) -> None:
        self._pad = _build_pad_tensors()
        self._entries: dict[str, CardTensors] = {}
        self._build(df, vocabs)

    def get(self, name: str) -> CardTensors:
        """Returns feature tensors for a card name.

        Args:
            name: Card name as it appears in cards.parquet.

        Returns:
            CardTensors for the card, or zero-valued PAD tensors when the name
            is not found in the lookup.
        """
        return self._entries.get(name, self._pad)

    def _build(self, df: pd.DataFrame, vocabs: CardVocabularies) -> None:
        for _, row in df.iterrows():
            self._entries[row["name"]] = _build_card_tensors(row, vocabs)


def _build_pad_tensors() -> CardTensors:
    return CardTensors(
        effect_embedding=torch.zeros(_EFFECT_DIM),
        categorical_indices=torch.zeros(2, dtype=torch.long),
    )


def _build_card_tensors(row: pd.Series, vocabs: CardVocabularies) -> CardTensors:
    """Converts a single cards DataFrame row into encoded feature tensors.

    Args:
        row: One row from the cards DataFrame.
        vocabs: Vocabulary mappings used to convert categorical strings to indices.

    Returns:
        CardTensors containing the PCA-compressed effect embedding and
        categorical indices for fine_category and level.
    """
    effect_embedding = torch.tensor(row["effect_embedding"], dtype=torch.float32)
    is_monster = pd.notna(row.get("attribute"))

    fine_cat_idx = vocabs.fine_category.get(str(row["fine_category"]), 0)
    level_idx = (
        vocabs.level.get(str(int(row["level"])), 0)
        if is_monster and pd.notna(row["level"])
        else 0
    )

    categorical_indices = torch.tensor([fine_cat_idx, level_idx], dtype=torch.long)

    return CardTensors(effect_embedding, categorical_indices)
