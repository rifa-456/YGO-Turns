from __future__ import annotations

import pandas as pd

from data.schema.constants import PAD_TOKEN


class CardVocabularies:
    """Categorical feature vocabularies derived from the cards dataset."""

    def __init__(
        self,
        fine_category: dict[str, int],
        level: dict[str, int],
    ) -> None:
        """Initialises the vocabulary with pre-built category-to-index mappings.

        Args:
            fine_category: Vocabulary mapping fine-grained card category strings to indices.
            level: Vocabulary mapping monster level strings to indices.
        """
        self.fine_category = fine_category
        self.level = level

    @classmethod
    def from_cards_df(cls, df: pd.DataFrame) -> CardVocabularies:
        """Builds all vocabularies from a cards DataFrame.

        Args:
            df: Cards DataFrame loaded from cards.parquet.

        Returns:
            CardVocabularies instance with mappings for every retained categorical feature.
        """
        return cls(
            fine_category=_build_vocab(
                df["fine_category"].dropna().astype(str).tolist()
            ),
            level=_build_vocab(df["level"].dropna().astype(int).astype(str).tolist()),
        )

    @property
    def sizes(self) -> dict[str, int]:
        """Returns the size of each vocabulary.

        Returns:
            Dict mapping each feature name to its vocabulary size, including
            the PAD index at 0.
        """
        return {
            "fine_category": len(self.fine_category),
            "level": len(self.level),
        }


def _build_vocab(values: list[str]) -> dict[str, int]:
    """Builds a string-to-int vocabulary with PAD_TOKEN fixed at index 0.

    Args:
        values: Flat list of observed string values; duplicates are collapsed.

    Returns:
        Dict mapping every unique value (plus PAD_TOKEN) to a stable integer index.
    """
    vocab: dict[str, int] = {PAD_TOKEN: 0}
    for v in sorted(set(values)):
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab
