from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from data.cards.models import (
    RawMonsterCard,
    RawSpellCard,
    RawTrapCard,
)
from data.cards.schema import CARDS_SCHEMA
from data.preprocessing.cards import derive_fine_category
from domain.enums import MonsterCategory


def to_record(
    raw: RawMonsterCard | RawSpellCard | RawTrapCard,
    embedding: np.ndarray,
) -> dict:
    """Builds flat parquet-compatible record."""

    fine_category = derive_fine_category(raw)

    base: dict = {
        "id": raw.id,
        "name": raw.name,
        "effect_text": raw.effect_text,
        "category": raw.category,
        "fine_category": fine_category.value,
        "monster_category": None,
        "spell_type": None,
        "trap_type": None,
        "attribute": None,
        "level": None,
        "atk": None,
        "defense": None,
        "monster_type": None,
        "is_effect": None,
        "effect_embedding": embedding.tolist(),
    }

    if isinstance(raw, RawMonsterCard):
        base.update(
            {
                "monster_category": raw.monster_category.value,
                "attribute": raw.attribute.value,
                "level": raw.level,
                "atk": raw.atk,
                "defense": raw.defense,
                "monster_type": raw.monster_type.value,
                "is_effect": raw.monster_category != MonsterCategory.NORMAL,
            }
        )

    elif isinstance(raw, RawSpellCard):
        base["spell_type"] = raw.spell_type.value

    else:
        base["trap_type"] = raw.trap_type.value

    return base


def build_arrow_table(
    raw_cards: list[RawMonsterCard | RawSpellCard | RawTrapCard],
    embeddings: np.ndarray,
) -> pa.Table:
    """Builds Arrow table from cards and embeddings."""

    records = [
        to_record(raw, embedding) for raw, embedding in zip(raw_cards, embeddings)
    ]

    columns = {field.name: [r[field.name] for r in records] for field in CARDS_SCHEMA}

    return pa.table(columns, schema=CARDS_SCHEMA)


def write_cards_parquet(
    raw_cards: list[RawMonsterCard | RawSpellCard | RawTrapCard],
    embeddings: np.ndarray,
    output_path: Path,
) -> None:
    """Writes cards parquet dataset."""

    table = build_arrow_table(raw_cards, embeddings)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = duckdb.connect()
    conn.register("cards", table)
    conn.execute(
        f"""
        COPY cards
        TO '{output_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )
    conn.close()
