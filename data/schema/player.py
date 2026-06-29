from __future__ import annotations

from dataclasses import dataclass

from data.schema.zones import (
    FieldSpellZoneColumns,
    MonsterZoneColumns,
    SpellZoneColumns,
)


@dataclass(frozen=True)
class PlayerColumns:
    life_points: str

    hand_size: str
    hand: list[str]

    deck_size: str
    deck_top: list[str]

    gy_size: str
    gy: list[str]

    banished_size: str
    banished: list[str]

    extra_size: str
    extra: list[str]

    monster_zones: MonsterZoneColumns
    spell_zones: SpellZoneColumns
    field_spell_zone: FieldSpellZoneColumns
