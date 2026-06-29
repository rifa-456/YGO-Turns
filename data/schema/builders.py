from __future__ import annotations

from data.schema.constants import (
    BANISHED_MAX,
    DECK_TOP_K,
    EQUIPS_MAX,
    EXTRA_MAX,
    GY_MAX,
    HAND_MAX,
    MONSTER_ZONE_COUNT,
    SPELL_ZONE_COUNT,
)
from data.schema.player import PlayerColumns
from data.schema.zones import (
    FieldSpellZoneColumns,
    MonsterZoneColumns,
    SpellZoneColumns,
)


def col(prefix: str, suffix: str) -> str:
    return f"{prefix}{suffix}"


def indexed(prefix: str, base: str, count: int) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive")

    return [f"{prefix}{base}_{i}" for i in range(count)]


def build_monster_zones(prefix: str) -> MonsterZoneColumns:
    base = f"{prefix}monster_zone"

    return MonsterZoneColumns(
        name=[f"{base}_{i}_name" for i in range(MONSTER_ZONE_COUNT)],
        atk=[f"{base}_{i}_atk" for i in range(MONSTER_ZONE_COUNT)],
        defense=[f"{base}_{i}_def" for i in range(MONSTER_ZONE_COUNT)],
        position=[f"{base}_{i}_position" for i in range(MONSTER_ZONE_COUNT)],
        face_down=[f"{base}_{i}_face_down" for i in range(MONSTER_ZONE_COUNT)],
        counters=[f"{base}_{i}_counters" for i in range(MONSTER_ZONE_COUNT)],
        equips=[
            [f"{base}_{i}_equip_{j}" for j in range(EQUIPS_MAX)]
            for i in range(MONSTER_ZONE_COUNT)
        ],
        equip_sizes=[f"{base}_{i}_equip_size" for i in range(MONSTER_ZONE_COUNT)],
    )


def build_spell_zones(prefix: str) -> SpellZoneColumns:
    base = f"{prefix}spell_zone"

    return SpellZoneColumns(
        name=[f"{base}_{i}_name" for i in range(SPELL_ZONE_COUNT)],
        face_down=[f"{base}_{i}_face_down" for i in range(SPELL_ZONE_COUNT)],
        counters=[f"{base}_{i}_counters" for i in range(SPELL_ZONE_COUNT)],
    )


def build_field_spell_zone(prefix: str) -> FieldSpellZoneColumns:
    """Builds column name descriptors for the field spell zone of one player.

    Args:
        prefix: Column name prefix, e.g. 'p1_' or 'p2_'.

    Returns:
        FieldSpellZoneColumns with name and counters column name strings.
    """
    base = f"{prefix}field_spell_zone"
    return FieldSpellZoneColumns(
        name=f"{base}_name",
        counters=f"{base}_counters",
    )


def build_player_columns(prefix: str) -> PlayerColumns:
    """Builds the full set of column name descriptors for one player.

    Args:
        prefix: Column name prefix, e.g. 'p1_' or 'p2_'.

    Returns:
        PlayerColumns instance with all column name strings populated.
    """
    return PlayerColumns(
        life_points=col(prefix, "life_points"),
        hand_size=col(prefix, "hand_size"),
        hand=indexed(prefix, "hand", HAND_MAX),
        deck_size=col(prefix, "deck_size"),
        deck_top=indexed(prefix, "deck_top", DECK_TOP_K),
        gy_size=col(prefix, "gy_size"),
        gy=indexed(prefix, "gy", GY_MAX),
        banished_size=col(prefix, "banished_size"),
        banished=indexed(prefix, "banished", BANISHED_MAX),
        extra_size=col(prefix, "extra_size"),
        extra=indexed(prefix, "extra", EXTRA_MAX),
        monster_zones=build_monster_zones(prefix),
        spell_zones=build_spell_zones(prefix),
        field_spell_zone=build_field_spell_zone(prefix),
    )
