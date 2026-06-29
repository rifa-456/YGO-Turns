from __future__ import annotations

from data.schema.player import PlayerColumns


def flatten_player_columns(player: PlayerColumns) -> list[str]:
    """Returns a flat ordered list of all column names for one player.

    Args:
        player: PlayerColumns instance describing all column names for the player.

    Returns:
        Ordered list of column name strings suitable for DataFrame selection.
    """
    columns = [
        player.life_points,
        player.hand_size,
        *player.hand,
        player.deck_size,
        *player.deck_top,
        player.gy_size,
        *player.gy,
        player.banished_size,
        *player.banished,
        player.extra_size,
        *player.extra,
    ]

    columns.extend(player.monster_zones.name)
    columns.extend(player.monster_zones.atk)
    columns.extend(player.monster_zones.defense)
    columns.extend(player.monster_zones.position)
    columns.extend(player.monster_zones.face_down)
    columns.extend(player.monster_zones.counters)
    for zone_equips in player.monster_zones.equips:
        columns.extend(zone_equips)
    columns.extend(player.monster_zones.equip_sizes)

    columns.extend(player.spell_zones.name)
    columns.extend(player.spell_zones.face_down)
    columns.extend(player.spell_zones.counters)

    columns.append(player.field_spell_zone.name)
    columns.append(player.field_spell_zone.counters)

    return columns
