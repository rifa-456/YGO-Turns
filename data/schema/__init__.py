from __future__ import annotations

from data.schema.builders import build_player_columns
from data.schema.constants import (
    DUEL_NUMBER,
    IS_P1_TURN,
    MATCH_ID,
    TARGET,
    TURN_NUMBER,
)
from data.schema.enums import PlayerPrefix
from data.schema.flatten import flatten_player_columns

P1 = build_player_columns(PlayerPrefix.P1.value)
P2 = build_player_columns(PlayerPrefix.P2.value)


ALL_COLUMNS = [
    MATCH_ID,
    DUEL_NUMBER,
    TURN_NUMBER,
    IS_P1_TURN,
    *flatten_player_columns(P1),
    *flatten_player_columns(P2),
    TARGET,
]


__all__ = [
    "P1",
    "P2",
    "ALL_COLUMNS",
    "MATCH_ID",
    "DUEL_NUMBER",
    "TURN_NUMBER",
    "IS_P1_TURN",
    "TARGET",
]
