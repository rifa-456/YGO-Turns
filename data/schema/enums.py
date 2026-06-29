from __future__ import annotations

from enum import Enum


class CardPosition(str, Enum):
    ATTACK = "ATK"
    DEFENSE = "DEF"


class PlayerPrefix(str, Enum):
    P1 = "p1_"
    P2 = "p2_"
