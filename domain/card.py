from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from domain.enums import (
    Attribute,
    CardCategory,
    MonsterCategory,
    MonsterRace,
    SpellType,
    TrapType,
)


@dataclass(frozen=True)
class Card:
    """Base immutable domain card."""

    id: int
    name: str
    text: str
    category: CardCategory


@dataclass(frozen=True)
class MonsterCard(Card):
    """Internal monster card representation."""

    category: Literal[CardCategory.MONSTER]
    monster_category: MonsterCategory
    attribute: Attribute
    race: MonsterRace
    level: int
    atk: int | None
    defense: int | None
    is_effect: bool


@dataclass(frozen=True)
class SpellCard(Card):
    """Internal spell card representation."""

    category: Literal[CardCategory.SPELL]
    spell_type: SpellType


@dataclass(frozen=True)
class TrapCard(Card):
    """Internal trap card representation."""

    category: Literal[CardCategory.TRAP]
    trap_type: TrapType


DomainCard = MonsterCard | SpellCard | TrapCard
