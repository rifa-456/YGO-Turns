from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from domain.enums import (
    Attribute,
    CardCategory,
    MonsterCategory,
    MonsterRace,
    SpellType,
    TrapType,
)


class _RawCardBase(BaseModel):
    model_config = {"populate_by_name": True}

    id: int
    name: str = Field(alias="n")
    effect_text: str = Field(alias="e")
    category: str = Field(alias="c")


def parse_stat(value: str) -> int | None:
    """Parses ATK/DEF stat values."""

    if value == "?":
        return None

    return int(value)


class RawMonsterCard(_RawCardBase):
    """Raw monster card entry."""

    category: Literal[CardCategory.MONSTER] = Field(alias="c")
    monster_category: MonsterCategory = Field(alias="mc")
    attribute: Attribute = Field(alias="att")
    level: int = Field(alias="lvl")
    monster_type: MonsterRace = Field(alias="typ")
    atk: int | None = Field(alias="atk")
    defense: int | None = Field(alias="def")
    is_effect: bool = Field(default=True, alias="ie")

    @field_validator("atk", mode="before")
    @classmethod
    def validate_atk(cls, value: str) -> int | None:
        return parse_stat(value)

    @field_validator("defense", mode="before")
    @classmethod
    def validate_defense(cls, value: str) -> int | None:
        return parse_stat(value)


class RawSpellCard(_RawCardBase):
    """Raw spell card entry."""

    category: Literal[CardCategory.SPELL] = Field(alias="c")
    spell_type: SpellType = Field(alias="typ")


class RawTrapCard(_RawCardBase):
    """Raw trap card entry."""

    category: Literal[CardCategory.TRAP] = Field(alias="c")
    trap_type: TrapType = Field(alias="typ")


RawCard = Annotated[
    RawMonsterCard | RawSpellCard | RawTrapCard,
    Field(discriminator="category"),
]