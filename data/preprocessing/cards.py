from __future__ import annotations

from data.cards.models import (
    RawMonsterCard,
    RawSpellCard,
    RawTrapCard,
)
from domain.enums import (
    FineCategory,
    MonsterCategory,
    SpellType,
    TrapType,
)

NO_EFFECT_TEXT: str = "This card has no effect."

MONSTER_CATEGORY_TO_FINE: dict[MonsterCategory, FineCategory] = {
    MonsterCategory.NORMAL: FineCategory.NORMAL_MONSTER,
    MonsterCategory.EFFECT: FineCategory.EFFECT_MONSTER,
    MonsterCategory.FUSION: FineCategory.FUSION_MONSTER,
    MonsterCategory.RITUAL: FineCategory.RITUAL_MONSTER,
    MonsterCategory.TOON: FineCategory.TOON_MONSTER,
    MonsterCategory.SPIRIT: FineCategory.SPIRIT_MONSTER,
    MonsterCategory.UNION: FineCategory.UNION_MONSTER,
    MonsterCategory.GEMINI: FineCategory.GEMINI_MONSTER,
    MonsterCategory.SYNCHRO: FineCategory.SYNCHRO_MONSTER,
    MonsterCategory.XYZ: FineCategory.XYZ_MONSTER,
    MonsterCategory.LINK: FineCategory.LINK_MONSTER,
}

SPELL_TYPE_TO_FINE: dict[SpellType, FineCategory] = {
    SpellType.NORMAL: FineCategory.SPELL_NORMAL,
    SpellType.CONTINUOUS: FineCategory.SPELL_CONTINUOUS,
    SpellType.EQUIP: FineCategory.SPELL_EQUIP,
    SpellType.QUICK_PLAY: FineCategory.SPELL_QUICK_PLAY,
    SpellType.FIELD: FineCategory.SPELL_FIELD,
    SpellType.RITUAL: FineCategory.SPELL_RITUAL,
}

TRAP_TYPE_TO_FINE: dict[TrapType, FineCategory] = {
    TrapType.NORMAL: FineCategory.TRAP_NORMAL,
    TrapType.CONTINUOUS: FineCategory.TRAP_CONTINUOUS,
    TrapType.COUNTER: FineCategory.TRAP_COUNTER,
}


def derive_fine_category(
    raw: RawMonsterCard | RawSpellCard | RawTrapCard,
) -> FineCategory:
    """Derives fine-grained card category from a raw card model.

    Args:
        raw: A validated raw card instance.

    Returns:
        The FineCategory corresponding to the card's type and subtype.

    Raises:
        ValueError: If a MonsterCategory has no entry in MONSTER_CATEGORY_TO_FINE.
    """
    if isinstance(raw, RawMonsterCard):
        if raw.monster_category not in MONSTER_CATEGORY_TO_FINE:
            raise ValueError(
                f"MonsterCategory '{raw.monster_category}' has no FineCategory mapping. "
                "Add it to MONSTER_CATEGORY_TO_FINE and FineCategory before proceeding."
            )
        return MONSTER_CATEGORY_TO_FINE[raw.monster_category]

    if isinstance(raw, RawSpellCard):
        return SPELL_TYPE_TO_FINE[raw.spell_type]

    return TRAP_TYPE_TO_FINE[raw.trap_type]


def embedding_text(
    raw: RawMonsterCard | RawSpellCard | RawTrapCard,
) -> str:
    """Returns text used for sentence embedding generation."""

    if (
        isinstance(raw, RawMonsterCard)
        and raw.monster_category == MonsterCategory.NORMAL
    ):
        return NO_EFFECT_TEXT

    return raw.effect_text
