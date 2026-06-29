from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.cards.models import (
    RawMonsterCard,
    RawSpellCard,
    RawTrapCard,
)
from data.cards.validation import RAW_CARD_ADAPTER
from domain.enums import CardCategory

_SUPPORTED_CATEGORIES: set[str] = {
    CardCategory.MONSTER.value,
    CardCategory.SPELL.value,
    CardCategory.TRAP.value,
}


def load_raw_cards(
    path: Path,
) -> list[RawMonsterCard | RawSpellCard | RawTrapCard]:
    """Reads and validates supported cards from a cards JSON file.

    Args:
        path: Path to a JSON file containing a top-level array of card objects.

    Returns:
        List of validated raw card models for Monster, Spell, and Trap cards.
    """

    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    filtered = [card for card in raw if card.get("c") in _SUPPORTED_CATEGORIES]

    return RAW_CARD_ADAPTER.validate_python(filtered)
