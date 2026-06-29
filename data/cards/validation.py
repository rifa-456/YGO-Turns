from __future__ import annotations

from pydantic import TypeAdapter

from data.cards.models import RawCard

RAW_CARD_ADAPTER: TypeAdapter[list[RawCard]] = TypeAdapter(list[RawCard])
