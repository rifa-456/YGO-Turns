from __future__ import annotations

import pyarrow as pa

CARDS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("effect_text", pa.string()),
        pa.field("category", pa.string()),
        pa.field("fine_category", pa.string()),
        pa.field("monster_category", pa.string()),
        pa.field("spell_type", pa.string()),
        pa.field("trap_type", pa.string()),
        pa.field("attribute", pa.string()),
        pa.field("level", pa.int8()),
        pa.field("atk", pa.int16(), nullable=True),
        pa.field("defense", pa.int16(), nullable=True),
        pa.field("monster_type", pa.string()),
        pa.field("is_effect", pa.bool_()),
        pa.field("effect_embedding", pa.list_(pa.float32())),
    ]
)
