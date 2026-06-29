from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.schema import IS_P1_TURN, P1, P2
from data.schema.constants import (
    EQUIPS_MAX,
    MONSTER_ZONE_COUNT,
    PAD_TOKEN,
    SPELL_ZONE_COUNT,
    TARGET,
)
from data.schema.zones import MonsterZoneColumns, SpellZoneColumns
from training.card_lookup import CardLookup, CardTensors

_ATK_ZONE_NORM: float = 3000.0
_DEF_ZONE_NORM: float = 3000.0
_COUNTER_NORM: float = 10.0
_LP_NORM: float = 16000.0
_DECK_NORM: float = 60.0
_HAND_NORM: float = 15.0

_STANDARD_ZONE_COLS: list[tuple[str, list[str]]] = [
    ("p1_hand", P1.hand),
    ("p1_gy", P1.gy),
    ("p1_banished", P1.banished),
    ("p1_extra", P1.extra),
    ("p2_hand", P2.hand),
    ("p2_gy", P2.gy),
    ("p2_banished", P2.banished),
    ("p2_extra", P2.extra),
]

_SPELL_ZONE_COLS: list[tuple[str, list[str]]] = [
    ("p1_spell_zones", P1.spell_zones.name),
    ("p2_spell_zones", P2.spell_zones.name),
]

_MONSTER_ZONE_COLS: list[tuple[str, list[str]]] = [
    ("p1_monster_zones", P1.monster_zones.name),
    ("p2_monster_zones", P2.monster_zones.name),
]

_DECK_ZONE_COLS: list[tuple[str, list[str]]] = [
    ("p1_deck_top", P1.deck_top),
    ("p2_deck_top", P2.deck_top),
]

_ALL_ZONE_COLS: list[tuple[str, list[str]]] = (
    _STANDARD_ZONE_COLS + _SPELL_ZONE_COLS + _MONSTER_ZONE_COLS + _DECK_ZONE_COLS
)

_SCALAR_COLS: list[str] = [
    P1.life_points,
    P2.life_points,
    IS_P1_TURN,
    P1.hand_size,
    P2.hand_size,
    P1.deck_size,
    P2.deck_size,
]

_SCALAR_NORMS: np.ndarray = np.array(
    [_LP_NORM, _LP_NORM, 1.0, _HAND_NORM, _HAND_NORM, _DECK_NORM, _DECK_NORM],
    dtype=np.float32,
)

SCALAR_DIM: int = len(_SCALAR_COLS)


class TurnsDataset(Dataset):
    """PyTorch Dataset over a preprocessed turns DataFrame."""

    def __init__(self, df: pd.DataFrame, card_lookup: CardLookup) -> None:
        """Initialises the dataset by pre-building all zone arrays from the DataFrame.

        Args:
            df: Preprocessed turns DataFrame produced by data.preprocessing.preprocess.
            card_lookup: Pre-built card name to feature tensor mapping.
        """
        self._lookup = card_lookup
        self._length = len(df)

        self._zone_names: dict[str, np.ndarray] = {
            zone_key: df[cols].values for zone_key, cols in _ALL_ZONE_COLS
        }

        self._spell_slot_numerics: dict[str, np.ndarray] = {
            "p1_spell_zones": _build_spell_slot_array(df, P1.spell_zones),
            "p2_spell_zones": _build_spell_slot_array(df, P2.spell_zones),
        }

        self._monster_slot_numerics: dict[str, np.ndarray] = {
            "p1_monster_zones": _build_monster_slot_array(df, P1.monster_zones),
            "p2_monster_zones": _build_monster_slot_array(df, P2.monster_zones),
        }

        self._equip_names: dict[str, np.ndarray] = {
            "p1_monster_zones": _build_equip_name_array(df, P1.monster_zones),
            "p2_monster_zones": _build_equip_name_array(df, P2.monster_zones),
        }

        self._scalars: np.ndarray = (
            df[_SCALAR_COLS].astype(np.float32).values / _SCALAR_NORMS
        )

        self._targets: np.ndarray = df[TARGET].astype(np.float32).values

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Builds the feature dict for a single turn.

        Args:
            idx: Integer row index into the dataset.

        Returns:
            Dict of named tensors consumed directly by WinPredictor.forward.
        """
        sample: dict[str, Tensor] = {}

        for zone_key, _ in _ALL_ZONE_COLS:
            sample.update(
                _encode_zone(zone_key, self._zone_names[zone_key][idx], self._lookup)
            )

        for zone_key, arr in self._spell_slot_numerics.items():
            sample[f"{zone_key}_slot_numerics"] = torch.from_numpy(arr[idx].copy())

        for zone_key, arr in self._monster_slot_numerics.items():
            sample[f"{zone_key}_slot_numerics"] = torch.from_numpy(arr[idx].copy())

        for zone_key in ("p1_monster_zones", "p2_monster_zones"):
            sample.update(
                _encode_equips(zone_key, self._equip_names[zone_key][idx], self._lookup)
            )

        sample["scalars"] = torch.from_numpy(self._scalars[idx].copy())
        sample["target"] = torch.tensor(self._targets[idx])
        return sample


def _safe_name(value: object) -> str:
    """Converts an arbitrary cell value to a card name string.

    Args:
        value: Raw value from a numpy object array cell.

    Returns:
        String card name, or PAD_TOKEN when the value is None or NaN.
    """
    if value is None:
        return PAD_TOKEN
    if isinstance(value, float) and np.isnan(value):
        return PAD_TOKEN
    return str(value)


def _encode_zone(
    zone_key: str,
    names: np.ndarray,
    lookup: CardLookup,
) -> dict[str, Tensor]:
    """Encodes a row of card name cells into stacked zone feature tensors.

    Args:
        zone_key: Zone identifier used as a prefix for the output dict keys.
        names: 1-D object array of card name strings for this zone.
        lookup: CardLookup used to resolve each name to its CardTensors.

    Returns:
        Dict with three tensors keyed by zone_key prefix: effect_emb, cat_idx,
        and mask.
    """
    safe = [_safe_name(n) for n in names]
    tensors: list[CardTensors] = [lookup.get(n) for n in safe]
    mask = torch.tensor([n != PAD_TOKEN for n in safe], dtype=torch.bool)

    return {
        f"{zone_key}_effect_emb": torch.stack([t.effect_embedding for t in tensors]),
        f"{zone_key}_cat_idx": torch.stack([t.categorical_indices for t in tensors]),
        f"{zone_key}_mask": mask,
    }


def _build_monster_slot_array(
    df: pd.DataFrame,
    zones: MonsterZoneColumns,
) -> np.ndarray:
    """Builds the per-player monster zone slot numerics array.

    Args:
        df: Preprocessed turns DataFrame.
        zones: MonsterZoneColumns for the target player.

    Returns:
        Float32 ndarray of shape (N, MONSTER_ZONE_COUNT, 5) containing
        normalised atk, def, position (ATK=1), face_down, and counters.
    """
    n = len(df)
    result = np.zeros((n, MONSTER_ZONE_COUNT, 5), dtype=np.float32)

    for i in range(MONSTER_ZONE_COUNT):
        result[:, i, 0] = (
            pd.to_numeric(df[zones.atk[i]], errors="coerce").fillna(0.0).values
            / _ATK_ZONE_NORM
        )
        result[:, i, 1] = (
            pd.to_numeric(df[zones.defense[i]], errors="coerce").fillna(0.0).values
            / _DEF_ZONE_NORM
        )
        result[:, i, 2] = (df[zones.position[i]] == "ATK").astype(np.float32).values
        result[:, i, 3] = df[zones.face_down[i]].fillna(False).astype(np.float32).values
        result[:, i, 4] = (
            pd.to_numeric(df[zones.counters[i]], errors="coerce").fillna(0.0).values
            / _COUNTER_NORM
        )

    return result


def _build_spell_slot_array(
    df: pd.DataFrame,
    zones: SpellZoneColumns,
) -> np.ndarray:
    """Builds the per-player spell zone slot numerics array.

    Args:
        df: Preprocessed turns DataFrame.
        zones: SpellZoneColumns for the target player.

    Returns:
        Float32 ndarray of shape (N, SPELL_ZONE_COUNT, 2) containing
        normalised face_down and counters for each spell zone slot.
    """
    n = len(df)
    result = np.zeros((n, SPELL_ZONE_COUNT, 2), dtype=np.float32)

    for i in range(SPELL_ZONE_COUNT):
        result[:, i, 0] = df[zones.face_down[i]].fillna(False).astype(np.float32).values
        result[:, i, 1] = (
            pd.to_numeric(df[zones.counters[i]], errors="coerce").fillna(0.0).values
            / _COUNTER_NORM
        )

    return result


def _build_equip_name_array(
    df: pd.DataFrame,
    zones: MonsterZoneColumns,
) -> np.ndarray:
    """Builds the per-player monster zone equip name array.

    Args:
        df: Preprocessed turns DataFrame.
        zones: MonsterZoneColumns for the target player.

    Returns:
        Object ndarray of shape (N, MONSTER_ZONE_COUNT, EQUIPS_MAX) containing
        card name strings for every equip slot, padded with PAD_TOKEN where absent.
    """
    n = len(df)
    result = np.empty((n, MONSTER_ZONE_COUNT, EQUIPS_MAX), dtype=object)
    for i in range(MONSTER_ZONE_COUNT):
        for j in range(EQUIPS_MAX):
            result[:, i, j] = df[zones.equips[i][j]].values
    return result


def _encode_equips(
    zone_key: str,
    equip_names: np.ndarray,
    lookup: CardLookup,
) -> dict[str, Tensor]:
    """Encodes all equip card slots for a monster zone row into stacked tensors.

    Args:
        zone_key: Zone identifier used as a prefix for the output dict keys.
        equip_names: Object array of shape (MONSTER_ZONE_COUNT, EQUIPS_MAX)
            containing card name strings for each equip slot.
        lookup: CardLookup used to resolve each name to its CardTensors.

    Returns:
        Dict with three tensors keyed by zone_key prefix: equip_effect_emb of
        shape (MONSTER_ZONE_COUNT, EQUIPS_MAX, effect_dim), equip_cat_idx of
        shape (MONSTER_ZONE_COUNT, EQUIPS_MAX, 2), and equip_mask of shape
        (MONSTER_ZONE_COUNT, EQUIPS_MAX).
    """
    effect_embs: list[Tensor] = []
    cat_idxs: list[Tensor] = []
    masks: list[Tensor] = []

    for i in range(MONSTER_ZONE_COUNT):
        slot_effect: list[Tensor] = []
        slot_cat: list[Tensor] = []
        slot_mask: list[bool] = []

        for j in range(EQUIPS_MAX):
            name = _safe_name(equip_names[i, j])
            card_tensors = lookup.get(name)
            slot_effect.append(card_tensors.effect_embedding)
            slot_cat.append(card_tensors.categorical_indices)
            slot_mask.append(name != PAD_TOKEN)

        effect_embs.append(torch.stack(slot_effect))
        cat_idxs.append(torch.stack(slot_cat))
        masks.append(torch.tensor(slot_mask, dtype=torch.bool))

    return {
        f"{zone_key}_equip_effect_emb": torch.stack(effect_embs),
        f"{zone_key}_equip_cat_idx": torch.stack(cat_idxs),
        f"{zone_key}_equip_mask": torch.stack(masks),
    }
