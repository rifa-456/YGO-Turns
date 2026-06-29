from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ZoneEncoder(nn.Module):
    """DeepSet encoder for a variable-occupancy game zone."""

    def __init__(self, card_vector_dim: int, output_dim: int) -> None:
        super().__init__()
        self._element_mlp = nn.Sequential(
            nn.Linear(card_vector_dim, output_dim),
            nn.ReLU(),
        )
        self._empty_zone_vec = nn.Parameter(torch.zeros(output_dim))

    def forward(self, card_vectors: Tensor, mask: Tensor) -> Tensor:
        """Pools card vectors into a single zone representation.

        Args:
            card_vectors: Float tensor of shape (B, S, card_vector_dim).
            mask: Bool tensor of shape (B, S). True = real card; False = PAD slot.

        Returns:
            Float tensor of shape (B, output_dim). Rows where every slot is PAD
            receive the learned empty_zone_vec instead of a zero vector.
        """
        x = self._element_mlp(card_vectors)

        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        is_empty = ~mask.any(dim=1)
        if is_empty.any():
            pooled = pooled.clone()
            pooled[is_empty] = self._empty_zone_vec

        return pooled


class SpellZoneEncoder(nn.Module):
    """DeepSet encoder for spell/trap zones with per-slot face-down and counter numerics."""

    _SLOT_NUMERIC_DIM: int = 2

    def __init__(self, card_vector_dim: int, output_dim: int) -> None:
        super().__init__()
        fused_input_dim = card_vector_dim + self._SLOT_NUMERIC_DIM
        self._element_mlp = nn.Sequential(
            nn.Linear(fused_input_dim, output_dim),
            nn.ReLU(),
        )
        self._empty_zone_vec = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        card_vectors: Tensor,
        slot_numerics: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Pools spell zone features into a single zone representation.

        Args:
            card_vectors: Float tensor of shape (B, S, card_vector_dim).
            slot_numerics: Float tensor of shape (B, S, 2) containing normalised
                face_down and counters for each slot.
            mask: Bool tensor of shape (B, S). True = occupied slot.

        Returns:
            Float tensor of shape (B, output_dim).
        """
        x = self._element_mlp(torch.cat([card_vectors, slot_numerics], dim=-1))

        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        is_empty = ~mask.any(dim=1)
        if is_empty.any():
            pooled = pooled.clone()
            pooled[is_empty] = self._empty_zone_vec

        return pooled


class MonsterZoneEncoder(nn.Module):
    """DeepSet encoder for monster zones with per-slot game state numerics and equip cards."""

    _SLOT_NUMERIC_DIM: int = 5

    def __init__(self, card_vector_dim: int, output_dim: int) -> None:
        super().__init__()
        fused_input_dim = 2 * card_vector_dim + self._SLOT_NUMERIC_DIM
        self._element_mlp = nn.Sequential(
            nn.Linear(fused_input_dim, output_dim),
            nn.ReLU(),
        )
        self._empty_zone_vec = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        card_vectors: Tensor,
        slot_numerics: Tensor,
        equip_vectors: Tensor,
        equip_mask: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Pools monster zone features into a single zone representation.

        Args:
            card_vectors: Float tensor of shape (B, 5, card_vector_dim).
            slot_numerics: Float tensor of shape (B, 5, 5) with per-slot
                game state values: atk, def, position, face_down, counters.
            equip_vectors: Float tensor of shape (B, 5, EQUIPS_MAX, card_vector_dim)
                holding CardEncoder outputs for each equip slot.
            equip_mask: Bool tensor of shape (B, 5, EQUIPS_MAX). True = occupied equip slot.
            mask: Bool tensor of shape (B, 5). True = occupied monster slot.

        Returns:
            Float tensor of shape (B, output_dim).
        """
        equip_mask_f = equip_mask.unsqueeze(-1).float()
        equip_sum = (equip_vectors * equip_mask_f).sum(dim=2)
        equip_count = equip_mask_f.sum(dim=2).clamp(min=1.0)
        pooled_equips = equip_sum / equip_count

        x = self._element_mlp(
            torch.cat([card_vectors, slot_numerics, pooled_equips], dim=-1)
        )

        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        is_empty = ~mask.any(dim=1)
        if is_empty.any():
            pooled = pooled.clone()
            pooled[is_empty] = self._empty_zone_vec

        return pooled
