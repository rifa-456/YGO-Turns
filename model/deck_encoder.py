from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DeckEncoder(nn.Module):
    """Encodes the top-K deck cards via learned positional weighted mean."""

    def __init__(self, deck_slots: int) -> None:
        super().__init__()
        self._positional_weights = nn.Parameter(torch.ones(deck_slots))

    def forward(self, card_vectors: Tensor, mask: Tensor) -> Tensor:
        """Produces a deck vector from ordered card vectors.

        Args:
            card_vectors: Float tensor of shape (B, deck_slots, card_vector_dim).
            mask: Bool tensor of shape (B, deck_slots). True = real card slot.

        Returns:
            Float tensor of shape (B, card_vector_dim).
        """
        weights = torch.softmax(self._positional_weights, dim=0)
        masked_weights = weights.unsqueeze(0) * mask.float()
        normalised = masked_weights / masked_weights.sum(dim=1, keepdim=True).clamp(
            min=1e-8
        )
        return (card_vectors * normalised.unsqueeze(-1)).sum(dim=1)
