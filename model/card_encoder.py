from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CardEncoder(nn.Module):
    """Projects PCA-compressed card features into a fixed-size card vector."""

    def __init__(
        self,
        effect_embedding_dim: int,
        fine_category_vocab_size: int,
        level_vocab_size: int,
        fine_category_embedding_dim: int,
        level_embedding_dim: int,
        card_vector_dim: int,
    ) -> None:
        """Initialises embedding tables and the fusion MLP.

        Args:
            effect_embedding_dim: Dimensionality of PCA-compressed effect embeddings.
            fine_category_vocab_size: Number of fine-category vocabulary entries.
            level_vocab_size: Number of level vocabulary entries.
            fine_category_embedding_dim: Embedding size for fine_category.
            level_embedding_dim: Embedding size for level.
            card_vector_dim: Output dimensionality of the card vector.
        """
        super().__init__()

        self._fine_category_emb = nn.Embedding(
            fine_category_vocab_size, fine_category_embedding_dim, padding_idx=0
        )
        self._level_emb = nn.Embedding(
            level_vocab_size, level_embedding_dim, padding_idx=0
        )

        fused_dim = (
            effect_embedding_dim + fine_category_embedding_dim + level_embedding_dim
        )

        self._fusion = nn.Sequential(
            nn.Linear(fused_dim, card_vector_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        effect_embeddings: Tensor,
        categorical_indices: Tensor,
    ) -> Tensor:
        """Encodes card features into a single card vector.

        Args:
            effect_embeddings: Float tensor of shape (..., effect_embedding_dim)
                holding PCA-compressed effect embeddings.
            categorical_indices: Long tensor of shape (..., 2) holding indices
                for fine_category and level respectively.

        Returns:
            Float tensor of shape (..., card_vector_dim).
        """
        cat = torch.cat(
            [
                self._fine_category_emb(categorical_indices[..., 0]),
                self._level_emb(categorical_indices[..., 1]),
            ],
            dim=-1,
        )
        return self._fusion(torch.cat([effect_embeddings, cat], dim=-1))
