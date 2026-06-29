from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor


class ScalarBaseline(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        """Initialises the MLP layers from scalar_dim to a single logit output.

        Args:
            scalar_dim: Number of input scalar features.
            hidden_dims: Sizes for each hidden layer.
            dropout: Dropout probability applied after each hidden activation.
        """
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = scalar_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self._net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Computes a win logit from the scalar feature vector.

        Args:
            batch: Dict containing at minimum a 'scalars' key with a float tensor
                of shape (B, scalar_dim).

        Returns:
            Float tensor of shape (B,) with raw logits.
        """
        return self._net(batch["scalars"]).squeeze(-1)
