from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from model.card_encoder import CardEncoder
from model.deck_encoder import DeckEncoder
from model.zone_encoder import MonsterZoneEncoder, SpellZoneEncoder, ZoneEncoder

_N_ZONES_PER_PLAYER: int = 7
_NUM_ZONES: int = _N_ZONES_PER_PLAYER * 2
_P2_ZONE_OFFSET: int = _N_ZONES_PER_PLAYER

_ZONE_POS_HAND: int = 0
_ZONE_POS_GY: int = 1
_ZONE_POS_BANISHED: int = 2
_ZONE_POS_EXTRA: int = 3
_ZONE_POS_SPELL: int = 4
_ZONE_POS_MONSTER: int = 5
_ZONE_POS_DECK: int = 6


class WinPredictor(nn.Module):
    def __init__(
        self,
        card_encoder: CardEncoder,
        zone_dim: int,
        card_vector_dim: int,
        deck_slots: int,
        scalar_dim: int,
        mlp_hidden_dims: Sequence[int],
        dropout: float,
        card_dropout_p: float = 0.0,
        zone_dropout_p: float = 0.0,
    ) -> None:
        """Initialises all zone encoders, aggregation components, and the prediction head.

        Args:
            card_encoder: Shared CardEncoder instance used across all zones.
            zone_dim: Output dimensionality for every zone encoder.
            card_vector_dim: Dimensionality of individual card vectors.
            deck_slots: Number of visible top-deck card slots.
            scalar_dim: Number of scalar game-state features.
            mlp_hidden_dims: Hidden layer sizes for the prediction MLP.
            dropout: Dropout probability applied after each hidden MLP layer.
            card_dropout_p: Probability of zeroing an entire card vector slot during training.
            zone_dropout_p: Probability of zeroing an entire zone vector during training.
        """
        super().__init__()

        self._card_encoder = card_encoder
        self._zone_encoder = ZoneEncoder(card_vector_dim, zone_dim)
        self._spell_zone_encoder = SpellZoneEncoder(card_vector_dim, zone_dim)
        self._monster_zone_encoder = MonsterZoneEncoder(card_vector_dim, zone_dim)
        self._deck_encoder = DeckEncoder(deck_slots)

        self._zone_embeddings = nn.Embedding(_NUM_ZONES, card_vector_dim)
        self._zone_pos_embeddings = nn.Embedding(_N_ZONES_PER_PLAYER, zone_dim)
        self._deck_proj = nn.Linear(card_vector_dim, zone_dim)
        self._zone_weights = nn.Parameter(torch.ones(_N_ZONES_PER_PLAYER))

        self._card_dropout_p = card_dropout_p
        self._zone_dropout_p = zone_dropout_p

        agg_dim = 3 * zone_dim + scalar_dim

        layers: list[nn.Module] = []
        in_dim = agg_dim
        for hidden_dim in mlp_hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self._head = nn.Sequential(*layers)

    def _apply_card_dropout(self, card_vecs: Tensor) -> Tensor:
        """Zeroes entire card vector slots with probability card_dropout_p during training.

        Args:
            card_vecs: Float tensor of shape (B, S, card_vector_dim).

        Returns:
            Tensor of the same shape with randomly selected slots zeroed out.
        """
        if not self.training or self._card_dropout_p == 0.0:
            return card_vecs
        keep = torch.bernoulli(
            torch.full(
                card_vecs.shape[:2],
                1.0 - self._card_dropout_p,
                device=card_vecs.device,
            )
        )
        return card_vecs * keep.unsqueeze(-1)

    def _apply_zone_dropout(self, zone_vectors: list[Tensor]) -> list[Tensor]:
        """Zeroes entire zone vectors with probability zone_dropout_p during training.

        Args:
            zone_vectors: List of float tensors each of shape (B, zone_dim).

        Returns:
            List of tensors of the same shapes with randomly selected zones zeroed out.
        """
        if not self.training or self._zone_dropout_p == 0.0:
            return zone_vectors
        result = []
        for vec in zone_vectors:
            keep = torch.bernoulli(
                torch.full(
                    (vec.shape[0],),
                    1.0 - self._zone_dropout_p,
                    device=vec.device,
                )
            )
            result.append(vec * keep.unsqueeze(-1))
        return result

    def _pool_player_zones(
        self,
        batch: dict[str, Tensor],
        prefix: str,
        zone_offset: int,
    ) -> Tensor:
        """Encodes all zones for one player into a single state vector.

        Applies zone-conditioned card encoding with card-slot dropout, runs each
        zone through its dedicated encoder, applies zone-level dropout, injects
        zone-type positional embeddings, and produces a learned weighted sum.

        Args:
            batch: Full batch dict of tensors from TurnsDataset.
            prefix: Player prefix string, either 'p1' or 'p2'.
            zone_offset: Integer offset added to every position index when
                looking up zone embeddings, separating P1 and P2 embedding rows.

        Returns:
            Float tensor of shape (B, zone_dim) representing the player state.
        """
        zone_vectors: list[Tensor] = []
        device = batch["scalars"].device

        for pos, suffix in [
            (_ZONE_POS_HAND, "hand"),
            (_ZONE_POS_GY, "gy"),
            (_ZONE_POS_BANISHED, "banished"),
            (_ZONE_POS_EXTRA, "extra"),
        ]:
            zone_name = f"{prefix}_{suffix}"
            zone_emb = self._zone_embeddings(
                torch.tensor(zone_offset + pos, device=device)
            )
            card_vecs = (
                self._card_encoder(
                    batch[f"{zone_name}_effect_emb"],
                    batch[f"{zone_name}_cat_idx"],
                )
                + zone_emb
            )
            card_vecs = self._apply_card_dropout(card_vecs)
            zone_vectors.append(
                self._zone_encoder(card_vecs, batch[f"{zone_name}_mask"])
            )

        zone_name = f"{prefix}_spell_zones"
        zone_emb = self._zone_embeddings(
            torch.tensor(zone_offset + _ZONE_POS_SPELL, device=device)
        )
        card_vecs = (
            self._card_encoder(
                batch[f"{zone_name}_effect_emb"],
                batch[f"{zone_name}_cat_idx"],
            )
            + zone_emb
        )
        card_vecs = self._apply_card_dropout(card_vecs)
        zone_vectors.append(
            self._spell_zone_encoder(
                card_vecs,
                batch[f"{zone_name}_slot_numerics"],
                batch[f"{zone_name}_mask"],
            )
        )

        zone_name = f"{prefix}_monster_zones"
        zone_emb = self._zone_embeddings(
            torch.tensor(zone_offset + _ZONE_POS_MONSTER, device=device)
        )
        card_vecs = (
            self._card_encoder(
                batch[f"{zone_name}_effect_emb"],
                batch[f"{zone_name}_cat_idx"],
            )
            + zone_emb
        )
        card_vecs = self._apply_card_dropout(card_vecs)
        equip_vecs = (
            self._card_encoder(
                batch[f"{zone_name}_equip_effect_emb"],
                batch[f"{zone_name}_equip_cat_idx"],
            )
            + zone_emb
        )
        orig_shape = equip_vecs.shape
        equip_vecs = self._apply_card_dropout(
            equip_vecs.view(orig_shape[0], -1, orig_shape[-1])
        ).view(orig_shape)
        zone_vectors.append(
            self._monster_zone_encoder(
                card_vecs,
                batch[f"{zone_name}_slot_numerics"],
                equip_vecs,
                batch[f"{zone_name}_equip_mask"],
                batch[f"{zone_name}_mask"],
            )
        )

        zone_name = f"{prefix}_deck_top"
        zone_emb = self._zone_embeddings(
            torch.tensor(zone_offset + _ZONE_POS_DECK, device=device)
        )
        card_vecs = (
            self._card_encoder(
                batch[f"{zone_name}_effect_emb"],
                batch[f"{zone_name}_cat_idx"],
            )
            + zone_emb
        )
        card_vecs = self._apply_card_dropout(card_vecs)
        deck_vec = self._deck_encoder(card_vecs, batch[f"{zone_name}_mask"])
        zone_vectors.append(self._deck_proj(deck_vec))

        zone_vectors = self._apply_zone_dropout(zone_vectors)

        stacked = torch.stack(zone_vectors, dim=1)
        pos_indices = torch.arange(_N_ZONES_PER_PLAYER, device=device)
        stacked = stacked + self._zone_pos_embeddings(pos_indices)
        weights = torch.softmax(self._zone_weights, dim=0)
        return (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Computes win logits for a batch of game-state turns.

        Encodes P1 and P2 zones independently into player state vectors, builds
        an interaction term via element-wise product, concatenates with scalar
        features, and passes the result through the prediction MLP.

        Args:
            batch: Dict of tensors produced by TurnsDataset.__getitem__.

        Returns:
            Float tensor of shape (B,) containing raw (pre-sigmoid) logits.
        """
        p1_state = self._pool_player_zones(batch, "p1", 0)
        p2_state = self._pool_player_zones(batch, "p2", _P2_ZONE_OFFSET)
        interaction = p1_state * p2_state
        agg = torch.cat([p1_state, p2_state, interaction, batch["scalars"]], dim=-1)
        return self._head(agg).squeeze(-1)
