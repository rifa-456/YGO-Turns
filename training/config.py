from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    effect_embedding_dim: int = 32
    card_vector_dim: int = 16
    fine_category_embedding_dim: int = 8
    level_embedding_dim: int = 4
    zone_dim: int = 16
    deck_slots: int = 5
    scalar_dim: int = 7
    mlp_hidden_dims: tuple[int, ...] = (64,)
    dropout: float = 0.4
    card_dropout_p: float = 0.2
    zone_dropout_p: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    val_fraction: float = 0.2
    seed: int = 42
    run_seed: int = 0
    num_workers: int = 4
    augment_swap: bool = True
    label_smoothing: float = 0.1
    weight_decay: float = 1e-3


@dataclass
class ScalarBaselineConfig:
    scalar_dim: int = 23
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    batch_size: int = 512
    max_epochs: int = 50
    val_fraction: float = 0.2
    seed: int = 42
    run_seed: int = 0
    num_workers: int = 4
    turns_parquet: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "ygo-turns.parquet"
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "checkpoints"
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    cards_parquet: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "cards.parquet"
    )
    turns_parquet: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "ygo-turns.parquet"
    )


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
