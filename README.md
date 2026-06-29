# YGO-Turns

Source code accompanying the paper. Implements a game-state win predictor for Yu-Gi-Oh! duels, trained on turn-level
snapshots extracted from recorded matches.

---

## Installation

Requires Python 3.12+.

```bash
git clone <repo-url>
cd YGO-Turns
uv sync
```

Dependencies are declared in `pyproject.toml`.

---

## Project Structure

```
.
├── data/
│   ├── schema/         # Column name descriptors and DataFrame schema constants
│   ├── cards/          # JSON loader, validated card models, schema, and writer
│   ├── turns/          # Turns parquet reader
│   ├── embedding/      # Sentence embedding computation and PCA compression
│   └── preprocessing/  # Card category derivation and embedding text extraction
├── domain/             # Card domain models and enumerations
├── model/              # Pure PyTorch nn.Module definitions
├── training/           # Dataset, DataModule, LightningModule, config, and augmentation
├── scripts/            # Runnable entry points
└── artifacts/          # Data files and saved checkpoints (not version-controlled)
```

### `data/schema/`

Defines the canonical column layout of the turns DataFrame.

| File           | Role                                                                  |
|----------------|-----------------------------------------------------------------------|
| `constants.py` | Integer caps (hand size, zone counts, etc.) and column name literals  |
| `enums.py`     | `CardPosition` and `PlayerPrefix` enumerations                        |
| `zones.py`     | Frozen dataclasses describing column-name groups for each zone type   |
| `builders.py`  | Factory functions that instantiate zone and player column descriptors |
| `player.py`    | `PlayerColumns` dataclass aggregating all per-player column groups    |
| `flatten.py`   | Produces a flat ordered list of all column names for a player         |

### `model/`

| File                 | Role                                                                                   |
|----------------------|----------------------------------------------------------------------------------------|
| `card_encoder.py`    | Fuses PCA effect embeddings with categorical embeddings into a card vector             |
| `zone_encoder.py`    | DeepSet pooling for standard, spell, and monster zones                                 |
| `deck_encoder.py`    | Learned positional weighted mean over the top-K deck cards                             |
| `win_predictor.py`   | Full model: encodes both players' zones, builds an interaction term, feeds an MLP head |
| `scalar_baseline.py` | Lightweight MLP operating only on scalar game-state features                           |

### `training/`

| File                | Role                                                                                                 |
|---------------------|------------------------------------------------------------------------------------------------------|
| `config.py`         | `Config`, `ModelConfig`, `TrainingConfig`, `PathConfig`, `ScalarBaselineConfig` dataclasses          |
| `vocabulary.py`     | Builds `fine_category` and `level` vocabularies from `cards.parquet`                                 |
| `card_lookup.py`    | Pre-indexes card name → `(effect_embedding, categorical_indices)` for O(1) lookup                    |
| `dataset.py`        | `TurnsDataset` — converts a turns DataFrame row into the full tensor dict consumed by `WinPredictor` |
| `scalar_dataset.py` | `ScalarDataset` — emits the 23-dimensional scalar feature vector only                                |
| `datamodule.py`     | `TurnsDataModule` — match-level train/val split, optional perspective-swap augmentation              |
| `module.py`         | `WinPredictorModule` — Lightning wrapper with BCE loss, label smoothing, AUROC/Brier logging         |
| `augmentation.py`   | `swap_perspective` — mirrors a batch by renaming `p1_*` ↔ `p2_*` and flipping targets                |

---

## Data Preparation

`cards.parquet` is not distributed with the repository due to copyright over card effect text. It must be generated
locally before running the full WinPredictor.

### Preparing `cards.json`

The build script expects a JSON file at `data/cards/cards.json` containing a top-level array of Yu-Gi-Oh! card objects.

**Fields required for all cards**

| Field | Type    | Description                                        |
|-------|---------|----------------------------------------------------|
| `id`  | integer | Unique card ID                                     |
| `n`   | string  | Card name                                          |
| `e`   | string  | Effect or flavour text                             |
| `c`   | string  | Card category: `"Monster"`, `"Spell"`, or `"Trap"` |

**Additional fields for monsters** (`"c": "Monster"`)

| Field | Type    | Valid values                                                                                                                                                                                                                                                                      |
|-------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mc`  | string  | `Normal`, `Effect`, `Fusion`, `Ritual`, `Toon`, `Spirit`, `Union`, `Gemini`, `Synchro`, `Xyz`, `Link`                                                                                                                                                                             |
| `att` | string  | `LIGHT`, `DARK`, `EARTH`, `WATER`, `FIRE`, `WIND`, `DIVINE`                                                                                                                                                                                                                       |
| `lvl` | integer | Card level                                                                                                                                                                                                                                                                        |
| `atk` | string  | ATK value, or `"?"` for variable                                                                                                                                                                                                                                                  |
| `def` | string  | DEF value, or `"?"` for variable                                                                                                                                                                                                                                                  |
| `typ` | string  | `Dragon`, `Spellcaster`, `Warrior`, `Machine`, `Zombie`, `Fiend`, `Fairy`, `Aqua`, `Pyro`, `Rock`, `Thunder`, `Dinosaur`, `Beast`, `Beast-Warrior`, `Winged Beast`, `Insect`, `Plant`, `Reptile`, `Psychic`, `Wyrm`, `Fish`, `Sea Serpent`, `Divine-Beast`, `Cyberse`, and others |

**Additional field for spells** (`"c": "Spell"`)

| Field | Type   | Valid values                                                     |
|-------|--------|------------------------------------------------------------------|
| `typ` | string | `Normal`, `Continuous`, `Equip`, `Quick-Play`, `Field`, `Ritual` |

**Additional field for traps** (`"c": "Trap"`)

| Field | Type   | Valid values                      |
|-------|--------|-----------------------------------|
| `typ` | string | `Normal`, `Continuous`, `Counter` |

### Building `cards.parquet`

Once `cards.json` is in place, run:

```bash
python scripts/build_cards.py
```

This will:

1. Filter cards to names present in `artifacts/turns.parquet`
2. Deduplicate to one entry per card name by lowest ID
3. Compute sentence embeddings using `all-MiniLM-L6-v2`
4. Compress embeddings to 32 dimensions via PCA
5. Write the result to `artifacts/cards.parquet`

---

## Running Experiments

### Logistic Regression Baseline

Operates on `artifacts/turns.parquet` only. No `cards.parquet` required.

```bash
python scripts/train_baseline_logistic.py
```

Saves the fitted model to `artifacts/baselines/scalar_logistic.joblib` and metrics to
`artifacts/baselines/scalar_logistic_metrics.json`.

### Scalar MLP Baseline

Operates on `artifacts/turns.parquet` only.

```bash
python scripts/train_baseline_mlp.py
```

Checkpoints are saved to `artifacts/checkpoints/`.

### Full WinPredictor

Requires both `artifacts/turns.parquet` and `artifacts/cards.parquet`.

```bash
python scripts/train.py
```

Hyperparameters are controlled through `training/config.py`. The top-3 checkpoints by validation AUROC are kept under
`artifacts/checkpoints/`.

### Exploratory Analysis

```bash
python scripts/eda.py              # Dataset statistics and zone occupancy tables
python scripts/card_similarity.py  # Nearest-neighbour retrieval over card embeddings
```

---

## Reproducibility

All train/val splits are match-level (no turn leakage across splits). The random seed defaults to `42` in all configs.
Set `TrainingConfig.seed` or
`ScalarBaselineConfig.seed` to change it.