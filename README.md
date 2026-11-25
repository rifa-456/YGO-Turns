# YGO-Turns

[![Paper](https://img.shields.io/badge/paper-IEEE%20CoG%202026-00629B)](https://doi.org/10.1109/CoG67430.2026.11778780)
[![Dataset](https://img.shields.io/badge/dataset-Hugging%20Face-FFD21E)](https://huggingface.co/datasets/Rifa456/YGO-Turns)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

YGO-Turns is a dataset of expert Yu-Gi-Oh! tournament play: the board state of both players at the
end of every turn, from 1,765 duels in 682 matches. This repository holds the code that trains win
predictors on it and reproduces its releases.

- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11778780), IEEE Conference on Games 2026
- **Dataset:** [Rifa456/YGO-Turns](https://huggingface.co/datasets/Rifa456/YGO-Turns) on Hugging Face,
  with every column explained
- **Changes since the paper:** [CHANGELOG.md](CHANGELOG.md)

## Results

| Model | Paper, v1.0 validation | v1.1 validation | v1.1 test | v1.1 test, 95% interval |
|---|---|---|---|---|
| Logistic regression | 0.747 | 0.743 | 0.725 | 0.670–0.772 |
| Scalar MLP | 0.755 ± 0.001 | 0.751 ± 0.002 | 0.733 ± 0.003 | 0.684–0.775 |
| BoardDSN | 0.716 ± 0.004 | 0.706 ± 0.006 | 0.721 ± 0.006 | 0.684–0.753 |

Scores are AUROC, mean ± standard deviation over five training seeds. Checkpoints are chosen on
validation. The interval resamples whole test matches. The three test intervals overlap widely, so
the order of the models on test is uncertain. v1.1 models train on 411 matches, the paper's on 548.

[results.json](artifacts/results.json) has every run, with Brier score and accuracy.

## Installation

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rifa-456/YGO-Turns.git
cd YGO-Turns
uv sync
```

The dataset downloads from Hugging Face on first use.

## Running the experiments

```bash
uv run python -m scripts.train_baseline_logistic
uv run python -m scripts.train_baseline_mlp --run-seed 0
uv run python -m scripts.train --run-seed 0
uv run python -m scripts.evaluate
```

Train the MLP and BoardDSN with seeds 0 to 4 before evaluating. Checkpoints go to
`artifacts/checkpoints/`, and the evaluation uses each seed's best checkpoint by validation AUROC.
BoardDSN (`scripts.train`) needs the card table described below; without it, evaluate with
`--models logistic mlp`. Hyperparameters are in `training/config.py`.

Two more scripts explore the data: `scripts.eda` prints statistics, and `scripts.card_similarity`
lists the nearest neighbours of card embeddings.

### Reproducing the paper

The paper uses v1.0. Check out the `v1.0` tag, which has the code, the dataset file and the trained
models of the paper:

```bash
git checkout v1.0
```

It splits matches with seed 42 into 548 for training and 136 for validation.

### Building v1.1

```bash
uv run python -m scripts.build_v1_1
```

This applies [corrections/v1.1.json](corrections/v1.1.json) to the v1.0 file and writes the three
splits to `artifacts/v1.1/`. Set `YGO_TURNS_DIR` to a folder of splits to train on local files.

## The card table

BoardDSN reads `artifacts/cards.parquet`, which is not distributed because of copyright over card
effect text. Put a card file at `data/cards/cards.json` and build the table:

```bash
uv run python -m scripts.build_cards
```

The script keeps the cards that appear in the dataset, one entry per name, and compresses
`all-MiniLM-L6-v2` embeddings of their effect text to 32 dimensions with PCA. The v1.1 results use
the card table built for the paper.

### The card file

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

## Project structure

```text
data/          dataset columns, card loading, embeddings
domain/        card models and enumerations
model/         PyTorch modules
training/      datasets, Lightning modules, configuration
scripts/       entry points
corrections/   what changes between dataset releases
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
| `datamodule.py`     | `TurnsDataModule` — train and validation data, optional perspective-swap augmentation                |
| `module.py`         | `WinPredictorModule` — Lightning wrapper with BCE loss, label smoothing, AUROC/Brier logging         |
| `turns.py`          | Reads a dataset split from Hugging Face or a local folder                                            |
| `augmentation.py`   | `swap_perspective` — mirrors a batch by renaming `p1_*` ↔ `p2_*` and flipping targets                |

## Citation

If you use YGO-Turns, please cite the paper:

```bibtex
@inproceedings{dasilvasantos2026ygoturns,
  author    = {da Silva Santos, Rafael and da Costa, Leonardo Ferreira and Barros, Ana Luiza B. P. and de Campos, Gustavo A. L.},
  title     = {{YGO-Turns}: A Large-Scale Tournament Dataset of Expert {Yu-Gi-Oh!} Duel States},
  booktitle = {2026 IEEE Conference on Games (CoG)},
  year      = {2026},
  pages     = {1--4},
  publisher = {IEEE},
  doi       = {10.1109/CoG67430.2026.11778780}
}
```

## License

The code is under the [MIT License](LICENSE). The dataset is under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Yu-Gi-Oh! card names belong to their
owners; this project is not affiliated with them.
