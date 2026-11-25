# Changelog

## 1.1

The v1.0 file the paper uses stays available under the `v1.0` tag. The 241 columns are the same.

### Fixed

- **Winners:** v1.0 marked all 31 duels without a recorded winner as lost by P1. The replays name
  the winner of 26 of them, 8 of which P1 won (110 rows relabelled). The other 5 ended in a
  disconnection and are removed (71 rows).
- **Repeated matches:** matches 222 and 494 were copies of 221 and 488 and are removed (95 rows).
- **Merged duels:** 15 duels held the turns of two duels, mostly a draw and the duel played after
  it, and are removed (354 rows).
- **Tokens:** token monsters were named after the picture the player chose for them, such as `1` or
  `7-3`. They are now named `Token` (4,114 cells).

### Added

- Train, validation and test splits by match. Validation has the same 4,673 rows as in the paper.

### Checks

- Replays were available for 676 of the 682 matches. Every row count and every label of those
  matches agrees with its replay. The 15 duels of the other 6 matches could not be checked.

### Code

- The dataset is read from Hugging Face, or from the folder in `YGO_TURNS_DIR`.
- The dataset file and the v1.0 models left the repository; both are in the `v1.0` tag.
- `scripts/build_v1_1.py` builds v1.1 from v1.0 and `corrections/v1.1.json`.
- `scripts/evaluate.py` scores the trained models.

## 1.0

The dataset, code and models of the paper.
