from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARDS_PARQUET = PROJECT_ROOT / "artifacts" / "cards.parquet"

PROBE_CARDS = [
    "Dark Magician",
    "Blue-Eyes White Dragon",
    "Pot of Greed",
    "Mirror Force",
    "Monster Reborn",
    "Raigeki",
    "Mystical Space Typhoon",
    "Exodia the Forbidden One",
    "Swords of Revealing Light",
]

TOP_K = 6
console = Console()


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Computes cosine similarities between query vectors and a corpus.

    Args:
        query: Float array of shape (Q, D).
        corpus: Float array of shape (N, D).

    Returns:
        Float array of shape (Q, N) with cosine similarities.
    """
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True).clip(min=1e-8)
    return query_norm @ corpus_norm.T


def build_similarity_table(
    card_name: str, neighbours: list[tuple[str, float]]
) -> Table:
    table = Table(
        title=f"Nearest Neighbours — {card_name}", header_style="bold magenta"
    )
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Card", min_width=40)
    table.add_column("Similarity", justify="right")

    for rank, (name, sim) in enumerate(neighbours, start=1):
        table.add_row(str(rank), name, f"{sim:.4f}")
    return table


def main() -> None:
    df = pd.read_parquet(CARDS_PARQUET)

    embeddings = np.stack(df["effect_embedding"].tolist()).astype(np.float32)
    names = df["name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(names)}

    missing = [c for c in PROBE_CARDS if c not in name_to_idx]
    if missing:
        console.print(
            f"[bold yellow]WARNING:[/bold yellow] cards not found in parquet: {missing}\n"
        )

    probes = [c for c in PROBE_CARDS if c in name_to_idx]
    probe_vecs = embeddings[[name_to_idx[c] for c in probes]]
    sims = cosine_similarity_matrix(probe_vecs, embeddings)

    for i, card_name in enumerate(probes):
        row = sims[i]
        top_indices = [idx for idx in np.argsort(row)[::-1] if names[idx] != card_name][
            :TOP_K
        ]
        neighbours = [(names[idx], row[idx]) for idx in top_indices]
        console.print(build_similarity_table(card_name, neighbours))
        console.print()


if __name__ == "__main__":
    main()
