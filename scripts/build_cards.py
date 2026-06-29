from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

from data.cards.loader import load_raw_cards
from data.cards.writer import write_cards_parquet
from data.embedding.encoder import (
    DEFAULT_EMBEDDING_MODEL,
    compress_embeddings,
    compute_embeddings,
)
from data.preprocessing.cards import embedding_text
from data.turns.reader import collect_card_names

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("build_cards")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "cards" / "cards.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "cards.parquet"


def _deduplicate_by_name(
    raw_cards: list,
) -> list:
    """Retains the first occurrence of each card name, resolved by ascending ID.

    When cards.json contains multiple entries per card name for different
    printings, keeping the lowest ID preserves the original print and ensures
    CardLookup receives a single deterministic entry per name.

    Args:
        raw_cards: List of raw card models as loaded from cards.json.

    Returns:
        Deduplicated list with one entry per unique card name.
    """
    seen: set[str] = set()
    deduped = []
    for card in sorted(raw_cards, key=lambda c: c.id):
        if card.name not in seen:
            seen.add(card.name)
            deduped.append(card)
    return deduped


def build_cards_dataset(
    json_path: Path = DEFAULT_JSON_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    turns_path: Path = PROJECT_ROOT / "artifacts" / "ygo-turns.parquet",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    n_components: int = 32,
) -> None:
    """Builds the cards parquet dataset with PCA-compressed effect embeddings.

    Card vocabulary is automatically restricted to names that appear in the
    turns parquet, discarding all cards irrelevant to the training format.

    Args:
        json_path: Path to the cards JSON file.
        output_path: Destination path for the output Parquet file.
        turns_path: Path to the preprocessed turns parquet used to derive
            the set of cards seen in actual games.
        model_name: Sentence-transformers model name for embedding generation.
        n_components: Target dimensionality for PCA compression of raw embeddings.
    """
    allowed_names = collect_card_names(turns_path)
    log.info("Unique card names found in turns: %d", len(allowed_names))

    raw_cards = load_raw_cards(json_path)
    raw_cards = [c for c in raw_cards if c.name in allowed_names]
    log.info("Cards retained after filtering: %d", len(raw_cards))

    raw_cards = _deduplicate_by_name(raw_cards)
    log.info("Cards retained after deduplication: %d", len(raw_cards))

    texts = [embedding_text(card) for card in raw_cards]
    embeddings = compute_embeddings(texts, model_name)
    embeddings = compress_embeddings(embeddings, n_components)

    write_cards_parquet(
        raw_cards=raw_cards,
        embeddings=embeddings,
        output_path=output_path,
    )
    log.info(
        "cards.parquet written to [bold]%s[/]", output_path, extra={"markup": True}
    )


def main() -> None:
    """CLI entrypoint."""
    build_cards_dataset()


if __name__ == "__main__":
    main()
