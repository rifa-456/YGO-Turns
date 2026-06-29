from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

log = logging.getLogger(__name__)


def compute_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Encodes texts into embedding vectors.

    Args:
        texts: List of strings to encode.
        model_name: Sentence-transformers model name.

    Returns:
        Float array of shape (N, D).
    """
    log.info("Loading embedding model [bold]%s[/]", model_name, extra={"markup": True})
    model = SentenceTransformer(model_name)

    log.info("Encoding %d texts", len(texts))
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def compress_embeddings(
    embeddings: np.ndarray,
    n_components: int = 32,
) -> np.ndarray:
    """Reduces embedding dimensionality via PCA fitted on the full card corpus.

    Args:
        embeddings: Float array of shape (N, D) produced by compute_embeddings.
        n_components: Target dimensionality after compression.

    Returns:
        Float32 array of shape (N, n_components).
    """
    from sklearn.decomposition import PCA

    log.info("Compressing embeddings to %d dimensions via PCA", n_components)
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(embeddings).astype(np.float32)
