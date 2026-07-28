"""
embedding.py — Generate vector representations of documents.

Strategy (hybrid, no external API needed):
  1. TF-IDF vectors for keyword-level matching (always available)
  2. Sentence-transformer embeddings for semantic understanding (if installed)
  3. Combined hybrid embedding when both are available

The module uses FAISS for efficient similarity search when available,
falling back to brute-force cosine similarity.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ─── Try importing optional heavy dependencies ────────────────────────────
_HAS_SENTENCE_TRANSFORMERS = False
_HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
    logger.info("sentence-transformers available — semantic embeddings enabled")
except ImportError:
    logger.info("sentence-transformers not installed — using TF-IDF only")

try:
    import faiss
    _HAS_FAISS = True
    logger.info("FAISS available — indexed search enabled")
except ImportError:
    logger.info("FAISS not installed — using brute-force similarity")


class EmbeddingEngine:
    """
    Generates and stores document embeddings.

    Supports three modes:
      - "tfidf": TF-IDF vectors only (always available)
      - "semantic": Sentence-transformer embeddings only
      - "hybrid": Weighted combination of both (default when available)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        semantic_model_name: str = "all-MiniLM-L6-v2",
        tfidf_weight: float = 0.35,
        semantic_weight: float = 0.65,
    ):
        self.mode = mode
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight

        # Downgrade mode if dependencies are missing
        if mode in ("semantic", "hybrid") and not _HAS_SENTENCE_TRANSFORMERS:
            logger.warning("Falling back to tfidf mode (sentence-transformers not installed)")
            self.mode = "tfidf"

        # Initialize TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),       # unigrams + bigrams
            sublinear_tf=True,        # apply log normalization
            min_df=1,
            stop_words="english",
        )

        # Initialize semantic model
        self.semantic_model = None
        if self.mode in ("semantic", "hybrid"):
            logger.info("Loading sentence-transformer: %s", semantic_model_name)
            self.semantic_model = SentenceTransformer(semantic_model_name)

        # Storage
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.semantic_matrix: Optional[np.ndarray] = None
        self.combined_matrix: Optional[np.ndarray] = None
        self.faiss_index = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit on corpus and return embedding matrix.
        Each row is a document vector, normalized to unit length.
        """
        n = len(texts)
        logger.info("Embedding %d documents in '%s' mode", n, self.mode)

        # ── TF-IDF ─────────────────────────────────────────────────────
        tfidf_sparse = self.tfidf.fit_transform(texts)
        self.tfidf_matrix = normalize(tfidf_sparse.toarray(), norm="l2")
        logger.info("TF-IDF matrix shape: %s", self.tfidf_matrix.shape)

        if self.mode == "tfidf":
            self.combined_matrix = self.tfidf_matrix
            self._build_faiss_index(self.combined_matrix)
            return self.combined_matrix

        # ── Semantic ───────────────────────────────────────────────────
        self.semantic_matrix = self.semantic_model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        logger.info("Semantic matrix shape: %s", self.semantic_matrix.shape)

        if self.mode == "semantic":
            self.combined_matrix = self.semantic_matrix
            self._build_faiss_index(self.combined_matrix)
            return self.combined_matrix

        # ── Hybrid: concatenate weighted vectors ───────────────────────
        tfidf_weighted = self.tfidf_matrix * self.tfidf_weight
        sem_weighted = self.semantic_matrix * self.semantic_weight
        self.combined_matrix = normalize(
            np.hstack([tfidf_weighted, sem_weighted]), norm="l2"
        )
        logger.info("Hybrid matrix shape: %s", self.combined_matrix.shape)
        self._build_faiss_index(self.combined_matrix)
        return self.combined_matrix

    def transform_query(self, query_text: str) -> np.ndarray:
        """
        Transform a single query (job description) using the fitted vectorizers.
        Returns a 1×D vector.
        """
        # TF-IDF
        tfidf_vec = normalize(
            self.tfidf.transform([query_text]).toarray(), norm="l2"
        )

        if self.mode == "tfidf":
            return tfidf_vec

        # Semantic
        sem_vec = self.semantic_model.encode(
            [query_text], normalize_embeddings=True
        )

        if self.mode == "semantic":
            return sem_vec

        # Hybrid
        tfidf_w = tfidf_vec * self.tfidf_weight
        sem_w = sem_vec * self.semantic_weight
        return normalize(np.hstack([tfidf_w, sem_w]), norm="l2")

    def _build_faiss_index(self, matrix: np.ndarray) -> None:
        """Build a FAISS index for fast nearest-neighbor search."""
        if not _HAS_FAISS:
            return
        d = matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)  # inner product = cosine for unit vecs
        self.faiss_index.add(matrix.astype(np.float32))
        logger.info("FAISS index built with %d vectors of dim %d", matrix.shape[0], d)

    def search_faiss(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index. Returns (scores, indices) arrays.
        Falls back to brute-force if FAISS not available.
        """
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(
                query_vec.astype(np.float32), top_k
            )
            return scores[0], indices[0]
        else:
            # Brute-force cosine similarity
            sims = (self.combined_matrix @ query_vec.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            return sims[top_idx], top_idx

    @property
    def embedding_dim(self) -> int:
        if self.combined_matrix is not None:
            return self.combined_matrix.shape[1]
        return 0
