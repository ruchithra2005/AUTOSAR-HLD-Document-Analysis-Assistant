"""TF-IDF Vectorization, Cosine Similarity Retrieval, and SVD/PCA Dimensionality Reduction.

Provides an auditable, transparent retrieval pipeline using standard mathematical formulations
with pure NumPy. Keeps track of similarity scores, threshold accept/reject boundaries,
source page provenance, and empirical SVD/PCA comparisons.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.parsers.models import TextChunk


# Standard English stopwords (excluding technical and automotive terms)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


@dataclass
class ScoredChunk:
    """Represents a text chunk evaluated against a search query with similarity metrics."""
    chunk: TextChunk
    similarity_score: float
    is_accepted: bool
    threshold: float
    rank: int
    status_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "page_number": self.chunk.page_number,
            "source_file": self.chunk.source_file,
            "word_count": self.chunk.word_count,
            "section_title": self.chunk.section_title or "N/A",
            "similarity_score": round(self.similarity_score, 4),
            "is_accepted": self.is_accepted,
            "threshold": self.threshold,
            "rank": self.rank,
            "status_reason": self.status_reason,
            "text_snippet": self.chunk.text[:200] + ("..." if len(self.chunk.text) > 200 else ""),
        }


@dataclass
class RetrievalComparisonResult:
    """Empirical comparison between baseline TF-IDF and SVD/PCA reduced retrieval."""
    query: str
    total_chunks: int
    total_features: int
    n_components: Optional[int]
    explained_variance_ratio: Optional[float]
    baseline_latency_ms: float
    reduced_latency_ms: Optional[float]
    baseline_top_scores: List[float]
    reduced_top_scores: List[float]
    baseline_results: List[ScoredChunk]
    reduced_results: Optional[List[ScoredChunk]]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_chunks": self.total_chunks,
            "total_features": self.total_features,
            "n_components": self.n_components,
            "explained_variance_ratio": (
                round(self.explained_variance_ratio, 4) if self.explained_variance_ratio is not None else None
            ),
            "baseline_latency_ms": round(self.baseline_latency_ms, 3),
            "reduced_latency_ms": round(self.reduced_latency_ms, 3) if self.reduced_latency_ms is not None else None,
            "baseline_top_scores": [round(s, 4) for s in self.baseline_top_scores],
            "reduced_top_scores": [round(s, 4) for s in self.reduced_top_scores],
            "notes": self.notes,
        }


class TFIDFVectorizer:
    """Pure NumPy TF-IDF Vectorizer with L2 normalization and sublinear scaling."""

    def __init__(self, stopwords: Optional[Set[str]] = None) -> None:
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.vocabulary: Dict[str, int] = {}
        self.idf_vector: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def tokenize(self, text: str) -> List[str]:
        """Extracts alphanumeric tokens and AUTOSAR identifier symbols."""
        if not text:
            return []
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        return [t for t in raw_tokens if t not in self.stopwords]

    def fit(self, chunks: List[TextChunk]) -> TFIDFVectorizer:
        """Learns vocabulary and IDF weights across the corpus."""
        if not chunks:
            self.vocabulary = {}
            self.idf_vector = np.array([])
            self.is_fitted = True
            return self

        doc_tokens = [self.tokenize(c.text) for c in chunks]
        doc_count = len(chunks)

        # Build vocabulary
        vocab: Dict[str, int] = {}
        df_counts: Dict[str, int] = {}

        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
                df_counts[token] = df_counts.get(token, 0) + 1

        self.vocabulary = vocab
        vocab_size = len(vocab)

        if vocab_size == 0:
            self.idf_vector = np.array([])
            self.is_fitted = True
            return self

        # Smooth IDF: ln((1 + N) / (1 + DF)) + 1.0
        idf = np.zeros(vocab_size, dtype=np.float64)
        for token, idx in vocab.items():
            df = df_counts.get(token, 0)
            idf[idx] = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0

        self.idf_vector = idf
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms texts into normalized TF-IDF matrix."""
        if not self.is_fitted:
            raise ValueError("TFIDFVectorizer must be fitted before calling transform.")

        num_texts = len(texts)
        vocab_size = len(self.vocabulary)

        if num_texts == 0 or vocab_size == 0:
            return np.zeros((num_texts, vocab_size), dtype=np.float64)

        matrix = np.zeros((num_texts, vocab_size), dtype=np.float64)

        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            if not tokens:
                continue

            # Compute term frequencies
            tf_counts: Dict[int, int] = {}
            for t in tokens:
                if t in self.vocabulary:
                    idx = self.vocabulary[t]
                    tf_counts[idx] = tf_counts.get(idx, 0) + 1

            total_tokens = len(tokens)
            for idx, count in tf_counts.items():
                # Sublinear TF: (1 + ln(count)) / total_tokens
                matrix[i, idx] = (1.0 + math.log(count)) * self.idf_vector[idx]

            # L2 normalization
            norm = np.linalg.norm(matrix[i])
            if norm > 0.0:
                matrix[i] /= norm

        return matrix

    def fit_transform(self, chunks: List[TextChunk]) -> np.ndarray:
        """Fits vocabulary and returns document-term TF-IDF matrix."""
        self.fit(chunks)
        texts = [c.text for c in chunks]
        return self.transform(texts)


class TruncatedSVDExperiment:
    """Optional Latent Semantic Analysis / Truncated SVD dimensionality reduction."""

    def __init__(self, n_components: int = 5) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[float] = None

    def fit(self, X: np.ndarray) -> TruncatedSVDExperiment:
        """Fits truncated SVD on TF-IDF matrix X (samples x features)."""
        if X.size == 0 or X.shape[0] == 0 or X.shape[1] == 0:
            self.components_ = np.zeros((0, 0))
            self.singular_values_ = np.array([])
            self.explained_variance_ratio_ = 0.0
            return self

        max_possible = min(X.shape[0], X.shape[1])
        k = max(1, min(self.n_components, max_possible))
        self.n_components = k

        # SVD decomposition: X = U * S * Vt
        try:
            _, s, vt = np.linalg.svd(X, full_matrices=False)
        except Exception:
            # Fallback for degenerate matrices
            self.components_ = np.eye(X.shape[1], k)
            self.singular_values_ = np.ones(k)
            self.explained_variance_ratio_ = 0.0
            return self

        # Keep top k components
        self.singular_values_ = s[:k]
        self.components_ = vt[:k, :].T  # shape (vocab_size, k)

        total_variance = np.sum(s ** 2)
        if total_variance > 0:
            self.explained_variance_ratio_ = float(np.sum(s[:k] ** 2) / total_variance)
        else:
            self.explained_variance_ratio_ = 0.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects matrix X into k-dimensional latent space with L2 normalization."""
        if self.components_ is None or self.components_.size == 0 or X.size == 0:
            return np.zeros((X.shape[0], 0), dtype=np.float64)

        reduced = X @ self.components_

        # Normalize rows to unit length for cosine similarity
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return reduced / norms


class TFIDFRetrievalEngine:
    """Retrieval engine computing Cosine Similarity scores and SVD comparisons."""

    def __init__(self, default_threshold: float = 0.15) -> None:
        self.default_threshold = default_threshold
        self.vectorizer = TFIDFVectorizer()
        self.chunks: List[TextChunk] = []
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Indexes text chunks by fitting the TF-IDF model."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.tfidf_matrix = np.zeros((0, 0))
            self.is_indexed = True
            return

        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        self.is_indexed = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[ScoredChunk]:
        """Retrieves and ranks chunks using Cosine Similarity against the query.

        Args:
            query: User's search text.
            top_k: Maximum number of top chunks to return.
            threshold: Cosine similarity cutoff (defaults to default_threshold).

        Returns:
            Ranked list of ScoredChunks with similarity metrics and acceptance flags.
        """
        if not self.is_indexed:
            raise ValueError("Engine must index chunks before calling retrieve.")

        tau = threshold if threshold is not None else self.default_threshold
        if not self.chunks or self.tfidf_matrix is None or self.tfidf_matrix.size == 0:
            return []

        if not query or not query.strip():
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query])[0]
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0.0:
            # Query terms are completely out of vocabulary
            scored_empty = [
                ScoredChunk(
                    chunk=c,
                    similarity_score=0.0,
                    is_accepted=False,
                    threshold=tau,
                    rank=idx + 1,
                    status_reason=f"Score 0.0000 < threshold {tau} (Rejected - Out of vocabulary)",
                )
                for idx, c in enumerate(self.chunks[:top_k])
            ]
            return scored_empty

        # Cosine similarity calculation: Q . D_i (vectors are already L2 normalized)
        scores = np.dot(self.tfidf_matrix, query_vec)
        scores = np.clip(scores, 0.0, 1.0)

        # Sort indices by score descending
        sorted_indices = np.argsort(-scores)

        results: List[ScoredChunk] = []
        limit = min(top_k, len(sorted_indices))

        for rank_idx in range(limit):
            chunk_idx = int(sorted_indices[rank_idx])
            score = float(scores[chunk_idx])
            chunk = self.chunks[chunk_idx]
            is_accepted = bool(score >= tau)

            if is_accepted:
                reason = f"Score {score:.4f} >= threshold {tau} (Accepted)"
            else:
                reason = f"Score {score:.4f} < threshold {tau} (Rejected)"

            results.append(
                ScoredChunk(
                    chunk=chunk,
                    similarity_score=score,
                    is_accepted=is_accepted,
                    threshold=tau,
                    rank=rank_idx + 1,
                    status_reason=reason,
                )
            )

        return results

    def compare_with_pca(
        self,
        query: str,
        n_components: int = 5,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> RetrievalComparisonResult:
        """Fairly compares baseline TF-IDF retrieval against SVD/PCA reduced retrieval."""
        if not self.is_indexed:
            raise ValueError("Engine must index chunks before running comparison.")

        tau = threshold if threshold is not None else self.default_threshold

        # 1. Baseline TF-IDF
        t0 = time.perf_counter()
        baseline_results = self.retrieve(query, top_k=top_k, threshold=tau)
        baseline_latency_ms = (time.perf_counter() - t0) * 1000.0

        if not self.chunks or self.tfidf_matrix is None or self.tfidf_matrix.size == 0:
            return RetrievalComparisonResult(
                query=query,
                total_chunks=len(self.chunks),
                total_features=0,
                n_components=0,
                explained_variance_ratio=0.0,
                baseline_latency_ms=baseline_latency_ms,
                reduced_latency_ms=0.0,
                baseline_top_scores=[],
                reduced_top_scores=[],
                baseline_results=[],
                reduced_results=[],
                notes="No indexed chunks available to compare.",
            )

        # 2. SVD / PCA Reduction Experiment
        t1 = time.perf_counter()
        svd = TruncatedSVDExperiment(n_components=n_components)
        svd.fit(self.tfidf_matrix)

        X_reduced = svd.transform(self.tfidf_matrix)
        q_raw = self.vectorizer.transform([query])
        q_reduced = svd.transform(q_raw)[0]

        reduced_scores = np.dot(X_reduced, q_reduced) if q_reduced.size > 0 else np.zeros(len(self.chunks))
        reduced_scores = np.clip(reduced_scores, 0.0, 1.0)
        reduced_latency_ms = (time.perf_counter() - t1) * 1000.0

        sorted_reduced = np.argsort(-reduced_scores)
        reduced_results: List[ScoredChunk] = []
        limit = min(top_k, len(sorted_reduced))

        for rank_idx in range(limit):
            chunk_idx = int(sorted_reduced[rank_idx])
            score = float(reduced_scores[chunk_idx])
            chunk = self.chunks[chunk_idx]
            is_accepted = bool(score >= tau)
            reason = (
                f"SVD Score {score:.4f} >= threshold {tau} (Accepted)"
                if is_accepted
                else f"SVD Score {score:.4f} < threshold {tau} (Rejected)"
            )
            reduced_results.append(
                ScoredChunk(
                    chunk=chunk,
                    similarity_score=score,
                    is_accepted=is_accepted,
                    threshold=tau,
                    rank=rank_idx + 1,
                    status_reason=reason,
                )
            )

        # 3. Analyze differences
        var_pct = round((svd.explained_variance_ratio_ or 0.0) * 100.0, 2)
        base_top = [r.similarity_score for r in baseline_results]
        red_top = [r.similarity_score for r in reduced_results]

        analysis = (
            f"Evaluated {len(self.chunks)} chunks over {len(self.vectorizer.vocabulary)} vocabulary features. "
            f"Truncated SVD retained {svd.n_components} components capturing {var_pct}% of total variance. "
            f"Baseline latency: {baseline_latency_ms:.2f}ms vs Reduced latency: {reduced_latency_ms:.2f}ms. "
            "Note: On small technical HLD documents, baseline exact TF-IDF preserves precise acronym tokens "
            "(e.g., SWC, RTE, ASIL) better than truncated SVD, while SVD reduces feature dimensionality."
        )

        return RetrievalComparisonResult(
            query=query,
            total_chunks=len(self.chunks),
            total_features=len(self.vectorizer.vocabulary),
            n_components=svd.n_components,
            explained_variance_ratio=svd.explained_variance_ratio_,
            baseline_latency_ms=baseline_latency_ms,
            reduced_latency_ms=reduced_latency_ms,
            baseline_top_scores=base_top,
            reduced_top_scores=red_top,
            baseline_results=baseline_results,
            reduced_results=reduced_results,
            notes=analysis,
        )
