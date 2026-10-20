"""Unit tests for TFIDFVectorizer, Cosine Similarity Retrieval, and SVD/PCA Comparison."""

import unittest
from pathlib import Path

import numpy as np

from src.analyzers.retrieval_engine import (
    ScoredChunk,
    TFIDFRetrievalEngine,
    TFIDFVectorizer,
    TruncatedSVDExperiment,
)
from src.parsers.chunker import DocumentChunker
from src.parsers.models import TextChunk
from src.parsers.pdf_parser import PDFParser


class TestTFIDFVectorizer(unittest.TestCase):
    def setUp(self):
        self.vectorizer = TFIDFVectorizer()
        self.sample_chunks = [
            TextChunk(
                chunk_id="chunk_1",
                text="The SWC_EngineControl component implements ppEngineSpeed port.",
                page_number=1,
                source_file="test.pdf",
                word_count=7,
                char_count=60,
                chunk_index=1,
            ),
            TextChunk(
                chunk_id="chunk_2",
                text="Runnable RE_CalculateTorque is triggered by 10ms timing event.",
                page_number=2,
                source_file="test.pdf",
                word_count=8,
                char_count=65,
                chunk_index=2,
            ),
        ]

    def test_tokenization_preserves_autosar_identifiers(self):
        """Tokens should preserve camelCase and underscore identifiers while ignoring stop words."""
        tokens = self.vectorizer.tokenize("The SWC_EngineControl and ppEngineSpeed are in the HLD.")
        self.assertIn("swc_enginecontrol", tokens)
        self.assertIn("ppenginespeed", tokens)
        self.assertIn("hld", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)
        self.assertNotIn("are", tokens)

    def test_fit_transform_shapes_and_normalization(self):
        """Matrix shape matches (num_chunks, vocab_size) and rows are unit normalized."""
        X = self.vectorizer.fit_transform(self.sample_chunks)
        self.assertEqual(X.shape[0], 2)
        self.assertGreater(X.shape[1], 0)

        # Check unit L2 normalization of rows
        for row in X:
            norm = np.linalg.norm(row)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_corpus_handling(self):
        """Fitting on empty corpus produces empty matrix without errors."""
        X = self.vectorizer.fit_transform([])
        self.assertEqual(X.shape, (0, 0))


class TestTFIDFRetrievalEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Chunk the real sample AUTOSAR HLD document for realistic retrieval tests."""
        sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {sample_path}")

        parser = PDFParser()
        pages = parser.extract_pages(sample_path)
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        cls.chunks, _ = chunker.chunk_pages(pages, source_file=sample_path.name)

        cls.engine = TFIDFRetrievalEngine(default_threshold=0.15)
        cls.engine.index_chunks(cls.chunks)

    def test_exact_query_retrieval_and_ranking(self):
        """Relevant query retrieves target chunk at top rank with high similarity."""
        query = "ppVehicleSpeed Sender-Receiver Interface IF_VehicleSpeed"
        results = self.engine.retrieve(query, top_k=3, threshold=0.10)

        self.assertGreater(len(results), 0)
        top_result = results[0]

        # Top chunk should originate from Page 2 where ppVehicleSpeed is defined
        self.assertEqual(top_result.chunk.page_number, 2)
        self.assertEqual(top_result.rank, 1)
        self.assertGreater(top_result.similarity_score, 0.20)
        self.assertTrue(top_result.is_accepted)
        self.assertIn("Accepted", top_result.status_reason)

    def test_threshold_cutoff_rejection(self):
        """Chunks scoring below the threshold are marked as rejected."""
        query = "VehicleSpeedManagement_SWC"
        # High threshold to test rejection on lower-scoring chunks
        high_threshold = 0.85
        results = self.engine.retrieve(query, top_k=5, threshold=high_threshold)

        self.assertGreater(len(results), 1)
        for res in results:
            if res.similarity_score < high_threshold:
                self.assertFalse(res.is_accepted)
                self.assertIn("Rejected", res.status_reason)
            else:
                self.assertTrue(res.is_accepted)
                self.assertIn("Accepted", res.status_reason)

    def test_out_of_vocabulary_query(self):
        """Query with terms not in document returns zero score and is_accepted=False."""
        query = "QuantumComputing Superconductor XYZ123"
        results = self.engine.retrieve(query, top_k=3, threshold=0.15)

        for res in results:
            self.assertEqual(res.similarity_score, 0.0)
            self.assertFalse(res.is_accepted)
            self.assertIn("Out of vocabulary", res.status_reason)

    def test_empty_query_returns_empty_results(self):
        """Empty or whitespace queries return empty result list."""
        self.assertEqual(self.engine.retrieve("", top_k=5), [])
        self.assertEqual(self.engine.retrieve("   ", top_k=5), [])

    def test_provenance_preservation(self):
        """Retrieved chunks preserve exact source file, page number, and chunk ID."""
        results = self.engine.retrieve("Dem Diagnostic Event Manager", top_k=2, threshold=0.05)
        for r in results:
            self.assertEqual(r.chunk.source_file, "sample_autosar_hld.pdf")
            self.assertIn(r.chunk.page_number, [1, 2, 3])
            self.assertTrue(len(r.chunk.chunk_id) > 0)


class TestTruncatedSVDExperiment(unittest.TestCase):
    def setUp(self):
        # Create a small term-document matrix (4 documents x 6 terms)
        np.random.seed(42)
        self.X = np.array([
            [1.0, 0.5, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.8, 0.9, 0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.7, 0.8, 0.3],
            [0.0, 0.1, 0.4, 0.0, 0.0, 0.9],
        ])

    def test_svd_dimension_reduction_and_variance(self):
        """SVD reduces features to k components and calculates explained variance ratio."""
        k = 2
        svd = TruncatedSVDExperiment(n_components=k)
        svd.fit(self.X)

        self.assertEqual(svd.n_components, k)
        self.assertIsNotNone(svd.explained_variance_ratio_)
        self.assertGreater(svd.explained_variance_ratio_, 0.0)
        self.assertLessEqual(svd.explained_variance_ratio_, 1.0)

        # Projected shape: (4 samples, 2 components)
        X_reduced = svd.transform(self.X)
        self.assertEqual(X_reduced.shape, (4, k))

    def test_excessive_components_clamped(self):
        """Requesting more components than min(samples, features) is safely clamped."""
        svd = TruncatedSVDExperiment(n_components=50)
        svd.fit(self.X)
        # Max possible is min(4, 6) = 4
        self.assertEqual(svd.n_components, 4)

    def test_empty_matrix_handling(self):
        """Handles empty matrix safely."""
        svd = TruncatedSVDExperiment(n_components=3)
        svd.fit(np.zeros((0, 0)))
        self.assertEqual(svd.explained_variance_ratio_, 0.0)
        reduced = svd.transform(np.zeros((0, 0)))
        self.assertEqual(reduced.shape, (0, 0))

    def test_compare_with_pca_execution(self):
        """Runs baseline vs SVD comparison on real indexed chunks and outputs empirical stats."""
        sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        parser = PDFParser()
        pages = parser.extract_pages(sample_path)
        chunker = DocumentChunker(chunk_size=40, chunk_overlap=10)
        chunks, _ = chunker.chunk_pages(pages, source_file="sample.pdf")

        engine = TFIDFRetrievalEngine(default_threshold=0.10)
        engine.index_chunks(chunks)

        comparison = engine.compare_with_pca("RE_CalculateSpeed timing event 10ms", n_components=3, top_k=3)

        self.assertGreater(comparison.total_chunks, 0)
        self.assertGreater(comparison.total_features, 0)
        self.assertEqual(comparison.n_components, 3)
        self.assertGreater(comparison.explained_variance_ratio, 0.0)
        self.assertGreaterEqual(comparison.baseline_latency_ms, 0.0)
        self.assertGreaterEqual(comparison.reduced_latency_ms, 0.0)
        self.assertEqual(len(comparison.baseline_results), 3)
        self.assertEqual(len(comparison.reduced_results), 3)
        self.assertIn("Truncated SVD retained", comparison.notes)


if __name__ == "__main__":
    unittest.main()
