"""Unit tests for IRMetricsCalculator and AUTOSARBenchmarkEvaluator."""

import unittest
from pathlib import Path

from src.analyzers.evaluator import (
    AUTOSAR_SAMPLE_BENCHMARK,
    AUTOSARBenchmarkEvaluator,
    IRMetricsCalculator,
)
from src.analyzers.retrieval_engine import TFIDFRetrievalEngine
from src.parsers.chunker import DocumentChunker
from src.parsers.pdf_parser import PDFParser


class TestIRMetricsCalculator(unittest.TestCase):
    """Deterministic tests with small, manually verifiable ground truth calculations."""

    def setUp(self):
        self.calc = IRMetricsCalculator()

    def test_precision_at_k_calculation(self):
        """Precision@K = (relevant in top K) / K."""
        expected_pages = {1, 2}

        # 2 of 3 are relevant: [1, 3, 2] -> relevant pages are 1 and 2
        p_3 = self.calc.calculate_precision_at_k([1, 3, 2], expected_pages, k=3)
        self.assertAlmostEqual(p_3, 2.0 / 3.0, places=4)

        # 0 of 2 are relevant: [3, 4]
        p_0 = self.calc.calculate_precision_at_k([3, 4], expected_pages, k=2)
        self.assertEqual(p_0, 0.0)

        # 1 of 1 is relevant: [1]
        p_1 = self.calc.calculate_precision_at_k([1], expected_pages, k=1)
        self.assertEqual(p_1, 1.0)

        # Edge cases: empty retrieved or k <= 0
        self.assertEqual(self.calc.calculate_precision_at_k([], expected_pages, k=3), 0.0)
        self.assertEqual(self.calc.calculate_precision_at_k([1], expected_pages, k=0), 0.0)

    def test_recall_at_k_calculation(self):
        """Recall@K = (recovered unique target pages) / (total target pages)."""
        expected_pages = {1, 2}

        # Retrieved contains only page 1 -> recall is 1/2 = 0.5
        r_half = self.calc.calculate_recall_at_k([1, 1, 3], expected_pages, k=3)
        self.assertEqual(r_half, 0.5)

        # Retrieved contains both page 1 and 2 -> recall is 2/2 = 1.0
        r_full = self.calc.calculate_recall_at_k([1, 2, 3], expected_pages, k=3)
        self.assertEqual(r_full, 1.0)

        # Retrieved contains neither -> recall is 0.0
        r_zero = self.calc.calculate_recall_at_k([3, 4], expected_pages, k=2)
        self.assertEqual(r_zero, 0.0)

        # Edge case: empty expected pages
        self.assertEqual(self.calc.calculate_recall_at_k([1], set(), k=1), 1.0)
        self.assertEqual(self.calc.calculate_recall_at_k([], expected_pages, k=3), 0.0)

    def test_f1_score_calculation(self):
        """F1 = 2 * (P * R) / (P + R)."""
        # Perfect P and R
        self.assertEqual(self.calc.calculate_f1_at_k(1.0, 1.0), 1.0)

        # Zero P or R
        self.assertEqual(self.calc.calculate_f1_at_k(0.0, 1.0), 0.0)
        self.assertEqual(self.calc.calculate_f1_at_k(0.0, 0.0), 0.0)

        # P = 0.5, R = 1.0 -> 2 * 0.5 / 1.5 = 2/3 = 0.6667
        f1 = self.calc.calculate_f1_at_k(0.5, 1.0)
        self.assertAlmostEqual(f1, 2.0 / 3.0, places=4)

    def test_reciprocal_rank_calculation(self):
        """RR = 1 / rank of first relevant result."""
        expected_pages = {2}

        # First relevant at rank 1: [2, 1, 3] -> RR = 1.0
        rr1, rank1 = self.calc.calculate_reciprocal_rank([2, 1, 3], expected_pages)
        self.assertEqual(rr1, 1.0)
        self.assertEqual(rank1, 1)

        # First relevant at rank 2: [1, 2, 3] -> RR = 0.5
        rr2, rank2 = self.calc.calculate_reciprocal_rank([1, 2, 3], expected_pages)
        self.assertEqual(rr2, 0.5)
        self.assertEqual(rank2, 2)

        # First relevant at rank 3: [3, 1, 2] -> RR = 1/3
        rr3, rank3 = self.calc.calculate_reciprocal_rank([3, 1, 2], expected_pages)
        self.assertAlmostEqual(rr3, 1.0 / 3.0, places=4)
        self.assertEqual(rank3, 3)

        # Not found: [3, 4] -> RR = 0.0, rank = None
        rr0, rank0 = self.calc.calculate_reciprocal_rank([3, 4], expected_pages)
        self.assertEqual(rr0, 0.0)
        self.assertIsNone(rank0)

    def test_citation_accuracy_calculation(self):
        """Citation accuracy is the fraction of top-K citations pointing to expected pages."""
        expected_pages = {1, 3}
        self.assertEqual(self.calc.calculate_citation_accuracy([1, 3], expected_pages, k=2), 1.0)
        self.assertEqual(self.calc.calculate_citation_accuracy([1, 2], expected_pages, k=2), 0.5)
        self.assertEqual(self.calc.calculate_citation_accuracy([2, 4], expected_pages, k=2), 0.0)


class TestAUTOSARBenchmarkEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Index the sample document to evaluate real retrieval against benchmark."""
        sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {sample_path}")

        parser = PDFParser()
        pages = parser.extract_pages(sample_path)
        chunker = DocumentChunker(chunk_size=45, chunk_overlap=10)
        cls.chunks, _ = chunker.chunk_pages(pages, source_file=sample_path.name)

        cls.engine = TFIDFRetrievalEngine(default_threshold=0.05)
        cls.engine.index_chunks(cls.chunks)
        cls.evaluator = AUTOSARBenchmarkEvaluator()

    def test_benchmark_dataset_integrity(self):
        """Verify all 10 benchmark queries are strictly grounded in 3-page sample HLD."""
        self.assertEqual(len(AUTOSAR_SAMPLE_BENCHMARK), 10)
        for bq in AUTOSAR_SAMPLE_BENCHMARK:
            self.assertTrue(bq.query_id.startswith("Q"))
            self.assertGreater(len(bq.query_text), 10)
            self.assertGreater(len(bq.expected_target_pages), 0)
            for p in bq.expected_target_pages:
                self.assertIn(p, [1, 2, 3])
            self.assertGreater(len(bq.expected_entities), 0)
            self.assertGreater(len(bq.ground_truth_answer), 10)

    def test_evaluate_engine_baseline_tfidf(self):
        """Runs evaluation on baseline TF-IDF engine and verifies metrics are computed."""
        report = self.evaluator.evaluate_engine(
            engine=self.engine,
            k=3,
            threshold=0.05,
            use_pca=False,
        )

        self.assertEqual(report.total_queries, 10)
        self.assertEqual(report.k, 3)
        self.assertEqual(report.retrieval_method, "Baseline TF-IDF")
        self.assertEqual(len(report.query_results), 10)

        # Check that metrics are in valid probability ranges
        self.assertGreaterEqual(report.mean_precision_at_k, 0.0)
        self.assertLessEqual(report.mean_precision_at_k, 1.0)
        self.assertGreaterEqual(report.mean_recall_at_k, 0.0)
        self.assertLessEqual(report.mean_recall_at_k, 1.0)
        self.assertGreaterEqual(report.mrr, 0.0)
        self.assertLessEqual(report.mrr, 1.0)
        self.assertGreaterEqual(report.mean_citation_accuracy, 0.0)

        # Baseline TF-IDF on our specific queries should achieve high relevance
        self.assertGreater(report.mrr, 0.5)

    def test_evaluate_engine_with_svd_comparison(self):
        """Evaluates SVD-reduced retrieval against the exact same benchmark queries."""
        report_svd = self.evaluator.evaluate_engine(
            engine=self.engine,
            k=3,
            threshold=0.05,
            use_pca=True,
            pca_components=4,
        )

        self.assertEqual(report_svd.total_queries, 10)
        self.assertIn("Truncated SVD", report_svd.retrieval_method)
        self.assertEqual(len(report_svd.query_results), 10)
        self.assertGreaterEqual(report_svd.mrr, 0.0)
        self.assertLessEqual(report_svd.mrr, 1.0)


if __name__ == "__main__":
    unittest.main()
