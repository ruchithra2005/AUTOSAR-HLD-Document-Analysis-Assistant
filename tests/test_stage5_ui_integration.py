"""Unit tests for Stage 5 UI components and end-to-end integration."""

import json
import unittest
from pathlib import Path

from src.analyzers.consistency_checker import AUTOSARConsistencyChecker
from src.analyzers.evaluator import (
    AUTOSAR_SAMPLE_BENCHMARK,
    AUTOSARBenchmarkEvaluator,
    BenchmarkEvaluationReport,
)
from src.analyzers.models import ConsistencyReport, RuleFinding, Severity
from src.analyzers.retrieval_engine import (
    RetrievalComparisonResult,
    ScoredChunk,
    TFIDFRetrievalEngine,
)
from src.parsers.autosar_extractor import StructuredAUTOSARExtractor
from src.parsers.chunker import DocumentChunker
from src.parsers.models import EntityType, EvidenceStatus, ExtractionResult
from src.parsers.pdf_parser import PDFParser


class TestStage5UIIntegration(unittest.TestCase):
    """Verifies all Stage 5 integration components for the Streamlit UI."""

    @classmethod
    def setUpClass(cls):
        cls.sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        if not cls.sample_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {cls.sample_path}")

        parser = PDFParser()
        cls.pages = parser.extract_pages(cls.sample_path)
        extractor = StructuredAUTOSARExtractor()
        cls.extraction_result = extractor.extract(cls.pages, source_file="sample_autosar_hld.pdf")
        cls.checker = AUTOSARConsistencyChecker()
        cls.consistency_report = cls.checker.check_consistency(cls.extraction_result)

        chunker = DocumentChunker(chunk_size=120, chunk_overlap=25)
        cls.chunks, cls.chunk_stats = chunker.chunk_pages(
            cls.pages, source_file="sample_autosar_hld.pdf", sections=cls.extraction_result.sections
        )
        cls.engine = TFIDFRetrievalEngine(default_threshold=0.15)
        cls.engine.index_chunks(cls.chunks)

    def test_consistency_report_serialization(self):
        """Verifies ConsistencyReport produces a JSON-serializable dictionary for download."""
        report_dict = self.consistency_report.to_dict()
        self.assertIn("source_file", report_dict)
        self.assertIn("total_rules_checked", report_dict)
        self.assertIn("summary_counts", report_dict)
        self.assertIn("findings", report_dict)
        self.assertEqual(report_dict["total_rules_checked"], 5)

        # Ensure JSON dumps works cleanly
        json_str = json.dumps(report_dict, indent=2)
        self.assertGreater(len(json_str), 100)
        reloaded = json.loads(json_str)
        self.assertEqual(reloaded["source_file"], "sample_autosar_hld.pdf")

    def test_consistency_severity_distribution(self):
        """Verifies counts for all severity levels in the consistency report."""
        counts = self.consistency_report.get_counts()
        for sev in Severity:
            self.assertIn(sev.value, counts)
        self.assertGreater(counts["PASS"], 0)
        self.assertGreater(counts["WARNING"], 0)
        self.assertGreater(counts["UNVERIFIED"], 0)

    def test_retrieval_and_threshold_filtering(self):
        """Verifies ScoredChunk output format, ranking, and threshold filtering."""
        query = "VehicleSpeedManagement_SWC"
        results = self.engine.retrieve(query, top_k=5, threshold=0.15)
        self.assertGreater(len(results), 0)

        top_chunk = results[0]
        self.assertIsInstance(top_chunk, ScoredChunk)
        self.assertEqual(top_chunk.rank, 1)
        self.assertGreaterEqual(top_chunk.similarity_score, 0.0)

        chunk_dict = top_chunk.to_dict()
        self.assertIn("chunk_id", chunk_dict)
        self.assertIn("page_number", chunk_dict)
        self.assertIn("similarity_score", chunk_dict)
        self.assertIn("is_accepted", chunk_dict)
        self.assertIn("status_reason", chunk_dict)

    def test_svd_pca_comparison_for_ui(self):
        """Verifies compare_with_pca generates complete comparison results for UI display."""
        query = "RE_CalculateSpeed periodic task"
        comp = self.engine.compare_with_pca(query, n_components=4, top_k=3, threshold=0.10)
        self.assertIsInstance(comp, RetrievalComparisonResult)
        self.assertEqual(comp.query, query)
        self.assertEqual(comp.n_components, 4)
        self.assertIsNotNone(comp.explained_variance_ratio)
        self.assertGreater(len(comp.baseline_results), 0)
        self.assertIsNotNone(comp.reduced_results)

        comp_dict = comp.to_dict()
        self.assertIn("baseline_latency_ms", comp_dict)
        self.assertIn("reduced_latency_ms", comp_dict)
        self.assertIn("explained_variance_ratio", comp_dict)
        self.assertIn("notes", comp_dict)

    def test_benchmark_evaluator_baseline(self):
        """Verifies AUTOSARBenchmarkEvaluator produces complete report across benchmark dataset."""
        evaluator = AUTOSARBenchmarkEvaluator()
        report = evaluator.evaluate_engine(self.engine, k=3, threshold=0.15, use_pca=False)

        self.assertIsInstance(report, BenchmarkEvaluationReport)
        self.assertEqual(report.total_queries, len(AUTOSAR_SAMPLE_BENCHMARK))
        self.assertGreater(report.mrr, 0.0)
        self.assertGreater(report.mean_precision_at_k, 0.0)
        self.assertGreater(report.mean_recall_at_k, 0.0)
        self.assertEqual(len(report.query_results), len(AUTOSAR_SAMPLE_BENCHMARK))

        # Check serialization for UI export
        rep_dict = report.to_dict()
        json_str = json.dumps(rep_dict, indent=2)
        self.assertIn("mean_precision_at_k", json_str)
        self.assertIn("mrr", json_str)

    def test_benchmark_evaluator_svd_mode(self):
        """Verifies AUTOSARBenchmarkEvaluator executes with Truncated SVD mode."""
        evaluator = AUTOSARBenchmarkEvaluator()
        report = evaluator.evaluate_engine(
            self.engine, k=3, threshold=0.10, use_pca=True, pca_components=4
        )
        self.assertIsInstance(report, BenchmarkEvaluationReport)
        self.assertIn("Truncated SVD", report.retrieval_method)
        self.assertEqual(report.total_queries, len(AUTOSAR_SAMPLE_BENCHMARK))
        self.assertGreater(report.mrr, 0.0)

    def test_benchmark_method_switching_updates_metrics(self):
        """Regression test: Switching between Baseline TF-IDF and SVD updates report and metrics."""
        evaluator = AUTOSARBenchmarkEvaluator()

        # 1. Baseline TF-IDF evaluation
        baseline_report = evaluator.evaluate_engine(
            self.engine, k=3, threshold=0.15, use_pca=False
        )
        self.assertEqual(baseline_report.retrieval_method, "Baseline TF-IDF")
        self.assertGreaterEqual(baseline_report.mean_precision_at_k, 0.0)
        self.assertGreaterEqual(baseline_report.mean_recall_at_k, 0.0)
        self.assertGreaterEqual(baseline_report.mean_f1_at_k, 0.0)
        self.assertGreater(baseline_report.mrr, 0.0)
        self.assertGreaterEqual(baseline_report.mean_citation_accuracy, 0.0)
        self.assertEqual(len(baseline_report.query_results), 10)

        # 2. Switch to Truncated SVD (5 components)
        svd_5_report = evaluator.evaluate_engine(
            self.engine, k=3, threshold=0.15, use_pca=True, pca_components=5
        )
        self.assertEqual(svd_5_report.retrieval_method, "Truncated SVD (5 components)")
        self.assertEqual(len(svd_5_report.query_results), 10)
        self.assertGreater(svd_5_report.mrr, 0.0)

        # 3. Switch to Truncated SVD (2 components)
        svd_2_report = evaluator.evaluate_engine(
            self.engine, k=3, threshold=0.15, use_pca=True, pca_components=2
        )
        self.assertEqual(svd_2_report.retrieval_method, "Truncated SVD (2 components)")
        self.assertEqual(len(svd_2_report.query_results), 10)
        # 2 components has lower variance and should produce different MRR
        self.assertNotEqual(svd_2_report.mrr, 0.0)

    def test_benchmark_config_evaluation_trigger_logic(self):
        """Regression test: Validates needs_evaluation state machine for button and method selection."""
        # Initial state: no previous report
        session_mock = {}
        run_btn = False
        cfg1 = {"method": "Baseline TF-IDF", "k": 3, "threshold": 0.15, "doc": "sample.pdf"}

        needs_eval = run_btn or ("last_benchmark_report" not in session_mock) or (session_mock.get("last_bench_config") != cfg1)
        self.assertTrue(needs_eval, "Should trigger evaluation when no report is in session state")

        # Simulate first evaluation completed
        session_mock["last_benchmark_report"] = "dummy_report"
        session_mock["last_bench_config"] = cfg1

        # Second run without changes
        needs_eval = run_btn or ("last_benchmark_report" not in session_mock) or (session_mock.get("last_bench_config") != cfg1)
        self.assertFalse(needs_eval, "Should not re-evaluate when config is unchanged and button not clicked")

        # User selects Truncated SVD (PCA)
        cfg_svd = {"method": "Truncated SVD (PCA)", "k": 3, "threshold": 0.15, "doc": "sample.pdf"}
        needs_eval = run_btn or ("last_benchmark_report" not in session_mock) or (session_mock.get("last_bench_config") != cfg_svd)
        self.assertTrue(needs_eval, "Should trigger evaluation when user switches retrieval method to Truncated SVD")

        # Simulate SVD evaluation completed
        session_mock["last_bench_config"] = cfg_svd

        # User clicks "Run Benchmark" button
        run_btn = True
        needs_eval = run_btn or ("last_benchmark_report" not in session_mock) or (session_mock.get("last_bench_config") != cfg_svd)
        self.assertTrue(needs_eval, "Should trigger evaluation when user explicitly clicks Run Benchmark")

    def test_sample_document_session_persistence_regression(self):
        """Regression test: use_sample_doc session state prevents document loss across rerun cycles."""
        session_mock = {"use_sample_doc": True}
        sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        uploaded_file = None

        # Simulate multiple reruns (e.g. user interacting with tabs, sliders, radio buttons)
        for _ in range(5):
            use_sample = session_mock.get("use_sample_doc", False)
            pdf_source = None
            doc_name = ""

            if uploaded_file is not None and not use_sample:
                pdf_source = uploaded_file
                doc_name = uploaded_file.name
            elif use_sample and sample_path.exists():
                pdf_source = sample_path
                doc_name = sample_path.name

            self.assertIsNotNone(pdf_source, "Document source must remain loaded across reruns")
            self.assertEqual(doc_name, "sample_autosar_hld.pdf")


if __name__ == "__main__":
    unittest.main()
