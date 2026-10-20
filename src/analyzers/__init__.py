"""AUTOSAR rule analyzers, consistency checker, retrieval engines, and evaluation benchmark package."""

from .consistency_checker import AUTOSARConsistencyChecker
from .evaluator import (
    AUTOSAR_SAMPLE_BENCHMARK,
    AUTOSARBenchmarkEvaluator,
    BenchmarkEvaluationReport,
    BenchmarkQuery,
    IRMetricsCalculator,
    QueryEvaluationResult,
)
from .models import ConsistencyReport, RuleFinding, Severity
from .retrieval_engine import (
    RetrievalComparisonResult,
    ScoredChunk,
    TFIDFRetrievalEngine,
    TFIDFVectorizer,
    TruncatedSVDExperiment,
)

__all__ = [
    "TFIDFVectorizer",
    "TruncatedSVDExperiment",
    "TFIDFRetrievalEngine",
    "ScoredChunk",
    "RetrievalComparisonResult",
    "AUTOSAR_SAMPLE_BENCHMARK",
    "BenchmarkQuery",
    "QueryEvaluationResult",
    "BenchmarkEvaluationReport",
    "IRMetricsCalculator",
    "AUTOSARBenchmarkEvaluator",
    "Severity",
    "RuleFinding",
    "ConsistencyReport",
    "AUTOSARConsistencyChecker",
]
