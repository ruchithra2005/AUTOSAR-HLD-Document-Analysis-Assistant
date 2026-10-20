"""Manually Labelled AUTOSAR Benchmark and Information Retrieval Metrics Module.

Contains verified ground-truth queries, expected target pages, expected architectural
entities, and factual reference answers grounded in data/input_docs/sample_autosar_hld.pdf.
Computes Precision@K, Recall@K, F1-Score, Mean Reciprocal Rank (MRR), and Citation Accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.analyzers.retrieval_engine import ScoredChunk, TFIDFRetrievalEngine


@dataclass
class BenchmarkQuery:
    """Represents a manually verified ground-truth evaluation item."""
    query_id: str
    query_text: str
    expected_target_pages: List[int]
    expected_entities: List[str]
    ground_truth_answer: str
    category: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryEvaluationResult:
    """Detailed evaluation metrics for a single benchmark query."""
    query_id: str
    query_text: str
    k: int
    retrieved_pages: List[int]
    expected_pages: List[int]
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    reciprocal_rank: float
    first_relevant_rank: Optional[int]
    citation_accuracy: float
    accepted_chunks_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkEvaluationReport:
    """Summary of IR performance across the complete benchmark dataset."""
    retrieval_method: str
    k: int
    threshold: float
    total_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_f1_at_k: float
    mrr: float
    mean_citation_accuracy: float
    query_results: List[QueryEvaluationResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mean_precision_at_k"] = round(self.mean_precision_at_k, 4)
        data["mean_recall_at_k"] = round(self.mean_recall_at_k, 4)
        data["mean_f1_at_k"] = round(self.mean_f1_at_k, 4)
        data["mrr"] = round(self.mrr, 4)
        data["mean_citation_accuracy"] = round(self.mean_citation_accuracy, 4)
        return data


# =============================================================================
# Manually Labelled AUTOSAR Benchmark Dataset (Ground Truth from sample_autosar_hld.pdf)
# =============================================================================

AUTOSAR_SAMPLE_BENCHMARK: List[BenchmarkQuery] = [
    BenchmarkQuery(
        query_id="Q01_PRIMARY_SWC",
        query_text="What is the primary Software Component defined in this HLD and what is its type?",
        expected_target_pages=[1],
        expected_entities=["VehicleSpeedManagement_SWC", "Application SWC"],
        ground_truth_answer="VehicleSpeedManagement_SWC is defined on Page 1 as an Application SWC.",
        category="SWC Architecture",
    ),
    BenchmarkQuery(
        query_id="Q02_CONNECTED_SWCS",
        query_text="Which other SWCs are connected to VehicleSpeedManagement_SWC?",
        expected_target_pages=[1],
        expected_entities=["WheelSpeedSensor_SWC", "InstrumentCluster_SWC", "EngineControl_SWC"],
        ground_truth_answer="WheelSpeedSensor_SWC, InstrumentCluster_SWC, and EngineControl_SWC are connected SWCs on Page 1.",
        category="SWC Architecture",
    ),
    BenchmarkQuery(
        query_id="Q03_BSW_DEPENDENCIES",
        query_text="What BSW modules and communication stack layers are specified as dependencies?",
        expected_target_pages=[1],
        expected_entities=["ComStack", "CanIf", "CanTp", "Com", "Dem", "NvM"],
        ground_truth_answer="Page 1 specifies ComStack (CanIf, CanTp, Com), Dem, and NvM.",
        category="BSW Dependencies",
    ),
    BenchmarkQuery(
        query_id="Q04_PPORT_VEHICLE_SPEED",
        query_text="What is the P-Port ppVehicleSpeed, its interface type, and data element?",
        expected_target_pages=[2],
        expected_entities=["ppVehicleSpeed", "IF_VehicleSpeed", "VehicleSpeed_kmh"],
        ground_truth_answer="Page 2 defines P-Port ppVehicleSpeed with Sender-Receiver interface IF_VehicleSpeed and data element VehicleSpeed_kmh.",
        category="Ports & Interfaces",
    ),
    BenchmarkQuery(
        query_id="Q05_RPORTS_SPECIFICATION",
        query_text="What R-Ports are defined in the document and what interfaces do they require?",
        expected_target_pages=[2],
        expected_entities=["rpWheelPulseRaw", "IF_WheelPulseData", "rpDiagStatus", "IF_DiagStatusService"],
        ground_truth_answer="Page 2 defines R-Port rpWheelPulseRaw (Sender-Receiver IF_WheelPulseData) and R-Port rpDiagStatus (Client-Server IF_DiagStatusService).",
        category="Ports & Interfaces",
    ),
    BenchmarkQuery(
        query_id="Q06_RUNNABLE_CALCULATE_SPEED",
        query_text="How is Runnable RE_CalculateSpeed triggered and which OS Task is it mapped to?",
        expected_target_pages=[2],
        expected_entities=["RE_CalculateSpeed", "Task_Periodic_10ms", "Timing Event", "10ms"],
        ground_truth_answer="Page 2 defines RE_CalculateSpeed triggered by RTE Timing Event (Period: 10ms) mapped to Task_Periodic_10ms.",
        category="Internal Behavior",
    ),
    BenchmarkQuery(
        query_id="Q07_RUNNABLE_EVENTS",
        query_text="What triggers are specified for RE_VSM_Init and RE_DiagEventUpdate?",
        expected_target_pages=[2],
        expected_entities=["RE_VSM_Init", "swc_init", "RE_DiagEventUpdate", "Data Received Event"],
        ground_truth_answer="Page 2 specifies RTE Init Event for RE_VSM_Init and Data Received Event on rpWheelPulseRaw for RE_DiagEventUpdate.",
        category="Internal Behavior",
    ),
    BenchmarkQuery(
        query_id="Q08_SAFETY_ASIL_E2E",
        query_text="What ISO 26262 ASIL rating and E2E protection profile are required?",
        expected_target_pages=[1, 3],
        expected_entities=["ASIL-B", "ISO 26262", "Profile 2", "IF_VehicleSpeed"],
        ground_truth_answer="ASIL-B rating is specified on Page 1 and Page 3. E2E Profile 2 is specified for IF_VehicleSpeed on Page 3.",
        category="Functional Safety",
    ),
    BenchmarkQuery(
        query_id="Q09_DIAGNOSTIC_MONITORING",
        query_text="Under what condition is Dem_SetEventStatus called?",
        expected_target_pages=[3],
        expected_entities=["Dem_SetEventStatus", "Dem", "25 km/h"],
        ground_truth_answer="Page 3 states Dem_SetEventStatus is called if wheel sensor differential > 25 km/h.",
        category="BSW Diagnostics",
    ),
    BenchmarkQuery(
        query_id="Q10_VERIFICATION_CHECKLIST_GAP",
        query_text="Which checklist item is marked as unverified or under review?",
        expected_target_pages=[3],
        expected_entities=["rpDiagStatus", "Server response timeout", "Under Review"],
        ground_truth_answer="Page 3 checklist has an unchecked item: Server response timeout for rpDiagStatus Client-Server invocation (Under Review).",
        category="Architectural Verification",
    ),
]


# =============================================================================
# Information Retrieval Evaluation Metrics Engine
# =============================================================================

class IRMetricsCalculator:
    """Calculates information retrieval metrics for evaluated queries."""

    @staticmethod
    def calculate_precision_at_k(retrieved_pages: List[int], expected_pages: Set[int], k: int) -> float:
        """Precision@K = (Number of relevant retrieved chunks in top K) / K."""
        if k <= 0 or not retrieved_pages:
            return 0.0
        top_k_pages = retrieved_pages[:k]
        relevant_count = sum(1 for p in top_k_pages if p in expected_pages)
        return float(relevant_count / len(top_k_pages))

    @staticmethod
    def calculate_recall_at_k(retrieved_pages: List[int], expected_pages: Set[int], k: int) -> float:
        """Recall@K = (Number of unique expected pages retrieved in top K) / (Total expected pages)."""
        if not expected_pages:
            return 1.0  # Vacuously true if no pages were needed
        if k <= 0 or not retrieved_pages:
            return 0.0
        top_k_pages = set(retrieved_pages[:k])
        recovered_pages = top_k_pages.intersection(expected_pages)
        return float(len(recovered_pages) / len(expected_pages))

    @staticmethod
    def calculate_f1_at_k(precision: float, recall: float) -> float:
        """F1-Score = 2 * (P * R) / (P + R)."""
        if precision + recall == 0.0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    @staticmethod
    def calculate_reciprocal_rank(retrieved_pages: List[int], expected_pages: Set[int]) -> Tuple[float, Optional[int]]:
        """Reciprocal Rank = 1 / rank of first relevant result."""
        for idx, page in enumerate(retrieved_pages):
            if page in expected_pages:
                rank = idx + 1
                return float(1.0 / rank), rank
        return 0.0, None

    @staticmethod
    def calculate_citation_accuracy(retrieved_pages: List[int], expected_pages: Set[int], k: int) -> float:
        """Citation Accuracy: Fraction of top-K citations that point to verified target pages."""
        if k <= 0 or not retrieved_pages:
            return 0.0
        top_k_pages = retrieved_pages[:k]
        valid_citations = sum(1 for p in top_k_pages if p in expected_pages)
        return float(valid_citations / len(top_k_pages))


class AUTOSARBenchmarkEvaluator:
    """Evaluates retrieval engines against the verified AUTOSAR benchmark dataset."""

    def __init__(self, benchmark: Optional[List[BenchmarkQuery]] = None) -> None:
        self.benchmark = benchmark or AUTOSAR_SAMPLE_BENCHMARK
        self.calculator = IRMetricsCalculator()

    def evaluate_engine(
        self,
        engine: TFIDFRetrievalEngine,
        k: int = 3,
        threshold: Optional[float] = None,
        use_pca: bool = False,
        pca_components: int = 5,
    ) -> BenchmarkEvaluationReport:
        """Evaluates engine retrieval quality across all benchmark queries."""
        tau = threshold if threshold is not None else engine.default_threshold
        query_results: List[QueryEvaluationResult] = []

        for bq in self.benchmark:
            expected_pages_set = set(bq.expected_target_pages)

            # Retrieve chunks using selected method
            if use_pca:
                comp_result = engine.compare_with_pca(
                    query=bq.query_text,
                    n_components=pca_components,
                    top_k=k,
                    threshold=tau,
                )
                scored_chunks = comp_result.reduced_results or []
            else:
                scored_chunks = engine.retrieve(
                    query=bq.query_text,
                    top_k=k,
                    threshold=tau,
                )

            retrieved_pages = [sc.chunk.page_number for sc in scored_chunks]
            accepted_count = sum(1 for sc in scored_chunks if sc.is_accepted)

            # Calculate individual metrics
            prec = self.calculator.calculate_precision_at_k(retrieved_pages, expected_pages_set, k=k)
            rec = self.calculator.calculate_recall_at_k(retrieved_pages, expected_pages_set, k=k)
            f1 = self.calculator.calculate_f1_at_k(prec, rec)
            rr, first_rank = self.calculator.calculate_reciprocal_rank(retrieved_pages, expected_pages_set)
            cite_acc = self.calculator.calculate_citation_accuracy(retrieved_pages, expected_pages_set, k=k)

            query_results.append(
                QueryEvaluationResult(
                    query_id=bq.query_id,
                    query_text=bq.query_text,
                    k=k,
                    retrieved_pages=retrieved_pages,
                    expected_pages=bq.expected_target_pages,
                    precision_at_k=round(prec, 4),
                    recall_at_k=round(rec, 4),
                    f1_at_k=round(f1, 4),
                    reciprocal_rank=round(rr, 4),
                    first_relevant_rank=first_rank,
                    citation_accuracy=round(cite_acc, 4),
                    accepted_chunks_count=accepted_count,
                )
            )

        total_q = len(query_results)
        if total_q == 0:
            return BenchmarkEvaluationReport(
                retrieval_method="Truncated SVD" if use_pca else "Baseline TF-IDF",
                k=k,
                threshold=tau,
                total_queries=0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_f1_at_k=0.0,
                mrr=0.0,
                mean_citation_accuracy=0.0,
                query_results=[],
                notes="No benchmark queries were evaluated.",
            )

        mean_prec = sum(r.precision_at_k for r in query_results) / total_q
        mean_rec = sum(r.recall_at_k for r in query_results) / total_q
        mean_f1 = sum(r.f1_at_k for r in query_results) / total_q
        mrr = sum(r.reciprocal_rank for r in query_results) / total_q
        mean_cite = sum(r.citation_accuracy for r in query_results) / total_q

        method_name = f"Truncated SVD ({pca_components} components)" if use_pca else "Baseline TF-IDF"
        notes = (
            f"Evaluated on {total_q} verified AUTOSAR HLD queries at K={k} with threshold={tau}. "
            "Relevance is grounded in source page targets verified against data/input_docs/sample_autosar_hld.pdf."
        )

        return BenchmarkEvaluationReport(
            retrieval_method=method_name,
            k=k,
            threshold=tau,
            total_queries=total_q,
            mean_precision_at_k=round(mean_prec, 4),
            mean_recall_at_k=round(mean_rec, 4),
            mean_f1_at_k=round(mean_f1, 4),
            mrr=round(mrr, 4),
            mean_citation_accuracy=round(mean_cite, 4),
            query_results=query_results,
            notes=notes,
        )
