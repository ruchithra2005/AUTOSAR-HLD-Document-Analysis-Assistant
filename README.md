# 🚗 AUTOSAR HLD Document Analysis Assistant
*(Tata Technologies TechPulse Internship Project)*

An AI-assisted engineering tool designed to review, verify, and analyze **AUTOSAR High-Level Design (HLD)** specifications for automotive software systems.

---

## 📌 Phase 1: Ingestion, Extraction & Document Inspection
- **PyMuPDF Engine:** High-performance, reliable PDF text and metadata extraction.
- **Document Details:** Page counts, word counts, file size, author/creation metadata, and encryption status.
- **AUTOSAR Domain Keyword Scanner:** Automatically scans for automotive software architectural terms.
- **Interactive Streamlit Dashboard:** File uploader, one-click sample loader, and page-by-page reader.

---

## 🧩 Phase 2: Structured Architecture Extraction & Evidence Assurance
- **Document Section & Heading Detection (`section_detector.py`):** Automatically detects numbered chapters (`1. ARCHITECTURAL OVERVIEW`) and technical headings with depth levels and page locations.
- **Candidate Entity Extraction (`autosar_extractor.py`):**
  - **Software Components (SWCs):** Target Application SWCs, Sensor/Actuator SWCs, and Connected SWCs.
  - **Ports:** P-Ports, R-Ports, PR-Ports, directions, and interface mappings.
  - **Port Interfaces:** Sender-Receiver and Client-Server interfaces with operations/data elements.
  - **Internal Behavior:** Runnables, trigger RTE Events, periodicity, and OS task mappings.
  - **BSW Modules:** Catalog classification for ComStack, Dem, NvM, OS, MCAL drivers, etc.
  - **Safety & Dependencies:** ISO 26262 ASIL ratings, E2E protection profiles, and checklist status.
- **Evidence & Provenance Tracking (`models.py`):**
  - Every extracted element links to its `source_file`, `page_number`, and `supporting_text` (verbatim excerpt).
  - Clear evidence status categorization (`✅ DETECTED`, `💡 INFERRED`, `⚠️ UNVERIFIED`).
- **Interactive Searchable Tables:** Filter by Entity Type, Evidence Status, or search by keyword; export complete structured model to JSON.
- **Zero Fabricated Rules:** Grounded strictly in textual evidence; no assumptions of compliance.

---

## 💬 Phase 3: Grounded Gemini LLM Chat & Architectural Review
- **Google Gemini API Integration (`src/llm/client.py`):**
  - Secure API key management via environment variable (`GEMINI_API_KEY`) or in-memory session override in the Streamlit sidebar (never logged or committed).
  - Uses Google's official `google-genai` SDK with support for `gemini-2.5-flash`, `gemini-2.5-pro`, and `gemini-1.5-flash`.
- **Domain Prompt Grounding (`src/llm/prompts.py`):**
  - Synthesizes document metadata, section structure, structured entities, and verbatim page text into grounded prompts.
  - **Strict Citation Mandate:** Requires answers to cite `[Source: <filename>, Page: <page_number>]` and quotes.
  - **Hallucination Prevention:** Instructs the model to explicitly state when evidence is insufficient instead of guessing.
  - **Separation of Facts vs Observations:** Distinguishes explicit document facts from architectural deductions.
  - **No False Compliance:** Forbids claiming standards compliance without authorized reference rules.
- **Interactive Streamlit Chat:**
  - Conversational Q&A tab with pre-built architectural question chips.
  - Error handling for missing keys, quota limits (429), network timeouts, and empty responses.
  - **Graceful Degradation:** All document parsing, section detection, structured extraction, search, and JSON export function completely without an API key.
- **Offline Mock Unit Testing (`tests/test_llm_client.py`):**
  - Full test suite with mocked Gemini responses; runs offline without requiring real API keys or incurring charges.

---

## 🛡️ Phase 4: Architectural Consistency & Rule Checking
- **Deterministic AUTOSAR Rules (`consistency_checker.py`):**
  - **AR-PORT-01:** Port-to-interface matching and unresolved interface detection.
  - **AR-RUN-01:** Runnable trigger and periodic activation timing verification.
  - **AR-RUN-02:** Runnable-to-OS-Task mapping checks.
  - **AR-IF-01:** Client-Server operation and invocation timeout auditing.
  - **AR-SAFE-01:** ISO 26262 ASIL safety rating and End-to-End (E2E) protection profile verification.
- **Evidence Provenance & Severity Categorization:**
  - Distinguishes `ERROR`, `WARNING`, `PASS`, and `UNVERIFIED` without fabricating compliance.
  - Flags findings for ISO 26262 functional safety review.
- **Standards Scope Disclaimer:** The checker executes defined architectural rules; it does not prove complete AUTOSAR standards compliance or replace formal schema/toolchain verification.

---

## 🔬 Phase 5: Streamlit UI Integration, AI/ML Transparency & Benchmark Studio
- **Architectural Consistency Checks UI:**
  - Real-time summary scorecard (PASS, WARNING, ERROR, UNVERIFIED).
  - Severity and Rule ID filters with search and ISO 26262 safety review toggles.
  - Detailed finding inspector with engineering recommendations, verbatim evidence excerpts, and JSON export.
- **AI/ML Transparency & Vector Retrieval Studio:**
  - **Preprocessing & Chunking Inspector:** Configurable sliding window (`chunk_size`, `chunk_overlap`), repeated header deduplication, word counts, and chunk browser.
  - **Transparent TF-IDF Cosine Retrieval:** Pure NumPy vectorization, sublinear TF scaling, smoothed IDF, and threshold accept/reject auditability with vocabulary weight inspect.
  - **SVD / PCA Dimensionality Reduction Experiment:** Empirical comparison between exact lexical TF-IDF and dense low-rank Truncated SVD representation with latency, dimension reduction, and explained variance metrics.
- **AUTOSAR Benchmark Evaluation Studio:**
  - 10 verified ground-truth queries grounded in `sample_autosar_hld.pdf` across 7 architectural categories.
  - Interactive evaluation controls with configurable Top-K and similarity cutoffs.
  - Summary Scorecard: Precision@K, Recall@K, F1-Score, Mean Reciprocal Rank (MRR), and Citation Accuracy.
  - Query-by-query breakdown table and downloadable JSON report.

---

## 📊 Evaluation Methodology & Empirical Benchmark Results

The system implements a transparent, reproducible Information Retrieval (IR) evaluation benchmark grounded in `data/input_docs/sample_autosar_hld.pdf`:

### 1. Benchmark Ground-Truth Dataset (`src/analyzers/evaluator.py`)
Comprises **10 manually verified domain queries** across 7 architectural categories:
- **SWC Architecture:** Primary component role, component classification, connected SWCs (`Q01`, `Q02`).
- **BSW Dependencies:** Communication stack (ComStack, CanIf, CanTp, Com), Dem, NvM modules (`Q03`).
- **Ports & Interfaces:** Sender-Receiver and Client-Server bindings, data elements (`ppVehicleSpeed`, `rpWheelPulseRaw`, `rpDiagStatus`) (`Q04`, `Q05`).
- **Internal Behavior:** Runnables (`RE_CalculateSpeed`, `RE_VSM_Init`), RTE event triggers (Timing, Init, Data Received), OS task mapping (`Task_Periodic_10ms`) (`Q06`, `Q07`).
- **Functional Safety:** ISO 26262 ASIL-B classification and End-to-End (E2E) Profile 2 data protection (`Q08`).
- **Diagnostics:** Diagnostic Event Manager (`Dem_SetEventStatus`) differential speed monitoring thresholds (`Q09`).
- **Architectural Verification:** Pending review checklist items and client-server response timeouts (`Q10`).

### 2. IR Mathematical Metrics & Evaluation Scope
* **Precision@K:** $\frac{|\text{Retrieved Top-}K \cap \text{Expected Pages}|}{K}$ — Fraction of retrieved chunks that are relevant.
* **Recall@K:** $\frac{|\text{Retrieved Top-}K \cap \text{Expected Pages}|}{|\text{Expected Target Pages}|}$ — Fraction of ground-truth target pages successfully retrieved.
* **F1-Score@K:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean of precision and recall.
* **Mean Reciprocal Rank (MRR):** $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$ — Average reciprocal rank of the first relevant document chunk.
* **Citation Accuracy:** Fraction of top-$K$ retrieved chunks that correctly cite ground-truth verified target pages.

> [!NOTE]
> **Retrieval Evaluation vs. Generative Answer Evaluation:**
> The benchmark strictly evaluates **Information Retrieval (IR) performance**—specifically, how accurately the TF-IDF and SVD vectorization engines retrieve and rank document chunks relative to expected target pages. It does **not** evaluate LLM-generated natural language answers, text readability, or generative citations (which would require generative metrics like ROUGE, BLEU, or an LLM-as-a-judge setup). "Citation Accuracy" refers to chunk source provenance mapping to expected pages, not generated LLM citations.

### 3. Empirical Benchmark Results (Sample AUTOSAR HLD, K=3, $\tau=0.15$)

| Evaluation Metric | Baseline TF-IDF | Truncated SVD (4 Components) | Truncated SVD (2 Components) |
|---|---|---|---|
| **Mean Precision@3** | 46.67% | 46.67% | 43.33% |
| **Mean Recall@3** | 100.00% | 100.00% | 100.00% |
| **Mean F1-Score@3** | 62.00% | 62.00% | 58.67% |
| **Mean Reciprocal Rank (MRR)** | **0.9500** | **0.9500** | 0.8333 |
| **Mean Citation Accuracy** | 46.67% | 46.67% | 43.33% |
| **Total Benchmark Queries** | 10 | 10 | 10 |

*Empirical Finding:* On compact technical HLD documents, baseline exact TF-IDF preserves specialized domain acronyms (`SWC`, `RTE`, `ASIL`, `Dem`) without topic blurring, achieving an MRR of 0.9500 (relevant page at rank #1 for 9 out of 10 queries). Low-rank SVD ($k=2$) captures broad latent semantic variance but exhibits slight acronym loss.

### 4. Benchmark Limitations & Dataset Scope
* **Curated Sample Size:** The benchmark currently evaluates 10 manually curated domain queries against the bundled 3-page sample AUTOSAR HLD (`sample_autosar_hld.pdf`).
* **Statistical Extensibility:** While these 10 queries thoroughly exercise all major sections of the sample HLD (SWCs, ports, runnables, BSW, safety), broader statistical benchmarking across large enterprise corpora (100+ pages, 100+ queries) is recommended for production scaling.
* **Scope Boundary:** The benchmark does not measure LLM response latency or generation tokens; those depend on external cloud API availability.

---

## ⚠️ Known Limitations & Academic Boundary

1. **Rule-Based Consistency Checking vs. Full Compliance:**
   * The architectural consistency checker operates strictly on a set of five defined heuristic rules (`AR-PORT-01`, `AR-RUN-01`, `AR-RUN-02`, `AR-IF-01`, `AR-SAFE-01`).
   * **Does not prove complete AUTOSAR standards compliance:** It evaluates high-level design consistency and flags gaps for engineering review. It does **not** prove, certify, or guarantee complete compliance with the full AUTOSAR Classic or Adaptive platform specifications, nor does it replace formal ARXML schema validation, RTE generation toolchains, or full ISO 26262 functional safety audits.
2. **Text-Based PDF Format:**
   * Text extraction uses PyMuPDF and requires digital-native, searchable PDF documents. Scanned image-only PDFs without an OCR text layer require an OCR preprocessing step before ingestion.
3. **Single-Document Architectural Scope:**
   * The current version reviews individual HLD specifications (`sample_autosar_hld.pdf` or uploaded PDF). Cross-document linkage across multi-file architectural repositories or external ARXML XML schemas is out of current single-document scope.
4. **Graceful Degradation vs. Cloud API Dependency:**
   * All core engineering features—PDF text parsing, section detection, structured entity extraction, deterministic consistency checks, TF-IDF retrieval, SVD experiments, and IR benchmark evaluation—operate **100% offline without network connectivity or API keys**.
   * Conversational review chat requires an active Google Gemini API key (`GEMINI_API_KEY`) and outbound internet access.
5. **Conservative Architectural Bounds (Zero Fabrication):**
   * The consistency checker and LLM prompt templates strictly enforce evidence boundaries. Incomplete or ambiguous specifications (e.g. unmapped aperiodic runnables or unverified timeouts) are intentionally classified as `UNVERIFIED` rather than assumed to be compliant.
6. **Benchmark Evaluation Scope:**
   * Benchmark metrics measure Information Retrieval (IR) performance on chunk and page-level retrieval; they do not benchmark natural language generation quality or generative LLM citations.

---

## 🗂️ Project Directory Structure

```text
AUTOSAR_HLD_Assistant/
├── app.py                      # Streamlit dashboard with full Stage 5 tabs
├── config.yaml                 # System configurations, model defaults & check categories
├── requirements.txt            # Python dependencies (Streamlit, PyMuPDF, google-genai, etc.)
├── .env.example                # Template for GEMINI_API_KEY
├── .gitignore                  # Git and submission exclusion rules
├── Evaluation_Results.json     # Reference export of benchmark evaluation results
├── README.md                   # Project documentation and quick-start guide
│
├── data/
│   ├── input_docs/             # Storage for HLD documents (includes sample_autosar_hld.pdf)
│   └── rules_guidelines/       # AUTOSAR specifications & checklists
│
├── src/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── models.py           # Dataclass models: EntityType, EvidenceStatus, ExtractionResult
│   │   ├── pdf_parser.py       # PyMuPDF-based text and metadata extractor
│   │   ├── section_detector.py # Document chapter & section heading detector
│   │   ├── autosar_extractor.py# Structured SWC, Port, Runnable & BSW extractor
│   │   └── chunker.py          # Sliding-window chunker with repeated line deduplication
│   ├── analyzers/              # Architecture, retrieval & benchmark evaluation modules
│   │   ├── __init__.py
│   │   ├── models.py           # RuleFinding, Severity, ConsistencyReport
│   │   ├── consistency_checker.py # Deterministic AUTOSAR architectural consistency checks
│   │   ├── retrieval_engine.py # TFIDFVectorizer, Cosine Retrieval & Truncated SVD
│   │   └── evaluator.py        # Benchmark queries, IRMetricsCalculator & BenchmarkEvaluator
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # GeminiClient with safe error handling & mockability
│   │   └── prompts.py          # AUTOSARPromptBuilder with grounding & citation rules
│   └── utils/                  # Utility helpers and report generators
│
├── reports/                    # Directory for generated analysis reports
└── tests/
    ├── __init__.py
    ├── test_pdf_parser.py      # Unit tests for PyMuPDF text parser
    ├── test_structured_extractor.py # Unit tests for Phase 2 structured extraction
    ├── test_llm_client.py      # Unit tests for Phase 3 LLM client & prompt grounding (mocked)
    ├── test_chunker.py         # Unit tests for document preprocessing & chunking
    ├── test_consistency_checker.py # Unit tests for architectural consistency rules
    ├── test_retrieval_engine.py# Unit tests for TF-IDF vectorization & SVD
    ├── test_evaluator.py       # Unit tests for IR metrics & benchmark evaluation
    └── test_stage5_ui_integration.py # Integration & regression tests for Stage 5 UI components
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- `pip` package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
To enable the AI chat interface:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
*(Alternatively, you can enter your API key directly in the Streamlit sidebar during your session).*

### 4. Run the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
*(Or directly `streamlit run app.py` if your Python Scripts directory is on your system PATH).*

Once launched, your browser will open to `http://localhost:8501`.

### 5. Running Automated Unit Tests
To run the complete automated test suite (71 tests across parsing, chunking, extraction, LLM grounding, consistency checking, retrieval, and benchmark evaluation):
```bash
python -m unittest discover -s tests -v
```

---

## 🏎️ AUTOSAR Concepts Refresher (For Beginners)

| Concept | Description |
|---|---|
| **SWC (Software Component)** | Atomic unit of application software (e.g., Engine Speed Controller, Sensor Abstraction). |
| **Ports & Interfaces** | Interaction points between SWCs: **Sender-Receiver** (data exchange) or **Client-Server** (function calls). |
| **Runnables** | Smallest executable code fragments within an SWC, triggered by RTE Events (e.g., 10ms timer, data arrival). |
| **RTE (Run-Time Environment)** | Middleware layer enabling SWC-to-SWC and SWC-to-BSW communication without hardware dependency. |
| **BSW (Basic Software)** | Standardized services (Communication stack, Memory, Diagnostic Event Manager, OS, Drivers). |
| **ASIL (Automotive Safety Integrity Level)** | Risk classification scheme defined by ISO 26262 (ASIL A to ASIL D). |

---

## 🗺️ Roadmap
- [x] **Phase 1:** Document Ingestion, PyMuPDF text extraction, metadata viewer & Streamlit UI.
- [x] **Phase 2:** Structured section detection, candidate SWC/Port/Runnable/BSW extraction, evidence provenance, and searchable tables.
- [x] **Phase 3:** Grounded Gemini LLM chat interface, citation enforcement, fact vs observation separation, and offline mock tests.
- [x] **Phase 4:** Architectural consistency & rule checking (Port mismatches, timing event conflicts, unmapped runnables).
- [x] **Phase 5:** Consistency checks, AI/ML transparency, and benchmark evaluation studio in Streamlit UI.
- [ ] **Phase 6:** Full interactive review agent with chat-based Q&A over the HLD.
