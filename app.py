"""AUTOSAR HLD Document Analysis Assistant - Streamlit Dashboard
Phase 4 & 5: Consistency Checks, AI/ML Transparency Studio & Benchmark Evaluation
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.analyzers import (
    AUTOSAR_SAMPLE_BENCHMARK,
    AUTOSARBenchmarkEvaluator,
    AUTOSARConsistencyChecker,
    BenchmarkEvaluationReport,
    BenchmarkQuery,
    ConsistencyReport,
    IRMetricsCalculator,
    QueryEvaluationResult,
    RetrievalComparisonResult,
    RuleFinding,
    ScoredChunk,
    Severity,
    TFIDFRetrievalEngine,
    TruncatedSVDExperiment,
)
from src.llm import (
    GeminiClient,
    GroundedResponse,
    LLMConfigurationError,
    LLMNetworkError,
    LLMQuotaExceededError,
    LLMResponseError,
)
from src.parsers import (
    DocumentChunker,
    EntityType,
    EvidenceStatus,
    ExtractionResult,
    PDFExtractionError,
    PDFParser,
    PreprocessingStats,
    SectionHeading,
    StructuredAUTOSARExtractor,
    TextChunk,
)

# Load environment variables
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AUTOSAR HLD Analysis Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_config() -> dict:
    """Loads application configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_sample_document_path() -> Optional[Path]:
    """Returns path to the pre-bundled sample AUTOSAR HLD document if present."""
    sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
    return sample_path if sample_path.exists() else None


def render_sidebar(config: dict, sample_pdf_path: Optional[Path]) -> tuple[Optional[object], bool, str, str]:
    """Renders sidebar controls, file uploaders, AI settings, and configuration stats."""
    with st.sidebar:
        st.header("📂 Document Input")
        st.caption("Upload an AUTOSAR High-Level Design (HLD) document to inspect its structure.")

        uploaded_file = st.file_uploader(
            label="Upload HLD PDF",
            type=["pdf"],
            help="Upload an AUTOSAR design specification in PDF format (up to 25MB).",
        )

        # Deactivate sample mode if user uploads a new file
        if uploaded_file is not None and st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            st.session_state["use_sample_doc"] = False
            st.session_state["last_uploaded_filename"] = uploaded_file.name

        if sample_pdf_path:
            st.divider()
            st.markdown("**Quick Start:** Don't have a PDF?")
            sample_active = st.session_state.get("use_sample_doc", False)
            if not sample_active:
                if st.button("📄 Load Sample AUTOSAR HLD", use_container_width=True, key="btn_load_sample"):
                    st.session_state["use_sample_doc"] = True
                    st.rerun()
            else:
                st.success(f"✅ Active: `{sample_pdf_path.name}`")
                if st.button("🗑️ Unload Sample Document", use_container_width=True, key="btn_unload_sample"):
                    st.session_state["use_sample_doc"] = False
                    st.session_state.pop("last_benchmark_report", None)
                    st.session_state.pop("last_bench_config", None)
                    st.rerun()

        use_sample = st.session_state.get("use_sample_doc", False)

        st.divider()
        st.subheader("🤖 Gemini AI Settings")
        
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key:
            st.success("✅ GEMINI_API_KEY detected in environment")
        else:
            st.info("ℹ️ No GEMINI_API_KEY found in .env")

        session_key = st.text_input(
            "Gemini API Key (optional override):",
            type="password",
            value=st.session_state.get("gemini_api_key_override", ""),
            help="Your API key is kept only in memory during this session and is never logged or saved to disk.",
        )
        if session_key:
            st.session_state["gemini_api_key_override"] = session_key.strip()

        model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
        default_model = config.get("llm", {}).get("model", "gemini-2.5-flash")
        selected_model = st.selectbox(
            "Model Selection:",
            model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            help="Recommended: gemini-2.5-flash for fast, grounded domain analysis.",
        )

        st.divider()
        st.subheader("ℹ️ Project Info")
        st.markdown(
            f"""
            - **Project:** TechPulse Internship
            - **App:** {config.get('app', {}).get('name', 'AUTOSAR HLD Assistant')}
            - **Version:** `{config.get('app', {}).get('version', '0.5.0')}`
            - **Phase:** `{config.get('app', {}).get('phase', 'Phase 4 & 5 - Rule Checks & AI Evaluation Studio')}`
            - **Engine:** `PyMuPDF` + `NumPy TF-IDF` + `Google Gemini`
            """
        )

        st.caption(
            "💡 **Graceful Degradation:** All document parsing, section detection, "
            "structured extraction, architectural consistency checks, TF-IDF retrieval, "
            "SVD experiments, benchmark evaluation, and JSON exports function completely offline without an API key."
        )

    return uploaded_file, use_sample, (session_key.strip() or env_key), selected_model


def format_attributes(attrs: Dict[str, Any]) -> str:
    """Formats attribute dictionaries into readable table strings."""
    if not attrs:
        return "-"
    parts = []
    for k, v in attrs.items():
        clean_key = k.replace("_", " ").title()
        parts.append(f"{clean_key}: {v}")
    return " | ".join(parts)


def format_status_badge(status: str) -> str:
    """Formats evidence status with recognizable icons."""
    if status == EvidenceStatus.DETECTED.value:
        return "✅ DETECTED"
    elif status == EvidenceStatus.INFERRED.value:
        return "💡 INFERRED"
    elif status == EvidenceStatus.UNVERIFIED.value:
        return "⚠️ UNVERIFIED"
    return status


def format_severity_badge(severity: str) -> str:
    """Formats rule finding severity with distinct color-coded icons."""
    if severity == Severity.ERROR.value:
        return "🔴 ERROR"
    elif severity == Severity.WARNING.value:
        return "🟡 WARNING"
    elif severity == Severity.PASS.value:
        return "🟢 PASS"
    elif severity == Severity.UNVERIFIED.value:
        return "⚪ UNVERIFIED"
    return severity


def main() -> None:
    config = load_config()
    sample_pdf_path = get_sample_document_path()

    uploaded_file, use_sample, active_api_key, selected_model = render_sidebar(config, sample_pdf_path)

    # Main dashboard header
    st.title("🚗 AUTOSAR HLD Document Analysis Assistant")
    st.markdown(
        "Automated Architectural Review & Grounded AI Assistant for **AUTOSAR Classic & Adaptive Platforms** "
        "*(Tata Technologies TechPulse Project)*"
    )
    st.markdown("---")

    # Determine document source: uploaded file or bundled sample
    pdf_source = None
    doc_display_name = ""

    if uploaded_file is not None and not use_sample:
        pdf_source = uploaded_file
        doc_display_name = uploaded_file.name
    elif use_sample and sample_pdf_path:
        pdf_source = sample_pdf_path
        doc_display_name = sample_pdf_path.name
        st.caption(f"📄 Active document: `{doc_display_name}` (Bundled Sample HLD)")

    if pdf_source is not None:
        if st.session_state.get("active_document_name") != doc_display_name:
            st.session_state["active_document_name"] = doc_display_name
            st.session_state.pop("last_benchmark_report", None)
            st.session_state.pop("last_bench_config", None)
    else:
        # Empty state welcome screen
        st.info("👈 Please upload an AUTOSAR HLD PDF from the sidebar, or click **'Load Sample AUTOSAR HLD'** to test!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Capabilities (Phase 1 to 5):")
            st.markdown(
                """
                - **1. Ingestion & Document Health:** PyMuPDF extraction, encryption & reading metrics.
                - **2. Section Detection:** Hierarchy-aware chapter and heading detector.
                - **3. Structured AUTOSAR Model:** SWCs, Ports, Interfaces, Runnables, RTE Events, BSW Modules.
                - **4. Evidence & Provenance Assurance:** Verified page citations and quotes for every entity.
                - **5. Grounded Gemini AI Chat:** Real-time engineering Q&A with strict grounding & citation enforcement.
                - **6. Consistency Checks:** Deterministic AUTOSAR architectural rules (AR-PORT-01, AR-RUN-01, AR-RUN-02, AR-IF-01, AR-SAFE-01).
                - **7. AI/ML Transparency Studio:** Sliding-window chunking, TF-IDF cosine similarity, threshold filtering & SVD/PCA.
                - **8. Benchmark Evaluation Studio:** Precision@K, Recall@K, F1, MRR & Citation Accuracy across 10 verified queries.
                """
            )
        with col2:
            st.markdown("### 🛡️ Grounding & Compliance Rules:")
            st.markdown(
                """
                - **Zero Fabrication:** The AI and analyzers only evaluate grounded HLD evidence.
                - **Page-Level Citations:** Every assertion requires exact document filename and page numbers.
                - **Explicit Severity Categorization:** PASS, WARNING, ERROR, and UNVERIFIED evidence states.
                - **Safe Offline Operation:** All parsing, chunking, consistency checking, vector retrieval, and benchmark evaluations function 100% offline without an API key.
                """
            )
        return

    # 1. Ingestion via PyMuPDF
    try:
        parser = PDFParser()
        with st.spinner("Extracting text and metadata with PyMuPDF..."):
            doc_overview = parser.get_document_overview(pdf_source, filename=doc_display_name)
    except PDFExtractionError as e:
        st.error(f"❌ PDF Extraction Error: {str(e)}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected Error while processing document: {str(e)}")
        return

    metadata = doc_overview["metadata"]
    pages = doc_overview["pages"]
    keywords_found = doc_overview["autosar_keywords_found"]

    # 2. Structured Extraction Pipeline (Phase 2)
    try:
        extractor = StructuredAUTOSARExtractor()
        with st.spinner("Extracting structured AUTOSAR architecture model & evidence..."):
            extraction_result: ExtractionResult = extractor.extract(pages, source_file=doc_display_name)
    except Exception as e:
        st.error(f"❌ Extraction Pipeline Error: {str(e)}")
        return

    summary_counts = extraction_result.get_summary_counts()
    total_entities = len(extraction_result.entities)

    # 2b. Consistency Checks Pipeline (Stage 5)
    try:
        consistency_checker = AUTOSARConsistencyChecker()
        consistency_report: ConsistencyReport = consistency_checker.check_consistency(extraction_result)
    except Exception as e:
        st.error(f"❌ Consistency Check Pipeline Error: {str(e)}")
        consistency_report = None

    # 2c. Document Chunking & Retrieval Indexing (Stage 5)
    chunk_cfg = config.get("chunking", {})
    default_chunk_size = chunk_cfg.get("default_chunk_size", 120)
    default_chunk_overlap = chunk_cfg.get("default_chunk_overlap", 25)
    dedup_headers = chunk_cfg.get("deduplicate_repeated_headers", True)

    try:
        chunker = DocumentChunker(
            chunk_size=default_chunk_size,
            chunk_overlap=default_chunk_overlap,
            deduplicate_repeated=dedup_headers,
        )
        chunks, chunk_stats = chunker.chunk_pages(
            pages, source_file=doc_display_name, sections=extraction_result.sections
        )
        retrieval_engine = TFIDFRetrievalEngine(default_threshold=0.15)
        retrieval_engine.index_chunks(chunks)
    except Exception as e:
        st.error(f"❌ Chunking & Retrieval Indexing Error: {str(e)}")
        chunks = []
        chunk_stats = None
        retrieval_engine = None

    # 3. Key Metrics Header Row
    consistency_counts = consistency_report.get_counts() if consistency_report else {}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("📄 Total Pages", doc_overview["total_pages"])
    m2.metric("📑 Sections Found", len(extraction_result.sections))
    m3.metric("🧩 SWCs Detected", summary_counts.get(EntityType.SWC.value, 0))
    m4.metric("🔌 Ports Identified", summary_counts.get(EntityType.PORT.value, 0))
    m5.metric("⚡ Runnables", summary_counts.get(EntityType.RUNNABLE.value, 0))
    m6.metric(
        "🛡️ Rule Checks",
        f"{len(consistency_report.findings)} Findings" if consistency_report else "N/A",
        f"{consistency_counts.get('PASS', 0)} Pass | {consistency_counts.get('WARNING', 0)} Warn | {consistency_counts.get('ERROR', 0)} Err"
        if consistency_report else None,
    )

    # 4. Document Metadata Expander
    with st.expander("🔍 Document Metadata & File Details", expanded=False):
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown(f"**Filename:** `{metadata['filename']}`")
            st.markdown(f"**Title:** {metadata['title']}")
            st.markdown(f"**Author / Architect:** {metadata['author']}")
            st.markdown(f"**Subject:** {metadata['subject']}")
        with meta_col2:
            st.markdown(f"**Producer:** {metadata['producer']}")
            st.markdown(f"**Created Date:** {metadata['creation_date']}")
            st.markdown(f"**Modified Date:** {metadata['mod_date']}")
            st.markdown(f"**File Size:** `{metadata['file_size_kb']} KB`")

    st.markdown("---")

    # 5. Primary Dashboard Tabs
    tab_chat, tab_consistency, tab_retrieval, tab_benchmark, tab_model, tab_sections, tab_pages, tab_full, tab_search = st.tabs([
        "💬 Grounded AI Engineering Chat",
        "🛡️ Architectural Consistency Checks",
        "🔬 AI/ML Retrieval & Transparency Studio",
        "📊 Benchmark Evaluation Studio",
        "🧩 Structured AUTOSAR Model",
        "📑 Sections & Structure",
        "📖 Page-by-Page Viewer",
        "📜 Full Document Text",
        "🔎 Raw Keyword Search",
    ])

    # =========================================================================
    # Tab 1: Grounded AI Engineering Chat (Phase 3)
    # =========================================================================
    with tab_chat:
        st.subheader("💬 AUTOSAR Engineering Chat Assistant")
        st.caption(
            "Ask technical questions about the uploaded HLD. Answers are strictly grounded in document evidence "
            "with verifiable page citations."
        )

        if not active_api_key:
            st.warning("🔑 **Gemini API Key Not Configured**")
            st.markdown(
                """
                To enable grounded AI analysis and chat:
                1. Add `GEMINI_API_KEY=your_key` to a local `.env` file, or
                2. Enter your key in the **Gemini AI Settings** section in the left sidebar.
                
                *Note: All other tabs (Structured Model, Sections, Page Viewer, Full Text, and JSON Download) remain 100% functional without an API key!*
                """
            )
        else:
            # Initialize conversation state
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Chat controls row
            c_col1, c_col2 = st.columns([4, 1])
            with c_col1:
                st.caption(f"Connected to Model: `{selected_model}` | Grounding: Active")
            with c_col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            # Suggested questions for quick start
            st.markdown("**Sample Architectural Queries:**")
            q_cols = st.columns(4)
            sample_queries = [
                "What SWCs are defined and what are their types?",
                "List all Ports, their interfaces and direction.",
                "Detail all Runnables, triggers, and OS tasks.",
                "What BSW modules and ASIL rating are specified?",
            ]
            clicked_query = None
            for idx, q_text in enumerate(sample_queries):
                if q_cols[idx].button(q_text, key=f"sample_q_{idx}", use_container_width=True):
                    clicked_query = q_text

            # Render existing chat conversation
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input field or button triggered prompt
            user_input = st.chat_input("Ask an architectural question about this AUTOSAR HLD...")
            prompt_to_run = clicked_query or user_input

            if prompt_to_run:
                # Append user prompt to state
                st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})
                with st.chat_message("user"):
                    st.markdown(prompt_to_run)

                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document evidence and generating grounded response..."):
                        try:
                            client = GeminiClient(api_key=active_api_key, model=selected_model)
                            response = client.generate_grounded_response(
                                user_query=prompt_to_run,
                                doc_overview=doc_overview,
                                extraction_result=extraction_result,
                                chat_history=st.session_state.chat_history[:-1],
                            )
                            st.markdown(response.content)
                            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                        except LLMConfigurationError as e:
                            st.error(f"🔑 Configuration Error: {str(e)}")
                        except LLMQuotaExceededError as e:
                            st.error(f"⏳ Quota Limit Reached: {str(e)}")
                        except LLMNetworkError as e:
                            st.error(f"🌐 Network Connection Error: {str(e)}")
                        except LLMResponseError as e:
                            st.error(f"⚠️ Response Error: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Unexpected Error: {str(e)}")

    # =========================================================================
    # Tab 2: Architectural Consistency Checks (Stage 5)
    # =========================================================================
    with tab_consistency:
        st.subheader("🛡️ Deterministic AUTOSAR Architectural Consistency Checks")
        st.caption(
            "Formal AUTOSAR rule evaluation with strict evidence provenance. Audits port-to-interface bindings, "
            "runnable triggers, OS task mappings, Client-Server timeout handling, and ISO 26262 ASIL / E2E protection."
        )

        if not consistency_report:
            st.warning("No consistency report available for this document.")
        else:
            c_counts = consistency_report.get_counts()

            # Severity Summary Cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📋 Rules Evaluated", consistency_report.total_rules_checked)
            c2.metric("🟢 PASS (Verified)", c_counts.get("PASS", 0))
            c3.metric("🟡 WARNING (Risk)", c_counts.get("WARNING", 0))
            c4.metric("🔴 ERROR (Violation)", c_counts.get("ERROR", 0))
            c5.metric("⚪ UNVERIFIED (Review)", c_counts.get("UNVERIFIED", 0))

            st.markdown("---")

            # Filters Row
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 2.5, 2])

            with f_col1:
                sev_options = ["ALL", "ERROR", "WARNING", "PASS", "UNVERIFIED"]
                selected_sev = st.selectbox("Filter by Severity:", sev_options, index=0, key="consist_filter_sev")

            with f_col2:
                available_rules = ["ALL"] + sorted(list({f.rule_id for f in consistency_report.findings}))
                selected_rule = st.selectbox("Filter by Rule ID:", available_rules, index=0, key="consist_filter_rule")

            with f_col3:
                finding_search = st.text_input(
                    "🔍 Search findings & evidence:",
                    placeholder="Search entity, rule, recommendation...",
                    key="consist_search",
                )

            with f_col4:
                st.write("")
                only_safety = st.checkbox(
                    "🛡️ Safety Review Only (ISO 26262)",
                    value=False,
                    help="Show only architectural findings flagged for Functional Safety Engineer inspection.",
                    key="consist_safety_only",
                )

            # Apply filters
            filtered_findings = consistency_report.findings
            if selected_sev != "ALL":
                filtered_findings = [f for f in filtered_findings if f.severity.value == selected_sev]
            if selected_rule != "ALL":
                filtered_findings = [f for f in filtered_findings if f.rule_id == selected_rule]
            if only_safety:
                filtered_findings = [f for f in filtered_findings if f.requires_safety_review]
            if finding_search.strip():
                q = finding_search.strip().lower()
                filtered_findings = [
                    f
                    for f in filtered_findings
                    if q in f.target_entity.lower()
                    or q in f.rule_id.lower()
                    or q in f.rule_name.lower()
                    or q in f.message.lower()
                    or q in f.recommendation.lower()
                    or q in f.supporting_evidence.lower()
                ]

            st.caption(f"Showing **{len(filtered_findings)}** of **{len(consistency_report.findings)}** evaluated findings.")

            if filtered_findings:
                # Tabular Overview
                finding_rows = []
                for f in filtered_findings:
                    finding_rows.append({
                        "Severity": format_severity_badge(f.severity.value),
                        "Rule ID": f.rule_id,
                        "Rule Name": f.rule_name,
                        "Target Entity": f.target_entity,
                        "Entity Type": f.target_entity_type,
                        "Page": f.page_number,
                        "Safety Review": "🛡️ Yes" if f.requires_safety_review else "No",
                        "Finding Message": f.message,
                        "Engineering Recommendation": f.recommendation,
                        "Supporting Evidence": f.supporting_evidence,
                    })
                df_findings = pd.DataFrame(finding_rows)
                st.dataframe(
                    df_findings,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Severity": st.column_config.TextColumn("Severity", width="small"),
                        "Rule ID": st.column_config.TextColumn("Rule ID", width="small"),
                        "Rule Name": st.column_config.TextColumn("Rule Name", width="medium"),
                        "Target Entity": st.column_config.TextColumn("Target Entity", width="medium"),
                        "Entity Type": st.column_config.TextColumn("Type", width="small"),
                        "Page": st.column_config.NumberColumn("Page", width="small"),
                        "Safety Review": st.column_config.TextColumn("Safety", width="small"),
                        "Finding Message": st.column_config.TextColumn("Message", width="large"),
                        "Engineering Recommendation": st.column_config.TextColumn("Actionable Recommendation", width="large"),
                        "Supporting Evidence": st.column_config.TextColumn("Evidence", width="large"),
                    },
                )

                # Expandable detailed card inspector
                st.markdown("#### 🔎 Detailed Findings & Actionable Recommendations")
                for idx, f in enumerate(filtered_findings):
                    badge = format_severity_badge(f.severity.value)
                    safety_tag = " | 🛡️ **SAFETY REVIEW REQUIRED**" if f.requires_safety_review else ""
                    expander_title = f"{badge} [{f.rule_id}] {f.target_entity} — {f.rule_name}{safety_tag}"

                    with st.expander(expander_title, expanded=(f.severity in (Severity.ERROR, Severity.WARNING))):
                        col_info1, col_info2, col_info3 = st.columns(3)
                        col_info1.markdown(f"**Target Entity:** `{f.target_entity}` ({f.target_entity_type})")
                        col_info2.markdown(f"**Document Page:** `Page {f.page_number}`")
                        col_info3.markdown(f"**Rule Specification:** `{f.rule_id} - {f.rule_name}`")

                        if f.severity == Severity.ERROR:
                            st.error(f"**Violation:** {f.message}")
                        elif f.severity == Severity.WARNING:
                            st.warning(f"**Risk / Inconsistency:** {f.message}")
                        elif f.severity == Severity.PASS:
                            st.success(f"**Verified:** {f.message}")
                        else:
                            st.info(f"**Unverified / Review:** {f.message}")

                        st.markdown(f"**💡 Actionable Recommendation:**\n> {f.recommendation}")
                        st.markdown(f"**📄 Verbatim Evidence Excerpt:**\n```text\n{f.supporting_evidence}\n```")
            else:
                st.info("No findings match the current filter criteria.")

            # Download Consistency Report JSON
            st.divider()
            consistency_json = json.dumps(consistency_report.to_dict(), indent=2)
            st.download_button(
                label="💾 Download Consistency Report JSON",
                data=consistency_json,
                file_name=f"{Path(metadata['filename']).stem}_consistency_report.json",
                mime="application/json",
                help="Download full consistency report with all evaluated rules, severities, and evidence excerpts.",
                use_container_width=True,
            )

    # =========================================================================
    # Tab 3: AI/ML Retrieval & Transparency Studio (Stage 5)
    # =========================================================================
    with tab_retrieval:
        st.subheader("🔬 AI/ML Vector Retrieval & Mathematical Transparency Studio")
        st.caption(
            "Auditable TF-IDF vectorization, Cosine Similarity retrieval, threshold filtering, "
            "and Latent Semantic Analysis / Truncated SVD dimensionality reduction."
        )

        r_section1, r_section2, r_section3 = st.tabs([
            "1️⃣ Preprocessing & Chunking",
            "2️⃣ TF-IDF Cosine Retrieval",
            "3️⃣ Truncated SVD Experiment",
        ])

        # --- Section 1: Preprocessing & Chunking ---
        with r_section1:
            st.markdown("#### 📄 Document Preprocessing & Sliding-Window Chunking")
            st.caption(
                "Inspect text sanitization, recurring header removal, and configurable sliding-window chunking."
            )

            chunk_col1, chunk_col2, chunk_col3 = st.columns([1.5, 1.5, 1])
            with chunk_col1:
                ui_chunk_size = st.slider(
                    "Chunk Size (words):",
                    min_value=30,
                    max_value=300,
                    value=default_chunk_size,
                    step=10,
                    key="ui_chunk_size",
                    help="Target word count per sliding window chunk.",
                )
            with chunk_col2:
                ui_chunk_overlap = st.slider(
                    "Chunk Overlap (words):",
                    min_value=0,
                    max_value=min(60, max(0, ui_chunk_size - 10)),
                    value=min(default_chunk_overlap, max(0, ui_chunk_size - 10)),
                    step=5,
                    key="ui_chunk_overlap",
                    help="Number of overlapping words between consecutive chunks.",
                )
            with chunk_col3:
                ui_dedup = st.checkbox(
                    "Deduplicate Recurring Headers",
                    value=dedup_headers,
                    key="ui_dedup",
                    help="Detects and eliminates lines recurring across >= 70% of pages.",
                )

            # Re-chunk on demand if parameters change
            if (
                ui_chunk_size != default_chunk_size
                or ui_chunk_overlap != default_chunk_overlap
                or ui_dedup != dedup_headers
            ):
                active_chunker = DocumentChunker(
                    chunk_size=ui_chunk_size,
                    chunk_overlap=ui_chunk_overlap,
                    deduplicate_repeated=ui_dedup,
                )
                active_chunks, active_stats = active_chunker.chunk_pages(
                    pages, source_file=doc_display_name, sections=extraction_result.sections
                )
                active_engine = TFIDFRetrievalEngine(default_threshold=0.15)
                active_engine.index_chunks(active_chunks)
            else:
                active_chunks = chunks
                active_stats = chunk_stats
                active_engine = retrieval_engine

            if active_stats:
                st.markdown("##### 📊 Preprocessing Diagnostics & Statistics")
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("📦 Total Chunks", active_stats.total_chunks)
                p2.metric("📝 Total Words", active_stats.total_words)
                p3.metric("📏 Avg Chunk Words", f"{active_stats.avg_chunk_words:.1f}")
                p4.metric("🧹 Cleaned Chars", active_stats.total_cleaned_characters)
                p5.metric("🚫 Repeated Lines Filtered", active_stats.repeated_lines_removed)

            # Chunks inspector
            with st.expander(f"📑 Inspect All {len(active_chunks)} Generated Chunks", expanded=False):
                chunk_table_rows = []
                for c in active_chunks:
                    chunk_table_rows.append({
                        "Chunk ID": c.chunk_id,
                        "Page": c.page_number,
                        "Section": c.section_title or "N/A",
                        "Words": c.word_count,
                        "Chars": c.char_count,
                        "Text Snippet": c.text[:150] + ("..." if len(c.text) > 150 else ""),
                    })
                st.dataframe(pd.DataFrame(chunk_table_rows), use_container_width=True, hide_index=True)

        # --- Section 2: TF-IDF Cosine Retrieval ---
        with r_section2:
            st.markdown("#### 🎯 Transparent TF-IDF Cosine Similarity Retrieval")
            st.caption(
                "Pure NumPy vectorization with sublinear TF scaling, smoothed IDF, L2 normalization, "
                "and explicit threshold accept/reject boundaries."
            )

            # Pre-built query chips
            st.markdown("**Suggested Automotive Test Queries:**")
            q_cols = st.columns(4)
            sample_retrieval_queries = [
                "Primary SWC VehicleSpeedManagement and type",
                "Periodic runnable RE_CalculateSpeed timing and OS task",
                "Sender-Receiver and Client-Server port interfaces",
                "ISO 26262 ASIL rating and E2E protection profile",
            ]
            clicked_retrieval_query = None
            for idx, q_txt in enumerate(sample_retrieval_queries):
                if q_cols[idx].button(q_txt, key=f"ret_sample_q_{idx}", use_container_width=True):
                    clicked_retrieval_query = q_txt

            user_retrieval_query = st.text_input(
                "Search query for similarity retrieval:",
                value=clicked_retrieval_query if clicked_retrieval_query else "VehicleSpeedManagement SWC",
                key="retrieval_user_query",
            )

            param_col1, param_col2 = st.columns(2)
            max_k_possible = max(1, len(active_chunks))
            with param_col1:
                ret_top_k = st.slider(
                    "Top-K Chunks to Retrieve:",
                    min_value=1,
                    max_value=min(10, max_k_possible),
                    value=min(5, max_k_possible),
                    key="ret_top_k",
                )
            with param_col2:
                ret_threshold = st.slider(
                    "Cosine Similarity Threshold (τ):",
                    min_value=0.00,
                    max_value=0.60,
                    value=0.15,
                    step=0.01,
                    key="ret_threshold",
                )

            if user_retrieval_query.strip() and active_engine:
                scored_results = active_engine.retrieve(user_retrieval_query, top_k=ret_top_k, threshold=ret_threshold)

                # Vocabulary transparency
                query_tokens = active_engine.vectorizer.tokenize(user_retrieval_query)
                matched_tokens = [t for t in query_tokens if t in active_engine.vectorizer.vocabulary]
                oov_tokens = [t for t in query_tokens if t not in active_engine.vectorizer.vocabulary]

                with st.expander("🧮 Mathematical Tokenization & Vocabulary Weights", expanded=False):
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.markdown(f"**Corpus Vocabulary Size:** `{len(active_engine.vectorizer.vocabulary)} unique terms`")
                        st.markdown(f"**Query Tokens Extracted:** `{', '.join(query_tokens) if query_tokens else 'None'}`")
                    with v_col2:
                        st.markdown(f"**Matched In-Vocabulary:** `{', '.join(matched_tokens) if matched_tokens else 'None'}`")
                        st.markdown(f"**Out-of-Vocabulary (OOV):** `{', '.join(oov_tokens) if oov_tokens else 'None'}`")

                    if matched_tokens:
                        vocab_rows = []
                        for t in set(matched_tokens):
                            idx = active_engine.vectorizer.vocabulary[t]
                            idf_val = float(active_engine.vectorizer.idf_vector[idx])
                            vocab_rows.append({"Token": t, "Vocab Index": idx, "Smoothed IDF Weight": round(idf_val, 4)})
                        st.dataframe(pd.DataFrame(vocab_rows), hide_index=True)

                st.markdown(f"##### 📋 Retrieval Results ({len(scored_results)} chunks evaluated):")
                if scored_results:
                    ret_rows = []
                    for sc in scored_results:
                        badge = "✅ ACCEPTED" if sc.is_accepted else "❌ REJECTED"
                        ret_rows.append({
                            "Rank": sc.rank,
                            "Status": badge,
                            "Cosine Score": f"{sc.similarity_score:.4f}",
                            "Threshold": f"{sc.threshold:.2f}",
                            "Page": sc.chunk.page_number,
                            "Section": sc.chunk.section_title or "N/A",
                            "Chunk ID": sc.chunk.chunk_id,
                            "Status Reason": sc.status_reason,
                            "Chunk Excerpt": sc.chunk.text[:200] + ("..." if len(sc.chunk.text) > 200 else ""),
                        })
                    st.dataframe(
                        pd.DataFrame(ret_rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small"),
                            "Status": st.column_config.TextColumn("Status", width="small"),
                            "Cosine Score": st.column_config.TextColumn("Score (Cosine)", width="small"),
                            "Threshold": st.column_config.TextColumn("Cutoff", width="small"),
                            "Page": st.column_config.NumberColumn("Page", width="small"),
                            "Section": st.column_config.TextColumn("Section", width="medium"),
                            "Chunk ID": st.column_config.TextColumn("Chunk ID", width="small"),
                            "Status Reason": st.column_config.TextColumn("Audit Reason", width="medium"),
                            "Chunk Excerpt": st.column_config.TextColumn("Excerpt", width="large"),
                        },
                    )

                    for sc in scored_results:
                        status_icon = "🟢" if sc.is_accepted else "⚪"
                        with st.expander(f"{status_icon} Rank #{sc.rank} — Page {sc.chunk.page_number} ({sc.chunk.chunk_id}) | Cosine Score: {sc.similarity_score:.4f}"):
                            st.markdown(f"**Evaluation Reason:** `{sc.status_reason}`")
                            st.markdown(f"**Section:** `{sc.chunk.section_title or 'General'}` | **Word Count:** `{sc.chunk.word_count}`")
                            st.markdown("**Verbatim Chunk Text:**")
                            st.text_area(label="", value=sc.chunk.text, height=120, key=f"chunk_txt_{sc.chunk.chunk_id}_{sc.rank}")
                else:
                    st.warning("No chunks available for retrieval.")

        # --- Section 3: Truncated SVD Experiment ---
        with r_section3:
            st.markdown("#### 📐 Latent Semantic Analysis / Truncated SVD Experiment")
            st.caption(
                "Compare exact lexical TF-IDF retrieval against dense low-rank SVD/PCA semantic representation."
            )

            max_svd = max(2, min(len(active_chunks), 15))
            svd_col1, svd_col2 = st.columns([2, 1])
            with svd_col1:
                n_svd_components = st.slider(
                    "SVD Latent Components (k):",
                    min_value=2,
                    max_value=max_svd,
                    value=min(5, max_svd),
                    key="svd_n_components",
                    help="Number of singular value dimensions to retain in the reduced latent space.",
                )
            with svd_col2:
                run_svd = st.button("🔬 Run SVD Comparison", use_container_width=True, key="btn_run_svd")

            if active_engine and (run_svd or "last_svd_query" in st.session_state):
                st.session_state["last_svd_query"] = user_retrieval_query
                comparison = active_engine.compare_with_pca(
                    query=user_retrieval_query,
                    n_components=n_svd_components,
                    top_k=ret_top_k,
                    threshold=ret_threshold,
                )

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Features -> Latent", f"{comparison.total_features} -> {comparison.n_components}")
                s2.metric("Explained Variance", f"{((comparison.explained_variance_ratio or 0.0) * 100):.1f}%")
                s3.metric("Baseline TF-IDF Latency", f"{comparison.baseline_latency_ms:.2f} ms")
                s4.metric("Reduced SVD Latency", f"{comparison.reduced_latency_ms:.2f} ms" if comparison.reduced_latency_ms else "N/A")

                st.info(f"💡 **Architectural Note:** {comparison.notes}")

                st.markdown("##### ⚖️ Side-by-Side Retrieval Ranking")
                comp_rows = []
                base_dict = {sc.chunk.chunk_id: sc for sc in comparison.baseline_results}
                red_dict = {sc.chunk.chunk_id: sc for sc in (comparison.reduced_results or [])}
                all_ids = list(base_dict.keys())

                for idx, cid in enumerate(all_ids):
                    b_sc = base_dict.get(cid)
                    r_sc = red_dict.get(cid)
                    b_score = b_sc.similarity_score if b_sc else 0.0
                    r_score = r_sc.similarity_score if r_sc else 0.0
                    comp_rows.append({
                        "Rank": idx + 1,
                        "Chunk ID": cid,
                        "Page": b_sc.chunk.page_number if b_sc else (r_sc.chunk.page_number if r_sc else 1),
                        "Baseline TF-IDF Score": f"{b_score:.4f}",
                        "SVD Reduced Score": f"{r_score:.4f}",
                        "Score Delta": f"{r_score - b_score:+.4f}",
                        "Baseline Accepted": "✅ Yes" if (b_sc and b_sc.is_accepted) else "❌ No",
                        "SVD Accepted": "✅ Yes" if (r_sc and r_sc.is_accepted) else "❌ No",
                    })

                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # Tab 4: Benchmark Evaluation Studio (Stage 5)
    # =========================================================================
    with tab_benchmark:
        st.subheader("📊 AUTOSAR Information Retrieval Benchmark Studio")
        st.caption(
            "Empirically evaluates the retrieval engine against 10 verified ground-truth queries grounded in "
            "`sample_autosar_hld.pdf` across 7 architectural categories. Computes Precision@K, Recall@K, "
            "F1-Score, Mean Reciprocal Rank (MRR), and Citation Accuracy."
        )

        # 1. Dataset Expander
        with st.expander(f"📚 View Manually Labelled Ground-Truth Benchmark Dataset ({len(AUTOSAR_SAMPLE_BENCHMARK)} Queries)", expanded=False):
            st.markdown(
                "Each query has been verified against the text of `data/input_docs/sample_autosar_hld.pdf`. "
                "Target pages and architectural entities serve as ground truth for IR metrics."
            )
            bq_rows = []
            for bq in AUTOSAR_SAMPLE_BENCHMARK:
                bq_rows.append({
                    "Query ID": bq.query_id,
                    "Category": bq.category,
                    "Query Text": bq.query_text,
                    "Target Pages": ", ".join(f"Page {p}" for p in bq.expected_target_pages),
                    "Expected Entities": ", ".join(bq.expected_entities),
                    "Ground Truth Reference Answer": bq.ground_truth_answer,
                })
            st.dataframe(
                pd.DataFrame(bq_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Query ID": st.column_config.TextColumn("ID", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Query Text": st.column_config.TextColumn("Query Text", width="large"),
                    "Target Pages": st.column_config.TextColumn("Target Pages", width="small"),
                    "Expected Entities": st.column_config.TextColumn("Expected Entities", width="medium"),
                    "Ground Truth Reference Answer": st.column_config.TextColumn("Reference Answer", width="large"),
                },
            )

        # 2. Benchmark Controls
        st.markdown("#### ⚙️ Benchmark Evaluation Configuration")
        b_col1, b_col2, b_col3, b_col4 = st.columns([1.5, 1.5, 2, 1.5])
        with b_col1:
            bench_k = st.slider("Evaluation K (Top-K):", min_value=1, max_value=5, value=3, key="bench_k")
        with b_col2:
            bench_threshold = st.slider("Similarity Cutoff (τ):", min_value=0.00, max_value=0.50, value=0.15, step=0.01, key="bench_threshold")
        with b_col3:
            bench_method = st.radio(
                "Retrieval Method:",
                ["Baseline TF-IDF", "Truncated SVD (PCA)"],
                horizontal=True,
                key="bench_method",
            )
        with b_col4:
            st.write("")
            run_benchmark_btn = st.button("🚀 Run Benchmark", use_container_width=True, key="btn_run_bench")

        use_pca_bench = (bench_method == "Truncated SVD (PCA)")
        pca_components = 5
        if use_pca_bench and retrieval_engine and retrieval_engine.chunks:
            max_components = max(2, min(len(retrieval_engine.chunks), 15))
            pca_components = st.slider(
                "SVD Latent Components:",
                min_value=2,
                max_value=max_components,
                value=min(5, max_components),
                key="bench_pca_components",
                help="Number of singular value decomposition components.",
            )

        bench_config = {
            "method": bench_method,
            "k": bench_k,
            "threshold": bench_threshold,
            "pca_components": pca_components if use_pca_bench else None,
            "doc": doc_display_name,
        }

        needs_evaluation = (
            run_benchmark_btn
            or "last_benchmark_report" not in st.session_state
            or st.session_state.get("last_bench_config") != bench_config
        )

        if retrieval_engine is not None and retrieval_engine.is_indexed:
            if needs_evaluation:
                with st.spinner(f"Evaluating {bench_method} against verified AUTOSAR benchmark..."):
                    evaluator = AUTOSARBenchmarkEvaluator()
                    bench_report = evaluator.evaluate_engine(
                        engine=retrieval_engine,
                        k=bench_k,
                        threshold=bench_threshold,
                        use_pca=use_pca_bench,
                        pca_components=pca_components,
                    )
                    st.session_state["last_benchmark_report"] = bench_report
                    st.session_state["last_bench_config"] = bench_config
            else:
                bench_report = st.session_state.get("last_benchmark_report")
        else:
            bench_report = None
            st.warning("⚠️ Retrieval engine is not ready or has no indexed chunks.")

        if bench_report:
            st.markdown("---")
            st.markdown(f"#### 🏆 Benchmark Performance Scorecard — `{bench_report.retrieval_method}` (K={bench_report.k}, τ={bench_report.threshold})")

            # 5 Key Metrics
            bm1, bm2, bm3, bm4, bm5 = st.columns(5)
            bm1.metric("🎯 Precision@K", f"{(bench_report.mean_precision_at_k * 100):.1f}%")
            bm2.metric("📥 Recall@K", f"{(bench_report.mean_recall_at_k * 100):.1f}%")
            bm3.metric("⚖️ F1-Score@K", f"{(bench_report.mean_f1_at_k * 100):.1f}%")
            bm4.metric("🥇 MRR", f"{bench_report.mrr:.4f}")
            bm5.metric("📌 Citation Accuracy", f"{(bench_report.mean_citation_accuracy * 100):.1f}%")

            st.caption(
                "💡 **Metric Definitions:** "
                "**Precision@K:** Fraction of retrieved top-K chunks that are relevant. "
                "**Recall@K:** Fraction of expected target pages captured in top-K. "
                "**MRR (Mean Reciprocal Rank):** Average reciprocal rank of the first relevant document chunk (1.0 = always rank 1). "
                "**Citation Accuracy:** Proportion of the top-K retrieved chunks whose source pages match the verified target pages. This metric evaluates retrieval provenance, not the correctness of generated AI answers."
            )

            # Query-by-Query Detailed Table
            st.markdown("##### 📋 Query-by-Query Evaluation Breakdown")
            q_res_rows = []
            for qr in bench_report.query_results:
                matched_pages_str = ", ".join(str(p) for p in qr.retrieved_pages)
                exp_pages_str = ", ".join(str(p) for p in qr.expected_pages)
                q_res_rows.append({
                    "Query ID": qr.query_id,
                    "Query Text": qr.query_text,
                    "Retrieved Pages": matched_pages_str,
                    "Expected Pages": exp_pages_str,
                    "Precision@K": f"{(qr.precision_at_k * 100):.1f}%",
                    "Recall@K": f"{(qr.recall_at_k * 100):.1f}%",
                    "F1-Score": f"{(qr.f1_at_k * 100):.1f}%",
                    "Reciprocal Rank": f"{qr.reciprocal_rank:.4f}",
                    "1st Rel Rank": qr.first_relevant_rank if qr.first_relevant_rank else "Not Found",
                    "Citation Accuracy": f"{(qr.citation_accuracy * 100):.1f}%",
                    "Accepted Chunks": qr.accepted_chunks_count,
                })

            st.dataframe(
                pd.DataFrame(q_res_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Query ID": st.column_config.TextColumn("Query ID", width="small"),
                    "Query Text": st.column_config.TextColumn("Query Text", width="large"),
                    "Retrieved Pages": st.column_config.TextColumn("Retrieved", width="small"),
                    "Expected Pages": st.column_config.TextColumn("Expected", width="small"),
                    "Precision@K": st.column_config.TextColumn("P@K", width="small"),
                    "Recall@K": st.column_config.TextColumn("R@K", width="small"),
                    "F1-Score": st.column_config.TextColumn("F1", width="small"),
                    "Reciprocal Rank": st.column_config.TextColumn("RR", width="small"),
                    "1st Rel Rank": st.column_config.TextColumn("Rank #1", width="small"),
                    "Citation Accuracy": st.column_config.TextColumn("Cite Acc", width="small"),
                    "Accepted Chunks": st.column_config.NumberColumn("Accepted", width="small"),
                },
            )

            # Download Benchmark Report JSON
            st.divider()
            bench_json = json.dumps(bench_report.to_dict(), indent=2)
            st.download_button(
                label="💾 Download Benchmark Evaluation Report JSON",
                data=bench_json,
                file_name=f"{Path(metadata['filename']).stem}_benchmark_report.json",
                mime="application/json",
                help="Download full benchmark report with query-level metrics and summary statistics.",
                use_container_width=True,
            )

    # =========================================================================
    # Tab 5: Structured AUTOSAR Model (Preserved from Phase 2)
    # =========================================================================
    with tab_model:
        st.subheader("🧩 Extracted AUTOSAR Architectural Entities")
        st.caption(
            "Every entity is grounded with source page numbers, supporting text, "
            "and explicit evidence status (Detected vs Inferred vs Unverified)."
        )

        # Evidence Status Legend Bar
        l_col1, l_col2, l_col3 = st.columns(3)
        l_col1.success("**✅ DETECTED:** Directly stated in text with concrete evidence.")
        l_col2.info("**💡 INFERRED:** Deduced from naming patterns or surrounding context.")
        l_col3.warning("**⚠️ UNVERIFIED:** Mentioned without full specification or marked under review.")

        # Entity Counts Mini Badges
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.caption(f"**SWCs:** {summary_counts.get(EntityType.SWC.value, 0)}")
        b2.caption(f"**Ports:** {summary_counts.get(EntityType.PORT.value, 0)}")
        b3.caption(f"**Interfaces:** {summary_counts.get(EntityType.INTERFACE.value, 0)}")
        b4.caption(f"**Runnables:** {summary_counts.get(EntityType.RUNNABLE.value, 0)}")
        b5.caption(f"**RTE Events:** {summary_counts.get(EntityType.RTE_EVENT.value, 0)}")
        b6.caption(f"**BSW Modules:** {summary_counts.get(EntityType.BSW_MODULE.value, 0)}")

        # Filters and Search
        f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2])

        with f_col1:
            all_types = ["ALL"] + [et.value for et in EntityType if et != EntityType.SECTION]
            selected_type = st.selectbox("Filter by Entity Type:", all_types, index=0)

        with f_col2:
            all_statuses = ["ALL"] + [es.value for es in EvidenceStatus]
            selected_status = st.selectbox("Filter by Evidence Status:", all_statuses, index=0)

        with f_col3:
            table_search = st.text_input("🔍 Search within entities & evidence:", placeholder="Type name or keyword...")

        # Filter entities based on selections
        filtered_entities = extraction_result.entities

        if selected_type != "ALL":
            filtered_entities = [e for e in filtered_entities if e.entity_type.value == selected_type]

        if selected_status != "ALL":
            filtered_entities = [e for e in filtered_entities if e.status.value == selected_status]

        if table_search.strip():
            query_str = table_search.strip().lower()
            filtered_entities = [
                e for e in filtered_entities
                if query_str in e.name.lower()
                or query_str in e.supporting_text.lower()
                or any(query_str in str(v).lower() for v in e.attributes.values())
            ]

        # Prepare Table Dataframe
        if filtered_entities:
            table_rows = []
            for e in filtered_entities:
                table_rows.append({
                    "Entity Type": e.entity_type.value,
                    "Name": e.name,
                    "Evidence Status": format_status_badge(e.status.value),
                    "Page": e.page_number,
                    "Architectural Attributes": format_attributes(e.attributes),
                    "Supporting Text (Evidence)": e.supporting_text,
                })

            df_entities = pd.DataFrame(table_rows)

            st.dataframe(
                df_entities,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Entity Type": st.column_config.TextColumn("Entity Type", width="small"),
                    "Name": st.column_config.TextColumn("Entity Name", width="medium"),
                    "Evidence Status": st.column_config.TextColumn("Evidence Status", width="small"),
                    "Page": st.column_config.NumberColumn("Page", width="small"),
                    "Architectural Attributes": st.column_config.TextColumn("Architectural Attributes", width="large"),
                    "Supporting Text (Evidence)": st.column_config.TextColumn("Supporting Text (Evidence)", width="large"),
                },
            )

            st.caption(f"Showing **{len(filtered_entities)}** of **{total_entities}** extracted entities.")
        else:
            st.warning("No architectural entities matched the selected filters.")

        # Download Structured Model JSON
        st.divider()
        json_content = extraction_result.to_json(indent=2)
        st.download_button(
            label="💾 Download Structured Model JSON",
            data=json_content,
            file_name=f"{Path(metadata['filename']).stem}_autosar_model.json",
            mime="application/json",
            help="Download the complete extracted architectural model including sections, entities, and evidence.",
            use_container_width=True,
        )

    # =========================================================================
    # Tab 3: Document Sections & Structure (Preserved from Phase 2)
    # =========================================================================
    with tab_sections:
        st.subheader("📑 Detected Document Sections & Headings")
        if extraction_result.sections:
            sec_rows = []
            for sec in extraction_result.sections:
                sec_rows.append({
                    "Section #": sec.section_number,
                    "Title": sec.title,
                    "Hierarchy Level": f"Level {sec.level}",
                    "Page": sec.page_number,
                    "Supporting Text": sec.supporting_text,
                })
            df_sections = pd.DataFrame(sec_rows)
            st.dataframe(
                df_sections,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Section #": st.column_config.TextColumn("Section #", width="small"),
                    "Title": st.column_config.TextColumn("Heading / Title", width="large"),
                    "Hierarchy Level": st.column_config.TextColumn("Depth", width="small"),
                    "Page": st.column_config.NumberColumn("Page", width="small"),
                    "Supporting Text": st.column_config.TextColumn("Verbatim Line", width="large"),
                },
            )
        else:
            st.warning("No explicit section headings were detected in this document.")

    # =========================================================================
    # Tab 4: Page-by-Page Viewer (Preserved from Phase 1)
    # =========================================================================
    with tab_pages:
        if doc_overview["total_pages"] == 0:
            st.warning("No pages found in this document.")
        else:
            selected_page_num = st.slider(
                "Select Page to Preview",
                min_value=1,
                max_value=doc_overview["total_pages"],
                value=1,
                help="Slide to preview any page of the uploaded document.",
            )
            page_info = pages[selected_page_num - 1]

            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.caption(f"**Page:** {page_info['page_number']} of {doc_overview['total_pages']}")
            col_p2.caption(f"**Words on Page:** {page_info['word_count']}")
            col_p3.caption(f"**Has Embedded Images:** {'Yes' if page_info['has_images'] else 'No'}")

            page_text = page_info["text"].strip()
            if page_text:
                st.text_area(
                    label=f"Content of Page {selected_page_num}",
                    value=page_text,
                    height=350,
                    key=f"text_area_page_{selected_page_num}",
                )
            else:
                st.warning("This page does not contain extractable plain text. It may contain scanned images or diagrams.")

    # =========================================================================
    # Tab 5: Full Document Text (Preserved from Phase 1)
    # =========================================================================
    with tab_full:
        full_text = "\n\n".join(
            f"=== Page {p['page_number']} ===\n{p['text']}" for p in pages
        )
        st.text_area(
            label="Complete Extracted Text",
            value=full_text,
            height=400,
            key="full_text_area",
        )
        st.download_button(
            label="💾 Download Extracted Text (.txt)",
            data=full_text,
            file_name=f"{Path(metadata['filename']).stem}_extracted.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # =========================================================================
    # Tab 6: Raw Keyword Search (Preserved from Phase 1)
    # =========================================================================
    with tab_search:
        st.subheader("🔎 Raw Text Search")
        search_query = st.text_input(
            "Search for any raw term or symbol:",
            placeholder="Type keyword here (e.g., 'Runnable', 'Sender-Receiver', 'Dem')...",
        )
        if search_query:
            query_lower = search_query.lower()
            matching_pages = []
            for page in pages:
                if query_lower in page["text"].lower():
                    occurrences = page["text"].lower().count(query_lower)
                    matching_pages.append((page["page_number"], occurrences, page["text"]))

            if matching_pages:
                st.success(
                    f"Found **{sum(m[1] for m in matching_pages)}** matches across **{len(matching_pages)}** page(s):"
                )
                for page_num, count, text in matching_pages:
                    with st.expander(f"Page {page_num} ({count} occurrence{'s' if count > 1 else ''})"):
                        idx = text.lower().find(query_lower)
                        start = max(0, idx - 100)
                        end = min(len(text), idx + 200)
                        snippet = text[start:end]
                        st.markdown(f"**Preview Snippet:**\n> ...`{snippet.replace(chr(10), ' ')}`...")
            else:
                st.warning(f"No occurrences of '{search_query}' found in this document.")


if __name__ == "__main__":
    main()
