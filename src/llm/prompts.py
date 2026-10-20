"""AUTOSAR Domain Prompt Engineering and Grounding Context Builder.

Ensures strict grounding, verifiable citations (page numbers and excerpts),
clear distinction between explicit document facts and architectural interpretations,
and mitigation of hallucinated compliance claims.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.parsers.models import ExtractionResult


class AUTOSARPromptBuilder:
    """Builds grounded prompt contexts for the Gemini LLM based on extracted HLD data."""

    SYSTEM_INSTRUCTION = """You are an expert Automotive Software Architect and AUTOSAR Review Assistant for the Tata Technologies TechPulse project.
Your task is to analyze High-Level Design (HLD) specifications and answer engineering queries.

STRICT GROUNDING & VERIFICATION RULES:
1. TRUTHFULNESS & GROUNDING: Base your answers ONLY on the supplied HLD document context and extracted architectural entities below.
2. CITATIONS: Every factual assertion MUST cite its location in the format: [Source: <filename>, Page: <page_number>] along with a brief supporting excerpt.
3. INSUFFICIENT EVIDENCE: If the document does not contain enough information to answer a question or verify an architectural aspect, state clearly: "The provided HLD does not contain sufficient evidence to answer this." DO NOT fabricate, guess, or assume missing parameters.
4. SEPARATION OF CONCERNS: Clearly separate your response into:
   - **Direct Answer**: Concise summary answering the user's question.
   - **Explicit Document Facts**: Direct facts explicitly stated in the HLD with page citations and quotes.
   - **Architectural Analysis & Potential Observations**: Logical deductions, potential gaps (e.g., missing timeouts, unmapped runnables, unverified connections).
   - **Unspecified / Missing Information**: Any relevant parameters or requirements not documented in the HLD.
5. NO UNSUBSTANTIATED COMPLIANCE CLAIMS: Never claim an architecture is "fully AUTOSAR compliant" or "ASIL compliant" unless the document explicitly provides verified evidence and traceability. Always note unverified items as such.
"""

    @classmethod
    def build_grounded_prompt(
        cls,
        user_query: str,
        doc_overview: Dict[str, Any],
        extraction_result: ExtractionResult,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Constructs a comprehensive, grounded prompt incorporating document context."""
        filename = doc_overview.get("metadata", {}).get("filename", "document.pdf")
        total_pages = doc_overview.get("total_pages", 0)

        # 1. Format Structured Entities
        structured_entities_summary = []
        for e in extraction_result.entities:
            attrs_str = ", ".join(f"{k}: {v}" for k, v in e.attributes.items())
            structured_entities_summary.append(
                f"- [{e.entity_type.value}] Name: '{e.name}' | Status: {e.status.value} | "
                f"Page: {e.page_number} | Details: {attrs_str} | Quote: \"{e.supporting_text}\""
            )

        entities_block = "\n".join(structured_entities_summary) if structured_entities_summary else "No entities extracted."

        # 2. Format Section Hierarchy
        sections_summary = []
        for s in extraction_result.sections:
            sections_summary.append(
                f"- Section {s.section_number}: \"{s.title}\" (Level {s.level}, Page {s.page_number})"
            )
        sections_block = "\n".join(sections_summary) if sections_summary else "No sections detected."

        # 3. Format Page-by-Page Raw Text
        pages_content = []
        for p in doc_overview.get("pages", []):
            page_num = p.get("page_number", 1)
            text = p.get("text", "").strip()
            pages_content.append(f"--- [START OF PAGE {page_num} ({filename})] ---\n{text}\n--- [END OF PAGE {page_num}] ---")

        raw_pages_block = "\n\n".join(pages_content) if pages_content else "No page text available."

        # 4. Optional Chat History Context
        history_block = ""
        if chat_history:
            history_lines = []
            for msg in chat_history[-6:]:  # Keep recent turns
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_block = "PRIOR CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""=== DOCUMENT METADATA ===
Source File: {filename}
Total Pages: {total_pages}

=== DETECTED DOCUMENT SECTIONS ===
{sections_block}

=== STRUCTURED AUTOSAR ARCHITECTURE MODEL ===
{entities_block}

=== VERBATIM DOCUMENT PAGE CONTENTS ===
{raw_pages_block}

{history_block}=== USER QUESTION ===
{user_query}

=== INSTRUCTIONS FOR YOUR RESPONSE ===
Provide a well-structured answer following the rules specified in your system instructions. Ensure citations [Source: {filename}, Page: <page_number>] accompany every factual statement.
"""
        return prompt
