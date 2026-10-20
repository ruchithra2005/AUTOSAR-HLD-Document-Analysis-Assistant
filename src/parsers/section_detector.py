"""Section and Heading Detector for AUTOSAR HLD documents.

Detects document structure, major chapters, sections, and subsections
across pages with full page number and provenance tracking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .models import SectionHeading


class SectionDetector:
    """Identifies section headings and structural divisions in document pages."""

    # Regex patterns for numbered headings (e.g., "1. ARCHITECTURAL OVERVIEW", "2.1 Ports")
    NUMBERED_HEADING_PATTERN = re.compile(
        r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Z][A-Za-z0-9\s&/\-_,]{2,80})$"
    )

    # Keywords typical for unnumbered top-level AUTOSAR HLD headings
    AUTOSAR_HEADING_KEYWORDS = [
        "ARCHITECTURAL OVERVIEW",
        "ARCHITECTURE OVERVIEW",
        "COMPONENT DESCRIPTION",
        "SOFTWARE COMPONENT DESCRIPTION",
        "PORT INTERFACES",
        "PORT INTERFACE SPECIFICATION",
        "DATA ELEMENTS",
        "RUNNABLES & RTE EVENT MAPPINGS",
        "RUNNABLE SPECIFICATION",
        "INTERNAL BEHAVIOR",
        "RTE EVENT MAPPINGS",
        "BSW & SAFETY REQUIREMENTS",
        "BSW DEPENDENCIES",
        "FUNCTIONAL SAFETY",
        "SAFETY REQUIREMENTS",
        "MEMORY MAPPING",
        "DIAGNOSTIC EVENT MANAGER",
        "VERIFICATION CHECKLIST",
    ]

    def detect_sections(self, pages: List[Dict[str, Any]], source_file: str = "document.pdf") -> List[SectionHeading]:
        """Scans page text and extracts candidate section headings."""
        detected_sections: List[SectionHeading] = []
        seen_titles = set()

        for page in pages:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]

            for line in lines:
                # 1. Check for standard numbered heading: e.g., "1. ARCHITECTURAL OVERVIEW"
                match = self.NUMBERED_HEADING_PATTERN.match(line)
                if match:
                    sec_num = match.group("number")
                    title = match.group("title").strip()
                    # Calculate depth level from dots (e.g. "1" -> 1, "1.2" -> 2)
                    level = sec_num.count(".") + 1
                    
                    # Avoid duplicate recordings if page headers repeat
                    dedup_key = (sec_num, title.upper())
                    if dedup_key not in seen_titles:
                        seen_titles.add(dedup_key)
                        detected_sections.append(
                            SectionHeading(
                                section_number=sec_num,
                                title=title,
                                page_number=page_num,
                                source_file=source_file,
                                supporting_text=line,
                                level=level,
                            )
                        )
                    continue

                # 2. Check for known unnumbered AUTOSAR headings
                normalized_upper = line.upper()
                for kw in self.AUTOSAR_HEADING_KEYWORDS:
                    if normalized_upper == kw or normalized_upper.startswith(f"{kw}:"):
                        dedup_key = ("UNNUMBERED", kw)
                        if dedup_key not in seen_titles:
                            seen_titles.add(dedup_key)
                            detected_sections.append(
                                SectionHeading(
                                    section_number="N/A",
                                    title=line,
                                    page_number=page_num,
                                    source_file=source_file,
                                    supporting_text=line,
                                    level=1,
                                )
                            )
                        break

        return detected_sections
