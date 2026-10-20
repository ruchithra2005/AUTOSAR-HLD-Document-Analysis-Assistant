"""Data models for structured AUTOSAR extraction in Phase 2.

Defines schemas for sections, SWCs, ports, interfaces, runnables,
RTE events, BSW modules, and dependencies with full provenance tracking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceStatus(str, Enum):
    """Categorization of extraction certainty / evidence status."""
    DETECTED = "DETECTED"        # Explicitly stated and verified in the source text
    INFERRED = "INFERRED"        # Derived from naming patterns or contextual proximity
    UNVERIFIED = "UNVERIFIED"    # Incomplete specification, draft status, or pending review


class EntityType(str, Enum):
    """Types of AUTOSAR architectural entities."""
    SECTION = "SECTION"
    SWC = "SWC"
    PORT = "PORT"
    INTERFACE = "INTERFACE"
    RUNNABLE = "RUNNABLE"
    RTE_EVENT = "RTE_EVENT"
    BSW_MODULE = "BSW_MODULE"
    DEPENDENCY = "DEPENDENCY"


@dataclass
class SectionHeading:
    """Represents a detected document section or heading."""
    section_number: str
    title: str
    page_number: int
    source_file: str
    supporting_text: str
    level: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedEntity:
    """Base class for all extracted AUTOSAR entities with evidence provenance."""
    name: str
    entity_type: EntityType
    page_number: int
    source_file: str
    supporting_text: str
    status: EvidenceStatus = EvidenceStatus.DETECTED
    attributes: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["entity_type"] = self.entity_type.value
        data["status"] = self.status.value
        return data


@dataclass
class ExtractionResult:
    """Container holding the complete structured extraction for a document."""
    source_file: str
    total_pages: int
    sections: List[SectionHeading] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    extraction_timestamp: str = ""

    def filter_by_type(self, entity_type: EntityType) -> List[ExtractedEntity]:
        """Returns only entities of the specified EntityType."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def filter_by_status(self, status: EvidenceStatus) -> List[ExtractedEntity]:
        """Returns only entities matching the specified EvidenceStatus."""
        return [e for e in self.entities if e.status == status]

    def get_summary_counts(self) -> Dict[str, int]:
        """Calculates counts per entity type."""
        counts: Dict[str, int] = {et.value: 0 for et in EntityType if et != EntityType.SECTION}
        for entity in self.entities:
            counts[entity.entity_type.value] = counts.get(entity.entity_type.value, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Converts complete extraction result into a serializable dictionary."""
        return {
            "source_file": self.source_file,
            "total_pages": self.total_pages,
            "extraction_timestamp": self.extraction_timestamp,
            "summary_counts": self.get_summary_counts(),
            "sections": [s.to_dict() for s in self.sections],
            "entities": [e.to_dict() for e in self.entities],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes extraction result into formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class TextChunk:
    """Represents a preprocessed, windowed chunk of text from the document."""
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    word_count: int
    char_count: int
    chunk_index: int
    section_title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessingStats:
    """Statistics detailing the preprocessing and chunking operations."""
    source_file: str
    total_raw_pages: int
    total_raw_characters: int
    total_cleaned_characters: int
    total_words: int
    total_chunks: int
    chunk_size_words: int
    chunk_overlap_words: int
    avg_chunk_words: float
    empty_pages_detected: List[int] = field(default_factory=list)
    repeated_lines_removed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
