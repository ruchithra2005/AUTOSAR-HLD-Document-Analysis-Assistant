"""Document parsers, chunking, and structured extraction package."""

from .chunker import DocumentChunker
from .models import (
    EntityType,
    EvidenceStatus,
    ExtractedEntity,
    ExtractionResult,
    PreprocessingStats,
    SectionHeading,
    TextChunk,
)
from .pdf_parser import PDFExtractionError, PDFParser
from .section_detector import SectionDetector
from .autosar_extractor import StructuredAUTOSARExtractor

__all__ = [
    "PDFParser",
    "PDFExtractionError",
    "SectionDetector",
    "StructuredAUTOSARExtractor",
    "DocumentChunker",
    "EntityType",
    "EvidenceStatus",
    "ExtractedEntity",
    "ExtractionResult",
    "SectionHeading",
    "TextChunk",
    "PreprocessingStats",
]
