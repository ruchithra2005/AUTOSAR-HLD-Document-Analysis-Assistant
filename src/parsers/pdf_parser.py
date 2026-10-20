"""PDF Parser Module using PyMuPDF (fitz) for AUTOSAR HLD documents.

Designed to be beginner-friendly, robust, and handle both local file paths
and uploaded in-memory file streams from Streamlit.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf  # Fallback for older installations
    except ImportError as err:
        pymupdf = None
        _import_error = err
    else:
        _import_error = None
else:
    _import_error = None


class PDFExtractionError(Exception):
    """Custom exception raised when PDF extraction encounters an error."""
    pass


class PDFParser:
    """Extracts text, metadata, and structural cues from PDF documents using PyMuPDF."""

    # Common AUTOSAR HLD architectural keywords to scan for quick document overview
    AUTOSAR_KEYWORDS = [
        "Software Component",
        "SWC",
        "Application SWC",
        "Sensor/Actuator SWC",
        "Composition",
        "Sender-Receiver",
        "Client-Server",
        "Port Interface",
        "P-Port",
        "R-Port",
        "PR-Port",
        "Runnable",
        "RTE",
        "RTE Event",
        "Timing Event",
        "Data Received Event",
        "BSW",
        "MCAL",
        "ComStack",
        "Dem",
        "Dcm",
        "NvM",
        "EcuM",
        "BswM",
        "OS Task",
        "ASIL",
        "Functional Safety",
    ]

    def __init__(self) -> None:
        if pymupdf is None:
            raise PDFExtractionError(
                f"PyMuPDF is not installed or failed to import: {_import_error}. "
                "Please run 'pip install pymupdf' to install it."
            )

    def _open_document(self, source: Union[str, Path, bytes, io.BytesIO]) -> pymupdf.Document:
        """Internal helper to safely open a PDF from path, bytes, or file-like object."""
        try:
            if isinstance(source, (str, Path)):
                path_obj = Path(source)
                if not path_obj.exists():
                    raise PDFExtractionError(f"File not found: {path_obj}")
                return pymupdf.open(str(path_obj))
            elif isinstance(source, bytes):
                return pymupdf.open(stream=source, filetype="pdf")
            elif hasattr(source, "read"):
                # Handle file-like objects (e.g., Streamlit UploadedFile)
                content = source.read()
                # Reset stream position if possible
                if hasattr(source, "seek"):
                    source.seek(0)
                return pymupdf.open(stream=content, filetype="pdf")
            else:
                raise PDFExtractionError(
                    f"Unsupported PDF source type: {type(source).__name__}. Expected file path, bytes, or file stream."
                )
        except Exception as e:
            if isinstance(e, PDFExtractionError):
                raise
            raise PDFExtractionError(f"Failed to open PDF document: {str(e)}") from e

    def get_metadata(self, source: Union[str, Path, bytes, io.BytesIO], filename: Optional[str] = None) -> Dict[str, Any]:
        """Extracts metadata and basic structural properties from the PDF."""
        doc = self._open_document(source)
        try:
            meta = doc.metadata or {}
            page_count = len(doc)
            is_encrypted = doc.is_encrypted

            # Calculate byte size if applicable
            file_size_kb = 0.0
            if isinstance(source, (str, Path)):
                file_size_kb = round(Path(source).stat().st_size / 1024, 2)
            elif isinstance(source, bytes):
                file_size_kb = round(len(source) / 1024, 2)
            elif hasattr(source, "size"):
                # Streamlit UploadedFile provides .size attribute
                file_size_kb = round(source.size / 1024, 2)

            resolved_filename = filename
            if not resolved_filename and isinstance(source, (str, Path)):
                resolved_filename = Path(source).name
            elif not resolved_filename and hasattr(source, "name"):
                resolved_filename = getattr(source, "name")

            return {
                "filename": resolved_filename or "document.pdf",
                "page_count": page_count,
                "file_size_kb": file_size_kb,
                "title": meta.get("title") or "Not Specified",
                "author": meta.get("author") or "Not Specified",
                "subject": meta.get("subject") or "Not Specified",
                "creator": meta.get("creator") or "Not Specified",
                "producer": meta.get("producer") or "Not Specified",
                "creation_date": meta.get("creationDate") or "Not Specified",
                "mod_date": meta.get("modDate") or "Not Specified",
                "is_encrypted": is_encrypted,
            }
        finally:
            doc.close()

    def extract_pages(self, source: Union[str, Path, bytes, io.BytesIO]) -> List[Dict[str, Any]]:
        """Extracts text and page-level metrics for each page in the PDF."""
        doc = self._open_document(source)
        pages_data: List[Dict[str, Any]] = []

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text") or ""
                # Strip out excessive null characters or weird binary artifacts if any
                cleaned_text = text.replace("\x00", "")

                words = cleaned_text.split()
                has_images = len(page.get_images()) > 0

                pages_data.append({
                    "page_number": page_index + 1,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "word_count": len(words),
                    "has_images": has_images,
                })

            return pages_data
        finally:
            doc.close()

    def extract_full_text(self, source: Union[str, Path, bytes, io.BytesIO]) -> str:
        """Extracts all text concatenated across all pages with page markers."""
        pages = self.extract_pages(source)
        full_text_chunks: List[str] = []
        for page in pages:
            full_text_chunks.append(
                f"--- [Page {page['page_number']}] ---\n{page['text']}\n"
            )
        return "\n".join(full_text_chunks)

    def scan_autosar_keywords(self, text: str) -> Dict[str, int]:
        """Scans extracted text for common AUTOSAR architectural terms and returns counts."""
        keyword_counts: Dict[str, int] = {}
        for keyword in self.AUTOSAR_KEYWORDS:
            # Word boundary matching where appropriate, case-insensitive
            escaped_kw = re.escape(keyword)
            matches = re.findall(rf"\b{escaped_kw}\b", text, flags=re.IGNORECASE)
            if matches:
                keyword_counts[keyword] = len(matches)
        return dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True))

    def get_document_overview(
        self, source: Union[str, Path, bytes, io.BytesIO], filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Provides a comprehensive, beginner-friendly summary of the parsed document."""
        metadata = self.get_metadata(source, filename=filename)
        pages = self.extract_pages(source)

        total_words = sum(p["word_count"] for p in pages)
        total_chars = sum(p["char_count"] for p in pages)
        combined_text = "\n".join(p["text"] for p in pages)

        autosar_detected = self.scan_autosar_keywords(combined_text)

        # Estimate reading time (average technical reading speed: ~150 words/minute)
        est_reading_minutes = max(1, round(total_words / 150)) if total_words > 0 else 0

        return {
            "metadata": metadata,
            "pages": pages,
            "total_pages": len(pages),
            "total_words": total_words,
            "total_characters": total_chars,
            "estimated_reading_minutes": est_reading_minutes,
            "autosar_keywords_found": autosar_detected,
            "has_text": total_words > 0,
        }
