"""Configurable Preprocessing and Chunking Engine for AUTOSAR HLD Documents.

Provides text cleaning, repeated header/footer detection, configurable sliding-window
chunking (size and overlap), page provenance preservation, and preprocessing diagnostics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .models import PreprocessingStats, SectionHeading, TextChunk


class DocumentChunker:
    """Cleans document text and divides it into overlapping chunks with provenance."""

    def __init__(
        self,
        chunk_size: int = 120,
        chunk_overlap: int = 25,
        deduplicate_repeated: bool = True,
    ) -> None:
        """Initializes chunker with configurable window parameters.

        Args:
            chunk_size: Target number of words per chunk (must be > 0).
            chunk_overlap: Number of overlapping words between consecutive chunks.
            deduplicate_repeated: Whether to detect and remove identical recurring headers/footers.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.deduplicate_repeated = deduplicate_repeated

    def clean_text(self, text: str) -> str:
        """Sanitizes text by removing null bytes, standardizing whitespace, and cleaning breaks."""
        if not text:
            return ""

        # Remove null bytes and non-printable control characters (except newline, tab)
        cleaned = text.replace("\x00", "")

        # Standardize carriage returns
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Replace non-breaking space
        cleaned = cleaned.replace("\xa0", " ")

        # Strip trailing spaces on lines and collapse multiple horizontal spaces
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]

        # Collapse 3 or more consecutive blank lines into 2
        collapsed_text = "\n".join(lines)
        collapsed_text = re.sub(r"\n{3,}", "\n\n", collapsed_text).strip()

        return collapsed_text

    def _detect_repeated_lines(self, pages: List[Dict[str, Any]]) -> List[str]:
        """Finds recurring lines that appear across 70% or more of pages (e.g. running headers)."""
        if len(pages) < 2:
            return []

        line_page_counts: Dict[str, int] = {}
        for p in pages:
            raw_text = p.get("text", "")
            cleaned = self.clean_text(raw_text)
            unique_lines_in_page = set(line for line in cleaned.splitlines() if len(line) > 5)
            for line in unique_lines_in_page:
                line_page_counts[line] = line_page_counts.get(line, 0) + 1

        threshold = max(2, int(len(pages) * 0.7))
        repeated = [line for line, count in line_page_counts.items() if count >= threshold]
        return repeated

    def _resolve_section(self, page_num: int, sections: Optional[List[SectionHeading]]) -> Optional[str]:
        """Finds the most specific section title active on or before the given page number."""
        if not sections:
            return None

        active_title: Optional[str] = None
        for s in sections:
            if s.page_number <= page_num:
                active_title = s.title
            else:
                break
        return active_title

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        source_file: str = "document.pdf",
        sections: Optional[List[SectionHeading]] = None,
    ) -> Tuple[List[TextChunk], PreprocessingStats]:
        """Processes document pages into overlapping TextChunks and produces preprocessing stats.

        Args:
            pages: List of page dictionaries with 'page_number' and 'text'.
            source_file: Filename of the source document.
            sections: Optional list of detected SectionHeadings to associate with chunks.

        Returns:
            Tuple of (list of TextChunks, PreprocessingStats).
        """
        raw_char_count = sum(len(p.get("text", "")) for p in pages)
        empty_pages: List[int] = []
        chunks: List[TextChunk] = []

        repeated_lines: List[str] = []
        if self.deduplicate_repeated:
            repeated_lines = self._detect_repeated_lines(pages)

        repeated_lines_removed_count = 0
        cleaned_char_count = 0
        total_words_count = 0
        chunk_counter = 0

        for page in pages:
            page_num = page.get("page_number", 1)
            raw_text = page.get("text", "")
            cleaned = self.clean_text(raw_text)

            # Strip recurring headers/footers if enabled
            if repeated_lines:
                lines = cleaned.splitlines()
                retained_lines = []
                for line in lines:
                    if line in repeated_lines:
                        repeated_lines_removed_count += 1
                    else:
                        retained_lines.append(line)
                cleaned = "\n".join(retained_lines).strip()

            cleaned_char_count += len(cleaned)
            words = cleaned.split()
            word_count = len(words)
            total_words_count += word_count

            if word_count == 0:
                empty_pages.append(page_num)
                continue

            section_title = self._resolve_section(page_num, sections)

            # If page text is smaller than or equal to chunk_size, create 1 chunk
            if word_count <= self.chunk_size:
                chunk_counter += 1
                chunk_id = f"chunk_p{page_num}_{chunk_counter}"
                chunk_text = " ".join(words)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        page_number=page_num,
                        source_file=source_file,
                        word_count=word_count,
                        char_count=len(chunk_text),
                        chunk_index=chunk_counter,
                        section_title=section_title,
                    )
                )
            else:
                # Sliding window chunking with configurable overlap
                step = max(1, self.chunk_size - self.chunk_overlap)
                start_idx = 0
                page_chunk_idx = 0

                while start_idx < word_count:
                    end_idx = min(start_idx + self.chunk_size, word_count)
                    chunk_words = words[start_idx:end_idx]
                    chunk_text = " ".join(chunk_words)

                    chunk_counter += 1
                    page_chunk_idx += 1
                    chunk_id = f"chunk_p{page_num}_{page_chunk_idx}"

                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            page_number=page_num,
                            source_file=source_file,
                            word_count=len(chunk_words),
                            char_count=len(chunk_text),
                            chunk_index=chunk_counter,
                            section_title=section_title,
                        )
                    )

                    # If end of page reached, exit loop
                    if end_idx >= word_count:
                        break
                    start_idx += step

        avg_words = (
            round(sum(c.word_count for c in chunks) / len(chunks), 2)
            if chunks
            else 0.0
        )

        stats = PreprocessingStats(
            source_file=source_file,
            total_raw_pages=len(pages),
            total_raw_characters=raw_char_count,
            total_cleaned_characters=cleaned_char_count,
            total_words=total_words_count,
            total_chunks=len(chunks),
            chunk_size_words=self.chunk_size,
            chunk_overlap_words=self.chunk_overlap,
            avg_chunk_words=avg_words,
            empty_pages_detected=empty_pages,
            repeated_lines_removed=repeated_lines_removed_count,
        )

        return chunks, stats
