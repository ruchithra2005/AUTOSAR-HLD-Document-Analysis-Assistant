"""Unit tests for DocumentChunker, text preprocessing, and chunk statistics."""

import unittest
from pathlib import Path

from src.parsers.chunker import DocumentChunker
from src.parsers.models import SectionHeading
from src.parsers.pdf_parser import PDFParser


class TestDocumentChunker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load actual sample AUTOSAR HLD document for realistic testing."""
        cls.sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        if not cls.sample_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {cls.sample_path}")

        cls.parser = PDFParser()
        cls.pages = cls.parser.extract_pages(cls.sample_path)

    def test_invalid_chunk_parameters(self):
        """Raises ValueError for invalid chunk_size and chunk_overlap."""
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=0)

        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=-10)

        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=-5)

        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=120)

    def test_clean_text_normalizations(self):
        """Cleans null bytes, non-breaking spaces, and excess newlines."""
        chunker = DocumentChunker()
        dirty = "AUTOSAR\x00 High-Level\r\nDesign\xa0Specification.\n\n\n\nSection 1."
        cleaned = chunker.clean_text(dirty)

        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("AUTOSAR High-Level\nDesign Specification.\n\nSection 1.", cleaned)

    def test_clean_text_empty(self):
        """Handles empty string safely."""
        chunker = DocumentChunker()
        self.assertEqual(chunker.clean_text(""), "")
        self.assertEqual(chunker.clean_text("   \n\t  "), "")

    def test_chunking_sample_hld_pdf(self):
        """Chunks the actual sample AUTOSAR HLD document and validates chunk properties."""
        chunk_size = 60
        chunk_overlap = 15
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks, stats = chunker.chunk_pages(self.pages, source_file="sample_autosar_hld.pdf")

        # Assert chunks exist
        self.assertGreater(len(chunks), 0)
        self.assertEqual(stats.total_chunks, len(chunks))
        self.assertEqual(stats.chunk_size_words, chunk_size)
        self.assertEqual(stats.chunk_overlap_words, chunk_overlap)
        self.assertGreater(stats.total_words, 100)
        self.assertGreater(stats.avg_chunk_words, 0.0)

        # Assert every chunk satisfies constraints
        for chunk in chunks:
            self.assertEqual(chunk.source_file, "sample_autosar_hld.pdf")
            self.assertIn(chunk.page_number, [1, 2, 3])
            self.assertLessEqual(chunk.word_count, chunk_size)
            self.assertGreater(chunk.word_count, 0)
            self.assertTrue(chunk.chunk_id.startswith(f"chunk_p{chunk.page_number}_"))

    def test_overlap_between_consecutive_chunks(self):
        """Verifies that consecutive chunks on the same page share overlapping words."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        sample_page = [{
            "page_number": 1,
            "text": "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 "
                    "word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 "
                    "word20 word21 word22 word23 word24 word25 word26 word27 word28 word29",
        }]

        chunks, stats = chunker.chunk_pages(sample_page, source_file="overlap_test.pdf")
        self.assertGreaterEqual(len(chunks), 2)

        c1_words = chunks[0].text.split()
        c2_words = chunks[1].text.split()

        # Last 5 words of c1 must match first 5 words of c2
        self.assertEqual(c1_words[-5:], c2_words[:5])

    def test_page_smaller_than_chunk_size(self):
        """A page with fewer words than chunk_size should produce exactly one chunk."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        short_page = [{
            "page_number": 1,
            "text": "Short page with only six words.",
        }]

        chunks, stats = chunker.chunk_pages(short_page, source_file="short.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].word_count, 6)
        self.assertEqual(chunks[0].text, "Short page with only six words.")

    def test_empty_and_missing_pages_detection(self):
        """Handles empty pages and logs them into empty_pages_detected."""
        chunker = DocumentChunker()
        pages_with_empty = [
            {"page_number": 1, "text": "Valid page content."},
            {"page_number": 2, "text": "   \n\x00   "},  # Effectively empty
            {"page_number": 3, "text": "Another valid page."},
        ]

        chunks, stats = chunker.chunk_pages(pages_with_empty, source_file="empty_test.pdf")
        self.assertEqual(len(chunks), 2)
        self.assertIn(2, stats.empty_pages_detected)
        self.assertNotIn(1, stats.empty_pages_detected)
        self.assertNotIn(3, stats.empty_pages_detected)

    def test_section_association_mapping(self):
        """Chunks inherit the active parent section heading."""
        sections = [
            SectionHeading(
                section_number="1",
                title="ARCHITECTURAL OVERVIEW",
                page_number=1,
                source_file="sample.pdf",
                supporting_text="1. ARCHITECTURAL OVERVIEW",
                level=1,
            ),
            SectionHeading(
                section_number="2",
                title="PORT INTERFACES",
                page_number=2,
                source_file="sample.pdf",
                supporting_text="2. PORT INTERFACES",
                level=1,
            ),
        ]

        pages = [
            {"page_number": 1, "text": "Content for section 1."},
            {"page_number": 2, "text": "Content for section 2."},
        ]

        chunker = DocumentChunker()
        chunks, _ = chunker.chunk_pages(pages, source_file="sample.pdf", sections=sections)

        self.assertEqual(chunks[0].section_title, "ARCHITECTURAL OVERVIEW")
        self.assertEqual(chunks[1].section_title, "PORT INTERFACES")


if __name__ == "__main__":
    unittest.main()
