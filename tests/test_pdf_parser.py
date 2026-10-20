"""Unit tests for PDFParser using PyMuPDF."""

import unittest
from pathlib import Path
import pymupdf

from src.parsers.pdf_parser import PDFParser, PDFExtractionError


class TestPDFParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary AUTOSAR HLD test PDF."""
        cls.test_dir = Path(__file__).parent
        cls.sample_pdf_path = cls.test_dir / "test_sample_hld.pdf"

        doc = pymupdf.open()
        
        # Page 1: Title and Architecture Overview
        page1 = doc.new_page()
        page1.insert_text(
            pymupdf.Point(50, 72),
            "AUTOSAR High-Level Design (HLD)\n"
            "Document ID: HLD-SWC-ENG-001\n"
            "Author: Automotive Systems Engineer\n\n"
            "1. Architectural Overview\n"
            "This document details the Software Component (SWC) architecture for Engine Control.\n"
            "It specifies Application SWC SWC_EngineSpeed and Sensor/Actuator SWC SWC_PedalSensor.\n",
            fontsize=11
        )
        
        # Page 2: Interfaces and Runnables
        page2 = doc.new_page()
        page2.insert_text(
            pymupdf.Point(50, 72),
            "2. Port Interfaces and RTE Mapping\n"
            "SWC_EngineSpeed implements P-Port ppEngineSpeed with Sender-Receiver interface IF_EngineSpeed.\n"
            "Runnable RE_CalculateSpeed is triggered by RTE Timing Event (period: 10ms).\n"
            "Safety Requirement: ASIL B functional safety rating.\n",
            fontsize=11
        )

        doc.save(str(cls.sample_pdf_path))
        doc.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up test PDF file."""
        if cls.sample_pdf_path.exists():
            cls.sample_pdf_path.unlink()

    def setUp(self):
        self.parser = PDFParser()

    def test_metadata_extraction(self):
        metadata = self.parser.get_metadata(self.sample_pdf_path)
        self.assertEqual(metadata["page_count"], 2)
        self.assertEqual(metadata["filename"], "test_sample_hld.pdf")
        self.assertFalse(metadata["is_encrypted"])
        self.assertGreater(metadata["file_size_kb"], 0)

    def test_page_extraction(self):
        pages = self.parser.extract_pages(self.sample_pdf_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertIn("SWC_EngineSpeed", pages[0]["text"])
        self.assertIn("RE_CalculateSpeed", pages[1]["text"])

    def test_autosar_keyword_scanning(self):
        overview = self.parser.get_document_overview(self.sample_pdf_path)
        self.assertEqual(overview["total_pages"], 2)
        self.assertTrue(overview["has_text"])
        
        keywords = overview["autosar_keywords_found"]
        self.assertIn("Software Component", keywords)
        self.assertIn("SWC", keywords)
        self.assertIn("Sender-Receiver", keywords)
        self.assertIn("Runnable", keywords)
        self.assertIn("RTE", keywords)
        self.assertIn("ASIL", keywords)

    def test_non_existent_file_raises_error(self):
        with self.assertRaises(PDFExtractionError):
            self.parser.get_metadata("non_existent_file.pdf")


if __name__ == "__main__":
    unittest.main()
