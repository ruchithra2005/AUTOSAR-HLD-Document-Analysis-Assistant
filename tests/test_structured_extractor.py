"""Unit tests for Phase 2 Structured AUTOSAR Extractor using the sample document."""

import json
import unittest
from pathlib import Path

from src.parsers.autosar_extractor import StructuredAUTOSARExtractor
from src.parsers.models import EntityType, EvidenceStatus, ExtractionResult
from src.parsers.pdf_parser import PDFParser
from src.parsers.section_detector import SectionDetector


class TestStructuredAUTOSARExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Locate sample document and parse its pages."""
        cls.sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        if not cls.sample_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {cls.sample_path}")

        cls.pdf_parser = PDFParser()
        cls.pages = cls.pdf_parser.extract_pages(cls.sample_path)
        cls.extractor = StructuredAUTOSARExtractor()

    def test_section_heading_detection(self):
        """Verify major numbered sections are detected with provenance."""
        sections = self.extractor.section_detector.detect_sections(
            self.pages, source_file=self.sample_path.name
        )
        self.assertGreaterEqual(len(sections), 4)

        sec_numbers = [s.section_number for s in sections]
        self.assertIn("1", sec_numbers)
        self.assertIn("2", sec_numbers)
        self.assertIn("3", sec_numbers)
        self.assertIn("4", sec_numbers)

        # Check section provenance
        for s in sections:
            self.assertEqual(s.source_file, self.sample_path.name)
            self.assertIn(s.page_number, [1, 2, 3])
            self.assertTrue(len(s.supporting_text) > 0)

    def test_swc_extraction(self):
        """Verify Software Components are extracted with correct roles and types."""
        result: ExtractionResult = self.extractor.extract(
            self.pages, source_file=self.sample_path.name
        )
        swcs = result.filter_by_type(EntityType.SWC)
        self.assertGreaterEqual(len(swcs), 1)

        swc_names = {s.name: s for s in swcs}
        self.assertIn("VehicleSpeedManagement_SWC", swc_names)

        target_swc = swc_names["VehicleSpeedManagement_SWC"]
        self.assertEqual(target_swc.status, EvidenceStatus.DETECTED)
        self.assertEqual(target_swc.attributes.get("component_type"), "Application SWC")
        self.assertEqual(target_swc.page_number, 1)
        self.assertIn("VehicleSpeedManagement_SWC", target_swc.supporting_text)

    def test_port_extraction(self):
        """Verify Ports are extracted with directions and interface mappings."""
        result = self.extractor.extract(self.pages, source_file=self.sample_path.name)
        ports = result.filter_by_type(EntityType.PORT)
        port_names = {p.name: p for p in ports}

        self.assertIn("ppVehicleSpeed", port_names)
        self.assertIn("rpWheelPulseRaw", port_names)
        self.assertIn("rpDiagStatus", port_names)

        # Check P-Port
        p_port = port_names["ppVehicleSpeed"]
        self.assertEqual(p_port.attributes.get("port_direction"), "Provided (P-Port)")
        self.assertEqual(p_port.attributes.get("associated_interface"), "IF_VehicleSpeed")
        self.assertEqual(p_port.page_number, 2)

        # Check R-Port
        r_port = port_names["rpWheelPulseRaw"]
        self.assertEqual(r_port.attributes.get("port_direction"), "Required (R-Port)")
        self.assertEqual(r_port.attributes.get("associated_interface"), "IF_WheelPulseData")

    def test_interface_extraction(self):
        """Verify Port Interfaces (Sender-Receiver, Client-Server) are extracted."""
        result = self.extractor.extract(self.pages, source_file=self.sample_path.name)
        interfaces = result.filter_by_type(EntityType.INTERFACE)
        if_names = {i.name: i for i in interfaces}

        self.assertIn("IF_VehicleSpeed", if_names)
        self.assertIn("IF_WheelPulseData", if_names)
        self.assertIn("IF_DiagStatusService", if_names)

        self.assertEqual(if_names["IF_VehicleSpeed"].attributes.get("interface_type"), "Sender-Receiver")
        self.assertEqual(if_names["IF_DiagStatusService"].attributes.get("interface_type"), "Client-Server")

    def test_runnable_and_event_extraction(self):
        """Verify Runnables, RTE event triggers, and OS tasks are extracted."""
        result = self.extractor.extract(self.pages, source_file=self.sample_path.name)
        runnables = result.filter_by_type(EntityType.RUNNABLE)
        runnable_names = {r.name: r for r in runnables}

        self.assertIn("RE_CalculateSpeed", runnable_names)
        self.assertIn("RE_VSM_Init", runnable_names)
        self.assertIn("RE_DiagEventUpdate", runnable_names)

        re_calc = runnable_names["RE_CalculateSpeed"]
        self.assertIn("Timing Event", re_calc.attributes.get("trigger_event", ""))
        self.assertIn("Task_Periodic_10ms", re_calc.attributes.get("mapped_os_task", ""))

        # Check corresponding RTE Events
        events = result.filter_by_type(EntityType.RTE_EVENT)
        self.assertGreaterEqual(len(events), 2)

    def test_bsw_module_extraction(self):
        """Verify BSW modules are cataloged."""
        result = self.extractor.extract(self.pages, source_file=self.sample_path.name)
        bsw_modules = result.filter_by_type(EntityType.BSW_MODULE)
        bsw_names = {b.name for b in bsw_modules}

        self.assertTrue({"Dem", "NvM", "ComStack"}.issubset(bsw_names))

    def test_evidence_status_distinction(self):
        """Verify clear separation between DETECTED, INFERRED, and UNVERIFIED."""
        result = self.extractor.extract(self.pages, source_file=self.sample_path.name)

        detected_entities = result.filter_by_status(EvidenceStatus.DETECTED)
        unverified_entities = result.filter_by_status(EvidenceStatus.UNVERIFIED)

        self.assertGreater(len(detected_entities), 0)
        # Unchecked checklist item should be UNVERIFIED
        self.assertGreaterEqual(len(unverified_entities), 1)

    def test_json_serialization(self):
        """Verify complete extraction result can be converted to valid JSON."""
        result = self.extractor.extract(self.pages, source_file=self.sample_path.name)
        json_str = result.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["source_file"], self.sample_path.name)
        self.assertIn("summary_counts", data)
        self.assertIn("sections", data)
        self.assertIn("entities", data)

    def test_safe_handling_of_empty_pages(self):
        """Extractor should handle empty page list gracefully."""
        result = self.extractor.extract([], source_file="empty.pdf")
        self.assertEqual(len(result.sections), 0)
        self.assertEqual(len(result.entities), 0)
        self.assertEqual(result.total_pages, 0)


if __name__ == "__main__":
    unittest.main()
