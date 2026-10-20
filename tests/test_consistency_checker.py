"""Unit tests for AUTOSARConsistencyChecker and architectural rule evaluations."""

import unittest
from pathlib import Path

from src.analyzers.consistency_checker import AUTOSARConsistencyChecker
from src.analyzers.models import ConsistencyReport, RuleFinding, Severity
from src.parsers.autosar_extractor import StructuredAUTOSARExtractor
from src.parsers.models import (
    EntityType,
    EvidenceStatus,
    ExtractedEntity,
    ExtractionResult,
)
from src.parsers.pdf_parser import PDFParser


class TestAUTOSARConsistencyCheckerSample(unittest.TestCase):
    """Evaluates consistency rules on the real sample_autosar_hld.pdf document."""

    @classmethod
    def setUpClass(cls):
        sample_path = Path("data/input_docs/sample_autosar_hld.pdf")
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample PDF not found at {sample_path}")

        parser = PDFParser()
        pages = parser.extract_pages(sample_path)
        extractor = StructuredAUTOSARExtractor()
        cls.extraction_result = extractor.extract(pages, source_file=sample_path.name)
        cls.checker = AUTOSARConsistencyChecker()
        cls.report = cls.checker.check_consistency(cls.extraction_result)

    def test_report_structure_and_counts(self):
        """Verifies report structure, total rules evaluated, and severity distribution."""
        self.assertEqual(self.report.source_file, "sample_autosar_hld.pdf")
        self.assertEqual(self.report.total_rules_checked, 5)
        self.assertGreater(len(self.report.findings), 5)

        counts = self.report.get_counts()
        self.assertGreater(counts["PASS"], 0)
        self.assertGreater(counts["WARNING"], 0)
        self.assertGreater(counts["UNVERIFIED"], 0)

    def test_ar_port_01_sample_ports_pass(self):
        """AR-PORT-01: Verifies all ports in sample HLD associate with valid interfaces."""
        port_findings = [f for f in self.report.findings if f.rule_id == "AR-PORT-01"]
        self.assertGreaterEqual(len(port_findings), 3)

        finding_names = {f.target_entity: f for f in port_findings}
        self.assertIn("ppVehicleSpeed", finding_names)
        self.assertIn("rpWheelPulseRaw", finding_names)
        self.assertIn("rpDiagStatus", finding_names)

        # All 3 sample ports reference defined interfaces in the sample PDF
        for f in port_findings:
            self.assertEqual(f.severity, Severity.PASS)
            self.assertIn("correctly", f.message.lower())
            self.assertEqual(f.page_number, 2)

    def test_ar_run_01_sample_runnable_triggers(self):
        """AR-RUN-01: Verifies periodic runnable period and event triggers."""
        run_findings = [f for f in self.report.findings if f.rule_id == "AR-RUN-01"]
        self.assertGreaterEqual(len(run_findings), 3)

        finding_names = {f.target_entity: f for f in run_findings}
        self.assertIn("RE_CalculateSpeed", finding_names)
        self.assertIn("RE_VSM_Init", finding_names)
        self.assertIn("RE_DiagEventUpdate", finding_names)

        # RE_CalculateSpeed has verified 10ms period
        re_calc = finding_names["RE_CalculateSpeed"]
        self.assertEqual(re_calc.severity, Severity.PASS)
        self.assertIn("10ms", re_calc.message)

    def test_ar_run_02_sample_task_mapping(self):
        """AR-RUN-02: Verifies task mapping and distinguishes unverified aperiodic tasks."""
        task_findings = [f for f in self.report.findings if f.rule_id == "AR-RUN-02"]
        finding_names = {f.target_entity: f for f in task_findings}

        # RE_CalculateSpeed is mapped to Task_Periodic_10ms -> PASS
        self.assertEqual(finding_names["RE_CalculateSpeed"].severity, Severity.PASS)
        self.assertIn("Task_Periodic_10ms", finding_names["RE_CalculateSpeed"].message)

        # Aperiodic runnables without task in HLD are marked UNVERIFIED (not ERROR)
        self.assertEqual(finding_names["RE_VSM_Init"].severity, Severity.UNVERIFIED)

    def test_ar_if_01_client_server_timeout_flagged(self):
        """AR-IF-01: Verifies that unchecked timeout for Client-Server interface is flagged as WARNING."""
        cs_findings = [f for f in self.report.findings if f.rule_id == "AR-IF-01"]
        self.assertGreater(len(cs_findings), 0)

        warning_finding = next((f for f in cs_findings if f.target_entity == "IF_DiagStatusService"), None)
        self.assertIsNotNone(warning_finding)
        self.assertEqual(warning_finding.severity, Severity.WARNING)
        self.assertIn("under review", warning_finding.message.lower())
        self.assertEqual(warning_finding.page_number, 3)

    def test_ar_safe_01_sample_safety_and_e2e(self):
        """AR-SAFE-01: Verifies ASIL-B + E2E Profile 2 pass and safety review is flagged."""
        safety_findings = [f for f in self.report.findings if f.rule_id == "AR-SAFE-01"]
        self.assertEqual(len(safety_findings), 1)

        f = safety_findings[0]
        self.assertEqual(f.severity, Severity.PASS)
        self.assertTrue(f.requires_safety_review)
        self.assertIn("ASIL-B", f.message)
        self.assertIn("Profile 2", f.message)


class TestAUTOSARConsistencyCheckerEdgeCases(unittest.TestCase):
    """Deterministic tests for error, warning, unverified, and edge-case synthetic inputs."""

    def setUp(self):
        self.checker = AUTOSARConsistencyChecker()

    def test_port_missing_interface_generates_error(self):
        """A port with no interface specified produces an ERROR."""
        ext_result = ExtractionResult(
            source_file="bad_port.pdf",
            total_pages=1,
            entities=[
                ExtractedEntity(
                    name="ppOrphanPort",
                    entity_type=EntityType.PORT,
                    page_number=1,
                    source_file="bad_port.pdf",
                    supporting_text="1. P-Port: ppOrphanPort",
                    attributes={"port_direction": "Provided (P-Port)", "associated_interface": "Not Specified"},
                )
            ],
        )

        report = self.checker.check_consistency(ext_result)
        findings = report.filter_by_severity(Severity.ERROR)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].rule_id, "AR-PORT-01")
        self.assertEqual(findings[0].target_entity, "ppOrphanPort")

    def test_port_unresolved_interface_generates_unverified(self):
        """A port referencing an interface not detailed in the HLD produces UNVERIFIED."""
        ext_result = ExtractionResult(
            source_file="external_if.pdf",
            total_pages=1,
            entities=[
                ExtractedEntity(
                    name="ppExternalPort",
                    entity_type=EntityType.PORT,
                    page_number=1,
                    source_file="external_if.pdf",
                    supporting_text="1. P-Port: ppExternalPort",
                    attributes={"port_direction": "Provided (P-Port)", "associated_interface": "IF_ExternalPackage"},
                )
            ],
        )

        report = self.checker.check_consistency(ext_result)
        findings = report.filter_by_severity(Severity.UNVERIFIED)
        self.assertTrue(any(f.rule_id == "AR-PORT-01" and f.target_entity == "ppExternalPort" for f in findings))

    def test_runnable_missing_trigger_generates_warning(self):
        """A runnable with no trigger specified produces a WARNING."""
        ext_result = ExtractionResult(
            source_file="bad_runnable.pdf",
            total_pages=1,
            entities=[
                ExtractedEntity(
                    name="RE_Untriggered",
                    entity_type=EntityType.RUNNABLE,
                    page_number=1,
                    source_file="bad_runnable.pdf",
                    supporting_text="Runnable: RE_Untriggered",
                    attributes={"trigger_event": "Not Specified"},
                )
            ],
        )

        report = self.checker.check_consistency(ext_result)
        findings = [f for f in report.findings if f.rule_id == "AR-RUN-01"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, Severity.WARNING)

    def test_periodic_runnable_missing_task_generates_warning(self):
        """A periodic runnable without an OS task generates a WARNING."""
        ext_result = ExtractionResult(
            source_file="unmapped.pdf",
            total_pages=1,
            entities=[
                ExtractedEntity(
                    name="RE_UnmappedPeriodic",
                    entity_type=EntityType.RUNNABLE,
                    page_number=1,
                    source_file="unmapped.pdf",
                    supporting_text="Runnable: RE_UnmappedPeriodic",
                    attributes={"trigger_event": "Timing Event (20ms)", "mapped_os_task": "Not Specified"},
                )
            ],
        )

        report = self.checker.check_consistency(ext_result)
        task_findings = [f for f in report.findings if f.rule_id == "AR-RUN-02"]
        self.assertEqual(len(task_findings), 1)
        self.assertEqual(task_findings[0].severity, Severity.WARNING)

    def test_client_server_without_operations_generates_error(self):
        """Client-Server interface with no operations generates an ERROR."""
        ext_result = ExtractionResult(
            source_file="bad_cs.pdf",
            total_pages=1,
            entities=[
                ExtractedEntity(
                    name="IF_EmptyCS",
                    entity_type=EntityType.INTERFACE,
                    page_number=1,
                    source_file="bad_cs.pdf",
                    supporting_text="Client-Server Interface (IF_EmptyCS)",
                    attributes={"interface_type": "Client-Server", "members": "Not Specified"},
                )
            ],
        )

        report = self.checker.check_consistency(ext_result)
        cs_findings = [f for f in report.findings if f.rule_id == "AR-IF-01"]
        self.assertEqual(len(cs_findings), 1)
        self.assertEqual(cs_findings[0].severity, Severity.ERROR)

    def test_asil_without_e2e_generates_warning_and_safety_flag(self):
        """ASIL safety rating declared without E2E protection generates a WARNING and requires safety review."""
        ext_result = ExtractionResult(
            source_file="unsafe.pdf",
            total_pages=1,
            entities=[
                ExtractedEntity(
                    name="FunctionalSafety_ASIL-C",
                    entity_type=EntityType.DEPENDENCY,
                    page_number=1,
                    source_file="unsafe.pdf",
                    supporting_text="ASIL Rating: ASIL-C",
                    attributes={"relation_type": "Safety Requirement", "safety_level": "ASIL-C"},
                )
            ],
        )

        report = self.checker.check_consistency(ext_result)
        safety_findings = [f for f in report.findings if f.rule_id == "AR-SAFE-01"]
        self.assertEqual(len(safety_findings), 1)
        self.assertEqual(safety_findings[0].severity, Severity.WARNING)
        self.assertTrue(safety_findings[0].requires_safety_review)

    def test_empty_extraction_result_safe_handling(self):
        """Handles empty extraction result gracefully with UNVERIFIED status, not errors."""
        ext_result = ExtractionResult(source_file="empty.pdf", total_pages=0, entities=[])
        report = self.checker.check_consistency(ext_result)

        self.assertEqual(report.total_rules_checked, 5)
        # Empty inputs should produce UNVERIFIED informational findings, zero crashes
        counts = report.get_counts()
        self.assertEqual(counts["ERROR"], 0)
        self.assertGreater(counts["UNVERIFIED"], 0)


if __name__ == "__main__":
    unittest.main()
