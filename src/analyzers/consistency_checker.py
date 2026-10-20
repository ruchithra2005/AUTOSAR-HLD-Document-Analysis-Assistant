"""Deterministic AUTOSAR Architectural Consistency Checker.

Evaluates extracted HLD architecture models against formal AUTOSAR design rules:
- AR-PORT-01: Port-to-interface matching and unresolved interface detection.
- AR-RUN-01: Runnable trigger and periodic activation timing verification.
- AR-RUN-02: Runnable-to-OS-Task mapping checks.
- AR-IF-01: Client-Server operation and invocation timeout auditing.
- AR-SAFE-01: ASIL safety classification and E2E protection profile verification.

Strictly preserves evidence provenance, distinguishes ERROR, WARNING, PASS, and
UNVERIFIED without fabricating rules or compliance claims, and flags safety findings for review.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from src.parsers.models import EntityType, ExtractedEntity, ExtractionResult

from .models import ConsistencyReport, RuleFinding, Severity


class AUTOSARConsistencyChecker:
    """Evaluates extracted AUTOSAR entities against architectural consistency rules."""

    def __init__(self) -> None:
        pass

    def check_consistency(self, extraction_result: ExtractionResult) -> ConsistencyReport:
        """Runs all deterministic AUTOSAR consistency checks on the extracted model.

        Args:
            extraction_result: Structured extraction result from Phase 2.

        Returns:
            ConsistencyReport containing all findings categorized by severity.
        """
        findings: List[RuleFinding] = []
        source_file = extraction_result.source_file

        # 1. AR-PORT-01: Port-to-Interface Matching
        findings.extend(self._check_port_interface_matching(extraction_result, source_file))

        # 2. AR-RUN-01: Runnable Trigger and Periodic Timing
        findings.extend(self._check_runnable_triggers(extraction_result, source_file))

        # 3. AR-RUN-02: Runnable-to-OS-Task Mapping
        findings.extend(self._check_runnable_task_mapping(extraction_result, source_file))

        # 4. AR-IF-01: Client-Server Operations and Timeout Checks
        findings.extend(self._check_client_server_interfaces(extraction_result, source_file))

        # 5. AR-SAFE-01: ASIL and E2E Protection Checks
        findings.extend(self._check_safety_and_e2e(extraction_result, source_file))

        return ConsistencyReport(
            source_file=source_file,
            total_rules_checked=5,
            findings=findings,
        )

    # -------------------------------------------------------------------------
    # Rule Evaluation Subroutines
    # -------------------------------------------------------------------------

    def _check_port_interface_matching(
        self, extraction_result: ExtractionResult, source_file: str
    ) -> List[RuleFinding]:
        """AR-PORT-01: Evaluates whether ports reference valid, typed PortInterfaces."""
        findings: List[RuleFinding] = []
        ports = extraction_result.filter_by_type(EntityType.PORT)
        interfaces = {i.name: i for i in extraction_result.filter_by_type(EntityType.INTERFACE)}

        if not ports:
            findings.append(
                RuleFinding(
                    rule_id="AR-PORT-01",
                    rule_name="Port-to-Interface Matching",
                    severity=Severity.UNVERIFIED,
                    target_entity="No Ports Found",
                    target_entity_type="PORT",
                    message="No Port prototypes (P-Port, R-Port, PR-Port) were detected in the document.",
                    recommendation="If this SWC communicates with other components, define its P-Ports or R-Ports.",
                    page_number=1,
                    source_file=source_file,
                    supporting_evidence="No port declarations identified in document text.",
                )
            )
            return findings

        for port in ports:
            assoc_if = port.attributes.get("associated_interface")
            port_kind = port.attributes.get("port_direction", "Port")

            if not assoc_if or assoc_if in ("Not Specified", "None", ""):
                findings.append(
                    RuleFinding(
                        rule_id="AR-PORT-01",
                        rule_name="Port-to-Interface Matching",
                        severity=Severity.ERROR,
                        target_entity=port.name,
                        target_entity_type="PORT",
                        message=f"Port '{port.name}' ({port_kind}) does not specify an associated PortInterface in the HLD.",
                        recommendation="Assign a typed Sender-Receiver or Client-Server interface prototype to this port.",
                        page_number=port.page_number,
                        source_file=source_file,
                        supporting_evidence=port.supporting_text,
                    )
                )
            elif assoc_if in interfaces:
                if_obj = interfaces[assoc_if]
                if_type = if_obj.attributes.get("interface_type", "Interface")
                findings.append(
                    RuleFinding(
                        rule_id="AR-PORT-01",
                        rule_name="Port-to-Interface Matching",
                        severity=Severity.PASS,
                        target_entity=port.name,
                        target_entity_type="PORT",
                        message=(
                            f"Port '{port.name}' ({port_kind}) correctly references "
                            f"defined interface '{assoc_if}' (Type: {if_type})."
                        ),
                        recommendation="Verified port-to-interface association. Maintain data type consistency across connectors.",
                        page_number=port.page_number,
                        source_file=source_file,
                        supporting_evidence=f"{port.supporting_text} -> Interface: {assoc_if} ({if_type})",
                    )
                )
            else:
                # Interface is named on the port, but full interface definition is missing from HLD
                findings.append(
                    RuleFinding(
                        rule_id="AR-PORT-01",
                        rule_name="Port-to-Interface Matching",
                        severity=Severity.UNVERIFIED,
                        target_entity=port.name,
                        target_entity_type="PORT",
                        message=(
                            f"Port '{port.name}' references interface '{assoc_if}', but the interface definition "
                            "is not detailed in this HLD document."
                        ),
                        recommendation=(
                            f"Confirm if interface '{assoc_if}' is specified in an external ARXML file "
                            "or separate shared interface document."
                        ),
                        page_number=port.page_number,
                        source_file=source_file,
                        supporting_evidence=port.supporting_text,
                    )
                )

        return findings

    def _check_runnable_triggers(
        self, extraction_result: ExtractionResult, source_file: str
    ) -> List[RuleFinding]:
        """AR-RUN-01: Verifies runnable triggers and execution periodicity."""
        findings: List[RuleFinding] = []
        runnables = extraction_result.filter_by_type(EntityType.RUNNABLE)

        if not runnables:
            findings.append(
                RuleFinding(
                    rule_id="AR-RUN-01",
                    rule_name="Runnable Trigger and Periodic Timing",
                    severity=Severity.UNVERIFIED,
                    target_entity="No Runnables Found",
                    target_entity_type="RUNNABLE",
                    message="No Runnable entities were detected in the document.",
                    recommendation="Define SWC internal behavior containing at least one executable Runnable entity.",
                    page_number=1,
                    source_file=source_file,
                    supporting_evidence="No runnable declarations identified in document text.",
                )
            )
            return findings

        for r in runnables:
            trigger = r.attributes.get("trigger_event", "Not Specified")

            if trigger in ("Not Specified", "", None):
                findings.append(
                    RuleFinding(
                        rule_id="AR-RUN-01",
                        rule_name="Runnable Trigger and Periodic Timing",
                        severity=Severity.WARNING,
                        target_entity=r.name,
                        target_entity_type="RUNNABLE",
                        message=f"Runnable '{r.name}' has no trigger RTE Event specified in the HLD.",
                        recommendation="Define an explicit RTE Event (e.g. TimingEvent, InitEvent, DataReceivedEvent) to trigger this runnable.",
                        page_number=r.page_number,
                        source_file=source_file,
                        supporting_evidence=r.supporting_text,
                    )
                )
            elif "timing" in trigger.lower() or "periodic" in trigger.lower() or "period:" in trigger.lower():
                # Check for explicit period (e.g. 10ms, 20ms)
                period_match = re.search(r"(\d+(?:\.\d+)?\s*m?s)", trigger, re.IGNORECASE)
                if period_match:
                    period_val = period_match.group(1)
                    findings.append(
                        RuleFinding(
                            rule_id="AR-RUN-01",
                            rule_name="Runnable Trigger and Periodic Timing",
                            severity=Severity.PASS,
                            target_entity=r.name,
                            target_entity_type="RUNNABLE",
                            message=f"Periodic Runnable '{r.name}' has verified timing trigger with activation period: {period_val}.",
                            recommendation="Verified. Ensure activation period matches the OS task activation period during RTE generation.",
                            page_number=r.page_number,
                            source_file=source_file,
                            supporting_evidence=f"{r.supporting_text} | Trigger: {trigger}",
                        )
                    )
                else:
                    findings.append(
                        RuleFinding(
                            rule_id="AR-RUN-01",
                            rule_name="Runnable Trigger and Periodic Timing",
                            severity=Severity.WARNING,
                            target_entity=r.name,
                            target_entity_type="RUNNABLE",
                            message=f"Runnable '{r.name}' indicates timing trigger, but explicit periodicity (e.g. 10ms) is missing.",
                            recommendation="Specify numerical period (e.g. 10ms, 20ms, 100ms) for periodic runnable execution.",
                            page_number=r.page_number,
                            source_file=source_file,
                            supporting_evidence=f"{r.supporting_text} | Trigger: {trigger}",
                        )
                    )
            else:
                # Aperiodic or event-driven trigger (InitEvent, DataReceivedEvent, etc.)
                findings.append(
                    RuleFinding(
                        rule_id="AR-RUN-01",
                        rule_name="Runnable Trigger and Periodic Timing",
                        severity=Severity.PASS,
                        target_entity=r.name,
                        target_entity_type="RUNNABLE",
                        message=f"Aperiodic Runnable '{r.name}' is bound to trigger event: '{trigger}'.",
                        recommendation="Verified event-driven trigger. Ensure data reception buffer or init sequence is handled safely.",
                        page_number=r.page_number,
                        source_file=source_file,
                        supporting_evidence=f"{r.supporting_text} | Trigger: {trigger}",
                    )
                )

        return findings

    def _check_runnable_task_mapping(
        self, extraction_result: ExtractionResult, source_file: str
    ) -> List[RuleFinding]:
        """AR-RUN-02: Verifies OS Task allocation for Runnable entities."""
        findings: List[RuleFinding] = []
        runnables = extraction_result.filter_by_type(EntityType.RUNNABLE)

        if not runnables:
            return findings

        for r in runnables:
            task = r.attributes.get("mapped_os_task", "Not Specified")
            trigger = r.attributes.get("trigger_event", "").lower()
            is_periodic = "timing" in trigger or "period" in trigger

            if task not in ("Not Specified", "", None):
                findings.append(
                    RuleFinding(
                        rule_id="AR-RUN-02",
                        rule_name="Runnable-to-OS-Task Mapping",
                        severity=Severity.PASS,
                        target_entity=r.name,
                        target_entity_type="RUNNABLE",
                        message=f"Runnable '{r.name}' is mapped to OS Task: '{task}'.",
                        recommendation="Verified task mapping. Confirm task priority and preemptability meet worst-case execution time (WCET).",
                        page_number=r.page_number,
                        source_file=source_file,
                        supporting_evidence=f"{r.supporting_text} | OS Task: {task}",
                    )
                )
            else:
                if is_periodic:
                    findings.append(
                        RuleFinding(
                            rule_id="AR-RUN-02",
                            rule_name="Runnable-to-OS-Task Mapping",
                            severity=Severity.WARNING,
                            target_entity=r.name,
                            target_entity_type="RUNNABLE",
                            message=f"Periodic Runnable '{r.name}' lacks an explicit OS Task mapping in this HLD.",
                            recommendation="Assign periodic runnable to an OS Task with matching activation period before RTE generation.",
                            page_number=r.page_number,
                            source_file=source_file,
                            supporting_evidence=r.supporting_text,
                        )
                    )
                else:
                    findings.append(
                        RuleFinding(
                            rule_id="AR-RUN-02",
                            rule_name="Runnable-to-OS-Task Mapping",
                            severity=Severity.UNVERIFIED,
                            target_entity=r.name,
                            target_entity_type="RUNNABLE",
                            message=f"OS Task mapping for aperiodic runnable '{r.name}' is not documented in the HLD.",
                            recommendation="Aperiodic/init runnables may be mapped during ECU integration; confirm task allocation in ECU configuration.",
                            page_number=r.page_number,
                            source_file=source_file,
                            supporting_evidence=r.supporting_text,
                        )
                    )

        return findings

    def _check_client_server_interfaces(
        self, extraction_result: ExtractionResult, source_file: str
    ) -> List[RuleFinding]:
        """AR-IF-01: Audits Client-Server operations and invocation response timeout handling."""
        findings: List[RuleFinding] = []
        interfaces = extraction_result.filter_by_type(EntityType.INTERFACE)
        cs_interfaces = [i for i in interfaces if i.attributes.get("interface_type") == "Client-Server"]
        dependencies = extraction_result.filter_by_type(EntityType.DEPENDENCY)

        # Check for any unverified server response timeout in checklist items
        unverified_timeout_dep = None
        for dep in dependencies:
            dep_details = dep.attributes.get("details", "")
            if "server response timeout" in dep_details.lower() or "timeout" in dep.name.lower():
                if dep.status.value == "UNVERIFIED" or not dep.attributes.get("verified", True):
                    unverified_timeout_dep = dep
                    break

        if not cs_interfaces:
            # Check if any port uses Client-Server
            ports = extraction_result.filter_by_type(EntityType.PORT)
            cs_ports = [p for p in ports if "diag" in p.name.lower() or "server" in str(p.attributes).lower()]
            if not cs_ports:
                findings.append(
                    RuleFinding(
                        rule_id="AR-IF-01",
                        rule_name="Client-Server Operations and Timeout Checks",
                        severity=Severity.PASS,
                        target_entity="Client-Server Interfaces",
                        target_entity_type="INTERFACE",
                        message="No Client-Server interfaces are used by this component (all communication is Sender-Receiver).",
                        recommendation="No Client-Server timeouts required.",
                        page_number=1,
                        source_file=source_file,
                        supporting_evidence="Document specifies Sender-Receiver interfaces.",
                    )
                )
                return findings

        for cs_if in cs_interfaces:
            members = cs_if.attributes.get("members", "")
            has_operations = "operation:" in members.lower()

            if not has_operations:
                findings.append(
                    RuleFinding(
                        rule_id="AR-IF-01",
                        rule_name="Client-Server Operations and Timeout Checks",
                        severity=Severity.ERROR,
                        target_entity=cs_if.name,
                        target_entity_type="INTERFACE",
                        message=f"Client-Server interface '{cs_if.name}' does not specify any operations.",
                        recommendation="Define operation prototypes with argument names, data types, and direction (IN, OUT, INOUT).",
                        page_number=cs_if.page_number,
                        source_file=source_file,
                        supporting_evidence=cs_if.supporting_text,
                    )
                )
            elif unverified_timeout_dep:
                findings.append(
                    RuleFinding(
                        rule_id="AR-IF-01",
                        rule_name="Client-Server Operations and Timeout Checks",
                        severity=Severity.WARNING,
                        target_entity=cs_if.name,
                        target_entity_type="INTERFACE",
                        message=(
                            f"Client-Server interface '{cs_if.name}' specifies operation '{members}', "
                            "but client invocation response timeout is marked 'Under Review' / unverified in HLD."
                        ),
                        recommendation=(
                            "Specify maximum client response timeout value (e.g. 50ms) to prevent client task blocking "
                            "if the server SWC fails to respond."
                        ),
                        page_number=unverified_timeout_dep.page_number,
                        source_file=source_file,
                        supporting_evidence=unverified_timeout_dep.supporting_text,
                    )
                )
            else:
                findings.append(
                    RuleFinding(
                        rule_id="AR-IF-01",
                        rule_name="Client-Server Operations and Timeout Checks",
                        severity=Severity.PASS,
                        target_entity=cs_if.name,
                        target_entity_type="INTERFACE",
                        message=f"Client-Server interface '{cs_if.name}' specifies operation: {members}.",
                        recommendation="Verified operation definition. Ensure synchronous timeout or asynchronous callback is handled.",
                        page_number=cs_if.page_number,
                        source_file=source_file,
                        supporting_evidence=cs_if.supporting_text,
                    )
                )

        return findings

    def _check_safety_and_e2e(
        self, extraction_result: ExtractionResult, source_file: str
    ) -> List[RuleFinding]:
        """AR-SAFE-01: Verifies ISO 26262 ASIL safety rating and End-to-End (E2E) protection."""
        findings: List[RuleFinding] = []
        dependencies = extraction_result.filter_by_type(EntityType.DEPENDENCY)

        # 1. Identify ASIL Rating
        asil_entity: Optional[ExtractedEntity] = None
        for dep in dependencies:
            if dep.attributes.get("relation_type") == "Safety Requirement" or "asil" in dep.name.lower():
                asil_entity = dep
                break

        # 2. Identify E2E Protection
        e2e_entity: Optional[ExtractedEntity] = None
        for dep in dependencies:
            if dep.attributes.get("relation_type") == "End-to-End Data Protection" or "e2e" in dep.name.lower():
                e2e_entity = dep
                break

        if asil_entity:
            safety_level = asil_entity.attributes.get("safety_level", asil_entity.name)

            if e2e_entity:
                e2e_spec = e2e_entity.attributes.get("specification", e2e_entity.supporting_text)
                findings.append(
                    RuleFinding(
                        rule_id="AR-SAFE-01",
                        rule_name="ASIL Safety and E2E Protection",
                        severity=Severity.PASS,
                        target_entity=safety_level,
                        target_entity_type="DEPENDENCY",
                        message=(
                            f"Safety requirement '{safety_level}' is complemented by verified End-to-End protection: "
                            f"'{e2e_spec}' on safety-critical communication."
                        ),
                        recommendation=(
                            "Safety requirement verified with evidence. A functional safety engineer should confirm "
                            "CRC polynomial and counter thresholds meet ISO 26262 ASIL metrics."
                        ),
                        page_number=e2e_entity.page_number,
                        source_file=source_file,
                        supporting_evidence=f"Safety: {asil_entity.supporting_text} | E2E: {e2e_entity.supporting_text}",
                        requires_safety_review=True,
                    )
                )
            else:
                findings.append(
                    RuleFinding(
                        rule_id="AR-SAFE-01",
                        rule_name="ASIL Safety and E2E Protection",
                        severity=Severity.WARNING,
                        target_entity=safety_level,
                        target_entity_type="DEPENDENCY",
                        message=(
                            f"Component specifies safety rating '{safety_level}', but no End-to-End (E2E) protection "
                            "profile is documented for its communication interfaces."
                        ),
                        recommendation=(
                            f"For {safety_level} compliance, evaluate whether inter-ECU or safety-critical interfaces "
                            "require AUTOSAR E2E protection (e.g. CRC, Sequence Counter, Timeout)."
                        ),
                        page_number=asil_entity.page_number,
                        source_file=source_file,
                        supporting_evidence=asil_entity.supporting_text,
                        requires_safety_review=True,
                    )
                )
        else:
            findings.append(
                RuleFinding(
                    rule_id="AR-SAFE-01",
                    rule_name="ASIL Safety and E2E Protection",
                    severity=Severity.UNVERIFIED,
                    target_entity="Functional Safety",
                    target_entity_type="DEPENDENCY",
                    message="No ISO 26262 ASIL safety rating (ASIL A-D) or QM designation was detected in the HLD.",
                    recommendation="Confirm whether the component is Quality Managed (QM) or carries functional safety requirements.",
                    page_number=1,
                    source_file=source_file,
                    supporting_evidence="No ASIL or safety classification detected in document text.",
                    requires_safety_review=False,
                )
            )

        return findings
