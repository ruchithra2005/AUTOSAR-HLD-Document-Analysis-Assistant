"""Structured AUTOSAR Entity Extractor.

Extracts candidate Software Components (SWCs), Ports, Interfaces, Runnables,
RTE Events, BSW Modules, and Dependencies from document pages with strict
evidence provenance tracking.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    EntityType,
    EvidenceStatus,
    ExtractedEntity,
    ExtractionResult,
    SectionHeading,
)
from .section_detector import SectionDetector


class StructuredAUTOSARExtractor:
    """Extracts structured AUTOSAR architecture elements from document pages."""

    # Standard AUTOSAR BSW modules catalog with domain categorization
    BSW_MODULE_CATALOG: Dict[str, str] = {
        "ComStack": "Communication Stack",
        "Com": "Communication Service",
        "CanIf": "CAN Interface",
        "CanTp": "CAN Transport Layer",
        "PduR": "PDU Router",
        "IpduM": "IPDU Multiplexer",
        "Dem": "Diagnostic Event Manager",
        "Dcm": "Diagnostic Communication Manager",
        "Det": "Default Error Tracer",
        "NvM": "Non-Volatile Memory Manager",
        "MemIf": "Memory Abstraction Interface",
        "Fee": "Flash EEPROM Emulation",
        "Fls": "Flash Driver",
        "EcuM": "ECU State Manager",
        "BswM": "Basic Software Mode Manager",
        "WdgM": "Watchdog Manager",
        "Os": "AUTOSAR Operating System",
        "OS": "AUTOSAR Operating System",
        "MCAL": "Microcontroller Abstraction Layer",
        "Can": "CAN Driver",
        "Lin": "LIN Driver",
        "Fr": "FlexRay Driver",
        "Eth": "Ethernet Driver",
        "Spi": "SPI Driver",
        "Adc": "ADC Driver",
        "Dio": "DIO Driver",
        "Port": "Port Driver",
        "Crc": "CRC Library",
    }

    def __init__(self) -> None:
        self.section_detector = SectionDetector()

    def extract(self, pages: List[Dict[str, Any]], source_file: str = "document.pdf") -> ExtractionResult:
        """Main extraction pipeline: detects sections and all AUTOSAR architectural entities."""
        # 1. Detect section structure
        sections: List[SectionHeading] = self.section_detector.detect_sections(pages, source_file=source_file)

        entities: List[ExtractedEntity] = []

        # 2. Iterate through pages and extract domain entities
        for page in pages:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")
            if not page_text.strip():
                continue

            # A. Extract Software Components (SWCs)
            entities.extend(self._extract_swcs(page_text, page_num, source_file))

            # B. Extract Ports
            entities.extend(self._extract_ports(page_text, page_num, source_file))

            # C. Extract Interfaces
            entities.extend(self._extract_interfaces(page_text, page_num, source_file))

            # D. Extract Runnables & RTE Events
            entities.extend(self._extract_runnables_and_events(page_text, page_num, source_file))

            # E. Extract BSW Modules & Service Dependencies
            entities.extend(self._extract_bsw_modules(page_text, page_num, source_file))

            # F. Extract Explicit Dependencies & Safety Requirements
            entities.extend(self._extract_dependencies_and_safety(page_text, page_num, source_file))

        # Deduplicate entities while preserving the most complete record
        deduped_entities = self._deduplicate_entities(entities)

        return ExtractionResult(
            source_file=source_file,
            total_pages=len(pages),
            sections=sections,
            entities=deduped_entities,
            extraction_timestamp=datetime.datetime.now().isoformat(),
        )

    # -------------------------------------------------------------------------
    # Entity Extraction Subroutines
    # -------------------------------------------------------------------------

    def _extract_swcs(self, text: str, page_num: int, source_file: str) -> List[ExtractedEntity]:
        """Extracts Software Components declared in the text."""
        swcs: List[ExtractedEntity] = []
        lines = text.splitlines()

        for line in lines:
            line_str = line.strip()

            # Pattern 1: Explicit declaration, e.g., "Component: VehicleSpeedManagement_SWC (Application SWC)"
            m1 = re.search(
                r"(?:Component|SWC)\s*:\s*(?P<name>[A-Za-z0-9_]+)(?:\s*\((?P<type>[^)]+)\))?",
                line_str,
                re.IGNORECASE,
            )
            if m1:
                swc_name = m1.group("name")
                swc_type = m1.group("type") or "Not Explicitly Specified"
                status = EvidenceStatus.DETECTED
                swcs.append(
                    ExtractedEntity(
                        name=swc_name,
                        entity_type=EntityType.SWC,
                        page_number=page_num,
                        source_file=source_file,
                        supporting_text=line_str,
                        status=status,
                        attributes={
                            "component_type": swc_type.strip(),
                            "declaration_role": "Primary Target Component",
                        },
                    )
                )

            # Pattern 2: Connected SWCs list, e.g., "Connected SWCs: WheelSpeedSensor_SWC, InstrumentCluster_SWC"
            m2 = re.search(r"Connected\s+SWCs?\s*:\s*(?P<list>[^\n]+)", line_str, re.IGNORECASE)
            if m2:
                items = [x.strip() for x in m2.group("list").split(",") if x.strip()]
                for item in items:
                    clean_name = re.sub(r"[^A-Za-z0-9_]", "", item)
                    if clean_name:
                        swcs.append(
                            ExtractedEntity(
                                name=clean_name,
                                entity_type=EntityType.SWC,
                                page_number=page_num,
                                source_file=source_file,
                                supporting_text=line_str,
                                status=EvidenceStatus.DETECTED,
                                attributes={
                                    "component_type": "Not Explicitly Specified",
                                    "declaration_role": "Connected SWC",
                                },
                            )
                        )

            # Pattern 3: Standalone SWC identifier naming pattern (e.g. SWC_EngineSpeed) if not already caught
            m3 = re.findall(r"\b(SWC_[A-Za-z0-9_]+|[A-Za-z0-9_]+_SWC)\b", line_str)
            for raw_name in m3:
                # Only add if it's not a generic word like "Connected_SWC"
                if raw_name.upper() not in {"SWC", "CONNECTED_SWC", "APPLICATION_SWC"}:
                    swcs.append(
                        ExtractedEntity(
                            name=raw_name,
                            entity_type=EntityType.SWC,
                            page_number=page_num,
                            source_file=source_file,
                            supporting_text=line_str,
                            status=EvidenceStatus.INFERRED,
                            attributes={
                                "component_type": "Inferred from Naming Convention",
                                "declaration_role": "Referenced Component",
                            },
                        )
                    )

        return swcs

    def _extract_ports(self, text: str, page_num: int, source_file: str) -> List[ExtractedEntity]:
        """Extracts P-Ports, R-Ports, and PR-Ports."""
        ports: List[ExtractedEntity] = []

        # Find port blocks (e.g., "1. P-Port: ppVehicleSpeed" or "R-Port: rpWheelPulseRaw")
        port_pattern = re.compile(
            r"(?P<kind>P-Port|R-Port|PR-Port)\s*:\s*(?P<name>[A-Za-z0-9_]+)",
            re.IGNORECASE,
        )

        for match in port_pattern.finditer(text):
            kind_str = match.group("kind").upper()
            port_name = match.group("name")

            # Extract surrounding context (up to 300 chars following the port declaration)
            start_pos = match.start()
            context_window = text[start_pos : start_pos + 350]

            # Detect associated interface
            if_match = re.search(
                r"Interface(?:\s+Type)?\s*:\s*(?:[^(\n]+\((?P<if_in_paren>[A-Za-z0-9_]+)\)|(?P<if_direct>[A-Za-z0-9_]+))",
                context_window,
                re.IGNORECASE,
            )
            associated_if = "Not Specified"
            if_status = EvidenceStatus.UNVERIFIED
            if if_match:
                associated_if = if_match.group("if_in_paren") or if_match.group("if_direct")
                if_status = EvidenceStatus.DETECTED

            # Detect data elements or operations
            elem_match = re.search(r"Data\s+Elements?\s*:\s*(?P<elem>[^\n]+)", context_window, re.IGNORECASE)
            op_match = re.search(r"Operation\s*:\s*(?P<op>[^\n]+)", context_window, re.IGNORECASE)

            payload_info = "Not Specified"
            if elem_match:
                payload_info = f"Data: {elem_match.group('elem').strip()}"
            elif op_match:
                payload_info = f"Operation: {op_match.group('op').strip()}"

            # Supporting line
            first_line = context_window.splitlines()[0].strip()

            port_direction = "Provided (P-Port)" if "P-PORT" in kind_str and "PR" not in kind_str else (
                "Required (R-Port)" if "R-PORT" in kind_str and "PR" not in kind_str else "Provided-Required (PR-Port)"
            )

            ports.append(
                ExtractedEntity(
                    name=port_name,
                    entity_type=EntityType.PORT,
                    page_number=page_num,
                    source_file=source_file,
                    supporting_text=first_line,
                    status=EvidenceStatus.DETECTED,
                    attributes={
                        "port_kind": kind_str,
                        "port_direction": port_direction,
                        "associated_interface": associated_if,
                        "interface_evidence_status": if_status.value,
                        "payload_details": payload_info,
                    },
                )
            )

        return ports

    def _extract_interfaces(self, text: str, page_num: int, source_file: str) -> List[ExtractedEntity]:
        """Extracts AUTOSAR Port Interfaces (Sender-Receiver, Client-Server)."""
        interfaces: List[ExtractedEntity] = []

        # Pattern: "Sender-Receiver Interface (IF_VehicleSpeed)" or "Client-Server Interface (IF_DiagStatusService)"
        if_pattern = re.compile(
            r"(?P<if_type>Sender-Receiver|Client-Server|Mode\s*Switch|Parameter)\s+(?:Port\s+)?Interface\s*(?:\((?P<name>[A-Za-z0-9_]+)\)|:\s*(?P<name2>[A-Za-z0-9_]+))?",
            re.IGNORECASE,
        )

        for match in if_pattern.finditer(text):
            if_type = match.group("if_type").title()
            if_name = match.group("name") or match.group("name2")

            # Extract window around match to extract data element or operation
            start_pos = match.start()
            context_window = text[max(0, start_pos - 50) : min(len(text), start_pos + 300)]
            first_line = [l.strip() for l in context_window.splitlines() if match.group(0) in l]
            supporting_line = first_line[0] if first_line else match.group(0)

            # If no explicit name inside parentheses, check if an interface name is mentioned nearby
            status = EvidenceStatus.DETECTED
            if not if_name:
                fallback_name = re.search(r"\b(IF_[A-Za-z0-9_]+)\b", context_window)
                if fallback_name:
                    if_name = fallback_name.group(1)
                    status = EvidenceStatus.INFERRED
                else:
                    if_name = f"Unnamed_{if_type.replace(' ', '')}_Interface"
                    status = EvidenceStatus.UNVERIFIED

            # Check for associated operations or data elements
            data_elem = re.search(r"Data\s+Elements?\s*:\s*(?P<elem>[^\n]+)", context_window, re.IGNORECASE)
            op_elem = re.search(r"Operation\s*:\s*(?P<op>[^\n]+)", context_window, re.IGNORECASE)

            details = []
            if data_elem:
                details.append(f"Data: {data_elem.group('elem').strip()}")
            if op_elem:
                details.append(f"Operation: {op_elem.group('op').strip()}")

            interfaces.append(
                ExtractedEntity(
                    name=if_name,
                    entity_type=EntityType.INTERFACE,
                    page_number=page_num,
                    source_file=source_file,
                    supporting_text=supporting_line,
                    status=status,
                    attributes={
                        "interface_type": if_type,
                        "members": ", ".join(details) if details else "Not Specified",
                    },
                )
            )

        return interfaces

    def _extract_runnables_and_events(self, text: str, page_num: int, source_file: str) -> List[ExtractedEntity]:
        """Extracts Runnables, their trigger RTE Events, and OS Task mappings."""
        extracted: List[ExtractedEntity] = []

        # Find runnable blocks (e.g., "1. Runnable: RE_CalculateSpeed")
        runnable_pattern = re.compile(
            r"Runnable\s*:\s*(?P<name>[A-Za-z0-9_]+)",
            re.IGNORECASE,
        )

        for match in runnable_pattern.finditer(text):
            runnable_name = match.group("name")
            start_pos = match.start()
            context_window = text[start_pos : start_pos + 450]

            # Detect Trigger / RTE Event
            trigger_match = re.search(
                r"Trigger\s*:\s*(?P<trigger>[^\n]+)",
                context_window,
                re.IGNORECASE,
            )
            trigger_str = trigger_match.group("trigger").strip() if trigger_match else "Not Specified"

            # Detect OS Task
            task_match = re.search(
                r"OS\s+Task\s*:\s*(?P<task>[^\n]+)",
                context_window,
                re.IGNORECASE,
            )
            task_str = task_match.group("task").strip() if task_match else "Not Specified"

            # Detect Description
            desc_match = re.search(
                r"Description\s*:\s*(?P<desc>[^\n]+)",
                context_window,
                re.IGNORECASE,
            )
            desc_str = desc_match.group("desc").strip() if desc_match else "Not Specified"

            first_line = context_window.splitlines()[0].strip()

            # 1. Add Runnable Entity
            extracted.append(
                ExtractedEntity(
                    name=runnable_name,
                    entity_type=EntityType.RUNNABLE,
                    page_number=page_num,
                    source_file=source_file,
                    supporting_text=first_line,
                    status=EvidenceStatus.DETECTED,
                    attributes={
                        "trigger_event": trigger_str,
                        "mapped_os_task": task_str,
                        "description": desc_str,
                    },
                )
            )

            # 2. Add Corresponding RTE Event if detected
            if trigger_match and trigger_str != "Not Specified":
                # Determine event category (Timing Event, Init Event, Data Received Event, etc.)
                event_name = f"Event_{runnable_name}"
                event_kind = "RTE Event"
                periodicity = "Aperiodic / Event-Driven"

                if "Timing" in trigger_str or "ms" in trigger_str:
                    event_kind = "TimingEvent"
                    period_match = re.search(r"(\d+(?:\.\d+)?\s*m?s)", trigger_str)
                    if period_match:
                        periodicity = period_match.group(1)
                elif "Init" in trigger_str:
                    event_kind = "InitEvent"
                    periodicity = "Initialization"
                elif "Data Received" in trigger_str:
                    event_kind = "DataReceivedEvent"
                elif "Operation" in trigger_str:
                    event_kind = "OperationInvokedEvent"

                extracted.append(
                    ExtractedEntity(
                        name=f"{runnable_name}_{event_kind}",
                        entity_type=EntityType.RTE_EVENT,
                        page_number=page_num,
                        source_file=source_file,
                        supporting_text=trigger_match.group(0),
                        status=EvidenceStatus.DETECTED,
                        attributes={
                            "event_category": event_kind,
                            "trigger_specification": trigger_str,
                            "associated_runnable": runnable_name,
                            "activation_period": periodicity,
                        },
                    )
                )

        return extracted

    def _extract_bsw_modules(self, text: str, page_num: int, source_file: str) -> List[ExtractedEntity]:
        """Extracts referenced Basic Software (BSW) modules and diagnostic stacks."""
        bsw_entities: List[ExtractedEntity] = []
        lines = text.splitlines()

        for line in lines:
            line_str = line.strip()

            # Scan line for known BSW module keys
            for module_name, category in self.BSW_MODULE_CATALOG.items():
                # Word boundary match to prevent substrings (e.g. "Can" in "Candidate")
                pattern = rf"\b{re.escape(module_name)}\b"
                if re.search(pattern, line_str):
                    # Check if line indicates dependence, service call, or stack usage
                    is_dependency_line = any(
                        term in line_str.lower()
                        for term in ["bsw", "dependenc", "stack", "call", "service", "dem_", "nvm", "manager", "driver"]
                    )
                    status = EvidenceStatus.DETECTED if is_dependency_line else EvidenceStatus.INFERRED

                    bsw_entities.append(
                        ExtractedEntity(
                            name=module_name,
                            entity_type=EntityType.BSW_MODULE,
                            page_number=page_num,
                            source_file=source_file,
                            supporting_text=line_str,
                            status=status,
                            attributes={
                                "bsw_domain": category,
                                "usage_context": line_str,
                            },
                        )
                    )

        return bsw_entities

    def _extract_dependencies_and_safety(self, text: str, page_num: int, source_file: str) -> List[ExtractedEntity]:
        """Extracts functional safety (ASIL), E2E protection, and architectural verification relationships."""
        relations: List[ExtractedEntity] = []

        # 1. ASIL / Safety standard (ISO 26262)
        asil_match = re.search(r"\b(ASIL[- ][A-D]|QM)\b", text, re.IGNORECASE)
        if asil_match:
            asil_grade = asil_match.group(1).upper()
            line_with_asil = [l.strip() for l in text.splitlines() if asil_match.group(0) in l]
            supporting = line_with_asil[0] if line_with_asil else asil_match.group(0)
            relations.append(
                ExtractedEntity(
                    name=f"FunctionalSafety_{asil_grade}",
                    entity_type=EntityType.DEPENDENCY,
                    page_number=page_num,
                    source_file=source_file,
                    supporting_text=supporting,
                    status=EvidenceStatus.DETECTED,
                    attributes={
                        "relation_type": "Safety Requirement",
                        "standard": "ISO 26262",
                        "safety_level": asil_grade,
                    },
                )
            )

        # 2. E2E Protection
        e2e_match = re.search(r"E2E\s+Protection\s*:\s*(?P<spec>[^\n]+)", text, re.IGNORECASE)
        if e2e_match:
            relations.append(
                ExtractedEntity(
                    name="E2E_Protection_Profile",
                    entity_type=EntityType.DEPENDENCY,
                    page_number=page_num,
                    source_file=source_file,
                    supporting_text=e2e_match.group(0),
                    status=EvidenceStatus.DETECTED,
                    attributes={
                        "relation_type": "End-to-End Data Protection",
                        "specification": e2e_match.group("spec").strip(),
                    },
                )
            )

        # 3. Verification checklist items (e.g. [ ] Server response timeout ... Under Review)
        checklist_items = re.findall(r"\[(?P<box>[ xX])\]\s*(?P<item>[^\n]+)", text)
        for box, item in checklist_items:
            is_checked = box.lower() == "x"
            status = EvidenceStatus.DETECTED if is_checked else EvidenceStatus.UNVERIFIED
            relations.append(
                ExtractedEntity(
                    name=f"VerificationCheck_{item[:30].replace(' ', '_')}",
                    entity_type=EntityType.DEPENDENCY,
                    page_number=page_num,
                    source_file=source_file,
                    supporting_text=f"[{box}] {item}",
                    status=status,
                    attributes={
                        "relation_type": "Checklist Verification Item",
                        "verified": is_checked,
                        "details": item.strip(),
                    },
                )
            )

        return relations

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Deduplicates extracted entities by name and entity_type, retaining the highest evidence status."""
        seen: Dict[Tuple[str, str], ExtractedEntity] = {}

        for entity in entities:
            key = (entity.entity_type.value, entity.name.strip())
            if key not in seen:
                seen[key] = entity
            else:
                existing = seen[key]
                # Prioritize DETECTED over INFERRED, and INFERRED over UNVERIFIED
                priority = {
                    EvidenceStatus.DETECTED: 3,
                    EvidenceStatus.INFERRED: 2,
                    EvidenceStatus.UNVERIFIED: 1,
                }
                if priority.get(entity.status, 0) > priority.get(existing.status, 0):
                    seen[key] = entity
                # Merge attributes if missing
                for k, v in entity.attributes.items():
                    if k not in existing.attributes or existing.attributes[k] in ("Not Specified", "Not Explicitly Specified"):
                        existing.attributes[k] = v

        return list(seen.values())
