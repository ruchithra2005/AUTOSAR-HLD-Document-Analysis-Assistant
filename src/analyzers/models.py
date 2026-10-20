"""Data models for AUTOSAR consistency checking and rule evaluation.

Defines schemas for RuleFindings, Severity levels (ERROR, WARNING, PASS, UNVERIFIED),
and aggregated ConsistencyReports with evidence provenance.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Classification of rule evaluation results."""
    ERROR = "ERROR"              # Definite architectural violation or missing mandatory element
    WARNING = "WARNING"          # Inconsistency, ambiguity, or potential runtime risk
    PASS = "PASS"                # Verified architectural rule grounded in concrete evidence
    UNVERIFIED = "UNVERIFIED"    # Incomplete or missing evidence; requires human review rather than violation


@dataclass
class RuleFinding:
    """Individual rule evaluation finding with complete evidence provenance."""
    rule_id: str                      # e.g., "AR-PORT-01"
    rule_name: str                    # e.g., "Port Interface Association"
    severity: Severity                # ERROR, WARNING, PASS, UNVERIFIED
    target_entity: str                # e.g., "ppVehicleSpeed"
    target_entity_type: str           # e.g., "PORT", "RUNNABLE", "INTERFACE"
    message: str                      # Detailed explanation of finding
    recommendation: str               # Actionable engineering remediation
    page_number: int                  # Page where target entity or evidence was found
    source_file: str                  # Source document name
    supporting_evidence: str          # Verbatim excerpt from document
    requires_safety_review: bool = False  # Flag for functional safety engineer review

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ConsistencyReport:
    """Container aggregating all rule findings for an evaluated HLD document."""
    source_file: str
    total_rules_checked: int
    findings: List[RuleFinding] = field(default_factory=list)
    evaluation_timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def get_counts(self) -> Dict[str, int]:
        """Returns summary counts of findings per severity."""
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        return counts

    def filter_by_severity(self, severity: Severity) -> List[RuleFinding]:
        """Returns findings matching the specified severity level."""
        return [f for f in self.findings if f.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "total_rules_checked": self.total_rules_checked,
            "evaluation_timestamp": self.evaluation_timestamp,
            "summary_counts": self.get_counts(),
            "findings": [f.to_dict() for f in self.findings],
        }
