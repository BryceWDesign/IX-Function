"""IX-Autonomy-Assurance-Case-Runtime handoff packets for IX-Function.

The assurance handoff converts IX-Function transfer evidence into traceable
claim, safety-gate, provenance, and human-authority records. It supports review
and falsification. It does not certify AGI, autonomy, safety, or deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.falsification import FalsificationVerdict
from ix_function.trial import TransferTrialResult, TrialStatus


class AssuranceHandoffStatus(StrEnum):
    """Status of an IX-Function assurance handoff."""

    READY_FOR_ASSURANCE_REVIEW = "ready_for_assurance_review"
    READY_FOR_FAILURE_DOSSIER = "ready_for_failure_dossier"
    BLOCKED = "blocked"


class AssuranceDecision(StrEnum):
    """Assurance decision for the transfer claim."""

    ALLOW_REVIEW_CLAIM = "allow_review_claim"
    BLOCK_CLAIM = "block_claim"
    REQUIRE_MORE_EVIDENCE = "require_more_evidence"


class SafetyGateDecision(StrEnum):
    """Safety-gate decision for transfer evidence."""

    ALLOW_REVIEW_ONLY = "allow_review_only"
    SAFE_HOLD = "safe_hold"
    VETO = "veto"


@dataclass(frozen=True, slots=True)
class AssuranceTraceLink:
    """Trace link from IX-Function evidence into an assurance case."""

    trace_id: str
    source_artifact_id: str
    target_claim_id: str
    relation: str
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class AssuranceSafetyGate:
    """Safety gate used before any assurance claim can be reviewed."""

    gate_id: str
    decision: SafetyGateDecision
    allowed: bool
    reason: str
    required_controls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssuranceClaim:
    """Bounded assurance claim derived from IX-Function evidence."""

    claim_id: str
    trial_id: str
    decision: AssuranceDecision
    claim_text: str
    supported: bool
    blocked_claims: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    uncertainty_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssuranceProvenanceBinding:
    """Provenance binding for assurance review."""

    provenance_id: str
    evidence_packet_id: str
    evidence_manifest_digest: str
    falsification_ledger_id: str
    negative_control_suite_id: str
    human_authority_required: bool


@dataclass(frozen=True, slots=True)
class AssuranceHandoffPacket:
    """Complete assurance-runtime handoff packet."""

    packet_id: str
    status: AssuranceHandoffStatus
    claim: AssuranceClaim
    safety_gates: tuple[AssuranceSafetyGate, ...]
    trace_links: tuple[AssuranceTraceLink, ...]
    provenance: AssuranceProvenanceBinding
    required_assurance_actions: tuple[str, ...]
    claim_boundary: str


def build_assurance_handoff_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> AssuranceHandoffPacket:
    """Build an assurance handoff from IX-Function trial evidence."""

    status = choose_assurance_handoff_status(result)
    claim = build_assurance_claim(result=result, evidence_packet=evidence_packet)
    safety_gates = build_assurance_safety_gates(result)
    trace_links = build_assurance_trace_links(
        result=result,
        evidence_packet=evidence_packet,
        claim_id=claim.claim_id,
    )
    provenance = AssuranceProvenanceBinding(
        provenance_id=f"{result.trial_id}:assurance-provenance",
        evidence_packet_id=evidence_packet.packet_id,
        evidence_manifest_digest=evidence_packet.manifest_sha256_digest,
        falsification_ledger_id=result.falsification_ledger.ledger_id,
        negative_control_suite_id=result.negative_control_suite.suite_id,
        human_authority_required=True,
    )

    return AssuranceHandoffPacket(
        packet_id=f"{result.trial_id}:assurance-handoff",
        status=status,
        claim=claim,
        safety_gates=safety_gates,
        trace_links=trace_links,
        provenance=provenance,
        required_assurance_actions=required_assurance_actions(result, status),
        claim_boundary=assurance_claim_boundary(),
    )


def choose_assurance_handoff_status(
    result: TransferTrialResult,
) -> AssuranceHandoffStatus:
    """Choose assurance handoff status from final trial result."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return AssuranceHandoffStatus.READY_FOR_ASSURANCE_REVIEW
    if result.reality_delta.observable_deltas:
        return AssuranceHandoffStatus.READY_FOR_FAILURE_DOSSIER
    return AssuranceHandoffStatus.BLOCKED


def build_assurance_claim(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> AssuranceClaim:
    """Build the bounded assurance claim for a transfer result."""

    supported = result.permits_bounded_evidence()
    decision = (
        AssuranceDecision.ALLOW_REVIEW_CLAIM
        if supported
        else AssuranceDecision.BLOCK_CLAIM
    )
    claim_text = (
        "IX-Function produced bounded, reviewable cross-domain causal-transfer "
        "evidence under uncertainty, falsification, and human authority gates."
        if supported
        else (
            "IX-Function did not produce bounded transfer evidence; the claim "
            "must remain blocked or downgraded until retesting resolves failures."
        )
    )

    return AssuranceClaim(
        claim_id=f"{result.trial_id}:assurance-claim",
        trial_id=result.trial_id,
        decision=decision,
        claim_text=claim_text,
        supported=supported,
        blocked_claims=(
            "AGI proof",
            "independent AGI validation",
            "deployment authorization",
            "safety certification",
            "production readiness",
            "self-approval",
        ),
        evidence_refs=(
            evidence_packet.packet_id,
            evidence_packet.manifest_sha256_digest,
            result.source_evidence.evidence_id,
            result.reality_delta.report_id,
            result.learning_update.update_id,
            result.falsification_ledger.ledger_id,
        ),
        uncertainty_refs=tuple(
            item.uncertainty_id for item in result.uncertainty_ledger.open_items()
        ),
    )


def build_assurance_safety_gates(
    result: TransferTrialResult,
) -> tuple[AssuranceSafetyGate, ...]:
    """Build safety gates for assurance review."""

    gates: list[AssuranceSafetyGate] = [
        AssuranceSafetyGate(
            gate_id="IXF-ASSURANCE-001",
            decision=SafetyGateDecision.ALLOW_REVIEW_ONLY,
            allowed=True,
            reason="Evidence may be reviewed but not treated as certification.",
            required_controls=(
                "Human authority required.",
                "No deployment authority is granted.",
                "No AGI proof language is allowed.",
            ),
        )
    ]

    if result.permits_bounded_evidence():
        gates.append(
            AssuranceSafetyGate(
                gate_id="IXF-ASSURANCE-002",
                decision=SafetyGateDecision.ALLOW_REVIEW_ONLY,
                allowed=True,
                reason="Bounded evidence passed trial gates.",
                required_controls=(
                    "Preserve uncertainty references.",
                    "Preserve falsification and negative-control results.",
                    "Require independent replay before stronger claims.",
                ),
            )
        )
    else:
        gates.append(
            AssuranceSafetyGate(
                gate_id="IXF-ASSURANCE-002",
                decision=SafetyGateDecision.SAFE_HOLD,
                allowed=False,
                reason="Trial did not permit bounded evidence.",
                required_controls=(
                    "Block claim promotion.",
                    "Bind failure evidence to the assurance dossier.",
                    *result.required_actions,
                ),
            )
        )

    if result.falsification_ledger.verdict is FalsificationVerdict.KILL_CLAIM:
        gates.append(
            AssuranceSafetyGate(
                gate_id="IXF-ASSURANCE-003",
                decision=SafetyGateDecision.VETO,
                allowed=False,
                reason="Falsification ledger killed the transfer claim.",
                required_controls=result.falsification_ledger.required_actions,
            )
        )

    return tuple(gates)


def build_assurance_trace_links(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
    claim_id: str,
) -> tuple[AssuranceTraceLink, ...]:
    """Build trace links from evidence artifacts to the assurance claim."""

    artifact_refs = (
        result.source_evidence.evidence_id,
        result.reality_delta.report_id,
        result.learning_update.update_id,
        result.uncertainty_ledger.ledger_id,
        result.falsification_ledger.ledger_id,
        result.negative_control_suite.suite_id,
    )
    return tuple(
        AssuranceTraceLink(
            trace_id=f"{result.trial_id}:trace:{index}",
            source_artifact_id=artifact_ref,
            target_claim_id=claim_id,
            relation="supports_or_constrains",
            evidence_digest=evidence_packet.manifest_sha256_digest,
        )
        for index, artifact_ref in enumerate(artifact_refs, start=1)
    )


def required_assurance_actions(
    result: TransferTrialResult,
    status: AssuranceHandoffStatus,
) -> tuple[str, ...]:
    """Return required assurance-runtime actions."""

    if status is AssuranceHandoffStatus.READY_FOR_ASSURANCE_REVIEW:
        return (
            "Create a human-reviewed assurance claim only.",
            "Bind claim to evidence packet digest and trace links.",
            "Preserve uncertainty, falsification, and negative controls.",
            "Do not certify AGI, autonomy, safety, or deployment readiness.",
        )

    if status is AssuranceHandoffStatus.READY_FOR_FAILURE_DOSSIER:
        return (
            "Create a failure dossier instead of a support claim.",
            "Block confidence promotion.",
            "Require retesting before any stronger assurance claim.",
            *result.required_actions,
        )

    return (
        "Block assurance handoff until complete evidence exists.",
        "Require human review before retry.",
        *result.required_actions,
    )


def assurance_claim_boundary() -> str:
    """Return fixed assurance handoff claim boundary."""

    return (
        "IX-Function assurance handoff is bounded review evidence. It is not "
        "AGI proof, safety certification, deployment authorization, production "
        "readiness, or independent validation."
    )


def validate_assurance_handoff_packet(
    packet: AssuranceHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors for an assurance handoff packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.claim.claim_id.strip():
        errors.append("claim_id must not be empty")
    if not packet.claim.trial_id.strip():
        errors.append("claim trial_id must not be empty")
    if not packet.claim.claim_text.strip():
        errors.append("claim_text must not be empty")
    if not packet.claim.blocked_claims:
        errors.append("blocked_claims must not be empty")
    if not packet.claim.evidence_refs:
        errors.append("claim evidence_refs must not be empty")
    if not packet.safety_gates:
        errors.append("at least one safety gate is required")
    if not packet.trace_links:
        errors.append("at least one trace link is required")
    if not packet.provenance.provenance_id.strip():
        errors.append("provenance_id must not be empty")
    if not packet.provenance.evidence_packet_id.strip():
        errors.append("provenance evidence_packet_id must not be empty")
    if not packet.provenance.evidence_manifest_digest.strip():
        errors.append("provenance evidence_manifest_digest must not be empty")
    if not packet.provenance.human_authority_required:
        errors.append("human_authority_required must remain true")
    if not packet.required_assurance_actions:
        errors.append("required_assurance_actions must not be empty")
    if packet.claim_boundary != assurance_claim_boundary():
        errors.append("claim_boundary must match fixed assurance boundary")

    gate_ids = [gate.gate_id for gate in packet.safety_gates]
    if len(set(gate_ids)) != len(gate_ids):
        errors.append("safety gates must use unique gate_id values")

    trace_ids = [trace.trace_id for trace in packet.trace_links]
    if len(set(trace_ids)) != len(trace_ids):
        errors.append("trace links must use unique trace_id values")

    for gate in packet.safety_gates:
        if not gate.gate_id.strip():
            errors.append("gate_id must not be empty")
        if not gate.reason.strip():
            errors.append(f"reason must not be empty for {gate.gate_id!r}")
        if not gate.required_controls:
            errors.append(
                f"required_controls must not be empty for {gate.gate_id!r}"
            )

    for trace in packet.trace_links:
        if not trace.source_artifact_id.strip():
            errors.append(f"source_artifact_id must not be empty for {trace.trace_id!r}")
        if trace.target_claim_id != packet.claim.claim_id:
            errors.append(f"trace {trace.trace_id!r} must target the packet claim")
        if not trace.evidence_digest.strip():
            errors.append(f"evidence_digest must not be empty for {trace.trace_id!r}")

    return tuple(errors)
