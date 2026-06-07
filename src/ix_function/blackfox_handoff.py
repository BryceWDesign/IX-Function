"""IX-BlackFox handoff packets for IX-Function.

The BlackFox handoff converts IX-Function transfer evidence into a governed
review bundle. BlackFox treats all model/cognition outputs as untrusted inputs
until policy gates, evidence bindings, and human review decide what can be
accepted for further evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.falsification import FalsificationVerdict
from ix_function.trial import TransferTrialResult, TrialStatus


class BlackFoxHandoffStatus(StrEnum):
    """Status of an IX-Function handoff to IX-BlackFox."""

    READY_FOR_REVIEW_BUNDLE = "ready_for_review_bundle"
    READY_FOR_FAILURE_REVIEW = "ready_for_failure_review"
    BLOCKED = "blocked"


class BlackFoxPolicyDecision(StrEnum):
    """Policy decision for a BlackFox review gate."""

    ALLOW_REVIEW_ONLY = "allow_review_only"
    BLOCK = "block"
    REQUIRE_HUMAN_APPROVAL = "require_human_approval"


@dataclass(frozen=True, slots=True)
class BlackFoxEvidenceBinding:
    """Evidence binding used by BlackFox review workflows."""

    binding_id: str
    trial_id: str
    evidence_packet_id: str
    evidence_manifest_digest: str
    trial_status: str
    falsification_verdict: str
    anti_theater_allowed: bool
    model_output_trusted: bool
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class BlackFoxPolicyGate:
    """Policy gate result for a BlackFox review bundle."""

    gate_id: str
    decision: BlackFoxPolicyDecision
    allowed: bool
    reason: str
    required_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlackFoxReviewBundle:
    """Review bundle for governed BlackFox ingestion."""

    review_id: str
    sandbox_required: bool
    egress_allowed: bool
    human_approval_required: bool
    policy_gates: tuple[BlackFoxPolicyGate, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlackFoxHandoffPacket:
    """Complete IX-BlackFox handoff packet."""

    packet_id: str
    status: BlackFoxHandoffStatus
    evidence_binding: BlackFoxEvidenceBinding
    review_bundle: BlackFoxReviewBundle
    required_blackfox_actions: tuple[str, ...]
    claim_boundary: str


def build_blackfox_handoff_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> BlackFoxHandoffPacket:
    """Build a BlackFox governance handoff from IX-Function trial evidence."""

    status = choose_blackfox_handoff_status(result)
    evidence_binding = build_blackfox_evidence_binding(
        result=result,
        evidence_packet=evidence_packet,
    )
    review_bundle = build_blackfox_review_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    return BlackFoxHandoffPacket(
        packet_id=f"{result.trial_id}:blackfox-handoff",
        status=status,
        evidence_binding=evidence_binding,
        review_bundle=review_bundle,
        required_blackfox_actions=required_blackfox_actions(result, status),
        claim_boundary=blackfox_claim_boundary(),
    )


def choose_blackfox_handoff_status(
    result: TransferTrialResult,
) -> BlackFoxHandoffStatus:
    """Choose BlackFox handoff status from IX-Function gates."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return BlackFoxHandoffStatus.READY_FOR_REVIEW_BUNDLE
    if result.reality_delta.observable_deltas:
        return BlackFoxHandoffStatus.READY_FOR_FAILURE_REVIEW
    return BlackFoxHandoffStatus.BLOCKED


def build_blackfox_evidence_binding(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> BlackFoxEvidenceBinding:
    """Build immutable evidence references for BlackFox review."""

    return BlackFoxEvidenceBinding(
        binding_id=f"{result.trial_id}:blackfox-evidence-binding",
        trial_id=result.trial_id,
        evidence_packet_id=evidence_packet.packet_id,
        evidence_manifest_digest=evidence_packet.manifest_sha256_digest,
        trial_status=result.status.value,
        falsification_verdict=result.falsification_ledger.verdict.value,
        anti_theater_allowed=result.anti_theater_gate.allowed,
        model_output_trusted=False,
        claim_boundary=blackfox_claim_boundary(),
    )


def build_blackfox_review_bundle(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> BlackFoxReviewBundle:
    """Build the policy-gated BlackFox review bundle."""

    return BlackFoxReviewBundle(
        review_id=f"{result.trial_id}:blackfox-review",
        sandbox_required=True,
        egress_allowed=False,
        human_approval_required=True,
        policy_gates=build_blackfox_policy_gates(result),
        evidence_refs=(
            evidence_packet.packet_id,
            evidence_packet.manifest_sha256_digest,
            result.source_evidence.evidence_id,
            result.reality_delta.report_id,
            result.learning_update.update_id,
            result.uncertainty_ledger.ledger_id,
            result.falsification_ledger.ledger_id,
            result.negative_control_suite.suite_id,
        ),
    )


def build_blackfox_policy_gates(
    result: TransferTrialResult,
) -> tuple[BlackFoxPolicyGate, ...]:
    """Build BlackFox policy gates for a transfer result."""

    gates: list[BlackFoxPolicyGate] = [
        BlackFoxPolicyGate(
            gate_id="IXF-BLACKFOX-001",
            decision=BlackFoxPolicyDecision.REQUIRE_HUMAN_APPROVAL,
            allowed=True,
            reason="IX-Function evidence requires human review before reuse.",
            required_actions=(
                "Treat all transfer evidence as untrusted input.",
                "Require reviewer signoff before downstream use.",
            ),
        ),
        BlackFoxPolicyGate(
            gate_id="IXF-BLACKFOX-002",
            decision=BlackFoxPolicyDecision.ALLOW_REVIEW_ONLY,
            allowed=True,
            reason="AGI, deployment, and production-readiness claims are blocked.",
            required_actions=(
                "Do not label this result as AGI proof.",
                "Do not use this handoff for deployment authorization.",
            ),
        ),
    ]

    if result.permits_bounded_evidence():
        gates.append(
            BlackFoxPolicyGate(
                gate_id="IXF-BLACKFOX-003",
                decision=BlackFoxPolicyDecision.ALLOW_REVIEW_ONLY,
                allowed=True,
                reason="Bounded evidence can enter a review bundle.",
                required_actions=(
                    "Bind review to evidence digest.",
                    "Preserve uncertainty and falsification records.",
                ),
            )
        )
    else:
        gates.append(
            BlackFoxPolicyGate(
                gate_id="IXF-BLACKFOX-003",
                decision=BlackFoxPolicyDecision.BLOCK,
                allowed=False,
                reason="Trial did not permit bounded evidence.",
                required_actions=(
                    "Do not strengthen downstream belief.",
                    *result.required_actions,
                ),
            )
        )

    if result.falsification_ledger.verdict is FalsificationVerdict.KILL_CLAIM:
        gates.append(
            BlackFoxPolicyGate(
                gate_id="IXF-BLACKFOX-004",
                decision=BlackFoxPolicyDecision.BLOCK,
                allowed=False,
                reason="Falsification ledger killed the transfer claim.",
                required_actions=result.falsification_ledger.required_actions,
            )
        )

    return tuple(gates)


def required_blackfox_actions(
    result: TransferTrialResult,
    status: BlackFoxHandoffStatus,
) -> tuple[str, ...]:
    """Return required BlackFox-side actions."""

    if status is BlackFoxHandoffStatus.READY_FOR_REVIEW_BUNDLE:
        return (
            "Create a review-only evidence bundle.",
            "Bind reviewer decision to the evidence packet digest.",
            "Treat model and transfer outputs as untrusted until human approval.",
            "Do not permit deployment, AGI, or production-readiness claims.",
        )

    if status is BlackFoxHandoffStatus.READY_FOR_FAILURE_REVIEW:
        return (
            "Create a failure-review bundle.",
            "Block confidence promotion.",
            "Preserve failed prediction and reality-delta records.",
            *result.required_actions,
        )

    return (
        "Block BlackFox promotion until complete evidence exists.",
        "Require human review before retry.",
        *result.required_actions,
    )


def blackfox_claim_boundary() -> str:
    """Return fixed BlackFox claim boundary."""

    return (
        "IX-Function BlackFox handoff is a governed review bundle. It treats "
        "transfer output as untrusted input and does not grant AGI proof, "
        "deployment authority, production readiness, or self-approval."
    )


def validate_blackfox_handoff_packet(
    packet: BlackFoxHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors for a BlackFox handoff packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.evidence_binding.binding_id.strip():
        errors.append("binding_id must not be empty")
    if not packet.evidence_binding.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not packet.evidence_binding.evidence_packet_id.strip():
        errors.append("evidence_packet_id must not be empty")
    if not packet.evidence_binding.evidence_manifest_digest.strip():
        errors.append("evidence_manifest_digest must not be empty")
    if packet.evidence_binding.model_output_trusted:
        errors.append("model_output_trusted must remain false")
    if packet.evidence_binding.claim_boundary != blackfox_claim_boundary():
        errors.append("evidence binding claim_boundary must match fixed boundary")
    if not packet.review_bundle.review_id.strip():
        errors.append("review_id must not be empty")
    if not packet.review_bundle.sandbox_required:
        errors.append("sandbox_required must remain true")
    if packet.review_bundle.egress_allowed:
        errors.append("egress_allowed must remain false")
    if not packet.review_bundle.human_approval_required:
        errors.append("human_approval_required must remain true")
    if not packet.review_bundle.policy_gates:
        errors.append("at least one policy gate is required")
    if not packet.review_bundle.evidence_refs:
        errors.append("review evidence_refs must not be empty")
    if not packet.required_blackfox_actions:
        errors.append("required_blackfox_actions must not be empty")
    if packet.claim_boundary != blackfox_claim_boundary():
        errors.append("claim_boundary must match fixed BlackFox boundary")

    gate_ids = [gate.gate_id for gate in packet.review_bundle.policy_gates]
    if len(set(gate_ids)) != len(gate_ids):
        errors.append("policy gates must use unique gate_id values")

    for gate in packet.review_bundle.policy_gates:
        if not gate.gate_id.strip():
            errors.append("gate_id must not be empty")
        if not gate.reason.strip():
            errors.append(f"reason must not be empty for {gate.gate_id!r}")
        if not gate.required_actions:
            errors.append(
                f"required_actions must not be empty for {gate.gate_id!r}"
            )

    return tuple(errors)
