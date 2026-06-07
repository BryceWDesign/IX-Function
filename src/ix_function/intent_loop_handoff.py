"""IX-IntentRealityLoop handoff packets for IX-Function.

The IntentRealityLoop handoff converts a completed IX-Function transfer trial
into an intent, permission, action, feedback, and memory-update package. The
handoff preserves the boundary that a transfer result may guide future behavior
only after permission, review, and reality feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.learning import LearningDisposition
from ix_function.trial import TransferTrialResult, TrialStatus


class IntentLoopHandoffStatus(StrEnum):
    """Status of an IX-Function handoff to IX-IntentRealityLoop."""

    READY_FOR_FEEDBACK_BINDING = "ready_for_feedback_binding"
    READY_FOR_FAILURE_BINDING = "ready_for_failure_binding"
    BLOCKED = "blocked"


class IntentPermissionState(StrEnum):
    """Permission state for future reuse of a transfer pattern."""

    ALLOW_REVIEW_ONLY = "allow_review_only"
    BLOCK_AUTOMATIC_REUSE = "block_automatic_reuse"
    REQUIRE_HUMAN_PERMISSION = "require_human_permission"


class RealityFeedbackKind(StrEnum):
    """Kind of reality feedback carried into IntentRealityLoop."""

    SUPPORTED_TRANSFER = "supported_transfer"
    FAILED_TRANSFER = "failed_transfer"
    MIXED_TRANSFER = "mixed_transfer"
    UNSCORABLE_TRANSFER = "unscorable_transfer"


@dataclass(frozen=True, slots=True)
class IntentBinding:
    """Human-intent binding for a transfer trial."""

    intent_id: str
    trial_id: str
    requested_transfer: str
    source_domain_id: str
    target_domain_id: str
    permitted_scope: str
    prohibited_scope: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PermissionBinding:
    """Permission boundary for future transfer reuse."""

    permission_id: str
    state: IntentPermissionState
    requires_human_review: bool
    allowed_next_actions: tuple[str, ...]
    blocked_next_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealityFeedbackBinding:
    """Reality feedback summary for IntentRealityLoop memory binding."""

    feedback_id: str
    kind: RealityFeedbackKind
    reality_delta_report_id: str
    confidence_delta: float
    outcome_summary: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntentMemoryBinding:
    """Memory update request for IntentRealityLoop."""

    memory_binding_id: str
    function_id: str
    disposition: LearningDisposition
    should_update_memory: bool
    should_quarantine: bool
    future_behavior_rules: tuple[str, ...]
    uncertainty_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntentRealityLoopHandoffPacket:
    """Complete IX-IntentRealityLoop handoff packet."""

    packet_id: str
    status: IntentLoopHandoffStatus
    intent_binding: IntentBinding
    permission_binding: PermissionBinding
    feedback_binding: RealityFeedbackBinding
    memory_binding: IntentMemoryBinding
    required_loop_actions: tuple[str, ...]
    claim_boundary: str


def build_intent_loop_handoff_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> IntentRealityLoopHandoffPacket:
    """Build an IntentRealityLoop handoff from a transfer trial result."""

    status = choose_intent_loop_handoff_status(result)
    intent_binding = build_intent_binding(result)
    permission_binding = build_permission_binding(result)
    feedback_binding = build_feedback_binding(
        result=result,
        evidence_packet=evidence_packet,
    )
    memory_binding = build_intent_memory_binding(result)

    return IntentRealityLoopHandoffPacket(
        packet_id=f"{result.trial_id}:intent-loop-handoff",
        status=status,
        intent_binding=intent_binding,
        permission_binding=permission_binding,
        feedback_binding=feedback_binding,
        memory_binding=memory_binding,
        required_loop_actions=required_intent_loop_actions(result, status),
        claim_boundary=intent_loop_claim_boundary(),
    )


def choose_intent_loop_handoff_status(
    result: TransferTrialResult,
) -> IntentLoopHandoffStatus:
    """Choose IntentRealityLoop handoff status from transfer-trial result."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return IntentLoopHandoffStatus.READY_FOR_FEEDBACK_BINDING
    if result.reality_delta.observable_deltas:
        return IntentLoopHandoffStatus.READY_FOR_FAILURE_BINDING
    return IntentLoopHandoffStatus.BLOCKED


def build_intent_binding(result: TransferTrialResult) -> IntentBinding:
    """Build the intent binding for a completed transfer trial."""

    return IntentBinding(
        intent_id=f"{result.trial_id}:intent",
        trial_id=result.trial_id,
        requested_transfer=(
            "Evaluate whether a learned causal function can transfer into a "
            "different target domain under pre-outcome prediction and feedback."
        ),
        source_domain_id=result.source_evidence.source_domain_id,
        target_domain_id=result.mapping.target_domain_id,
        permitted_scope=(
            "Bounded causal-transfer evaluation, evidence packaging, and "
            "future-behavior guidance under human review."
        ),
        prohibited_scope=(
            "AGI proof",
            "deployment authorization",
            "automatic production use",
            "self-approval",
            "unsupervised operational autonomy",
        ),
    )


def build_permission_binding(result: TransferTrialResult) -> PermissionBinding:
    """Build the permission state for future transfer reuse."""

    if result.permits_bounded_evidence():
        return PermissionBinding(
            permission_id=f"{result.trial_id}:permission",
            state=IntentPermissionState.REQUIRE_HUMAN_PERMISSION,
            requires_human_review=True,
            allowed_next_actions=(
                "Use as bounded evidence in a human-reviewed packet.",
                "Schedule a held-out-domain retest.",
                "Create a new pre-outcome prediction before future reuse.",
            ),
            blocked_next_actions=(
                "Do not treat the result as truth.",
                "Do not reuse automatically without a new prediction.",
                "Do not represent the result as AGI proof.",
            ),
        )

    return PermissionBinding(
        permission_id=f"{result.trial_id}:permission",
        state=IntentPermissionState.BLOCK_AUTOMATIC_REUSE,
        requires_human_review=True,
        allowed_next_actions=(
            "Bind failure evidence to memory.",
            "Retest only after resolving required actions.",
            "Preserve uncertainty and falsification records.",
        ),
        blocked_next_actions=(
            "Do not strengthen confidence.",
            "Do not promote skill reuse.",
            "Do not create deployment or AGI claims.",
        ),
    )


def build_feedback_binding(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> RealityFeedbackBinding:
    """Build the reality-feedback binding for IntentRealityLoop."""

    return RealityFeedbackBinding(
        feedback_id=f"{result.trial_id}:feedback",
        kind=reality_feedback_kind(result),
        reality_delta_report_id=result.reality_delta.report_id,
        confidence_delta=result.reality_delta.confidence_delta,
        outcome_summary=(
            f"Reality-delta status={result.reality_delta.status.value}; "
            f"mean_score={result.reality_delta.mean_score:.6f}; "
            f"trial_status={result.status.value}."
        ),
        evidence_refs=(
            evidence_packet.packet_id,
            evidence_packet.manifest_sha256_digest,
            result.reality_delta.report_id,
            result.learning_update.update_id,
            result.falsification_ledger.ledger_id,
        ),
    )


def build_intent_memory_binding(result: TransferTrialResult) -> IntentMemoryBinding:
    """Build the memory-binding request for IntentRealityLoop."""

    should_quarantine = not result.permits_bounded_evidence() or (
        result.learning_update.disposition is LearningDisposition.QUARANTINE
    )
    return IntentMemoryBinding(
        memory_binding_id=f"{result.trial_id}:intent-memory-binding",
        function_id=result.learning_update.function_id,
        disposition=result.learning_update.disposition,
        should_update_memory=True,
        should_quarantine=should_quarantine,
        future_behavior_rules=result.learning_update.future_planning_rules,
        uncertainty_refs=tuple(
            item.uncertainty_id for item in result.uncertainty_ledger.open_items()
        ),
    )


def reality_feedback_kind(result: TransferTrialResult) -> RealityFeedbackKind:
    """Classify trial reality feedback for IntentRealityLoop."""

    status = result.reality_delta.status.value
    if status == "supported":
        return RealityFeedbackKind.SUPPORTED_TRANSFER
    if status == "mixed":
        return RealityFeedbackKind.MIXED_TRANSFER
    if status == "failed":
        return RealityFeedbackKind.FAILED_TRANSFER
    return RealityFeedbackKind.UNSCORABLE_TRANSFER


def required_intent_loop_actions(
    result: TransferTrialResult,
    status: IntentLoopHandoffStatus,
) -> tuple[str, ...]:
    """Return required IntentRealityLoop-side actions."""

    if status is IntentLoopHandoffStatus.READY_FOR_FEEDBACK_BINDING:
        return (
            "Bind reality feedback to memory as bounded evidence only.",
            "Require human permission before any future reuse.",
            "Require a new pre-outcome prediction for every future transfer.",
            "Preserve uncertainty and falsification references.",
        )

    if status is IntentLoopHandoffStatus.READY_FOR_FAILURE_BINDING:
        return (
            "Bind failed or downgraded feedback as negative evidence.",
            "Block automatic reuse until retesting resolves required actions.",
            "Preserve the failed prediction and outcome delta.",
            *result.required_actions,
        )

    return (
        "Block memory promotion until complete feedback records exist.",
        "Require human review before retry.",
        *result.required_actions,
    )


def intent_loop_claim_boundary() -> str:
    """Return fixed claim boundary for IntentRealityLoop handoff."""

    return (
        "IX-Function IntentRealityLoop handoff binds intent, permission, reality "
        "feedback, and memory guidance. It is not AGI proof, not self-approval, "
        "and not deployment authority."
    )


def validate_intent_loop_handoff_packet(
    packet: IntentRealityLoopHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors for an IntentRealityLoop handoff packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.intent_binding.intent_id.strip():
        errors.append("intent_id must not be empty")
    if not packet.intent_binding.trial_id.strip():
        errors.append("intent trial_id must not be empty")
    if not packet.intent_binding.source_domain_id.strip():
        errors.append("source_domain_id must not be empty")
    if not packet.intent_binding.target_domain_id.strip():
        errors.append("target_domain_id must not be empty")
    if not packet.intent_binding.prohibited_scope:
        errors.append("prohibited_scope must not be empty")
    if not packet.permission_binding.permission_id.strip():
        errors.append("permission_id must not be empty")
    if not packet.permission_binding.blocked_next_actions:
        errors.append("blocked_next_actions must not be empty")
    if not packet.feedback_binding.feedback_id.strip():
        errors.append("feedback_id must not be empty")
    if not packet.feedback_binding.evidence_refs:
        errors.append("feedback evidence_refs must not be empty")
    if not packet.memory_binding.memory_binding_id.strip():
        errors.append("memory_binding_id must not be empty")
    if not packet.memory_binding.function_id.strip():
        errors.append("memory function_id must not be empty")
    if not packet.required_loop_actions:
        errors.append("required_loop_actions must not be empty")
    if packet.claim_boundary != intent_loop_claim_boundary():
        errors.append("claim_boundary must match fixed IntentRealityLoop boundary")

    return tuple(errors)
