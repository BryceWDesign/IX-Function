"""IX-CognitionKernel handoff packets for IX-Function.

The Kernel handoff converts a finished IX-Function transfer trial into bounded
belief, memory, and skill-candidate records. The handoff does not tell the
Kernel that a causal transfer is true. It tells the Kernel what evidence was
produced, what changed, what uncertainty remains, and what future behavior must
be constrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.learning import LearningDisposition
from ix_function.trial import TransferTrialResult, TrialStatus
from ix_function.uncertainty import UncertaintySeverity


class KernelHandoffStatus(StrEnum):
    """Status of an IX-Function handoff to IX-CognitionKernel."""

    READY_FOR_REVIEW = "ready_for_review"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"


class KernelMemoryUpdateKind(StrEnum):
    """Memory update categories for Kernel ingestion."""

    FUTURE_PLANNING_RULE = "future_planning_rule"
    TRANSFER_FAILURE = "transfer_failure"
    TRANSFER_SUPPORT = "transfer_support"
    UNCERTAINTY_NOTE = "uncertainty_note"


@dataclass(frozen=True, slots=True)
class KernelBeliefUpdate:
    """Bounded belief update candidate for IX-CognitionKernel."""

    belief_id: str
    function_id: str
    trial_id: str
    disposition: LearningDisposition
    confidence_delta: float
    revised_confidence: float
    evidence_refs: tuple[str, ...]
    uncertainty_refs: tuple[str, ...]
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class KernelMemoryUpdate:
    """Memory update candidate derived from IX-Function evidence."""

    memory_id: str
    kind: KernelMemoryUpdateKind
    function_id: str
    content: str
    evidence_refs: tuple[str, ...]
    quarantine_recommended: bool


@dataclass(frozen=True, slots=True)
class KernelSkillCandidate:
    """Reusable skill candidate only when transfer evidence remains bounded."""

    skill_id: str
    function_id: str
    trial_id: str
    allowed_for_reuse: bool
    reuse_conditions: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    blocking_reason: str | None = None


@dataclass(frozen=True, slots=True)
class KernelHandoffPacket:
    """Complete IX-CognitionKernel handoff packet."""

    packet_id: str
    status: KernelHandoffStatus
    trial_id: str
    evidence_packet_id: str
    belief_update: KernelBeliefUpdate
    memory_updates: tuple[KernelMemoryUpdate, ...]
    skill_candidate: KernelSkillCandidate
    required_kernel_actions: tuple[str, ...]
    claim_boundary: str


def build_kernel_handoff_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> KernelHandoffPacket:
    """Build a Kernel handoff packet from a trial result and evidence packet."""

    status = choose_kernel_handoff_status(result)
    evidence_refs = evidence_reference_tuple(result, evidence_packet)
    uncertainty_refs = tuple(
        item.uncertainty_id for item in result.uncertainty_ledger.open_items()
    )

    belief_update = KernelBeliefUpdate(
        belief_id=f"{result.trial_id}:kernel-belief-update",
        function_id=result.learning_update.function_id,
        trial_id=result.trial_id,
        disposition=result.learning_update.disposition,
        confidence_delta=result.learning_update.confidence_delta,
        revised_confidence=result.learning_update.revised_confidence,
        evidence_refs=evidence_refs,
        uncertainty_refs=uncertainty_refs,
        claim_boundary=kernel_claim_boundary(),
    )

    memory_updates = build_kernel_memory_updates(
        result=result,
        evidence_refs=evidence_refs,
    )
    skill_candidate = build_kernel_skill_candidate(
        result=result,
        evidence_refs=evidence_refs,
    )

    return KernelHandoffPacket(
        packet_id=f"{result.trial_id}:kernel-handoff",
        status=status,
        trial_id=result.trial_id,
        evidence_packet_id=evidence_packet.packet_id,
        belief_update=belief_update,
        memory_updates=memory_updates,
        skill_candidate=skill_candidate,
        required_kernel_actions=required_kernel_actions(result, status),
        claim_boundary=kernel_claim_boundary(),
    )


def choose_kernel_handoff_status(result: TransferTrialResult) -> KernelHandoffStatus:
    """Choose Kernel handoff status from final transfer-trial state."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return KernelHandoffStatus.READY_FOR_REVIEW
    if result.learning_update.disposition is LearningDisposition.QUARANTINE:
        return KernelHandoffStatus.QUARANTINED
    return KernelHandoffStatus.BLOCKED


def build_kernel_memory_updates(
    *,
    result: TransferTrialResult,
    evidence_refs: tuple[str, ...],
) -> tuple[KernelMemoryUpdate, ...]:
    """Build bounded Kernel memory-update candidates."""

    updates: list[KernelMemoryUpdate] = []
    function_id = result.learning_update.function_id
    quarantine_recommended = (
        result.learning_update.disposition is LearningDisposition.QUARANTINE
        or not result.permits_bounded_evidence()
    )

    if result.permits_bounded_evidence():
        updates.append(
            KernelMemoryUpdate(
                memory_id=f"{result.trial_id}:memory:transfer-support",
                kind=KernelMemoryUpdateKind.TRANSFER_SUPPORT,
                function_id=function_id,
                content=(
                    "Transfer trial produced bounded support. Future reuse "
                    "requires a new pre-outcome prediction and fresh scoring."
                ),
                evidence_refs=evidence_refs,
                quarantine_recommended=False,
            )
        )
    else:
        updates.append(
            KernelMemoryUpdate(
                memory_id=f"{result.trial_id}:memory:transfer-failure",
                kind=KernelMemoryUpdateKind.TRANSFER_FAILURE,
                function_id=function_id,
                content=(
                    "Transfer trial did not permit bounded evidence. Future "
                    "plans must not promote this transfer without retesting."
                ),
                evidence_refs=evidence_refs,
                quarantine_recommended=True,
            )
        )

    for index, rule in enumerate(result.learning_update.future_planning_rules, start=1):
        updates.append(
            KernelMemoryUpdate(
                memory_id=f"{result.trial_id}:memory:future-rule:{index}",
                kind=KernelMemoryUpdateKind.FUTURE_PLANNING_RULE,
                function_id=function_id,
                content=rule,
                evidence_refs=evidence_refs,
                quarantine_recommended=quarantine_recommended,
            )
        )

    for index, item in enumerate(result.uncertainty_ledger.open_items(), start=1):
        updates.append(
            KernelMemoryUpdate(
                memory_id=f"{result.trial_id}:memory:uncertainty:{index}",
                kind=KernelMemoryUpdateKind.UNCERTAINTY_NOTE,
                function_id=function_id,
                content=f"{item.severity.value}: {item.statement}",
                evidence_refs=(item.uncertainty_id, *evidence_refs),
                quarantine_recommended=(
                    item.severity is UncertaintySeverity.BLOCKING
                    or quarantine_recommended
                ),
            )
        )

    return tuple(updates)


def build_kernel_skill_candidate(
    *,
    result: TransferTrialResult,
    evidence_refs: tuple[str, ...],
) -> KernelSkillCandidate:
    """Build a bounded skill candidate for future cross-domain reuse."""

    allowed_for_reuse = (
        result.permits_bounded_evidence()
        and result.learning_update.disposition in {
            LearningDisposition.PROMOTE,
            LearningDisposition.RETAIN,
        }
    )
    if not allowed_for_reuse:
        return KernelSkillCandidate(
            skill_id=f"{result.trial_id}:skill-candidate",
            function_id=result.learning_update.function_id,
            trial_id=result.trial_id,
            allowed_for_reuse=False,
            reuse_conditions=(),
            evidence_refs=evidence_refs,
            blocking_reason=(
                "Trial did not permit bounded reuse or learning disposition "
                "was not promote/retain."
            ),
        )

    return KernelSkillCandidate(
        skill_id=f"{result.trial_id}:skill-candidate",
        function_id=result.learning_update.function_id,
        trial_id=result.trial_id,
        allowed_for_reuse=True,
        reuse_conditions=(
            "Require source-target domain separation.",
            "Require target mapping coverage before prediction.",
            "Require a committed pre-outcome prediction.",
            "Require reality-delta scoring before confidence increase.",
            "Preserve uncertainty and falsification records.",
        ),
        evidence_refs=evidence_refs,
        blocking_reason=None,
    )


def evidence_reference_tuple(
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> tuple[str, ...]:
    """Return stable evidence references for Kernel handoff records."""

    return (
        evidence_packet.packet_id,
        evidence_packet.manifest_sha256_digest,
        result.source_evidence.evidence_id,
        result.mapping.function_id,
        result.prediction_readiness.prediction_id,
        result.reality_delta.report_id,
        result.learning_update.update_id,
        result.uncertainty_ledger.ledger_id,
        result.falsification_ledger.ledger_id,
        result.negative_control_suite.suite_id,
    )


def required_kernel_actions(
    result: TransferTrialResult,
    status: KernelHandoffStatus,
) -> tuple[str, ...]:
    """Return required Kernel-side actions for the handoff packet."""

    if status is KernelHandoffStatus.READY_FOR_REVIEW:
        return (
            "Ingest as candidate evidence only, not truth.",
            "Bind confidence update to the cited evidence packet digest.",
            "Carry uncertainty notes into Kernel memory.",
            "Require a new prediction before future cross-domain reuse.",
        )

    if status is KernelHandoffStatus.QUARANTINED:
        return (
            "Quarantine the transfer pattern.",
            "Block automatic future reuse.",
            "Require retesting before any confidence increase.",
            *result.required_actions,
        )

    return (
        "Do not strengthen Kernel belief from this trial.",
        "Preserve failure and uncertainty records.",
        "Require human review before any reuse attempt.",
        *result.required_actions,
    )


def kernel_claim_boundary() -> str:
    """Return the fixed claim boundary for Kernel handoff artifacts."""

    return (
        "IX-Function Kernel handoff is bounded causal-transfer evidence. It is "
        "not truth, not AGI proof, not deployment authority, and not automatic "
        "memory promotion."
    )


def validate_kernel_handoff_packet(
    packet: KernelHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors for a Kernel handoff packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not packet.evidence_packet_id.strip():
        errors.append("evidence_packet_id must not be empty")
    if packet.belief_update.trial_id != packet.trial_id:
        errors.append("belief_update trial_id must match packet trial_id")
    if not packet.memory_updates:
        errors.append("at least one memory update is required")
    if packet.skill_candidate.trial_id != packet.trial_id:
        errors.append("skill_candidate trial_id must match packet trial_id")
    if not packet.required_kernel_actions:
        errors.append("required_kernel_actions must not be empty")
    if packet.claim_boundary != kernel_claim_boundary():
        errors.append("claim_boundary must match fixed Kernel boundary")

    memory_ids = [memory.memory_id for memory in packet.memory_updates]
    if len(set(memory_ids)) != len(memory_ids):
        errors.append("memory updates must use unique memory_id values")

    for memory in packet.memory_updates:
        if not memory.memory_id.strip():
            errors.append("memory_id must not be empty")
        if not memory.content.strip():
            errors.append(f"content must not be empty for {memory.memory_id!r}")
        if not memory.evidence_refs:
            errors.append(
                f"evidence_refs must not be empty for {memory.memory_id!r}"
            )

    if not packet.belief_update.evidence_refs:
        errors.append("belief_update evidence_refs must not be empty")
    if not packet.skill_candidate.evidence_refs:
        errors.append("skill_candidate evidence_refs must not be empty")

    return tuple(errors)
