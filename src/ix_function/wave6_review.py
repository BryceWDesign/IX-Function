"""Independent replication and Wave 6 review gate for IX-Function.

This module is the final code-level guard before README/public positioning.
It decides whether an IX-Function trial can enter bounded Wave 6 review, needs
additional replication/model review, or must be blocked.

The gate does not prove AGI. It only checks whether the evidence chain is
complete enough for serious external review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket, validate_evidence_packet
from ix_function.handoff_bundle import (
    IntegratedDonorHandoffBundle,
    validate_integrated_handoff_bundle,
)
from ix_function.model_review import (
    ModelProviderReviewReport,
    ModelReviewGateStatus,
    validate_model_provider_review_report,
)
from ix_function.trial import TransferTrialResult


class ReplicationStepKind(StrEnum):
    """Kinds of steps required for independent replication."""

    ARTIFACT_VALIDATION = "artifact_validation"
    CLAIM_BOUNDARY_CHECK = "claim_boundary_check"
    EVIDENCE_REPLAY = "evidence_replay"
    FALSIFICATION_REPLAY = "falsification_replay"
    MODEL_REVIEW_REPLAY = "model_review_replay"
    NEGATIVE_CONTROL_REPLAY = "negative_control_replay"


class ReplicationReadinessStatus(StrEnum):
    """Independent replication readiness status."""

    READY_FOR_EXTERNAL_REPLAY = "ready_for_external_replay"
    NEEDS_MODEL_REVIEW = "needs_model_review"
    BLOCKED = "blocked"


class Wave6ReviewDecision(StrEnum):
    """Final bounded Wave 6 review decision."""

    READY_FOR_BOUNDED_WAVE6_REVIEW = "ready_for_bounded_wave6_review"
    REQUIRE_INDEPENDENT_REPLICATION = "require_independent_replication"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ReplicationStep:
    """A deterministic step an outside reviewer should be able to replay."""

    step_id: str
    kind: ReplicationStepKind
    statement: str
    required_evidence_refs: tuple[str, ...]
    expected_result: str
    kill_if_missing: bool


@dataclass(frozen=True, slots=True)
class IndependentReplicationPacket:
    """Replay packet for independent IX-Function replication."""

    packet_id: str
    trial_id: str
    status: ReplicationReadinessStatus
    evidence_packet_id: str
    evidence_manifest_digest: str
    integrated_bundle_id: str
    model_review_report_id: str
    replication_steps: tuple[ReplicationStep, ...]
    expected_outputs: tuple[str, ...]
    kill_criteria: tuple[str, ...]
    required_reviewer_actions: tuple[str, ...]
    claim_boundary: str

    def is_ready_for_external_replay(self) -> bool:
        """Return whether this packet is ready for external replay."""

        return self.status is ReplicationReadinessStatus.READY_FOR_EXTERNAL_REPLAY


@dataclass(frozen=True, slots=True)
class Wave6ReviewGateResult:
    """Final IX-Function Wave 6 review gate result."""

    gate_id: str
    trial_id: str
    decision: Wave6ReviewDecision
    replication_packet_id: str
    integrated_bundle_id: str
    model_review_report_id: str
    allowed_claims: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    validation_errors: tuple[str, ...]
    required_actions: tuple[str, ...]
    claim_boundary: str

    def permits_bounded_wave6_review(self) -> bool:
        """Return whether bounded Wave 6 review language is permitted."""

        return (
            self.decision is Wave6ReviewDecision.READY_FOR_BOUNDED_WAVE6_REVIEW
            and not self.validation_errors
        )


def build_independent_replication_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
    integrated_bundle: IntegratedDonorHandoffBundle,
    model_review_report: ModelProviderReviewReport,
) -> IndependentReplicationPacket:
    """Build an independent replication packet from the final evidence chain."""

    status = choose_replication_readiness_status(
        result=result,
        integrated_bundle=integrated_bundle,
        model_review_report=model_review_report,
    )

    return IndependentReplicationPacket(
        packet_id=f"{result.trial_id}:independent-replication",
        trial_id=result.trial_id,
        status=status,
        evidence_packet_id=evidence_packet.packet_id,
        evidence_manifest_digest=evidence_packet.manifest_sha256_digest,
        integrated_bundle_id=integrated_bundle.bundle_id,
        model_review_report_id=model_review_report.report_id,
        replication_steps=build_replication_steps(
            result=result,
            evidence_packet=evidence_packet,
            integrated_bundle=integrated_bundle,
            model_review_report=model_review_report,
        ),
        expected_outputs=expected_replication_outputs(result),
        kill_criteria=replication_kill_criteria(),
        required_reviewer_actions=required_replication_actions(status),
        claim_boundary=replication_claim_boundary(),
    )


def choose_replication_readiness_status(
    *,
    result: TransferTrialResult,
    integrated_bundle: IntegratedDonorHandoffBundle,
    model_review_report: ModelProviderReviewReport,
) -> ReplicationReadinessStatus:
    """Choose independent replication readiness status."""

    if not result.permits_bounded_evidence():
        return ReplicationReadinessStatus.BLOCKED
    if not integrated_bundle.is_ready_for_wave6_review():
        return ReplicationReadinessStatus.BLOCKED
    if model_review_report.status is ModelReviewGateStatus.BLOCKED:
        return ReplicationReadinessStatus.BLOCKED
    if model_review_report.status is not ModelReviewGateStatus.READY_FOR_HUMAN_REVIEW:
        return ReplicationReadinessStatus.NEEDS_MODEL_REVIEW
    return ReplicationReadinessStatus.READY_FOR_EXTERNAL_REPLAY


def build_replication_steps(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
    integrated_bundle: IntegratedDonorHandoffBundle,
    model_review_report: ModelProviderReviewReport,
) -> tuple[ReplicationStep, ...]:
    """Build replay steps required for outside review."""

    return (
        ReplicationStep(
            step_id=f"{result.trial_id}:replication:artifact-validation",
            kind=ReplicationStepKind.ARTIFACT_VALIDATION,
            statement=(
                "Validate evidence packet artifact digests and manifest digest "
                "before interpreting any result."
            ),
            required_evidence_refs=(
                evidence_packet.packet_id,
                evidence_packet.manifest_sha256_digest,
            ),
            expected_result="Evidence packet validates without digest mismatch.",
            kill_if_missing=True,
        ),
        ReplicationStep(
            step_id=f"{result.trial_id}:replication:evidence-replay",
            kind=ReplicationStepKind.EVIDENCE_REPLAY,
            statement=(
                "Replay source learning, mapping, pre-outcome prediction, "
                "reality-delta scoring, and learning update."
            ),
            required_evidence_refs=(
                result.source_evidence.evidence_id,
                result.mapping.function_id,
                result.prediction_readiness.prediction_id,
                result.reality_delta.report_id,
                result.learning_update.update_id,
            ),
            expected_result=(
                f"Trial status remains {result.status.value!r} with reality-delta "
                f"status {result.reality_delta.status.value!r}."
            ),
            kill_if_missing=True,
        ),
        ReplicationStep(
            step_id=f"{result.trial_id}:replication:falsification",
            kind=ReplicationStepKind.FALSIFICATION_REPLAY,
            statement=(
                "Replay falsification criteria and confirm kill criteria can "
                "block or downgrade the claim."
            ),
            required_evidence_refs=(
                result.falsification_ledger.ledger_id,
                result.falsification_ledger.verdict.value,
            ),
            expected_result=(
                "Falsification verdict is preserved and cannot be overridden by "
                "model output."
            ),
            kill_if_missing=True,
        ),
        ReplicationStep(
            step_id=f"{result.trial_id}:replication:negative-controls",
            kind=ReplicationStepKind.NEGATIVE_CONTROL_REPLAY,
            statement=(
                "Replay negative-control results and confirm bad evidence is "
                "not promoted."
            ),
            required_evidence_refs=(result.negative_control_suite.suite_id,),
            expected_result=(
                "Negative-control suite has no failed controls for this review "
                "packet."
            ),
            kill_if_missing=True,
        ),
        ReplicationStep(
            step_id=f"{result.trial_id}:replication:model-review",
            kind=ReplicationStepKind.MODEL_REVIEW_REPLAY,
            statement=(
                "Replay model-provider review and confirm outputs remain bounded, "
                "multi-provider, uncertainty-aware, and human-review-bound."
            ),
            required_evidence_refs=(
                model_review_report.report_id,
                *(
                    review.output_id
                    for review in model_review_report.reviews
                ),
            ),
            expected_result="Model-provider review is ready for human review.",
            kill_if_missing=True,
        ),
        ReplicationStep(
            step_id=f"{result.trial_id}:replication:claim-boundary",
            kind=ReplicationStepKind.CLAIM_BOUNDARY_CHECK,
            statement=(
                "Check every donor handoff and the integrated bundle for fixed "
                "anti-overclaim boundaries."
            ),
            required_evidence_refs=(
                integrated_bundle.bundle_id,
                integrated_bundle.kernel_handoff.packet_id,
                integrated_bundle.worldtwin_handoff.packet_id,
                integrated_bundle.intent_loop_handoff.packet_id,
                integrated_bundle.blackfox_handoff.packet_id,
                integrated_bundle.assurance_handoff.packet_id,
                integrated_bundle.ix_contract_handoff.packet_id,
            ),
            expected_result=(
                "All handoff boundaries block AGI proof, deployment authority, "
                "production readiness, and self-approval claims."
            ),
            kill_if_missing=True,
        ),
    )


def build_wave6_review_gate(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
    integrated_bundle: IntegratedDonorHandoffBundle,
    model_review_report: ModelProviderReviewReport,
) -> Wave6ReviewGateResult:
    """Build final Wave 6 review gate result."""

    replication_packet = build_independent_replication_packet(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_review_report,
    )
    validation_errors = wave6_validation_errors(
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_review_report,
        replication_packet=replication_packet,
    )
    decision = choose_wave6_review_decision(
        result=result,
        replication_packet=replication_packet,
        model_review_report=model_review_report,
        validation_errors=validation_errors,
    )

    return Wave6ReviewGateResult(
        gate_id=f"{result.trial_id}:wave6-review-gate",
        trial_id=result.trial_id,
        decision=decision,
        replication_packet_id=replication_packet.packet_id,
        integrated_bundle_id=integrated_bundle.bundle_id,
        model_review_report_id=model_review_report.report_id,
        allowed_claims=allowed_wave6_claims(decision),
        blocked_claims=blocked_wave6_claims(),
        validation_errors=validation_errors,
        required_actions=required_wave6_actions(
            decision=decision,
            validation_errors=validation_errors,
            replication_packet=replication_packet,
            model_review_report=model_review_report,
            result=result,
        ),
        claim_boundary=wave6_review_claim_boundary(),
    )


def choose_wave6_review_decision(
    *,
    result: TransferTrialResult,
    replication_packet: IndependentReplicationPacket,
    model_review_report: ModelProviderReviewReport,
    validation_errors: tuple[str, ...],
) -> Wave6ReviewDecision:
    """Choose final bounded Wave 6 review decision."""

    if validation_errors:
        return Wave6ReviewDecision.BLOCKED
    if not result.permits_bounded_evidence():
        return Wave6ReviewDecision.BLOCKED
    if model_review_report.status is ModelReviewGateStatus.BLOCKED:
        return Wave6ReviewDecision.BLOCKED
    if not replication_packet.is_ready_for_external_replay():
        return Wave6ReviewDecision.REQUIRE_INDEPENDENT_REPLICATION
    return Wave6ReviewDecision.READY_FOR_BOUNDED_WAVE6_REVIEW


def wave6_validation_errors(
    *,
    evidence_packet: EvidencePacket,
    integrated_bundle: IntegratedDonorHandoffBundle,
    model_review_report: ModelProviderReviewReport,
    replication_packet: IndependentReplicationPacket,
) -> tuple[str, ...]:
    """Return validation errors across final Wave 6 review inputs."""

    errors: list[str] = []
    errors.extend(_prefix_errors("evidence", validate_evidence_packet(evidence_packet)))
    errors.extend(
        _prefix_errors(
            "integrated_bundle",
            validate_integrated_handoff_bundle(integrated_bundle),
        )
    )
    errors.extend(
        _prefix_errors(
            "model_review",
            validate_model_provider_review_report(model_review_report),
        )
    )
    errors.extend(
        _prefix_errors(
            "replication",
            validate_independent_replication_packet(replication_packet),
        )
    )

    if integrated_bundle.evidence_packet_id != evidence_packet.packet_id:
        errors.append("wave6: integrated bundle evidence_packet_id mismatch")
    if model_review_report.trial_id != integrated_bundle.trial_id:
        errors.append("wave6: model review trial_id mismatch")
    if replication_packet.trial_id != integrated_bundle.trial_id:
        errors.append("wave6: replication packet trial_id mismatch")

    return tuple(errors)


def validate_independent_replication_packet(
    packet: IndependentReplicationPacket,
) -> tuple[str, ...]:
    """Return validation errors for an independent replication packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not packet.evidence_packet_id.strip():
        errors.append("evidence_packet_id must not be empty")
    if not packet.evidence_manifest_digest.strip():
        errors.append("evidence_manifest_digest must not be empty")
    if not packet.integrated_bundle_id.strip():
        errors.append("integrated_bundle_id must not be empty")
    if not packet.model_review_report_id.strip():
        errors.append("model_review_report_id must not be empty")
    if not packet.replication_steps:
        errors.append("at least one replication step is required")
    if not packet.expected_outputs:
        errors.append("expected_outputs must not be empty")
    if not packet.kill_criteria:
        errors.append("kill_criteria must not be empty")
    if not packet.required_reviewer_actions:
        errors.append("required_reviewer_actions must not be empty")
    if packet.claim_boundary != replication_claim_boundary():
        errors.append("claim_boundary must match fixed replication boundary")

    step_ids = [step.step_id for step in packet.replication_steps]
    if len(set(step_ids)) != len(step_ids):
        errors.append("replication steps must use unique step_id values")

    for step in packet.replication_steps:
        if not step.step_id.strip():
            errors.append("step_id must not be empty")
        if not step.statement.strip():
            errors.append(f"statement must not be empty for {step.step_id!r}")
        if not step.required_evidence_refs:
            errors.append(
                f"required_evidence_refs must not be empty for {step.step_id!r}"
            )
        if not step.expected_result.strip():
            errors.append(
                f"expected_result must not be empty for {step.step_id!r}"
            )

    return tuple(errors)


def validate_wave6_review_gate_result(
    result: Wave6ReviewGateResult,
) -> tuple[str, ...]:
    """Return validation errors for a Wave 6 review gate result."""

    errors: list[str] = []
    if not result.gate_id.strip():
        errors.append("gate_id must not be empty")
    if not result.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not result.replication_packet_id.strip():
        errors.append("replication_packet_id must not be empty")
    if not result.integrated_bundle_id.strip():
        errors.append("integrated_bundle_id must not be empty")
    if not result.model_review_report_id.strip():
        errors.append("model_review_report_id must not be empty")
    if not result.allowed_claims:
        errors.append("allowed_claims must not be empty")
    if not result.blocked_claims:
        errors.append("blocked_claims must not be empty")
    if not result.required_actions:
        errors.append("required_actions must not be empty")
    if result.claim_boundary != wave6_review_claim_boundary():
        errors.append("claim_boundary must match fixed Wave 6 boundary")
    if (
        result.decision is Wave6ReviewDecision.READY_FOR_BOUNDED_WAVE6_REVIEW
        and result.validation_errors
    ):
        errors.append("ready Wave 6 review gate must not contain validation_errors")

    return tuple(errors)


def expected_replication_outputs(result: TransferTrialResult) -> tuple[str, ...]:
    """Return expected outputs an external replay should reproduce."""

    return (
        f"trial_status={result.status.value}",
        f"reality_delta_status={result.reality_delta.status.value}",
        f"learning_disposition={result.learning_update.disposition.value}",
        f"falsification_verdict={result.falsification_ledger.verdict.value}",
        f"anti_theater_allowed={result.anti_theater_gate.allowed}",
    )


def replication_kill_criteria() -> tuple[str, ...]:
    """Return final independent replication kill criteria."""

    return (
        "Evidence packet digest mismatch.",
        "Prediction is not committed before outcome observation.",
        "Reality-delta score cannot be reproduced.",
        "Falsification ledger is missing or bypassed.",
        "Negative controls fail to reject bad evidence.",
        (
            "Model-provider output overclaims AGI proof, deployment, or "
            "production readiness."
        ),
        "Human review boundary is missing.",
        "Uncertainty is omitted or laundered into certainty.",
    )


def required_replication_actions(
    status: ReplicationReadinessStatus,
) -> tuple[str, ...]:
    """Return required actions for replication readiness status."""

    if status is ReplicationReadinessStatus.READY_FOR_EXTERNAL_REPLAY:
        return (
            "Send packet for independent replay.",
            "Require reviewer to reproduce expected outputs.",
            "Record any mismatch as falsification pressure.",
            "Do not represent replay readiness as AGI proof.",
        )

    if status is ReplicationReadinessStatus.NEEDS_MODEL_REVIEW:
        return (
            "Collect clean multi-provider model review before external replay.",
            "Block overclaiming provider outputs.",
            "Require uncertainty and human-review acknowledgement.",
        )

    return (
        "Block independent replay package until trial and handoff gates pass.",
        "Preserve failure evidence for review.",
        "Do not promote bounded Wave 6 language.",
    )


def required_wave6_actions(
    *,
    decision: Wave6ReviewDecision,
    validation_errors: tuple[str, ...],
    replication_packet: IndependentReplicationPacket,
    model_review_report: ModelProviderReviewReport,
    result: TransferTrialResult,
) -> tuple[str, ...]:
    """Return required actions for the final Wave 6 review decision."""

    if validation_errors:
        return tuple(
            f"Resolve Wave 6 validation error: {error}"
            for error in validation_errors
        )

    if decision is Wave6ReviewDecision.READY_FOR_BOUNDED_WAVE6_REVIEW:
        return (
            "Permit bounded Wave 6 review language only.",
            "Submit evidence packet and donor handoffs for independent replay.",
            "Bind reviewer decisions to evidence digests.",
            "Preserve uncertainty, falsification, negative controls, and model review.",
            "Do not claim AGI proof.",
        )

    if decision is Wave6ReviewDecision.REQUIRE_INDEPENDENT_REPLICATION:
        return (
            *replication_packet.required_reviewer_actions,
            *model_review_report.required_actions,
            "Do not advance to bounded Wave 6 review until these are resolved.",
        )

    return (
        "Block bounded Wave 6 review language.",
        *result.required_actions,
        *model_review_report.required_actions,
    )


def allowed_wave6_claims(
    decision: Wave6ReviewDecision,
) -> tuple[str, ...]:
    """Return allowed public/review claims for a Wave 6 decision."""

    if decision is Wave6ReviewDecision.READY_FOR_BOUNDED_WAVE6_REVIEW:
        return (
            "bounded Wave 6 review candidate evidence",
            "cross-domain causal-transfer trial evidence",
            "independent replay package ready for review",
        )

    if decision is Wave6ReviewDecision.REQUIRE_INDEPENDENT_REPLICATION:
        return (
            "pre-review evidence package",
            "replication required before Wave 6 review",
        )

    return (
        "blocked evidence record",
        "failure or invalid review artifact",
    )


def blocked_wave6_claims() -> tuple[str, ...]:
    """Return blocked claims for every Wave 6 gate result."""

    return (
        "AGI proof",
        "certified AGI",
        "independent AGI validation",
        "deployment authority",
        "production readiness",
        "self-approval",
        "unsupervised operational autonomy",
    )


def replication_claim_boundary() -> str:
    """Return fixed independent replication boundary."""

    return (
        "IX-Function independent replication packet is replay readiness evidence. "
        "It is not AGI proof, not independent validation by itself, not deployment "
        "authority, and not self-approval."
    )


def wave6_review_claim_boundary() -> str:
    """Return fixed Wave 6 review gate boundary."""

    return (
        "IX-Function Wave 6 review gate may allow bounded review-candidate "
        "language only. It is not AGI proof, not independent validation, not "
        "deployment authority, and not production readiness."
    )


def _prefix_errors(prefix: str, errors: tuple[str, ...]) -> tuple[str, ...]:
    """Prefix validation errors with their review section."""

    return tuple(f"{prefix}: {error}" for error in errors)
