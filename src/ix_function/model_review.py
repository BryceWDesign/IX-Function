"""Model-provider review gate for IX-Function.

IX-Function treats every model output as untrusted interpretation. A provider
can summarize, critique, or propose reuse, but it cannot overrule evidence,
falsification, uncertainty, human review, or claim boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.trial import TransferTrialResult


class ModelProviderKind(StrEnum):
    """Provider families used for model-output review."""

    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    OPENAI = "openai"
    OTHER = "other"


class ModelOutputClaimKind(StrEnum):
    """Claim kinds a model may attempt to make about IX-Function evidence."""

    AGI_PROOF = "agi_proof"
    BOUNDED_TRANSFER_EVIDENCE = "bounded_transfer_evidence"
    DEPLOYMENT_AUTHORITY = "deployment_authority"
    FAILURE_INTERPRETATION = "failure_interpretation"
    FUTURE_REUSE_GUIDANCE = "future_reuse_guidance"
    PRODUCTION_READINESS = "production_readiness"


class ModelReviewDecision(StrEnum):
    """Decision assigned to a model output."""

    ACCEPT_BOUNDED = "accept_bounded"
    BLOCK = "block"
    REVISE_REQUIRED = "revise_required"


class ModelReviewGateStatus(StrEnum):
    """Aggregate status for model-provider review."""

    BLOCKED = "blocked"
    READY_FOR_HUMAN_REVIEW = "ready_for_human_review"
    REVISE_REQUIRED = "revise_required"


@dataclass(frozen=True, slots=True)
class ModelProviderOutput:
    """One model-provider interpretation of an IX-Function evidence packet."""

    output_id: str
    provider_kind: ModelProviderKind
    provider_name: str
    model_name: str
    prompt_id: str
    response_text: str
    declared_claims: tuple[ModelOutputClaimKind, ...]
    cited_evidence_refs: tuple[str, ...]
    uncertainty_acknowledged: bool
    human_review_acknowledged: bool


@dataclass(frozen=True, slots=True)
class ModelOutputReview:
    """Review result for a single model-provider output."""

    output_id: str
    provider_kind: ModelProviderKind
    model_name: str
    decision: ModelReviewDecision
    blocking_reasons: tuple[str, ...]
    revision_reasons: tuple[str, ...]
    accepted_evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelProviderReviewReport:
    """Aggregate review report across one or more model-provider outputs."""

    report_id: str
    trial_id: str
    status: ModelReviewGateStatus
    reviews: tuple[ModelOutputReview, ...]
    provider_kinds: tuple[ModelProviderKind, ...]
    multi_provider_coverage: bool
    required_actions: tuple[str, ...]
    claim_boundary: str

    def accepted_reviews(self) -> tuple[ModelOutputReview, ...]:
        """Return model outputs accepted as bounded review input."""

        return tuple(
            review
            for review in self.reviews
            if review.decision is ModelReviewDecision.ACCEPT_BOUNDED
        )

    def blocked_reviews(self) -> tuple[ModelOutputReview, ...]:
        """Return blocked model outputs."""

        return tuple(
            review
            for review in self.reviews
            if review.decision is ModelReviewDecision.BLOCK
        )


BLOCKED_RESPONSE_TERMS: tuple[str, ...] = (
    "agi proven",
    "artificial general intelligence proven",
    "certified agi",
    "deployment authorized",
    "independent agi validation",
    "production ready",
    "self-approved",
    "unsupervised autonomy approved",
)

BLOCKED_CLAIM_KINDS: frozenset[ModelOutputClaimKind] = frozenset(
    {
        ModelOutputClaimKind.AGI_PROOF,
        ModelOutputClaimKind.DEPLOYMENT_AUTHORITY,
        ModelOutputClaimKind.PRODUCTION_READINESS,
    }
)


def review_model_output(
    *,
    output: ModelProviderOutput,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> ModelOutputReview:
    """Review one model output against deterministic IX-Function evidence."""

    blocking_reasons: list[str] = []
    revision_reasons: list[str] = []

    blocking_reasons.extend(validate_model_output_shape(output))
    blocking_reasons.extend(blocked_claim_reasons(output))
    blocking_reasons.extend(blocked_response_term_reasons(output.response_text))

    known_refs = known_evidence_refs(result=result, evidence_packet=evidence_packet)
    cited_known_refs = tuple(
        ref for ref in output.cited_evidence_refs if ref in known_refs
    )
    unknown_refs = tuple(
        ref for ref in output.cited_evidence_refs if ref not in known_refs
    )
    if unknown_refs:
        revision_reasons.append(
            "model output cited unknown evidence refs: " + ", ".join(unknown_refs)
        )

    required_refs = required_model_evidence_refs(
        result=result,
        evidence_packet=evidence_packet,
    )
    missing_required_refs = tuple(
        ref for ref in required_refs if ref not in cited_known_refs
    )
    if missing_required_refs:
        revision_reasons.append(
            "model output omitted required evidence refs: "
            + ", ".join(missing_required_refs)
        )

    if not output.uncertainty_acknowledged:
        revision_reasons.append("model output did not acknowledge uncertainty")
    if not output.human_review_acknowledged:
        revision_reasons.append("model output did not acknowledge human review")
    if (
        ModelOutputClaimKind.BOUNDED_TRANSFER_EVIDENCE
        not in output.declared_claims
        and result.permits_bounded_evidence()
    ):
        revision_reasons.append(
            "model output did not declare bounded transfer evidence language"
        )

    if blocking_reasons:
        decision = ModelReviewDecision.BLOCK
    elif revision_reasons:
        decision = ModelReviewDecision.REVISE_REQUIRED
    else:
        decision = ModelReviewDecision.ACCEPT_BOUNDED

    return ModelOutputReview(
        output_id=output.output_id,
        provider_kind=output.provider_kind,
        model_name=output.model_name,
        decision=decision,
        blocking_reasons=tuple(blocking_reasons),
        revision_reasons=tuple(revision_reasons),
        accepted_evidence_refs=cited_known_refs,
    )


def build_model_provider_review_report(
    *,
    report_id: str,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
    outputs: tuple[ModelProviderOutput, ...],
    required_provider_count: int = 2,
) -> ModelProviderReviewReport:
    """Build a multi-provider review report for model interpretations."""

    reviews = tuple(
        review_model_output(
            output=output,
            result=result,
            evidence_packet=evidence_packet,
        )
        for output in outputs
    )
    provider_kinds = tuple(sorted({output.provider_kind for output in outputs}))
    multi_provider_coverage = len(provider_kinds) >= required_provider_count
    status = choose_model_review_gate_status(
        reviews=reviews,
        has_outputs=bool(outputs),
        multi_provider_coverage=multi_provider_coverage,
    )

    return ModelProviderReviewReport(
        report_id=report_id,
        trial_id=result.trial_id,
        status=status,
        reviews=reviews,
        provider_kinds=provider_kinds,
        multi_provider_coverage=multi_provider_coverage,
        required_actions=required_model_review_actions(
            status=status,
            reviews=reviews,
            multi_provider_coverage=multi_provider_coverage,
        ),
        claim_boundary=model_review_claim_boundary(),
    )


def choose_model_review_gate_status(
    *,
    reviews: tuple[ModelOutputReview, ...],
    has_outputs: bool,
    multi_provider_coverage: bool,
) -> ModelReviewGateStatus:
    """Choose aggregate model-provider review status."""

    if not has_outputs:
        return ModelReviewGateStatus.BLOCKED
    if any(review.decision is ModelReviewDecision.BLOCK for review in reviews):
        return ModelReviewGateStatus.BLOCKED
    if (
        not multi_provider_coverage
        or any(
            review.decision is ModelReviewDecision.REVISE_REQUIRED
            for review in reviews
        )
    ):
        return ModelReviewGateStatus.REVISE_REQUIRED
    return ModelReviewGateStatus.READY_FOR_HUMAN_REVIEW


def known_evidence_refs(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> frozenset[str]:
    """Return evidence refs model outputs are allowed to cite."""

    artifact_refs = tuple(
        artifact.artifact_id for artifact in evidence_packet.artifacts
    )
    return frozenset(
        (
            evidence_packet.packet_id,
            evidence_packet.manifest_sha256_digest,
            result.trial_id,
            result.source_evidence.evidence_id,
            result.mapping.function_id,
            result.mapping.target_domain_id,
            result.prediction_readiness.prediction_id,
            result.reality_delta.report_id,
            result.learning_update.update_id,
            result.uncertainty_ledger.ledger_id,
            result.falsification_ledger.ledger_id,
            result.negative_control_suite.suite_id,
            *artifact_refs,
        )
    )


def required_model_evidence_refs(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> tuple[str, ...]:
    """Return evidence refs every acceptable model review must cite."""

    return (
        evidence_packet.packet_id,
        evidence_packet.manifest_sha256_digest,
        result.reality_delta.report_id,
        result.uncertainty_ledger.ledger_id,
        result.falsification_ledger.ledger_id,
    )


def validate_model_output_shape(output: ModelProviderOutput) -> tuple[str, ...]:
    """Return structural validation errors for a model output."""

    errors: list[str] = []
    if not output.output_id.strip():
        errors.append("output_id must not be empty")
    if not output.provider_name.strip():
        errors.append("provider_name must not be empty")
    if not output.model_name.strip():
        errors.append("model_name must not be empty")
    if not output.prompt_id.strip():
        errors.append("prompt_id must not be empty")
    if not output.response_text.strip():
        errors.append("response_text must not be empty")
    if not output.declared_claims:
        errors.append("declared_claims must not be empty")
    if not output.cited_evidence_refs:
        errors.append("cited_evidence_refs must not be empty")
    return tuple(errors)


def blocked_claim_reasons(output: ModelProviderOutput) -> tuple[str, ...]:
    """Return blocked reasons from declared model claim kinds."""

    blocked = tuple(
        claim for claim in output.declared_claims if claim in BLOCKED_CLAIM_KINDS
    )
    return tuple(
        f"model output declared blocked claim kind {claim.value!r}"
        for claim in blocked
    )


def blocked_response_term_reasons(response_text: str) -> tuple[str, ...]:
    """Return blocked reasons from forbidden response text."""

    normalized = " ".join(response_text.lower().split())
    return tuple(
        f"model output used blocked response term {term!r}"
        for term in BLOCKED_RESPONSE_TERMS
        if term in normalized
    )


def required_model_review_actions(
    *,
    status: ModelReviewGateStatus,
    reviews: tuple[ModelOutputReview, ...],
    multi_provider_coverage: bool,
) -> tuple[str, ...]:
    """Return required actions for aggregate model-provider review."""

    if status is ModelReviewGateStatus.READY_FOR_HUMAN_REVIEW:
        return (
            "Route accepted bounded model interpretations to human review.",
            "Bind review to cited evidence refs and packet digest.",
            "Do not allow any provider output to override falsification gates.",
        )

    actions: list[str] = []
    if not reviews:
        actions.append("Collect at least one model-provider interpretation.")
    if not multi_provider_coverage:
        actions.append("Collect review outputs from at least two provider kinds.")

    for review in reviews:
        for reason in review.blocking_reasons:
            actions.append(f"Block {review.output_id}: {reason}")
        for reason in review.revision_reasons:
            actions.append(f"Revise {review.output_id}: {reason}")

    if not actions:
        actions.append("Revise model-provider review before human interpretation.")

    return tuple(actions)


def model_review_claim_boundary() -> str:
    """Return fixed model-review claim boundary."""

    return (
        "IX-Function model-provider review treats model outputs as untrusted "
        "interpretations. It does not grant AGI proof, deployment authority, "
        "production readiness, independent validation, or self-approval."
    )


def validate_model_provider_review_report(
    report: ModelProviderReviewReport,
) -> tuple[str, ...]:
    """Return validation errors for a model-provider review report."""

    errors: list[str] = []
    if not report.report_id.strip():
        errors.append("report_id must not be empty")
    if not report.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not report.reviews:
        errors.append("at least one model output review is required")
    if not report.provider_kinds:
        errors.append("provider_kinds must not be empty")
    if not report.required_actions:
        errors.append("required_actions must not be empty")
    if report.claim_boundary != model_review_claim_boundary():
        errors.append("claim_boundary must match fixed model-review boundary")
    if (
        report.status is ModelReviewGateStatus.READY_FOR_HUMAN_REVIEW
        and not report.multi_provider_coverage
    ):
        errors.append("ready report must have multi-provider coverage")

    output_ids = [review.output_id for review in report.reviews]
    if len(set(output_ids)) != len(output_ids):
        errors.append("model output reviews must use unique output_id values")

    for review in report.reviews:
        if not review.output_id.strip():
            errors.append("review output_id must not be empty")
        if not review.model_name.strip():
            errors.append(f"model_name must not be empty for {review.output_id!r}")
        if (
            review.decision is ModelReviewDecision.BLOCK
            and not review.blocking_reasons
        ):
            errors.append(
                f"blocked review {review.output_id!r} must include blocking reasons"
            )
        if (
            review.decision is ModelReviewDecision.REVISE_REQUIRED
            and not review.revision_reasons
        ):
            errors.append(
                f"revision review {review.output_id!r} must include revision reasons"
            )

    return tuple(errors)
