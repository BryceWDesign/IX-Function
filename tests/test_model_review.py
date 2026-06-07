from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.model_review import (
    ModelOutputClaimKind,
    ModelProviderKind,
    ModelProviderOutput,
    ModelReviewDecision,
    ModelReviewGateStatus,
    blocked_response_term_reasons,
    build_model_provider_review_report,
    model_review_claim_boundary,
    required_model_evidence_refs,
    review_model_output,
    validate_model_output_shape,
    validate_model_provider_review_report,
)
from ix_function.trial import run_transfer_trial
from tests.fixtures import make_trial_input


def make_output(
    *,
    output_id: str = "model-output-001",
    provider_kind: ModelProviderKind = ModelProviderKind.OPENAI,
    response_text: str = "Bounded transfer evidence with uncertainty preserved.",
    declared_claims: tuple[ModelOutputClaimKind, ...] = (
        ModelOutputClaimKind.BOUNDED_TRANSFER_EVIDENCE,
    ),
) -> ModelProviderOutput:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    return ModelProviderOutput(
        output_id=output_id,
        provider_kind=provider_kind,
        provider_name=provider_kind.value,
        model_name=f"{provider_kind.value}-review-model",
        prompt_id="review-prompt-001",
        response_text=response_text,
        declared_claims=declared_claims,
        cited_evidence_refs=required_model_evidence_refs(
            result=result,
            evidence_packet=evidence_packet,
        ),
        uncertainty_acknowledged=True,
        human_review_acknowledged=True,
    )


def test_validate_model_output_shape_accepts_complete_output() -> None:
    assert validate_model_output_shape(make_output()) == ()


def test_validate_model_output_shape_rejects_empty_fields() -> None:
    output = ModelProviderOutput(
        output_id="",
        provider_kind=ModelProviderKind.OTHER,
        provider_name="",
        model_name="",
        prompt_id="",
        response_text="",
        declared_claims=(),
        cited_evidence_refs=(),
        uncertainty_acknowledged=False,
        human_review_acknowledged=False,
    )

    errors = validate_model_output_shape(output)

    assert "output_id must not be empty" in errors
    assert "provider_name must not be empty" in errors
    assert "declared_claims must not be empty" in errors
    assert "cited_evidence_refs must not be empty" in errors


def test_blocked_response_term_reasons_detects_overclaim_language() -> None:
    reasons = blocked_response_term_reasons(
        "This is AGI proven and production ready."
    )

    assert "model output used blocked response term 'agi proven'" in reasons
    assert "model output used blocked response term 'production ready'" in reasons


def test_review_model_output_accepts_bounded_interpretation() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    review = review_model_output(
        output=make_output(),
        result=result,
        evidence_packet=evidence_packet,
    )

    assert review.decision is ModelReviewDecision.ACCEPT_BOUNDED
    assert review.blocking_reasons == ()
    assert review.revision_reasons == ()
    assert evidence_packet.packet_id in review.accepted_evidence_refs


def test_review_model_output_blocks_declared_agi_claim() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    review = review_model_output(
        output=make_output(
            response_text="This is AGI proven.",
            declared_claims=(ModelOutputClaimKind.AGI_PROOF,),
        ),
        result=result,
        evidence_packet=evidence_packet,
    )

    assert review.decision is ModelReviewDecision.BLOCK
    assert any("blocked claim kind" in reason for reason in review.blocking_reasons)
    assert any("blocked response term" in reason for reason in review.blocking_reasons)


def test_review_model_output_requires_uncertainty_and_human_review() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    output = ModelProviderOutput(
        output_id="model-output-002",
        provider_kind=ModelProviderKind.ANTHROPIC,
        provider_name="anthropic",
        model_name="anthropic-review-model",
        prompt_id="review-prompt-001",
        response_text="Bounded transfer evidence.",
        declared_claims=(ModelOutputClaimKind.BOUNDED_TRANSFER_EVIDENCE,),
        cited_evidence_refs=(evidence_packet.packet_id,),
        uncertainty_acknowledged=False,
        human_review_acknowledged=False,
    )

    review = review_model_output(
        output=output,
        result=result,
        evidence_packet=evidence_packet,
    )

    assert review.decision is ModelReviewDecision.REVISE_REQUIRED
    assert "model output did not acknowledge uncertainty" in review.revision_reasons
    assert "model output did not acknowledge human review" in review.revision_reasons
    assert any(
        "omitted required evidence refs" in reason
        for reason in review.revision_reasons
    )


def test_multi_provider_report_ready_for_human_review_when_clean() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    outputs = (
        make_output(
            output_id="model-output-openai",
            provider_kind=ModelProviderKind.OPENAI,
        ),
        make_output(
            output_id="model-output-anthropic",
            provider_kind=ModelProviderKind.ANTHROPIC,
        ),
    )

    report = build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=outputs,
    )

    assert report.status is ModelReviewGateStatus.READY_FOR_HUMAN_REVIEW
    assert report.multi_provider_coverage
    assert len(report.accepted_reviews()) == 2
    assert report.blocked_reviews() == ()
    assert report.claim_boundary == model_review_claim_boundary()
    assert validate_model_provider_review_report(report) == ()


def test_multi_provider_report_requires_revision_for_single_provider() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    report = build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=(make_output(),),
    )

    assert report.status is ModelReviewGateStatus.REVISE_REQUIRED
    assert not report.multi_provider_coverage
    assert "Collect review outputs from at least two provider kinds." in (
        report.required_actions
    )


def test_multi_provider_report_blocks_overclaiming_provider() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    outputs = (
        make_output(
            output_id="model-output-openai",
            provider_kind=ModelProviderKind.OPENAI,
        ),
        make_output(
            output_id="model-output-google",
            provider_kind=ModelProviderKind.GOOGLE,
            response_text="This provides deployment authorized certainty.",
            declared_claims=(ModelOutputClaimKind.DEPLOYMENT_AUTHORITY,),
        ),
    )

    report = build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=outputs,
    )

    assert report.status is ModelReviewGateStatus.BLOCKED
    assert report.blocked_reviews()
    assert any(
        action.startswith("Block model-output-google")
        for action in report.required_actions
    )


def test_validate_model_provider_review_report_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    report = build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=(
            make_output(
                output_id="model-output-openai",
                provider_kind=ModelProviderKind.OPENAI,
            ),
            make_output(
                output_id="model-output-anthropic",
                provider_kind=ModelProviderKind.ANTHROPIC,
            ),
        ),
    )
    invalid_report = type(report)(
        report_id=report.report_id,
        trial_id=report.trial_id,
        status=report.status,
        reviews=report.reviews,
        provider_kinds=report.provider_kinds,
        multi_provider_coverage=report.multi_provider_coverage,
        required_actions=report.required_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_model_provider_review_report(invalid_report)

    assert "claim_boundary must match fixed model-review boundary" in errors
