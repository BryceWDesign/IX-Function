from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.handoff_bundle import build_integrated_handoff_bundle
from ix_function.model_review import (
    ModelOutputClaimKind,
    ModelProviderKind,
    ModelProviderOutput,
    build_model_provider_review_report,
    required_model_evidence_refs,
)
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.trial import TransferTrialInput, TransferTrialResult, run_transfer_trial
from ix_function.wave6_review import (
    ReplicationReadinessStatus,
    Wave6ReviewDecision,
    build_independent_replication_packet,
    build_wave6_review_gate,
    replication_claim_boundary,
    validate_independent_replication_packet,
    validate_wave6_review_gate_result,
    wave6_review_claim_boundary,
)
from tests.fixtures import make_trial_input


def make_model_output(
    *,
    result: TransferTrialResult,
    output_id: str,
    provider_kind: ModelProviderKind,
    response_text: str = "Bounded transfer evidence with uncertainty preserved.",
    declared_claims: tuple[ModelOutputClaimKind, ...] = (
        ModelOutputClaimKind.BOUNDED_TRANSFER_EVIDENCE,
    ),
) -> ModelProviderOutput:
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


def make_clean_model_report(result: TransferTrialResult):
    evidence_packet = build_trial_evidence_packet(result)
    return build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=(
            make_model_output(
                result=result,
                output_id="model-output-openai",
                provider_kind=ModelProviderKind.OPENAI,
            ),
            make_model_output(
                result=result,
                output_id="model-output-anthropic",
                provider_kind=ModelProviderKind.ANTHROPIC,
            ),
        ),
    )


def make_single_provider_model_report(result: TransferTrialResult):
    evidence_packet = build_trial_evidence_packet(result)
    return build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=(
            make_model_output(
                result=result,
                output_id="model-output-openai",
                provider_kind=ModelProviderKind.OPENAI,
            ),
        ),
    )


def make_overclaiming_model_report(result: TransferTrialResult):
    evidence_packet = build_trial_evidence_packet(result)
    return build_model_provider_review_report(
        report_id="trial-001:model-review",
        result=result,
        evidence_packet=evidence_packet,
        outputs=(
            make_model_output(
                result=result,
                output_id="model-output-openai",
                provider_kind=ModelProviderKind.OPENAI,
            ),
            make_model_output(
                result=result,
                output_id="model-output-google",
                provider_kind=ModelProviderKind.GOOGLE,
                response_text="This is AGI proven and production ready.",
                declared_claims=(ModelOutputClaimKind.AGI_PROOF,),
            ),
        ),
    )


def make_failed_result() -> TransferTrialResult:
    trial_input = make_trial_input()
    failed_input = TransferTrialInput(
        trial_id=trial_input.trial_id,
        domain_pair=trial_input.domain_pair,
        causal_function=trial_input.causal_function,
        source_baseline=trial_input.source_baseline,
        source_intervention=trial_input.source_intervention,
        source_outcome=trial_input.source_outcome,
        source_support_reasons=trial_input.source_support_reasons,
        source_uncertainty_notes=trial_input.source_uncertainty_notes,
        target_baseline=trial_input.target_baseline,
        target_intervention=trial_input.target_intervention,
        target_outcome=OutcomeRecord(
            domain_id="ci-pipeline",
            outcome_id="target-outcome-failed",
            observed_after_intervention_id="increase-worker-count",
            values=(
                MeasuredValue(
                    observable_name="Completion Throughput Output",
                    value=80.0,
                    evidence_id="target-outcome-failed-completion",
                ),
            ),
            result_summary="Completion changed far outside prediction.",
        ),
        prediction=trial_input.prediction,
    )
    return run_transfer_trial(failed_input)


def test_independent_replication_packet_ready_when_all_gates_are_clean() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_clean_model_report(result)

    packet = build_independent_replication_packet(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )

    assert packet.packet_id == "trial-001:independent-replication"
    assert packet.status is ReplicationReadinessStatus.READY_FOR_EXTERNAL_REPLAY
    assert packet.is_ready_for_external_replay()
    assert len(packet.replication_steps) == 6
    assert "AGI proof" in packet.kill_criteria[-3]
    assert packet.claim_boundary == replication_claim_boundary()
    assert validate_independent_replication_packet(packet) == ()


def test_independent_replication_packet_needs_model_review_for_single_provider() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_single_provider_model_report(result)

    packet = build_independent_replication_packet(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )

    assert packet.status is ReplicationReadinessStatus.NEEDS_MODEL_REVIEW
    assert not packet.is_ready_for_external_replay()


def test_wave6_review_gate_ready_when_replication_and_model_review_are_clean() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_clean_model_report(result)

    gate = build_wave6_review_gate(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )

    assert gate.gate_id == "trial-001:wave6-review-gate"
    assert gate.decision is Wave6ReviewDecision.READY_FOR_BOUNDED_WAVE6_REVIEW
    assert gate.permits_bounded_wave6_review()
    assert "AGI proof" in gate.blocked_claims
    assert gate.validation_errors == ()
    assert gate.claim_boundary == wave6_review_claim_boundary()
    assert validate_wave6_review_gate_result(gate) == ()


def test_wave6_review_gate_requires_replication_when_model_coverage_is_weak() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_single_provider_model_report(result)

    gate = build_wave6_review_gate(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )

    assert gate.decision is Wave6ReviewDecision.REQUIRE_INDEPENDENT_REPLICATION
    assert not gate.permits_bounded_wave6_review()
    assert "Do not advance to bounded Wave 6 review" in gate.required_actions[-1]


def test_wave6_review_gate_blocks_overclaiming_model_output() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_overclaiming_model_report(result)

    gate = build_wave6_review_gate(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )

    assert gate.decision is Wave6ReviewDecision.BLOCKED
    assert not gate.permits_bounded_wave6_review()
    assert "Block bounded Wave 6 review language." in gate.required_actions


def test_wave6_review_gate_blocks_failed_trial() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_clean_model_report(result)

    gate = build_wave6_review_gate(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )

    assert gate.decision is Wave6ReviewDecision.BLOCKED
    assert not gate.permits_bounded_wave6_review()


def test_validate_wave6_review_gate_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    integrated_bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    model_report = make_clean_model_report(result)
    gate = build_wave6_review_gate(
        result=result,
        evidence_packet=evidence_packet,
        integrated_bundle=integrated_bundle,
        model_review_report=model_report,
    )
    invalid_gate = type(gate)(
        gate_id=gate.gate_id,
        trial_id=gate.trial_id,
        decision=gate.decision,
        replication_packet_id=gate.replication_packet_id,
        integrated_bundle_id=gate.integrated_bundle_id,
        model_review_report_id=gate.model_review_report_id,
        allowed_claims=gate.allowed_claims,
        blocked_claims=gate.blocked_claims,
        validation_errors=gate.validation_errors,
        required_actions=gate.required_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_wave6_review_gate_result(invalid_gate)

    assert "claim_boundary must match fixed Wave 6 boundary" in errors
