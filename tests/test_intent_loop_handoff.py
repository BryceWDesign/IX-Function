from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.intent_loop_handoff import (
    IntentLoopHandoffStatus,
    IntentPermissionState,
    RealityFeedbackKind,
    build_intent_loop_handoff_packet,
    choose_intent_loop_handoff_status,
    intent_loop_claim_boundary,
    validate_intent_loop_handoff_packet,
)
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.trial import TransferTrialInput, run_transfer_trial
from tests.fixtures import make_trial_input


def make_failed_result():
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


def test_choose_intent_loop_status_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())

    assert (
        choose_intent_loop_handoff_status(result)
        is IntentLoopHandoffStatus.READY_FOR_FEEDBACK_BINDING
    )


def test_choose_intent_loop_status_for_failed_trial() -> None:
    result = make_failed_result()

    assert (
        choose_intent_loop_handoff_status(result)
        is IntentLoopHandoffStatus.READY_FOR_FAILURE_BINDING
    )


def test_build_intent_loop_handoff_packet_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_intent_loop_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.packet_id == "trial-001:intent-loop-handoff"
    assert handoff.status is IntentLoopHandoffStatus.READY_FOR_FEEDBACK_BINDING
    assert handoff.intent_binding.source_domain_id == "water-flow"
    assert handoff.intent_binding.target_domain_id == "ci-pipeline"
    assert "AGI proof" in handoff.intent_binding.prohibited_scope
    assert handoff.permission_binding.state is (
        IntentPermissionState.REQUIRE_HUMAN_PERMISSION
    )
    assert handoff.permission_binding.requires_human_review
    assert handoff.feedback_binding.kind is RealityFeedbackKind.SUPPORTED_TRANSFER
    assert handoff.feedback_binding.confidence_delta > 0
    assert handoff.memory_binding.should_update_memory
    assert handoff.memory_binding.should_quarantine is False
    assert handoff.claim_boundary == intent_loop_claim_boundary()
    assert validate_intent_loop_handoff_packet(handoff) == ()


def test_build_intent_loop_handoff_packet_for_failed_trial_quarantines_reuse() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_intent_loop_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.status is IntentLoopHandoffStatus.READY_FOR_FAILURE_BINDING
    assert handoff.permission_binding.state is (
        IntentPermissionState.BLOCK_AUTOMATIC_REUSE
    )
    assert handoff.feedback_binding.kind is RealityFeedbackKind.FAILED_TRANSFER
    assert handoff.feedback_binding.confidence_delta < 0
    assert handoff.memory_binding.should_quarantine
    assert "Bind failed" in handoff.required_loop_actions[0]
    assert validate_intent_loop_handoff_packet(handoff) == ()


def test_validate_intent_loop_handoff_packet_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    handoff = build_intent_loop_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_handoff = type(handoff)(
        packet_id=handoff.packet_id,
        status=handoff.status,
        intent_binding=handoff.intent_binding,
        permission_binding=handoff.permission_binding,
        feedback_binding=handoff.feedback_binding,
        memory_binding=handoff.memory_binding,
        required_loop_actions=handoff.required_loop_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_intent_loop_handoff_packet(invalid_handoff)

    assert "claim_boundary must match fixed IntentRealityLoop boundary" in errors
