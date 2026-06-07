from __future__ import annotations

from ix_function.blackfox_handoff import (
    BlackFoxHandoffStatus,
    BlackFoxPolicyDecision,
    blackfox_claim_boundary,
    build_blackfox_handoff_packet,
    choose_blackfox_handoff_status,
    validate_blackfox_handoff_packet,
)
from ix_function.evidence import build_trial_evidence_packet
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.trial import TransferTrialInput, TransferTrialResult, run_transfer_trial
from tests.fixtures import make_trial_input


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


def test_choose_blackfox_handoff_status_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())

    assert (
        choose_blackfox_handoff_status(result)
        is BlackFoxHandoffStatus.READY_FOR_REVIEW_BUNDLE
    )


def test_choose_blackfox_handoff_status_for_failed_trial() -> None:
    result = make_failed_result()

    assert (
        choose_blackfox_handoff_status(result)
        is BlackFoxHandoffStatus.READY_FOR_FAILURE_REVIEW
    )


def test_build_blackfox_handoff_packet_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_blackfox_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.packet_id == "trial-001:blackfox-handoff"
    assert handoff.status is BlackFoxHandoffStatus.READY_FOR_REVIEW_BUNDLE
    assert handoff.evidence_binding.trial_id == "trial-001"
    assert handoff.evidence_binding.evidence_packet_id == "trial-001:evidence-packet"
    assert handoff.evidence_binding.model_output_trusted is False
    assert handoff.review_bundle.sandbox_required
    assert handoff.review_bundle.egress_allowed is False
    assert handoff.review_bundle.human_approval_required
    assert any(
        gate.decision is BlackFoxPolicyDecision.REQUIRE_HUMAN_APPROVAL
        for gate in handoff.review_bundle.policy_gates
    )
    assert handoff.claim_boundary == blackfox_claim_boundary()
    assert validate_blackfox_handoff_packet(handoff) == ()


def test_blackfox_failed_trial_handoff_blocks_promotion() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_blackfox_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.status is BlackFoxHandoffStatus.READY_FOR_FAILURE_REVIEW
    assert any(
        gate.decision is BlackFoxPolicyDecision.BLOCK
        for gate in handoff.review_bundle.policy_gates
    )
    assert "Block confidence promotion." in handoff.required_blackfox_actions
    assert validate_blackfox_handoff_packet(handoff) == ()


def test_validate_blackfox_handoff_packet_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    handoff = build_blackfox_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_handoff = type(handoff)(
        packet_id=handoff.packet_id,
        status=handoff.status,
        evidence_binding=handoff.evidence_binding,
        review_bundle=handoff.review_bundle,
        required_blackfox_actions=handoff.required_blackfox_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_blackfox_handoff_packet(invalid_handoff)

    assert "claim_boundary must match fixed BlackFox boundary" in errors
