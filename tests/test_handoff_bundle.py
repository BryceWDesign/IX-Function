from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.handoff_bundle import (
    IntegratedHandoffStatus,
    build_integrated_handoff_bundle,
    integrated_handoff_claim_boundary,
    validate_integrated_handoff_bundle,
)
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


def test_build_integrated_handoff_bundle_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert bundle.bundle_id == "trial-001:integrated-donor-handoff"
    assert bundle.status is IntegratedHandoffStatus.READY_FOR_WAVE6_REVIEW
    assert bundle.is_ready_for_wave6_review()
    assert bundle.trial_id == "trial-001"
    assert bundle.evidence_packet_id == "trial-001:evidence-packet"
    assert bundle.kernel_handoff.packet_id == "trial-001:kernel-handoff"
    assert bundle.worldtwin_handoff.packet_id == "trial-001:worldtwin-handoff"
    assert bundle.intent_loop_handoff.packet_id == "trial-001:intent-loop-handoff"
    assert bundle.blackfox_handoff.packet_id == "trial-001:blackfox-handoff"
    assert bundle.assurance_handoff.packet_id == "trial-001:assurance-handoff"
    assert bundle.ix_contract_handoff.packet_id == "trial-001:ix-contract-handoff"
    assert bundle.validation_errors == ()
    assert bundle.claim_boundary == integrated_handoff_claim_boundary()
    assert validate_integrated_handoff_bundle(bundle) == ()


def test_integrated_bundle_for_failed_trial_enters_failure_review() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert bundle.status is IntegratedHandoffStatus.READY_FOR_FAILURE_REVIEW
    assert not bundle.is_ready_for_wave6_review()
    assert bundle.validation_errors == ()
    assert "Review all donor handoffs as failure" in bundle.required_review_actions[0]
    assert validate_integrated_handoff_bundle(bundle) == ()


def test_integrated_bundle_validation_detects_trial_id_mismatch() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_bundle = type(bundle)(
        bundle_id=bundle.bundle_id,
        status=bundle.status,
        trial_id="wrong-trial",
        evidence_packet_id=bundle.evidence_packet_id,
        kernel_handoff=bundle.kernel_handoff,
        worldtwin_handoff=bundle.worldtwin_handoff,
        intent_loop_handoff=bundle.intent_loop_handoff,
        blackfox_handoff=bundle.blackfox_handoff,
        assurance_handoff=bundle.assurance_handoff,
        ix_contract_handoff=bundle.ix_contract_handoff,
        validation_errors=bundle.validation_errors,
        required_review_actions=bundle.required_review_actions,
        claim_boundary=bundle.claim_boundary,
    )

    errors = validate_integrated_handoff_bundle(invalid_bundle)

    assert "all donor handoffs must reference bundle trial_id" in errors


def test_integrated_bundle_validation_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    bundle = build_integrated_handoff_bundle(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_bundle = type(bundle)(
        bundle_id=bundle.bundle_id,
        status=bundle.status,
        trial_id=bundle.trial_id,
        evidence_packet_id=bundle.evidence_packet_id,
        kernel_handoff=bundle.kernel_handoff,
        worldtwin_handoff=bundle.worldtwin_handoff,
        intent_loop_handoff=bundle.intent_loop_handoff,
        blackfox_handoff=bundle.blackfox_handoff,
        assurance_handoff=bundle.assurance_handoff,
        ix_contract_handoff=bundle.ix_contract_handoff,
        validation_errors=bundle.validation_errors,
        required_review_actions=bundle.required_review_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_integrated_handoff_bundle(invalid_bundle)

    assert "claim_boundary must match fixed integrated boundary" in errors
