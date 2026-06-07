from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.ix_contract_handoff import (
    IXContractAuthority,
    IXContractHandoffStatus,
    IXContractStepKind,
    build_ix_contract_handoff_packet,
    choose_ix_contract_handoff_status,
    ix_contract_claim_boundary,
    validate_ix_contract_handoff_packet,
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


def test_choose_ix_contract_status_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())

    assert (
        choose_ix_contract_handoff_status(result)
        is IXContractHandoffStatus.READY_FOR_REVIEW_CONTRACT
    )


def test_choose_ix_contract_status_for_failed_trial() -> None:
    result = make_failed_result()

    assert (
        choose_ix_contract_handoff_status(result)
        is IXContractHandoffStatus.READY_FOR_FAILURE_CONTRACT
    )


def test_build_ix_contract_handoff_packet_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_ix_contract_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.packet_id == "trial-001:ix-contract-handoff"
    assert handoff.status is IXContractHandoffStatus.READY_FOR_REVIEW_CONTRACT
    assert handoff.trial_id == "trial-001"
    assert handoff.contract_id == "trial-001:ix-review-contract"
    assert len(handoff.preconditions) == 4
    assert len(handoff.steps) == 6
    assert handoff.boundary.authority is IXContractAuthority.HUMAN_REVIEW_REQUIRED
    assert handoff.boundary.human_review_required
    assert "AGI proof" in handoff.boundary.prohibited_claims
    assert "bounded causal-transfer evidence" in handoff.boundary.allowed_claims
    assert handoff.claim_boundary == ix_contract_claim_boundary()
    assert validate_ix_contract_handoff_packet(handoff) == ()


def test_ix_contract_handoff_preserves_required_step_kinds() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_ix_contract_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    step_kinds = {step.kind for step in handoff.steps}

    assert step_kinds == {
        IXContractStepKind.ASSERT_BOUNDARY,
        IXContractStepKind.BIND_EVIDENCE,
        IXContractStepKind.CHECK_FALSIFICATION,
        IXContractStepKind.CHECK_PERMISSION,
        IXContractStepKind.PRESERVE_UNCERTAINTY,
        IXContractStepKind.REQUIRE_HUMAN_REVIEW,
    }


def test_build_ix_contract_handoff_packet_for_failed_trial_blocks_promotion() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_ix_contract_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.status is IXContractHandoffStatus.READY_FOR_FAILURE_CONTRACT
    assert "failure evidence" in handoff.boundary.allowed_claims
    assert "Block claim promotion." in handoff.required_ix_actions
    assert validate_ix_contract_handoff_packet(handoff) == ()


def test_validate_ix_contract_handoff_packet_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    handoff = build_ix_contract_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_handoff = type(handoff)(
        packet_id=handoff.packet_id,
        status=handoff.status,
        trial_id=handoff.trial_id,
        contract_id=handoff.contract_id,
        preconditions=handoff.preconditions,
        steps=handoff.steps,
        boundary=handoff.boundary,
        required_ix_actions=handoff.required_ix_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_ix_contract_handoff_packet(invalid_handoff)

    assert "claim_boundary must match fixed IX contract boundary" in errors
