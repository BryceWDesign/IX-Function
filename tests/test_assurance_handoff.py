from __future__ import annotations

from ix_function.assurance_handoff import (
    AssuranceDecision,
    AssuranceHandoffStatus,
    SafetyGateDecision,
    assurance_claim_boundary,
    build_assurance_handoff_packet,
    choose_assurance_handoff_status,
    validate_assurance_handoff_packet,
)
from ix_function.evidence import build_trial_evidence_packet
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.trial import (
    TransferTrialInput,
    TransferTrialResult,
    run_transfer_trial,
)
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


def test_choose_assurance_handoff_status_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())

    assert (
        choose_assurance_handoff_status(result)
        is AssuranceHandoffStatus.READY_FOR_ASSURANCE_REVIEW
    )


def test_choose_assurance_handoff_status_for_failed_trial() -> None:
    result = make_failed_result()

    assert (
        choose_assurance_handoff_status(result)
        is AssuranceHandoffStatus.READY_FOR_FAILURE_DOSSIER
    )


def test_build_assurance_handoff_packet_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_assurance_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.packet_id == "trial-001:assurance-handoff"
    assert handoff.status is AssuranceHandoffStatus.READY_FOR_ASSURANCE_REVIEW
    assert handoff.claim.claim_id == "trial-001:assurance-claim"
    assert handoff.claim.decision is AssuranceDecision.ALLOW_REVIEW_CLAIM
    assert handoff.claim.supported
    assert "AGI proof" in handoff.claim.blocked_claims
    assert handoff.safety_gates[0].decision is SafetyGateDecision.ALLOW_REVIEW_ONLY
    assert handoff.trace_links
    assert handoff.provenance.human_authority_required
    assert handoff.claim_boundary == assurance_claim_boundary()
    assert validate_assurance_handoff_packet(handoff) == ()


def test_assurance_handoff_for_failed_trial_blocks_claim() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_assurance_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.status is AssuranceHandoffStatus.READY_FOR_FAILURE_DOSSIER
    assert handoff.claim.decision is AssuranceDecision.BLOCK_CLAIM
    assert handoff.claim.supported is False
    assert any(
        gate.decision in {SafetyGateDecision.SAFE_HOLD, SafetyGateDecision.VETO}
        for gate in handoff.safety_gates
    )
    assert "Create a failure dossier" in handoff.required_assurance_actions[0]
    assert validate_assurance_handoff_packet(handoff) == ()


def test_validate_assurance_handoff_packet_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    handoff = build_assurance_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_handoff = type(handoff)(
        packet_id=handoff.packet_id,
        status=handoff.status,
        claim=handoff.claim,
        safety_gates=handoff.safety_gates,
        trace_links=handoff.trace_links,
        provenance=handoff.provenance,
        required_assurance_actions=handoff.required_assurance_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_assurance_handoff_packet(invalid_handoff)

    assert "claim_boundary must match fixed assurance boundary" in errors
