from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.trial import TransferTrialInput, run_transfer_trial
from ix_function.worldtwin_handoff import (
    WorldTwinAdaptationAction,
    WorldTwinHandoffStatus,
    build_worldtwin_handoff_packet,
    choose_worldtwin_handoff_status,
    validate_worldtwin_handoff_packet,
    worldtwin_claim_boundary,
)
from tests.fixtures import make_trial_input


def make_failed_result() -> object:
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


def test_choose_worldtwin_handoff_status_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())

    assert (
        choose_worldtwin_handoff_status(result)
        is WorldTwinHandoffStatus.READY_FOR_SCENARIO_REPLAY
    )


def test_choose_worldtwin_handoff_status_for_failed_delta_trial() -> None:
    result = make_failed_result()

    assert (
        choose_worldtwin_handoff_status(result)
        is WorldTwinHandoffStatus.READY_FOR_FAILURE_REPLAY
    )


def test_build_worldtwin_handoff_packet_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_worldtwin_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.packet_id == "trial-001:worldtwin-handoff"
    assert handoff.status is WorldTwinHandoffStatus.READY_FOR_SCENARIO_REPLAY
    assert handoff.scenario_packet.scenario_id == "trial-001:worldtwin-scenario"
    assert handoff.scenario_packet.function_id == "causal-bottleneck-v1"
    assert handoff.scenario_packet.source_domain_id == "water-flow"
    assert handoff.scenario_packet.target_domain_id == "ci-pipeline"
    assert handoff.scenario_packet.variable_bindings
    assert handoff.scenario_packet.outcome_delta_bindings
    assert handoff.scenario_packet.evidence_refs[0] == "trial-001:evidence-packet"
    assert handoff.adaptation_packet.action is (
        WorldTwinAdaptationAction.PRESERVE_AS_SUPPORTED_CASE
    )
    assert "AGI proof" in handoff.adaptation_packet.blocked_claims
    assert handoff.claim_boundary == worldtwin_claim_boundary()
    assert validate_worldtwin_handoff_packet(handoff) == ()


def test_worldtwin_handoff_for_failed_trial_recommends_retest() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_worldtwin_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.status is WorldTwinHandoffStatus.READY_FOR_FAILURE_REPLAY
    assert handoff.adaptation_packet.action is (
        WorldTwinAdaptationAction.RETEST_WITH_NARROWER_RANGE
    )
    assert "strong transfer support" in handoff.adaptation_packet.blocked_claims
    assert "Do not increase confidence" in handoff.required_worldtwin_actions[1]
    assert validate_worldtwin_handoff_packet(handoff) == ()


def test_validate_worldtwin_handoff_packet_detects_bad_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    handoff = build_worldtwin_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_handoff = type(handoff)(
        packet_id=handoff.packet_id,
        status=handoff.status,
        scenario_packet=handoff.scenario_packet,
        adaptation_packet=handoff.adaptation_packet,
        required_worldtwin_actions=handoff.required_worldtwin_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_worldtwin_handoff_packet(invalid_handoff)

    assert "claim_boundary must match fixed WorldTwin boundary" in errors
