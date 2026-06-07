from __future__ import annotations

from ix_function.falsification import FalsificationVerdict
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.prediction import TransferPrediction
from ix_function.trial import (
    TransferTrialInput,
    TrialStatus,
    choose_trial_status,
    evaluate_bounded_claim_request,
    required_actions_for_trial,
    run_transfer_trial,
    validate_transfer_trial_input,
)
from ix_function.uncertainty import EvidenceClaimStrength
from tests.fixtures import make_trial_input


def test_validate_transfer_trial_input_accepts_complete_trial() -> None:
    errors = validate_transfer_trial_input(make_trial_input())

    assert errors == ()


def test_run_transfer_trial_allows_bounded_evidence_for_clean_chain() -> None:
    result = run_transfer_trial(make_trial_input())

    assert result.trial_id == "trial-001"
    assert result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED
    assert result.permits_bounded_evidence()
    assert result.source_evidence.blocking_errors == ()
    assert result.mapping.coverage_score == 1.0
    assert result.prediction_readiness.is_ready()
    assert result.reality_delta.is_supported()
    assert result.learning_update.should_update_future_behavior()
    assert result.uncertainty_gate.allowed
    assert result.falsification_ledger.verdict.value == "allow_bounded_evidence"
    assert result.negative_control_suite.passed()
    assert result.anti_theater_gate.allowed
    assert result.blocking_errors == ()


def test_run_transfer_trial_blocks_failed_transfer_outcome() -> None:
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

    result = run_transfer_trial(failed_input)

    assert result.status is TrialStatus.BLOCKED
    assert not result.permits_bounded_evidence()
    assert result.reality_delta.is_supported() is False
    assert result.uncertainty_gate.allowed is False
    assert result.required_actions


def test_validate_transfer_trial_input_blocks_prediction_intervention_mismatch(
) -> None:
    trial_input = make_trial_input()
    invalid_prediction = TransferPrediction(
        prediction_id=trial_input.prediction.prediction_id,
        function_id=trial_input.prediction.function_id,
        target_domain_id=trial_input.prediction.target_domain_id,
        source_evidence_id=trial_input.prediction.source_evidence_id,
        target_intervention_id="different-intervention",
        mapping_quality=trial_input.prediction.mapping_quality,
        predicted_observables=trial_input.prediction.predicted_observables,
        assumptions=trial_input.prediction.assumptions,
        uncertainty_notes=trial_input.prediction.uncertainty_notes,
        created_before_outcome=True,
    )
    invalid_input = TransferTrialInput(
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
        target_outcome=trial_input.target_outcome,
        prediction=invalid_prediction,
    )

    errors = validate_transfer_trial_input(invalid_input)

    assert "prediction target_intervention_id must match target intervention" in errors


def test_choose_trial_status_prioritizes_invalid_before_other_gates() -> None:
    status = choose_trial_status(
        blocking_errors=("bad input",),
        uncertainty_allowed=True,
        falsification_verdict=FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE,
        anti_theater_allowed=True,
    )

    assert status is TrialStatus.INVALID


def test_required_actions_for_allowed_trial_are_bounded() -> None:
    trial_result = run_transfer_trial(make_trial_input())

    actions = required_actions_for_trial(
        status=TrialStatus.BOUNDED_EVIDENCE_ALLOWED,
        blocking_errors=(),
        uncertainty_gate=trial_result.uncertainty_gate,
        falsification_ledger=trial_result.falsification_ledger,
        anti_theater_gate=trial_result.anti_theater_gate,
    )

    assert actions == (
        "Allow bounded IX-Function transfer evidence language.",
        "Preserve uncertainty, falsification, and negative-control records.",
        "Do not represent this result as AGI proof.",
    )


def test_evaluate_bounded_claim_request_allows_candidate_evidence() -> None:
    result = run_transfer_trial(make_trial_input())

    allowed, actions = evaluate_bounded_claim_request(
        result,
        EvidenceClaimStrength.BOUNDED_CANDIDATE_EVIDENCE,
    )

    assert allowed
    assert "Do not represent this result as AGI proof." in actions


def test_evaluate_bounded_claim_request_blocks_failed_trial() -> None:
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
    result = run_transfer_trial(failed_input)

    allowed, actions = evaluate_bounded_claim_request(
        result,
        EvidenceClaimStrength.BOUNDED_CANDIDATE_EVIDENCE,
    )

    assert not allowed
    assert actions[0] == "Transfer trial does not permit bounded evidence."
