from __future__ import annotations

from ix_function.mapping import MappingQuality
from ix_function.observation import (
    DomainSnapshot,
    MeasuredValue,
    OutcomeRecord,
)
from ix_function.prediction import (
    PredictedObservable,
    PredictionDirection,
    TransferPrediction,
)
from ix_function.reality_delta import (
    OutcomeMatch,
    TransferOutcomeStatus,
    build_reality_delta_report,
    classify_transfer_outcome,
    confidence_delta_for_status,
    mean_observable_score,
    numeric_value,
    range_matches_prediction,
    score_predicted_observable,
    validate_reality_delta_inputs,
)


def make_prediction() -> TransferPrediction:
    return TransferPrediction(
        prediction_id="prediction-001",
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        source_evidence_id="source-trial-001:source-learning",
        target_intervention_id="increase-worker-count",
        mapping_quality=MappingQuality.COMPLETE,
        predicted_observables=(
            PredictedObservable(
                observable_name="Completion Time",
                direction=PredictionDirection.LIMITED_CHANGE,
                expected_min=118.0,
                expected_max=125.0,
                tolerance=5.0,
                rationale=(
                    "Downstream test-stage bottleneck should keep completion "
                    "time near baseline."
                ),
            ),
        ),
        assumptions=("Slowest downstream stage remains unchanged.",),
        uncertainty_notes=("Hidden parallelism may weaken the transfer.",),
        created_before_outcome=True,
    )


def make_baseline(value: float = 121.0) -> DomainSnapshot:
    return DomainSnapshot(
        domain_id="ci-pipeline",
        snapshot_id="baseline-001",
        captured_at_label="before-target-intervention",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value=value,
                evidence_id="baseline-completion-time",
            ),
        ),
        source="ci-fixture",
    )


def make_outcome(value: float = 122.0) -> OutcomeRecord:
    return OutcomeRecord(
        domain_id="ci-pipeline",
        outcome_id="outcome-001",
        observed_after_intervention_id="increase-worker-count",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value=value,
                evidence_id="outcome-completion-time",
            ),
        ),
        result_summary="Completion time remained close to baseline.",
    )


def test_numeric_value_rejects_bool_and_non_numeric_values() -> None:
    assert numeric_value(True) is None
    assert numeric_value(False) is None
    assert numeric_value("12.0") is None
    assert numeric_value(None) is None
    assert numeric_value(4) == 4.0
    assert numeric_value(4.5) == 4.5


def test_range_matches_prediction_with_tolerance() -> None:
    predicted = make_prediction().predicted_observables[0]

    assert range_matches_prediction(predicted, observed=127.0)
    assert not range_matches_prediction(predicted, observed=131.0)


def test_score_predicted_observable_marks_supported_prediction() -> None:
    predicted = make_prediction().predicted_observables[0]

    delta = score_predicted_observable(
        predicted=predicted,
        baseline_value=121.0,
        observed_value=122.0,
    )

    assert delta.outcome_match is OutcomeMatch.MATCHED
    assert delta.direction_matched
    assert delta.range_matched is True
    assert delta.numeric_delta == 1.0
    assert delta.score == 1.0


def test_score_predicted_observable_marks_partial_prediction() -> None:
    predicted = make_prediction().predicted_observables[0]

    delta = score_predicted_observable(
        predicted=predicted,
        baseline_value=121.0,
        observed_value=130.0,
    )

    assert delta.outcome_match is OutcomeMatch.PARTIAL
    assert delta.direction_matched is False
    assert delta.range_matched is True
    assert delta.score == 0.5


def test_score_predicted_observable_marks_failed_prediction() -> None:
    predicted = make_prediction().predicted_observables[0]

    delta = score_predicted_observable(
        predicted=predicted,
        baseline_value=121.0,
        observed_value=90.0,
    )

    assert delta.outcome_match is OutcomeMatch.MISSED
    assert delta.direction_matched is False
    assert delta.range_matched is False
    assert delta.score == 0.0


def test_score_predicted_observable_marks_unscorable_missing_values() -> None:
    predicted = make_prediction().predicted_observables[0]

    delta = score_predicted_observable(
        predicted=predicted,
        baseline_value=None,
        observed_value=122.0,
    )

    assert delta.outcome_match is OutcomeMatch.UNSCORABLE
    assert delta.numeric_delta is None
    assert "baseline value is missing" in delta.notes


def test_classify_transfer_outcome_distinguishes_result_types() -> None:
    matched = score_predicted_observable(
        predicted=make_prediction().predicted_observables[0],
        baseline_value=121.0,
        observed_value=122.0,
    )
    missed = score_predicted_observable(
        predicted=make_prediction().predicted_observables[0],
        baseline_value=121.0,
        observed_value=90.0,
    )

    assert classify_transfer_outcome((matched,)) is TransferOutcomeStatus.SUPPORTED
    assert classify_transfer_outcome((matched, missed)) is TransferOutcomeStatus.MIXED
    assert classify_transfer_outcome((missed,)) is TransferOutcomeStatus.FAILED
    assert classify_transfer_outcome(()) is TransferOutcomeStatus.UNSCORABLE


def test_mean_observable_score_handles_empty_unscorable_collection() -> None:
    assert mean_observable_score(()) == 0.0


def test_confidence_delta_for_status_is_bounded_and_conservative() -> None:
    assert confidence_delta_for_status(TransferOutcomeStatus.SUPPORTED, 1.0) == 0.12
    assert confidence_delta_for_status(TransferOutcomeStatus.MIXED, 0.5) == 0.02
    assert confidence_delta_for_status(TransferOutcomeStatus.FAILED, 0.0) == -0.12
    assert confidence_delta_for_status(TransferOutcomeStatus.UNSCORABLE, 0.0) == -0.08


def test_validate_reality_delta_inputs_accepts_matching_records() -> None:
    assert (
        validate_reality_delta_inputs(
            make_prediction(),
            make_baseline(),
            make_outcome(),
        )
        == ()
    )


def test_validate_reality_delta_inputs_blocks_mismatched_intervention() -> None:
    outcome = OutcomeRecord(
        domain_id="ci-pipeline",
        outcome_id="outcome-002",
        observed_after_intervention_id="different-intervention",
        values=make_outcome().values,
        result_summary="Outcome references the wrong intervention.",
    )

    errors = validate_reality_delta_inputs(
        make_prediction(),
        make_baseline(),
        outcome,
    )

    assert "outcome must reference prediction target_intervention_id" in errors


def test_build_reality_delta_report_marks_supported_transfer() -> None:
    report = build_reality_delta_report(
        make_prediction(),
        make_baseline(),
        make_outcome(),
    )

    assert report.report_id == "prediction-001:reality-delta"
    assert report.status is TransferOutcomeStatus.SUPPORTED
    assert report.is_supported()
    assert report.mean_score == 1.0
    assert report.confidence_delta == 0.12
    assert report.blocking_errors == ()
    assert "bounded transfer evidence rather than AGI proof" in (
        report.uncertainty_notes[0]
    )


def test_build_reality_delta_report_handles_all_unscorable_numeric_values() -> None:
    baseline = DomainSnapshot(
        domain_id="ci-pipeline",
        snapshot_id="baseline-text",
        captured_at_label="before-target-intervention",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value="not numeric",
                evidence_id="baseline-text-value",
            ),
        ),
        source="ci-fixture",
    )
    outcome = OutcomeRecord(
        domain_id="ci-pipeline",
        outcome_id="outcome-text",
        observed_after_intervention_id="increase-worker-count",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value="still not numeric",
                evidence_id="outcome-text-value",
            ),
        ),
        result_summary="Outcome values were not numeric.",
    )

    report = build_reality_delta_report(make_prediction(), baseline, outcome)

    assert report.status is TransferOutcomeStatus.UNSCORABLE
    assert report.mean_score == 0.0
    assert report.confidence_delta == -0.08
    assert "could not be scored" in report.uncertainty_notes[0]


def test_build_reality_delta_report_blocks_invalid_input_records() -> None:
    bad_baseline = DomainSnapshot(
        domain_id="wrong-domain",
        snapshot_id="baseline-002",
        captured_at_label="before-target-intervention",
        values=make_baseline().values,
        source="ci-fixture",
    )

    report = build_reality_delta_report(
        make_prediction(),
        bad_baseline,
        make_outcome(),
    )

    assert report.status is TransferOutcomeStatus.UNSCORABLE
    assert report.mean_score == 0.0
    assert report.confidence_delta == -0.12
    assert (
        "prediction target_domain_id must match baseline domain_id"
        in report.blocking_errors
    )
