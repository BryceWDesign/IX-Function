from __future__ import annotations

from ix_function.mapping import MappingQuality, SlotMapping, TransferMapping
from ix_function.prediction import (
    PredictedObservable,
    PredictionDirection,
    PredictionStatus,
    TransferPrediction,
    assess_prediction_readiness,
    direction_matches_delta,
    validate_predicted_observable,
    validate_transfer_prediction,
)


def make_complete_mapping() -> TransferMapping:
    return TransferMapping(
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        slot_mappings=(
            SlotMapping(
                slot_id="upstream_capacity_intervention",
                observable_name="Worker Count",
                score=1.0,
                uncertainty_notes=(),
            ),
            SlotMapping(
                slot_id="downstream_constraint",
                observable_name="Slowest Downstream Stage Time",
                score=1.0,
                uncertainty_notes=(),
            ),
            SlotMapping(
                slot_id="final_output",
                observable_name="Completion Time",
                score=1.0,
                uncertainty_notes=(),
            ),
        ),
        quality=MappingQuality.COMPLETE,
        coverage_score=1.0,
        ambiguity_score=0.0,
        warnings=(),
    )


def make_ready_prediction() -> TransferPrediction:
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
                    "If the slowest downstream stage remains the limit, more "
                    "workers should not materially improve completion time."
                ),
            ),
        ),
        assumptions=(
            "The slowest downstream stage remains unchanged during the trial.",
            "Completion time measures the full pipeline, not one parallel segment.",
        ),
        uncertainty_notes=(
            "Hidden parallelism could weaken the bottleneck transfer.",
        ),
        created_before_outcome=True,
    )


def test_predicted_observable_normalizes_name_and_detects_range() -> None:
    predicted = PredictedObservable(
        observable_name=" Completion   Time ",
        direction=PredictionDirection.NO_CHANGE,
        rationale="Expect no meaningful change.",
        expected_min=10.0,
        expected_max=11.0,
    )

    assert predicted.normalized_observable_name() == "completion_time"
    assert predicted.has_numeric_range()


def test_validate_predicted_observable_accepts_complete_numeric_prediction() -> None:
    predicted = make_ready_prediction().predicted_observables[0]

    assert validate_predicted_observable(predicted) == ()


def test_validate_predicted_observable_rejects_unknown_and_bad_range() -> None:
    predicted = PredictedObservable(
        observable_name="Completion Time",
        direction=PredictionDirection.UNKNOWN,
        rationale="Unknown is not a committed prediction.",
        expected_min=20.0,
        expected_max=10.0,
        tolerance=-1.0,
    )

    errors = validate_predicted_observable(predicted)

    assert (
        "predicted observable 'Completion Time' must not use unknown direction"
        in errors
    )
    assert "predicted observable 'Completion Time' has min above max" in errors
    assert "predicted observable 'Completion Time' has negative tolerance" in errors


def test_validate_transfer_prediction_accepts_ready_prediction() -> None:
    prediction = make_ready_prediction()
    mapping = make_complete_mapping()

    assert validate_transfer_prediction(prediction, mapping) == ()


def test_assess_prediction_readiness_marks_ready_prediction() -> None:
    readiness = assess_prediction_readiness(
        make_ready_prediction(),
        make_complete_mapping(),
    )

    assert readiness.status is PredictionStatus.READY
    assert readiness.is_ready()
    assert readiness.blocking_errors == ()


def test_prediction_blocks_outcome_leakage() -> None:
    valid_prediction = make_ready_prediction()
    invalid_prediction = TransferPrediction(
        prediction_id=valid_prediction.prediction_id,
        function_id=valid_prediction.function_id,
        target_domain_id=valid_prediction.target_domain_id,
        source_evidence_id=valid_prediction.source_evidence_id,
        target_intervention_id=valid_prediction.target_intervention_id,
        mapping_quality=valid_prediction.mapping_quality,
        predicted_observables=valid_prediction.predicted_observables,
        assumptions=valid_prediction.assumptions,
        uncertainty_notes=valid_prediction.uncertainty_notes,
        created_before_outcome=False,
        forbidden_outcome_ids=("outcome-001",),
    )

    errors = validate_transfer_prediction(invalid_prediction, make_complete_mapping())

    assert "prediction must be created before outcome observation" in errors
    assert "prediction must not reference outcome identifiers" in errors


def test_prediction_blocks_observables_not_present_in_mapping() -> None:
    valid_prediction = make_ready_prediction()
    invalid_prediction = TransferPrediction(
        prediction_id=valid_prediction.prediction_id,
        function_id=valid_prediction.function_id,
        target_domain_id=valid_prediction.target_domain_id,
        source_evidence_id=valid_prediction.source_evidence_id,
        target_intervention_id=valid_prediction.target_intervention_id,
        mapping_quality=valid_prediction.mapping_quality,
        predicted_observables=(
            PredictedObservable(
                observable_name="Unmapped Metric",
                direction=PredictionDirection.INCREASE,
                rationale="This metric is not mapped.",
            ),
        ),
        assumptions=valid_prediction.assumptions,
        uncertainty_notes=valid_prediction.uncertainty_notes,
        created_before_outcome=True,
    )

    errors = validate_transfer_prediction(invalid_prediction, make_complete_mapping())

    assert (
        "predicted observable 'Unmapped Metric' is not present in the transfer "
        "mapping"
    ) in errors


def test_prediction_blocks_unusable_mapping() -> None:
    valid_prediction = make_ready_prediction()
    insufficient_mapping = TransferMapping(
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        slot_mappings=make_complete_mapping().slot_mappings[:1],
        quality=MappingQuality.INSUFFICIENT,
        coverage_score=0.333333,
        ambiguity_score=0.0,
        warnings=("missing target observables",),
    )
    invalid_prediction = TransferPrediction(
        prediction_id=valid_prediction.prediction_id,
        function_id=valid_prediction.function_id,
        target_domain_id=valid_prediction.target_domain_id,
        source_evidence_id=valid_prediction.source_evidence_id,
        target_intervention_id=valid_prediction.target_intervention_id,
        mapping_quality=MappingQuality.INSUFFICIENT,
        predicted_observables=valid_prediction.predicted_observables,
        assumptions=valid_prediction.assumptions,
        uncertainty_notes=valid_prediction.uncertainty_notes,
        created_before_outcome=True,
    )

    errors = validate_transfer_prediction(invalid_prediction, insufficient_mapping)

    assert "mapping is not usable for prediction" in errors
    assert (
        "predicted observable 'Completion Time' is not present in the transfer "
        "mapping"
    ) in errors


def test_direction_matches_delta_uses_tolerance() -> None:
    assert direction_matches_delta(
        PredictionDirection.INCREASE,
        baseline=10.0,
        observed=12.0,
        tolerance=0.5,
    )
    assert direction_matches_delta(
        PredictionDirection.DECREASE,
        baseline=10.0,
        observed=8.0,
        tolerance=0.5,
    )
    assert direction_matches_delta(
        PredictionDirection.LIMITED_CHANGE,
        baseline=10.0,
        observed=10.3,
        tolerance=0.5,
    )
    assert not direction_matches_delta(
        PredictionDirection.NO_CHANGE,
        baseline=10.0,
        observed=11.0,
        tolerance=0.5,
    )
    assert not direction_matches_delta(
        PredictionDirection.UNKNOWN,
        baseline=10.0,
        observed=10.0,
        tolerance=0.5,
    )
