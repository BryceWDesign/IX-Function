from __future__ import annotations

from ix_function.causal_function import (
    CausalFamily,
    CausalFunction,
    CausalMechanism,
    CausalSlotRole,
    CausalVariableSlot,
)
from ix_function.domain import ObservableRole
from ix_function.learning import (
    ConfidenceBand,
    LearningDisposition,
    build_learning_update,
    choose_learning_disposition,
    clamp_confidence,
    classify_confidence,
    combine_uncertainty_notes,
    future_planning_rules_for,
    validate_learning_inputs,
)
from ix_function.prediction import PredictionDirection
from ix_function.reality_delta import (
    ObservableDelta,
    OutcomeMatch,
    RealityDeltaReport,
    TransferOutcomeStatus,
)


def make_function(confidence: float = 0.55) -> CausalFunction:
    return CausalFunction(
        function_id="prediction-001",
        name="Downstream Bottleneck Limit",
        family=CausalFamily.BOTTLENECK,
        summary="Upstream increases cannot beat a downstream bottleneck.",
        variable_slots=(
            CausalVariableSlot(
                slot_id="upstream_capacity_intervention",
                role=CausalSlotRole.INTERVENTION,
                description="Increase upstream capacity.",
                semantic_tags=("capacity", "intervention", "upstream"),
                compatible_observable_roles=(ObservableRole.INTERVENTION,),
            ),
            CausalVariableSlot(
                slot_id="final_output",
                role=CausalSlotRole.OUTPUT,
                description="Final output after intervention.",
                semantic_tags=("output", "throughput"),
                compatible_observable_roles=(ObservableRole.OUTPUT,),
            ),
        ),
        mechanisms=(
            CausalMechanism(
                mechanism_id="limited-output-gain",
                family=CausalFamily.BOTTLENECK,
                premise="A downstream stage is already limiting output.",
                expected_effect="Increasing upstream capacity produces little gain.",
                assumptions=("Downstream limit remains unchanged.",),
            ),
        ),
        prior_confidence=confidence,
        uncertainty_notes=("Hidden parallel paths may invalidate the mechanism.",),
    )


def make_report(
    status: TransferOutcomeStatus = TransferOutcomeStatus.SUPPORTED,
    confidence_delta: float = 0.12,
    blocking_errors: tuple[str, ...] = (),
) -> RealityDeltaReport:
    return RealityDeltaReport(
        report_id="prediction-001:reality-delta",
        prediction_id="prediction-001",
        target_domain_id="ci-pipeline",
        target_intervention_id="increase-worker-count",
        outcome_id="outcome-001",
        observable_deltas=(
            ObservableDelta(
                observable_name="Completion Time",
                predicted_direction=PredictionDirection.LIMITED_CHANGE,
                baseline_value=121.0,
                observed_value=122.0,
                numeric_delta=1.0,
                direction_matched=True,
                range_matched=True,
                outcome_match=OutcomeMatch.MATCHED,
                score=1.0,
                notes=(),
            ),
        ),
        status=status,
        mean_score=1.0,
        confidence_delta=confidence_delta,
        uncertainty_notes=("Observed outcome supported bounded transfer evidence.",),
        blocking_errors=blocking_errors,
    )


def test_clamp_confidence_bounds_values() -> None:
    assert clamp_confidence(-0.5) == 0.0
    assert clamp_confidence(0.3456789) == 0.345679
    assert clamp_confidence(1.5) == 1.0


def test_classify_confidence_uses_conservative_bands() -> None:
    assert classify_confidence(0.1) is ConfidenceBand.QUARANTINED
    assert classify_confidence(0.3) is ConfidenceBand.LOW
    assert classify_confidence(0.6) is ConfidenceBand.MEDIUM
    assert classify_confidence(0.8) is ConfidenceBand.HIGH


def test_choose_learning_disposition_promotes_supported_confident_transfer() -> None:
    assert (
        choose_learning_disposition(TransferOutcomeStatus.SUPPORTED, 0.67)
        is LearningDisposition.PROMOTE
    )


def test_choose_learning_disposition_quarantines_low_failed_transfer() -> None:
    assert (
        choose_learning_disposition(TransferOutcomeStatus.FAILED, 0.31)
        is LearningDisposition.QUARANTINE
    )


def test_future_planning_rules_preserve_behavior_change() -> None:
    rules = future_planning_rules_for(
        status=TransferOutcomeStatus.FAILED,
        disposition=LearningDisposition.QUARANTINE,
        family_value="bottleneck",
    )

    assert len(rules) == 2
    assert "Quarantine" in rules[0]
    assert "Block automatic reuse" in rules[1]


def test_combine_uncertainty_notes_preserves_sources() -> None:
    notes = combine_uncertainty_notes(
        ("Function uncertainty.",),
        ("Reality uncertainty.",),
    )

    assert notes == (
        "function: Function uncertainty.",
        "reality_delta: Reality uncertainty.",
    )


def test_validate_learning_inputs_accepts_matching_function_and_report() -> None:
    assert validate_learning_inputs(make_function(), make_report()) == ()


def test_validate_learning_inputs_blocks_lineage_mismatch() -> None:
    function = make_function()
    report = RealityDeltaReport(
        report_id="different-prediction:reality-delta",
        prediction_id="different-prediction",
        target_domain_id="ci-pipeline",
        target_intervention_id="increase-worker-count",
        outcome_id="outcome-001",
        observable_deltas=make_report().observable_deltas,
        status=TransferOutcomeStatus.SUPPORTED,
        mean_score=1.0,
        confidence_delta=0.12,
        uncertainty_notes=("Supported but mismatched.",),
    )

    errors = validate_learning_inputs(function, report)

    assert (
        "causal_function function_id does not match report prediction lineage"
        in errors
    )


def test_build_learning_update_promotes_supported_transfer() -> None:
    update = build_learning_update(make_function(), make_report())

    assert update.update_id == "prediction-001:reality-delta:learning-update"
    assert update.function_id == "prediction-001"
    assert update.report_id == "prediction-001:reality-delta"
    assert update.source_confidence == 0.55
    assert update.confidence_delta == 0.12
    assert update.revised_confidence == 0.67
    assert update.confidence_band is ConfidenceBand.MEDIUM
    assert update.disposition is LearningDisposition.PROMOTE
    assert update.should_update_future_behavior()
    assert update.blocking_errors == ()
    assert "cautious reuse" in update.future_planning_rules[0]


def test_build_learning_update_quarantines_failed_low_confidence_transfer() -> None:
    report = make_report(
        status=TransferOutcomeStatus.FAILED,
        confidence_delta=-0.12,
    )

    update = build_learning_update(make_function(confidence=0.45), report)

    assert update.revised_confidence == 0.33
    assert update.confidence_band is ConfidenceBand.LOW
    assert update.disposition is LearningDisposition.QUARANTINE
    assert "Quarantine" in update.future_planning_rules[0]


def test_build_learning_update_weakens_invalid_report_without_promoting() -> None:
    report = make_report(
        status=TransferOutcomeStatus.UNSCORABLE,
        confidence_delta=-0.08,
        blocking_errors=("outcome missing predicted observable",),
    )

    update = build_learning_update(make_function(), report)

    assert update.disposition is LearningDisposition.WEAKEN
    assert update.confidence_delta == -0.08
    assert update.blocking_errors == ("reality_delta report contains blocking errors",)
    assert update.should_update_future_behavior() is False
