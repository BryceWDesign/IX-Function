from __future__ import annotations

from ix_function.causal_function import (
    CausalFamily,
    CausalFunction,
    CausalMechanism,
    CausalSlotRole,
    CausalVariableSlot,
)
from ix_function.domain import (
    DomainKind,
    DomainProfile,
    Observable,
    ObservableRole,
    ValueKind,
)
from ix_function.observation import (
    DomainSnapshot,
    InterventionRecord,
    MeasuredValue,
    OutcomeRecord,
)
from ix_function.source_learning import (
    SourceLearningStatus,
    SourceLearningTrial,
    evaluate_source_learning_trial,
    validate_source_learning_trial,
)


def make_flow_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="water-flow",
        name="Water Flow",
        kind=DomainKind.FLOW,
        summary="A bounded flow domain with upstream input and downstream output.",
        observables=(
            Observable(
                name="Input Rate",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.REAL,
                description="Upstream capacity intervention in liters per second.",
                unit="liters_per_second",
            ),
            Observable(
                name="Output Rate",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Final downstream throughput output.",
                unit="liters_per_second",
            ),
            Observable(
                name="System State",
                role=ObservableRole.STATE,
                value_kind=ValueKind.CATEGORICAL,
                description="Whether the channel is saturated or unconstrained.",
            ),
        ),
    )


def make_bottleneck_function() -> CausalFunction:
    return CausalFunction(
        function_id="causal-bottleneck-v1",
        name="Downstream Bottleneck Limit",
        family=CausalFamily.BOTTLENECK,
        summary="More upstream input cannot exceed a saturated downstream channel.",
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
                description="Final output after the intervention.",
                semantic_tags=("output", "throughput", "completion"),
                compatible_observable_roles=(ObservableRole.OUTPUT,),
            ),
        ),
        mechanisms=(
            CausalMechanism(
                mechanism_id="saturated-output-limit",
                family=CausalFamily.BOTTLENECK,
                premise="The downstream path is already saturated.",
                expected_effect="Increasing input produces little output gain.",
                assumptions=("The downstream channel remains unchanged.",),
            ),
        ),
        prior_confidence=0.55,
        uncertainty_notes=("Hidden bypass channels may invalidate the mechanism.",),
        learned_from_domain_id="water-flow",
    )


def make_supported_trial() -> SourceLearningTrial:
    return SourceLearningTrial(
        trial_id="source-trial-001",
        source_domain=make_flow_domain(),
        causal_function=make_bottleneck_function(),
        baseline_snapshot=DomainSnapshot(
            domain_id="water-flow",
            snapshot_id="baseline-001",
            captured_at_label="before-intervention",
            values=(
                MeasuredValue(
                    observable_name="Output Rate",
                    value=10.0,
                    evidence_id="baseline-output-rate",
                ),
                MeasuredValue(
                    observable_name="System State",
                    value="saturated",
                    evidence_id="baseline-state",
                ),
            ),
            source="controlled-flow-fixture",
        ),
        intervention=InterventionRecord(
            domain_id="water-flow",
            intervention_id="increase-input-rate",
            values=(
                MeasuredValue(
                    observable_name="Input Rate",
                    value=20.0,
                    evidence_id="intervention-input-rate",
                ),
            ),
            rationale="Increase upstream input to test the downstream limit.",
        ),
        outcome=OutcomeRecord(
            domain_id="water-flow",
            outcome_id="outcome-001",
            observed_after_intervention_id="increase-input-rate",
            values=(
                MeasuredValue(
                    observable_name="Output Rate",
                    value=10.2,
                    evidence_id="outcome-output-rate",
                ),
                MeasuredValue(
                    observable_name="System State",
                    value="saturated",
                    evidence_id="outcome-state",
                ),
            ),
            result_summary="Output barely changed after increased input.",
        ),
        support_reasons=(
            "Outcome remained close to baseline after intervention.",
            "State evidence preserved downstream saturation.",
        ),
        uncertainty_notes=(
            "The source fixture is controlled and may omit noisy real systems.",
        ),
    )


def test_validate_source_learning_trial_accepts_supported_trial() -> None:
    trial = make_supported_trial()

    assert validate_source_learning_trial(trial) == ()


def test_evaluate_source_learning_trial_marks_strong_source_support() -> None:
    trial = make_supported_trial()

    evidence = evaluate_source_learning_trial(trial)

    assert evidence.evidence_id == "source-trial-001:source-learning"
    assert evidence.function_id == "causal-bottleneck-v1"
    assert evidence.source_domain_id == "water-flow"
    assert evidence.status is SourceLearningStatus.SUPPORTED
    assert evidence.confidence_delta == 0.08
    assert evidence.adjusted_confidence(0.55) == 0.63
    assert evidence.blocking_errors == ()


def test_evaluate_source_learning_trial_marks_weak_support_when_uncertainty_dominates() -> None:
    supported = make_supported_trial()
    weak_trial = SourceLearningTrial(
        trial_id=supported.trial_id,
        source_domain=supported.source_domain,
        causal_function=supported.causal_function,
        baseline_snapshot=supported.baseline_snapshot,
        intervention=supported.intervention,
        outcome=supported.outcome,
        support_reasons=("Output changed less than input.",),
        uncertainty_notes=(
            "Unknown downstream geometry.",
            "Unknown measurement noise.",
            "Unknown bypass behavior.",
        ),
    )

    evidence = evaluate_source_learning_trial(weak_trial)

    assert evidence.status is SourceLearningStatus.WEAKLY_SUPPORTED
    assert evidence.confidence_delta == 0.01


def test_evaluate_source_learning_trial_blocks_invalid_records() -> None:
    supported = make_supported_trial()
    invalid_trial = SourceLearningTrial(
        trial_id=supported.trial_id,
        source_domain=supported.source_domain,
        causal_function=supported.causal_function,
        baseline_snapshot=supported.baseline_snapshot,
        intervention=supported.intervention,
        outcome=OutcomeRecord(
            domain_id="water-flow",
            outcome_id="outcome-002",
            observed_after_intervention_id="different-intervention",
            values=supported.outcome.values,
            result_summary="Outcome points at the wrong intervention.",
        ),
        support_reasons=supported.support_reasons,
        uncertainty_notes=supported.uncertainty_notes,
    )

    evidence = evaluate_source_learning_trial(invalid_trial)

    assert evidence.status is SourceLearningStatus.BLOCKED
    assert evidence.confidence_delta == -0.1
    assert "outcome must reference the trial intervention" in evidence.blocking_errors
    assert evidence.adjusted_confidence(0.05) == 0.0


def test_validate_source_learning_trial_blocks_mismatched_function_source() -> None:
    supported = make_supported_trial()
    mismatched_function = CausalFunction(
        function_id=supported.causal_function.function_id,
        name=supported.causal_function.name,
        family=supported.causal_function.family,
        summary=supported.causal_function.summary,
        variable_slots=supported.causal_function.variable_slots,
        mechanisms=supported.causal_function.mechanisms,
        prior_confidence=supported.causal_function.prior_confidence,
        uncertainty_notes=supported.causal_function.uncertainty_notes,
        learned_from_domain_id="different-domain",
    )
    invalid_trial = SourceLearningTrial(
        trial_id=supported.trial_id,
        source_domain=supported.source_domain,
        causal_function=mismatched_function,
        baseline_snapshot=supported.baseline_snapshot,
        intervention=supported.intervention,
        outcome=supported.outcome,
        support_reasons=supported.support_reasons,
        uncertainty_notes=supported.uncertainty_notes,
    )

    errors = validate_source_learning_trial(invalid_trial)

    assert (
        "causal_function learned_from_domain_id must be empty or match source "
        "domain_id"
    ) in errors
