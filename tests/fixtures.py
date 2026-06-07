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
    DomainPair,
    DomainProfile,
    Observable,
    ObservableRole,
    ValueKind,
)
from ix_function.mapping import MappingQuality
from ix_function.observation import (
    DomainSnapshot,
    InterventionRecord,
    MeasuredValue,
    OutcomeRecord,
)
from ix_function.prediction import (
    PredictedObservable,
    PredictionDirection,
    TransferPrediction,
)
from ix_function.trial import TransferTrialInput


def make_source_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="water-flow",
        name="Water Flow",
        kind=DomainKind.FLOW,
        summary="A flow domain with upstream input and downstream output.",
        observables=(
            Observable(
                name="Input Rate",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.REAL,
                description="Upstream capacity intervention.",
            ),
            Observable(
                name="Output Rate",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Final downstream throughput output.",
            ),
            Observable(
                name="System State",
                role=ObservableRole.STATE,
                value_kind=ValueKind.CATEGORICAL,
                description="Whether the flow path is saturated.",
            ),
        ),
    )


def make_target_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="ci-pipeline",
        name="CI Pipeline",
        kind=DomainKind.COMPUTING,
        summary="A pipeline with workers and completion output.",
        observables=(
            Observable(
                name="Worker Count",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.INTEGER,
                description="Upstream capacity intervention for pipeline workers.",
            ),
            Observable(
                name="Completion Throughput Output",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Final output and completion throughput after tests.",
                unit="seconds",
            ),
        ),
    )


def make_function() -> CausalFunction:
    return CausalFunction(
        function_id="causal-bottleneck-v1",
        name="Downstream Bottleneck Limit",
        family=CausalFamily.BOTTLENECK,
        summary="Increasing upstream capacity should not beat a downstream limit.",
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
                semantic_tags=("output", "throughput", "completion"),
                compatible_observable_roles=(ObservableRole.OUTPUT,),
            ),
        ),
        mechanisms=(
            CausalMechanism(
                mechanism_id="limited-output-gain",
                family=CausalFamily.BOTTLENECK,
                premise="A downstream stage remains limiting.",
                expected_effect="More upstream capacity yields limited output change.",
                assumptions=("Downstream limit remains unchanged.",),
            ),
        ),
        prior_confidence=0.55,
        uncertainty_notes=("Hidden parallelism may weaken the mechanism.",),
        learned_from_domain_id="water-flow",
    )


def make_prediction() -> TransferPrediction:
    return TransferPrediction(
        prediction_id="prediction-001",
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        source_evidence_id="trial-001:source-learning",
        target_intervention_id="increase-worker-count",
        mapping_quality=MappingQuality.COMPLETE,
        predicted_observables=(
            PredictedObservable(
                observable_name="Completion Throughput Output",
                direction=PredictionDirection.LIMITED_CHANGE,
                expected_min=118.0,
                expected_max=126.0,
                tolerance=5.0,
                rationale="Bottleneck transfer predicts limited completion change.",
            ),
        ),
        assumptions=("Slowest downstream stage remains unchanged.",),
        uncertainty_notes=("Hidden parallelism may weaken transfer.",),
        created_before_outcome=True,
    )


def make_trial_input() -> TransferTrialInput:
    return TransferTrialInput(
        trial_id="trial-001",
        domain_pair=DomainPair(
            source=make_source_domain(),
            target=make_target_domain(),
            transfer_purpose="Transfer bottleneck structure from flow to CI.",
        ),
        causal_function=make_function(),
        source_baseline=DomainSnapshot(
            domain_id="water-flow",
            snapshot_id="source-baseline",
            captured_at_label="source-before",
            values=(
                MeasuredValue(
                    observable_name="Output Rate",
                    value=10.0,
                    evidence_id="source-baseline-output",
                ),
            ),
            source="source-fixture",
        ),
        source_intervention=InterventionRecord(
            domain_id="water-flow",
            intervention_id="increase-input-rate",
            values=(
                MeasuredValue(
                    observable_name="Input Rate",
                    value=20.0,
                    evidence_id="source-input-intervention",
                ),
            ),
            rationale="Increase upstream input to test downstream limit.",
        ),
        source_outcome=OutcomeRecord(
            domain_id="water-flow",
            outcome_id="source-outcome",
            observed_after_intervention_id="increase-input-rate",
            values=(
                MeasuredValue(
                    observable_name="Output Rate",
                    value=10.2,
                    evidence_id="source-outcome-output",
                ),
                MeasuredValue(
                    observable_name="System State",
                    value="saturated",
                    evidence_id="source-outcome-state",
                ),
            ),
            result_summary="Output barely changed after increased input.",
        ),
        source_support_reasons=(
            "Output remained close to baseline after intervention.",
            "State evidence preserved saturation.",
        ),
        source_uncertainty_notes=(
            "Controlled fixture may omit noisy real-world effects.",
        ),
        target_baseline=DomainSnapshot(
            domain_id="ci-pipeline",
            snapshot_id="target-baseline",
            captured_at_label="target-before",
            values=(
                MeasuredValue(
                    observable_name="Completion Throughput Output",
                    value=122.0,
                    evidence_id="target-baseline-completion",
                ),
            ),
            source="target-fixture",
        ),
        target_intervention=InterventionRecord(
            domain_id="ci-pipeline",
            intervention_id="increase-worker-count",
            values=(
                MeasuredValue(
                    observable_name="Worker Count",
                    value=8,
                    evidence_id="target-worker-intervention",
                ),
            ),
            rationale="Increase workers to test bottleneck transfer.",
        ),
        target_outcome=OutcomeRecord(
            domain_id="ci-pipeline",
            outcome_id="target-outcome",
            observed_after_intervention_id="increase-worker-count",
            values=(
                MeasuredValue(
                    observable_name="Completion Throughput Output",
                    value=123.0,
                    evidence_id="target-outcome-completion",
                ),
            ),
            result_summary="Completion remained close to baseline.",
        ),
        prediction=make_prediction(),
    )
