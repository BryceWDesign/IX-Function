from __future__ import annotations

from ix_function.falsification import (
    FalsificationLedger,
    FalsificationVerdict,
)
from ix_function.learning import (
    ConfidenceBand,
    LearningDisposition,
    TransferLearningUpdate,
)
from ix_function.mapping import MappingQuality, SlotMapping, TransferMapping
from ix_function.negative_control import (
    NegativeControlEvaluation,
    NegativeControlKind,
    NegativeControlSpec,
    NegativeControlStatus,
    NegativeControlSuite,
    build_negative_control_suite,
    default_negative_controls,
    evaluate_anti_theater_gate,
    evaluate_negative_control,
    validate_negative_control_spec,
    validate_negative_control_suite,
)
from ix_function.reality_delta import RealityDeltaReport, TransferOutcomeStatus


def make_mapping(
    quality: MappingQuality = MappingQuality.INSUFFICIENT,
) -> TransferMapping:
    return TransferMapping(
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        slot_mappings=(
            SlotMapping(
                slot_id="final_output",
                observable_name="Completion Time",
                score=0.5,
                uncertainty_notes=("weak mapping",),
            ),
        ),
        quality=quality,
        coverage_score=0.5 if quality is MappingQuality.INSUFFICIENT else 1.0,
        ambiguity_score=0.0,
        warnings=("mapping warning",)
        if quality is not MappingQuality.COMPLETE
        else (),
    )


def make_report(
    status: TransferOutcomeStatus = TransferOutcomeStatus.FAILED,
) -> RealityDeltaReport:
    return RealityDeltaReport(
        report_id="prediction-001:reality-delta",
        prediction_id="prediction-001",
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        target_intervention_id="increase-worker-count",
        outcome_id="outcome-001",
        observable_deltas=(),
        status=status,
        mean_score=0.0,
        confidence_delta=-0.12,
        uncertainty_notes=("Outcome failed transfer.",),
        blocking_errors=(),
    )


def make_learning_update(
    disposition: LearningDisposition = LearningDisposition.WEAKEN,
    blocking_errors: tuple[str, ...] = (),
) -> TransferLearningUpdate:
    return TransferLearningUpdate(
        update_id="prediction-001:reality-delta:learning-update",
        function_id="causal-bottleneck-v1",
        report_id="prediction-001:reality-delta",
        source_confidence=0.55,
        confidence_delta=-0.12,
        revised_confidence=0.43,
        confidence_band=ConfidenceBand.LOW,
        disposition=disposition,
        future_planning_rules=("Weaken future reuse.",),
        uncertainty_notes=("Learning weakened transfer.",),
        blocking_errors=blocking_errors,
    )


def make_falsification_ledger(
    verdict: FalsificationVerdict = FalsificationVerdict.KILL_CLAIM,
) -> FalsificationLedger:
    return FalsificationLedger(
        ledger_id="falsification-ledger-001",
        function_id="causal-bottleneck-v1",
        evaluations=(),
        verdict=verdict,
        required_actions=("Block or downgrade negative-control claim.",),
    )


def test_default_negative_controls_are_stable() -> None:
    controls = default_negative_controls()

    assert len(controls) == 5
    assert controls[0].control_id == "IXF-NEG-001"
    assert controls[-1].control_id == "IXF-NEG-005"
    assert {control.kind for control in controls} == {
        NegativeControlKind.EXPECTED_FAILURE,
        NegativeControlKind.INSUFFICIENT_MAPPING,
        NegativeControlKind.OUTCOME_LEAKAGE,
        NegativeControlKind.SAME_DOMAIN_THEATER,
        NegativeControlKind.SHUFFLED_MAPPING,
    }


def test_validate_negative_control_spec_accepts_complete_spec() -> None:
    spec = default_negative_controls()[0]

    assert validate_negative_control_spec(spec) == ()


def test_validate_negative_control_spec_rejects_empty_fields() -> None:
    spec = NegativeControlSpec(
        control_id="",
        kind=NegativeControlKind.EXPECTED_FAILURE,
        purpose="",
        expected_blocking_behavior="",
    )

    errors = validate_negative_control_spec(spec)

    assert "control_id must not be empty" in errors
    assert "purpose must not be empty for ''" in errors
    assert "expected_blocking_behavior must not be empty for ''" in errors


def test_clean_positive_chain_marks_controls_not_applicable() -> None:
    suite = build_negative_control_suite(
        suite_id="negative-suite-positive",
        mapping=make_mapping(MappingQuality.COMPLETE),
        report=make_report(TransferOutcomeStatus.SUPPORTED),
        learning_update=make_learning_update(LearningDisposition.PROMOTE),
        falsification_ledger=make_falsification_ledger(
            FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE
        ),
    )

    assert suite.passed()
    assert suite.failed_controls() == ()
    assert all(
        evaluation.status is NegativeControlStatus.NOT_APPLICABLE
        for evaluation in suite.evaluations
    )


def test_insufficient_mapping_control_passes_when_mapping_is_insufficient() -> None:
    spec = default_negative_controls()[0]

    evaluation = evaluate_negative_control(
        spec=spec,
        mapping=make_mapping(MappingQuality.INSUFFICIENT),
        report=make_report(),
        learning_update=make_learning_update(),
        falsification_ledger=make_falsification_ledger(),
    )

    assert evaluation.status is NegativeControlStatus.PASSED
    assert evaluation.is_clean()
    assert "blocked" in evaluation.reason


def test_insufficient_mapping_control_is_not_applicable_to_clean_mapping() -> None:
    spec = default_negative_controls()[0]

    evaluation = evaluate_negative_control(
        spec=spec,
        mapping=make_mapping(MappingQuality.COMPLETE),
        report=make_report(TransferOutcomeStatus.SUPPORTED),
        learning_update=make_learning_update(LearningDisposition.PROMOTE),
        falsification_ledger=make_falsification_ledger(
            FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE
        ),
    )

    assert evaluation.status is NegativeControlStatus.NOT_APPLICABLE
    assert evaluation.is_clean()


def test_expected_failure_control_passes_when_failure_weakens_learning() -> None:
    spec = default_negative_controls()[2]

    evaluation = evaluate_negative_control(
        spec=spec,
        mapping=make_mapping(),
        report=make_report(TransferOutcomeStatus.FAILED),
        learning_update=make_learning_update(LearningDisposition.QUARANTINE),
        falsification_ledger=make_falsification_ledger(),
    )

    assert evaluation.status is NegativeControlStatus.PASSED
    assert "weakened" in evaluation.reason


def test_expected_failure_control_fails_when_failure_is_promoted() -> None:
    spec = default_negative_controls()[2]

    evaluation = evaluate_negative_control(
        spec=spec,
        mapping=make_mapping(MappingQuality.COMPLETE),
        report=make_report(TransferOutcomeStatus.FAILED),
        learning_update=make_learning_update(LearningDisposition.PROMOTE),
        falsification_ledger=make_falsification_ledger(
            FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE
        ),
    )

    assert evaluation.status is NegativeControlStatus.FAILED
    assert "did not weaken" in evaluation.reason


def test_outcome_leakage_control_passes_when_report_has_blockers() -> None:
    spec = default_negative_controls()[3]
    report = RealityDeltaReport(
        report_id="prediction-001:reality-delta",
        prediction_id="prediction-001",
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        target_intervention_id="increase-worker-count",
        outcome_id="outcome-001",
        observable_deltas=(),
        status=TransferOutcomeStatus.UNSCORABLE,
        mean_score=0.0,
        confidence_delta=-0.12,
        uncertainty_notes=("Lineage was invalid.",),
        blocking_errors=("prediction was not pre-outcome",),
    )

    evaluation = evaluate_negative_control(
        spec=spec,
        mapping=make_mapping(),
        report=report,
        learning_update=make_learning_update(),
        falsification_ledger=make_falsification_ledger(),
    )

    assert evaluation.status is NegativeControlStatus.PASSED
    assert "blocked" in evaluation.reason


def test_build_negative_control_suite_collects_evaluations() -> None:
    suite = build_negative_control_suite(
        suite_id="negative-suite-001",
        mapping=make_mapping(),
        report=make_report(),
        learning_update=make_learning_update(),
        falsification_ledger=make_falsification_ledger(),
    )

    assert suite.suite_id == "negative-suite-001"
    assert len(suite.evaluations) == 5
    assert validate_negative_control_suite(suite) == ()


def test_anti_theater_gate_allows_clean_positive_chain() -> None:
    suite = build_negative_control_suite(
        suite_id="negative-suite-positive",
        mapping=make_mapping(MappingQuality.COMPLETE),
        report=make_report(TransferOutcomeStatus.SUPPORTED),
        learning_update=make_learning_update(LearningDisposition.PROMOTE),
        falsification_ledger=make_falsification_ledger(
            FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE
        ),
    )

    result = evaluate_anti_theater_gate(suite)

    assert result.allowed
    assert result.failed_control_ids == ()
    assert "positive transfer chain" in result.reason


def test_anti_theater_gate_allows_clean_negative_controls() -> None:
    suite = build_negative_control_suite(
        suite_id="negative-suite-002",
        mapping=make_mapping(),
        report=make_report(),
        learning_update=make_learning_update(),
        falsification_ledger=make_falsification_ledger(),
    )

    result = evaluate_anti_theater_gate(suite)

    assert result.allowed
    assert result.failed_control_ids == ()
    assert "negative controls" in result.reason


def test_anti_theater_gate_blocks_failed_negative_controls() -> None:
    failed = NegativeControlEvaluation(
        control_id="IXF-NEG-FAIL",
        kind=NegativeControlKind.EXPECTED_FAILURE,
        status=NegativeControlStatus.FAILED,
        reason="Failed outcome was promoted.",
        evidence_refs=("report",),
    )
    suite = NegativeControlSuite(
        suite_id="negative-suite-003",
        evaluations=(failed,),
    )

    result = evaluate_anti_theater_gate(suite)

    assert result.allowed is False
    assert result.failed_control_ids == ("IXF-NEG-FAIL",)
    assert result.required_actions == (
        "Fix negative control IXF-NEG-FAIL: Failed outcome was promoted.",
    )


def test_anti_theater_gate_blocks_invalid_suite() -> None:
    suite = NegativeControlSuite(
        suite_id="",
        evaluations=(),
    )

    result = evaluate_anti_theater_gate(suite)

    assert result.allowed is False
    assert result.required_actions == (
        "suite_id must not be empty",
        "at least one negative-control evaluation is required",
    )


def test_validate_negative_control_suite_rejects_duplicate_control_ids() -> None:
    evaluation = NegativeControlEvaluation(
        control_id="duplicate",
        kind=NegativeControlKind.SHUFFLED_MAPPING,
        status=NegativeControlStatus.PASSED,
        reason="Passed.",
        evidence_refs=("ref",),
    )
    suite = NegativeControlSuite(
        suite_id="negative-suite-004",
        evaluations=(evaluation, evaluation),
    )

    errors = validate_negative_control_suite(suite)

    assert "negative-control evaluations must use unique control_id values" in errors
