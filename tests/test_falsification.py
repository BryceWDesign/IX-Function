from __future__ import annotations

from ix_function.falsification import (
    CriterionEvaluation,
    CriterionStatus,
    FalsificationSeverity,
    FalsificationVerdict,
    build_falsification_ledger,
    choose_falsification_verdict,
    default_wave6_kill_criteria,
    required_actions_for_verdict,
    validate_falsification_ledger,
)
from ix_function.learning import (
    ConfidenceBand,
    LearningDisposition,
    TransferLearningUpdate,
)
from ix_function.mapping import MappingQuality, SlotMapping, TransferMapping
from ix_function.reality_delta import RealityDeltaReport, TransferOutcomeStatus
from ix_function.uncertainty import (
    UncertaintyItem,
    UncertaintyKind,
    UncertaintyLedger,
    UncertaintySeverity,
    UncertaintyState,
)


def make_mapping(quality: MappingQuality = MappingQuality.COMPLETE) -> TransferMapping:
    return TransferMapping(
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        slot_mappings=(
            SlotMapping(
                slot_id="final_output",
                observable_name="Completion Time",
                score=1.0,
                uncertainty_notes=(),
            ),
        ),
        quality=quality,
        coverage_score=1.0 if quality is not MappingQuality.INSUFFICIENT else 0.5,
        ambiguity_score=0.0,
        warnings=(),
    )


def make_report(
    status: TransferOutcomeStatus = TransferOutcomeStatus.SUPPORTED,
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
        mean_score=1.0 if status is TransferOutcomeStatus.SUPPORTED else 0.0,
        confidence_delta=0.12 if status is TransferOutcomeStatus.SUPPORTED else -0.12,
        uncertainty_notes=("Outcome scoring note.",),
        blocking_errors=(),
    )


def make_uncertainty_ledger(blocking: bool = False) -> UncertaintyLedger:
    if not blocking:
        return UncertaintyLedger(
            ledger_id="uncertainty-ledger-001",
            items=(),
        )

    return UncertaintyLedger(
        ledger_id="uncertainty-ledger-001",
        items=(
            UncertaintyItem(
                uncertainty_id="blocking-uncertainty-001",
                kind=UncertaintyKind.OUTCOME,
                severity=UncertaintySeverity.BLOCKING,
                state=UncertaintyState.ESCALATED,
                source_id="prediction-001:reality-delta",
                statement="Outcome could not support the transfer.",
                mitigation="Block the stronger claim and retest.",
            ),
        ),
    )


def make_learning_update(
    usable: bool = True,
) -> TransferLearningUpdate:
    return TransferLearningUpdate(
        update_id="prediction-001:reality-delta:learning-update",
        function_id="causal-bottleneck-v1",
        report_id="prediction-001:reality-delta",
        source_confidence=0.55,
        confidence_delta=0.12,
        revised_confidence=0.67,
        confidence_band=ConfidenceBand.MEDIUM,
        disposition=LearningDisposition.PROMOTE,
        future_planning_rules=(
            "Permit cautious reuse with a new pre-outcome prediction.",
        )
        if usable
        else (),
        uncertainty_notes=("Learning uncertainty note.",),
        blocking_errors=() if usable else ("learning blocker",),
    )


def test_default_wave6_kill_criteria_are_stable_and_complete() -> None:
    criteria = default_wave6_kill_criteria()

    assert len(criteria) == 6
    assert criteria[0].criterion_id == "IXF-FALSIFY-001"
    assert criteria[-1].criterion_id == "IXF-FALSIFY-006"
    assert any(
        criterion.severity is FalsificationSeverity.KILL
        for criterion in criteria
    )


def test_build_falsification_ledger_allows_bounded_evidence_when_clean() -> None:
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-001",
        function_id="causal-bottleneck-v1",
        is_cross_domain=True,
        mapping=make_mapping(),
        report=make_report(),
        uncertainty_ledger=make_uncertainty_ledger(),
        learning_update=make_learning_update(),
    )

    assert ledger.verdict is FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE
    assert ledger.kill_evaluations() == ()
    assert validate_falsification_ledger(ledger) == ()
    assert ledger.required_actions == (
        "Allow bounded IX-Function evidence language.",
        "Do not represent this result as AGI proof.",
    )


def test_build_falsification_ledger_kills_same_domain_theater() -> None:
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-002",
        function_id="causal-bottleneck-v1",
        is_cross_domain=False,
        mapping=make_mapping(),
        report=make_report(),
        uncertainty_ledger=make_uncertainty_ledger(),
        learning_update=make_learning_update(),
    )

    assert ledger.verdict is FalsificationVerdict.KILL_CLAIM
    assert ledger.kill_evaluations()
    assert any("Source and target domains were not" in action for action in ledger.required_actions)


def test_build_falsification_ledger_kills_insufficient_mapping() -> None:
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-003",
        function_id="causal-bottleneck-v1",
        is_cross_domain=True,
        mapping=make_mapping(MappingQuality.INSUFFICIENT),
        report=make_report(),
        uncertainty_ledger=make_uncertainty_ledger(),
        learning_update=make_learning_update(),
    )

    assert ledger.verdict is FalsificationVerdict.KILL_CLAIM
    assert any(
        evaluation.criterion_id == "IXF-FALSIFY-002"
        for evaluation in ledger.kill_evaluations()
    )


def test_build_falsification_ledger_downgrades_failed_outcome() -> None:
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-004",
        function_id="causal-bottleneck-v1",
        is_cross_domain=True,
        mapping=make_mapping(),
        report=make_report(TransferOutcomeStatus.FAILED),
        uncertainty_ledger=make_uncertainty_ledger(),
        learning_update=make_learning_update(),
    )

    assert ledger.verdict is FalsificationVerdict.DOWNGRADE_CLAIM
    assert ledger.failed_evaluations()
    assert any("Reality-delta status was 'failed'" in action for action in ledger.required_actions)


def test_build_falsification_ledger_kills_blocking_uncertainty() -> None:
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-005",
        function_id="causal-bottleneck-v1",
        is_cross_domain=True,
        mapping=make_mapping(),
        report=make_report(),
        uncertainty_ledger=make_uncertainty_ledger(blocking=True),
        learning_update=make_learning_update(),
    )

    assert ledger.verdict is FalsificationVerdict.KILL_CLAIM
    assert any(
        evaluation.criterion_id == "IXF-FALSIFY-004"
        for evaluation in ledger.kill_evaluations()
    )


def test_build_falsification_ledger_downgrades_missing_future_behavior() -> None:
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-006",
        function_id="causal-bottleneck-v1",
        is_cross_domain=True,
        mapping=make_mapping(),
        report=make_report(),
        uncertainty_ledger=make_uncertainty_ledger(),
        learning_update=make_learning_update(usable=False),
    )

    assert ledger.verdict is FalsificationVerdict.DOWNGRADE_CLAIM
    assert any(
        evaluation.criterion_id == "IXF-FALSIFY-005"
        for evaluation in ledger.failed_evaluations()
    )


def test_choose_falsification_verdict_prioritizes_kill_over_downgrade() -> None:
    evaluations = (
        CriterionEvaluation(
            criterion_id="warning",
            status=CriterionStatus.WARNING,
            severity=FalsificationSeverity.WARNING,
            reason="Warning.",
            evidence_refs=("warning-ref",),
        ),
        CriterionEvaluation(
            criterion_id="downgrade",
            status=CriterionStatus.FAILED,
            severity=FalsificationSeverity.DOWNGRADE,
            reason="Downgrade.",
            evidence_refs=("downgrade-ref",),
        ),
        CriterionEvaluation(
            criterion_id="kill",
            status=CriterionStatus.FAILED,
            severity=FalsificationSeverity.KILL,
            reason="Kill.",
            evidence_refs=("kill-ref",),
        ),
    )

    assert choose_falsification_verdict(evaluations) is FalsificationVerdict.KILL_CLAIM


def test_required_actions_for_verdict_names_failed_criteria() -> None:
    evaluations = (
        CriterionEvaluation(
            criterion_id="IXF-FALSIFY-003",
            status=CriterionStatus.FAILED,
            severity=FalsificationSeverity.DOWNGRADE,
            reason="Outcome failed.",
            evidence_refs=("report",),
        ),
    )

    actions = required_actions_for_verdict(
        FalsificationVerdict.DOWNGRADE_CLAIM,
        evaluations,
    )

    assert actions == (
        "Downgrade claim due to IXF-FALSIFY-003: Outcome failed.",
    )


def test_validate_falsification_ledger_rejects_duplicate_criterion_ids() -> None:
    evaluation = CriterionEvaluation(
        criterion_id="duplicate",
        status=CriterionStatus.PASSED,
        severity=FalsificationSeverity.WARNING,
        reason="Passed.",
        evidence_refs=("ref",),
    )
    ledger = build_falsification_ledger(
        ledger_id="falsification-ledger-007",
        function_id="causal-bottleneck-v1",
        is_cross_domain=True,
        mapping=make_mapping(),
        report=make_report(),
        uncertainty_ledger=make_uncertainty_ledger(),
        learning_update=make_learning_update(),
        criteria=(),
    )
    invalid_ledger = type(ledger)(
        ledger_id="falsification-ledger-007",
        function_id="causal-bottleneck-v1",
        evaluations=(evaluation, evaluation),
        verdict=FalsificationVerdict.REQUIRE_RETEST,
        required_actions=("Retest duplicate criteria.",),
    )

    errors = validate_falsification_ledger(invalid_ledger)

    assert "criterion evaluations must use unique criterion_id values" in errors
