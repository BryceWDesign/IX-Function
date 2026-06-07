from __future__ import annotations

from ix_function.learning import (
    ConfidenceBand,
    LearningDisposition,
    TransferLearningUpdate,
)
from ix_function.mapping import MappingQuality, SlotMapping, TransferMapping
from ix_function.prediction import (
    PredictedObservable,
    PredictionDirection,
    TransferPrediction,
)
from ix_function.reality_delta import (
    RealityDeltaReport,
    TransferOutcomeStatus,
)
from ix_function.uncertainty import (
    EvidenceClaimStrength,
    UncertaintyItem,
    UncertaintyKind,
    UncertaintyLedger,
    UncertaintySeverity,
    UncertaintyState,
    build_uncertainty_ledger,
    evaluate_claim_strength_gate,
    evaluate_uncertainty_gate,
    items_from_learning_update,
    items_from_mapping,
    items_from_prediction,
    items_from_reality_delta,
    validate_uncertainty_ledger,
)


def make_mapping(quality: MappingQuality = MappingQuality.AMBIGUOUS) -> TransferMapping:
    return TransferMapping(
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        slot_mappings=(
            SlotMapping(
                slot_id="final_output",
                observable_name="Completion Time",
                score=0.96,
                uncertainty_notes=("nearest alternative 'Duration' scored 0.920",),
            ),
        ),
        quality=quality,
        coverage_score=1.0 if quality is not MappingQuality.INSUFFICIENT else 0.5,
        ambiguity_score=1.0 if quality is MappingQuality.AMBIGUOUS else 0.0,
        warnings=("target observable reuse requires review",),
    )


def make_clean_mapping() -> TransferMapping:
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
        quality=MappingQuality.COMPLETE,
        coverage_score=1.0,
        ambiguity_score=0.0,
        warnings=(),
    )


def make_prediction() -> TransferPrediction:
    return TransferPrediction(
        prediction_id="prediction-001",
        function_id="causal-bottleneck-v1",
        target_domain_id="ci-pipeline",
        source_evidence_id="source-trial-001:source-learning",
        target_intervention_id="increase-worker-count",
        mapping_quality=MappingQuality.AMBIGUOUS,
        predicted_observables=(
            PredictedObservable(
                observable_name="Completion Time",
                direction=PredictionDirection.LIMITED_CHANGE,
                rationale="Bottleneck transfer predicts limited completion change.",
                expected_min=118.0,
                expected_max=125.0,
                tolerance=5.0,
            ),
        ),
        assumptions=("Slowest stage remains unchanged.",),
        uncertainty_notes=("Hidden parallelism may weaken the transfer.",),
        created_before_outcome=True,
    )


def make_report(
    status: TransferOutcomeStatus = TransferOutcomeStatus.MIXED,
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
        mean_score=0.5,
        confidence_delta=0.02,
        uncertainty_notes=("Observed outcome partially supported prediction.",),
        blocking_errors=(),
    )


def make_update() -> TransferLearningUpdate:
    return TransferLearningUpdate(
        update_id="prediction-001:reality-delta:learning-update",
        function_id="causal-bottleneck-v1",
        report_id="prediction-001:reality-delta",
        source_confidence=0.55,
        confidence_delta=0.02,
        revised_confidence=0.57,
        confidence_band=ConfidenceBand.MEDIUM,
        disposition=LearningDisposition.RETAIN,
        future_planning_rules=("Use narrower prediction next time.",),
        uncertainty_notes=("reality_delta: mixed transfer result.",),
        blocking_errors=(),
    )


def test_uncertainty_item_identifies_blocking_conditions() -> None:
    blocking = UncertaintyItem(
        uncertainty_id="uncertainty-001",
        kind=UncertaintyKind.OUTCOME,
        severity=UncertaintySeverity.BLOCKING,
        state=UncertaintyState.OPEN,
        source_id="report-001",
        statement="Outcome failed.",
        mitigation="Quarantine transfer.",
    )
    escalated = UncertaintyItem(
        uncertainty_id="uncertainty-002",
        kind=UncertaintyKind.MAPPING,
        severity=UncertaintySeverity.MEDIUM,
        state=UncertaintyState.ESCALATED,
        source_id="mapping-001",
        statement="Mapping is escalated.",
        mitigation="Review mapping.",
    )

    assert blocking.is_blocking()
    assert escalated.is_blocking()


def test_items_from_mapping_preserves_ambiguity_warnings_and_slot_notes() -> None:
    items = items_from_mapping(make_mapping())

    assert len(items) == 3
    assert items[0].kind is UncertaintyKind.AMBIGUITY
    assert items[0].severity is UncertaintySeverity.MEDIUM
    assert items[1].statement == "target observable reuse requires review"
    assert "nearest alternative" in items[2].statement


def test_items_from_mapping_escalates_insufficient_mapping() -> None:
    items = items_from_mapping(make_mapping(MappingQuality.INSUFFICIENT))

    assert any(item.severity is UncertaintySeverity.BLOCKING for item in items)
    assert any(item.state is UncertaintyState.ESCALATED for item in items)


def test_items_from_prediction_preserves_assumptions_and_notes() -> None:
    items = items_from_prediction(make_prediction())

    assert len(items) == 2
    assert items[0].kind is UncertaintyKind.ASSUMPTION
    assert items[0].statement == "Slowest stage remains unchanged."
    assert items[1].statement == "Hidden parallelism may weaken the transfer."


def test_items_from_reality_delta_escalates_failed_result() -> None:
    items = items_from_reality_delta(make_report(TransferOutcomeStatus.FAILED))

    assert items[0].severity is UncertaintySeverity.BLOCKING
    assert items[0].state is UncertaintyState.ESCALATED
    assert "failed the transfer prediction" in items[0].statement


def test_items_from_reality_delta_escalates_unscorable_result() -> None:
    items = items_from_reality_delta(make_report(TransferOutcomeStatus.UNSCORABLE))

    assert items[0].kind is UncertaintyKind.MEASUREMENT
    assert items[0].severity is UncertaintySeverity.BLOCKING


def test_items_from_learning_update_preserves_learning_uncertainty() -> None:
    items = items_from_learning_update(make_update())

    assert len(items) == 1
    assert items[0].kind is UncertaintyKind.TRANSFER
    assert items[0].statement == "reality_delta: mixed transfer result."


def test_build_uncertainty_ledger_combines_full_chain() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-001",
        mapping=make_mapping(),
        prediction=make_prediction(),
        report=make_report(),
        learning_update=make_update(),
    )

    assert ledger.ledger_id == "ledger-001"
    assert len(ledger.items) >= 7
    assert validate_uncertainty_ledger(ledger) == ()
    assert ledger.maximum_severity() is UncertaintySeverity.HIGH


def test_uncertainty_gate_allows_nonblocking_open_uncertainty() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-002",
        mapping=make_mapping(),
        prediction=make_prediction(),
        report=make_report(),
        learning_update=make_update(),
    )

    result = evaluate_uncertainty_gate(ledger)

    assert result.allowed
    assert result.blocking_ids == ()
    assert "bounded candidate evidence" in result.reason


def test_uncertainty_gate_blocks_escalated_uncertainty() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-003",
        mapping=make_mapping(MappingQuality.INSUFFICIENT),
        prediction=make_prediction(),
        report=make_report(TransferOutcomeStatus.UNSCORABLE),
        learning_update=make_update(),
    )

    result = evaluate_uncertainty_gate(ledger)

    assert result.allowed is False
    assert result.maximum_severity is UncertaintySeverity.BLOCKING
    assert result.blocking_ids
    assert "blocked stronger transfer claims" in result.reason


def test_claim_strength_gate_allows_internal_notes_with_blockers() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-004",
        mapping=make_mapping(MappingQuality.INSUFFICIENT),
        prediction=make_prediction(),
        report=make_report(TransferOutcomeStatus.UNSCORABLE),
        learning_update=make_update(),
    )

    result = evaluate_claim_strength_gate(
        ledger,
        EvidenceClaimStrength.INTERNAL_NOTE,
    )

    assert result.allowed
    assert result.required_actions == (
        "Preserve all uncertainty items with the internal note.",
    )
    assert "does not promote" in result.reason


def test_claim_strength_gate_blocks_candidate_evidence_with_blockers() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-005",
        mapping=make_mapping(MappingQuality.INSUFFICIENT),
        prediction=make_prediction(),
        report=make_report(TransferOutcomeStatus.UNSCORABLE),
        learning_update=make_update(),
    )

    result = evaluate_claim_strength_gate(
        ledger,
        EvidenceClaimStrength.BOUNDED_CANDIDATE_EVIDENCE,
    )

    assert result.allowed is False
    assert result.blocking_ids
    assert "blocked" in result.reason


def test_claim_strength_gate_blocks_strong_support_with_high_open_items() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-006",
        mapping=make_mapping(),
        prediction=make_prediction(),
        report=make_report(TransferOutcomeStatus.MIXED),
        learning_update=make_update(),
    )

    result = evaluate_claim_strength_gate(
        ledger,
        EvidenceClaimStrength.STRONG_TRANSFER_SUPPORT,
    )

    assert result.allowed is False
    assert result.blocking_ids
    assert "high-severity open uncertainty" in result.reason


def test_claim_strength_gate_allows_bounded_evidence_with_nonblocking_items() -> None:
    ledger = build_uncertainty_ledger(
        ledger_id="ledger-007",
        mapping=make_clean_mapping(),
        prediction=make_prediction(),
        report=make_report(TransferOutcomeStatus.SUPPORTED),
        learning_update=make_update(),
    )

    result = evaluate_claim_strength_gate(
        ledger,
        EvidenceClaimStrength.BOUNDED_CANDIDATE_EVIDENCE,
    )

    assert result.allowed
    assert result.blocking_ids == ()
    assert "bounded IX-Function evidence language" in result.reason


def test_claim_strength_gate_blocks_invalid_ledgers() -> None:
    ledger = UncertaintyLedger(
        ledger_id="",
        items=(),
    )

    result = evaluate_claim_strength_gate(
        ledger,
        EvidenceClaimStrength.BOUNDED_CANDIDATE_EVIDENCE,
    )

    assert result.allowed is False
    assert result.required_actions == ("ledger_id must not be empty",)


def test_validate_uncertainty_ledger_rejects_duplicate_item_ids() -> None:
    item = UncertaintyItem(
        uncertainty_id="duplicate",
        kind=UncertaintyKind.TRANSFER,
        severity=UncertaintySeverity.LOW,
        state=UncertaintyState.OPEN,
        source_id="source",
        statement="Statement.",
        mitigation="Mitigation.",
    )
    ledger = UncertaintyLedger(
        ledger_id="ledger-008",
        items=(item, item),
    )

    errors = validate_uncertainty_ledger(ledger)

    assert "uncertainty item identifiers must be unique" in errors
