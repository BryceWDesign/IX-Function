from __future__ import annotations

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
    validate_intervention_against_domain,
    validate_outcome_against_domain,
    validate_snapshot_against_domain,
    value_matches_kind,
)


def make_ci_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="ci-pipeline",
        name="CI Pipeline",
        kind=DomainKind.COMPUTING,
        summary="A pipeline with a worker intervention and completion output.",
        observables=(
            Observable(
                name="Worker Count",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.INTEGER,
                description="Number of workers assigned to the pipeline.",
            ),
            Observable(
                name="Completion Time",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Measured completion time.",
                unit="seconds",
            ),
            Observable(
                name="Pipeline State",
                role=ObservableRole.STATE,
                value_kind=ValueKind.CATEGORICAL,
                description="Terminal pipeline state.",
            ),
        ),
    )


def test_value_matches_kind_preserves_bool_integer_boundary() -> None:
    assert value_matches_kind(True, ValueKind.BOOLEAN)
    assert not value_matches_kind(True, ValueKind.INTEGER)
    assert value_matches_kind(3, ValueKind.INTEGER)
    assert value_matches_kind(3, ValueKind.REAL)
    assert value_matches_kind(3.5, ValueKind.REAL)
    assert not value_matches_kind(False, ValueKind.REAL)
    assert value_matches_kind("passed", ValueKind.CATEGORICAL)
    assert not value_matches_kind("   ", ValueKind.CATEGORICAL)


def test_snapshot_validates_known_observables_and_value_types() -> None:
    snapshot = DomainSnapshot(
        domain_id="ci-pipeline",
        snapshot_id="snapshot-001",
        captured_at_label="before-transfer",
        values=(
            MeasuredValue(
                observable_name="Worker Count",
                value=4,
                evidence_id="evidence-worker-count",
            ),
            MeasuredValue(
                observable_name="Completion Time",
                value=121.5,
                evidence_id="evidence-completion-time",
            ),
        ),
        source="unit-test-fixture",
    )

    assert validate_snapshot_against_domain(make_ci_domain(), snapshot) == ()


def test_snapshot_rejects_unknown_observable_and_wrong_type() -> None:
    snapshot = DomainSnapshot(
        domain_id="ci-pipeline",
        snapshot_id="snapshot-002",
        captured_at_label="after-transfer",
        values=(
            MeasuredValue(
                observable_name="Worker Count",
                value=True,
                evidence_id="bad-worker-count",
            ),
            MeasuredValue(
                observable_name="Missing Observable",
                value="unknown",
                evidence_id="bad-missing-observable",
            ),
        ),
        source="unit-test-fixture",
    )

    errors = validate_snapshot_against_domain(make_ci_domain(), snapshot)

    assert "value for 'Worker Count' does not match declared kind 'integer'" in errors
    assert "unknown measured observable 'Missing Observable'" in errors


def test_snapshot_rejects_duplicate_measurements() -> None:
    snapshot = DomainSnapshot(
        domain_id="ci-pipeline",
        snapshot_id="snapshot-003",
        captured_at_label="duplicate-check",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value=120.0,
                evidence_id="first-completion-time",
            ),
            MeasuredValue(
                observable_name="completion   time",
                value=118.0,
                evidence_id="second-completion-time",
            ),
        ),
        source="unit-test-fixture",
    )

    errors = validate_snapshot_against_domain(make_ci_domain(), snapshot)

    assert "duplicate measured observable 'completion_time'" in errors


def test_intervention_record_allows_only_intervention_observables() -> None:
    intervention = InterventionRecord(
        domain_id="ci-pipeline",
        intervention_id="intervention-001",
        values=(
            MeasuredValue(
                observable_name="Worker Count",
                value=8,
                evidence_id="evidence-worker-intervention",
            ),
        ),
        rationale="Increase workers to test whether completion time improves.",
    )

    assert validate_intervention_against_domain(make_ci_domain(), intervention) == ()


def test_intervention_record_rejects_output_observable() -> None:
    intervention = InterventionRecord(
        domain_id="ci-pipeline",
        intervention_id="intervention-002",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value=90.0,
                evidence_id="bad-output-as-intervention",
            ),
        ),
        rationale="This incorrectly writes the output as an intervention.",
    )

    errors = validate_intervention_against_domain(make_ci_domain(), intervention)

    assert (
        "intervention value 'Completion Time' must target an intervention observable"
        in errors
    )


def test_outcome_record_allows_output_and_state_observables() -> None:
    outcome = OutcomeRecord(
        domain_id="ci-pipeline",
        outcome_id="outcome-001",
        observed_after_intervention_id="intervention-001",
        values=(
            MeasuredValue(
                observable_name="Completion Time",
                value=102.0,
                evidence_id="evidence-completion-outcome",
            ),
            MeasuredValue(
                observable_name="Pipeline State",
                value="passed",
                evidence_id="evidence-state-outcome",
            ),
        ),
        result_summary="Pipeline completed successfully after the intervention.",
    )

    assert validate_outcome_against_domain(make_ci_domain(), outcome) == ()


def test_outcome_record_rejects_intervention_observable() -> None:
    outcome = OutcomeRecord(
        domain_id="ci-pipeline",
        outcome_id="outcome-002",
        observed_after_intervention_id="intervention-001",
        values=(
            MeasuredValue(
                observable_name="Worker Count",
                value=8,
                evidence_id="bad-intervention-as-outcome",
            ),
        ),
        result_summary="Incorrectly reported intervention as outcome.",
    )

    errors = validate_outcome_against_domain(make_ci_domain(), outcome)

    assert (
        "outcome value 'Worker Count' must target an output or state observable"
        in errors
    )
