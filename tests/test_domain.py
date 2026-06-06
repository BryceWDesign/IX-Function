from __future__ import annotations

from ix_function.domain import (
    DomainConstraint,
    DomainKind,
    DomainPair,
    DomainProfile,
    Observable,
    ObservableRole,
    ValueKind,
    validate_domain_pair,
    validate_domain_profile,
)


def make_flow_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="water-flow",
        name="Water Flow",
        kind=DomainKind.FLOW,
        summary="A bounded flow domain with capacity and downstream output.",
        observables=(
            Observable(
                name="Input Rate",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.REAL,
                description="Water entering the upstream segment per second.",
                unit="liters_per_second",
            ),
            Observable(
                name="Pipe Capacity",
                role=ObservableRole.CONSTRAINT,
                value_kind=ValueKind.REAL,
                description="Maximum downstream carrying capacity.",
                unit="liters_per_second",
            ),
            Observable(
                name="Output Rate",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Water leaving the downstream segment per second.",
                unit="liters_per_second",
            ),
        ),
        constraints=(
            DomainConstraint(
                constraint_id="FLOW-CAPACITY",
                description="Output cannot exceed downstream pipe capacity.",
                affected_observables=("Pipe Capacity", "Output Rate"),
            ),
        ),
    )


def make_computing_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="ci-pipeline",
        name="CI Pipeline",
        kind=DomainKind.COMPUTING,
        summary="A test pipeline with parallelism and a limiting test stage.",
        observables=(
            Observable(
                name="Worker Count",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.INTEGER,
                description="Number of workers assigned to the pipeline.",
            ),
            Observable(
                name="Slowest Stage Time",
                role=ObservableRole.CONSTRAINT,
                value_kind=ValueKind.REAL,
                description="Duration of the slowest required test stage.",
                unit="seconds",
            ),
            Observable(
                name="Completion Time",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Total time until the pipeline completes.",
                unit="seconds",
            ),
        ),
    )


def test_observable_normalizes_names_for_evidence_keys() -> None:
    observable = Observable(
        name="  Output   Rate ",
        role=ObservableRole.OUTPUT,
        value_kind=ValueKind.REAL,
        description="Measured output rate.",
    )

    assert observable.normalized_name() == "output_rate"


def test_domain_profile_indexes_observables_by_normalized_name() -> None:
    profile = make_flow_domain()

    assert set(profile.observable_index()) == {
        "input_rate",
        "pipe_capacity",
        "output_rate",
    }
    assert profile.has_role(ObservableRole.INTERVENTION)
    assert len(profile.observables_by_role(ObservableRole.OUTPUT)) == 1


def test_validate_domain_profile_accepts_complete_domain() -> None:
    assert validate_domain_profile(make_flow_domain()) == ()


def test_validate_domain_profile_rejects_duplicate_observables() -> None:
    profile = DomainProfile(
        domain_id="bad-domain",
        name="Bad Domain",
        kind=DomainKind.SYNTHETIC,
        summary="A malformed domain profile.",
        observables=(
            Observable(
                name="Output Rate",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="First output.",
            ),
            Observable(
                name="output   rate",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.REAL,
                description="Duplicate name after normalization.",
            ),
        ),
    )

    errors = validate_domain_profile(profile)

    assert "observable names must be unique after normalization" in errors


def test_validate_domain_profile_rejects_unknown_constraint_references() -> None:
    profile = DomainProfile(
        domain_id="bad-constraint-domain",
        name="Bad Constraint Domain",
        kind=DomainKind.SYNTHETIC,
        summary="A domain with a broken constraint reference.",
        observables=(
            Observable(
                name="Intervention",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.REAL,
                description="A valid intervention.",
            ),
            Observable(
                name="Output",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="A valid output.",
            ),
        ),
        constraints=(
            DomainConstraint(
                constraint_id="BROKEN",
                description="References a missing observable.",
                affected_observables=("Missing Observable",),
            ),
        ),
    )

    errors = validate_domain_profile(profile)

    assert (
        "constraint 'BROKEN' references unknown observable 'Missing Observable'"
        in errors
    )


def test_validate_domain_pair_accepts_cross_domain_pair() -> None:
    pair = DomainPair(
        source=make_flow_domain(),
        target=make_computing_domain(),
        transfer_purpose="Test bottleneck transfer from flow to CI.",
    )

    assert pair.is_cross_domain()
    assert validate_domain_pair(pair) == ()


def test_validate_domain_pair_rejects_same_domain_kind() -> None:
    pair = DomainPair(
        source=make_flow_domain(),
        target=DomainProfile(
            domain_id="second-flow-domain",
            name="Second Flow Domain",
            kind=DomainKind.FLOW,
            summary="Another flow-shaped domain.",
            observables=make_flow_domain().observables,
        ),
        transfer_purpose="This should not count as cross-domain transfer.",
    )

    errors = validate_domain_pair(pair)

    assert "source and target must use different domain kinds" in errors
