"""Observation records for IX-Function domain trials.

This module keeps measured values separate from causal interpretations. A value
can be observed, validated, and preserved as evidence before any transfer engine
is allowed to claim that it learned or predicted a causal relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, assert_never

from ix_function.domain import DomainProfile, ObservableRole, ValueKind

ObservationValue: TypeAlias = bool | int | float | str


@dataclass(frozen=True, slots=True)
class MeasuredValue:
    """A single measured value tied to a named domain observable."""

    observable_name: str
    value: ObservationValue
    evidence_id: str
    notes: str = ""

    def normalized_observable_name(self) -> str:
        """Return the normalized observable key used by domain profiles."""

        return "_".join(self.observable_name.strip().lower().split())


@dataclass(frozen=True, slots=True)
class DomainSnapshot:
    """A set of measured values captured for a domain at one logical moment."""

    domain_id: str
    snapshot_id: str
    captured_at_label: str
    values: tuple[MeasuredValue, ...]
    source: str

    def value_index(self) -> dict[str, MeasuredValue]:
        """Return measured values keyed by normalized observable name."""

        return {value.normalized_observable_name(): value for value in self.values}


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """A committed intervention before the outcome is observed."""

    domain_id: str
    intervention_id: str
    values: tuple[MeasuredValue, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """Observed outcome values produced after a named intervention."""

    domain_id: str
    outcome_id: str
    observed_after_intervention_id: str
    values: tuple[MeasuredValue, ...]
    result_summary: str


def value_matches_kind(value: ObservationValue, value_kind: ValueKind) -> bool:
    """Return whether a measured value matches the observable value kind."""

    if value_kind is ValueKind.BOOLEAN:
        return isinstance(value, bool)
    if value_kind is ValueKind.CATEGORICAL:
        return isinstance(value, str) and bool(value.strip())
    if value_kind is ValueKind.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if value_kind is ValueKind.REAL:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_kind is ValueKind.TEXT:
        return isinstance(value, str)
    assert_never(value_kind)


def validate_snapshot_against_domain(
    profile: DomainProfile,
    snapshot: DomainSnapshot,
) -> tuple[str, ...]:
    """Return validation errors for a domain snapshot."""

    errors: list[str] = []
    if snapshot.domain_id != profile.domain_id:
        errors.append(
            f"snapshot domain_id {snapshot.domain_id!r} does not match "
            f"profile domain_id {profile.domain_id!r}"
        )
    if not snapshot.snapshot_id.strip():
        errors.append("snapshot_id must not be empty")
    if not snapshot.captured_at_label.strip():
        errors.append("captured_at_label must not be empty")
    if not snapshot.source.strip():
        errors.append("source must not be empty")
    if not snapshot.values:
        errors.append("snapshot must include at least one measured value")

    observable_index = profile.observable_index()
    seen_names: set[str] = set()
    for measured in snapshot.values:
        normalized_name = measured.normalized_observable_name()
        if not normalized_name:
            errors.append("measured observable_name must not be empty")
            continue
        if normalized_name in seen_names:
            errors.append(f"duplicate measured observable {normalized_name!r}")
        seen_names.add(normalized_name)

        observable = observable_index.get(normalized_name)
        if observable is None:
            errors.append(f"unknown measured observable {measured.observable_name!r}")
            continue
        if not measured.evidence_id.strip():
            errors.append(
                f"evidence_id must not be empty for {measured.observable_name!r}"
            )
        if not value_matches_kind(measured.value, observable.value_kind):
            errors.append(
                f"value for {measured.observable_name!r} does not match "
                f"declared kind {observable.value_kind.value!r}"
            )

    return tuple(errors)


def validate_intervention_against_domain(
    profile: DomainProfile,
    intervention: InterventionRecord,
) -> tuple[str, ...]:
    """Return validation errors for an intervention record."""

    snapshot_errors = validate_snapshot_against_domain(
        profile,
        DomainSnapshot(
            domain_id=intervention.domain_id,
            snapshot_id=intervention.intervention_id,
            captured_at_label="intervention",
            values=intervention.values,
            source="intervention_record",
        ),
    )
    errors = list(snapshot_errors)
    if not intervention.rationale.strip():
        errors.append("intervention rationale must not be empty")

    observable_index = profile.observable_index()
    for measured in intervention.values:
        observable = observable_index.get(measured.normalized_observable_name())
        if (
            observable is not None
            and observable.role is not ObservableRole.INTERVENTION
        ):
            errors.append(
                f"intervention value {measured.observable_name!r} must target an "
                "intervention observable"
            )

    return tuple(errors)


def validate_outcome_against_domain(
    profile: DomainProfile,
    outcome: OutcomeRecord,
) -> tuple[str, ...]:
    """Return validation errors for an outcome record."""

    snapshot_errors = validate_snapshot_against_domain(
        profile,
        DomainSnapshot(
            domain_id=outcome.domain_id,
            snapshot_id=outcome.outcome_id,
            captured_at_label="outcome",
            values=outcome.values,
            source="outcome_record",
        ),
    )
    errors = list(snapshot_errors)
    if not outcome.observed_after_intervention_id.strip():
        errors.append("observed_after_intervention_id must not be empty")
    if not outcome.result_summary.strip():
        errors.append("result_summary must not be empty")

    observable_index = profile.observable_index()
    for measured in outcome.values:
        observable = observable_index.get(measured.normalized_observable_name())
        if observable is not None and observable.role not in {
            ObservableRole.OUTPUT,
            ObservableRole.STATE,
        }:
            errors.append(
                f"outcome value {measured.observable_name!r} must target an "
                "output or state observable"
            )

    return tuple(errors)
