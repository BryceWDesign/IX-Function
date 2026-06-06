"""Domain and observable models for cross-domain causal transfer.

IX-Function treats a domain as a bounded measurement space. A transfer attempt
must name the source domain, target domain, observable variables, constraints,
and intervention surface before any prediction can count as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DomainKind(str, Enum):
    """High-level domain family used to prevent accidental same-domain trials."""

    COMPUTING = "computing"
    CONTROL = "control"
    ENERGY = "energy"
    FLOW = "flow"
    LOGISTICS = "logistics"
    MEMORY = "memory"
    SECURITY = "security"
    SYNTHETIC = "synthetic"


class ObservableRole(str, Enum):
    """Role an observable plays inside a causal-transfer trial."""

    CONTEXT = "context"
    CONSTRAINT = "constraint"
    INPUT = "input"
    INTERVENTION = "intervention"
    OUTPUT = "output"
    STATE = "state"


class ValueKind(str, Enum):
    """Portable value categories for domain observables."""

    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    REAL = "real"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Observable:
    """A named variable that can be mapped, predicted, or observed."""

    name: str
    role: ObservableRole
    value_kind: ValueKind
    description: str
    unit: str | None = None

    def normalized_name(self) -> str:
        """Return a stable lowercase key for matching and evidence records."""

        return "_".join(self.name.strip().lower().split())


@dataclass(frozen=True, slots=True)
class DomainConstraint:
    """A declared limit that can invalidate or weaken a transfer attempt."""

    constraint_id: str
    description: str
    affected_observables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DomainProfile:
    """A bounded source or target domain for causal transfer."""

    domain_id: str
    name: str
    kind: DomainKind
    summary: str
    observables: tuple[Observable, ...]
    constraints: tuple[DomainConstraint, ...] = ()

    def observable_index(self) -> dict[str, Observable]:
        """Return observables keyed by normalized name."""

        return {observable.normalized_name(): observable for observable in self.observables}

    def observables_by_role(self, role: ObservableRole) -> tuple[Observable, ...]:
        """Return all observables with the requested role."""

        return tuple(observable for observable in self.observables if observable.role is role)

    def has_role(self, role: ObservableRole) -> bool:
        """Return whether the domain exposes at least one observable for a role."""

        return any(observable.role is role for observable in self.observables)


@dataclass(frozen=True, slots=True)
class DomainPair:
    """A source-target pair for testing whether transfer is cross-domain."""

    source: DomainProfile
    target: DomainProfile
    transfer_purpose: str

    def is_cross_domain(self) -> bool:
        """Return True when the trial crosses declared domain families."""

        return self.source.kind is not self.target.kind


def validate_domain_profile(profile: DomainProfile) -> tuple[str, ...]:
    """Return validation errors for a domain profile.

    Empty return means the profile is usable for a transfer trial. The function
    avoids raising so failures can be carried into reviewable evidence records.
    """

    errors: list[str] = []
    if not profile.domain_id.strip():
        errors.append("domain_id must not be empty")
    if not profile.name.strip():
        errors.append("name must not be empty")
    if not profile.summary.strip():
        errors.append("summary must not be empty")
    if not profile.observables:
        errors.append("at least one observable is required")

    normalized_names = [observable.normalized_name() for observable in profile.observables]
    if len(set(normalized_names)) != len(normalized_names):
        errors.append("observable names must be unique after normalization")

    if profile.observables and not profile.has_role(ObservableRole.OUTPUT):
        errors.append("at least one output observable is required")
    if profile.observables and not profile.has_role(ObservableRole.INTERVENTION):
        errors.append("at least one intervention observable is required")

    observable_names = set(normalized_names)
    for constraint in profile.constraints:
        if not constraint.constraint_id.strip():
            errors.append("constraint_id must not be empty")
        if not constraint.description.strip():
            errors.append(f"constraint {constraint.constraint_id!r} must have a description")
        for affected in constraint.affected_observables:
            if "_".join(affected.strip().lower().split()) not in observable_names:
                errors.append(
                    f"constraint {constraint.constraint_id!r} references unknown "
                    f"observable {affected!r}"
                )

    return tuple(errors)


def validate_domain_pair(pair: DomainPair) -> tuple[str, ...]:
    """Return validation errors for a source-target transfer pair."""

    errors = [f"source: {error}" for error in validate_domain_profile(pair.source)]
    errors.extend(f"target: {error}" for error in validate_domain_profile(pair.target))
    if not pair.transfer_purpose.strip():
        errors.append("transfer_purpose must not be empty")
    if pair.source.domain_id == pair.target.domain_id:
        errors.append("source and target domain_id must differ")
    if not pair.is_cross_domain():
        errors.append("source and target must use different domain kinds")
    return tuple(errors)
