"""Reusable causal function models for IX-Function.

A causal function is the portable structure IX-Function tries to move from one
domain into another. It is intentionally more abstract than a domain profile and
more constrained than free-form model text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.domain import ObservableRole


class CausalFamily(StrEnum):
    """Supported causal families for early cross-domain transfer trials."""

    BOTTLENECK = "bottleneck"
    FEEDBACK_DELAY = "feedback_delay"
    QUARANTINE = "quarantine"
    RESOURCE_ALLOCATION = "resource_allocation"
    SATURATION = "saturation"
    THRESHOLD = "threshold"


class CausalSlotRole(StrEnum):
    """Abstract role required by a causal function."""

    CONSTRAINT = "constraint"
    CONTEXT = "context"
    INTERVENTION = "intervention"
    OUTPUT = "output"
    STATE = "state"


@dataclass(frozen=True, slots=True)
class CausalVariableSlot:
    """A portable variable slot inside a causal function."""

    slot_id: str
    role: CausalSlotRole
    description: str
    semantic_tags: tuple[str, ...]
    compatible_observable_roles: tuple[ObservableRole, ...]

    def normalized_id(self) -> str:
        """Return a stable lowercase key for mapping and evidence records."""

        return "_".join(self.slot_id.strip().lower().split())

    def normalized_tags(self) -> tuple[str, ...]:
        """Return lowercase semantic tags with empty tags removed."""

        return tuple(
            "_".join(tag.strip().lower().split())
            for tag in self.semantic_tags
            if tag.strip()
        )


@dataclass(frozen=True, slots=True)
class CausalMechanism:
    """A mechanism claim that can be transferred, predicted, and tested."""

    mechanism_id: str
    family: CausalFamily
    premise: str
    expected_effect: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CausalFunction:
    """A reusable causal structure learned or declared for transfer testing."""

    function_id: str
    name: str
    family: CausalFamily
    summary: str
    variable_slots: tuple[CausalVariableSlot, ...]
    mechanisms: tuple[CausalMechanism, ...]
    prior_confidence: float
    uncertainty_notes: tuple[str, ...]
    learned_from_domain_id: str | None = None

    def slot_index(self) -> dict[str, CausalVariableSlot]:
        """Return variable slots keyed by normalized slot identifier."""

        return {slot.normalized_id(): slot for slot in self.variable_slots}

    def slots_by_role(self, role: CausalSlotRole) -> tuple[CausalVariableSlot, ...]:
        """Return variable slots using the requested abstract role."""

        return tuple(slot for slot in self.variable_slots if slot.role is role)

    def required_semantic_tags(self) -> tuple[str, ...]:
        """Return unique normalized semantic tags required by the function."""

        tags: set[str] = set()
        for slot in self.variable_slots:
            tags.update(slot.normalized_tags())
        return tuple(sorted(tags))


@dataclass(frozen=True, slots=True)
class CausalSignature:
    """Compact transfer signature derived from a causal function."""

    function_id: str
    family: CausalFamily
    slot_count: int
    required_roles: tuple[CausalSlotRole, ...]
    required_semantic_tags: tuple[str, ...]
    mechanism_ids: tuple[str, ...]


def build_causal_signature(function: CausalFunction) -> CausalSignature:
    """Build a compact signature used by mapping and falsification layers."""

    required_roles = tuple(sorted({slot.role for slot in function.variable_slots}))
    mechanism_ids = tuple(mechanism.mechanism_id for mechanism in function.mechanisms)
    return CausalSignature(
        function_id=function.function_id,
        family=function.family,
        slot_count=len(function.variable_slots),
        required_roles=required_roles,
        required_semantic_tags=function.required_semantic_tags(),
        mechanism_ids=mechanism_ids,
    )


def validate_causal_slot(slot: CausalVariableSlot) -> tuple[str, ...]:
    """Return validation errors for a causal variable slot."""

    errors: list[str] = []
    if not slot.slot_id.strip():
        errors.append("slot_id must not be empty")
    if not slot.description.strip():
        errors.append(f"slot {slot.slot_id!r} must have a description")
    if not slot.normalized_tags():
        errors.append(f"slot {slot.slot_id!r} must include semantic_tags")
    if not slot.compatible_observable_roles:
        errors.append(
            f"slot {slot.slot_id!r} must include compatible_observable_roles"
        )
    return tuple(errors)


def validate_causal_mechanism(mechanism: CausalMechanism) -> tuple[str, ...]:
    """Return validation errors for a causal mechanism."""

    errors: list[str] = []
    if not mechanism.mechanism_id.strip():
        errors.append("mechanism_id must not be empty")
    if not mechanism.premise.strip():
        errors.append(f"mechanism {mechanism.mechanism_id!r} must have a premise")
    if not mechanism.expected_effect.strip():
        errors.append(
            f"mechanism {mechanism.mechanism_id!r} must have an expected_effect"
        )
    if not mechanism.assumptions:
        errors.append(
            f"mechanism {mechanism.mechanism_id!r} must declare assumptions"
        )
    elif any(not assumption.strip() for assumption in mechanism.assumptions):
        errors.append(
            f"mechanism {mechanism.mechanism_id!r} has an empty assumption"
        )
    return tuple(errors)


def validate_causal_function(function: CausalFunction) -> tuple[str, ...]:
    """Return validation errors for a causal function."""

    errors: list[str] = []
    if not function.function_id.strip():
        errors.append("function_id must not be empty")
    if not function.name.strip():
        errors.append("name must not be empty")
    if not function.summary.strip():
        errors.append("summary must not be empty")
    if not 0.0 <= function.prior_confidence <= 1.0:
        errors.append("prior_confidence must be between 0.0 and 1.0")
    if not function.uncertainty_notes:
        errors.append("uncertainty_notes must not be empty")
    elif any(not note.strip() for note in function.uncertainty_notes):
        errors.append("uncertainty_notes must not contain empty notes")

    if not function.variable_slots:
        errors.append("at least one variable slot is required")
    if not function.mechanisms:
        errors.append("at least one causal mechanism is required")

    slot_ids = [slot.normalized_id() for slot in function.variable_slots]
    if len(set(slot_ids)) != len(slot_ids):
        errors.append("variable slot identifiers must be unique after normalization")

    if function.variable_slots and not function.slots_by_role(
        CausalSlotRole.INTERVENTION
    ):
        errors.append("at least one intervention slot is required")
    if function.variable_slots and not function.slots_by_role(CausalSlotRole.OUTPUT):
        errors.append("at least one output slot is required")

    for slot in function.variable_slots:
        errors.extend(validate_causal_slot(slot))

    mechanism_ids = [mechanism.mechanism_id for mechanism in function.mechanisms]
    if len(set(mechanism_ids)) != len(mechanism_ids):
        errors.append("mechanism identifiers must be unique")

    for mechanism in function.mechanisms:
        errors.extend(validate_causal_mechanism(mechanism))
        if mechanism.family is not function.family:
            errors.append(
                f"mechanism {mechanism.mechanism_id!r} family "
                f"{mechanism.family.value!r} does not match function family "
                f"{function.family.value!r}"
            )

    return tuple(errors)
