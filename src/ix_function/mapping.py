"""Transfer mapping engine for causal functions and target domains.

The mapping layer proposes how abstract causal slots could attach to concrete
observables in a target domain. It does not execute transfer and does not treat
mapping success as proof. Ambiguity is preserved so later prediction and
falsification layers can weaken or block claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from ix_function.causal_function import CausalFunction, CausalVariableSlot
from ix_function.domain import DomainProfile, Observable

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class MappingQuality(StrEnum):
    """Quality state for a proposed source-function to target-domain mapping."""

    COMPLETE = "complete"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class SlotCandidate:
    """Candidate observable for a single causal variable slot."""

    slot_id: str
    observable_name: str
    role_compatible: bool
    semantic_overlap: tuple[str, ...]
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SlotMapping:
    """Committed mapping from one causal slot to one target observable."""

    slot_id: str
    observable_name: str
    score: float
    uncertainty_notes: tuple[str, ...]

    def normalized_slot_id(self) -> str:
        """Return the stable normalized slot identifier."""

        return "_".join(self.slot_id.strip().lower().split())

    def normalized_observable_name(self) -> str:
        """Return the stable normalized observable name."""

        return "_".join(self.observable_name.strip().lower().split())


@dataclass(frozen=True, slots=True)
class TransferMapping:
    """A proposed mapping from a causal function to a target domain."""

    function_id: str
    target_domain_id: str
    slot_mappings: tuple[SlotMapping, ...]
    quality: MappingQuality
    coverage_score: float
    ambiguity_score: float
    warnings: tuple[str, ...]

    def mapping_index(self) -> dict[str, SlotMapping]:
        """Return slot mappings keyed by normalized slot identifier."""

        return {
            mapping.normalized_slot_id(): mapping
            for mapping in self.slot_mappings
        }

    def is_usable_for_prediction(self) -> bool:
        """Return whether the mapping can support a bounded prediction trial."""

        return (
            self.quality in {MappingQuality.COMPLETE, MappingQuality.AMBIGUOUS}
            and self.coverage_score == 1.0
        )


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Return deterministic lowercase tokens for semantic matching."""

    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def observable_semantic_tokens(observable: Observable) -> tuple[str, ...]:
    """Return tokens from the observable's inspectable semantic fields."""

    joined = " ".join(
        part
        for part in (
            observable.name,
            observable.description,
            observable.unit or "",
        )
        if part
    )
    return normalized_tokens(joined)


def build_slot_candidates(
    slot: CausalVariableSlot,
    domain: DomainProfile,
) -> tuple[SlotCandidate, ...]:
    """Rank target-domain observables for one causal variable slot."""

    candidates: list[SlotCandidate] = []
    slot_tags = set(slot.normalized_tags())
    for observable in domain.observables:
        role_compatible = observable.role in slot.compatible_observable_roles
        observable_tokens = set(observable_semantic_tokens(observable))
        overlap = tuple(sorted(slot_tags & observable_tokens))
        role_score = 0.65 if role_compatible else 0.0
        semantic_score = 0.0
        if slot_tags:
            semantic_score = 0.35 * (len(overlap) / len(slot_tags))
        score = round(role_score + semantic_score, 6)

        reasons: list[str] = []
        if role_compatible:
            reasons.append(
                f"observable role {observable.role.value!r} is compatible"
            )
        if overlap:
            reasons.append(f"semantic overlap: {', '.join(overlap)}")
        if not reasons:
            reasons.append("no role or semantic support")

        candidates.append(
            SlotCandidate(
                slot_id=slot.slot_id,
                observable_name=observable.name,
                role_compatible=role_compatible,
                semantic_overlap=overlap,
                score=score,
                reasons=tuple(reasons),
            )
        )

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.observable_name),
        )
    )


def propose_transfer_mapping(
    function: CausalFunction,
    target_domain: DomainProfile,
    *,
    min_slot_score: float = 0.65,
    ambiguity_delta: float = 0.08,
) -> TransferMapping:
    """Propose a causal-slot to target-observable mapping.

    The algorithm is intentionally transparent and conservative. It only uses
    declared role compatibility and token overlap from observable descriptions.
    It does not contain domain-pair special cases.
    """

    warnings: list[str] = []
    slot_mappings: list[SlotMapping] = []
    ambiguous_slots = 0

    for slot in function.variable_slots:
        candidates = build_slot_candidates(slot, target_domain)
        viable = tuple(
            candidate for candidate in candidates if candidate.score >= min_slot_score
        )
        if not viable:
            warnings.append(
                f"slot {slot.slot_id!r} has no target observable above score "
                f"threshold {min_slot_score:.2f}"
            )
            continue

        best = viable[0]
        uncertainty_notes: list[str] = []
        if len(viable) > 1 and viable[0].score - viable[1].score <= ambiguity_delta:
            ambiguous_slots += 1
            uncertainty_notes.append(
                f"nearest alternative {viable[1].observable_name!r} scored "
                f"{viable[1].score:.3f}"
            )

        if not best.semantic_overlap:
            uncertainty_notes.append(
                "mapping is role-compatible but has no semantic tag overlap"
            )

        slot_mappings.append(
            SlotMapping(
                slot_id=slot.slot_id,
                observable_name=best.observable_name,
                score=best.score,
                uncertainty_notes=tuple(uncertainty_notes),
            )
        )

    mapped_slots = len(slot_mappings)
    total_slots = len(function.variable_slots)
    coverage_score = round(mapped_slots / total_slots, 6) if total_slots else 0.0
    ambiguity_score = round(ambiguous_slots / mapped_slots, 6) if mapped_slots else 0.0

    duplicate_observables = _duplicate_normalized_observables(slot_mappings)
    for duplicate in duplicate_observables:
        warnings.append(
            f"target observable {duplicate!r} is mapped by multiple causal slots"
        )

    if coverage_score < 1.0 or duplicate_observables:
        quality = MappingQuality.INSUFFICIENT
    elif ambiguous_slots:
        quality = MappingQuality.AMBIGUOUS
    else:
        quality = MappingQuality.COMPLETE

    return TransferMapping(
        function_id=function.function_id,
        target_domain_id=target_domain.domain_id,
        slot_mappings=tuple(slot_mappings),
        quality=quality,
        coverage_score=coverage_score,
        ambiguity_score=ambiguity_score,
        warnings=tuple(warnings),
    )


def validate_transfer_mapping(mapping: TransferMapping) -> tuple[str, ...]:
    """Return validation errors for a proposed transfer mapping."""

    errors: list[str] = []
    if not mapping.function_id.strip():
        errors.append("function_id must not be empty")
    if not mapping.target_domain_id.strip():
        errors.append("target_domain_id must not be empty")
    if not 0.0 <= mapping.coverage_score <= 1.0:
        errors.append("coverage_score must be between 0.0 and 1.0")
    if not 0.0 <= mapping.ambiguity_score <= 1.0:
        errors.append("ambiguity_score must be between 0.0 and 1.0")

    slot_ids = [
        mapping_item.normalized_slot_id()
        for mapping_item in mapping.slot_mappings
    ]
    if len(set(slot_ids)) != len(slot_ids):
        errors.append("slot mappings must use unique slot identifiers")

    duplicate_observables = _duplicate_normalized_observables(mapping.slot_mappings)
    for duplicate in duplicate_observables:
        errors.append(
            f"target observable {duplicate!r} must not be mapped more than once"
        )

    for mapping_item in mapping.slot_mappings:
        if not mapping_item.slot_id.strip():
            errors.append("slot mapping slot_id must not be empty")
        if not mapping_item.observable_name.strip():
            errors.append("slot mapping observable_name must not be empty")
        if not 0.0 <= mapping_item.score <= 1.0:
            errors.append(
                f"score for slot {mapping_item.slot_id!r} must be between 0.0 and 1.0"
            )
        if any(not note.strip() for note in mapping_item.uncertainty_notes):
            errors.append(
                f"uncertainty note for slot {mapping_item.slot_id!r} must not be empty"
            )

    return tuple(errors)


def _duplicate_normalized_observables(
    mappings: tuple[SlotMapping, ...] | list[SlotMapping],
) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for mapping in mappings:
        normalized = mapping.normalized_observable_name()
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    return tuple(sorted(duplicates))
