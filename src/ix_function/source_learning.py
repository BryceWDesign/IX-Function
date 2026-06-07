"""Source-domain learning evidence for IX-Function transfer trials.

The source-learning layer records why a causal function has bounded support in
its source domain before IX-Function attempts cross-domain transfer. It does not
prove universality, AGI, deployment readiness, or automatic target-domain reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.causal_function import CausalFunction, validate_causal_function
from ix_function.domain import DomainProfile, validate_domain_profile
from ix_function.observation import (
    DomainSnapshot,
    InterventionRecord,
    OutcomeRecord,
    validate_intervention_against_domain,
    validate_outcome_against_domain,
    validate_snapshot_against_domain,
)


class SourceLearningStatus(StrEnum):
    """Review status for source-domain support evidence."""

    SUPPORTED = "supported"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SourceLearningTrial:
    """Committed source-domain records for one causal learning claim."""

    trial_id: str
    source_domain: DomainProfile
    causal_function: CausalFunction
    baseline_snapshot: DomainSnapshot
    intervention: InterventionRecord
    outcome: OutcomeRecord
    support_reasons: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceLearningEvidence:
    """Bounded evidence that a causal function is source-supported."""

    evidence_id: str
    trial_id: str
    function_id: str
    source_domain_id: str
    status: SourceLearningStatus
    support_reasons: tuple[str, ...]
    confidence_delta: float
    uncertainty_notes: tuple[str, ...]
    blocking_errors: tuple[str, ...] = ()

    def is_supported(self) -> bool:
        """Return whether the source-domain evidence is usable for transfer."""

        return self.status is SourceLearningStatus.SUPPORTED


def evaluate_source_learning_trial(
    trial: SourceLearningTrial,
) -> SourceLearningEvidence:
    """Evaluate source-domain support without making target-domain claims."""

    blocking_errors = validate_source_learning_trial(trial)
    if blocking_errors:
        return SourceLearningEvidence(
            evidence_id=f"{trial.trial_id}:source-learning",
            trial_id=trial.trial_id,
            function_id=trial.causal_function.function_id,
            source_domain_id=trial.source_domain.domain_id,
            status=SourceLearningStatus.BLOCKED,
            support_reasons=trial.support_reasons,
            confidence_delta=0.0,
            uncertainty_notes=trial.uncertainty_notes,
            blocking_errors=blocking_errors,
        )

    return SourceLearningEvidence(
        evidence_id=f"{trial.trial_id}:source-learning",
        trial_id=trial.trial_id,
        function_id=trial.causal_function.function_id,
        source_domain_id=trial.source_domain.domain_id,
        status=SourceLearningStatus.SUPPORTED,
        support_reasons=trial.support_reasons,
        confidence_delta=0.05,
        uncertainty_notes=trial.uncertainty_notes,
        blocking_errors=(),
    )


def validate_source_learning_trial(
    trial: SourceLearningTrial,
) -> tuple[str, ...]:
    """Return blocking errors for source-domain causal learning evidence."""

    errors: list[str] = []
    if not trial.trial_id.strip():
        errors.append("trial_id must not be empty")

    errors.extend(
        f"source domain: {error}"
        for error in validate_domain_profile(trial.source_domain)
    )
    errors.extend(
        f"causal function: {error}"
        for error in validate_causal_function(trial.causal_function)
    )
    errors.extend(
        f"source baseline: {error}"
        for error in validate_snapshot_against_domain(
            trial.source_domain,
            trial.baseline_snapshot,
        )
    )
    errors.extend(
        f"source intervention: {error}"
        for error in validate_intervention_against_domain(
            trial.source_domain,
            trial.intervention,
        )
    )
    errors.extend(
        f"source outcome: {error}"
        for error in validate_outcome_against_domain(
            trial.source_domain,
            trial.outcome,
        )
    )

    if (
        trial.causal_function.learned_from_domain_id is not None
        and trial.causal_function.learned_from_domain_id
        != trial.source_domain.domain_id
    ):
        errors.append(
            "causal_function learned_from_domain_id must match source domain_id"
        )
    if (
        trial.outcome.observed_after_intervention_id
        != trial.intervention.intervention_id
    ):
        errors.append("source outcome must reference source intervention")
    if not trial.support_reasons:
        errors.append("support_reasons must not be empty")
    elif any(not reason.strip() for reason in trial.support_reasons):
        errors.append("support_reasons must not contain empty reasons")
    if not trial.uncertainty_notes:
        errors.append("uncertainty_notes must not be empty")
    elif any(not note.strip() for note in trial.uncertainty_notes):
        errors.append("uncertainty_notes must not contain empty notes")

    return tuple(errors)
