"""Source-domain learning evidence for IX-Function.

This module records whether a causal function has source-domain support before
it is allowed to become a transfer candidate. It does not prove the function is
universal; it only creates reviewable evidence that the function was learned or
supported in a bounded source domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.causal_function import CausalFunction
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
    """Status of a causal function inside its source-domain evidence record."""

    SUPPORTED = "supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SourceLearningEvidence:
    """Evidence that a causal function has support in a source domain."""

    evidence_id: str
    function_id: str
    source_domain_id: str
    baseline_snapshot_id: str
    intervention_id: str
    outcome_id: str
    status: SourceLearningStatus
    confidence_delta: float
    support_reasons: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]
    blocking_errors: tuple[str, ...] = ()

    def adjusted_confidence(self, prior_confidence: float) -> float:
        """Return confidence after applying the bounded source-learning delta."""

        return min(1.0, max(0.0, round(prior_confidence + self.confidence_delta, 6)))


@dataclass(frozen=True, slots=True)
class SourceLearningTrial:
    """A bounded source-domain trial used to support a causal function."""

    trial_id: str
    source_domain: DomainProfile
    causal_function: CausalFunction
    baseline_snapshot: DomainSnapshot
    intervention: InterventionRecord
    outcome: OutcomeRecord
    support_reasons: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]


def evaluate_source_learning_trial(
    trial: SourceLearningTrial,
    *,
    strong_support_threshold: int = 2,
) -> SourceLearningEvidence:
    """Evaluate whether a source trial supports a causal function.

    The function is deliberately conservative. It validates the domain and all
    observations first, blocks malformed records, then assigns bounded support
    based on declared reasons and uncertainty instead of pretending certainty.
    """

    blocking_errors = list(validate_source_learning_trial(trial))
    if blocking_errors:
        return SourceLearningEvidence(
            evidence_id=f"{trial.trial_id}:source-learning",
            function_id=trial.causal_function.function_id,
            source_domain_id=trial.source_domain.domain_id,
            baseline_snapshot_id=trial.baseline_snapshot.snapshot_id,
            intervention_id=trial.intervention.intervention_id,
            outcome_id=trial.outcome.outcome_id,
            status=SourceLearningStatus.BLOCKED,
            confidence_delta=-0.1,
            support_reasons=(),
            uncertainty_notes=(
                "Source learning was blocked because the trial record is invalid.",
            ),
            blocking_errors=tuple(blocking_errors),
        )

    support_count = len(trial.support_reasons)
    uncertainty_count = len(trial.uncertainty_notes)

    if support_count >= strong_support_threshold and uncertainty_count <= 2:
        status = SourceLearningStatus.SUPPORTED
        confidence_delta = 0.08
    else:
        status = SourceLearningStatus.WEAKLY_SUPPORTED
        confidence_delta = 0.03

    if uncertainty_count >= support_count:
        confidence_delta = min(confidence_delta, 0.01)

    return SourceLearningEvidence(
        evidence_id=f"{trial.trial_id}:source-learning",
        function_id=trial.causal_function.function_id,
        source_domain_id=trial.source_domain.domain_id,
        baseline_snapshot_id=trial.baseline_snapshot.snapshot_id,
        intervention_id=trial.intervention.intervention_id,
        outcome_id=trial.outcome.outcome_id,
        status=status,
        confidence_delta=confidence_delta,
        support_reasons=trial.support_reasons,
        uncertainty_notes=trial.uncertainty_notes,
        blocking_errors=(),
    )


def validate_source_learning_trial(trial: SourceLearningTrial) -> tuple[str, ...]:
    """Return validation errors for a source-learning trial."""

    errors: list[str] = []
    if not trial.trial_id.strip():
        errors.append("trial_id must not be empty")
    if trial.causal_function.learned_from_domain_id not in {
        None,
        trial.source_domain.domain_id,
    }:
        errors.append(
            "causal_function learned_from_domain_id must be empty or match "
            "source domain_id"
        )
    if not trial.support_reasons:
        errors.append("support_reasons must not be empty")
    elif any(not reason.strip() for reason in trial.support_reasons):
        errors.append("support_reasons must not contain empty reasons")
    if not trial.uncertainty_notes:
        errors.append("uncertainty_notes must not be empty")
    elif any(not note.strip() for note in trial.uncertainty_notes):
        errors.append("uncertainty_notes must not contain empty notes")

    errors.extend(
        f"source domain: {error}"
        for error in validate_domain_profile(trial.source_domain)
    )
    errors.extend(
        f"baseline snapshot: {error}"
        for error in validate_snapshot_against_domain(
            trial.source_domain,
            trial.baseline_snapshot,
        )
    )
    errors.extend(
        f"intervention: {error}"
        for error in validate_intervention_against_domain(
            trial.source_domain,
            trial.intervention,
        )
    )
    errors.extend(
        f"outcome: {error}"
        for error in validate_outcome_against_domain(
            trial.source_domain,
            trial.outcome,
        )
    )

    if trial.baseline_snapshot.domain_id != trial.source_domain.domain_id:
        errors.append("baseline snapshot must belong to source domain")
    if trial.intervention.domain_id != trial.source_domain.domain_id:
        errors.append("intervention must belong to source domain")
    if trial.outcome.domain_id != trial.source_domain.domain_id:
        errors.append("outcome must belong to source domain")
    if (
        trial.outcome.observed_after_intervention_id
        != trial.intervention.intervention_id
    ):
        errors.append("outcome must reference the trial intervention")

    return tuple(errors)
