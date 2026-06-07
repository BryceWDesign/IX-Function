"""Pre-outcome prediction records for IX-Function transfer trials.

A transfer trial only becomes meaningful when the system commits to a measurable
prediction before the outcome is observed. This module records that commitment
and blocks prediction records that leak, reference, or depend on outcome data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from ix_function.mapping import MappingQuality, TransferMapping


class PredictionDirection(StrEnum):
    """Expected direction of a target observable after intervention."""

    DECREASE = "decrease"
    INCREASE = "increase"
    LIMITED_CHANGE = "limited_change"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class PredictionStatus(StrEnum):
    """Validation status for a pre-outcome prediction record."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class PredictedObservable:
    """A measurable expected outcome for one target observable."""

    observable_name: str
    direction: PredictionDirection
    rationale: str
    expected_min: float | None = None
    expected_max: float | None = None
    tolerance: float | None = None

    def normalized_observable_name(self) -> str:
        """Return the stable normalized observable name."""

        return "_".join(self.observable_name.strip().lower().split())

    def has_numeric_range(self) -> bool:
        """Return whether the prediction includes a bounded numeric range."""

        return self.expected_min is not None and self.expected_max is not None


@dataclass(frozen=True, slots=True)
class TransferPrediction:
    """A committed prediction for a cross-domain transfer attempt."""

    prediction_id: str
    function_id: str
    target_domain_id: str
    source_evidence_id: str
    target_intervention_id: str
    mapping_quality: MappingQuality
    predicted_observables: tuple[PredictedObservable, ...]
    assumptions: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]
    created_before_outcome: bool
    forbidden_outcome_ids: tuple[str, ...] = ()

    def predicted_observable_index(self) -> dict[str, PredictedObservable]:
        """Return predicted observables keyed by normalized observable name."""

        return {
            observable.normalized_observable_name(): observable
            for observable in self.predicted_observables
        }


@dataclass(frozen=True, slots=True)
class PredictionReadiness:
    """Result of checking whether a prediction can enter a transfer trial."""

    status: PredictionStatus
    prediction_id: str
    blocking_errors: tuple[str, ...]

    def is_ready(self) -> bool:
        """Return whether the prediction can be used for outcome scoring."""

        return self.status is PredictionStatus.READY


def validate_predicted_observable(
    observable: PredictedObservable,
) -> tuple[str, ...]:
    """Return validation errors for one predicted observable."""

    errors: list[str] = []
    if not observable.observable_name.strip():
        errors.append("predicted observable_name must not be empty")
    if not observable.rationale.strip():
        errors.append(
            f"predicted observable {observable.observable_name!r} must have rationale"
        )
    if observable.direction is PredictionDirection.UNKNOWN:
        errors.append(
            f"predicted observable {observable.observable_name!r} must not use "
            "unknown direction"
        )
    if observable.expected_min is None and observable.expected_max is not None:
        errors.append(
            f"predicted observable {observable.observable_name!r} has max without min"
        )
    if observable.expected_min is not None and observable.expected_max is None:
        errors.append(
            f"predicted observable {observable.observable_name!r} has min without max"
        )
    if (
        observable.expected_min is not None
        and observable.expected_max is not None
        and observable.expected_min > observable.expected_max
    ):
        errors.append(
            f"predicted observable {observable.observable_name!r} has min above max"
        )
    if observable.tolerance is not None and observable.tolerance < 0.0:
        errors.append(
            f"predicted observable {observable.observable_name!r} has negative "
            "tolerance"
        )

    return tuple(errors)


def validate_transfer_prediction(
    prediction: TransferPrediction,
    mapping: TransferMapping,
) -> tuple[str, ...]:
    """Return validation errors for a pre-outcome transfer prediction."""

    errors: list[str] = []
    if not prediction.prediction_id.strip():
        errors.append("prediction_id must not be empty")
    if prediction.function_id != mapping.function_id:
        errors.append("prediction function_id must match mapping function_id")
    if prediction.target_domain_id != mapping.target_domain_id:
        errors.append("prediction target_domain_id must match mapping target_domain_id")
    if not prediction.source_evidence_id.strip():
        errors.append("source_evidence_id must not be empty")
    if not prediction.target_intervention_id.strip():
        errors.append("target_intervention_id must not be empty")
    if prediction.mapping_quality is not mapping.quality:
        errors.append("prediction mapping_quality must match mapping quality")
    if not prediction.created_before_outcome:
        errors.append("prediction must be created before outcome observation")
    if prediction.forbidden_outcome_ids:
        errors.append("prediction must not reference outcome identifiers")
    if not mapping.is_usable_for_prediction():
        errors.append("mapping is not usable for prediction")
    if not prediction.predicted_observables:
        errors.append("at least one predicted observable is required")
    if not prediction.assumptions:
        errors.append("assumptions must not be empty")
    elif any(not assumption.strip() for assumption in prediction.assumptions):
        errors.append("assumptions must not contain empty assumptions")
    if not prediction.uncertainty_notes:
        errors.append("uncertainty_notes must not be empty")
    elif any(not note.strip() for note in prediction.uncertainty_notes):
        errors.append("uncertainty_notes must not contain empty notes")

    predicted_names = [
        observable.normalized_observable_name()
        for observable in prediction.predicted_observables
    ]
    if len(set(predicted_names)) != len(predicted_names):
        errors.append("predicted observables must be unique after normalization")

    mapped_target_names = {
        slot_mapping.normalized_observable_name()
        for slot_mapping in mapping.slot_mappings
    }
    for predicted in prediction.predicted_observables:
        errors.extend(validate_predicted_observable(predicted))
        if predicted.normalized_observable_name() not in mapped_target_names:
            errors.append(
                f"predicted observable {predicted.observable_name!r} is not present "
                "in the transfer mapping"
            )

    return tuple(errors)


def assess_prediction_readiness(
    prediction: TransferPrediction,
    mapping: TransferMapping,
) -> PredictionReadiness:
    """Assess whether a prediction is ready for outcome scoring."""

    errors = validate_transfer_prediction(prediction, mapping)
    status = PredictionStatus.READY if not errors else PredictionStatus.BLOCKED
    return PredictionReadiness(
        status=status,
        prediction_id=prediction.prediction_id,
        blocking_errors=errors,
    )


def direction_matches_delta(
    direction: PredictionDirection,
    *,
    baseline: float,
    observed: float,
    tolerance: float,
) -> bool:
    """Return whether an observed numeric delta matches the expected direction."""

    delta = observed - baseline
    if direction is PredictionDirection.INCREASE:
        return delta > tolerance
    if direction is PredictionDirection.DECREASE:
        return delta < -tolerance
    if direction is PredictionDirection.LIMITED_CHANGE:
        return abs(delta) <= tolerance
    if direction is PredictionDirection.NO_CHANGE:
        return abs(delta) <= tolerance
    if direction is PredictionDirection.UNKNOWN:
        return False
    assert_never(direction)
