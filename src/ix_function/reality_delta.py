"""Reality-delta scoring for IX-Function transfer predictions.

Reality-delta records compare committed pre-outcome predictions against observed
target outcomes. This is where IX-Function begins separating useful transfer
from attractive but failed narrative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.observation import (
    DomainSnapshot,
    MeasuredValue,
    ObservationValue,
    OutcomeRecord,
)
from ix_function.prediction import (
    PredictedObservable,
    PredictionDirection,
    TransferPrediction,
    direction_matches_delta,
)


class OutcomeMatch(StrEnum):
    """Per-observable prediction outcome state."""

    MATCHED = "matched"
    PARTIAL = "partial"
    MISSED = "missed"
    UNSCORABLE = "unscorable"


class TransferOutcomeStatus(StrEnum):
    """Overall status for a scored transfer prediction."""

    SUPPORTED = "supported"
    MIXED = "mixed"
    FAILED = "failed"
    UNSCORABLE = "unscorable"


@dataclass(frozen=True, slots=True)
class ObservableDelta:
    """Reality-delta score for one predicted target observable."""

    observable_name: str
    predicted_direction: PredictionDirection
    baseline_value: ObservationValue | None
    observed_value: ObservationValue | None
    numeric_delta: float | None
    direction_matched: bool
    range_matched: bool | None
    outcome_match: OutcomeMatch
    score: float
    notes: tuple[str, ...]

    def normalized_observable_name(self) -> str:
        """Return the stable normalized observable name."""

        return "_".join(self.observable_name.strip().lower().split())


@dataclass(frozen=True, slots=True)
class RealityDeltaReport:
    """Scored result of comparing one prediction against target reality."""

    report_id: str
    prediction_id: str
    target_domain_id: str
    target_intervention_id: str
    outcome_id: str
    observable_deltas: tuple[ObservableDelta, ...]
    status: TransferOutcomeStatus
    mean_score: float
    confidence_delta: float
    uncertainty_notes: tuple[str, ...]
    blocking_errors: tuple[str, ...] = ()

    def is_supported(self) -> bool:
        """Return whether the transfer outcome supports the prediction."""

        return self.status is TransferOutcomeStatus.SUPPORTED


def build_reality_delta_report(
    prediction: TransferPrediction,
    baseline: DomainSnapshot,
    outcome: OutcomeRecord,
) -> RealityDeltaReport:
    """Compare a pre-outcome prediction against baseline and outcome records."""

    blocking_errors = validate_reality_delta_inputs(prediction, baseline, outcome)
    if blocking_errors:
        return RealityDeltaReport(
            report_id=f"{prediction.prediction_id}:reality-delta",
            prediction_id=prediction.prediction_id,
            target_domain_id=prediction.target_domain_id,
            target_intervention_id=prediction.target_intervention_id,
            outcome_id=outcome.outcome_id,
            observable_deltas=(),
            status=TransferOutcomeStatus.UNSCORABLE,
            mean_score=0.0,
            confidence_delta=-0.12,
            uncertainty_notes=(
                "Reality-delta scoring was blocked by invalid input records.",
            ),
            blocking_errors=blocking_errors,
        )

    baseline_index = baseline.value_index()
    outcome_index = outcome_value_index(outcome)
    observable_deltas = tuple(
        score_predicted_observable(
            predicted=predicted,
            baseline_value=(
                baseline_index[predicted.normalized_observable_name()].value
                if predicted.normalized_observable_name() in baseline_index
                else None
            ),
            observed_value=(
                outcome_index[predicted.normalized_observable_name()].value
                if predicted.normalized_observable_name() in outcome_index
                else None
            ),
        )
        for predicted in prediction.predicted_observables
    )

    status = classify_transfer_outcome(observable_deltas)
    scored_deltas = tuple(
        delta
        for delta in observable_deltas
        if delta.outcome_match is not OutcomeMatch.UNSCORABLE
    )
    mean_score = mean_observable_score(scored_deltas)
    uncertainty_notes = build_reality_uncertainty_notes(observable_deltas, status)
    confidence_delta = confidence_delta_for_status(status, mean_score)

    return RealityDeltaReport(
        report_id=f"{prediction.prediction_id}:reality-delta",
        prediction_id=prediction.prediction_id,
        target_domain_id=prediction.target_domain_id,
        target_intervention_id=prediction.target_intervention_id,
        outcome_id=outcome.outcome_id,
        observable_deltas=observable_deltas,
        status=status,
        mean_score=mean_score,
        confidence_delta=confidence_delta,
        uncertainty_notes=uncertainty_notes,
        blocking_errors=(),
    )


def score_predicted_observable(
    *,
    predicted: PredictedObservable,
    baseline_value: ObservationValue | None,
    observed_value: ObservationValue | None,
) -> ObservableDelta:
    """Score one predicted observable against baseline and observed value."""

    notes: list[str] = []
    baseline_numeric = numeric_value(baseline_value)
    observed_numeric = numeric_value(observed_value)

    if baseline_value is None:
        notes.append("baseline value is missing")
    if observed_value is None:
        notes.append("observed value is missing")

    if baseline_numeric is None or observed_numeric is None:
        return ObservableDelta(
            observable_name=predicted.observable_name,
            predicted_direction=predicted.direction,
            baseline_value=baseline_value,
            observed_value=observed_value,
            numeric_delta=None,
            direction_matched=False,
            range_matched=None,
            outcome_match=OutcomeMatch.UNSCORABLE,
            score=0.0,
            notes=tuple(notes or ("numeric baseline and observed values are required",)),
        )

    tolerance = predicted.tolerance if predicted.tolerance is not None else 0.0
    direction_matched = direction_matches_delta(
        predicted.direction,
        baseline=baseline_numeric,
        observed=observed_numeric,
        tolerance=tolerance,
    )
    range_matched = range_matches_prediction(
        predicted,
        observed=observed_numeric,
    )
    numeric_delta = round(observed_numeric - baseline_numeric, 6)

    if direction_matched and range_matched is not False:
        outcome_match = OutcomeMatch.MATCHED
        score = 1.0
    elif direction_matched or range_matched is True:
        outcome_match = OutcomeMatch.PARTIAL
        score = 0.5
        notes.append("prediction partially matched observed outcome")
    else:
        outcome_match = OutcomeMatch.MISSED
        score = 0.0
        notes.append("prediction missed observed outcome")

    if range_matched is None:
        notes.append("prediction did not include a numeric expected range")

    return ObservableDelta(
        observable_name=predicted.observable_name,
        predicted_direction=predicted.direction,
        baseline_value=baseline_value,
        observed_value=observed_value,
        numeric_delta=numeric_delta,
        direction_matched=direction_matched,
        range_matched=range_matched,
        outcome_match=outcome_match,
        score=score,
        notes=tuple(notes),
    )


def classify_transfer_outcome(
    observable_deltas: tuple[ObservableDelta, ...],
) -> TransferOutcomeStatus:
    """Classify the overall transfer outcome from per-observable deltas."""

    if not observable_deltas:
        return TransferOutcomeStatus.UNSCORABLE
    if all(delta.outcome_match is OutcomeMatch.UNSCORABLE for delta in observable_deltas):
        return TransferOutcomeStatus.UNSCORABLE
    if all(delta.outcome_match is OutcomeMatch.MATCHED for delta in observable_deltas):
        return TransferOutcomeStatus.SUPPORTED
    if any(
        delta.outcome_match in {OutcomeMatch.MATCHED, OutcomeMatch.PARTIAL}
        for delta in observable_deltas
    ):
        return TransferOutcomeStatus.MIXED
    return TransferOutcomeStatus.FAILED


def mean_observable_score(observable_deltas: tuple[ObservableDelta, ...]) -> float:
    """Return mean score while safely handling all-unscorable outcomes."""

    if not observable_deltas:
        return 0.0
    return round(
        sum(delta.score for delta in observable_deltas) / len(observable_deltas),
        6,
    )


def confidence_delta_for_status(
    status: TransferOutcomeStatus,
    mean_score: float,
) -> float:
    """Return bounded confidence delta from the scored reality outcome."""

    if status is TransferOutcomeStatus.SUPPORTED:
        return round(0.12 * mean_score, 6)
    if status is TransferOutcomeStatus.MIXED:
        return round(0.04 * mean_score, 6)
    if status is TransferOutcomeStatus.FAILED:
        return -0.12
    if status is TransferOutcomeStatus.UNSCORABLE:
        return -0.08
    raise AssertionError(f"Unhandled transfer outcome status: {status!r}")


def range_matches_prediction(
    predicted: PredictedObservable,
    *,
    observed: float,
) -> bool | None:
    """Return whether an observed numeric value falls inside prediction range."""

    if predicted.expected_min is None or predicted.expected_max is None:
        return None

    tolerance = predicted.tolerance if predicted.tolerance is not None else 0.0
    return (
        predicted.expected_min - tolerance
        <= observed
        <= predicted.expected_max + tolerance
    )


def numeric_value(value: ObservationValue | None) -> float | None:
    """Return numeric value as float while rejecting bool as a numeric shortcut."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def outcome_value_index(outcome: OutcomeRecord) -> dict[str, MeasuredValue]:
    """Return outcome measured values keyed by normalized observable name."""

    return {value.normalized_observable_name(): value for value in outcome.values}


def build_reality_uncertainty_notes(
    observable_deltas: tuple[ObservableDelta, ...],
    status: TransferOutcomeStatus,
) -> tuple[str, ...]:
    """Build conservative uncertainty notes from scored deltas."""

    notes: list[str] = []
    if status is TransferOutcomeStatus.SUPPORTED:
        notes.append(
            "Observed outcome supported the committed prediction, but this is "
            "bounded transfer evidence rather than AGI proof."
        )
    elif status is TransferOutcomeStatus.MIXED:
        notes.append(
            "Observed outcome partially supported the prediction; confidence "
            "must remain limited."
        )
    elif status is TransferOutcomeStatus.FAILED:
        notes.append(
            "Observed outcome failed the prediction; transfer confidence must "
            "be reduced."
        )
    else:
        notes.append(
            "Observed outcome could not be scored; transfer confidence must not "
            "increase."
        )

    for delta in observable_deltas:
        for note in delta.notes:
            notes.append(f"{delta.observable_name}: {note}")

    return tuple(notes)


def validate_reality_delta_inputs(
    prediction: TransferPrediction,
    baseline: DomainSnapshot,
    outcome: OutcomeRecord,
) -> tuple[str, ...]:
    """Return blocking errors before scoring a reality-delta report."""

    errors: list[str] = []
    if prediction.target_domain_id != baseline.domain_id:
        errors.append("prediction target_domain_id must match baseline domain_id")
    if prediction.target_domain_id != outcome.domain_id:
        errors.append("prediction target_domain_id must match outcome domain_id")
    if outcome.observed_after_intervention_id != prediction.target_intervention_id:
        errors.append("outcome must reference prediction target_intervention_id")

    baseline_names = set(baseline.value_index())
    outcome_names = set(outcome_value_index(outcome))
    for predicted in prediction.predicted_observables:
        predicted_name = predicted.normalized_observable_name()
        if predicted_name not in baseline_names:
            errors.append(
                f"baseline is missing predicted observable {predicted.observable_name!r}"
            )
        if predicted_name not in outcome_names:
            errors.append(
                f"outcome is missing predicted observable {predicted.observable_name!r}"
            )

    return tuple(errors)
