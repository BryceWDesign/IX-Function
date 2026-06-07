"""Learning updates from scored transfer outcomes.

IX-Function only matters if reality feedback changes future behavior. This
module converts scored reality-delta reports into bounded confidence revisions,
planning cautions, and promotion or quarantine decisions for causal functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.causal_function import CausalFunction
from ix_function.reality_delta import RealityDeltaReport, TransferOutcomeStatus


class ConfidenceBand(StrEnum):
    """Coarse confidence band after reality feedback is applied."""

    QUARANTINED = "quarantined"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LearningDisposition(StrEnum):
    """How future planning should treat the causal function."""

    PROMOTE = "promote"
    RETAIN = "retain"
    WEAKEN = "weaken"
    QUARANTINE = "quarantine"


@dataclass(frozen=True, slots=True)
class TransferLearningUpdate:
    """Bounded learning update produced from a reality-delta report."""

    update_id: str
    function_id: str
    report_id: str
    source_confidence: float
    confidence_delta: float
    revised_confidence: float
    confidence_band: ConfidenceBand
    disposition: LearningDisposition
    future_planning_rules: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]
    blocking_errors: tuple[str, ...] = ()

    def should_update_future_behavior(self) -> bool:
        """Return whether the update contains behavior-changing guidance."""

        return bool(self.future_planning_rules) and not self.blocking_errors


def build_learning_update(
    causal_function: CausalFunction,
    report: RealityDeltaReport,
) -> TransferLearningUpdate:
    """Build a future-behavior update from scored reality feedback."""

    blocking_errors = validate_learning_inputs(causal_function, report)
    revised_confidence = clamp_confidence(
        causal_function.prior_confidence + report.confidence_delta
    )
    confidence_band = classify_confidence(revised_confidence)

    if blocking_errors:
        return TransferLearningUpdate(
            update_id=f"{report.report_id}:learning-update",
            function_id=causal_function.function_id,
            report_id=report.report_id,
            source_confidence=causal_function.prior_confidence,
            confidence_delta=min(report.confidence_delta, 0.0),
            revised_confidence=revised_confidence,
            confidence_band=confidence_band,
            disposition=LearningDisposition.WEAKEN,
            future_planning_rules=(
                "Do not use this transfer result to strengthen future plans "
                "until blocking errors are resolved.",
            ),
            uncertainty_notes=(
                "Learning update was generated from an invalid function/report pair.",
            ),
            blocking_errors=blocking_errors,
        )

    disposition = choose_learning_disposition(report.status, revised_confidence)
    return TransferLearningUpdate(
        update_id=f"{report.report_id}:learning-update",
        function_id=causal_function.function_id,
        report_id=report.report_id,
        source_confidence=causal_function.prior_confidence,
        confidence_delta=report.confidence_delta,
        revised_confidence=revised_confidence,
        confidence_band=confidence_band,
        disposition=disposition,
        future_planning_rules=future_planning_rules_for(
            status=report.status,
            disposition=disposition,
            family_value=causal_function.family.value,
        ),
        uncertainty_notes=combine_uncertainty_notes(
            causal_function.uncertainty_notes,
            report.uncertainty_notes,
        ),
        blocking_errors=(),
    )


def clamp_confidence(value: float) -> float:
    """Clamp confidence into the closed interval [0.0, 1.0]."""

    return min(1.0, max(0.0, round(value, 6)))


def classify_confidence(confidence: float) -> ConfidenceBand:
    """Classify bounded confidence into a reviewable coarse band."""

    if confidence < 0.2:
        return ConfidenceBand.QUARANTINED
    if confidence < 0.5:
        return ConfidenceBand.LOW
    if confidence < 0.75:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.HIGH


def choose_learning_disposition(
    status: TransferOutcomeStatus,
    revised_confidence: float,
) -> LearningDisposition:
    """Choose how future planning should treat the causal function."""

    if status is TransferOutcomeStatus.SUPPORTED:
        if revised_confidence >= 0.6:
            return LearningDisposition.PROMOTE
        return LearningDisposition.RETAIN
    if status is TransferOutcomeStatus.MIXED:
        if revised_confidence < 0.35:
            return LearningDisposition.WEAKEN
        return LearningDisposition.RETAIN
    if status is TransferOutcomeStatus.FAILED:
        if revised_confidence < 0.4:
            return LearningDisposition.QUARANTINE
        return LearningDisposition.WEAKEN
    if status is TransferOutcomeStatus.UNSCORABLE:
        if revised_confidence < 0.25:
            return LearningDisposition.QUARANTINE
        return LearningDisposition.WEAKEN
    raise AssertionError(f"Unhandled transfer outcome status: {status!r}")


def future_planning_rules_for(
    *,
    status: TransferOutcomeStatus,
    disposition: LearningDisposition,
    family_value: str,
) -> tuple[str, ...]:
    """Return behavior-changing planning rules for future transfer attempts."""

    if status is TransferOutcomeStatus.SUPPORTED:
        return (
            f"Permit cautious reuse of {family_value!r} causal structure when "
            "a future target domain exposes comparable roles and measurable "
            "outcomes.",
            "Require a new pre-outcome prediction for every future transfer; "
            "prior support is not automatic approval.",
        )

    if status is TransferOutcomeStatus.MIXED:
        return (
            f"Retain {family_value!r} causal structure as plausible but require "
            "additional target-domain measurements before stronger reuse.",
            "Prefer narrower predictions and larger uncertainty bands for the "
            "next related transfer trial.",
        )

    if status is TransferOutcomeStatus.FAILED:
        if disposition is LearningDisposition.QUARANTINE:
            return (
                f"Quarantine {family_value!r} transfer for this target pattern "
                "until new source or target evidence explains the failure.",
                "Block automatic reuse in future plans that share the same "
                "failed mapping pattern.",
            )
        return (
            f"Weaken {family_value!r} transfer confidence for future plans.",
            "Require an alternative causal explanation before trying the same "
            "mapping pattern again.",
        )

    if status is TransferOutcomeStatus.UNSCORABLE:
        return (
            f"Do not strengthen {family_value!r} transfer confidence from this "
            "trial because the outcome could not be scored.",
            "Require complete baseline and outcome measurements before the next "
            "learning update.",
        )

    raise AssertionError(f"Unhandled transfer outcome status: {status!r}")


def combine_uncertainty_notes(
    function_notes: tuple[str, ...],
    report_notes: tuple[str, ...],
) -> tuple[str, ...]:
    """Combine function and report uncertainty notes without losing provenance."""

    notes: list[str] = []
    notes.extend(f"function: {note}" for note in function_notes)
    notes.extend(f"reality_delta: {note}" for note in report_notes)
    if not notes:
        notes.append("No uncertainty notes were provided; confidence must not rise.")
    return tuple(notes)


def validate_learning_inputs(
    causal_function: CausalFunction,
    report: RealityDeltaReport,
) -> tuple[str, ...]:
    """Return blocking errors for a function/report learning update."""

    errors: list[str] = []
    if causal_function.function_id != report.function_id:
        errors.append("causal_function function_id must match report function_id")
    if report.blocking_errors:
        errors.append("reality_delta report contains blocking errors")
    if not 0.0 <= causal_function.prior_confidence <= 1.0:
        errors.append("causal_function prior_confidence must be between 0.0 and 1.0")
    if not report.report_id.strip():
        errors.append("report_id must not be empty")
    if not report.prediction_id.strip():
        errors.append("report prediction_id must not be empty")
    if not report.function_id.strip():
        errors.append("report function_id must not be empty")
    if not report.uncertainty_notes:
        errors.append("report uncertainty_notes must not be empty")
    return tuple(errors)
