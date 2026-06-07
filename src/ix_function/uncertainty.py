"""Uncertainty preservation for IX-Function.

IX-Function must not turn ambiguous mappings, weak observations, unsupported
assumptions, or failed outcomes into false certainty. This module creates a
portable uncertainty ledger that later evidence, falsification, and handoff
layers can carry forward without laundering uncertainty away.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.learning import TransferLearningUpdate
from ix_function.mapping import MappingQuality, TransferMapping
from ix_function.prediction import TransferPrediction
from ix_function.reality_delta import RealityDeltaReport, TransferOutcomeStatus


class UncertaintyKind(StrEnum):
    """Category of uncertainty preserved in the transfer evidence chain."""

    AMBIGUITY = "ambiguity"
    ASSUMPTION = "assumption"
    BLOCKING_ERROR = "blocking_error"
    MAPPING = "mapping"
    MEASUREMENT = "measurement"
    OUTCOME = "outcome"
    TRANSFER = "transfer"


class UncertaintySeverity(StrEnum):
    """Severity of uncertainty for review and claim gating."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKING = "blocking"


class UncertaintyState(StrEnum):
    """Lifecycle state for one uncertainty item."""

    OPEN = "open"
    REDUCED = "reduced"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class EvidenceClaimStrength(StrEnum):
    """Claim strength requested from an uncertainty ledger."""

    INTERNAL_NOTE = "internal_note"
    BOUNDED_CANDIDATE_EVIDENCE = "bounded_candidate_evidence"
    STRONG_TRANSFER_SUPPORT = "strong_transfer_support"


@dataclass(frozen=True, slots=True)
class UncertaintyItem:
    """One preserved uncertainty item with provenance."""

    uncertainty_id: str
    kind: UncertaintyKind
    severity: UncertaintySeverity
    state: UncertaintyState
    source_id: str
    statement: str
    mitigation: str

    def is_blocking(self) -> bool:
        """Return whether this item blocks stronger transfer claims."""

        return (
            self.severity is UncertaintySeverity.BLOCKING
            or self.state is UncertaintyState.ESCALATED
        )


@dataclass(frozen=True, slots=True)
class UncertaintyLedger:
    """Reviewable set of uncertainty items for a transfer evidence chain."""

    ledger_id: str
    items: tuple[UncertaintyItem, ...]

    def blocking_items(self) -> tuple[UncertaintyItem, ...]:
        """Return items that block stronger transfer claims."""

        return tuple(item for item in self.items if item.is_blocking())

    def open_items(self) -> tuple[UncertaintyItem, ...]:
        """Return unresolved uncertainty items."""

        return tuple(
            item
            for item in self.items
            if item.state in {UncertaintyState.OPEN, UncertaintyState.ESCALATED}
        )

    def high_open_items(self) -> tuple[UncertaintyItem, ...]:
        """Return open high-severity uncertainty items."""

        return tuple(
            item
            for item in self.open_items()
            if item.severity in {
                UncertaintySeverity.HIGH,
                UncertaintySeverity.BLOCKING,
            }
        )

    def maximum_severity(self) -> UncertaintySeverity:
        """Return the highest severity present in the ledger."""

        severity_order = {
            UncertaintySeverity.LOW: 0,
            UncertaintySeverity.MEDIUM: 1,
            UncertaintySeverity.HIGH: 2,
            UncertaintySeverity.BLOCKING: 3,
        }
        if not self.items:
            return UncertaintySeverity.LOW
        return max(self.items, key=lambda item: severity_order[item.severity]).severity

    def permits_candidate_evidence(self) -> bool:
        """Return whether uncertainty still allows bounded candidate evidence."""

        return not self.blocking_items()


@dataclass(frozen=True, slots=True)
class UncertaintyGateResult:
    """Result of evaluating whether uncertainty blocks a transfer claim."""

    ledger_id: str
    allowed: bool
    maximum_severity: UncertaintySeverity
    blocking_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class ClaimStrengthGateResult:
    """Result of evaluating a requested evidence claim strength."""

    ledger_id: str
    requested_strength: EvidenceClaimStrength
    allowed: bool
    blocking_ids: tuple[str, ...]
    required_actions: tuple[str, ...]
    reason: str


def build_uncertainty_ledger(
    *,
    ledger_id: str,
    mapping: TransferMapping,
    prediction: TransferPrediction,
    report: RealityDeltaReport,
    learning_update: TransferLearningUpdate,
) -> UncertaintyLedger:
    """Build a ledger from mapping, prediction, outcome, and learning artifacts."""

    items: list[UncertaintyItem] = []
    items.extend(items_from_mapping(mapping))
    items.extend(items_from_prediction(prediction))
    items.extend(items_from_reality_delta(report))
    items.extend(items_from_learning_update(learning_update))

    return UncertaintyLedger(
        ledger_id=ledger_id,
        items=tuple(items),
    )


def evaluate_uncertainty_gate(ledger: UncertaintyLedger) -> UncertaintyGateResult:
    """Evaluate whether preserved uncertainty blocks stronger claims."""

    blocking_items = ledger.blocking_items()
    blocking_ids = tuple(item.uncertainty_id for item in blocking_items)
    if blocking_items:
        return UncertaintyGateResult(
            ledger_id=ledger.ledger_id,
            allowed=False,
            maximum_severity=ledger.maximum_severity(),
            blocking_ids=blocking_ids,
            reason=(
                "Uncertainty gate blocked stronger transfer claims because one "
                "or more uncertainty items are blocking or escalated."
            ),
        )

    return UncertaintyGateResult(
        ledger_id=ledger.ledger_id,
        allowed=True,
        maximum_severity=ledger.maximum_severity(),
        blocking_ids=(),
        reason=(
            "Uncertainty gate allows bounded candidate evidence while preserving "
            "all open uncertainty items for review."
        ),
    )


def evaluate_claim_strength_gate(
    ledger: UncertaintyLedger,
    requested_strength: EvidenceClaimStrength,
) -> ClaimStrengthGateResult:
    """Gate a requested evidence claim strength against preserved uncertainty."""

    validation_errors = validate_uncertainty_ledger(ledger)
    if validation_errors:
        return ClaimStrengthGateResult(
            ledger_id=ledger.ledger_id,
            requested_strength=requested_strength,
            allowed=False,
            blocking_ids=(),
            required_actions=validation_errors,
            reason="Claim-strength gate blocked because the ledger is invalid.",
        )

    if requested_strength is EvidenceClaimStrength.INTERNAL_NOTE:
        return ClaimStrengthGateResult(
            ledger_id=ledger.ledger_id,
            requested_strength=requested_strength,
            allowed=True,
            blocking_ids=(),
            required_actions=(
                "Preserve all uncertainty items with the internal note.",
            ),
            reason=(
                "Internal uncertainty documentation is allowed, but it does not "
                "promote the transfer claim."
            ),
        )

    blocking_items = ledger.blocking_items()
    if blocking_items:
        return ClaimStrengthGateResult(
            ledger_id=ledger.ledger_id,
            requested_strength=requested_strength,
            allowed=False,
            blocking_ids=tuple(item.uncertainty_id for item in blocking_items),
            required_actions=tuple(item.mitigation for item in blocking_items),
            reason=(
                "Requested claim strength is blocked by escalated or blocking "
                "uncertainty items."
            ),
        )

    high_open_items = ledger.high_open_items()
    if (
        requested_strength is EvidenceClaimStrength.STRONG_TRANSFER_SUPPORT
        and high_open_items
    ):
        return ClaimStrengthGateResult(
            ledger_id=ledger.ledger_id,
            requested_strength=requested_strength,
            allowed=False,
            blocking_ids=tuple(item.uncertainty_id for item in high_open_items),
            required_actions=tuple(item.mitigation for item in high_open_items),
            reason=(
                "Strong transfer support is blocked until high-severity open "
                "uncertainty is reduced or resolved."
            ),
        )

    return ClaimStrengthGateResult(
        ledger_id=ledger.ledger_id,
        requested_strength=requested_strength,
        allowed=True,
        blocking_ids=(),
        required_actions=(
            "Carry remaining uncertainty into the evidence packet.",
            "Do not represent this result as AGI proof.",
        ),
        reason=(
            "Requested claim strength is allowed within bounded IX-Function "
            "evidence language."
        ),
    )


def items_from_mapping(mapping: TransferMapping) -> tuple[UncertaintyItem, ...]:
    """Convert mapping warnings and ambiguity into uncertainty items."""

    items: list[UncertaintyItem] = []
    if mapping.quality is MappingQuality.AMBIGUOUS:
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{mapping.function_id}:mapping:ambiguity",
                kind=UncertaintyKind.AMBIGUITY,
                severity=UncertaintySeverity.MEDIUM,
                state=UncertaintyState.OPEN,
                source_id=mapping.target_domain_id,
                statement=(
                    "Transfer mapping is usable but ambiguous; alternative slot "
                    "bindings remain plausible."
                ),
                mitigation=(
                    "Require pre-outcome prediction and reality-delta scoring "
                    "before increasing transfer confidence."
                ),
            )
        )
    if mapping.quality is MappingQuality.INSUFFICIENT:
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{mapping.function_id}:mapping:insufficient",
                kind=UncertaintyKind.MAPPING,
                severity=UncertaintySeverity.BLOCKING,
                state=UncertaintyState.ESCALATED,
                source_id=mapping.target_domain_id,
                statement="Transfer mapping is insufficient for prediction.",
                mitigation="Add measurable target observables or block the trial.",
            )
        )

    for index, warning in enumerate(mapping.warnings, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{mapping.function_id}:mapping:warning:{index}",
                kind=UncertaintyKind.MAPPING,
                severity=UncertaintySeverity.HIGH,
                state=UncertaintyState.OPEN,
                source_id=mapping.target_domain_id,
                statement=warning,
                mitigation="Resolve mapping warning before making stronger claims.",
            )
        )

    for slot_mapping in mapping.slot_mappings:
        for index, note in enumerate(slot_mapping.uncertainty_notes, start=1):
            items.append(
                UncertaintyItem(
                    uncertainty_id=(
                        f"{mapping.function_id}:mapping:"
                        f"{slot_mapping.normalized_slot_id()}:note:{index}"
                    ),
                    kind=UncertaintyKind.MAPPING,
                    severity=UncertaintySeverity.MEDIUM,
                    state=UncertaintyState.OPEN,
                    source_id=mapping.target_domain_id,
                    statement=note,
                    mitigation=(
                        "Retain note in evidence and require outcome scoring "
                        "before confidence increases."
                    ),
                )
            )

    return tuple(items)


def items_from_prediction(
    prediction: TransferPrediction,
) -> tuple[UncertaintyItem, ...]:
    """Convert prediction assumptions and uncertainty notes into ledger items."""

    items: list[UncertaintyItem] = []
    for index, assumption in enumerate(prediction.assumptions, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{prediction.prediction_id}:assumption:{index}",
                kind=UncertaintyKind.ASSUMPTION,
                severity=UncertaintySeverity.MEDIUM,
                state=UncertaintyState.OPEN,
                source_id=prediction.prediction_id,
                statement=assumption,
                mitigation=(
                    "Verify or weaken this assumption during outcome review."
                ),
            )
        )

    for index, note in enumerate(prediction.uncertainty_notes, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{prediction.prediction_id}:prediction-note:{index}",
                kind=UncertaintyKind.TRANSFER,
                severity=UncertaintySeverity.MEDIUM,
                state=UncertaintyState.OPEN,
                source_id=prediction.prediction_id,
                statement=note,
                mitigation="Carry this uncertainty into reality-delta scoring.",
            )
        )

    return tuple(items)


def items_from_reality_delta(
    report: RealityDeltaReport,
) -> tuple[UncertaintyItem, ...]:
    """Convert reality-delta notes and failures into uncertainty items."""

    items: list[UncertaintyItem] = []
    if report.status is TransferOutcomeStatus.FAILED:
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{report.report_id}:failed-transfer",
                kind=UncertaintyKind.OUTCOME,
                severity=UncertaintySeverity.BLOCKING,
                state=UncertaintyState.ESCALATED,
                source_id=report.report_id,
                statement="Reality-delta scoring failed the transfer prediction.",
                mitigation=(
                    "Reduce confidence, block promotion, and require an "
                    "alternative causal explanation."
                ),
            )
        )
    elif report.status is TransferOutcomeStatus.UNSCORABLE:
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{report.report_id}:unscorable-transfer",
                kind=UncertaintyKind.MEASUREMENT,
                severity=UncertaintySeverity.BLOCKING,
                state=UncertaintyState.ESCALATED,
                source_id=report.report_id,
                statement="Reality-delta scoring could not score the transfer.",
                mitigation=(
                    "Collect complete numeric baseline and outcome measurements "
                    "before any confidence increase."
                ),
            )
        )
    elif report.status is TransferOutcomeStatus.MIXED:
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{report.report_id}:mixed-transfer",
                kind=UncertaintyKind.OUTCOME,
                severity=UncertaintySeverity.HIGH,
                state=UncertaintyState.OPEN,
                source_id=report.report_id,
                statement="Reality-delta scoring produced a mixed transfer result.",
                mitigation="Require narrower retesting before promotion.",
            )
        )

    for index, error in enumerate(report.blocking_errors, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{report.report_id}:blocking-error:{index}",
                kind=UncertaintyKind.BLOCKING_ERROR,
                severity=UncertaintySeverity.BLOCKING,
                state=UncertaintyState.ESCALATED,
                source_id=report.report_id,
                statement=error,
                mitigation="Resolve the blocking error and rerun scoring.",
            )
        )

    for index, note in enumerate(report.uncertainty_notes, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{report.report_id}:outcome-note:{index}",
                kind=UncertaintyKind.OUTCOME,
                severity=UncertaintySeverity.LOW,
                state=UncertaintyState.OPEN,
                source_id=report.report_id,
                statement=note,
                mitigation="Preserve this note in the final evidence packet.",
            )
        )

    return tuple(items)


def items_from_learning_update(
    update: TransferLearningUpdate,
) -> tuple[UncertaintyItem, ...]:
    """Convert learning-update uncertainty and blockers into ledger items."""

    items: list[UncertaintyItem] = []
    for index, error in enumerate(update.blocking_errors, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{update.update_id}:blocking-error:{index}",
                kind=UncertaintyKind.BLOCKING_ERROR,
                severity=UncertaintySeverity.BLOCKING,
                state=UncertaintyState.ESCALATED,
                source_id=update.update_id,
                statement=error,
                mitigation="Resolve learning-update blocker before reuse.",
            )
        )

    for index, note in enumerate(update.uncertainty_notes, start=1):
        items.append(
            UncertaintyItem(
                uncertainty_id=f"{update.update_id}:learning-note:{index}",
                kind=UncertaintyKind.TRANSFER,
                severity=UncertaintySeverity.LOW,
                state=UncertaintyState.OPEN,
                source_id=update.update_id,
                statement=note,
                mitigation="Carry this learning uncertainty into future planning.",
            )
        )

    return tuple(items)


def validate_uncertainty_ledger(ledger: UncertaintyLedger) -> tuple[str, ...]:
    """Return validation errors for an uncertainty ledger."""

    errors: list[str] = []
    if not ledger.ledger_id.strip():
        errors.append("ledger_id must not be empty")

    item_ids = [item.uncertainty_id for item in ledger.items]
    if len(set(item_ids)) != len(item_ids):
        errors.append("uncertainty item identifiers must be unique")

    for item in ledger.items:
        if not item.uncertainty_id.strip():
            errors.append("uncertainty_id must not be empty")
        if not item.source_id.strip():
            errors.append(f"source_id must not be empty for {item.uncertainty_id!r}")
        if not item.statement.strip():
            errors.append(
                f"statement must not be empty for {item.uncertainty_id!r}"
            )
        if not item.mitigation.strip():
            errors.append(
                f"mitigation must not be empty for {item.uncertainty_id!r}"
            )

    return tuple(errors)
