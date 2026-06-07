"""Falsification ledger for IX-Function.

A serious AGI-candidate donor cannot only collect positive evidence. It must
also preserve kill criteria that can block, downgrade, or quarantine transfer
claims when the evidence fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.learning import TransferLearningUpdate
from ix_function.mapping import TransferMapping
from ix_function.reality_delta import RealityDeltaReport, TransferOutcomeStatus
from ix_function.uncertainty import UncertaintyLedger


class FalsificationKind(StrEnum):
    """Kind of falsification pressure applied to a transfer chain."""

    DOMAIN_TRANSFER = "domain_transfer"
    MAPPING_VALIDITY = "mapping_validity"
    OUTCOME_SUPPORT = "outcome_support"
    REPRODUCIBILITY = "reproducibility"
    UNCERTAINTY = "uncertainty"
    FUTURE_BEHAVIOR = "future_behavior"


class FalsificationSeverity(StrEnum):
    """Severity of a failed criterion."""

    WARNING = "warning"
    DOWNGRADE = "downgrade"
    KILL = "kill"


class CriterionStatus(StrEnum):
    """Result status for one falsification criterion."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class FalsificationVerdict(StrEnum):
    """Final verdict for a transfer evidence chain."""

    ALLOW_BOUNDED_EVIDENCE = "allow_bounded_evidence"
    REQUIRE_RETEST = "require_retest"
    DOWNGRADE_CLAIM = "downgrade_claim"
    KILL_CLAIM = "kill_claim"


@dataclass(frozen=True, slots=True)
class FalsificationCriterion:
    """A criterion that can block or weaken transfer evidence."""

    criterion_id: str
    kind: FalsificationKind
    severity: FalsificationSeverity
    statement: str
    required_evidence: str


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    """Evaluation result for one falsification criterion."""

    criterion_id: str
    status: CriterionStatus
    severity: FalsificationSeverity
    reason: str
    evidence_refs: tuple[str, ...]

    def blocks_claim(self) -> bool:
        """Return whether this evaluation blocks the stronger claim."""

        return (
            self.status is CriterionStatus.FAILED
            and self.severity is FalsificationSeverity.KILL
        )

    def downgrades_claim(self) -> bool:
        """Return whether this evaluation downgrades the claim."""

        return self.status is CriterionStatus.FAILED and self.severity in {
            FalsificationSeverity.DOWNGRADE,
            FalsificationSeverity.KILL,
        }


@dataclass(frozen=True, slots=True)
class FalsificationLedger:
    """Reviewable falsification result for a transfer evidence chain."""

    ledger_id: str
    function_id: str
    evaluations: tuple[CriterionEvaluation, ...]
    verdict: FalsificationVerdict
    required_actions: tuple[str, ...]

    def failed_evaluations(self) -> tuple[CriterionEvaluation, ...]:
        """Return failed criterion evaluations."""

        return tuple(
            evaluation
            for evaluation in self.evaluations
            if evaluation.status is CriterionStatus.FAILED
        )

    def kill_evaluations(self) -> tuple[CriterionEvaluation, ...]:
        """Return failed kill-severity evaluations."""

        return tuple(
            evaluation for evaluation in self.evaluations if evaluation.blocks_claim()
        )


def default_wave6_kill_criteria() -> tuple[FalsificationCriterion, ...]:
    """Return the default IX-Function falsification criteria for Wave 6."""

    return (
        FalsificationCriterion(
            criterion_id="IXF-FALSIFY-001",
            kind=FalsificationKind.DOMAIN_TRANSFER,
            severity=FalsificationSeverity.KILL,
            statement="The trial must be cross-domain and not same-family theater.",
            required_evidence=(
                "Source and target domains must have distinct declared domain kinds."
            ),
        ),
        FalsificationCriterion(
            criterion_id="IXF-FALSIFY-002",
            kind=FalsificationKind.MAPPING_VALIDITY,
            severity=FalsificationSeverity.KILL,
            statement="The target mapping must be usable before prediction.",
            required_evidence=(
                "Transfer mapping quality must be complete or ambiguous with "
                "full slot coverage."
            ),
        ),
        FalsificationCriterion(
            criterion_id="IXF-FALSIFY-003",
            kind=FalsificationKind.OUTCOME_SUPPORT,
            severity=FalsificationSeverity.DOWNGRADE,
            statement="The committed prediction must receive outcome support.",
            required_evidence=(
                "Reality-delta scoring must be supported or mixed; failed and "
                "unscorable outcomes cannot promote transfer."
            ),
        ),
        FalsificationCriterion(
            criterion_id="IXF-FALSIFY-004",
            kind=FalsificationKind.UNCERTAINTY,
            severity=FalsificationSeverity.KILL,
            statement="Blocking uncertainty must block stronger claims.",
            required_evidence=(
                "Uncertainty ledger must contain no blocking or escalated items "
                "for bounded candidate evidence."
            ),
        ),
        FalsificationCriterion(
            criterion_id="IXF-FALSIFY-005",
            kind=FalsificationKind.FUTURE_BEHAVIOR,
            severity=FalsificationSeverity.DOWNGRADE,
            statement="Reality feedback must change future planning behavior.",
            required_evidence=(
                "Learning update must include future planning rules and no "
                "blocking errors."
            ),
        ),
        FalsificationCriterion(
            criterion_id="IXF-FALSIFY-006",
            kind=FalsificationKind.REPRODUCIBILITY,
            severity=FalsificationSeverity.WARNING,
            statement="The trial must identify evidence refs for replay.",
            required_evidence=(
                "Mapping, prediction, reality-delta, learning, and uncertainty "
                "records must expose stable identifiers."
            ),
        ),
    )


def build_falsification_ledger(
    *,
    ledger_id: str,
    function_id: str,
    is_cross_domain: bool,
    mapping: TransferMapping,
    report: RealityDeltaReport,
    uncertainty_ledger: UncertaintyLedger,
    learning_update: TransferLearningUpdate,
    criteria: tuple[FalsificationCriterion, ...] | None = None,
) -> FalsificationLedger:
    """Evaluate default or supplied falsification criteria."""

    active_criteria = criteria or default_wave6_kill_criteria()
    evaluations = tuple(
        evaluate_criterion(
            criterion=criterion,
            function_id=function_id,
            is_cross_domain=is_cross_domain,
            mapping=mapping,
            report=report,
            uncertainty_ledger=uncertainty_ledger,
            learning_update=learning_update,
        )
        for criterion in active_criteria
    )
    verdict = choose_falsification_verdict(evaluations)
    return FalsificationLedger(
        ledger_id=ledger_id,
        function_id=function_id,
        evaluations=evaluations,
        verdict=verdict,
        required_actions=required_actions_for_verdict(verdict, evaluations),
    )


def evaluate_criterion(
    *,
    criterion: FalsificationCriterion,
    function_id: str,
    is_cross_domain: bool,
    mapping: TransferMapping,
    report: RealityDeltaReport,
    uncertainty_ledger: UncertaintyLedger,
    learning_update: TransferLearningUpdate,
) -> CriterionEvaluation:
    """Evaluate one falsification criterion against the transfer chain."""

    if criterion.kind is FalsificationKind.DOMAIN_TRANSFER:
        return _evaluate_domain_transfer(criterion, is_cross_domain)

    if criterion.kind is FalsificationKind.MAPPING_VALIDITY:
        return _evaluate_mapping_validity(criterion, mapping)

    if criterion.kind is FalsificationKind.OUTCOME_SUPPORT:
        return _evaluate_outcome_support(criterion, report)

    if criterion.kind is FalsificationKind.UNCERTAINTY:
        return _evaluate_uncertainty(criterion, uncertainty_ledger)

    if criterion.kind is FalsificationKind.FUTURE_BEHAVIOR:
        return _evaluate_future_behavior(criterion, learning_update)

    if criterion.kind is FalsificationKind.REPRODUCIBILITY:
        return _evaluate_reproducibility(
            criterion=criterion,
            function_id=function_id,
            mapping=mapping,
            report=report,
            uncertainty_ledger=uncertainty_ledger,
            learning_update=learning_update,
        )

    raise AssertionError(f"Unhandled falsification criterion kind: {criterion.kind!r}")


def choose_falsification_verdict(
    evaluations: tuple[CriterionEvaluation, ...],
) -> FalsificationVerdict:
    """Choose final falsification verdict from criterion evaluations."""

    if any(evaluation.blocks_claim() for evaluation in evaluations):
        return FalsificationVerdict.KILL_CLAIM
    if any(
        evaluation.status is CriterionStatus.FAILED
        and evaluation.severity is FalsificationSeverity.DOWNGRADE
        for evaluation in evaluations
    ):
        return FalsificationVerdict.DOWNGRADE_CLAIM
    if any(evaluation.status is CriterionStatus.WARNING for evaluation in evaluations):
        return FalsificationVerdict.REQUIRE_RETEST
    return FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE


def required_actions_for_verdict(
    verdict: FalsificationVerdict,
    evaluations: tuple[CriterionEvaluation, ...],
) -> tuple[str, ...]:
    """Return required actions for a falsification verdict."""

    failed = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.status is not CriterionStatus.PASSED
    )
    if verdict is FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE:
        return (
            "Allow bounded IX-Function evidence language.",
            "Do not represent this result as AGI proof.",
        )

    if verdict is FalsificationVerdict.REQUIRE_RETEST:
        return tuple(
            f"Retest or document warning for {evaluation.criterion_id}: "
            f"{evaluation.reason}"
            for evaluation in failed
        )

    if verdict is FalsificationVerdict.DOWNGRADE_CLAIM:
        return tuple(
            f"Downgrade claim due to {evaluation.criterion_id}: "
            f"{evaluation.reason}"
            for evaluation in failed
        )

    if verdict is FalsificationVerdict.KILL_CLAIM:
        return tuple(
            f"Block claim due to {evaluation.criterion_id}: {evaluation.reason}"
            for evaluation in failed
        )

    raise AssertionError(f"Unhandled falsification verdict: {verdict!r}")


def validate_falsification_ledger(ledger: FalsificationLedger) -> tuple[str, ...]:
    """Return validation errors for a falsification ledger."""

    errors: list[str] = []
    if not ledger.ledger_id.strip():
        errors.append("ledger_id must not be empty")
    if not ledger.function_id.strip():
        errors.append("function_id must not be empty")
    if not ledger.evaluations:
        errors.append("at least one criterion evaluation is required")
    if not ledger.required_actions:
        errors.append("required_actions must not be empty")

    criterion_ids = [evaluation.criterion_id for evaluation in ledger.evaluations]
    if len(set(criterion_ids)) != len(criterion_ids):
        errors.append("criterion evaluations must use unique criterion_id values")

    for evaluation in ledger.evaluations:
        if not evaluation.criterion_id.strip():
            errors.append("criterion_id must not be empty")
        if not evaluation.reason.strip():
            errors.append(
                f"reason must not be empty for {evaluation.criterion_id!r}"
            )
        if not evaluation.evidence_refs:
            errors.append(
                f"evidence_refs must not be empty for {evaluation.criterion_id!r}"
            )

    return tuple(errors)


def _passed(
    criterion: FalsificationCriterion,
    reason: str,
    evidence_refs: tuple[str, ...],
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        status=CriterionStatus.PASSED,
        severity=criterion.severity,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def _warning(
    criterion: FalsificationCriterion,
    reason: str,
    evidence_refs: tuple[str, ...],
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        status=CriterionStatus.WARNING,
        severity=criterion.severity,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def _failed(
    criterion: FalsificationCriterion,
    reason: str,
    evidence_refs: tuple[str, ...],
) -> CriterionEvaluation:
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        status=CriterionStatus.FAILED,
        severity=criterion.severity,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def _evaluate_domain_transfer(
    criterion: FalsificationCriterion,
    is_cross_domain: bool,
) -> CriterionEvaluation:
    if is_cross_domain:
        return _passed(
            criterion,
            "Source and target domains were declared cross-domain.",
            ("domain-pair",),
        )
    return _failed(
        criterion,
        "Source and target domains were not declared cross-domain.",
        ("domain-pair",),
    )


def _evaluate_mapping_validity(
    criterion: FalsificationCriterion,
    mapping: TransferMapping,
) -> CriterionEvaluation:
    if mapping.is_usable_for_prediction():
        return _passed(
            criterion,
            "Mapping is usable for bounded prediction.",
            (mapping.function_id, mapping.target_domain_id),
        )
    return _failed(
        criterion,
        "Mapping is insufficient and cannot support prediction.",
        (mapping.function_id, mapping.target_domain_id),
    )


def _evaluate_outcome_support(
    criterion: FalsificationCriterion,
    report: RealityDeltaReport,
) -> CriterionEvaluation:
    if report.status in {
        TransferOutcomeStatus.SUPPORTED,
        TransferOutcomeStatus.MIXED,
    }:
        return _passed(
            criterion,
            f"Reality-delta status was {report.status.value!r}.",
            (report.report_id,),
        )
    return _failed(
        criterion,
        f"Reality-delta status was {report.status.value!r}.",
        (report.report_id,),
    )


def _evaluate_uncertainty(
    criterion: FalsificationCriterion,
    uncertainty_ledger: UncertaintyLedger,
) -> CriterionEvaluation:
    blocking_items = uncertainty_ledger.blocking_items()
    if not blocking_items:
        return _passed(
            criterion,
            "Uncertainty ledger has no blocking items.",
            (uncertainty_ledger.ledger_id,),
        )
    return _failed(
        criterion,
        "Uncertainty ledger contains blocking or escalated items.",
        (
            uncertainty_ledger.ledger_id,
            *(item.uncertainty_id for item in blocking_items),
        ),
    )


def _evaluate_future_behavior(
    criterion: FalsificationCriterion,
    learning_update: TransferLearningUpdate,
) -> CriterionEvaluation:
    if learning_update.should_update_future_behavior():
        return _passed(
            criterion,
            "Learning update contains future behavior guidance.",
            (learning_update.update_id,),
        )
    return _failed(
        criterion,
        "Learning update does not contain usable future behavior guidance.",
        (learning_update.update_id,),
    )


def _evaluate_reproducibility(
    *,
    criterion: FalsificationCriterion,
    function_id: str,
    mapping: TransferMapping,
    report: RealityDeltaReport,
    uncertainty_ledger: UncertaintyLedger,
    learning_update: TransferLearningUpdate,
) -> CriterionEvaluation:
    refs = (
        function_id,
        mapping.function_id,
        mapping.target_domain_id,
        report.report_id,
        uncertainty_ledger.ledger_id,
        learning_update.update_id,
    )
    if all(ref.strip() for ref in refs):
        return _passed(
            criterion,
            "Stable evidence references are present for replay.",
            refs,
        )
    return _warning(
        criterion,
        "One or more evidence references are missing or empty.",
        tuple(ref for ref in refs if ref.strip()) or ("missing-evidence-ref",),
    )
