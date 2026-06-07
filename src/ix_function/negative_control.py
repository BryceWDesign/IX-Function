"""Negative controls for IX-Function.

Negative controls prove the harness does not pass every transfer attempt. A
serious cross-domain causal-transfer system must be able to reject bad mappings,
same-domain theater, unsupported outcomes, and leaked-answer predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.falsification import FalsificationLedger, FalsificationVerdict
from ix_function.learning import LearningDisposition, TransferLearningUpdate
from ix_function.mapping import MappingQuality, TransferMapping
from ix_function.reality_delta import RealityDeltaReport, TransferOutcomeStatus


class NegativeControlKind(StrEnum):
    """Negative-control pattern used to pressure the transfer chain."""

    EXPECTED_FAILURE = "expected_failure"
    INSUFFICIENT_MAPPING = "insufficient_mapping"
    OUTCOME_LEAKAGE = "outcome_leakage"
    SAME_DOMAIN_THEATER = "same_domain_theater"
    SHUFFLED_MAPPING = "shuffled_mapping"


class NegativeControlStatus(StrEnum):
    """Outcome of a negative-control evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class NegativeControlSpec:
    """A negative control that should not strengthen transfer confidence."""

    control_id: str
    kind: NegativeControlKind
    purpose: str
    expected_blocking_behavior: str


@dataclass(frozen=True, slots=True)
class NegativeControlEvaluation:
    """Evaluation result for one negative control."""

    control_id: str
    kind: NegativeControlKind
    status: NegativeControlStatus
    reason: str
    evidence_refs: tuple[str, ...]

    def is_clean(self) -> bool:
        """Return whether the negative control behaved as expected."""

        return self.status is NegativeControlStatus.PASSED


@dataclass(frozen=True, slots=True)
class NegativeControlSuite:
    """Reviewable negative-control results for a transfer evidence chain."""

    suite_id: str
    evaluations: tuple[NegativeControlEvaluation, ...]

    def failed_controls(self) -> tuple[NegativeControlEvaluation, ...]:
        """Return negative controls that failed to reject bad evidence."""

        return tuple(
            evaluation
            for evaluation in self.evaluations
            if evaluation.status is NegativeControlStatus.FAILED
        )

    def passed(self) -> bool:
        """Return whether every negative control behaved as expected."""

        return bool(self.evaluations) and not self.failed_controls()


@dataclass(frozen=True, slots=True)
class AntiTheaterGateResult:
    """Gate result that blocks promotion when negative controls fail."""

    suite_id: str
    allowed: bool
    failed_control_ids: tuple[str, ...]
    reason: str
    required_actions: tuple[str, ...]


def default_negative_controls() -> tuple[NegativeControlSpec, ...]:
    """Return baseline negative controls for Wave 6 causal-transfer trials."""

    return (
        NegativeControlSpec(
            control_id="IXF-NEG-001",
            kind=NegativeControlKind.INSUFFICIENT_MAPPING,
            purpose="Confirm insufficient mappings cannot support prediction.",
            expected_blocking_behavior=(
                "Mapping quality must be insufficient or falsification must kill "
                "the claim."
            ),
        ),
        NegativeControlSpec(
            control_id="IXF-NEG-002",
            kind=NegativeControlKind.SAME_DOMAIN_THEATER,
            purpose="Confirm same-domain transfer theater cannot pass as Wave 6.",
            expected_blocking_behavior=(
                "Falsification verdict must kill or downgrade the claim when "
                "source and target are not genuinely cross-domain."
            ),
        ),
        NegativeControlSpec(
            control_id="IXF-NEG-003",
            kind=NegativeControlKind.EXPECTED_FAILURE,
            purpose="Confirm failed outcomes reduce or block transfer confidence.",
            expected_blocking_behavior=(
                "Reality-delta failure must produce a failed outcome, weakened "
                "learning, quarantine, downgrade, or kill verdict."
            ),
        ),
        NegativeControlSpec(
            control_id="IXF-NEG-004",
            kind=NegativeControlKind.OUTCOME_LEAKAGE,
            purpose="Confirm leaked-answer predictions cannot be promoted.",
            expected_blocking_behavior=(
                "Blocked or invalid prediction lineage must prevent promotion."
            ),
        ),
        NegativeControlSpec(
            control_id="IXF-NEG-005",
            kind=NegativeControlKind.SHUFFLED_MAPPING,
            purpose="Confirm arbitrary slot mappings cannot create strong support.",
            expected_blocking_behavior=(
                "Ambiguous or insufficient mapping warnings must remain visible "
                "and block strong support when unresolved."
            ),
        ),
    )


def evaluate_negative_control(
    *,
    spec: NegativeControlSpec,
    mapping: TransferMapping,
    report: RealityDeltaReport,
    learning_update: TransferLearningUpdate,
    falsification_ledger: FalsificationLedger,
) -> NegativeControlEvaluation:
    """Evaluate one negative control against the transfer evidence chain."""

    if spec.kind is NegativeControlKind.INSUFFICIENT_MAPPING:
        passed = (
            mapping.quality is MappingQuality.INSUFFICIENT
            or falsification_ledger.verdict is FalsificationVerdict.KILL_CLAIM
        )
        return _evaluation_from_bool(
            spec=spec,
            passed=passed,
            passed_reason="Insufficient mapping was blocked.",
            failed_reason="Insufficient mapping was allowed to behave as support.",
            evidence_refs=(mapping.function_id, mapping.target_domain_id),
        )

    if spec.kind is NegativeControlKind.SAME_DOMAIN_THEATER:
        passed = falsification_ledger.verdict in {
            FalsificationVerdict.KILL_CLAIM,
            FalsificationVerdict.DOWNGRADE_CLAIM,
        }
        return _evaluation_from_bool(
            spec=spec,
            passed=passed,
            passed_reason="Same-domain theater was not promoted.",
            failed_reason="Same-domain theater was allowed as bounded evidence.",
            evidence_refs=(falsification_ledger.ledger_id,),
        )

    if spec.kind is NegativeControlKind.EXPECTED_FAILURE:
        passed = (
            report.status in {
                TransferOutcomeStatus.FAILED,
                TransferOutcomeStatus.UNSCORABLE,
            }
            and learning_update.disposition in {
                LearningDisposition.WEAKEN,
                LearningDisposition.QUARANTINE,
            }
        )
        return _evaluation_from_bool(
            spec=spec,
            passed=passed,
            passed_reason="Failed or unscorable outcome weakened the transfer.",
            failed_reason="Failed outcome did not weaken future behavior.",
            evidence_refs=(report.report_id, learning_update.update_id),
        )

    if spec.kind is NegativeControlKind.OUTCOME_LEAKAGE:
        passed = (
            report.status is TransferOutcomeStatus.UNSCORABLE
            or bool(report.blocking_errors)
            or bool(learning_update.blocking_errors)
            or falsification_ledger.verdict is FalsificationVerdict.KILL_CLAIM
        )
        return _evaluation_from_bool(
            spec=spec,
            passed=passed,
            passed_reason="Leaked or invalid outcome lineage was blocked.",
            failed_reason="Outcome leakage did not block promotion.",
            evidence_refs=(report.report_id, learning_update.update_id),
        )

    if spec.kind is NegativeControlKind.SHUFFLED_MAPPING:
        passed = (
            mapping.quality in {
                MappingQuality.AMBIGUOUS,
                MappingQuality.INSUFFICIENT,
            }
            or bool(mapping.warnings)
            or falsification_ledger.verdict is not (
                FalsificationVerdict.ALLOW_BOUNDED_EVIDENCE
            )
        )
        return _evaluation_from_bool(
            spec=spec,
            passed=passed,
            passed_reason="Shuffled or weak mapping remained visible to gates.",
            failed_reason="Shuffled mapping was allowed as clean support.",
            evidence_refs=(mapping.function_id, mapping.target_domain_id),
        )

    raise AssertionError(f"Unhandled negative control kind: {spec.kind!r}")


def build_negative_control_suite(
    *,
    suite_id: str,
    mapping: TransferMapping,
    report: RealityDeltaReport,
    learning_update: TransferLearningUpdate,
    falsification_ledger: FalsificationLedger,
    controls: tuple[NegativeControlSpec, ...] | None = None,
) -> NegativeControlSuite:
    """Evaluate a suite of negative controls."""

    active_controls = controls or default_negative_controls()
    return NegativeControlSuite(
        suite_id=suite_id,
        evaluations=tuple(
            evaluate_negative_control(
                spec=control,
                mapping=mapping,
                report=report,
                learning_update=learning_update,
                falsification_ledger=falsification_ledger,
            )
            for control in active_controls
        ),
    )


def evaluate_anti_theater_gate(
    suite: NegativeControlSuite,
) -> AntiTheaterGateResult:
    """Block promotion when negative controls fail."""

    validation_errors = validate_negative_control_suite(suite)
    if validation_errors:
        return AntiTheaterGateResult(
            suite_id=suite.suite_id,
            allowed=False,
            failed_control_ids=(),
            reason="Anti-theater gate blocked because the suite is invalid.",
            required_actions=validation_errors,
        )

    failed_controls = suite.failed_controls()
    if failed_controls:
        return AntiTheaterGateResult(
            suite_id=suite.suite_id,
            allowed=False,
            failed_control_ids=tuple(control.control_id for control in failed_controls),
            reason=(
                "Anti-theater gate blocked because one or more negative controls "
                "failed to reject bad evidence."
            ),
            required_actions=tuple(
                f"Fix negative control {control.control_id}: {control.reason}"
                for control in failed_controls
            ),
        )

    return AntiTheaterGateResult(
        suite_id=suite.suite_id,
        allowed=True,
        failed_control_ids=(),
        reason=(
            "Anti-theater gate allows bounded evidence because negative controls "
            "did not promote bad transfer evidence."
        ),
        required_actions=(
            "Preserve negative-control results in the evidence packet.",
            "Do not represent negative-control passage as AGI proof.",
        ),
    )


def validate_negative_control_spec(spec: NegativeControlSpec) -> tuple[str, ...]:
    """Return validation errors for one negative-control spec."""

    errors: list[str] = []
    if not spec.control_id.strip():
        errors.append("control_id must not be empty")
    if not spec.purpose.strip():
        errors.append(f"purpose must not be empty for {spec.control_id!r}")
    if not spec.expected_blocking_behavior.strip():
        errors.append(
            f"expected_blocking_behavior must not be empty for {spec.control_id!r}"
        )
    return tuple(errors)


def validate_negative_control_suite(
    suite: NegativeControlSuite,
) -> tuple[str, ...]:
    """Return validation errors for a negative-control suite."""

    errors: list[str] = []
    if not suite.suite_id.strip():
        errors.append("suite_id must not be empty")
    if not suite.evaluations:
        errors.append("at least one negative-control evaluation is required")

    control_ids = [evaluation.control_id for evaluation in suite.evaluations]
    if len(set(control_ids)) != len(control_ids):
        errors.append("negative-control evaluations must use unique control_id values")

    for evaluation in suite.evaluations:
        if not evaluation.control_id.strip():
            errors.append("evaluation control_id must not be empty")
        if not evaluation.reason.strip():
            errors.append(f"reason must not be empty for {evaluation.control_id!r}")
        if not evaluation.evidence_refs:
            errors.append(
                f"evidence_refs must not be empty for {evaluation.control_id!r}"
            )

    return tuple(errors)


def _evaluation_from_bool(
    *,
    spec: NegativeControlSpec,
    passed: bool,
    passed_reason: str,
    failed_reason: str,
    evidence_refs: tuple[str, ...],
) -> NegativeControlEvaluation:
    return NegativeControlEvaluation(
        control_id=spec.control_id,
        kind=spec.kind,
        status=NegativeControlStatus.PASSED
        if passed
        else NegativeControlStatus.FAILED,
        reason=passed_reason if passed else failed_reason,
        evidence_refs=evidence_refs,
    )
