"""End-to-end transfer trial orchestration for IX-Function.

The trial layer binds the core IX-Function evidence chain:

source learning -> target mapping -> pre-outcome prediction -> reality delta ->
learning update -> uncertainty ledger -> falsification -> negative controls.

This module does not invent outcomes and does not claim AGI. It only assembles
already-committed records into a reviewable transfer trial result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.causal_function import CausalFunction, validate_causal_function
from ix_function.domain import DomainPair, validate_domain_pair
from ix_function.falsification import (
    FalsificationLedger,
    FalsificationVerdict,
    build_falsification_ledger,
    validate_falsification_ledger,
)
from ix_function.learning import (
    TransferLearningUpdate,
    build_learning_update,
)
from ix_function.mapping import (
    TransferMapping,
    propose_transfer_mapping,
    validate_transfer_mapping,
)
from ix_function.negative_control import (
    AntiTheaterGateResult,
    NegativeControlSuite,
    build_negative_control_suite,
    evaluate_anti_theater_gate,
    validate_negative_control_suite,
)
from ix_function.observation import (
    DomainSnapshot,
    InterventionRecord,
    OutcomeRecord,
    validate_intervention_against_domain,
    validate_outcome_against_domain,
    validate_snapshot_against_domain,
)
from ix_function.prediction import (
    PredictionReadiness,
    TransferPrediction,
    assess_prediction_readiness,
)
from ix_function.reality_delta import (
    RealityDeltaReport,
    build_reality_delta_report,
)
from ix_function.source_learning import (
    SourceLearningEvidence,
    SourceLearningTrial,
    evaluate_source_learning_trial,
)
from ix_function.uncertainty import (
    EvidenceClaimStrength,
    UncertaintyGateResult,
    UncertaintyLedger,
    build_uncertainty_ledger,
    evaluate_claim_strength_gate,
    evaluate_uncertainty_gate,
    validate_uncertainty_ledger,
)


class TrialStatus(StrEnum):
    """Final orchestration status for a transfer trial."""

    BOUNDED_EVIDENCE_ALLOWED = "bounded_evidence_allowed"
    BLOCKED = "blocked"
    DOWNGRADED = "downgraded"
    INVALID = "invalid"
    RETEST_REQUIRED = "retest_required"


@dataclass(frozen=True, slots=True)
class TransferTrialInput:
    """Committed input records required to run a transfer trial."""

    trial_id: str
    domain_pair: DomainPair
    causal_function: CausalFunction
    source_baseline: DomainSnapshot
    source_intervention: InterventionRecord
    source_outcome: OutcomeRecord
    source_support_reasons: tuple[str, ...]
    source_uncertainty_notes: tuple[str, ...]
    target_baseline: DomainSnapshot
    target_intervention: InterventionRecord
    target_outcome: OutcomeRecord
    prediction: TransferPrediction


@dataclass(frozen=True, slots=True)
class TransferTrialResult:
    """End-to-end result for one IX-Function transfer trial."""

    trial_id: str
    status: TrialStatus
    source_evidence: SourceLearningEvidence
    mapping: TransferMapping
    prediction_readiness: PredictionReadiness
    reality_delta: RealityDeltaReport
    learning_update: TransferLearningUpdate
    uncertainty_ledger: UncertaintyLedger
    uncertainty_gate: UncertaintyGateResult
    falsification_ledger: FalsificationLedger
    negative_control_suite: NegativeControlSuite
    anti_theater_gate: AntiTheaterGateResult
    blocking_errors: tuple[str, ...]
    required_actions: tuple[str, ...]

    def permits_bounded_evidence(self) -> bool:
        """Return whether the trial permits bounded evidence language."""

        return self.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED


def run_transfer_trial(trial_input: TransferTrialInput) -> TransferTrialResult:
    """Run a full IX-Function transfer trial from committed records."""

    preflight_errors = validate_transfer_trial_input(trial_input)
    source_trial = SourceLearningTrial(
        trial_id=trial_input.trial_id,
        source_domain=trial_input.domain_pair.source,
        causal_function=trial_input.causal_function,
        baseline_snapshot=trial_input.source_baseline,
        intervention=trial_input.source_intervention,
        outcome=trial_input.source_outcome,
        support_reasons=trial_input.source_support_reasons,
        uncertainty_notes=trial_input.source_uncertainty_notes,
    )
    source_evidence = evaluate_source_learning_trial(source_trial)
    mapping = propose_transfer_mapping(
        trial_input.causal_function,
        trial_input.domain_pair.target,
    )
    prediction_readiness = assess_prediction_readiness(
        trial_input.prediction,
        mapping,
    )
    reality_delta = build_reality_delta_report(
        trial_input.prediction,
        trial_input.target_baseline,
        trial_input.target_outcome,
    )
    learning_update = build_learning_update(
        trial_input.causal_function,
        reality_delta,
    )
    uncertainty_ledger = build_uncertainty_ledger(
        ledger_id=f"{trial_input.trial_id}:uncertainty-ledger",
        mapping=mapping,
        prediction=trial_input.prediction,
        report=reality_delta,
        learning_update=learning_update,
    )
    uncertainty_gate = evaluate_uncertainty_gate(uncertainty_ledger)
    falsification_ledger = build_falsification_ledger(
        ledger_id=f"{trial_input.trial_id}:falsification-ledger",
        function_id=trial_input.causal_function.function_id,
        is_cross_domain=trial_input.domain_pair.is_cross_domain(),
        mapping=mapping,
        report=reality_delta,
        uncertainty_ledger=uncertainty_ledger,
        learning_update=learning_update,
    )
    negative_control_suite = build_negative_control_suite(
        suite_id=f"{trial_input.trial_id}:negative-control-suite",
        mapping=mapping,
        report=reality_delta,
        learning_update=learning_update,
        falsification_ledger=falsification_ledger,
    )
    anti_theater_gate = evaluate_anti_theater_gate(negative_control_suite)

    blocking_errors = tuple(
        error
        for error in (
            *preflight_errors,
            *source_evidence.blocking_errors,
            *prediction_readiness.blocking_errors,
            *reality_delta.blocking_errors,
            *learning_update.blocking_errors,
            *validate_uncertainty_ledger(uncertainty_ledger),
            *validate_falsification_ledger(falsification_ledger),
            *validate_negative_control_suite(negative_control_suite),
        )
    )
    status = choose_trial_status(
        blocking_errors=blocking_errors,
        uncertainty_allowed=uncertainty_gate.allowed,
        falsification_verdict=falsification_ledger.verdict,
        anti_theater_allowed=anti_theater_gate.allowed,
    )

    return TransferTrialResult(
        trial_id=trial_input.trial_id,
        status=status,
        source_evidence=source_evidence,
        mapping=mapping,
        prediction_readiness=prediction_readiness,
        reality_delta=reality_delta,
        learning_update=learning_update,
        uncertainty_ledger=uncertainty_ledger,
        uncertainty_gate=uncertainty_gate,
        falsification_ledger=falsification_ledger,
        negative_control_suite=negative_control_suite,
        anti_theater_gate=anti_theater_gate,
        blocking_errors=blocking_errors,
        required_actions=required_actions_for_trial(
            status=status,
            blocking_errors=blocking_errors,
            uncertainty_gate=uncertainty_gate,
            falsification_ledger=falsification_ledger,
            anti_theater_gate=anti_theater_gate,
        ),
    )


def choose_trial_status(
    *,
    blocking_errors: tuple[str, ...],
    uncertainty_allowed: bool,
    falsification_verdict: FalsificationVerdict,
    anti_theater_allowed: bool,
) -> TrialStatus:
    """Choose final transfer-trial status from all gates."""

    if blocking_errors:
        return TrialStatus.INVALID
    if not uncertainty_allowed or not anti_theater_allowed:
        return TrialStatus.BLOCKED
    if falsification_verdict is FalsificationVerdict.KILL_CLAIM:
        return TrialStatus.BLOCKED
    if falsification_verdict is FalsificationVerdict.DOWNGRADE_CLAIM:
        return TrialStatus.DOWNGRADED
    if falsification_verdict is FalsificationVerdict.REQUIRE_RETEST:
        return TrialStatus.RETEST_REQUIRED
    return TrialStatus.BOUNDED_EVIDENCE_ALLOWED


def required_actions_for_trial(
    *,
    status: TrialStatus,
    blocking_errors: tuple[str, ...],
    uncertainty_gate: UncertaintyGateResult,
    falsification_ledger: FalsificationLedger,
    anti_theater_gate: AntiTheaterGateResult,
) -> tuple[str, ...]:
    """Return required review actions for a trial status."""

    if status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return (
            "Allow bounded IX-Function transfer evidence language.",
            "Preserve uncertainty, falsification, and negative-control records.",
            "Do not represent this result as AGI proof.",
        )

    if status is TrialStatus.INVALID:
        return tuple(
            f"Fix invalid trial input or artifact: {error}"
            for error in blocking_errors
        )

    actions: list[str] = []
    if not uncertainty_gate.allowed:
        actions.extend(
            f"Resolve uncertainty blocker {blocking_id}."
            for blocking_id in uncertainty_gate.blocking_ids
        )
    if not anti_theater_gate.allowed:
        actions.extend(anti_theater_gate.required_actions)
    actions.extend(falsification_ledger.required_actions)

    if not actions:
        actions.append("Retest trial before making stronger transfer claims.")

    return tuple(actions)


def validate_transfer_trial_input(trial_input: TransferTrialInput) -> tuple[str, ...]:
    """Return validation errors for a transfer trial input package."""

    errors: list[str] = []
    if not trial_input.trial_id.strip():
        errors.append("trial_id must not be empty")

    errors.extend(
        f"domain pair: {error}" for error in validate_domain_pair(trial_input.domain_pair)
    )
    errors.extend(
        f"causal function: {error}"
        for error in validate_causal_function(trial_input.causal_function)
    )
    errors.extend(
        f"target baseline: {error}"
        for error in validate_snapshot_against_domain(
            trial_input.domain_pair.target,
            trial_input.target_baseline,
        )
    )
    errors.extend(
        f"target intervention: {error}"
        for error in validate_intervention_against_domain(
            trial_input.domain_pair.target,
            trial_input.target_intervention,
        )
    )
    errors.extend(
        f"target outcome: {error}"
        for error in validate_outcome_against_domain(
            trial_input.domain_pair.target,
            trial_input.target_outcome,
        )
    )

    mapping = propose_transfer_mapping(
        trial_input.causal_function,
        trial_input.domain_pair.target,
    )
    errors.extend(
        f"transfer mapping: {error}" for error in validate_transfer_mapping(mapping)
    )

    if trial_input.prediction.function_id != trial_input.causal_function.function_id:
        errors.append("prediction function_id must match causal function_id")
    if trial_input.prediction.target_domain_id != trial_input.domain_pair.target.domain_id:
        errors.append("prediction target_domain_id must match target domain_id")
    if (
        trial_input.prediction.target_intervention_id
        != trial_input.target_intervention.intervention_id
    ):
        errors.append("prediction target_intervention_id must match target intervention")
    if (
        trial_input.target_outcome.observed_after_intervention_id
        != trial_input.target_intervention.intervention_id
    ):
        errors.append("target outcome must reference target intervention")

    return tuple(errors)


def evaluate_bounded_claim_request(
    result: TransferTrialResult,
    requested_strength: EvidenceClaimStrength,
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate whether a finished trial can support requested claim strength."""

    claim_gate = evaluate_claim_strength_gate(
        result.uncertainty_ledger,
        requested_strength,
    )
    if not result.permits_bounded_evidence():
        return (
            False,
            (
                "Transfer trial does not permit bounded evidence.",
                *result.required_actions,
            ),
        )
    if not claim_gate.allowed:
        return (
            False,
            (
                claim_gate.reason,
                *claim_gate.required_actions,
            ),
        )
    return (
        True,
        (
            claim_gate.reason,
            "Do not represent this result as AGI proof.",
        ),
    )
