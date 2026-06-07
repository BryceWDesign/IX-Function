"""IX declarative contract handoff packets for IX-Function.

The IX handoff converts IX-Function trial evidence into a declarative contract
shape that can be reviewed, replayed, and constrained. The contract is not an
execution grant. It is a bounded representation of intent, preconditions,
evidence requirements, prohibited claims, and human authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.trial import TransferTrialResult, TrialStatus


class IXContractHandoffStatus(StrEnum):
    """Status of an IX declarative contract handoff."""

    READY_FOR_REVIEW_CONTRACT = "ready_for_review_contract"
    READY_FOR_FAILURE_CONTRACT = "ready_for_failure_contract"
    BLOCKED = "blocked"


class IXContractStepKind(StrEnum):
    """Kinds of declarative steps exported to IX."""

    ASSERT_BOUNDARY = "assert_boundary"
    BIND_EVIDENCE = "bind_evidence"
    CHECK_FALSIFICATION = "check_falsification"
    CHECK_PERMISSION = "check_permission"
    PRESERVE_UNCERTAINTY = "preserve_uncertainty"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class IXContractAuthority(StrEnum):
    """Authority state for an IX contract."""

    HUMAN_REVIEW_REQUIRED = "human_review_required"
    NO_AUTOMATIC_AUTHORITY = "no_automatic_authority"


@dataclass(frozen=True, slots=True)
class IXContractPrecondition:
    """Precondition that must hold before contract review."""

    precondition_id: str
    statement: str
    evidence_ref: str
    required: bool


@dataclass(frozen=True, slots=True)
class IXContractStep:
    """Declarative contract step for IX review."""

    step_id: str
    kind: IXContractStepKind
    statement: str
    evidence_refs: tuple[str, ...]
    required: bool


@dataclass(frozen=True, slots=True)
class IXContractBoundary:
    """Boundary terms that the IX contract must preserve."""

    boundary_id: str
    allowed_claims: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    authority: IXContractAuthority
    human_review_required: bool


@dataclass(frozen=True, slots=True)
class IXContractHandoffPacket:
    """Complete IX declarative contract handoff packet."""

    packet_id: str
    status: IXContractHandoffStatus
    trial_id: str
    contract_id: str
    preconditions: tuple[IXContractPrecondition, ...]
    steps: tuple[IXContractStep, ...]
    boundary: IXContractBoundary
    required_ix_actions: tuple[str, ...]
    claim_boundary: str


def build_ix_contract_handoff_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> IXContractHandoffPacket:
    """Build an IX declarative contract handoff from transfer evidence."""

    status = choose_ix_contract_handoff_status(result)
    return IXContractHandoffPacket(
        packet_id=f"{result.trial_id}:ix-contract-handoff",
        status=status,
        trial_id=result.trial_id,
        contract_id=f"{result.trial_id}:ix-review-contract",
        preconditions=build_ix_preconditions(
            result=result,
            evidence_packet=evidence_packet,
        ),
        steps=build_ix_contract_steps(result=result, evidence_packet=evidence_packet),
        boundary=build_ix_contract_boundary(result),
        required_ix_actions=required_ix_actions(result, status),
        claim_boundary=ix_contract_claim_boundary(),
    )


def choose_ix_contract_handoff_status(
    result: TransferTrialResult,
) -> IXContractHandoffStatus:
    """Choose IX handoff status from transfer trial result."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return IXContractHandoffStatus.READY_FOR_REVIEW_CONTRACT
    if result.reality_delta.observable_deltas:
        return IXContractHandoffStatus.READY_FOR_FAILURE_CONTRACT
    return IXContractHandoffStatus.BLOCKED


def build_ix_preconditions(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> tuple[IXContractPrecondition, ...]:
    """Build required preconditions for IX contract review."""

    return (
        IXContractPrecondition(
            precondition_id=f"{result.trial_id}:precondition:evidence-packet",
            statement="Evidence packet must be present and digest-bound.",
            evidence_ref=evidence_packet.packet_id,
            required=True,
        ),
        IXContractPrecondition(
            precondition_id=f"{result.trial_id}:precondition:prediction-before-outcome",
            statement="Prediction must have been committed before outcome scoring.",
            evidence_ref=result.prediction_readiness.prediction_id,
            required=True,
        ),
        IXContractPrecondition(
            precondition_id=f"{result.trial_id}:precondition:falsification",
            statement="Falsification ledger must be preserved with the contract.",
            evidence_ref=result.falsification_ledger.ledger_id,
            required=True,
        ),
        IXContractPrecondition(
            precondition_id=f"{result.trial_id}:precondition:human-authority",
            statement="Human review remains required for any downstream reuse.",
            evidence_ref=result.trial_id,
            required=True,
        ),
    )


def build_ix_contract_steps(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> tuple[IXContractStep, ...]:
    """Build declarative IX contract steps."""

    return (
        IXContractStep(
            step_id=f"{result.trial_id}:step:bind-evidence",
            kind=IXContractStepKind.BIND_EVIDENCE,
            statement=(
                "Bind review to the IX-Function evidence packet and manifest "
                "digest."
            ),
            evidence_refs=(
                evidence_packet.packet_id,
                evidence_packet.manifest_sha256_digest,
            ),
            required=True,
        ),
        IXContractStep(
            step_id=f"{result.trial_id}:step:preserve-uncertainty",
            kind=IXContractStepKind.PRESERVE_UNCERTAINTY,
            statement="Preserve all open uncertainty references during review.",
            evidence_refs=tuple(
                item.uncertainty_id for item in result.uncertainty_ledger.open_items()
            )
            or (result.uncertainty_ledger.ledger_id,),
            required=True,
        ),
        IXContractStep(
            step_id=f"{result.trial_id}:step:check-falsification",
            kind=IXContractStepKind.CHECK_FALSIFICATION,
            statement="Apply falsification verdict before any claim promotion.",
            evidence_refs=(
                result.falsification_ledger.ledger_id,
                result.falsification_ledger.verdict.value,
            ),
            required=True,
        ),
        IXContractStep(
            step_id=f"{result.trial_id}:step:check-permission",
            kind=IXContractStepKind.CHECK_PERMISSION,
            statement="Require human permission before future transfer reuse.",
            evidence_refs=(result.trial_id,),
            required=True,
        ),
        IXContractStep(
            step_id=f"{result.trial_id}:step:assert-boundary",
            kind=IXContractStepKind.ASSERT_BOUNDARY,
            statement=(
                "Assert that this contract does not grant AGI proof, deployment "
                "authority, or automatic execution authority."
            ),
            evidence_refs=(result.trial_id,),
            required=True,
        ),
        IXContractStep(
            step_id=f"{result.trial_id}:step:human-review",
            kind=IXContractStepKind.REQUIRE_HUMAN_REVIEW,
            statement="Route final interpretation to human review.",
            evidence_refs=(result.trial_id,),
            required=True,
        ),
    )


def build_ix_contract_boundary(result: TransferTrialResult) -> IXContractBoundary:
    """Build fixed IX contract boundary terms."""

    allowed_claims = (
        "bounded causal-transfer evidence",
        "reviewable trial result",
        "falsifiable candidate evidence",
    )
    prohibited_claims = (
        "AGI proof",
        "certified AGI",
        "deployment authorization",
        "production readiness",
        "self-approval",
        "unsupervised operational autonomy",
    )
    if not result.permits_bounded_evidence():
        allowed_claims = (
            "failure evidence",
            "blocked transfer record",
            "retest requirement",
        )

    return IXContractBoundary(
        boundary_id=f"{result.trial_id}:ix-boundary",
        allowed_claims=allowed_claims,
        prohibited_claims=prohibited_claims,
        authority=IXContractAuthority.HUMAN_REVIEW_REQUIRED,
        human_review_required=True,
    )


def required_ix_actions(
    result: TransferTrialResult,
    status: IXContractHandoffStatus,
) -> tuple[str, ...]:
    """Return required IX-side review actions."""

    if status is IXContractHandoffStatus.READY_FOR_REVIEW_CONTRACT:
        return (
            "Review contract as bounded evidence only.",
            "Bind all review steps to the evidence packet digest.",
            "Preserve uncertainty and falsification references.",
            "Require human authority before future reuse.",
        )

    if status is IXContractHandoffStatus.READY_FOR_FAILURE_CONTRACT:
        return (
            "Review contract as failure or downgrade evidence.",
            "Block claim promotion.",
            "Require retest before any future reuse contract.",
            *result.required_actions,
        )

    return (
        "Block IX contract promotion until complete evidence exists.",
        "Require human review before retry.",
        *result.required_actions,
    )


def ix_contract_claim_boundary() -> str:
    """Return fixed IX contract claim boundary."""

    return (
        "IX-Function IX contract handoff is a declarative review contract. It is "
        "not AGI proof, not execution permission, not deployment authority, and "
        "not self-approval."
    )


def validate_ix_contract_handoff_packet(
    packet: IXContractHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors for an IX contract handoff packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not packet.contract_id.strip():
        errors.append("contract_id must not be empty")
    if not packet.preconditions:
        errors.append("at least one precondition is required")
    if not packet.steps:
        errors.append("at least one contract step is required")
    if not packet.boundary.boundary_id.strip():
        errors.append("boundary_id must not be empty")
    if not packet.boundary.allowed_claims:
        errors.append("allowed_claims must not be empty")
    if not packet.boundary.prohibited_claims:
        errors.append("prohibited_claims must not be empty")
    if not packet.boundary.human_review_required:
        errors.append("human_review_required must remain true")
    if packet.boundary.authority is not IXContractAuthority.HUMAN_REVIEW_REQUIRED:
        errors.append("boundary authority must require human review")
    if not packet.required_ix_actions:
        errors.append("required_ix_actions must not be empty")
    if packet.claim_boundary != ix_contract_claim_boundary():
        errors.append("claim_boundary must match fixed IX contract boundary")

    precondition_ids = [
        precondition.precondition_id for precondition in packet.preconditions
    ]
    if len(set(precondition_ids)) != len(precondition_ids):
        errors.append("preconditions must use unique precondition_id values")

    step_ids = [step.step_id for step in packet.steps]
    if len(set(step_ids)) != len(step_ids):
        errors.append("steps must use unique step_id values")

    for precondition in packet.preconditions:
        if not precondition.precondition_id.strip():
            errors.append("precondition_id must not be empty")
        if not precondition.statement.strip():
            errors.append(
                f"statement must not be empty for {precondition.precondition_id!r}"
            )
        if not precondition.evidence_ref.strip():
            errors.append(
                f"evidence_ref must not be empty for "
                f"{precondition.precondition_id!r}"
            )

    for step in packet.steps:
        if not step.step_id.strip():
            errors.append("step_id must not be empty")
        if not step.statement.strip():
            errors.append(f"statement must not be empty for {step.step_id!r}")
        if not step.evidence_refs:
            errors.append(f"evidence_refs must not be empty for {step.step_id!r}")

    return tuple(errors)
