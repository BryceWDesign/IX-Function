"""Project doctrine and claim boundaries for IX-Function.

The doctrine layer exists before any causal-transfer implementation so later
modules cannot silently convert a transfer trial into an AGI or autonomy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClaimLevel(str, Enum):
    """Allowed maturity language for IX-Function outputs."""

    RESEARCH_PROTOTYPE = "research_prototype"
    GOVERNED_TRANSFER_HARNESS = "governed_transfer_harness"
    AGI_CANDIDATE_EVIDENCE = "agi_candidate_evidence"


@dataclass(frozen=True, slots=True)
class DoctrineRule:
    """A non-negotiable project rule with a stable identifier."""

    rule_id: str
    statement: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ClaimBoundary:
    """Result of checking whether a public or machine claim is allowed."""

    requested_claim: str
    allowed: bool
    maximum_allowed_level: ClaimLevel
    reason: str


DOCTRINE_RULES: tuple[DoctrineRule, ...] = (
    DoctrineRule(
        rule_id="IXF-DOCTRINE-001",
        statement="IX-Function is not an AGI claim.",
        rationale=(
            "Cross-domain causal transfer evidence may support review, but it "
            "does not by itself prove artificial general intelligence."
        ),
    ),
    DoctrineRule(
        rule_id="IXF-DOCTRINE-002",
        statement="Prediction must precede outcome observation.",
        rationale=(
            "A transfer trial is not valid evidence when the system sees the "
            "answer before committing to a measurable prediction."
        ),
    ),
    DoctrineRule(
        rule_id="IXF-DOCTRINE-003",
        statement="Reality feedback outranks narrative confidence.",
        rationale=(
            "The system must revise confidence from observed outcomes instead "
            "of preserving flattering explanations after failed transfer."
        ),
    ),
    DoctrineRule(
        rule_id="IXF-DOCTRINE-004",
        statement="Uncertainty must be preserved, not laundered into certainty.",
        rationale=(
            "Ambiguous mappings, noisy outcomes, and partial transfer results "
            "must remain visible in evidence artifacts."
        ),
    ),
    DoctrineRule(
        rule_id="IXF-DOCTRINE-005",
        statement="Negative controls must be able to fail the claim.",
        rationale=(
            "A harness that passes every transfer attempt is not falsifying the "
            "claim; it is only manufacturing approval."
        ),
    ),
    DoctrineRule(
        rule_id="IXF-DOCTRINE-006",
        statement="Human authority remains outside the model loop.",
        rationale=(
            "Model output, transfer receipts, and confidence updates can inform "
            "review but cannot self-authorize AGI, autonomy, or deployment claims."
        ),
    ),
)

BLOCKED_CLAIM_TERMS: tuple[str, ...] = (
    "agi proven",
    "artificial general intelligence proven",
    "certified agi",
    "independent agi validation",
    "production autonomy",
    "production-ready autonomy",
    "self-authorizing",
    "unsupervised operational autonomy",
)


def doctrine_index() -> dict[str, DoctrineRule]:
    """Return doctrine rules keyed by their stable rule identifier."""

    return {rule.rule_id: rule for rule in DOCTRINE_RULES}


def evaluate_claim_boundary(requested_claim: str) -> ClaimBoundary:
    """Check whether a requested claim violates IX-Function boundaries."""

    normalized = " ".join(requested_claim.lower().split())
    blocked_terms = tuple(term for term in BLOCKED_CLAIM_TERMS if term in normalized)
    if blocked_terms:
        return ClaimBoundary(
            requested_claim=requested_claim,
            allowed=False,
            maximum_allowed_level=ClaimLevel.AGI_CANDIDATE_EVIDENCE,
            reason=(
                "Requested claim uses blocked maturity language: "
                f"{', '.join(blocked_terms)}. IX-Function can produce "
                "AGI-candidate evidence, not proof of AGI or deployment authority."
            ),
        )

    return ClaimBoundary(
        requested_claim=requested_claim,
        allowed=True,
        maximum_allowed_level=ClaimLevel.AGI_CANDIDATE_EVIDENCE,
        reason=(
            "Claim is inside IX-Function boundaries when supported by evidence, "
            "falsification records, uncertainty, and human review."
        ),
    )
