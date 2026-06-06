from __future__ import annotations

from ix_function.doctrine import (
    BLOCKED_CLAIM_TERMS,
    DOCTRINE_RULES,
    ClaimLevel,
    doctrine_index,
    evaluate_claim_boundary,
)


def test_doctrine_rules_have_stable_unique_ids() -> None:
    index = doctrine_index()

    assert len(DOCTRINE_RULES) >= 6
    assert len(index) == len(DOCTRINE_RULES)
    assert all(rule_id.startswith("IXF-DOCTRINE-") for rule_id in index)


def test_doctrine_includes_core_transfer_constraints() -> None:
    statements = "\n".join(rule.statement for rule in DOCTRINE_RULES)

    assert "not an AGI claim" in statements
    assert "Prediction must precede outcome observation" in statements
    assert "Uncertainty must be preserved" in statements
    assert "Negative controls" in statements
    assert "Human authority" in statements


def test_blocked_claim_terms_are_not_empty() -> None:
    assert "agi proven" in BLOCKED_CLAIM_TERMS
    assert "production-ready autonomy" in BLOCKED_CLAIM_TERMS


def test_evaluate_claim_boundary_blocks_agi_proof_language() -> None:
    boundary = evaluate_claim_boundary(
        "This system is AGI proven and ready for production autonomy."
    )

    assert boundary.allowed is False
    assert boundary.maximum_allowed_level is ClaimLevel.AGI_CANDIDATE_EVIDENCE
    assert "blocked maturity language" in boundary.reason
    assert "agi proven" in boundary.reason


def test_evaluate_claim_boundary_allows_bounded_candidate_language() -> None:
    boundary = evaluate_claim_boundary(
        "IX-Function produced governed AGI-candidate evidence for review."
    )

    assert boundary.allowed is True
    assert boundary.maximum_allowed_level is ClaimLevel.AGI_CANDIDATE_EVIDENCE
    assert "human review" in boundary.reason
