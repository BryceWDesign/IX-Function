from __future__ import annotations

from ix_function.evidence import build_trial_evidence_packet
from ix_function.kernel_handoff import (
    KernelHandoffStatus,
    KernelMemoryUpdateKind,
    build_kernel_handoff_packet,
    build_kernel_memory_updates,
    choose_kernel_handoff_status,
    kernel_claim_boundary,
    validate_kernel_handoff_packet,
)
from ix_function.observation import MeasuredValue, OutcomeRecord
from ix_function.trial import (
    TransferTrialInput,
    TransferTrialResult,
    run_transfer_trial,
)
from tests.fixtures import make_trial_input


def make_failed_result() -> TransferTrialResult:
    trial_input = make_trial_input()
    failed_input = TransferTrialInput(
        trial_id=trial_input.trial_id,
        domain_pair=trial_input.domain_pair,
        causal_function=trial_input.causal_function,
        source_baseline=trial_input.source_baseline,
        source_intervention=trial_input.source_intervention,
        source_outcome=trial_input.source_outcome,
        source_support_reasons=trial_input.source_support_reasons,
        source_uncertainty_notes=trial_input.source_uncertainty_notes,
        target_baseline=trial_input.target_baseline,
        target_intervention=trial_input.target_intervention,
        target_outcome=OutcomeRecord(
            domain_id="ci-pipeline",
            outcome_id="target-outcome-failed",
            observed_after_intervention_id="increase-worker-count",
            values=(
                MeasuredValue(
                    observable_name="Completion Throughput Output",
                    value=80.0,
                    evidence_id="target-outcome-failed-completion",
                ),
            ),
            result_summary="Completion changed far outside prediction.",
        ),
        prediction=trial_input.prediction,
    )
    return run_transfer_trial(failed_input)


def test_choose_kernel_handoff_status_marks_success_ready_for_review() -> None:
    result = run_transfer_trial(make_trial_input())

    assert choose_kernel_handoff_status(result) is KernelHandoffStatus.READY_FOR_REVIEW


def test_choose_kernel_handoff_status_blocks_failed_trial() -> None:
    result = make_failed_result()

    assert choose_kernel_handoff_status(result) is KernelHandoffStatus.BLOCKED


def test_build_kernel_handoff_packet_for_successful_trial() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_kernel_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.packet_id == "trial-001:kernel-handoff"
    assert handoff.status is KernelHandoffStatus.READY_FOR_REVIEW
    assert handoff.trial_id == "trial-001"
    assert handoff.evidence_packet_id == "trial-001:evidence-packet"
    assert handoff.belief_update.function_id == "causal-bottleneck-v1"
    assert handoff.belief_update.confidence_delta > 0
    assert handoff.belief_update.claim_boundary == kernel_claim_boundary()
    assert handoff.skill_candidate.allowed_for_reuse
    assert handoff.skill_candidate.blocking_reason is None
    assert validate_kernel_handoff_packet(handoff) == ()


def test_kernel_handoff_preserves_future_rules_and_uncertainty_memory() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_kernel_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    memory_kinds = {memory.kind for memory in handoff.memory_updates}

    assert KernelMemoryUpdateKind.TRANSFER_SUPPORT in memory_kinds
    assert KernelMemoryUpdateKind.FUTURE_PLANNING_RULE in memory_kinds
    assert KernelMemoryUpdateKind.UNCERTAINTY_NOTE in memory_kinds
    assert all(memory.evidence_refs for memory in handoff.memory_updates)


def test_build_kernel_memory_updates_marks_failed_trial_for_quarantine() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)
    refs = (
        evidence_packet.packet_id,
        evidence_packet.manifest_sha256_digest,
    )

    updates = build_kernel_memory_updates(result=result, evidence_refs=refs)

    assert updates[0].kind is KernelMemoryUpdateKind.TRANSFER_FAILURE
    assert updates[0].quarantine_recommended
    assert any(memory.quarantine_recommended for memory in updates)


def test_build_kernel_handoff_packet_blocks_failed_trial_reuse() -> None:
    result = make_failed_result()
    evidence_packet = build_trial_evidence_packet(result)

    handoff = build_kernel_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    assert handoff.status is KernelHandoffStatus.BLOCKED
    assert handoff.skill_candidate.allowed_for_reuse is False
    assert handoff.skill_candidate.blocking_reason is not None
    assert "Do not strengthen Kernel belief" in handoff.required_kernel_actions[0]
    assert validate_kernel_handoff_packet(handoff) == ()


def test_validate_kernel_handoff_packet_detects_bad_claim_boundary() -> None:
    result = run_transfer_trial(make_trial_input())
    evidence_packet = build_trial_evidence_packet(result)
    handoff = build_kernel_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    invalid_handoff = type(handoff)(
        packet_id=handoff.packet_id,
        status=handoff.status,
        trial_id=handoff.trial_id,
        evidence_packet_id=handoff.evidence_packet_id,
        belief_update=handoff.belief_update,
        memory_updates=handoff.memory_updates,
        skill_candidate=handoff.skill_candidate,
        required_kernel_actions=handoff.required_kernel_actions,
        claim_boundary="bad boundary",
    )

    errors = validate_kernel_handoff_packet(invalid_handoff)

    assert "claim_boundary must match fixed Kernel boundary" in errors
