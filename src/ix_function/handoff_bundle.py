"""Integrated donor handoff bundle for IX-Function.

The bundle gathers all donor-facing handoffs into one review object:

- IX-CognitionKernel
- IX-BlackFox-WorldTwin
- IX-IntentRealityLoop
- IX-BlackFox
- IX-Autonomy-Assurance-Case-Runtime
- IX declarative contract layer

This bundle is a review and validation artifact. It is not AGI proof, not
deployment permission, and not automatic downstream authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.assurance_handoff import (
    AssuranceHandoffPacket,
    build_assurance_handoff_packet,
    validate_assurance_handoff_packet,
)
from ix_function.blackfox_handoff import (
    BlackFoxHandoffPacket,
    build_blackfox_handoff_packet,
    validate_blackfox_handoff_packet,
)
from ix_function.evidence import EvidencePacket, validate_evidence_packet
from ix_function.intent_loop_handoff import (
    IntentRealityLoopHandoffPacket,
    build_intent_loop_handoff_packet,
    validate_intent_loop_handoff_packet,
)
from ix_function.ix_contract_handoff import (
    IXContractHandoffPacket,
    build_ix_contract_handoff_packet,
    validate_ix_contract_handoff_packet,
)
from ix_function.kernel_handoff import (
    KernelHandoffPacket,
    build_kernel_handoff_packet,
    validate_kernel_handoff_packet,
)
from ix_function.trial import TransferTrialResult, TrialStatus
from ix_function.worldtwin_handoff import (
    WorldTwinHandoffPacket,
    build_worldtwin_handoff_packet,
    validate_worldtwin_handoff_packet,
)


class IntegratedHandoffStatus(StrEnum):
    """Status of the integrated donor handoff bundle."""

    READY_FOR_WAVE6_REVIEW = "ready_for_wave6_review"
    READY_FOR_FAILURE_REVIEW = "ready_for_failure_review"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class IntegratedDonorHandoffBundle:
    """Complete donor handoff bundle for one IX-Function transfer trial."""

    bundle_id: str
    status: IntegratedHandoffStatus
    trial_id: str
    evidence_packet_id: str
    kernel_handoff: KernelHandoffPacket
    worldtwin_handoff: WorldTwinHandoffPacket
    intent_loop_handoff: IntentRealityLoopHandoffPacket
    blackfox_handoff: BlackFoxHandoffPacket
    assurance_handoff: AssuranceHandoffPacket
    ix_contract_handoff: IXContractHandoffPacket
    validation_errors: tuple[str, ...]
    required_review_actions: tuple[str, ...]
    claim_boundary: str

    def is_ready_for_wave6_review(self) -> bool:
        """Return whether the integrated bundle can enter bounded Wave 6 review."""

        return (
            self.status is IntegratedHandoffStatus.READY_FOR_WAVE6_REVIEW
            and not self.validation_errors
        )


def build_integrated_handoff_bundle(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> IntegratedDonorHandoffBundle:
    """Build every donor handoff and validate the integrated bundle."""

    kernel_handoff = build_kernel_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    worldtwin_handoff = build_worldtwin_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    intent_loop_handoff = build_intent_loop_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    blackfox_handoff = build_blackfox_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    assurance_handoff = build_assurance_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    ix_contract_handoff = build_ix_contract_handoff_packet(
        result=result,
        evidence_packet=evidence_packet,
    )

    validation_errors = integrated_validation_errors(
        result=result,
        evidence_packet=evidence_packet,
        kernel_handoff=kernel_handoff,
        worldtwin_handoff=worldtwin_handoff,
        intent_loop_handoff=intent_loop_handoff,
        blackfox_handoff=blackfox_handoff,
        assurance_handoff=assurance_handoff,
        ix_contract_handoff=ix_contract_handoff,
    )
    status = choose_integrated_handoff_status(
        result=result,
        validation_errors=validation_errors,
    )

    return IntegratedDonorHandoffBundle(
        bundle_id=f"{result.trial_id}:integrated-donor-handoff",
        status=status,
        trial_id=result.trial_id,
        evidence_packet_id=evidence_packet.packet_id,
        kernel_handoff=kernel_handoff,
        worldtwin_handoff=worldtwin_handoff,
        intent_loop_handoff=intent_loop_handoff,
        blackfox_handoff=blackfox_handoff,
        assurance_handoff=assurance_handoff,
        ix_contract_handoff=ix_contract_handoff,
        validation_errors=validation_errors,
        required_review_actions=required_integrated_review_actions(
            result=result,
            status=status,
            validation_errors=validation_errors,
        ),
        claim_boundary=integrated_handoff_claim_boundary(),
    )


def integrated_validation_errors(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
    kernel_handoff: KernelHandoffPacket,
    worldtwin_handoff: WorldTwinHandoffPacket,
    intent_loop_handoff: IntentRealityLoopHandoffPacket,
    blackfox_handoff: BlackFoxHandoffPacket,
    assurance_handoff: AssuranceHandoffPacket,
    ix_contract_handoff: IXContractHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors across all donor handoffs."""

    errors: list[str] = []
    errors.extend(
        _prefix_errors("evidence", validate_evidence_packet(evidence_packet))
    )
    errors.extend(
        _prefix_errors("kernel", validate_kernel_handoff_packet(kernel_handoff))
    )
    errors.extend(
        _prefix_errors(
            "worldtwin",
            validate_worldtwin_handoff_packet(worldtwin_handoff),
        )
    )
    errors.extend(
        _prefix_errors(
            "intent_loop",
            validate_intent_loop_handoff_packet(intent_loop_handoff),
        )
    )
    errors.extend(
        _prefix_errors(
            "blackfox",
            validate_blackfox_handoff_packet(blackfox_handoff),
        )
    )
    errors.extend(
        _prefix_errors(
            "assurance",
            validate_assurance_handoff_packet(assurance_handoff),
        )
    )
    errors.extend(
        _prefix_errors(
            "ix_contract",
            validate_ix_contract_handoff_packet(ix_contract_handoff),
        )
    )

    expected_trial_id = result.trial_id
    trial_links = (
        kernel_handoff.trial_id,
        worldtwin_handoff.scenario_packet.trial_id,
        intent_loop_handoff.intent_binding.trial_id,
        blackfox_handoff.evidence_binding.trial_id,
        assurance_handoff.claim.trial_id,
        ix_contract_handoff.trial_id,
    )
    if any(trial_id != expected_trial_id for trial_id in trial_links):
        errors.append("integrated: all donor handoffs must reference result trial_id")

    expected_evidence_packet_id = evidence_packet.packet_id
    evidence_links = (
        kernel_handoff.evidence_packet_id,
        blackfox_handoff.evidence_binding.evidence_packet_id,
        assurance_handoff.provenance.evidence_packet_id,
    )
    if any(packet_id != expected_evidence_packet_id for packet_id in evidence_links):
        errors.append(
            "integrated: donor handoffs must reference the same evidence packet_id"
        )

    return tuple(errors)


def choose_integrated_handoff_status(
    *,
    result: TransferTrialResult,
    validation_errors: tuple[str, ...],
) -> IntegratedHandoffStatus:
    """Choose integrated bundle status from trial result and validation state."""

    if validation_errors:
        return IntegratedHandoffStatus.BLOCKED
    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return IntegratedHandoffStatus.READY_FOR_WAVE6_REVIEW
    if result.reality_delta.observable_deltas:
        return IntegratedHandoffStatus.READY_FOR_FAILURE_REVIEW
    return IntegratedHandoffStatus.BLOCKED


def required_integrated_review_actions(
    *,
    result: TransferTrialResult,
    status: IntegratedHandoffStatus,
    validation_errors: tuple[str, ...],
) -> tuple[str, ...]:
    """Return required review actions for the integrated donor bundle."""

    if validation_errors:
        return tuple(
            f"Resolve integrated handoff validation error: {error}"
            for error in validation_errors
        )

    if status is IntegratedHandoffStatus.READY_FOR_WAVE6_REVIEW:
        return (
            "Review all donor handoffs as bounded IX-Function evidence.",
            "Bind every review decision to the evidence packet digest.",
            "Preserve uncertainty, falsification, and negative-control records.",
            "Require human authority before any downstream reuse.",
            "Do not represent this bundle as AGI proof.",
        )

    if status is IntegratedHandoffStatus.READY_FOR_FAILURE_REVIEW:
        return (
            "Review all donor handoffs as failure or downgrade evidence.",
            "Block confidence promotion.",
            "Require held-out retesting before future transfer reuse.",
            *result.required_actions,
        )

    return (
        "Block integrated donor handoff until complete evidence exists.",
        "Require human review before retry.",
        *result.required_actions,
    )


def validate_integrated_handoff_bundle(
    bundle: IntegratedDonorHandoffBundle,
) -> tuple[str, ...]:
    """Return validation errors for the integrated donor bundle itself."""

    errors: list[str] = []
    if not bundle.bundle_id.strip():
        errors.append("bundle_id must not be empty")
    if not bundle.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not bundle.evidence_packet_id.strip():
        errors.append("evidence_packet_id must not be empty")
    if not bundle.required_review_actions:
        errors.append("required_review_actions must not be empty")
    if bundle.claim_boundary != integrated_handoff_claim_boundary():
        errors.append("claim_boundary must match fixed integrated boundary")

    trial_links = (
        bundle.kernel_handoff.trial_id,
        bundle.worldtwin_handoff.scenario_packet.trial_id,
        bundle.intent_loop_handoff.intent_binding.trial_id,
        bundle.blackfox_handoff.evidence_binding.trial_id,
        bundle.assurance_handoff.claim.trial_id,
        bundle.ix_contract_handoff.trial_id,
    )
    if any(trial_id != bundle.trial_id for trial_id in trial_links):
        errors.append("all donor handoffs must reference bundle trial_id")

    if (
        bundle.kernel_handoff.evidence_packet_id != bundle.evidence_packet_id
        or bundle.blackfox_handoff.evidence_binding.evidence_packet_id
        != bundle.evidence_packet_id
        or bundle.assurance_handoff.provenance.evidence_packet_id
        != bundle.evidence_packet_id
    ):
        errors.append("core donor handoffs must reference bundle evidence_packet_id")

    if (
        bundle.status is IntegratedHandoffStatus.READY_FOR_WAVE6_REVIEW
        and bundle.validation_errors
    ):
        errors.append("ready bundle must not contain validation_errors")

    return tuple(errors)


def integrated_handoff_claim_boundary() -> str:
    """Return fixed integrated-bundle claim boundary."""

    return (
        "IX-Function integrated donor handoff is bounded review evidence across "
        "the donor repos. It is not AGI proof, not independent validation, not "
        "deployment authority, and not self-approval."
    )


def _prefix_errors(prefix: str, errors: tuple[str, ...]) -> tuple[str, ...]:
    """Prefix validation errors with their donor section."""

    return tuple(f"{prefix}: {error}" for error in errors)
