"""IX-BlackFox-WorldTwin handoff packets for IX-Function.

The WorldTwin handoff converts an IX-Function transfer trial into a scenario,
prediction, outcome-delta, and adaptation package. The handoff is designed for
simulation/replay review, not for proving truth or AGI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import EvidencePacket
from ix_function.prediction import PredictionDirection
from ix_function.reality_delta import OutcomeMatch
from ix_function.trial import TransferTrialResult, TrialStatus


class WorldTwinHandoffStatus(StrEnum):
    """Status of a WorldTwin handoff package."""

    READY_FOR_SCENARIO_REPLAY = "ready_for_scenario_replay"
    READY_FOR_FAILURE_REPLAY = "ready_for_failure_replay"
    BLOCKED = "blocked"


class WorldTwinAdaptationAction(StrEnum):
    """Adaptation action suggested by IX-Function evidence."""

    EXPAND_SCENARIO = "expand_scenario"
    PRESERVE_AS_SUPPORTED_CASE = "preserve_as_supported_case"
    QUARANTINE_SCENARIO = "quarantine_scenario"
    RETEST_WITH_NARROWER_RANGE = "retest_with_narrower_range"


@dataclass(frozen=True, slots=True)
class WorldTwinVariableBinding:
    """Binding from a causal slot to a WorldTwin scenario variable."""

    slot_id: str
    target_observable_name: str
    mapping_score: float
    uncertainty_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorldTwinPredictionBinding:
    """Prediction binding for WorldTwin replay."""

    observable_name: str
    direction: PredictionDirection
    expected_min: float | None
    expected_max: float | None
    tolerance: float | None
    rationale: str


@dataclass(frozen=True, slots=True)
class WorldTwinOutcomeDeltaBinding:
    """Observed outcome delta for WorldTwin reality comparison."""

    observable_name: str
    outcome_match: OutcomeMatch
    baseline_value: str
    observed_value: str
    numeric_delta: float | None
    score: float
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorldTwinScenarioPacket:
    """Scenario packet for WorldTwin replay."""

    scenario_id: str
    trial_id: str
    function_id: str
    source_domain_id: str
    target_domain_id: str
    transfer_purpose: str
    variable_bindings: tuple[WorldTwinVariableBinding, ...]
    prediction_bindings: tuple[WorldTwinPredictionBinding, ...]
    outcome_delta_bindings: tuple[WorldTwinOutcomeDeltaBinding, ...]
    assumptions: tuple[str, ...]
    uncertainty_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorldTwinAdaptationPacket:
    """Adaptation packet derived from a scenario replay result."""

    adaptation_id: str
    action: WorldTwinAdaptationAction
    confidence_delta: float
    recommended_next_scenarios: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class WorldTwinHandoffPacket:
    """Complete IX-BlackFox-WorldTwin handoff packet."""

    packet_id: str
    status: WorldTwinHandoffStatus
    scenario_packet: WorldTwinScenarioPacket
    adaptation_packet: WorldTwinAdaptationPacket
    required_worldtwin_actions: tuple[str, ...]
    claim_boundary: str


def build_worldtwin_handoff_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> WorldTwinHandoffPacket:
    """Build a WorldTwin replay handoff from a completed transfer trial."""

    status = choose_worldtwin_handoff_status(result)
    scenario_packet = build_worldtwin_scenario_packet(
        result=result,
        evidence_packet=evidence_packet,
    )
    adaptation_packet = build_worldtwin_adaptation_packet(result)
    return WorldTwinHandoffPacket(
        packet_id=f"{result.trial_id}:worldtwin-handoff",
        status=status,
        scenario_packet=scenario_packet,
        adaptation_packet=adaptation_packet,
        required_worldtwin_actions=required_worldtwin_actions(result, status),
        claim_boundary=worldtwin_claim_boundary(),
    )


def choose_worldtwin_handoff_status(
    result: TransferTrialResult,
) -> WorldTwinHandoffStatus:
    """Choose WorldTwin handoff status from trial outcome."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return WorldTwinHandoffStatus.READY_FOR_SCENARIO_REPLAY
    if result.reality_delta.observable_deltas:
        return WorldTwinHandoffStatus.READY_FOR_FAILURE_REPLAY
    return WorldTwinHandoffStatus.BLOCKED


def build_worldtwin_scenario_packet(
    *,
    result: TransferTrialResult,
    evidence_packet: EvidencePacket,
) -> WorldTwinScenarioPacket:
    """Build a WorldTwin scenario packet from trial artifacts."""

    return WorldTwinScenarioPacket(
        scenario_id=f"{result.trial_id}:worldtwin-scenario",
        trial_id=result.trial_id,
        function_id=result.learning_update.function_id,
        source_domain_id=result.source_evidence.source_domain_id,
        target_domain_id=result.mapping.target_domain_id,
        transfer_purpose=(
            "Replay cross-domain causal transfer and compare committed "
            "prediction against observed target reality."
        ),
        variable_bindings=tuple(
            WorldTwinVariableBinding(
                slot_id=mapping.slot_id,
                target_observable_name=mapping.observable_name,
                mapping_score=mapping.score,
                uncertainty_notes=mapping.uncertainty_notes,
            )
            for mapping in result.mapping.slot_mappings
        ),
        prediction_bindings=tuple(
            WorldTwinPredictionBinding(
                observable_name=delta.observable_name,
                direction=delta.predicted_direction,
                expected_min=_prediction_expected_min(result, delta.observable_name),
                expected_max=_prediction_expected_max(result, delta.observable_name),
                tolerance=_prediction_tolerance(result, delta.observable_name),
                rationale=_prediction_rationale(result, delta.observable_name),
            )
            for delta in result.reality_delta.observable_deltas
        ),
        outcome_delta_bindings=tuple(
            WorldTwinOutcomeDeltaBinding(
                observable_name=delta.observable_name,
                outcome_match=delta.outcome_match,
                baseline_value=repr(delta.baseline_value),
                observed_value=repr(delta.observed_value),
                numeric_delta=delta.numeric_delta,
                score=delta.score,
                notes=delta.notes,
            )
            for delta in result.reality_delta.observable_deltas
        ),
        assumptions=tuple(
            item.statement
            for item in result.uncertainty_ledger.items
            if item.kind.value == "assumption"
        ),
        uncertainty_refs=tuple(
            item.uncertainty_id for item in result.uncertainty_ledger.open_items()
        ),
        evidence_refs=(
            evidence_packet.packet_id,
            evidence_packet.manifest_sha256_digest,
            result.reality_delta.report_id,
            result.falsification_ledger.ledger_id,
        ),
    )


def build_worldtwin_adaptation_packet(
    result: TransferTrialResult,
) -> WorldTwinAdaptationPacket:
    """Build WorldTwin adaptation guidance from IX-Function gates."""

    if result.status is TrialStatus.BOUNDED_EVIDENCE_ALLOWED:
        return WorldTwinAdaptationPacket(
            adaptation_id=f"{result.trial_id}:worldtwin-adaptation",
            action=WorldTwinAdaptationAction.PRESERVE_AS_SUPPORTED_CASE,
            confidence_delta=result.reality_delta.confidence_delta,
            recommended_next_scenarios=(
                "Replay same causal function against a held-out target domain.",
                "Retest with different baseline values before stronger claims.",
            ),
            blocked_claims=("AGI proof", "deployment authority"),
            rationale=(
                "Scenario produced bounded transfer support while preserving "
                "uncertainty and falsification records."
            ),
        )

    if result.reality_delta.observable_deltas:
        return WorldTwinAdaptationPacket(
            adaptation_id=f"{result.trial_id}:worldtwin-adaptation",
            action=WorldTwinAdaptationAction.RETEST_WITH_NARROWER_RANGE,
            confidence_delta=result.reality_delta.confidence_delta,
            recommended_next_scenarios=(
                "Replay failed or mixed observable deltas with narrower ranges.",
                "Introduce alternate causal explanation before reuse.",
            ),
            blocked_claims=("strong transfer support", "AGI proof"),
            rationale=(
                "Scenario did not permit bounded evidence; WorldTwin should "
                "retain the failed or mixed reality delta for retesting."
            ),
        )

    return WorldTwinAdaptationPacket(
        adaptation_id=f"{result.trial_id}:worldtwin-adaptation",
        action=WorldTwinAdaptationAction.QUARANTINE_SCENARIO,
        confidence_delta=0.0,
        recommended_next_scenarios=(
            "Collect complete baseline, intervention, and outcome records.",
        ),
        blocked_claims=("bounded candidate evidence", "AGI proof"),
        rationale=(
            "Scenario could not be replayed because the trial did not contain "
            "scorable outcome deltas."
        ),
    )


def required_worldtwin_actions(
    result: TransferTrialResult,
    status: WorldTwinHandoffStatus,
) -> tuple[str, ...]:
    """Return required WorldTwin-side actions."""

    if status is WorldTwinHandoffStatus.READY_FOR_SCENARIO_REPLAY:
        return (
            "Replay scenario as bounded transfer evidence only.",
            "Compare future WorldTwin predictions against fresh outcomes.",
            "Carry uncertainty references into the scenario record.",
            "Do not treat scenario support as AGI proof.",
        )

    if status is WorldTwinHandoffStatus.READY_FOR_FAILURE_REPLAY:
        return (
            "Replay failed or mixed scenario as negative evidence.",
            "Do not increase confidence from this scenario.",
            "Use outcome deltas to construct a narrower retest.",
            *result.required_actions,
        )

    return (
        "Block scenario replay until complete measurable records exist.",
        "Do not promote the transfer claim.",
        *result.required_actions,
    )


def worldtwin_claim_boundary() -> str:
    """Return fixed WorldTwin claim boundary."""

    return (
        "IX-Function WorldTwin handoff is replay evidence for bounded causal "
        "transfer. It is not AGI proof, not a validated world model, and not "
        "deployment authority."
    )


def validate_worldtwin_handoff_packet(
    packet: WorldTwinHandoffPacket,
) -> tuple[str, ...]:
    """Return validation errors for a WorldTwin handoff packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.scenario_packet.scenario_id.strip():
        errors.append("scenario_id must not be empty")
    if not packet.scenario_packet.trial_id.strip():
        errors.append("trial_id must not be empty")
    if not packet.scenario_packet.function_id.strip():
        errors.append("function_id must not be empty")
    if not packet.scenario_packet.source_domain_id.strip():
        errors.append("source_domain_id must not be empty")
    if not packet.scenario_packet.target_domain_id.strip():
        errors.append("target_domain_id must not be empty")
    if not packet.scenario_packet.variable_bindings:
        errors.append("at least one variable binding is required")
    if not packet.scenario_packet.evidence_refs:
        errors.append("scenario evidence_refs must not be empty")
    if not packet.adaptation_packet.adaptation_id.strip():
        errors.append("adaptation_id must not be empty")
    if not packet.adaptation_packet.blocked_claims:
        errors.append("adaptation blocked_claims must not be empty")
    if not packet.required_worldtwin_actions:
        errors.append("required_worldtwin_actions must not be empty")
    if packet.claim_boundary != worldtwin_claim_boundary():
        errors.append("claim_boundary must match fixed WorldTwin boundary")

    return tuple(errors)


def _prediction_expected_min(
    result: TransferTrialResult,
    observable_name: str,
) -> float | None:
    prediction = result.reality_delta.prediction_id
    _ = prediction
    for delta_name, expected_min, _, _, _ in _prediction_lookup(result):
        if delta_name == observable_name:
            return expected_min
    return None


def _prediction_expected_max(
    result: TransferTrialResult,
    observable_name: str,
) -> float | None:
    for delta_name, _, expected_max, _, _ in _prediction_lookup(result):
        if delta_name == observable_name:
            return expected_max
    return None


def _prediction_tolerance(
    result: TransferTrialResult,
    observable_name: str,
) -> float | None:
    for delta_name, _, _, tolerance, _ in _prediction_lookup(result):
        if delta_name == observable_name:
            return tolerance
    return None


def _prediction_rationale(
    result: TransferTrialResult,
    observable_name: str,
) -> str:
    for delta_name, _, _, _, rationale in _prediction_lookup(result):
        if delta_name == observable_name:
            return rationale
    return "Prediction rationale was not available in the scored delta."


def _prediction_lookup(
    result: TransferTrialResult,
) -> tuple[tuple[str, float | None, float | None, float | None, str], ...]:
    """Return prediction fields recoverable from scored observable deltas.

    The TransferTrialResult stores scored deltas, not the full original
    prediction object. This lookup intentionally preserves only what can be
    reconstructed from the delta names and the trial evidence chain.
    """

    return tuple(
        (
            delta.observable_name,
            None,
            None,
            None,
            (
                "Prediction was committed before outcome and scored in "
                f"{result.reality_delta.report_id}."
            ),
        )
        for delta in result.reality_delta.observable_deltas
    )
