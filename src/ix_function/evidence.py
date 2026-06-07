"""Deterministic evidence packets for IX-Function.

IX-Function evidence must be replayable and reviewable. This module creates
stable JSON payloads and SHA-256 digests for trial artifacts without claiming
that a digest proves truth, AGI, or operational readiness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from typing import Any, TypeAlias, cast

from ix_function.trial import TransferTrialResult

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class EvidenceArtifactType(StrEnum):
    """Artifact type names used in IX-Function evidence packets."""

    ANTI_THEATER_GATE = "anti_theater_gate"
    FALSIFICATION_LEDGER = "falsification_ledger"
    LEARNING_UPDATE = "learning_update"
    MAPPING = "mapping"
    NEGATIVE_CONTROL_SUITE = "negative_control_suite"
    PREDICTION_READINESS = "prediction_readiness"
    REALITY_DELTA = "reality_delta"
    SOURCE_EVIDENCE = "source_evidence"
    TRIAL_SUMMARY = "trial_summary"
    UNCERTAINTY_GATE = "uncertainty_gate"
    UNCERTAINTY_LEDGER = "uncertainty_ledger"


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    """A single deterministic evidence artifact with a stable digest."""

    artifact_id: str
    artifact_type: EvidenceArtifactType
    payload: JsonObject
    sha256_digest: str


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """A set of evidence artifacts with a manifest digest."""

    packet_id: str
    artifacts: tuple[EvidenceArtifact, ...]
    manifest_sha256_digest: str

    def artifact_index(self) -> dict[str, EvidenceArtifact]:
        """Return artifacts keyed by artifact identifier."""

        return {artifact.artifact_id: artifact for artifact in self.artifacts}


def canonical_json(value: JsonValue) -> str:
    """Return deterministic JSON for hashing and review."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_for_json(value: JsonValue) -> str:
    """Return SHA-256 digest for deterministic JSON content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def to_json_value(value: object) -> JsonValue:
    """Convert supported IX-Function values into deterministic JSON values."""

    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        converted: JsonObject = {}
        for key, item in value.items():
            converted[str(key)] = to_json_value(item)
        return converted
    if is_dataclass(value) and not isinstance(value, type):
        raw = asdict(cast(Any, value))
        return to_json_value(raw)

    raise TypeError(f"Unsupported evidence JSON value: {type(value).__name__}")


def artifact_payload_for_object(value: object) -> JsonObject:
    """Convert a dataclass-like object into a JSON object payload."""

    payload = to_json_value(value)
    if not isinstance(payload, dict):
        raise TypeError("Evidence artifact payload must serialize to a JSON object")
    return payload


def create_evidence_artifact(
    *,
    artifact_id: str,
    artifact_type: EvidenceArtifactType,
    payload: JsonObject,
) -> EvidenceArtifact:
    """Create a deterministic evidence artifact."""

    digest_payload: JsonObject = {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type.value,
        "payload": payload,
    }
    return EvidenceArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        payload=payload,
        sha256_digest=sha256_for_json(digest_payload),
    )


def create_evidence_artifact_from_object(
    *,
    artifact_id: str,
    artifact_type: EvidenceArtifactType,
    value: object,
) -> EvidenceArtifact:
    """Create an evidence artifact from a supported dataclass-like object."""

    return create_evidence_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        payload=artifact_payload_for_object(value),
    )


def build_evidence_packet(
    *,
    packet_id: str,
    artifacts: tuple[EvidenceArtifact, ...],
) -> EvidencePacket:
    """Build a packet manifest around deterministic evidence artifacts."""

    manifest_payload = packet_manifest_payload(packet_id, artifacts)
    return EvidencePacket(
        packet_id=packet_id,
        artifacts=artifacts,
        manifest_sha256_digest=sha256_for_json(manifest_payload),
    )


def packet_manifest_payload(
    packet_id: str,
    artifacts: tuple[EvidenceArtifact, ...],
) -> JsonObject:
    """Return deterministic manifest payload for an evidence packet."""

    return {
        "packet_id": packet_id,
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type.value,
                "sha256_digest": artifact.sha256_digest,
            }
            for artifact in artifacts
        ],
    }


def trial_summary_payload(result: TransferTrialResult) -> JsonObject:
    """Return a compact JSON payload summarizing one transfer trial result."""

    return {
        "trial_id": result.trial_id,
        "status": result.status.value,
        "permits_bounded_evidence": result.permits_bounded_evidence(),
        "source_evidence": {
            "evidence_id": result.source_evidence.evidence_id,
            "status": result.source_evidence.status.value,
            "confidence_delta": result.source_evidence.confidence_delta,
            "blocking_errors": list(result.source_evidence.blocking_errors),
        },
        "mapping": {
            "function_id": result.mapping.function_id,
            "target_domain_id": result.mapping.target_domain_id,
            "quality": result.mapping.quality.value,
            "coverage_score": result.mapping.coverage_score,
            "ambiguity_score": result.mapping.ambiguity_score,
            "warnings": list(result.mapping.warnings),
        },
        "prediction_readiness": {
            "prediction_id": result.prediction_readiness.prediction_id,
            "status": result.prediction_readiness.status.value,
            "blocking_errors": list(result.prediction_readiness.blocking_errors),
        },
        "reality_delta": {
            "report_id": result.reality_delta.report_id,
            "function_id": result.reality_delta.function_id,
            "status": result.reality_delta.status.value,
            "mean_score": result.reality_delta.mean_score,
            "confidence_delta": result.reality_delta.confidence_delta,
            "blocking_errors": list(result.reality_delta.blocking_errors),
        },
        "learning_update": {
            "update_id": result.learning_update.update_id,
            "disposition": result.learning_update.disposition.value,
            "revised_confidence": result.learning_update.revised_confidence,
            "confidence_band": result.learning_update.confidence_band.value,
            "future_planning_rules": list(
                result.learning_update.future_planning_rules
            ),
            "blocking_errors": list(result.learning_update.blocking_errors),
        },
        "uncertainty_gate": {
            "ledger_id": result.uncertainty_gate.ledger_id,
            "allowed": result.uncertainty_gate.allowed,
            "maximum_severity": result.uncertainty_gate.maximum_severity.value,
            "blocking_ids": list(result.uncertainty_gate.blocking_ids),
            "reason": result.uncertainty_gate.reason,
        },
        "falsification": {
            "ledger_id": result.falsification_ledger.ledger_id,
            "verdict": result.falsification_ledger.verdict.value,
            "failed_count": len(result.falsification_ledger.failed_evaluations()),
            "kill_count": len(result.falsification_ledger.kill_evaluations()),
        },
        "anti_theater": {
            "suite_id": result.anti_theater_gate.suite_id,
            "allowed": result.anti_theater_gate.allowed,
            "failed_control_ids": list(result.anti_theater_gate.failed_control_ids),
            "reason": result.anti_theater_gate.reason,
        },
        "blocking_errors": list(result.blocking_errors),
        "required_actions": list(result.required_actions),
        "claim_boundary": (
            "This packet is bounded transfer evidence only. It is not AGI proof, "
            "deployment authorization, certification, or independent validation."
        ),
    }


def build_trial_evidence_packet(result: TransferTrialResult) -> EvidencePacket:
    """Build a deterministic evidence packet for a transfer trial."""

    artifacts = (
        create_evidence_artifact(
            artifact_id=f"{result.trial_id}:trial-summary",
            artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
            payload=trial_summary_payload(result),
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:source-evidence",
            artifact_type=EvidenceArtifactType.SOURCE_EVIDENCE,
            value=result.source_evidence,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:mapping",
            artifact_type=EvidenceArtifactType.MAPPING,
            value=result.mapping,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:prediction-readiness",
            artifact_type=EvidenceArtifactType.PREDICTION_READINESS,
            value=result.prediction_readiness,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:reality-delta",
            artifact_type=EvidenceArtifactType.REALITY_DELTA,
            value=result.reality_delta,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:learning-update",
            artifact_type=EvidenceArtifactType.LEARNING_UPDATE,
            value=result.learning_update,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:uncertainty-ledger",
            artifact_type=EvidenceArtifactType.UNCERTAINTY_LEDGER,
            value=result.uncertainty_ledger,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:uncertainty-gate",
            artifact_type=EvidenceArtifactType.UNCERTAINTY_GATE,
            value=result.uncertainty_gate,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:falsification-ledger",
            artifact_type=EvidenceArtifactType.FALSIFICATION_LEDGER,
            value=result.falsification_ledger,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:negative-control-suite",
            artifact_type=EvidenceArtifactType.NEGATIVE_CONTROL_SUITE,
            value=result.negative_control_suite,
        ),
        create_evidence_artifact_from_object(
            artifact_id=f"{result.trial_id}:anti-theater-gate",
            artifact_type=EvidenceArtifactType.ANTI_THEATER_GATE,
            value=result.anti_theater_gate,
        ),
    )
    return build_evidence_packet(
        packet_id=f"{result.trial_id}:evidence-packet",
        artifacts=artifacts,
    )


def validate_evidence_artifact(artifact: EvidenceArtifact) -> tuple[str, ...]:
    """Return validation errors for one evidence artifact."""

    errors: list[str] = []
    if not artifact.artifact_id.strip():
        errors.append("artifact_id must not be empty")
    if not artifact.payload:
        errors.append(f"payload must not be empty for {artifact.artifact_id!r}")
    if not artifact.sha256_digest.strip():
        errors.append(f"sha256_digest must not be empty for {artifact.artifact_id!r}")

    expected = sha256_for_json(
        {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type.value,
            "payload": artifact.payload,
        }
    )
    if artifact.sha256_digest != expected:
        errors.append(f"sha256_digest mismatch for {artifact.artifact_id!r}")

    return tuple(errors)


def validate_evidence_packet(packet: EvidencePacket) -> tuple[str, ...]:
    """Return validation errors for an evidence packet."""

    errors: list[str] = []
    if not packet.packet_id.strip():
        errors.append("packet_id must not be empty")
    if not packet.artifacts:
        errors.append("at least one evidence artifact is required")

    artifact_ids = [artifact.artifact_id for artifact in packet.artifacts]
    if len(set(artifact_ids)) != len(artifact_ids):
        errors.append("evidence artifacts must use unique artifact_id values")

    for artifact in packet.artifacts:
        errors.extend(validate_evidence_artifact(artifact))

    expected_manifest_digest = sha256_for_json(
        packet_manifest_payload(packet.packet_id, packet.artifacts)
    )
    if packet.manifest_sha256_digest != expected_manifest_digest:
        errors.append("manifest_sha256_digest mismatch")

    return tuple(errors)
