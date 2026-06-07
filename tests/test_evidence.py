from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ix_function.evidence import (
    EvidenceArtifact,
    EvidenceArtifactType,
    EvidencePacket,
    artifact_payload_for_object,
    build_evidence_packet,
    build_trial_evidence_packet,
    canonical_json,
    create_evidence_artifact,
    create_evidence_artifact_from_object,
    packet_manifest_payload,
    sha256_for_json,
    to_json_value,
    validate_evidence_artifact,
    validate_evidence_packet,
)
from ix_function.trial import run_transfer_trial
from tests.fixtures import make_trial_input


class FixtureState(StrEnum):
    READY = "ready"


@dataclass(frozen=True, slots=True)
class FixtureArtifact:
    artifact_id: str
    state: FixtureState
    notes: tuple[str, ...]


def test_canonical_json_is_deterministic_for_key_order() -> None:
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_for_json(left) == sha256_for_json(right)


def test_to_json_value_serializes_dataclasses_enums_and_tuples() -> None:
    fixture = FixtureArtifact(
        artifact_id="fixture-001",
        state=FixtureState.READY,
        notes=("bounded evidence only",),
    )

    assert to_json_value(fixture) == {
        "artifact_id": "fixture-001",
        "state": "ready",
        "notes": ["bounded evidence only"],
    }
    assert artifact_payload_for_object(fixture)["state"] == "ready"


def test_create_evidence_artifact_hashes_identity_type_and_payload() -> None:
    artifact = create_evidence_artifact(
        artifact_id="trial-001:trial-summary",
        artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
        payload={
            "trial_id": "trial-001",
            "status": "bounded_evidence_allowed",
        },
    )

    assert artifact.artifact_id == "trial-001:trial-summary"
    assert artifact.artifact_type is EvidenceArtifactType.TRIAL_SUMMARY
    assert len(artifact.sha256_digest) == 64
    assert validate_evidence_artifact(artifact) == ()


def test_create_evidence_artifact_from_object_uses_serialized_payload() -> None:
    fixture = FixtureArtifact(
        artifact_id="fixture-001",
        state=FixtureState.READY,
        notes=("bounded evidence only",),
    )

    artifact = create_evidence_artifact_from_object(
        artifact_id="fixture-001",
        artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
        value=fixture,
    )

    assert artifact.payload == {
        "artifact_id": "fixture-001",
        "state": "ready",
        "notes": ["bounded evidence only"],
    }
    assert validate_evidence_artifact(artifact) == ()


def test_validate_evidence_artifact_detects_digest_mismatch() -> None:
    artifact = EvidenceArtifact(
        artifact_id="trial-001:trial-summary",
        artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
        payload={"trial_id": "trial-001"},
        sha256_digest="bad-digest",
    )

    errors = validate_evidence_artifact(artifact)

    assert "sha256_digest mismatch for 'trial-001:trial-summary'" in errors


def test_build_evidence_packet_creates_manifest_digest() -> None:
    artifact = create_evidence_artifact(
        artifact_id="trial-001:trial-summary",
        artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
        payload={
            "trial_id": "trial-001",
            "claim_boundary": "bounded evidence only",
        },
    )
    packet = build_evidence_packet(
        packet_id="trial-001:evidence-packet",
        artifacts=(artifact,),
    )

    assert packet.packet_id == "trial-001:evidence-packet"
    assert packet.artifact_index()["trial-001:trial-summary"] == artifact
    assert len(packet.manifest_sha256_digest) == 64
    assert validate_evidence_packet(packet) == ()
    assert packet_manifest_payload(packet.packet_id, packet.artifacts) == {
        "packet_id": "trial-001:evidence-packet",
        "artifacts": [
            {
                "artifact_id": "trial-001:trial-summary",
                "artifact_type": "trial_summary",
                "sha256_digest": artifact.sha256_digest,
            },
        ],
    }


def test_build_trial_evidence_packet_serializes_real_trial_result() -> None:
    result = run_transfer_trial(make_trial_input())

    packet = build_trial_evidence_packet(result)
    artifact_index = packet.artifact_index()

    assert packet.packet_id == "trial-001:evidence-packet"
    assert len(packet.artifacts) == 11
    assert validate_evidence_packet(packet) == ()
    assert artifact_index["trial-001:trial-summary"].payload["status"] == (
        "bounded_evidence_allowed"
    )
    assert artifact_index["trial-001:reality-delta"].payload["status"] == "supported"
    assert artifact_index["trial-001:learning-update"].payload[
        "future_planning_rules"
    ]
    assert artifact_index["trial-001:trial-summary"].payload["claim_boundary"] == (
        "This packet is bounded transfer evidence only. It is not AGI proof, "
        "deployment authorization, certification, or independent validation."
    )


def test_validate_evidence_packet_detects_duplicate_artifacts() -> None:
    artifact = create_evidence_artifact(
        artifact_id="duplicate",
        artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
        payload={"trial_id": "trial-001"},
    )
    packet = build_evidence_packet(
        packet_id="trial-001:evidence-packet",
        artifacts=(artifact, artifact),
    )

    errors = validate_evidence_packet(packet)

    assert "evidence artifacts must use unique artifact_id values" in errors


def test_validate_evidence_packet_detects_manifest_mismatch() -> None:
    artifact = create_evidence_artifact(
        artifact_id="trial-001:trial-summary",
        artifact_type=EvidenceArtifactType.TRIAL_SUMMARY,
        payload={"trial_id": "trial-001"},
    )
    packet = EvidencePacket(
        packet_id="trial-001:evidence-packet",
        artifacts=(artifact,),
        manifest_sha256_digest="bad-manifest",
    )

    errors = validate_evidence_packet(packet)

    assert "manifest_sha256_digest mismatch" in errors
