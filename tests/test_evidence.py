from __future__ import annotations

from ix_function.evidence import (
    EvidenceArtifact,
    EvidenceArtifactType,
    EvidencePacket,
    build_evidence_packet,
    canonical_json,
    create_evidence_artifact,
    packet_manifest_payload,
    sha256_for_json,
    validate_evidence_artifact,
    validate_evidence_packet,
)


def test_canonical_json_is_deterministic_for_key_order() -> None:
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_for_json(left) == sha256_for_json(right)


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
