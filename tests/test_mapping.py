from __future__ import annotations

from ix_function.causal_function import (
    CausalFamily,
    CausalFunction,
    CausalMechanism,
    CausalSlotRole,
    CausalVariableSlot,
)
from ix_function.domain import (
    DomainKind,
    DomainProfile,
    Observable,
    ObservableRole,
    ValueKind,
)
from ix_function.mapping import (
    MappingQuality,
    SlotMapping,
    TransferMapping,
    build_slot_candidates,
    normalized_tokens,
    observable_semantic_tokens,
    propose_transfer_mapping,
    validate_transfer_mapping,
)


def make_bottleneck_function() -> CausalFunction:
    return CausalFunction(
        function_id="causal-bottleneck-v1",
        name="Downstream Bottleneck Limit",
        family=CausalFamily.BOTTLENECK,
        summary="Upstream capacity cannot beat a downstream constraint.",
        variable_slots=(
            CausalVariableSlot(
                slot_id="upstream_capacity_intervention",
                role=CausalSlotRole.INTERVENTION,
                description="A change that increases upstream capacity.",
                semantic_tags=("capacity", "intervention", "upstream"),
                compatible_observable_roles=(ObservableRole.INTERVENTION,),
            ),
            CausalVariableSlot(
                slot_id="downstream_constraint",
                role=CausalSlotRole.CONSTRAINT,
                description="A limiting downstream stage.",
                semantic_tags=("constraint", "downstream", "limit"),
                compatible_observable_roles=(ObservableRole.CONSTRAINT,),
            ),
            CausalVariableSlot(
                slot_id="final_output",
                role=CausalSlotRole.OUTPUT,
                description="The final measured output.",
                semantic_tags=("output", "throughput", "completion"),
                compatible_observable_roles=(ObservableRole.OUTPUT,),
            ),
        ),
        mechanisms=(
            CausalMechanism(
                mechanism_id="bottleneck-no-output-gain",
                family=CausalFamily.BOTTLENECK,
                premise="The downstream stage is saturated.",
                expected_effect="More upstream capacity produces limited output gain.",
                assumptions=("Downstream limit remains unchanged.",),
            ),
        ),
        prior_confidence=0.62,
        uncertainty_notes=("Hidden parallel paths may invalidate transfer.",),
    )


def make_ci_domain() -> DomainProfile:
    return DomainProfile(
        domain_id="ci-pipeline",
        name="CI Pipeline",
        kind=DomainKind.COMPUTING,
        summary="A pipeline with workers, slow stages, and completion time.",
        observables=(
            Observable(
                name="Worker Count",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.INTEGER,
                description="Upstream capacity intervention for pipeline workers.",
            ),
            Observable(
                name="Slowest Downstream Stage Time",
                role=ObservableRole.CONSTRAINT,
                value_kind=ValueKind.REAL,
                description="Downstream limit or constraint in seconds.",
                unit="seconds",
            ),
            Observable(
                name="Completion Throughput Output",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Final output and completion throughput after tests.",
                unit="seconds",
            ),
        ),
    )


def test_normalized_tokens_extracts_simple_terms() -> None:
    assert normalized_tokens("Downstream-limit / CI_v2") == (
        "downstream",
        "limit",
        "ci",
        "v2",
    )


def test_observable_semantic_tokens_uses_declared_fields() -> None:
    observable = make_ci_domain().observables[1]

    tokens = observable_semantic_tokens(observable)

    assert "downstream" in tokens
    assert "constraint" in tokens
    assert "seconds" in tokens


def test_build_slot_candidates_prefers_role_and_semantic_overlap() -> None:
    function = make_bottleneck_function()
    slot = function.variable_slots[1]

    candidates = build_slot_candidates(slot, make_ci_domain())

    assert candidates[0].observable_name == "Slowest Downstream Stage Time"
    assert candidates[0].role_compatible is True
    assert candidates[0].semantic_overlap == ("constraint", "downstream", "limit")
    assert candidates[0].score == 1.0


def test_propose_transfer_mapping_creates_complete_mapping() -> None:
    mapping = propose_transfer_mapping(make_bottleneck_function(), make_ci_domain())

    assert mapping.quality is MappingQuality.COMPLETE
    assert mapping.coverage_score == 1.0
    assert mapping.ambiguity_score == 0.0
    assert mapping.warnings == ()
    assert mapping.is_usable_for_prediction()
    assert validate_transfer_mapping(mapping) == ()
    assert mapping.mapping_index()["final_output"].observable_name == (
        "Completion Throughput Output"
    )


def test_propose_transfer_mapping_preserves_role_only_uncertainty() -> None:
    sparse_domain = DomainProfile(
        domain_id="sparse-target",
        name="Sparse Target",
        kind=DomainKind.SYNTHETIC,
        summary="A target with roles but no helpful semantic labels.",
        observables=(
            Observable(
                name="A",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.REAL,
                description="Generic value.",
            ),
            Observable(
                name="B",
                role=ObservableRole.CONSTRAINT,
                value_kind=ValueKind.REAL,
                description="Generic value.",
            ),
            Observable(
                name="C",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Generic value.",
            ),
        ),
    )

    mapping = propose_transfer_mapping(make_bottleneck_function(), sparse_domain)

    assert mapping.quality is MappingQuality.COMPLETE
    assert all(
        "mapping is role-compatible but has no semantic tag overlap"
        in slot_mapping.uncertainty_notes
        for slot_mapping in mapping.slot_mappings
    )


def test_propose_transfer_mapping_marks_ambiguous_near_ties() -> None:
    ambiguous_domain = DomainProfile(
        domain_id="ambiguous-target",
        name="Ambiguous Target",
        kind=DomainKind.COMPUTING,
        summary="A target with two plausible output observables.",
        observables=(
            *make_ci_domain().observables,
            Observable(
                name="Alternative Completion Output",
                role=ObservableRole.OUTPUT,
                value_kind=ValueKind.REAL,
                description="Final output and completion throughput alternative.",
            ),
        ),
    )

    mapping = propose_transfer_mapping(make_bottleneck_function(), ambiguous_domain)

    assert mapping.quality is MappingQuality.AMBIGUOUS
    assert mapping.ambiguity_score > 0.0
    assert any(
        "nearest alternative" in note
        for slot_mapping in mapping.slot_mappings
        for note in slot_mapping.uncertainty_notes
    )


def test_propose_transfer_mapping_blocks_insufficient_coverage() -> None:
    insufficient_domain = DomainProfile(
        domain_id="insufficient-target",
        name="Insufficient Target",
        kind=DomainKind.SECURITY,
        summary="A target that lacks output and constraint observables.",
        observables=(
            Observable(
                name="Worker Count",
                role=ObservableRole.INTERVENTION,
                value_kind=ValueKind.INTEGER,
                description="Upstream capacity intervention.",
            ),
        ),
    )

    mapping = propose_transfer_mapping(
        make_bottleneck_function(),
        insufficient_domain,
        min_slot_score=0.65,
    )

    assert mapping.quality is MappingQuality.INSUFFICIENT
    assert mapping.coverage_score < 1.0
    assert not mapping.is_usable_for_prediction()
    assert any("has no target observable" in warning for warning in mapping.warnings)


def test_validate_transfer_mapping_rejects_duplicate_target_observable() -> None:
    mapping = TransferMapping(
        function_id="causal-bottleneck-v1",
        target_domain_id="bad-target",
        slot_mappings=(
            SlotMapping(
                slot_id="slot-a",
                observable_name="Shared Observable",
                score=0.7,
                uncertainty_notes=(),
            ),
            SlotMapping(
                slot_id="slot-b",
                observable_name="shared   observable",
                score=0.8,
                uncertainty_notes=(),
            ),
        ),
        quality=MappingQuality.INSUFFICIENT,
        coverage_score=1.0,
        ambiguity_score=0.0,
        warnings=("duplicate observable for test",),
    )

    errors = validate_transfer_mapping(mapping)

    assert (
        "target observable 'shared_observable' must not be mapped more than once"
        in errors
    )
