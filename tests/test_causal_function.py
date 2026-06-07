from __future__ import annotations

from ix_function.causal_function import (
    CausalFamily,
    CausalFunction,
    CausalMechanism,
    CausalSlotRole,
    CausalVariableSlot,
    build_causal_signature,
    validate_causal_function,
    validate_causal_mechanism,
    validate_causal_slot,
)
from ix_function.domain import ObservableRole


def make_bottleneck_function() -> CausalFunction:
    return CausalFunction(
        function_id="causal-bottleneck-v1",
        name="Downstream Bottleneck Limit",
        family=CausalFamily.BOTTLENECK,
        summary=(
            "Increasing upstream capacity does not improve final output when a "
            "downstream constraint is already the limiting factor."
        ),
        variable_slots=(
            CausalVariableSlot(
                slot_id="upstream_capacity_intervention",
                role=CausalSlotRole.INTERVENTION,
                description="A change that increases available upstream capacity.",
                semantic_tags=("capacity", "intervention", "upstream"),
                compatible_observable_roles=(ObservableRole.INTERVENTION,),
            ),
            CausalVariableSlot(
                slot_id="downstream_constraint",
                role=CausalSlotRole.CONSTRAINT,
                description="A limiting stage or channel after the intervention.",
                semantic_tags=("constraint", "downstream", "limit"),
                compatible_observable_roles=(ObservableRole.CONSTRAINT,),
            ),
            CausalVariableSlot(
                slot_id="final_output",
                role=CausalSlotRole.OUTPUT,
                description="The final measured output after the intervention.",
                semantic_tags=("output", "throughput", "completion"),
                compatible_observable_roles=(ObservableRole.OUTPUT,),
            ),
        ),
        mechanisms=(
            CausalMechanism(
                mechanism_id="bottleneck-no-output-gain",
                family=CausalFamily.BOTTLENECK,
                premise="Downstream capacity is already saturated.",
                expected_effect=(
                    "Additional upstream capacity produces little or no output "
                    "improvement."
                ),
                assumptions=(
                    "The downstream constraint remains unchanged.",
                    "The output metric measures the full system, not one segment.",
                ),
            ),
        ),
        prior_confidence=0.61,
        uncertainty_notes=(
            "Transfer may fail when the target system has hidden parallel paths.",
        ),
        learned_from_domain_id="water-flow",
    )


def test_validate_causal_slot_accepts_complete_slot() -> None:
    slot = make_bottleneck_function().variable_slots[0]

    assert validate_causal_slot(slot) == ()
    assert slot.normalized_id() == "upstream_capacity_intervention"
    assert slot.normalized_tags() == ("capacity", "intervention", "upstream")


def test_validate_causal_slot_rejects_missing_tags_and_roles() -> None:
    slot = CausalVariableSlot(
        slot_id="bad-slot",
        role=CausalSlotRole.OUTPUT,
        description="A malformed slot.",
        semantic_tags=(),
        compatible_observable_roles=(),
    )

    errors = validate_causal_slot(slot)

    assert "slot 'bad-slot' must include semantic_tags" in errors
    assert "slot 'bad-slot' must include compatible_observable_roles" in errors


def test_validate_causal_mechanism_accepts_complete_mechanism() -> None:
    mechanism = make_bottleneck_function().mechanisms[0]

    assert validate_causal_mechanism(mechanism) == ()


def test_validate_causal_function_accepts_complete_function() -> None:
    function = make_bottleneck_function()

    assert validate_causal_function(function) == ()
    assert set(function.slot_index()) == {
        "upstream_capacity_intervention",
        "downstream_constraint",
        "final_output",
    }
    assert function.required_semantic_tags() == (
        "capacity",
        "completion",
        "constraint",
        "downstream",
        "intervention",
        "limit",
        "output",
        "throughput",
        "upstream",
    )


def test_build_causal_signature_summarizes_transfer_requirements() -> None:
    signature = build_causal_signature(make_bottleneck_function())

    assert signature.function_id == "causal-bottleneck-v1"
    assert signature.family is CausalFamily.BOTTLENECK
    assert signature.slot_count == 3
    assert signature.required_roles == (
        CausalSlotRole.CONSTRAINT,
        CausalSlotRole.INTERVENTION,
        CausalSlotRole.OUTPUT,
    )
    assert signature.mechanism_ids == ("bottleneck-no-output-gain",)


def test_validate_causal_function_rejects_missing_intervention_slot() -> None:
    valid_function = make_bottleneck_function()
    invalid_function = CausalFunction(
        function_id=valid_function.function_id,
        name=valid_function.name,
        family=valid_function.family,
        summary=valid_function.summary,
        variable_slots=valid_function.variable_slots[1:],
        mechanisms=valid_function.mechanisms,
        prior_confidence=valid_function.prior_confidence,
        uncertainty_notes=valid_function.uncertainty_notes,
        learned_from_domain_id=valid_function.learned_from_domain_id,
    )

    errors = validate_causal_function(invalid_function)

    assert "at least one intervention slot is required" in errors


def test_validate_causal_function_rejects_mismatched_mechanism_family() -> None:
    valid_function = make_bottleneck_function()
    invalid_function = CausalFunction(
        function_id=valid_function.function_id,
        name=valid_function.name,
        family=CausalFamily.SATURATION,
        summary=valid_function.summary,
        variable_slots=valid_function.variable_slots,
        mechanisms=valid_function.mechanisms,
        prior_confidence=valid_function.prior_confidence,
        uncertainty_notes=valid_function.uncertainty_notes,
        learned_from_domain_id=valid_function.learned_from_domain_id,
    )

    errors = validate_causal_function(invalid_function)

    assert (
        "mechanism 'bottleneck-no-output-gain' family 'bottleneck' does not "
        "match function family 'saturation'"
    ) in errors


def test_validate_causal_function_rejects_duplicate_slots_and_bad_confidence() -> None:
    valid_function = make_bottleneck_function()
    duplicated_slot = valid_function.variable_slots[0]
    invalid_function = CausalFunction(
        function_id=valid_function.function_id,
        name=valid_function.name,
        family=valid_function.family,
        summary=valid_function.summary,
        variable_slots=(
            duplicated_slot,
            duplicated_slot,
            valid_function.variable_slots[-1],
        ),
        mechanisms=valid_function.mechanisms,
        prior_confidence=1.2,
        uncertainty_notes=valid_function.uncertainty_notes,
        learned_from_domain_id=valid_function.learned_from_domain_id,
    )

    errors = validate_causal_function(invalid_function)

    assert "prior_confidence must be between 0.0 and 1.0" in errors
    assert "variable slot identifiers must be unique after normalization" in errors
