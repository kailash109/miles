"""CPU tests for the slot-sorted packing layout (no torch)."""

from __future__ import annotations

import pytest

from miles.training_engine.continuous_packer import plan_packed_layout
from tests.training_engine.helpers import make_example


def test_layout_sorted_by_slot_and_counts_consistent():
    # Examples deliberately out of slot order.
    examples = [
        make_example("jobC", slot=2, n=3),
        make_example("jobA", slot=0, n=5),
        make_example("jobC2", slot=2, n=2),
        make_example("jobB", slot=1, n=4),
    ]
    layout = plan_packed_layout(examples, max_slots=4, loss_type="sft")

    # Sorted by physical slot.
    slots = [ex.slot for ex in layout.sorted_examples]
    assert slots == sorted(slots)

    # adapter_token_counts sums to total tokens.
    assert sum(layout.adapter_token_counts) == layout.total_tokens == (3 + 5 + 2 + 4)
    # Per-slot counts: slot0=5, slot1=4, slot2=3+2=5, slot3=0.
    assert layout.adapter_token_counts == [5, 4, 5, 0]


def test_job_ranges_match_token_spans():
    examples = [
        make_example("a", slot=0, n=2),
        make_example("b", slot=1, n=3),
    ]
    layout = plan_packed_layout(examples, max_slots=2, loss_type="sft")

    # Slot 0 (job a) occupies [0,2); slot 1 (job b) occupies [2,5).
    assert layout.job_ranges["a"] == [(0, 2)]
    assert layout.job_ranges["b"] == [(2, 5)]
    # Ranges are contiguous and cover the whole packed sequence.
    flat = [r for ranges in layout.job_ranges.values() for r in ranges]
    flat.sort()
    assert flat[0][0] == 0
    assert flat[-1][1] == layout.total_tokens


def test_job_token_counts_use_loss_mask():
    # 4 tokens, only 2 trainable.
    examples = [make_example("a", slot=0, n=4, n_loss_tokens=2)]
    layout = plan_packed_layout(examples, max_slots=1, loss_type="sft")
    assert layout.job_token_counts["a"] == 2
    # adapter_token_counts counts ALL tokens (for the forward pass), not just loss.
    assert layout.adapter_token_counts == [4]


def test_mixed_loss_types_rejected():
    examples = [
        make_example("a", slot=0, n=2, loss_type="sft"),
        make_example("b", slot=1, n=2, loss_type="grpo"),
    ]
    with pytest.raises(ValueError, match="mixed loss types"):
        plan_packed_layout(examples, max_slots=2, loss_type="sft")


def test_slot_out_of_range_rejected():
    examples = [make_example("a", slot=5, n=2)]
    with pytest.raises(ValueError, match="out of range"):
        plan_packed_layout(examples, max_slots=2, loss_type="sft")


def test_empty_rejected():
    with pytest.raises(ValueError, match="no examples"):
        plan_packed_layout([], max_slots=2, loss_type="sft")
