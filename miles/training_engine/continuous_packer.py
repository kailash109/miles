"""Pack selected jobs' examples into slot-sorted Megatron microbatches.

The hard invariant from ``MultiLoRALinear`` is that flattened tokens must be
sorted by physical adapter slot, and ``adapter_token_counts[i]`` must equal the
number of contiguous tokens routed to slot ``i``:

    [ slot 0 tokens | slot 1 tokens | slot 2 tokens | ... ]

The layout computation is kept torch-free in ``plan_packed_layout`` so it can be
unit-tested without the GPU stack; ``pack_examples_by_slot`` wraps it into torch
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .schemas import TrainExample


@dataclass
class PackedLayout:
    """Torch-free description of one packed training step."""

    sorted_examples: list[TrainExample]
    adapter_token_counts: list[int]  # length == max_slots
    job_ranges: dict[str, list[tuple[int, int]]]
    job_token_counts: dict[str, int]  # loss/action tokens per job
    total_tokens: int
    loss_type: str
    slots_used: list[int] = field(default_factory=list)


def plan_packed_layout(
    examples: Sequence[TrainExample],
    *,
    max_slots: int,
    loss_type: str,
) -> PackedLayout:
    """Compute the slot-sorted layout for a set of examples (no torch)."""
    if not examples:
        raise ValueError("no examples to pack")
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")

    seen_loss_types = {ex.loss_type for ex in examples}
    if seen_loss_types != {loss_type}:
        raise ValueError(
            f"mixed loss types in one training quantum are not supported in MVP: "
            f"{sorted(seen_loss_types)} (expected only {loss_type!r})"
        )

    for ex in examples:
        if ex.slot is None:
            raise ValueError(f"example for job {ex.job_id} has no physical slot assigned")
        if not 0 <= ex.slot < max_slots:
            raise ValueError(f"slot {ex.slot} out of range [0, {max_slots})")

    # Sort by physical slot (required), then longest-first within a slot to keep
    # packing stable/deterministic.
    ordered = sorted(examples, key=lambda ex: (int(ex.slot), -len(ex.input_ids)))

    adapter_token_counts = [0] * max_slots
    job_ranges: dict[str, list[tuple[int, int]]] = {}
    job_token_counts: dict[str, int] = {}
    slots_used: list[int] = []

    cursor = 0
    for ex in ordered:
        n = len(ex.input_ids)
        start, end = cursor, cursor + n
        adapter_token_counts[ex.slot] += n
        if ex.slot not in slots_used:
            slots_used.append(ex.slot)
        job_ranges.setdefault(ex.job_id, []).append((start, end))
        job_token_counts[ex.job_id] = job_token_counts.get(ex.job_id, 0) + int(sum(ex.loss_mask))
        cursor = end

    total_tokens = cursor
    assert sum(adapter_token_counts) == total_tokens, "adapter_token_counts must sum to total tokens"

    return PackedLayout(
        sorted_examples=ordered,
        adapter_token_counts=adapter_token_counts,
        job_ranges=job_ranges,
        job_token_counts=job_token_counts,
        total_tokens=total_tokens,
        loss_type=loss_type,
        slots_used=slots_used,
    )


def pack_examples_by_slot(
    examples: Sequence[TrainExample],
    *,
    max_slots: int,
    pad_to_multiple: int,
    loss_type: str,
) -> dict:
    """Build the torch batch dict consumed by the runner's forward step.

    Padding (to ``pad_to_multiple``) is appended to the last used slot with
    ``attention_mask``/``loss_mask`` zeroed, so it never contributes to loss or
    gradients while preserving the ``adapter_token_counts`` sum invariant.
    """
    import torch

    layout = plan_packed_layout(examples, max_slots=max_slots, loss_type=loss_type)

    input_ids: list[int] = []
    attention_mask: list[int] = []
    loss_mask: list[int] = []
    labels: list[int] = []
    old_logprobs: list[float] = []
    ref_logprobs: list[float] = []
    advantages: list[float] = []
    returns: list[float] = []
    rewards: list[float] = []
    group_ids: list[str | None] = []

    for ex in layout.sorted_examples:
        input_ids.extend(ex.input_ids)
        attention_mask.extend(ex.attention_mask)
        loss_mask.extend(ex.loss_mask)
        if loss_type == "sft":
            labels.extend(ex.labels or [])
        else:
            old_logprobs.extend(ex.old_logprobs or [])
            if ex.ref_logprobs is not None:
                ref_logprobs.extend(ex.ref_logprobs)
            if ex.advantages is not None:
                advantages.extend(ex.advantages)
            if ex.returns is not None:
                returns.extend(ex.returns)
            if ex.rewards is not None:
                rewards.extend(ex.rewards)
            group_ids.append(ex.group_id)

    adapter_token_counts = list(layout.adapter_token_counts)

    # Pad the flat sequence to a multiple, attributing pad tokens to the last
    # used slot so tokens-per-adapter still sums to the total token count.
    if pad_to_multiple and pad_to_multiple > 1:
        remainder = len(input_ids) % pad_to_multiple
        if remainder:
            pad = pad_to_multiple - remainder
            last_slot = layout.slots_used[-1]
            input_ids.extend([0] * pad)
            attention_mask.extend([0] * pad)
            loss_mask.extend([0] * pad)
            adapter_token_counts[last_slot] += pad
            if loss_type == "sft":
                labels.extend([-100] * pad)
            else:
                old_logprobs.extend([0.0] * pad)
                if ref_logprobs:
                    ref_logprobs.extend([0.0] * pad)

    batch: dict = {
        "tokens": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int32),
        "loss_masks": torch.tensor(loss_mask, dtype=torch.float32),
        "adapter_token_counts": torch.tensor(adapter_token_counts, dtype=torch.int32),
        "job_ranges": layout.job_ranges,
        "job_token_counts": layout.job_token_counts,
        "loss_type": loss_type,
    }

    if loss_type == "sft":
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
    else:
        batch["rollout_log_probs"] = torch.tensor(old_logprobs, dtype=torch.float32)
        if ref_logprobs:
            batch["ref_log_probs"] = torch.tensor(ref_logprobs, dtype=torch.float32)
        if advantages:
            batch["advantages"] = torch.tensor(advantages, dtype=torch.float32)
        if returns:
            batch["returns"] = torch.tensor(returns, dtype=torch.float32)
        if rewards:
            batch["rewards"] = torch.tensor(rewards, dtype=torch.float32)
        batch["group_ids"] = group_ids

    return batch
