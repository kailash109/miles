"""Turn a plan's leased payloads into slot-stamped, packed microbatches.

Resolves each lease's Ray ObjectRef, converts the payload (pre-tokenized SFT
``TrainExample``s or an RL ``ExternalTrajectoryBatch``) into ``TrainExample``s
stamped with the plan's physical slot, then emits one microbatch per sequence
(safe causal attention). Runs inside the Megatron training process.
"""

from __future__ import annotations

import ray

from .continuous_packer import pack_examples_by_slot
from .plan import TrainStepPlan
from .schemas import ExternalTrajectoryBatch, TrainExample


def _to_rows(value) -> list[list]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        value = tolist()
    return [list(row) for row in value]


def rl_examples_from_batch(
    batch: ExternalTrajectoryBatch, job_id: str, slot: int, loss_type: str
) -> list[TrainExample]:
    input_rows = _to_rows(batch.input_ids)
    attn_rows = _to_rows(batch.attention_mask)
    action_rows = _to_rows(batch.action_mask)
    old_rows = _to_rows(batch.old_logprobs)
    ref_rows = _to_rows(batch.ref_logprobs) if batch.ref_logprobs is not None else None
    adv_rows = _to_rows(batch.advantages) if batch.advantages is not None else None
    rew_rows = _to_rows(batch.rewards) if batch.rewards is not None else None
    group_ids = batch.group_ids or [None] * len(input_rows)

    examples = []
    for i, ids in enumerate(input_rows):
        examples.append(
            TrainExample(
                job_id=job_id,
                slot=slot,
                loss_type=loss_type,
                adapter_version=batch.adapter_version,
                input_ids=[int(t) for t in ids],
                attention_mask=[int(a) for a in attn_rows[i]],
                loss_mask=[int(a) for a in action_rows[i]],
                old_logprobs=[float(x) for x in old_rows[i]],
                ref_logprobs=[float(x) for x in ref_rows[i]] if ref_rows else None,
                advantages=[float(x) for x in adv_rows[i]] if adv_rows else None,
                rewards=[float(x) for x in rew_rows[i]] if rew_rows else None,
                group_id=group_ids[i] if i < len(group_ids) else None,
            )
        )
    return examples


class BatchMaterializer:
    def __init__(self, args, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer

    def materialize(self, plan: TrainStepPlan) -> list[dict]:
        examples: list[TrainExample] = []
        for job_id, leases in plan.leases.items():
            slot = plan.job_to_slot[job_id]
            payloads = ray.get([lease.payload_ref for lease in leases])
            for payload in payloads:
                examples.extend(self._payload_to_examples(payload, job_id, slot, plan.loss_type))

        # One sequence per microbatch (normal causal attention); grads accumulate
        # across microbatches into disjoint per-slot params in one optimizer step.
        return [
            pack_examples_by_slot(
                [ex], max_slots=self.args.multi_lora_n_adapters, pad_to_multiple=1, loss_type=plan.loss_type
            )
            for ex in examples
        ]

    def _payload_to_examples(self, payload, job_id, slot, loss_type) -> list[TrainExample]:
        if isinstance(payload, list):  # pre-tokenized SFT TrainExamples
            for ex in payload:
                ex.slot = slot
            return payload
        return rl_examples_from_batch(payload, job_id, slot, loss_type)
