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

        if not examples:
            return []

        # Data-parallel sharding: each DP rank trains a *disjoint* subset of the
        # plan's sequences. Megatron's finalize_model_grads sums grads across the
        # DP(-CP) group and (with per-token loss) sums the token normalizer too,
        # so this reproduces single-GPU semantics over the full global batch
        # while giving DP-x throughput. TP/CP ranks share a DP coordinate, so
        # they get the *same* shard (required for collective-correct TP forward).
        #
        # A rank may legitimately get an EMPTY shard (fewer sequences than DP
        # ranks). That is fine: the executor still drives the end-of-step DP
        # collectives (grad finalize + optimizer step) for an empty shard so all
        # ranks stay in lockstep -- without wasting a forward/backward.
        dp_rank, dp_size = self._dp_rank_and_size()
        local = examples[dp_rank::dp_size] if dp_size > 1 else examples

        # One sequence per microbatch (normal causal attention); grads accumulate
        # across microbatches into disjoint per-slot params in one optimizer step.
        return [
            pack_examples_by_slot(
                [ex], max_slots=self.args.multi_lora_n_adapters, pad_to_multiple=1, loss_type=plan.loss_type
            )
            for ex in local
        ]

    def _dp_rank_and_size(self) -> tuple[int, int]:
        try:
            from miles.backends.training_utils.parallel import get_parallel_state

            dp = get_parallel_state().intra_dp
            return dp.rank, dp.size
        except Exception:
            return 0, 1

    def _payload_to_examples(self, payload, job_id, slot, loss_type) -> list[TrainExample]:
        if isinstance(payload, list):  # pre-tokenized SFT TrainExamples
            for ex in payload:
                ex.slot = slot
            return payload
        return rl_examples_from_batch(payload, job_id, slot, loss_type)
