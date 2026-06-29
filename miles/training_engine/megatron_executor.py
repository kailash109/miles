"""Worker-side execution of an immutable TrainStepPlan.

A worker receives the same plan as every other rank and: prepares slots, resolves
+ materializes leased data, runs exactly one Miles/Megatron training step with a
continuous forward/loss, exports dirty adapters, and returns a WorkerStepResult.
No scheduling/controller/store logic lives here.
"""

from __future__ import annotations

import os
from typing import Iterator

import torch
from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

from miles.backends.megatron_utils.model import train_with_custom_forward_step

from .adapter_slot_executor import AdapterSlotExecutor
from .batch_materializer import BatchMaterializer
from .continuous_loss import continuous_rl_loss, continuous_sft_loss, gather_current_logprobs
from .plan import TrainStepPlan
from .results import WorkerStepResult


class _OneShotDataIterator(Iterator):
    def __init__(self, microbatches: list[dict]):
        self._it = iter(microbatches)

    def __next__(self):
        return next(self._it)


def _global_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def make_continuous_forward_step(loss_type: str, job_specs: dict, per_token_loss: bool):
    """Build a Megatron forward_step returning the ``(loss, normalizer, log)`` triple.

    Mirrors Miles' ``loss_function`` contract: the normalizer is always a
    *tensor* (Megatron sums + clamps it). With per-token loss the returned loss
    is the unnormalized token sum and the normalizer is the token count, so
    ``finalize_model_grads`` divides gradients by the total tokens across
    microbatches. Otherwise we return the per-token mean and normalizer 1, and
    the schedule averages over microbatches.

    Returns ``(forward_step, per_job_accum)``. ``per_job_accum`` is a mutable
    ``job_id -> {"loss_sum", "loss_tokens", "reward_sum"}`` dict that the loss
    func accumulates (TP-reduced raw sums) across microbatches; the caller
    DP-reduces it after the step to log per-adapter loss/reward.
    """
    per_job_accum: dict[str, dict[str, torch.Tensor]] = {}

    def forward_step(data_iterator, model):
        micro = next(data_iterator)
        device = torch.cuda.current_device()
        for key, value in list(micro.items()):
            if torch.is_tensor(value):
                micro[key] = value.to(device)
        set_tokens_per_adapter_slot(model, micro["adapter_token_counts"])
        logits = model(
            input_ids=micro["tokens"].unsqueeze(0),
            position_ids=None,
            attention_mask=None,
        )

        def loss_func(output):
            if loss_type == "sft":
                sum_loss, num_tokens, per_job = continuous_sft_loss(output, micro, job_specs)
            else:
                model_logprobs = gather_current_logprobs(output, micro["tokens"])
                sum_loss, num_tokens, per_job = continuous_rl_loss(model_logprobs, micro, job_specs)

            # Accumulate per-adapter raw sums across microbatches (detached).
            for job_id, sums in per_job.items():
                acc = per_job_accum.setdefault(job_id, {})
                for key, val in sums.items():
                    acc[key] = acc.get(key, torch.zeros((), device=val.device)) + val

            # Token COUNT: Megatron accumulates the normalizer in an integer
            # tensor across microbatches, so it must be int (loss_masks are
            # float). Clamp with an int to avoid a float->int cast error.
            num_tokens = num_tokens.to(torch.long).clamp_min(1)
            if per_token_loss:
                loss = sum_loss
                normalizer = num_tokens
            else:
                loss = sum_loss / num_tokens
                normalizer = torch.ones((), dtype=torch.long, device=loss.device)
            # Report the token-sum of loss + the token count (count first), so
            # ``aggregate_train_losses`` yields the mean per-token loss regardless
            # of the backprop normalization chosen above.
            report_values = torch.stack([num_tokens.detach().float(), sum_loss.detach().float()])
            return loss, normalizer, {"keys": ["loss"], "values": report_values}

        return logits, loss_func

    return forward_step, per_job_accum


def _reduce_per_job_metrics(
    per_job_accum: dict[str, dict[str, torch.Tensor]],
    job_ids: list[str],
    loss_type: str,
) -> dict[str, float]:
    """DP-reduce accumulated per-adapter raw sums into per-adapter + aggregate metrics.

    Mirrors ``aggregate_train_losses``' data+context-parallel reduction, so sums
    are correct across DP shards. Only the last pipeline stage populates
    ``per_job_accum`` (that's where the loss is computed), matching where the
    aggregate loss is reduced. Emits per-adapter
    ``{job_id}/train/{loss,reward,tokens,n_sequences,avg_response_len}`` plus
    step-level aggregates ``train/{total_tokens,n_sequences,avg_response_len}``.

    ``job_ids`` MUST be the plan's full ``selected_jobs`` (identical on every
    rank). The reduction tensor is shaped/ordered by it -- NOT by the locally
    observed jobs -- so the cross-DP all-reduce shapes always match even when a
    rank's data shard only touched a subset of the step's jobs.
    """
    from megatron.core import mpu

    # Loss (and thus per-job accumulation) only exists on the last pipeline
    # stage; matching where the aggregate loss is reduced avoids a cross-stage
    # collective mismatch.
    if not job_ids or not mpu.is_pipeline_last_stage():
        return {}

    import torch.distributed as dist

    from miles.backends.training_utils.parallel import get_parallel_state

    # Reward only exists for RL losses; derive from the (rank-invariant)
    # loss_type so every rank builds an identically shaped reduction tensor.
    has_reward = loss_type != "sft"
    keys = ["loss_sum", "loss_tokens", "n_seq"] + (["reward_sum"] if has_reward else [])
    device = torch.cuda.current_device()
    stats = torch.zeros((len(job_ids), len(keys)), device=device)
    for i, job_id in enumerate(job_ids):
        acc = per_job_accum.get(job_id, {})
        for c, key in enumerate(keys):
            val = acc.get(key)
            if val is not None:
                stats[i, c] = val

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=get_parallel_state().intra_dp_cp.group)

    col = {key: c for c, key in enumerate(keys)}
    out: dict[str, float] = {}
    total_tokens = 0.0
    total_seqs = 0.0
    for i, job_id in enumerate(job_ids):
        tokens = float(stats[i, col["loss_tokens"]])
        n_seq = float(stats[i, col["n_seq"]])
        total_tokens += tokens
        total_seqs += n_seq
        out[f"{job_id}/train/loss"] = float(stats[i, col["loss_sum"]]) / max(tokens, 1.0)
        out[f"{job_id}/train/tokens"] = tokens
        out[f"{job_id}/train/n_sequences"] = n_seq
        out[f"{job_id}/train/avg_response_len"] = tokens / max(n_seq, 1.0)
        if has_reward:
            out[f"{job_id}/train/reward"] = float(stats[i, col["reward_sum"]]) / max(tokens, 1.0)

    out["train/total_tokens"] = total_tokens
    out["train/n_sequences"] = total_seqs
    out["train/avg_response_len"] = total_tokens / max(total_seqs, 1.0)
    return out


class MegatronPlanExecutor:
    def __init__(self, args, model, optimizer, opt_param_scheduler, tokenizer=None, *, writer=None, hf_iterator=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.slot_executor = AdapterSlotExecutor(args, model, optimizer, writer=writer, hf_iterator=hf_iterator)
        self.materializer = BatchMaterializer(args, tokenizer)

    def execute(self, plan: TrainStepPlan) -> WorkerStepResult:
        rank = _global_rank()
        try:
            self.slot_executor.prepare_slots(plan)
            microbatches = self.materializer.materialize(plan)
            if rank == 0:
                total_tokens = sum(int(mb["tokens"].shape[0]) for mb in microbatches)
                print(
                    f"[worker] running forward/backward: step={plan.engine_step} "
                    f"loss={plan.loss_type} jobs={list(plan.selected_jobs)} "
                    f"slots={plan.job_to_slot} tokens={total_tokens}",
                    flush=True,
                )
            # Decode a few sample rollouts BEFORE _train (microbatch tensors are
            # still on CPU; _train moves them to GPU in place).
            samples = self._maybe_collect_samples(plan, microbatches, rank)
            metrics = self._train(plan, microbatches)
            if samples:
                metrics["_rollout_samples"] = samples
            if rank == 0:
                print(
                    f"[worker] step={plan.engine_step} done "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"grad_norm={metrics.get('grad_norm', float('nan')):.3f} "
                    f"jobs={list(plan.selected_jobs)}",
                    flush=True,
                )
            written = self.slot_executor.write_dirty_adapter_files(plan)
            return WorkerStepResult(
                plan_id=plan.plan_id, rank=rank, ok=True, metrics=metrics, written_files=written
            )
        except Exception as exc:  # noqa: BLE001 - report failure to the coordinator
            import traceback

            return WorkerStepResult(
                plan_id=plan.plan_id,
                rank=rank,
                ok=False,
                error=f"{exc!r}\n{traceback.format_exc()}",
            )

    def _maybe_collect_samples(self, plan: TrainStepPlan, microbatches: list[dict], rank: int) -> list[dict]:
        """Decode up to one rollout per adapter as readable text, for W&B inspection.

        Gated by ``ENGINE_LOG_SAMPLES_EVERY`` (steps; 0 = off) and rank 0 only.
        Splits prompt vs response via the action/loss mask and reads the reward
        from the (response-broadcast) per-token reward. Cheap because it runs at
        most every N steps over a handful of sequences.
        """
        interval = int(os.environ.get("ENGINE_LOG_SAMPLES_EVERY", "0"))
        tokenizer = self.materializer.tokenizer
        if interval <= 0 or rank != 0 or tokenizer is None:
            return []
        if plan.engine_step % interval != 0:
            return []
        max_chars = int(os.environ.get("ENGINE_LOG_SAMPLES_MAX_CHARS", "4000"))

        seen: set[str] = set()
        out: list[dict] = []
        for mb in microbatches:
            ranges = mb.get("job_ranges") or {}
            if len(ranges) != 1:
                continue
            job_id = next(iter(ranges))
            if job_id in seen:
                continue
            seen.add(job_id)

            tokens = mb["tokens"]
            loss_mask = mb["loss_masks"]
            attn = mb.get("attention_mask")
            resp_sel = loss_mask > 0
            prompt_sel = (loss_mask == 0) if attn is None else ((attn > 0) & (loss_mask == 0))
            response_ids = tokens[resp_sel].tolist()
            prompt_ids = tokens[prompt_sel].tolist()

            reward = None
            rewards = mb.get("rewards")
            if rewards is not None and rewards.numel() > 0:
                resp_rewards = rewards[resp_sel]
                if resp_rewards.numel() > 0:
                    reward = float(resp_rewards[0])

            out.append({
                "engine_step": int(plan.engine_step),
                "adapter": job_id,
                "reward": reward,
                "response_len": len(response_ids),
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False)[:max_chars],
                "response": tokenizer.decode(response_ids, skip_special_tokens=False)[:max_chars],
            })
        return out

    def _train(self, plan: TrainStepPlan, microbatches: list[dict]) -> dict:
        # Every DP rank must drive the same per-step collectives (grad finalize,
        # optimizer step, loss all-reduce, per-job all-reduce) exactly once, in
        # the same order, even when this rank's data shard is empty -- otherwise
        # the ranks that have data deadlock on the grad all-reduce. The data
        # volume only changes the forward compute, never the collective count.
        if microbatches:
            metrics, per_job_accum = self._train_with_data(plan, microbatches)
        else:
            metrics, per_job_accum = self._sync_step_without_data()
        metrics.update(_reduce_per_job_metrics(per_job_accum, list(plan.selected_jobs), plan.loss_type))
        return metrics

    def _train_with_data(self, plan: TrainStepPlan, microbatches: list[dict]) -> tuple[dict, dict]:
        max_seq_len = max(int(mb["tokens"].shape[0]) for mb in microbatches)
        per_token_loss = bool(getattr(self.args, "calculate_per_token_loss", True))
        forward_step, per_job_accum = make_continuous_forward_step(
            plan.loss_type, plan.job_to_loss, per_token_loss
        )
        metrics = train_with_custom_forward_step(
            self.args,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            forward_step_func=forward_step,
            data_iterator=_OneShotDataIterator(microbatches),
            num_microbatches=len(microbatches),
            seq_length=max_seq_len,
            micro_batch_size=1,
        )
        return metrics, per_job_accum

    def _sync_step_without_data(self) -> tuple[dict, dict]:
        """Empty-shard step: run the end-of-step collectives + optimizer step
        with a zero contribution, so this rank stays lockstep with the ranks
        that trained -- no dummy forward, no normalizer perturbation.

        Mirrors, in order, the collectives ``train_with_custom_forward_step``
        runs on a data rank: grad finalize + optimizer step (via
        ``sync_train_step_no_data``), then the ``aggregate_train_losses``
        intra_dp_cp all-reduce (matched here with a zero-filled contribution in
        the forward step's fixed ``["loss"]`` report shape). The per-job metric
        all-reduce is driven by the shared ``_reduce_per_job_metrics`` call in
        ``_train`` with an empty accumulator.
        """
        from megatron.core import mpu

        from miles.backends.megatron_utils.model import sync_train_step_no_data
        from miles.backends.training_utils.log_utils import aggregate_train_losses

        metrics = sync_train_step_no_data(
            self.args, self.model, self.optimizer, self.opt_param_scheduler
        )
        # The loss all-reduce only runs on the last pipeline stage on data ranks
        # (that's where forward_backward returns loss dicts), so mirror it there.
        # forward_step reports {"keys": ["loss"], "values": [num_tokens, sum_loss]}
        # (see make_continuous_forward_step). Contribute zeros of that exact shape
        # so the reduce matches the data ranks; afterward this rank holds the same
        # global mean loss they do.
        if mpu.is_pipeline_last_stage():
            keys = ["loss"]
            zero = [{"keys": keys, "values": torch.zeros(len(keys) + 1, device=torch.cuda.current_device())}]
            metrics.update({k: float(v) for k, v in aggregate_train_losses(zero).items()})
        return metrics, {}
