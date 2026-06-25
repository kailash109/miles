"""Continuous training runner: drives the Megatron MultiLoRA model.

Runs inside the Megatron training process (Modal GPU). It pulls runnable jobs
from the controller, makes their adapters hot via the pager, packs a slot-sorted
batch, runs one Megatron forward/backward + optimizer step, commits progress,
and publishes dirty adapter artifacts.

The controller may be a plain ``TrainingJobController`` (single-rank, in-actor)
or a Ray actor handle (multi-process); ``_ctl`` adapts both. A pre-built
``(model, optimizer, opt_scheduler)`` can be injected so the runner reuses the
model a Miles train actor already built instead of constructing its own.
"""

from __future__ import annotations

import time
from typing import Iterator

import ray
import torch
from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot
from megatron.core import mpu
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import get_model_config

from miles.backends.megatron_utils.model import finalize_model_grads_with_empty_cache
from miles.backends.training_utils.loss import calculate_log_probs_and_entropy

from .adapter_pager import AdapterPager
from .artifact_publisher import AdapterArtifactPublisher
from .continuous_loss import continuous_rl_loss, continuous_sft_loss
from .continuous_packer import pack_examples_by_slot
from .schemas import ExternalTrajectoryBatch, TrainExample, TrainingJobRuntime, TrainingJobState


def _to_rows(value) -> list[list]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        value = tolist()
    return [list(row) for row in value]


def materialize_rl_examples(
    job: TrainingJobRuntime,
    batches: list[ExternalTrajectoryBatch],
) -> list[TrainExample]:
    """Convert external trajectory batches into per-sequence TrainExamples."""
    loss_type = job.spec.loss.type
    examples: list[TrainExample] = []
    for batch in batches:
        input_rows = _to_rows(batch.input_ids)
        attn_rows = _to_rows(batch.attention_mask)
        action_rows = _to_rows(batch.action_mask)
        old_rows = _to_rows(batch.old_logprobs)
        ref_rows = _to_rows(batch.ref_logprobs) if batch.ref_logprobs is not None else None
        adv_rows = _to_rows(batch.advantages) if batch.advantages is not None else None
        rew_rows = _to_rows(batch.rewards) if batch.rewards is not None else None
        group_ids = batch.group_ids or [None] * len(input_rows)
        for i, ids in enumerate(input_rows):
            examples.append(
                TrainExample(
                    job_id=job.spec.job_id,
                    slot=job.slot,
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


class _OneShotDataIterator(Iterator):
    def __init__(self, microbatches: list[dict]):
        self._it = iter(microbatches)

    def __next__(self):
        return next(self._it)


class ContinuousTrainingRunner:
    def __init__(
        self,
        args,
        controller,
        scheduler,
        trajectory_store,
        example_store,
        checkpoint_store,
        *,
        model=None,
        optimizer=None,
        opt_scheduler=None,
    ):
        self.args = args
        # The controller is always a Ray actor handle (created via
        # make_training_job_controller). All reads/writes go through .remote().
        self.controller = controller
        self.scheduler = scheduler
        self.trajectory_store = trajectory_store
        self.example_store = example_store
        self.checkpoint_store = checkpoint_store

        if model is None:
            from miles.backends.megatron_utils.multi_lora import (
                initialize_multi_lora_model_and_optimizer,
            )

            self.model, self.optimizer, self.opt_scheduler, self.iteration = (
                initialize_multi_lora_model_and_optimizer(args, role="actor")
            )
        else:
            self.model, self.optimizer, self.opt_scheduler, self.iteration = (
                model,
                optimizer,
                opt_scheduler,
                0,
            )

        self.pager = AdapterPager(
            args=args,
            model=self.model,
            optimizer=self.optimizer,
            controller=controller,
            scheduler=scheduler,
            checkpoint_store=checkpoint_store,
        )
        self.artifacts = AdapterArtifactPublisher(args=args)

    # -- loops ---------------------------------------------------------------

    def run_forever(self) -> None:
        idle_sleep = getattr(self.args, "continuous_idle_sleep_s", 0.01)
        while True:
            if not self._engine_step():
                time.sleep(idle_sleep)

    def run_bounded(self, max_engine_steps: int) -> dict:
        """Run until all jobs are terminal/out-of-data or step budget hit."""
        for _ in range(max_engine_steps):
            jobs = ray.get(self.controller.snapshot_jobs.remote())
            if jobs and all(j.state == TrainingJobState.COMPLETED for j in jobs.values()):
                break
            if not self._engine_step():
                # Nothing runnable: stop if no one has data left.
                if all(
                    j.state in {TrainingJobState.COMPLETED, TrainingJobState.WAITING_FOR_DATA}
                    for j in jobs.values()
                ):
                    break
        return self._summary()

    def _engine_step(self) -> bool:
        """One scheduling round. Returns True if a training step happened."""
        # Accrue credit server-side (mutates real state), then read a snapshot.
        ray.get(self.controller.accrue_deficits.remote(self.scheduler.base_quantum_tokens))
        jobs = ray.get(self.controller.snapshot_jobs.remote())
        selected = self.scheduler.select_training_jobs(jobs)
        if not selected:
            return False
        self._train_selected(selected, jobs)
        return True

    def _train_selected(
        self,
        selected: list[tuple[str, int]],
        jobs: dict[str, TrainingJobRuntime],
    ) -> None:
        for job_id, _tokens in selected:
            self.pager.ensure_hot(jobs[job_id], jobs)

        jobs = ray.get(self.controller.snapshot_jobs.remote())  # slots may have changed

        examples: list[TrainExample] = []
        for job_id, target_tokens in selected:
            job = jobs[job_id]
            for ex in self._pull_examples(job, target_tokens):
                ex.slot = job.slot
                examples.append(ex)
        if not examples:
            return

        for loss_type in sorted({ex.loss_type for ex in examples}):
            self._train_examples([e for e in examples if e.loss_type == loss_type], jobs)

    def _pull_examples(self, job: TrainingJobRuntime, target_tokens: int) -> list[TrainExample]:
        if job.spec.loss.type == "sft":
            return self.example_store.pop_for_job(job.spec.job_id, target_tokens)
        batches = self.trajectory_store.pop_for_job(job.spec.job_id, target_tokens)
        return materialize_rl_examples(job, batches)

    def _train_examples(
        self,
        examples: list[TrainExample],
        jobs: dict[str, TrainingJobRuntime],
    ) -> None:
        loss_type = examples[0].loss_type
        # Layout is used only for slot-sorted ordering + per-job token accounting.
        from .continuous_packer import plan_packed_layout

        layout = plan_packed_layout(
            examples, max_slots=self.args.multi_lora_n_adapters, loss_type=loss_type
        )
        job_ids = list(layout.job_ranges.keys())
        job_specs = {jid: jobs[jid].spec for jid in job_ids}

        # One example per microbatch: each sequence keeps normal causal attention
        # (no cross-sequence contamination), and grads accumulate across
        # microbatches into the disjoint per-slot adapter params within one
        # optimizer step. ``adapter_token_counts`` is one-hot per microbatch.
        microbatches = [
            pack_examples_by_slot(
                [ex], max_slots=self.args.multi_lora_n_adapters, pad_to_multiple=1, loss_type=loss_type
            )
            for ex in layout.sorted_examples
        ]

        ray.get(self.controller.mark_active_step.remote(job_ids))
        self._run_megatron_step(microbatches, job_specs, loss_type)
        ray.get(
            self.controller.commit_step.remote(
                {jid: {"trained_tokens": int(layout.job_token_counts[jid])} for jid in job_ids}
            )
        )
        self._maybe_publish(job_ids)

    def _run_megatron_step(self, microbatches: list[dict], job_specs: dict, loss_type: str) -> None:
        device = torch.cuda.current_device()
        max_seq_len = max(int(mb["tokens"].shape[0]) for mb in microbatches)

        # Mirror Miles' train()/train_one_step config setup. grad_scale_func and
        # finalize_model_grads_func are what make the *distributed* optimizer
        # correct: finalize_model_grads reduce-scatters grads from the DDP grad
        # buffer into the optimizer (needed even at DP=1, and the DP all-reduce
        # at DP>1). Without these, optimizer.step() steps on un-finalized grads.
        config = get_model_config(self.model[0])
        config.grad_scale_func = self.optimizer.scale_loss
        config.timers = None
        config.finalize_model_grads_func = finalize_model_grads_with_empty_cache

        def forward_step(data_iterator, model):
            micro = next(data_iterator)
            set_tokens_per_adapter_slot(model, micro["adapter_token_counts"].to(device))
            tokens = micro["tokens"].to(device).unsqueeze(0)  # [1, seq]
            output_tensor = model(input_ids=tokens, position_ids=None, attention_mask=None)

            def loss_fn(output):
                if loss_type == "sft":
                    return continuous_sft_loss(output, micro, job_specs)
                # RL: convert logits -> per-token current-policy logprobs (TP-aware)
                # before the clipped policy-gradient loss.
                model_logprobs = self._token_logprobs(output, micro["tokens"].to(device))
                return continuous_rl_loss(model_logprobs, micro, job_specs)

            return output_tensor, loss_fn

        # Zero grad buffers + optimizer state before the step (Miles ordering).
        for model_chunk in self.model:
            if hasattr(model_chunk, "zero_grad_buffer"):
                model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        forward_backward = get_forward_backward_func()
        forward_backward(
            forward_step_func=forward_step,
            data_iterator=_OneShotDataIterator(microbatches),
            model=self.model,
            num_microbatches=len(microbatches),
            seq_length=max_seq_len,
            micro_batch_size=1,
            forward_only=False,
        )

        update_successful, _grad_norm, _num_zeros = self.optimizer.step()
        if update_successful:
            self.opt_scheduler.step(increment=getattr(self.args, "global_batch_size", 1))
        for model_chunk in self.model:
            if hasattr(model_chunk, "zero_grad_buffer"):
                model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

    def _token_logprobs(self, logits, tokens):
        """Per-token current-policy logprobs aligned to the [T] token layout.

        ``lp[t] = log_softmax(logits[t-1])[tokens[t]]`` (TP-aware via Miles'
        ``calculate_log_probs_and_entropy``); ``lp[0] = 0`` (no predecessor).
        Returned shape is [T], matching ``loss_masks``/``rollout_log_probs`` so
        ``continuous_rl_loss`` can slice it by ``job_ranges``.
        """
        seq_logits = logits.reshape(-1, logits.shape[-1]).float()  # [T, V]
        total = seq_logits.shape[0]
        # logits[t] predicts token t+1, so align logits[0:T-1] with tokens[1:T].
        log_prob, _ = calculate_log_probs_and_entropy(
            seq_logits[: total - 1],
            tokens[1:total],
            mpu.get_tensor_model_parallel_group(),
            with_entropy=False,
            chunk_size=getattr(self.args, "log_probs_chunk_size", -1),
            true_on_policy=getattr(self.args, "true_on_policy_mode", False),
        )
        lp_full = torch.zeros(total, device=seq_logits.device, dtype=log_prob.dtype)
        lp_full[1:total] = log_prob.squeeze(-1)
        return lp_full

    def _maybe_publish(self, job_ids: list[str]) -> None:
        jobs = ray.get(self.controller.snapshot_jobs.remote())
        for job_id in job_ids:
            job = jobs[job_id]
            if not job.dirty_since_publish:
                continue
            if job.trained_steps % max(1, job.spec.budget.publish_every_steps) != 0:
                continue
            uri = self.artifacts.publish(job, self.model)
            ray.get(self.controller.mark_published.remote(job_id, uri))
            if ray.get(self.controller.budget_exhausted.remote(job_id)):
                ray.get(self.controller.complete_job.remote(job_id, uri))

    def _summary(self) -> dict:
        jobs = ray.get(self.controller.snapshot_jobs.remote())
        return {
            jid: {
                "state": j.state.value,
                "adapter_version": j.current_adapter_version,
                "latest_adapter_uri": j.latest_adapter_uri,
                "trained_steps": j.trained_steps,
                "trained_tokens": j.trained_tokens,
            }
            for jid, j in jobs.items()
        }
