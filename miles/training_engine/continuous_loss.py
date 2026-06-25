"""Per-job-normalized losses for continuous MultiLoRA training.

Loss is normalized per logical job and then summed. Because each job's LoRA
slot is a disjoint set of parameters, summing per-job means is safe and avoids
the largest job dominating the gradient.

This runs inside the Megatron training process (Modal GPU), so torch/Megatron
are imported normally.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu


def continuous_loss(output_tensor, batch: dict, job_specs: dict[str, Any]):
    loss_type = batch["loss_type"]
    if loss_type == "sft":
        return continuous_sft_loss(output_tensor, batch, job_specs)
    if loss_type in {"grpo", "ppo"}:
        return continuous_rl_loss(output_tensor, batch, job_specs)
    raise ValueError(f"unsupported loss_type: {loss_type}")


def _tp_all_reduce_sum(value):
    """Sum a scalar across the tensor-model-parallel group."""
    if dist.is_initialized():
        dist.all_reduce(value, group=mpu.get_tensor_model_parallel_group())
    return value


def continuous_sft_loss(logits, batch: dict, job_specs: dict[str, Any]):
    """Token-level cross-entropy, normalized per job over its loss tokens."""
    labels = batch["labels"].to(logits.device)
    loss_masks = batch["loss_masks"].to(logits.device)

    # Model may return logits as [T, vocab] or [1, T, vocab]; flatten to [T, vocab].
    logits = logits.reshape(-1, logits.shape[-1])

    token_losses = torch.nn.functional.cross_entropy(
        logits.float(),
        labels.clamp_min(0),
        reduction="none",
    )

    total = torch.zeros((), device=logits.device)
    metrics: dict[str, Any] = {}
    for job_id, ranges in batch["job_ranges"].items():
        local_sum = torch.zeros((), device=logits.device)
        local_count = torch.zeros((), device=logits.device)
        for start, end in ranges:
            mask = loss_masks[start:end]
            local_sum = local_sum + (token_losses[start:end] * mask).sum()
            local_count = local_count + mask.sum()
        global_sum = _tp_all_reduce_sum(local_sum)
        global_count = _tp_all_reduce_sum(local_count).clamp_min(1.0)
        job_loss = global_sum / global_count
        total = total + job_loss
        metrics[f"{job_id}/loss"] = job_loss.detach()
        metrics[f"{job_id}/tokens"] = global_count.detach()
    return total, metrics


def _advantages_for_range(batch: dict, job_id: str, start: int, end: int, spec):
    source = spec.loss.advantage_source
    if source == "provided":
        advantages = batch.get("advantages")
        if advantages is None:
            raise ValueError(f"job {job_id}: advantage_source=provided requires advantages")
        return advantages[start:end]

    rewards = batch.get("rewards")
    if rewards is None:
        raise ValueError(f"job {job_id}: advantage_source={source} requires rewards")
    adv = rewards[start:end]
    if source == "trainer_from_group_rewards":
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    return adv


def continuous_rl_loss(model_logprobs, batch: dict, job_specs: dict[str, Any]):
    """Clipped policy-gradient (GRPO/PPO) from external behavior logprobs."""
    old_logprobs = batch["rollout_log_probs"]
    action_mask = batch["loss_masks"]
    total = torch.zeros((), device=model_logprobs.device)
    metrics: dict[str, Any] = {}

    for job_id, ranges in batch["job_ranges"].items():
        spec = job_specs[job_id]
        loss_terms = []
        mask_terms = []
        for start, end in ranges:
            pi_logp = model_logprobs[start:end]
            old_logp = old_logprobs[start:end]
            mask = action_mask[start:end]
            advantages = _advantages_for_range(batch, job_id, start, end, spec)

            log_ratio = pi_logp - old_logp
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - spec.loss.clip_eps, 1.0 + spec.loss.clip_eps)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages)

            if spec.loss.kl_ref in {"base", "provided"} and batch.get("ref_log_probs") is not None:
                ref_logp = batch["ref_log_probs"][start:end]
                ref_delta = ref_logp - pi_logp
                kl = torch.exp(ref_delta) - ref_delta - 1.0
                token_loss = policy_loss + spec.loss.kl_coef * kl
            else:
                token_loss = policy_loss

            loss_terms.append(token_loss)
            mask_terms.append(mask)

        losses = torch.cat(loss_terms)
        masks = torch.cat(mask_terms)
        global_sum = _tp_all_reduce_sum((losses * masks).sum())
        global_count = _tp_all_reduce_sum(masks.sum()).clamp_min(1.0)
        job_loss = global_sum / global_count
        total = total + job_loss
        metrics[f"{job_id}/loss"] = job_loss.detach()
    return total, metrics
