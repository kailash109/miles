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

from miles.backends.training_utils.loss import calculate_log_probs_and_entropy


def gather_current_logprobs(logits, input_ids):
    """Per-token current-policy logprobs, aligned to token positions.

    ``out[p] = log_softmax(logits[p-1])[input_ids[p]]`` (TP-correct via Miles'
    ``calculate_log_probs_and_entropy``); ``out[0] = 0``. This token-position
    basis matches the external ``old_logprobs`` / ``action_mask`` the RL loss
    slices, so no off-by-one against the behavior policy. The loss applies the
    action mask, so positions outside the action span are harmless.
    """
    seq_logits = logits.reshape(-1, logits.shape[-1]).float()  # [T, V]
    total = seq_logits.shape[0]
    input_ids = input_ids.reshape(-1).to(seq_logits.device)
    out = torch.zeros(total, device=seq_logits.device, dtype=torch.float32)
    if total >= 2:
        log_prob, _ = calculate_log_probs_and_entropy(
            seq_logits[: total - 1],
            input_ids[1:total].clamp_min(0),
            mpu.get_tensor_model_parallel_group(),
            with_entropy=False,
        )
        out[1:total] = log_prob.squeeze(-1).to(out.dtype)
    return out


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
    """Token-level cross-entropy.

    Returns ``(sum_loss, num_tokens, per_job)`` -- the *unnormalized* summed
    token loss and the loss-token count -- matching Miles' loss contract so the
    caller can emit the Megatron ``(loss, normalizer, log)`` tuple and let
    ``finalize_model_grads`` do per-token gradient normalization. ``per_job`` maps
    ``job_id -> {"loss_sum", "loss_tokens"}`` (TP-reduced raw sums, detached) so
    the caller can aggregate per-adapter means across microbatches/DP for logging.
    """
    labels = batch["labels"].to(logits.device)
    loss_masks = batch["loss_masks"].to(logits.device)

    # Model may return logits as [T, vocab] or [1, T, vocab]; flatten to [T, vocab].
    logits = logits.reshape(-1, logits.shape[-1])

    token_losses = torch.nn.functional.cross_entropy(
        logits.float(),
        labels.clamp_min(0),
        reduction="none",
    )
    masked = token_losses * loss_masks

    # Per-job detached raw sums for logging (loss_sum, loss_tokens, n_seq). Per-rank
    # (not TP-reduced): the caller DP-reduces across data-parallel ranks. Computed
    # under no_grad so this logging branch never feeds autograd.
    per_job: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for job_id, ranges in batch["job_ranges"].items():
            ls = torch.zeros((), device=logits.device)
            lt = torch.zeros((), device=logits.device)
            for start, end in ranges:
                ls += masked[start:end].sum()
                lt += loss_masks[start:end].sum()
            per_job[job_id] = {
                "loss_sum": ls,
                "loss_tokens": lt,
                "n_seq": torch.tensor(float(len(ranges)), device=logits.device),
            }

    sum_loss = _tp_all_reduce_sum(masked.sum())
    num_tokens = _tp_all_reduce_sum(loss_masks.sum())
    return sum_loss, num_tokens, per_job


def _advantages_for_range(batch: dict, job_id: str, start: int, end: int, spec):
    source = spec.advantage_source
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
    """Clipped policy-gradient (GRPO/PPO) from external behavior logprobs.

    Returns ``(sum_loss, num_tokens, per_job)`` over action tokens, same
    contract as :func:`continuous_sft_loss`. ``per_job`` additionally carries a
    ``reward_sum`` (sum of per-token rewards, falling back to advantages when the
    batch has no raw rewards) so the caller can log per-adapter mean reward.
    """
    device = model_logprobs.device
    old_logprobs = batch["rollout_log_probs"].to(device)
    action_mask = batch["loss_masks"].to(device)
    raw_rewards = batch.get("rewards")
    total_sum = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)

    # Per-job detached raw sums for logging (loss_sum, loss_tokens, reward_sum,
    # n_seq), kept separate from the grad-carrying total loss. Per-rank (not
    # TP-reduced); the caller DP-reduces across data-parallel ranks.
    per_job: dict[str, dict[str, torch.Tensor]] = {}

    for job_id, ranges in batch["job_ranges"].items():
        spec = job_specs[job_id]  # a LossSpec
        job_loss = torch.zeros((), device=device)
        job_tokens = torch.zeros((), device=device)
        job_reward = torch.zeros((), device=device)
        for start, end in ranges:
            pi_logp = model_logprobs[start:end]
            old_logp = old_logprobs[start:end]
            mask = action_mask[start:end]
            advantages = _advantages_for_range(batch, job_id, start, end, spec).to(device)

            log_ratio = pi_logp - old_logp
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - spec.clip_eps, 1.0 + spec.clip_eps)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages)

            if spec.kl_ref in {"base", "provided"} and batch.get("ref_log_probs") is not None:
                ref_logp = batch["ref_log_probs"][start:end].to(device)
                ref_delta = ref_logp - pi_logp
                kl = torch.exp(ref_delta) - ref_delta - 1.0
                token_loss = policy_loss + spec.kl_coef * kl
            else:
                token_loss = policy_loss

            masked_loss = (token_loss * mask).sum()
            masked_tokens = mask.sum()
            total_sum = total_sum + masked_loss
            total_tokens = total_tokens + masked_tokens

            job_loss = job_loss + masked_loss
            job_tokens = job_tokens + masked_tokens
            # Prefer the raw reward signal; fall back to advantages when the
            # batch only carries advantages (advantage_source="provided").
            reward_slice = raw_rewards[start:end].to(device) if raw_rewards is not None else advantages
            job_reward = job_reward + (reward_slice * mask).sum()

        per_job[job_id] = {
            "loss_sum": job_loss.detach(),
            "loss_tokens": job_tokens.detach(),
            "reward_sum": job_reward.detach(),
            "n_seq": torch.tensor(float(len(ranges)), device=device),
        }

    total_sum = _tp_all_reduce_sum(total_sum)
    total_tokens = _tp_all_reduce_sum(total_tokens)
    return total_sum, total_tokens, per_job
