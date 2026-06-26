"""Worker-side execution of an immutable TrainStepPlan.

A worker receives the same plan as every other rank and: prepares slots, resolves
+ materializes leased data, runs exactly one Miles/Megatron training step with a
continuous forward/loss, exports dirty adapters, and returns a WorkerStepResult.
No scheduling/controller/store logic lives here.
"""

from __future__ import annotations

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
    """

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
                sum_loss, num_tokens, metrics = continuous_sft_loss(output, micro, job_specs)
            else:
                model_logprobs = gather_current_logprobs(output, micro["tokens"])
                sum_loss, num_tokens, metrics = continuous_rl_loss(model_logprobs, micro, job_specs)

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

    return forward_step


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
            metrics = self._train(plan, microbatches)
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

    def _train(self, plan: TrainStepPlan, microbatches: list[dict]) -> dict:
        if not microbatches:
            return {"skipped": True}
        max_seq_len = max(int(mb["tokens"].shape[0]) for mb in microbatches)
        per_token_loss = bool(getattr(self.args, "calculate_per_token_loss", True))
        forward_step = make_continuous_forward_step(
            plan.loss_type, plan.job_to_loss, per_token_loss
        )
        return train_with_custom_forward_step(
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
