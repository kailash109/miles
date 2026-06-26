import dataclasses
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def is_multi_lora_enabled(args: Namespace) -> bool:
    return getattr(args, "multi_lora", False)


def create_multi_lora(args: Namespace):
    """Create a MultiLoRA instance from training args."""
    from megatron.bridge.peft.multi_lora import MultiLoRA

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_megatron, exclude_mtp_vision_modules

    lora_type_name = getattr(args, "lora_type", "lora").lower()
    if lora_type_name == "canonical_lora":
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA
        lora_cls = CanonicalLoRA
    else:
        from megatron.bridge.peft.lora import LoRA
        lora_cls = LoRA

    target_modules = convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls)
    if "Qwen3.5" in args.hf_checkpoint:
        target_modules = exclude_mtp_vision_modules(target_modules)

    return MultiLoRA(
        target_modules=target_modules,
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def build_multi_lora_model(args: Namespace):
    """Build Megatron model with MultiLoRA layers via megatron-bridge.

    Returns DDP-wrapped model chunks. Does NOT register adapters or load checkpoints —
    that happens after the optimizer is created.
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig

    from miles.backends.megatron_utils.bridge_lora_helpers import _make_value_model_hook

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    provider.context_parallel_size = args.context_parallel_size
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_router_load_balancing_type = "none"
    provider.finalize()

    multi_lora = create_multi_lora(args)

    def apply_hook(model_chunks):
        transformed = multi_lora(model_chunks, training=True)
        multi_lora.set_params_to_save(transformed)
        return transformed

    provider.register_pre_wrap_hook(apply_hook)

    is_value_model = (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        provider.register_pre_wrap_hook(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    ddp_config.finalize()

    if args.offload_train:
        from miles.backends.megatron_utils.lora_utils import patch_param_grad_buffer_for_colocate_mode_lora
        patch_param_grad_buffer_for_colocate_mode_lora()

    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)
    return model, multi_lora


def initialize_multi_lora_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
):
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.ci_utils import check_model_hashes, check_peak_gpu_memory_after_load
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler
    from miles.utils.memory_utils import clear_memory

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync

    model, multi_lora = build_multi_lora_model(args)
    model[0].role = role

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None
    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)

    # Hide adapter params so the bridge's conversion-task walk doesn't see them
    # while loading the base checkpoint.
    from megatron.bridge.peft.multi_lora_layers import hide_adapters

    clear_memory()
    with hide_adapters(model):
        iteration, _ = load_checkpoint(
            model, optimizer, opt_param_scheduler,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
    check_peak_gpu_memory_after_load(args)
    clear_memory()
    check_model_hashes(args, model, iteration)
    opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    # Install every adapter the controller already knows about, before the
    # caller's backuper takes its snapshot. See the docstring for why.
    from .update_weight.multi_lora_sync import load_pending_adapters
    load_pending_adapters(args, model, optimizer)

    return model, optimizer, opt_param_scheduler, iteration


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None

    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    for step_dir in step_dirs:
        all_present = all(
            (step_dir / f"adapter_megatron_tp{tp}_pp{pp}.pt").exists()
            for tp in range(tp_size)
            for pp in range(pp_size)
        )
        if all_present:
            return step_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    return None


def iter_named_multi_lora_modules(model):
    """Yield ``(stable_module_prefix, MultiLoRALinear)`` pairs.

    The prefix is stable across process restarts and slot moves (it is the
    model-chunk index + the module's ``named_modules`` path), unlike Python
    object ids, so it is safe as a durable optimizer-state key.
    """
    models = model if isinstance(model, list) else [model]
    for chunk_i, model_chunk in enumerate(models):
        for name, module in model_chunk.named_modules():
            if module.__class__.__name__ == "MultiLoRALinear":
                yield f"chunk{chunk_i}.{name}", module


def iter_adapter_named_params_for_slot(model, idx: int):
    """Yield ``(stable_name, param)`` for adapter slot ``idx``.

    The name is deliberately *slot-independent* (``...adapter.<param>``, not
    ``...adapters.<idx>.<param>``) so optimizer state captured from one slot can
    be restored into a different slot when a logical job migrates.
    """
    for module_prefix, module in iter_named_multi_lora_modules(model):
        adapter = module.adapters[idx]
        for name, param in adapter.named_parameters():
            yield f"{module_prefix}.adapter.{name}", param


def capture_optimizer_state_for_adapter(optimizer, model, idx: int) -> dict:
    """Snapshot (CPU) Adam state for adapter slot ``idx`` keyed by stable name."""
    name_by_main_param_id: dict[int, str] = {}
    for stable_name, param in iter_adapter_named_params_for_slot(model, idx):
        main = getattr(param, "main_param", None)
        key_param = main if main is not None else param
        name_by_main_param_id[id(key_param)] = stable_name

    captured: dict[tuple[int, str], dict] = {}
    chained = getattr(optimizer, "chained_optimizers", [optimizer])
    for opt_i, chained_optimizer in enumerate(chained):
        inner = getattr(chained_optimizer, "optimizer", chained_optimizer)
        for param, state in inner.state.items():
            stable_name = name_by_main_param_id.get(id(param))
            if stable_name is None:
                continue
            captured[(opt_i, stable_name)] = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in state.items()
            }
    return captured


def restore_optimizer_state_for_adapter(optimizer, model, idx: int, captured: dict) -> None:
    """Restore Adam state captured by ``capture_optimizer_state_for_adapter``."""
    param_by_stable_name: dict[str, Any] = {}
    for stable_name, param in iter_adapter_named_params_for_slot(model, idx):
        main = getattr(param, "main_param", None)
        key_param = main if main is not None else param
        param_by_stable_name[stable_name] = key_param

    chained = getattr(optimizer, "chained_optimizers", [optimizer])
    for opt_i, chained_optimizer in enumerate(chained):
        inner = getattr(chained_optimizer, "optimizer", chained_optimizer)
        for (saved_opt_i, stable_name), state in captured.items():
            if saved_opt_i != opt_i:
                continue
            param = param_by_stable_name.get(stable_name)
            if param is None:
                continue
            inner.state[param] = {
                k: (v.to(device=param.device) if torch.is_tensor(v) else v)
                for k, v in state.items()
            }


def zero_optimizer_state_for_adapter(optimizer, model, idx: int) -> None:
    from megatron.bridge.peft.multi_lora_layers import (
        MultiLoRALinear,
        _iter_multi_lora_modules,
    )

    target_main_params = set()
    for module in _iter_multi_lora_modules(model):
        if not isinstance(module, MultiLoRALinear):
            continue
        adapter = module.adapters[idx]
        for param in adapter.parameters():
            main = getattr(param, "main_param", None)
            target_main_params.add(id(main if main is not None else param))

    chained = getattr(optimizer, "chained_optimizers", [optimizer])
    for chained_optimizer in chained:
        inner = getattr(chained_optimizer, "optimizer", chained_optimizer)
        for param, state in inner.state.items():
            if id(param) not in target_main_params:
                continue
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
