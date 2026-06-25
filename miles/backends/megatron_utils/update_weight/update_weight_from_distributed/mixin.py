from argparse import Namespace
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef

from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import timer

from ...lora_utils import LORA_ADAPTER_NAME, build_lora_sync_config, is_lora_weight_name
from ...megatron_to_hf import convert_to_hf
from ...sglang import FlattenedTensorBucket, MultiprocessingSerializer
from ..common import (
    _check_weight_sync_results,
    all_gather_param,
    collect_named_tensors_for_weight_transfer,
    post_process_weights,
)
from ..hf_weight_iterator_base import HfWeightIteratorBase


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (
        (".adapter." in name or ".adapters." in name) and ("linear_in" in name or "linear_out" in name)
    )


class DistBucketedWeightUpdateMixin:
    """Mixin providing bucketed TP/EP all-gather, HF format conversion, pre-process/post-process
        and the weight updating pipeline.

    Requires the consuming class to set:
        self.args: Namespace with update_weight_buffer_size (as the bucket size).
        self.model: Sequence[torch.nn.Module] (Megatron model chunks).
        self.model_name: str (for HF conversion).
        self.quantization_config: dict | None.
        self._is_source: bool (whether it's the rank broadcasting weights after `all_gather`).
        self.weight_version: int.
        self.rollout_engines: Sequence[ActorHandle]. engines of rollout side.
        self._group_name: str. Identifier shown in the tqdm progress bar.
        self._update_weight_implementation(converted_named_tensors, pbar) -> None
            Transfer a bucket of HF-format ``(name, tensor)`` pairs to rollout
            engines (via NCCL broadcast, p2p write, etc.).
    """

    def _init_lora(
        self,
        *,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        model_name: str,
        quantization_config: dict | None,
        is_lora: bool,
        is_multi_lora: bool = False,
    ) -> None:
        """Initialize LoRA state for distributed weight updaters."""
        self.is_lora = is_lora
        self.is_multi_lora = is_multi_lora
        self._lora_loaded = False
        self._lora_base_synced = False
        self._multi_lora_loaded: set[str] = set()

        if self.is_lora or self.is_multi_lora:
            self._hf_weight_iterator = HfWeightIteratorBase.create(
                args=args,
                model=model,
                model_name=model_name,
                quantization_config=quantization_config,
                is_lora=True,
            )
            if self.is_lora and not self.is_multi_lora:
                self._lora_config = build_lora_sync_config(args)

    def _gather_and_update_non_expert_weights(
        self,
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None = None,
    ) -> None:
        """
        Bucketed TP all-gather + HF conversion for non-expert parameters.
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        After `all_gather`, update weights/buffer_size on source, do nothing on non-source.
        """

        buffer_size = 0
        converted_named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=False):
            if (getattr(self, "is_lora", False) or getattr(self, "is_multi_lora", False)) and _is_adapter_param_name(
                name
            ):
                continue
            name = name.replace(".to_wrap.", ".")
            param = all_gather_param(self.args, name, param)
            if not self._is_source:
                continue

            param_size = param.numel() * param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                update_bucket_weight_func(converted_named_tensors, pbar)
                converted_named_tensors = []
                buffer_size = 0

            converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            buffer_size += param_size

        if converted_named_tensors:
            update_bucket_weight_func(converted_named_tensors, pbar)

    def _gather_and_update_expert_weights(
        self,
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None = None,
    ) -> None:
        """
        Bucketed TP + EP all-gather + HF conversion for expert parameters.
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=True):
            if (getattr(self, "is_lora", False) or getattr(self, "is_multi_lora", False)) and _is_adapter_param_name(
                name
            ):
                continue
            name = name.replace(".to_wrap.", ".")
            param = all_gather_param(self.args, name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size and named_tensors:
                self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)
                named_tensors = []
                buffer_size = 0

            named_tensors.append((name, param))
            buffer_size += param_size

        if named_tensors:
            self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)

    def _update_expert_bucket_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None,
    ) -> None:
        """
        Gather EP → HF → update weights. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names: list[list[str] | None] = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for ep_names in all_names:
            assert len(named_tensors) == len(
                ep_names
            ), f"mismatch names length: {len(named_tensors)} != {len(ep_names)}"

        all_gathered_params: list[list[tuple[str, torch.Tensor]]] = [
            [] for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, ep_names in enumerate(all_names):
                all_gathered_params[ep_rank].append((ep_names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_source:
            return

        flat_gathered = sum(all_gathered_params, [])

        converted_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for name, param in flat_gathered:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        update_bucket_weight_func(converted_hf_tensors, pbar)

    def _pause_and_prepare_engines(self) -> None:
        """Pause rollout engines, flush cache, and run pre-process if needed."""
        if dist.get_rank() == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    rollout_engines=self.rollout_engines,
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                )

    def _finalize_and_resume_engines(self, post_load_weights: bool = False) -> None:
        """Run post-process if needed and resume rollout engines."""
        if dist.get_rank() == 0:
            # post_process_quantization is related to the process_weights_after_loading
            # in the sglang rollout side, which should always be invoked after weight
            # updating.
            post_process_weights(
                rollout_engines=self.rollout_engines,
                restore_weights_before_load=False,
                post_process_quantization=True,
                post_load_weights=post_load_weights,
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    @torch.no_grad()
    def update_weights(self) -> None:
        """Orchestrate the full weight-update lifecycle.

        Non-LoRA: pause → base non-expert (TP) → base expert (EP) → resume.
        LoRA: pause → base weights once → adapter weights every call → resume.
        """
        self.weight_version += 1

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            adapter_enabled = getattr(self, "is_lora", False) or getattr(self, "is_multi_lora", False)
            base_was_synced = getattr(self, "_lora_base_synced", False)
            if not (adapter_enabled and base_was_synced):
                if adapter_enabled:
                    self._sync_base_weights_from_iterator()
                else:
                    pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None

                    self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)
                    dist.barrier(group=get_gloo_group())
                    self._gather_and_update_expert_weights(self._update_weight_implementation, pbar)
                    dist.barrier(group=get_gloo_group())

            if getattr(self, "is_multi_lora", False):
                self._sync_multi_lora_weights()
                dist.barrier(group=get_gloo_group())
            elif getattr(self, "is_lora", False):
                self._sync_lora_weights()
                dist.barrier(group=get_gloo_group())

            if adapter_enabled and not base_was_synced:
                self._lora_base_synced = True

        with timer("finalize_and_resume_engines"):
            self._finalize_and_resume_engines()
            dist.barrier(group=get_gloo_group())

    def _sync_base_weights_from_iterator(self) -> None:
        """Sync frozen base weights through the configured HF iterator.

        In bridge mode this avoids the legacy direct converter, which does not
        understand Qwen3.5-VL vision parameters.
        """
        pbar = tqdm(desc=f"[{self._group_name}] Update base weights", total=0) if self._is_source else None
        megatron_local_weights = self.weights_getter()
        base_ctx = nullcontext()
        if getattr(self, "is_multi_lora", False):
            from megatron.bridge.peft.multi_lora_layers import hide_adapters

            base_ctx = hide_adapters(self.model)

        with base_ctx:
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            ):
                if self._is_source:
                    self._update_weight_implementation(list(hf_named_tensors), pbar)
        dist.barrier(group=get_gloo_group())

    def _sync_lora_weights(self) -> None:
        """Sync a single LoRA adapter to rollout engines via Ray tensor RPCs."""
        all_refs: list[ObjectRef] = []
        lora_chunk_count = 0

        if self._is_source and self._lora_loaded:
            ray.get([engine.unload_lora_adapter.remote(lora_name=LORA_ADAPTER_NAME) for engine in self.rollout_engines])

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks({}, weight_type="lora"):
            weight_tensors = [(n, t) for n, t in hf_named_tensors if is_lora_weight_name(n)]
            if not weight_tensors:
                continue
            lora_chunk_count += 1

            if self._is_source:
                all_refs.extend(
                    self._load_lora_adapter_from_tensors(
                        adapter_name=LORA_ADAPTER_NAME,
                        lora_config=self._lora_config,
                        weight_tensors=weight_tensors,
                    )
                )

        if lora_chunk_count == 0:
            raise RuntimeError(
                "LoRA weight sync failed: the weight iterator produced zero chunks. "
                "No adapter weights were sent to the rollout engine. This usually means "
                "the Megatron-Bridge or SGLang version is incompatible."
            )

        if all_refs:
            _check_weight_sync_results(ray.get(all_refs), is_lora=True)
        if self._is_source:
            self._lora_loaded = True

    def _sync_multi_lora_weights(self) -> None:
        """Sync every active multi-LoRA adapter and unload drained adapters."""
        from miles.ray.multi_lora_controller import get_multi_lora_controller
        from miles.utils.adapter_config import AdapterState

        configs = ray.get(get_multi_lora_controller().adapter_configs.remote())
        for name, config in configs.items():
            if config.state == AdapterState.ACTIVE:
                self._sync_one_multi_lora_adapter(name, config, lora_loaded=name in self._multi_lora_loaded)
                self._multi_lora_loaded.add(name)
            elif config.state == AdapterState.DRAINED and name in self._multi_lora_loaded:
                if self._is_source:
                    ray.get([engine.unload_lora_adapter.remote(lora_name=name) for engine in self.rollout_engines])
                self._multi_lora_loaded.discard(name)

    def _sync_one_multi_lora_adapter(self, adapter_name: str, config, lora_loaded: bool) -> None:
        """Push one named adapter to all rollout engines."""
        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        from ..multi_lora_sync import slice_lora_to_rank

        lora_config = build_lora_sync_config(self.args)
        lora_config["r"] = config.rank
        lora_config["lora_alpha"] = config.alpha

        all_refs: list[ObjectRef] = []
        lora_chunk_count = 0

        if self._is_source and lora_loaded:
            ray.get([engine.unload_lora_adapter.remote(lora_name=adapter_name) for engine in self.rollout_engines])

        with expose_adapter_slot(self.model, config.slot):
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks({}, weight_type="lora"):
                weight_tensors = [
                    (n, slice_lora_to_rank(n, t, config.rank))
                    for n, t in hf_named_tensors
                    if is_lora_weight_name(n)
                ]
                if not weight_tensors:
                    continue
                lora_chunk_count += 1

                if self._is_source:
                    all_refs.extend(
                        self._load_lora_adapter_from_tensors(
                            adapter_name=adapter_name,
                            lora_config=lora_config,
                            weight_tensors=weight_tensors,
                        )
                    )

        if lora_chunk_count == 0:
            raise RuntimeError(
                f"Multi-LoRA weight sync failed for adapter {adapter_name}: "
                "the weight iterator produced zero LoRA chunks."
            )

        if all_refs:
            _check_weight_sync_results(ray.get(all_refs), is_lora=True)

    def _load_lora_adapter_from_tensors(
        self,
        *,
        adapter_name: str,
        lora_config: dict,
        weight_tensors: list[tuple[str, torch.Tensor]],
    ) -> list[ObjectRef]:
        bucket = FlattenedTensorBucket(named_tensors=weight_tensors)
        serialized = MultiprocessingSerializer.serialize(
            {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            },
            output_str=True,
        )
        # The server picks its shard via ``serialized_named_tensors[tp_rank]`` and
        # the LoRA module re-shards internally, so every inference TP rank receives
        # the same full adapter blob.
        serialized_named_tensors = [serialized] * self.args.rollout_num_gpus_per_engine
        return [
            engine.load_lora_adapter_from_tensors.remote(
                lora_name=adapter_name,
                config_dict=lora_config,
                serialized_named_tensors=serialized_named_tensors,
                load_format="flattened_bucket",
            )
            for engine in self.rollout_engines
        ]
