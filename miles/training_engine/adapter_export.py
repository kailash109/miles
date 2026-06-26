"""HF-PEFT export of a single resident adapter slot (for sglang disk-load).

Reuses the exact Megatron->HF LoRA conversion that ``update_weights`` uses
(``HfWeightIteratorBridge.get_hf_weight_chunks(weight_type="lora")`` +
``slice_lora_to_rank``), so the on-disk adapter matches what sglang's tensor
loader already accepts. Writes a standard PEFT directory
(``adapter_config.json`` + ``adapter_model.safetensors``) atomically.

Runs inside the Megatron training process (torch/Megatron available).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def capture_slot_hf_cpu(hf_iterator, model, slot: int, rank: int) -> dict[str, Any]:
    """Snapshot a slot's LoRA weights as HF-named CPU tensors (must run before the
    slot is cleared)."""
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
    from miles.backends.megatron_utils.update_weight.multi_lora_sync import slice_lora_to_rank

    out: dict[str, Any] = {}
    with expose_adapter_slot(model, slot):
        for chunk in hf_iterator.get_hf_weight_chunks({}, weight_type="lora"):
            for name, tensor in chunk:
                if is_lora_weight_name(name):
                    out[name] = slice_lora_to_rank(name, tensor, rank).detach().to("cpu").contiguous().clone()
    return out


def write_hf_peft_adapter(
    out_dir: str,
    tensors: dict[str, Any],
    *,
    args,
    rank: int,
    alpha: int,
    target_modules: list[str] | tuple[str, ...],
) -> None:
    """Atomically write a PEFT adapter dir (config + safetensors) to ``out_dir``.

    Atomicity: weights are written to a temp file then ``os.replace``d into place,
    so a reader that finds ``adapter_model.safetensors`` always sees a complete file.
    """
    from safetensors.torch import save_file

    from miles.backends.megatron_utils.lora_utils import build_lora_sync_config

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = dict(build_lora_sync_config(args))
    config["r"] = int(rank)
    config["lora_alpha"] = int(alpha)
    if target_modules:
        config["target_modules"] = list(target_modules)
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    # Write weights to a temp file in the same dir, then atomically rename so the
    # final path only appears once fully written.
    fd, tmp_path = tempfile.mkstemp(dir=str(out), suffix=".safetensors.tmp")
    os.close(fd)
    save_file(tensors, tmp_path)
    os.replace(tmp_path, out / "adapter_model.safetensors")
