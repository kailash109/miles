"""Ray-job driver for a real GPU run of the continuous MultiLoRA engine.

Mirrors ``examples/multi_lora/train_multi_lora.py``'s bootstrap but train-only:
it allocates a Miles train actor (which builds the real Megatron MultiLoRA model
via ``initialize_multi_lora_model_and_optimizer``), then runs the continuous
training engine inside that actor over a couple of synthetic SFT jobs and
prints the resulting adapter-artifact versions.

Launched by ``run_engine.sh`` via ``ray job submit``. SFT-only, external-rollout
mode (no SGLang here).
"""

from __future__ import annotations

import asyncio
import json
import os

from miles.ray.placement_group import allocate_train_group, create_placement_groups
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger


def _synthetic_records(name: str, n: int = 32) -> list[dict]:
    return [
        {"prompt": f"{name} question {i}: what is {i} + {i}?", "completion": f" The answer is {2 * i}."}
        for i in range(n)
    ]


def _build_engine_cfg(output_root: str, target_modules: list[str]) -> dict:
    return {
        "jobs": [
            {
                "name": "alpha",
                "rank": 8,
                "alpha": 16,
                "target_modules": target_modules,
                "output_uri": f"{output_root}/alpha",
                "max_steps": 4,
                "records": _synthetic_records("alpha"),
            },
            {
                "name": "beta",
                "rank": 8,
                "alpha": 16,
                "target_modules": target_modules,
                "output_uri": f"{output_root}/beta",
                "max_steps": 4,
                "records": _synthetic_records("beta"),
            },
        ],
        "scheduler": {
            "train_tokens_per_step": 4096,
            "max_adapters_per_step": 2,
            "base_quantum_tokens": 1024,
        },
        "tokens_per_update": 256,
        "max_engine_steps": 200,
    }


async def main(args) -> None:
    configure_logger()
    pgs = create_placement_groups(args)

    # The Miles multi-LoRA model build calls load_pending_adapters(), which
    # looks up the legacy hot-slot controller actor. Create it (empty) so model
    # init succeeds; the continuous engine manages slots itself afterward.
    #
    from miles.ray.multi_lora_controller import create_multi_lora_controller

    controller = create_multi_lora_controller(  # noqa: F841 (kept alive intentionally)
        args.multi_lora_n_adapters, args.lora_rank
    )

    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="actor",
        with_ref=False,
    )
    # Builds the real Megatron MultiLoRA model + optimizer on each rank.
    await actor_model.init()

    output_root = os.environ.get("ENGINE_OUTPUT_ROOT", "/root/engine_artifacts")
    target_modules = (
        list(args.target_modules)
        if isinstance(args.target_modules, (list, tuple))
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    engine_cfg = _build_engine_cfg(output_root, target_modules)

    print("[engine] submitting jobs and running continuous SFT engine...", flush=True)
    summary = await actor_model.run_continuous_sft_engine(engine_cfg)
    print("[engine] done. job summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
