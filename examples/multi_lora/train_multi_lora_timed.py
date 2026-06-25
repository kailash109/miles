"""Timed multi-LoRA submission test.

Starts a persistent trainer, then submits the adapters in ``--multi-lora-dir``
one at a time on a fixed interval, simulating periodically pushing multi-LoRA
"jobs" to a long-running trainer.

The trainer loop and the submitter run as two coroutines in a single process
(one Ray job), so this fits Modal's one-app-per-container model: a single
``modal run`` launches the whole thing, and the "periodic submission" happens
on an in-process timer rather than via separate app invocations.

The trainer reacts purely to controller state via its existing lifecycle hooks
(``load_pending_adapters``, idle gate, ``unload_drained_adapters``); it has no
knowledge of the submission timer. Before the first submission it simply idles.

Tunables (env vars, easy to override from the launcher's runtime env):
  MULTI_LORA_SUBMIT_INITIAL_S   delay before the first submission (default 30)
  MULTI_LORA_SUBMIT_INTERVAL_S  seconds between submissions      (default 60)

The trainer loop runs indefinitely (same as the dynamic driver); stop the Modal
app when the test is done.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# examples/ is not a package; ensure the sibling driver is importable whether or
# not Python has already put this file's directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from miles.ray.multi_lora_controller import create_multi_lora_controller
from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.tracking_utils import init_tracking

from train_multi_lora_dynamic import run_trainer

logger = logging.getLogger(__name__)

SUBMIT_INITIAL_S = float(os.environ.get("MULTI_LORA_SUBMIT_INITIAL_S", "30"))
SUBMIT_INTERVAL_S = float(os.environ.get("MULTI_LORA_SUBMIT_INTERVAL_S", "60"))


def _discover_adapter_dirs(multi_lora_dir: Path) -> list[Path]:
    """Direct children of ``multi_lora_dir`` that hold an adapter.yaml, sorted."""
    return sorted(d for d in multi_lora_dir.iterdir() if (d / "adapter.yaml").exists())


def _banner(msg: str) -> None:
    """Print a high-visibility line to stdout so it stands out in Modal logs."""
    line = "=" * 78
    print(f"\n{line}\n[MULTI-LORA SUBMIT] {msg}\n{line}", flush=True)


async def run_schedule(controller, multi_lora_dir: Path) -> None:
    """Register adapters one at a time on a fixed interval."""
    adapter_dirs = _discover_adapter_dirs(multi_lora_dir)
    total = len(adapter_dirs)
    _banner(
        f"{total} adapters queued from {multi_lora_dir} | "
        f"initial delay {SUBMIT_INITIAL_S}s, interval {SUBMIT_INTERVAL_S}s"
    )

    await asyncio.sleep(SUBMIT_INITIAL_S)

    for i, adapter_dir in enumerate(adapter_dirs):
        info = await controller.register_adapter.remote(str(adapter_dir))
        slot = info.get("slot") if isinstance(info, dict) else None
        _banner(f"({i + 1}/{total}) registered '{adapter_dir.name}' -> slot {slot} (PENDING)")
        if i < total - 1:
            print(f"[MULTI-LORA SUBMIT] next adapter in {SUBMIT_INTERVAL_S}s", flush=True)
            await asyncio.sleep(SUBMIT_INTERVAL_S)

    _banner(f"all {total} adapters submitted; trainer continues")


async def main(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    # No startup registration — the schedule task drives all submissions.
    controller = create_multi_lora_controller(args.multi_lora_n_adapters, args.lora_rank)
    args.data_source_path = "miles.rollout.multi_lora_data_source.MultiLoRADataSource"
    args.custom_generate_state_path = "miles.ray.multi_lora_controller.MultiLoRAGenerateState"

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    shared_state = [args.start_rollout_id]

    await asyncio.gather(
        run_trainer(args, controller, rollout_manager, actor_model, num_rollout_per_epoch, shared_state),
        run_schedule(controller, Path(args.multi_lora_dir)),
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
