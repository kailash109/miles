"""CLI/arg additions for the continuous training engine (plan §20).

Call ``add_training_engine_arguments(parser)`` from the Miles argument layer to
wire these in. Kept separate so it has no import-time dependency on the rest of
the engine.

In continuous-engine mode, ``--multi-lora-n-adapters`` means *max hot GPU
slots*, not max logical jobs.
"""

from __future__ import annotations

import argparse


def add_training_engine_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training engine (continuous MultiLoRA)")
    group.add_argument(
        "--training-engine",
        action="store_true",
        default=False,
        help="Enable the long-lived continuous MultiLoRA training engine.",
    )
    group.add_argument(
        "--training-engine-mode",
        type=str,
        default="external_rollout",
        choices=["external_rollout", "colocated"],
        help="external_rollout (default) consumes externally generated trajectories; "
        "colocated launches/pushes to local SGLang (Phase 5, optional).",
    )
    group.add_argument(
        "--continuous-train-token-budget",
        type=int,
        default=32768,
        help="Max trained tokens per engine step across all selected adapters.",
    )
    group.add_argument(
        "--continuous-max-adapters-per-step",
        type=int,
        default=8,
        help="Max distinct adapters packed into one training step.",
    )
    group.add_argument(
        "--continuous-base-quantum-tokens",
        type=int,
        default=8192,
        help="Scheduling credit (deficit) granted per runnable job per round.",
    )
    group.add_argument(
        "--continuous-idle-sleep-s",
        type=float,
        default=0.01,
        help="Sleep when no job is runnable.",
    )
    group.add_argument(
        "--continuous-enable-preemption",
        action="store_true",
        default=True,
        help="Allow evicting hot adapters to make room (Phase 3).",
    )
    group.add_argument(
        "--continuous-checkpoint-store",
        type=str,
        default=None,
        help="Base URI for internal preemption/resume checkpoints.",
    )
    group.add_argument(
        "--continuous-artifact-store",
        type=str,
        default=None,
        help="Base URI for published adapter artifacts.",
    )
    group.add_argument("--continuous-api-host", type=str, default="0.0.0.0")
    group.add_argument("--continuous-api-port", type=int, default=8080)
