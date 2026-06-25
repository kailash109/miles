"""Continuous-batched MultiLoRA training engine.

Only import-light, torch/ray-free symbols are re-exported here so importing the
package on a CPU-only machine is safe. The GPU pieces (``runner``,
``adapter_pager``, ``continuous_loss``, ``checkpoint_store``,
``artifact_publisher``) import torch/Megatron and must be imported directly from
their submodules inside the training process.
"""

from .scheduler import ContinuousTrainingScheduler
from .schemas import (
    AdapterSpec,
    BudgetSpec,
    DatasetSpec,
    ExternalTrajectoryBatch,
    LossSpec,
    OptimizerSpec,
    SchedulingSpec,
    TrainExample,
    TrainingJobRuntime,
    TrainingJobSpec,
    TrainingJobState,
    validate_trajectory_batch,
)

__all__ = [
    "ContinuousTrainingScheduler",
    "AdapterSpec",
    "BudgetSpec",
    "DatasetSpec",
    "ExternalTrajectoryBatch",
    "LossSpec",
    "OptimizerSpec",
    "SchedulingSpec",
    "TrainExample",
    "TrainingJobRuntime",
    "TrainingJobSpec",
    "TrainingJobState",
    "validate_trajectory_batch",
]
