"""Continuous-batched MultiLoRA training engine (central-coordinator design).

Only import-light, torch/ray-free symbols are re-exported here so importing the
package on a CPU-only machine is safe. The worker-side GPU modules
(``megatron_executor``, ``adapter_slot_executor``, ``batch_materializer``,
``continuous_loss``, ``checkpoint_store``) import torch/Megatron and must be
imported directly from their submodules inside the training process.
"""

from .batch_store import BatchStore
from .client import LoraTrainingClient, ServiceClient, TrainingClient, build_job_spec
from .coordinator import TrainingCoordinator, make_training_coordinator
from .plan import BatchLease, ExportRequest, SlotOnload, SlotPreemption, TrainStepPlan
from .results import (
    CommitResult,
    CreateJobResponse,
    SubmitBatchResponse,
    WorkerHealth,
    WorkerStepResult,
)
from .scheduler import ContinuousTrainingScheduler
from .schemas import (
    AdapterSpec,
    BatchingPolicy,
    BudgetSpec,
    DatasetSpec,
    Execution,
    ExternalTrajectoryBatch,
    Lifecycle,
    LossSpec,
    OptimizerSpec,
    QueueLimits,
    Readiness,
    Residency,
    SchedulingSpec,
    SelectedJob,
    TrainExample,
    TrainingJobRuntime,
    TrainingJobSpec,
    validate_trajectory_batch,
)

__all__ = [
    "BatchStore",
    "TrainingClient",
    "ServiceClient",
    "LoraTrainingClient",
    "build_job_spec",
    "TrainingCoordinator",
    "make_training_coordinator",
    "TrainStepPlan",
    "SlotOnload",
    "SlotPreemption",
    "BatchLease",
    "ExportRequest",
    "CommitResult",
    "CreateJobResponse",
    "SubmitBatchResponse",
    "WorkerHealth",
    "WorkerStepResult",
    "ContinuousTrainingScheduler",
    "AdapterSpec",
    "BatchingPolicy",
    "BudgetSpec",
    "DatasetSpec",
    "Execution",
    "ExternalTrajectoryBatch",
    "Lifecycle",
    "LossSpec",
    "OptimizerSpec",
    "QueueLimits",
    "Readiness",
    "Residency",
    "SchedulingSpec",
    "SelectedJob",
    "TrainExample",
    "TrainingJobRuntime",
    "TrainingJobSpec",
    "validate_trajectory_batch",
]
