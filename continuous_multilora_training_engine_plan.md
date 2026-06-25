# Continuous-Batched MultiLoRA Training Engine on Osmosis Miles + Megatron-Bridge

**Audience:** coding agent with local access to the Osmosis `miles` repository and the Osmosis `Megatron-Bridge` fork.

**Goal:** implement a long-lived, continuous-batched MultiLoRA training engine on top of the existing Osmosis Miles/Megatron-Bridge MultiLoRA stack. The engine should accept training jobs at arbitrary times, schedule them fairly over a fixed number of GPU-resident LoRA slots, train adapters with the existing Megatron forward/backward infrastructure, and return versioned LoRA adapter artifacts. The design should be compatible with a future fully disaggregated setup where inference/rollout is external and the training engine consumes externally generated trajectories and old logprobs.

**Primary target mode:** external-rollout/disaggregated training.

In this mode, the training engine does **not** own SGLang/vLLM, does **not** launch rollouts, and does **not** push weights into inference engines. It accepts SFT examples or RL trajectory batches generated elsewhere, trains the appropriate LoRA adapter, and publishes immutable adapter artifacts. A colocated-rollout plugin can be added later, but should not be the core architecture.

---

## 0. Current codebase facts to verify before editing

The plan below assumes the following branch layout and current implementation patterns. The coding agent should verify locally because these forks may move.

### Miles repository

Expected branch/example path:

```text
Osmosis-AI/miles@multilora_no_colocate
```

Important files:

```text
miles/backends/megatron_utils/multi_lora.py
miles/backends/megatron_utils/model.py
miles/backends/megatron_utils/lora_utils.py
miles/backends/megatron_utils/update_weight/multi_lora_sync.py
miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py
miles/backends/sglang_utils/sglang_engine.py
miles/ray/multi_lora_controller.py
miles/utils/adapter_config.py
```

Current useful behavior:

1. `miles/backends/megatron_utils/multi_lora.py` builds the Megatron model with a Megatron-Bridge `MultiLoRA` transform via a pre-wrap hook. It creates `MultiLoRA(target_modules=..., n_adapters=args.multi_lora_n_adapters, dim=args.lora_rank, alpha=args.lora_alpha, ...)`, then calls `provider.register_pre_wrap_hook(apply_hook)`. This is the correct place to keep using existing Megatron initialization.
2. The same file has `initialize_multi_lora_model_and_optimizer(...)`, which initializes the model, optimizer, checkpoint loading, hides adapters during base checkpoint load, and calls `load_pending_adapters(...)` after optimizer creation.
3. The same file has `zero_optimizer_state_for_adapter(optimizer, model, idx)`, which walks `MultiLoRALinear` modules, finds parameters for adapter slot `idx`, resolves Megatron `main_param` if present, and zeroes Adam state such as `exp_avg` and `exp_avg_sq`. This should be generalized into capture/restore functions for preemption/offload.
4. `miles/ray/multi_lora_controller.py` is currently a fixed-slot Ray actor with lifecycle states like `PENDING -> ACTIVE -> DRAINING -> DRAINED -> REMOVED`. It uses a finite `free_slots` pool. It is a hot-slot registry, not a durable multi-user job controller.
5. `miles/backends/megatron_utils/update_weight/multi_lora_sync.py` already has logic for loading pending adapters, saving per-adapter checkpoints, clearing slots, and zeroing optimizer state for drained adapters. Factor this into reusable slot install/save/clear helpers.
6. `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` currently pushes active MultiLoRA adapters into colocated SGLang via `send_one_multi_lora_adapter(...)`. In the new external-rollout mode, this path must be optional and disabled by default.
7. `miles/backends/sglang_utils/sglang_engine.py` can initialize external or internal SGLang engines and configures LoRA serving when MultiLoRA is enabled. The continuous training engine should not depend on this in external-rollout mode.

### Megatron-Bridge fork

Expected branch/example path:

```text
Osmosis-AI/Megatron-Bridge@multilora_may25
```

Important files:

```text
src/megatron/bridge/peft/multi_lora.py
src/megatron/bridge/peft/multi_lora_layers.py
```

Current useful behavior:

1. `MultiLoRALinear` wraps a Megatron parallel linear with `n_adapters` adapter slots stored in `self.adapters = nn.ModuleList([...])`.
2. Every wrapped layer expects `tokens_per_adapter` to be set before forward. This tensor has length `n_adapters`; entry `i` is the number of **contiguous flattened tokens** routed to adapter slot `i` in the upcoming microbatch. The batch must be sorted by physical adapter slot.
3. Forward path computes base linear output, flattens tokens, creates `offsets = tokens_per_adapter.cumsum(...)`, stacks adapter `A/B` weights, runs `torch._grouped_mm` for `x @ A`, performs the TP all-reduce/all-gather depending on whether the wrapped base layer is row/column parallel, runs `torch._grouped_mm` for the `B` projection, applies `alpha / rank` per token, matches output layout, and adds the LoRA delta to the base output.
4. Important helpers already exist:

```python
set_tokens_per_adapter_slot(model, tokens_per_adapter)
init_adapter_slot(model, idx, rank, alpha)
clear_adapter_slot(model, idx)
load_adapter(model, idx, state_dict)
expose_adapter_slot(model, idx)
hide_adapters(model)
```

Do **not** rewrite this path in the first implementation. The continuous training engine should feed it correctly.

---

## 1. Product/API target

The user-facing API should resemble a training service rather than a launched monolithic experiment.

### SFT-style request

```python
from miles.training_engine.client import TrainingClient

client = TrainingClient("http://trainer-host:8080")

job = client.create_lora_training_job(
    base_model="qwen3.6-35b-a3b",
    adapter={
        "name": "math-sft-v1",
        "rank": 16,
        "alpha": 32,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "init": "random",  # or {"adapter_uri": "..."}
    },
    dataset={
        "uri": "s3://bucket/math_sft.jsonl",
        "format": "prompt_completion_jsonl",
        "prompt_key": "prompt",
        "completion_key": "completion",
    },
    loss={"type": "sft"},
    optimizer={"type": "adamw", "lr": 2e-4, "weight_decay": 0.0},
    budget={"max_steps": 500, "tokens_per_update": 8192},
    output_uri="s3://bucket/adapters/math-sft-v1/",
)

result = job.wait()
print(result.adapter_uri)
```

### External-rollout RL request

```python
job = client.create_lora_training_job(
    base_model="qwen3.6-35b-a3b",
    adapter={
        "name": "agent-grpo-v1",
        "rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "init": "random",
    },
    loss={
        "type": "grpo",
        "clip_eps": 0.2,
        "kl_coef": 0.01,
        "kl_ref": "base",
        "advantage_source": "trainer_from_group_rewards",
    },
    optimizer={"type": "adamw", "lr": 1e-4},
    budget={
        "max_steps": 300,
        "tokens_per_update": 8192,
        "max_policy_lag": 4,
    },
    output_uri="s3://bucket/adapters/agent-grpo-v1/",
)

# The response returns the initial adapter version and artifact URI.
initial = job.initial_adapter()
external_inference.load_lora(name=job.id, uri=initial.adapter_uri)

# External inference/rollout system generates completions + old logprobs.
client.submit_trajectory_batch(
    job_id=job.id,
    adapter_version=initial.version,
    batch={
        "input_ids": ...,        # prompt + completion tokens
        "attention_mask": ...,
        "action_mask": ...,     # generated/action tokens to train on
        "old_logprobs": ...,    # behavior-policy logprobs from external inference
        "rewards": ...,
        "group_ids": ...,       # for GRPO
        "ref_logprobs": None,   # optional; trainer can compute base ref
        "metadata": {...},
    },
)

next_adapter = job.wait_for_adapter_version(after_version=initial.version)
external_inference.load_lora(name=job.id, uri=next_adapter.adapter_uri)
```

The training engine should work whether the external inference system is SGLang, vLLM, Modal endpoints, or user-owned infra. The contract is adapter artifacts and trajectory batches labeled with the adapter version that produced them.

---

## 2. Architectural model

### Core principle

Independent user jobs should be independent in the **control plane**, but fused in the **GPU training process**.

```text
External clients / rollout systems
        │
        ▼
Training API + durable job controller
        │
        ▼
Continuous scheduler + adapter pager
        │
        ▼
Existing Miles/Megatron model + MultiLoRA slots
        │
        ▼
Versioned adapter artifact registry
```

### Separation of identities

```text
Logical job id:
  durable user-facing identity, e.g. job_abc

Physical adapter slot:
  GPU-resident slot index in MultiLoRALinear, e.g. slot 3

Adapter artifact version:
  immutable exported adapter checkpoint, e.g. job_abc/v000012
```

A logical job may move between physical slots over time:

```text
job_abc -> slot 3 -> preempted -> cold checkpoint -> slot 7 -> completed
```

All externally generated RL data must carry the logical `job_id` and the `adapter_version` used to generate it. The training engine maps `job_id` to the current hot slot only at training time.

---

## 3. Continuous batching semantics

This is analogous to vLLM/SGLang continuous batching, but the safe scheduling unit is not prefill/decode. It is a **training quantum**.

A training quantum is:

```text
1. select runnable jobs
2. make selected adapters hot in GPU slots
3. pull examples/trajectories for selected jobs
4. pack examples by physical slot
5. set adapter_token_counts
6. run Megatron forward/backward over one training step
7. optimizer step selected adapters
8. export/publish dirty adapter versions
9. optionally checkpoint/preempt/complete jobs
```

Preemption is only safe between training quanta. Do not preempt in the middle of a forward/backward/pipeline-parallel step.

### Batch invariant

The existing `MultiLoRALinear` requires flattened tokens to be sorted by adapter slot:

```text
[ tokens for slot 0 | tokens for slot 1 | tokens for slot 2 | ... ]
```

Then pass:

```python
adapter_token_counts = torch.tensor([
    n_tokens_for_slot_0,
    n_tokens_for_slot_1,
    n_tokens_for_slot_2,
    ...,
], dtype=torch.int32, device="cuda")

set_tokens_per_adapter_slot(model, adapter_token_counts)
```

Never pass arbitrary unsorted per-token adapter ids into the current model. If a coding agent wants unsorted routing later, that requires a Megatron-Bridge layer change and likely a gather/sort/scatter path. Do not do it in MVP.

---

## 4. New package layout

Add a new package in Miles:

```text
miles/training_engine/
  __init__.py
  schemas.py
  client.py
  api_server.py
  job_controller.py
  scheduler.py
  adapter_pager.py
  trajectory_store.py
  dataset_worker.py
  continuous_packer.py
  continuous_loss.py
  runner.py
  checkpoint_store.py
  artifact_publisher.py
  events.py
  metrics.py
```

The first implementation can keep everything in-process/Ray actors and expose HTTP later, but define the schemas as if API boundaries already exist.

---

## 5. Schemas

Create `miles/training_engine/schemas.py`.

Use Pydantic if Miles already uses it for API boundaries; otherwise dataclasses are fine internally. Prefer explicit schemas because external rollout requires strict validation.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class TrainingJobState(str, Enum):
    QUEUED = "queued"
    WAITING_FOR_DATA = "waiting_for_data"
    TRAIN_READY = "train_ready"
    LOADING = "loading"
    HOT_IDLE = "hot_idle"
    ACTIVE_STEP = "active_step"
    PREEMPTING = "preempting"
    COLD_READY = "cold_ready"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    rank: int
    alpha: int
    target_modules: list[str]
    init: Literal["random", "adapter_uri"] = "random"
    adapter_uri: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    uri: str | None = None
    format: Literal[
        "prompt_completion_jsonl",
        "prompt_jsonl",
        "external_trajectory_batches",
    ] = "external_trajectory_batches"
    prompt_key: str = "prompt"
    completion_key: str | None = "completion"


@dataclass(frozen=True)
class LossSpec:
    type: Literal["sft", "dpo", "grpo", "ppo"]
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    kl_ref: Literal["base", "provided", "none"] = "base"
    advantage_source: Literal[
        "provided",
        "trainer_from_rewards",
        "trainer_from_group_rewards",
    ] = "provided"


@dataclass(frozen=True)
class OptimizerSpec:
    type: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class BudgetSpec:
    max_steps: int | None = None
    max_train_tokens: int | None = None
    tokens_per_update: int = 8192
    checkpoint_every_steps: int = 25
    publish_every_steps: int = 1
    max_policy_lag: int = 4


@dataclass(frozen=True)
class SchedulingSpec:
    priority: int = 1
    preemptible: bool = True
    min_hot_steps: int = 2
    max_consecutive_steps: int = 8
    min_tokens_per_train_quantum: int = 2048


@dataclass(frozen=True)
class TrainingJobSpec:
    job_id: str
    user_id: str
    base_model: str
    base_model_revision: str | None
    adapter: AdapterSpec
    dataset: DatasetSpec
    loss: LossSpec
    optimizer: OptimizerSpec
    budget: BudgetSpec
    scheduling: SchedulingSpec
    output_uri: str


@dataclass
class TrainingJobRuntime:
    spec: TrainingJobSpec
    state: TrainingJobState = TrainingJobState.QUEUED

    # Physical hot slot; None when cold/not loaded.
    slot: int | None = None
    hot_since_engine_step: int | None = None

    # Progress.
    trained_steps: int = 0
    trained_tokens: int = 0
    current_adapter_version: int = 0
    latest_adapter_uri: str | None = None

    # Queue/accounting.
    ready_train_tokens: int = 0
    ready_batch_ids: list[str] = field(default_factory=list)
    deficit_tokens: int = 0
    consecutive_steps: int = 0

    # Checkpoint/offload.
    cold_checkpoint_uri: str | None = None
    dirty_since_publish: bool = False

    # Error/reporting.
    last_error: str | None = None
```

### External trajectory schema

The external rollout system submits tokenized data. The training engine should not rely on reconstructing chat templates from raw text for RL.

```python
@dataclass
class ExternalTrajectoryBatch:
    job_id: str
    adapter_version: int

    # Compatibility fields. Store hashes at job creation and validate here.
    base_model_hash: str | None
    tokenizer_hash: str | None
    lora_config_hash: str | None

    # Shape: [num_sequences, seq_len]. Can also be serialized as ragged lists.
    input_ids: Any
    attention_mask: Any
    action_mask: Any

    # Behavior-policy logprobs from external inference.
    old_logprobs: Any

    # RL supervision.
    rewards: Any | None = None
    advantages: Any | None = None
    returns: Any | None = None
    group_ids: list[str] | None = None
    ref_logprobs: Any | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
```

Validation rules:

```python
assert batch.adapter_version <= job.current_adapter_version
assert job.current_adapter_version - batch.adapter_version <= job.spec.budget.max_policy_lag
assert shape(input_ids) == shape(attention_mask) == shape(action_mask)
assert shape(old_logprobs) == shape(action_mask)
assert action_mask.sum() > 0
```

---

## 6. Job controller

Create `miles/training_engine/job_controller.py` as a Ray actor.

Do **not** replace `miles/ray/multi_lora_controller.py` initially. The new controller owns logical jobs. The existing controller can remain a hot-slot adapter registry for compatibility with existing MultiLoRA flows.

```python
import ray
from .schemas import TrainingJobRuntime, TrainingJobSpec, TrainingJobState


@ray.remote(num_cpus=0)
class TrainingJobController:
    def __init__(self, base_model: str, max_hot_slots: int):
        self.base_model = base_model
        self.max_hot_slots = max_hot_slots
        self.jobs: dict[str, TrainingJobRuntime] = {}
        self.hot_job_by_slot: dict[int, str] = {}
        self.free_slots: set[int] = set(range(max_hot_slots))
        self.engine_step: int = 0

    def submit_job(self, spec: TrainingJobSpec) -> str:
        self._validate_new_job(spec)
        if spec.job_id in self.jobs:
            raise ValueError(f"duplicate job_id: {spec.job_id}")
        rt = TrainingJobRuntime(spec=spec)
        rt.state = TrainingJobState.WAITING_FOR_DATA
        self.jobs[spec.job_id] = rt
        return spec.job_id

    def get_job(self, job_id: str) -> TrainingJobRuntime:
        return self.jobs[job_id]

    def snapshot_jobs(self) -> dict[str, TrainingJobRuntime]:
        # For MVP, returning dataclasses is fine. For large scale, return compact views.
        return dict(self.jobs)

    def mark_train_batch_ready(self, job_id: str, batch_id: str, token_count: int) -> None:
        rt = self.jobs[job_id]
        rt.ready_batch_ids.append(batch_id)
        rt.ready_train_tokens += token_count
        if rt.state in {TrainingJobState.WAITING_FOR_DATA, TrainingJobState.COLD_READY, TrainingJobState.HOT_IDLE}:
            rt.state = TrainingJobState.TRAIN_READY

    def reserve_free_slot(self, job_id: str) -> int | None:
        if not self.free_slots:
            return None
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.hot_job_by_slot[slot] = job_id
        rt = self.jobs[job_id]
        rt.slot = slot
        rt.state = TrainingJobState.LOADING
        return slot

    def mark_hot(self, job_id: str, slot: int) -> None:
        rt = self.jobs[job_id]
        rt.slot = slot
        rt.hot_since_engine_step = self.engine_step
        rt.state = TrainingJobState.HOT_IDLE

    def mark_active_step(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            self.jobs[job_id].state = TrainingJobState.ACTIVE_STEP

    def commit_step(self, updates: dict[str, dict]) -> None:
        self.engine_step += 1
        for job_id, u in updates.items():
            rt = self.jobs[job_id]
            rt.trained_steps += 1
            rt.trained_tokens += int(u.get("trained_tokens", 0))
            rt.current_adapter_version += 1
            rt.ready_train_tokens = max(0, rt.ready_train_tokens - int(u.get("trained_tokens", 0)))
            rt.dirty_since_publish = True
            rt.consecutive_steps += 1
            rt.deficit_tokens -= int(u.get("trained_tokens", 0))
            rt.state = TrainingJobState.HOT_IDLE if rt.ready_train_tokens > 0 else TrainingJobState.WAITING_FOR_DATA

    def mark_cold(self, job_id: str, checkpoint_uri: str) -> None:
        rt = self.jobs[job_id]
        if rt.slot is not None:
            self.hot_job_by_slot.pop(rt.slot, None)
            self.free_slots.add(rt.slot)
        rt.slot = None
        rt.cold_checkpoint_uri = checkpoint_uri
        rt.hot_since_engine_step = None
        rt.consecutive_steps = 0
        if rt.ready_train_tokens > 0:
            rt.state = TrainingJobState.TRAIN_READY
        else:
            rt.state = TrainingJobState.COLD_READY

    def mark_published(self, job_id: str, adapter_uri: str) -> None:
        rt = self.jobs[job_id]
        rt.latest_adapter_uri = adapter_uri
        rt.dirty_since_publish = False

    def complete_job(self, job_id: str, adapter_uri: str) -> None:
        rt = self.jobs[job_id]
        rt.latest_adapter_uri = adapter_uri
        rt.state = TrainingJobState.COMPLETED

    def fail_job(self, job_id: str, error: str) -> None:
        rt = self.jobs[job_id]
        rt.state = TrainingJobState.FAILED
        rt.last_error = error

    def _validate_new_job(self, spec: TrainingJobSpec) -> None:
        if spec.base_model != self.base_model:
            raise ValueError(f"job base_model {spec.base_model} != engine base_model {self.base_model}")
        if spec.adapter.rank <= 0:
            raise ValueError("adapter rank must be positive")
```

Notes:

1. The controller must not directly touch Megatron tensors.
2. All state transitions that involve GPU operations are orchestrated by the runner/pager, then reflected back into the controller.
3. Durable persistence can be added later by writing `TrainingJobRuntime` snapshots to a DB/object store.

---

## 7. Trajectory/data store

Create `miles/training_engine/trajectory_store.py`.

For MVP, store batches in memory or local disk. For production, store in object store and keep only metadata in memory.

```python
class TrajectoryStore:
    def __init__(self):
        self._batches: dict[str, ExternalTrajectoryBatch] = {}
        self._by_job: dict[str, list[str]] = {}

    def put(self, batch: ExternalTrajectoryBatch) -> tuple[str, int]:
        batch_id = make_batch_id(batch.job_id)
        token_count = count_action_tokens(batch.action_mask)
        self._batches[batch_id] = batch
        self._by_job.setdefault(batch.job_id, []).append(batch_id)
        return batch_id, token_count

    def pop_for_job(self, job_id: str, target_tokens: int) -> list[ExternalTrajectoryBatch]:
        out = []
        tokens = 0
        ids = self._by_job.get(job_id, [])
        kept = []
        for batch_id in ids:
            if tokens >= target_tokens:
                kept.append(batch_id)
                continue
            batch = self._batches.pop(batch_id)
            out.append(batch)
            tokens += count_action_tokens(batch.action_mask)
        self._by_job[job_id] = kept
        return out
```

Add an ingestion API endpoint later:

```python
@app.post("/v1/jobs/{job_id}/trajectory_batches")
async def submit_trajectory_batch(job_id: str, batch: ExternalTrajectoryBatch):
    job = await controller.get_job.remote(job_id)
    validate_trajectory_batch(job, batch)
    batch_id, token_count = trajectory_store.put(batch)
    await controller.mark_train_batch_ready.remote(job_id, batch_id, token_count)
    return {"batch_id": batch_id, "accepted": True, "token_count": token_count}
```

---

## 8. Dataset worker for SFT

For SFT jobs with dataset URIs, create `miles/training_engine/dataset_worker.py`.

The standardized MVP dataset format:

```json
{"prompt": "...", "completion": "..."}
```

Worker output should be converted into the same internal `TrainExample` shape used by external trajectories, so the packer does not care whether data came from SFT JSONL or external RL rollouts.

```python
@dataclass
class TrainExample:
    job_id: str
    slot: int | None
    loss_type: Literal["sft", "grpo", "ppo", "dpo"]
    adapter_version: int | None

    input_ids: list[int]
    attention_mask: list[int]
    loss_mask: list[int]

    # SFT.
    labels: list[int] | None = None

    # RL.
    old_logprobs: list[float] | None = None
    rewards: list[float] | None = None
    advantages: list[float] | None = None
    returns: list[float] | None = None
    ref_logprobs: list[float] | None = None
    group_id: str | None = None
```

---

## 9. Continuous scheduler

Create `miles/training_engine/scheduler.py`.

Use deficit-based weighted fair scheduling. Credit accrues to runnable jobs; spending happens in trained tokens. This prevents one job with a huge backlog from monopolizing the engine.

```python
class ContinuousTrainingScheduler:
    def __init__(
        self,
        *,
        train_tokens_per_step: int,
        max_adapters_per_step: int,
        base_quantum_tokens: int,
    ):
        self.train_tokens_per_step = train_tokens_per_step
        self.max_adapters_per_step = max_adapters_per_step
        self.base_quantum_tokens = base_quantum_tokens

    def accrue_deficits(self, jobs: dict[str, TrainingJobRuntime]) -> None:
        for job in jobs.values():
            if job.state in {
                TrainingJobState.TRAIN_READY,
                TrainingJobState.HOT_IDLE,
                TrainingJobState.COLD_READY,
            } and job.ready_train_tokens > 0:
                job.deficit_tokens += self.base_quantum_tokens * job.spec.scheduling.priority

    def select_training_jobs(
        self,
        jobs: dict[str, TrainingJobRuntime],
    ) -> list[tuple[str, int]]:
        candidates = []
        for job_id, job in jobs.items():
            if job.state in {TrainingJobState.FAILED, TrainingJobState.CANCELLED, TrainingJobState.COMPLETED}:
                continue
            if job.ready_train_tokens < job.spec.scheduling.min_tokens_per_train_quantum:
                continue
            if job.deficit_tokens < job.spec.scheduling.min_tokens_per_train_quantum:
                continue
            if job.state == TrainingJobState.ACTIVE_STEP:
                continue
            candidates.append(job)

        # Higher deficit first; prefer already-hot jobs when fairness is close; prevent hogging.
        candidates.sort(
            key=lambda j: (
                -j.deficit_tokens,
                0 if j.slot is not None else 1,
                j.consecutive_steps,
                -j.spec.scheduling.priority,
            )
        )

        selected: list[tuple[str, int]] = []
        remaining = self.train_tokens_per_step
        for job in candidates:
            if len(selected) >= self.max_adapters_per_step:
                break
            if job.consecutive_steps >= job.spec.scheduling.max_consecutive_steps:
                continue
            want = min(
                remaining,
                job.spec.budget.tokens_per_update,
                job.ready_train_tokens,
                job.deficit_tokens,
            )
            if want < job.spec.scheduling.min_tokens_per_train_quantum:
                continue
            selected.append((job.spec.job_id, int(want)))
            remaining -= int(want)
            if remaining <= 0:
                break
        return selected

    def choose_preemption_victim(
        self,
        hot_jobs: list[TrainingJobRuntime],
        incoming_job: TrainingJobRuntime,
        engine_step: int,
    ) -> TrainingJobRuntime | None:
        candidates = []
        for job in hot_jobs:
            if not job.spec.scheduling.preemptible:
                continue
            if job.state == TrainingJobState.ACTIVE_STEP:
                continue
            if job.hot_since_engine_step is not None:
                hot_steps = engine_step - job.hot_since_engine_step
                if hot_steps < job.spec.scheduling.min_hot_steps:
                    continue
            candidates.append(job)

        if not candidates:
            return None

        # Prefer evicting idle / over-served / long-consecutive jobs.
        candidates.sort(
            key=lambda j: (
                j.ready_train_tokens > 0,
                j.deficit_tokens,
                -j.consecutive_steps,
                j.spec.scheduling.priority,
            )
        )
        return candidates[0]
```

Scheduler constraints:

1. Only schedule jobs with ready training data.
2. Do not keep jobs hot just because external rollout is slow.
3. Enforce `max_consecutive_steps` to avoid hogging.
4. Enforce `min_hot_steps` to avoid preemption thrash.
5. Select a bounded number of adapters per step to avoid tiny per-adapter batches and poor grouped-mm efficiency.

---

## 10. Adapter pager

Create `miles/training_engine/adapter_pager.py`.

The pager maps many logical jobs onto fewer physical MultiLoRA slots.

### Responsibilities

```text
ensure_hot(job_id) -> slot
preempt(job_id) -> save adapter + optimizer + scheduler state, clear slot
load(job_id, slot) -> initialize slot, load adapter weights, restore optimizer state
complete(job_id) -> final export, clear slot
```

### Required helper changes in existing Miles code

In `miles/backends/megatron_utils/multi_lora.py`, add stable parameter iteration and optimizer state capture/restore.

Current `zero_optimizer_state_for_adapter(...)` is a useful starting point, but it only zeroes state. Add:

```python
def iter_adapter_named_params_for_slot(model, idx: int):
    """Yield stable names and parameter objects for adapter slot idx."""
    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, _iter_multi_lora_modules

    for module_prefix, module in iter_named_multi_lora_modules(model):
        if not isinstance(module, MultiLoRALinear):
            continue
        adapter = module.adapters[idx]
        for name, param in adapter.named_parameters():
            yield f"{module_prefix}.adapters.{idx}.{name}", param


def iter_named_multi_lora_modules(model):
    models = model if isinstance(model, list) else [model]
    for chunk_i, model_chunk in enumerate(models):
        for name, module in model_chunk.named_modules():
            if module.__class__.__name__ == "MultiLoRALinear":
                yield f"chunk{chunk_i}.{name}", module
```

Then:

```python
def capture_optimizer_state_for_adapter(optimizer, model, idx: int) -> dict:
    name_by_main_param_id = {}
    for stable_name, param in iter_adapter_named_params_for_slot(model, idx):
        main = getattr(param, "main_param", None)
        key_param = main if main is not None else param
        name_by_main_param_id[id(key_param)] = stable_name

    captured = {}
    chained = getattr(optimizer, "chained_optimizers", [optimizer])
    for opt_i, chained_optimizer in enumerate(chained):
        inner = getattr(chained_optimizer, "optimizer", chained_optimizer)
        for param, state in inner.state.items():
            stable_name = name_by_main_param_id.get(id(param))
            if stable_name is None:
                continue
            captured[(opt_i, stable_name)] = {
                k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                for k, v in state.items()
            }
    return captured


def restore_optimizer_state_for_adapter(optimizer, model, idx: int, captured: dict) -> None:
    param_by_stable_name = {}
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
                k: v.to(device=param.device) if torch.is_tensor(v) else v
                for k, v in state.items()
            }
```

Do not use Python `id(param)` as the durable key. Use a stable module path and adapter parameter name. Python object IDs change across process restarts and slot moves.

### Pager class

```python
class AdapterPager:
    def __init__(self, args, model, optimizer, controller, scheduler, checkpoint_store):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.controller = controller
        self.scheduler = scheduler
        self.checkpoint_store = checkpoint_store

    def ensure_hot(self, job: TrainingJobRuntime, all_jobs: dict[str, TrainingJobRuntime]) -> int:
        if job.slot is not None:
            return job.slot

        slot = ray.get(self.controller.reserve_free_slot.remote(job.spec.job_id))
        if slot is None:
            victim = self._choose_victim(job, all_jobs)
            if victim is None:
                raise RuntimeError("no free slot and no preemptible victim")
            self.preempt(victim)
            slot = ray.get(self.controller.reserve_free_slot.remote(job.spec.job_id))
            assert slot is not None

        self._load_job_into_slot(job, slot)
        ray.get(self.controller.mark_hot.remote(job.spec.job_id, slot))
        return slot

    def preempt(self, job: TrainingJobRuntime) -> str:
        assert job.slot is not None
        checkpoint_uri = self._save_hot_job_checkpoint(job)
        self._clear_slot(job.slot)
        ray.get(self.controller.mark_cold.remote(job.spec.job_id, checkpoint_uri))
        return checkpoint_uri

    def _load_job_into_slot(self, job: TrainingJobRuntime, slot: int) -> None:
        from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter
        from miles.backends.megatron_utils.multi_lora import restore_optimizer_state_for_adapter

        init_adapter_slot(
            self.model,
            slot,
            rank=job.spec.adapter.rank,
            alpha=job.spec.adapter.alpha,
        )

        state = None
        if job.cold_checkpoint_uri is not None:
            state = self.checkpoint_store.load_training_checkpoint(job.cold_checkpoint_uri)
        elif job.spec.adapter.init == "adapter_uri":
            state = self.checkpoint_store.load_adapter_init(job.spec.adapter.adapter_uri)

        if state is not None:
            if "adapter_megatron" in state:
                load_adapter(self.model, slot, state["adapter_megatron"])
            if "optimizer" in state:
                restore_optimizer_state_for_adapter(self.optimizer, self.model, slot, state["optimizer"])

        self.optimizer.reload_model_params()

    def _save_hot_job_checkpoint(self, job: TrainingJobRuntime) -> str:
        assert job.slot is not None
        from miles.backends.megatron_utils.multi_lora import capture_optimizer_state_for_adapter

        adapter_state = self.checkpoint_store.extract_megatron_adapter_state(
            model=self.model,
            slot=job.slot,
        )
        optimizer_state = capture_optimizer_state_for_adapter(
            self.optimizer,
            self.model,
            job.slot,
        )
        return self.checkpoint_store.write_training_checkpoint(
            job=job,
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )

    def _clear_slot(self, slot: int) -> None:
        from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot
        from miles.backends.megatron_utils.multi_lora import zero_optimizer_state_for_adapter

        clear_adapter_slot(self.model, slot)
        zero_optimizer_state_for_adapter(self.optimizer, self.model, slot)
        self.optimizer.reload_model_params()

    def _choose_victim(self, incoming_job, all_jobs):
        hot = [j for j in all_jobs.values() if j.slot is not None]
        return self.scheduler.choose_preemption_victim(
            hot_jobs=hot,
            incoming_job=incoming_job,
            engine_step=max(j.trained_steps for j in all_jobs.values()) if all_jobs else 0,
        )
```

### Important implementation detail

Megatron distributed optimizer internals may require `optimizer.reload_model_params()` after loading/clearing slots. Existing Miles cleanup already calls it after zeroing drained adapter state; follow that pattern.

---

## 11. Checkpoint store and artifact publisher

Create `miles/training_engine/checkpoint_store.py` and `miles/training_engine/artifact_publisher.py`.

### Internal training checkpoint

Needed for preemption/resume. It must include:

```text
job spec or job_id
adapter version
adapter Megatron-native shard tensors for this TP/PP rank
optimizer state for this logical adapter
scheduler/progress metadata
rank/alpha/target module metadata
```

Example:

```python
class CheckpointStore:
    def write_training_checkpoint(self, job, adapter_state, optimizer_state) -> str:
        uri = f"{job.spec.output_uri}/internal/checkpoints/v{job.current_adapter_version:06d}/"
        payload = {
            "job_id": job.spec.job_id,
            "adapter_version": job.current_adapter_version,
            "rank": job.spec.adapter.rank,
            "alpha": job.spec.adapter.alpha,
            "adapter_megatron": adapter_state,
            "optimizer": optimizer_state,
            "trained_steps": job.trained_steps,
            "trained_tokens": job.trained_tokens,
        }
        write_rank_local_payload(uri, payload)
        return uri
```

### External artifact

Needed for clients/inference systems. It should be immutable by version and should usually be HF PEFT-compatible:

```text
{output_uri}/versions/v000012/
  adapter_model.safetensors
  adapter_config.json
  metadata.json
```

Use existing Miles/Megatron-Bridge export helpers where possible. Existing `save_multi_lora_checkpoints(...)` already writes both a Megatron-native shard and HF PEFT `adapter_model.safetensors`/`adapter_config.json`. Factor out a function that exports one slot to a specific output URI without requiring the old adapter directory layout.

```python
class AdapterArtifactPublisher:
    def publish(self, job: TrainingJobRuntime, model) -> str:
        assert job.slot is not None
        # Use expose_adapter_slot(model, job.slot), bridge.export_adapter_weights(...),
        # slice_lora_to_rank(...), and safetensors save, similar to save_multi_lora_checkpoints.
        uri = f"{job.spec.output_uri}/versions/v{job.current_adapter_version:06d}/"
        export_hf_peft_adapter(
            model=model,
            slot=job.slot,
            rank=job.spec.adapter.rank,
            alpha=job.spec.adapter.alpha,
            target_modules=job.spec.adapter.target_modules,
            output_uri=uri,
        )
        write_json({
            "job_id": job.spec.job_id,
            "adapter_version": job.current_adapter_version,
            "base_model": job.spec.base_model,
            "rank": job.spec.adapter.rank,
            "alpha": job.spec.adapter.alpha,
            "target_modules": job.spec.adapter.target_modules,
            "trained_steps": job.trained_steps,
            "trained_tokens": job.trained_tokens,
        }, f"{uri}/metadata.json")
        return uri
```

In external-rollout mode, this artifact publisher replaces direct SGLang update calls.

---

## 12. Continuous packer

Create `miles/training_engine/continuous_packer.py`.

The packer turns selected jobs' examples into one or more Megatron microbatches. MVP can require one loss type per training quantum; mixed SFT+GRPO in one step can come later.

```python
@dataclass
class PackedContinuousBatch:
    microbatches: list[dict]
    job_token_counts: dict[str, int]
    job_loss_weights: dict[str, float]
    loss_type: str
```

Packer sketch:

```python
def pack_examples_by_slot(
    examples: list[TrainExample],
    *,
    max_slots: int,
    pad_to_multiple: int,
    loss_type: str,
) -> dict:
    if not examples:
        raise ValueError("no examples")

    # V1: one loss family per packed step.
    if {ex.loss_type for ex in examples} != {loss_type}:
        raise ValueError("mixed loss types in one training quantum are not supported in MVP")

    # Required by MultiLoRALinear.
    examples.sort(key=lambda ex: (int(ex.slot), -len(ex.input_ids)))

    input_ids = []
    attention_mask = []
    loss_mask = []
    labels = []
    old_logprobs = []
    ref_logprobs = []
    rewards = []
    advantages = []
    returns = []
    group_ids = []

    adapter_token_counts = torch.zeros(max_slots, dtype=torch.int32)
    job_ranges: dict[str, list[tuple[int, int]]] = {}
    job_token_counts: dict[str, int] = {}

    cursor = 0
    for ex in examples:
        assert ex.slot is not None
        n = len(ex.input_ids)
        start, end = cursor, cursor + n

        input_ids.extend(ex.input_ids)
        attention_mask.extend(ex.attention_mask)
        loss_mask.extend(ex.loss_mask)

        adapter_token_counts[ex.slot] += n
        job_ranges.setdefault(ex.job_id, []).append((start, end))
        job_token_counts[ex.job_id] = job_token_counts.get(ex.job_id, 0) + int(sum(ex.loss_mask))

        if loss_type == "sft":
            labels.extend(ex.labels)
        else:
            old_logprobs.extend(ex.old_logprobs)
            if ex.ref_logprobs is not None:
                ref_logprobs.extend(ex.ref_logprobs)
            if ex.advantages is not None:
                advantages.extend(ex.advantages)
            if ex.returns is not None:
                returns.extend(ex.returns)
            if ex.rewards is not None:
                rewards.extend(ex.rewards)
            group_ids.append(ex.group_id)

        cursor = end

    batch = {
        "tokens": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int32),
        "loss_masks": torch.tensor(loss_mask, dtype=torch.float32),
        "adapter_token_counts": adapter_token_counts,
        "job_ranges": job_ranges,
        "job_token_counts": job_token_counts,
        "loss_type": loss_type,
    }

    if loss_type == "sft":
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
    else:
        batch["rollout_log_probs"] = torch.tensor(old_logprobs, dtype=torch.float32)
        if ref_logprobs:
            batch["ref_log_probs"] = torch.tensor(ref_logprobs, dtype=torch.float32)
        if advantages:
            batch["advantages"] = torch.tensor(advantages, dtype=torch.float32)
        if returns:
            batch["returns"] = torch.tensor(returns, dtype=torch.float32)
        if rewards:
            batch["rewards"] = torch.tensor(rewards, dtype=torch.float32)
        batch["group_ids"] = group_ids

    return finalize_for_megatron(batch, pad_to_multiple=pad_to_multiple)
```

`finalize_for_megatron(...)` must produce whatever packed sequence metadata the existing Miles/Megatron forward path expects. Reuse the current Miles batch preparation utilities rather than inventing new packed-sequence semantics.

---

## 13. Continuous loss

Create `miles/training_engine/continuous_loss.py`.

Loss must be normalized per logical job, not globally, otherwise the largest job dominates. Because LoRA slots are disjoint parameters, summing per-job normalized losses is safe.

```python
def continuous_loss(output_tensor, batch: dict, job_specs: dict[str, TrainingJobSpec]):
    loss_type = batch["loss_type"]
    if loss_type == "sft":
        return continuous_sft_loss(output_tensor, batch, job_specs)
    if loss_type in {"grpo", "ppo"}:
        return continuous_rl_loss(output_tensor, batch, job_specs)
    raise ValueError(f"unsupported loss_type: {loss_type}")
```

### SFT

```python
def continuous_sft_loss(logits, batch, job_specs):
    token_losses = cross_entropy_per_token(logits, batch["labels"])
    total = 0.0
    metrics = {}

    for job_id, ranges in batch["job_ranges"].items():
        local_sum = torch.tensor(0.0, device=logits.device)
        local_count = torch.tensor(0.0, device=logits.device)
        for start, end in ranges:
            mask = batch["loss_masks"][start:end]
            local_sum = local_sum + (token_losses[start:end] * mask).sum()
            local_count = local_count + mask.sum()

        global_sum = all_reduce_sum_if_needed(local_sum)
        global_count = all_reduce_sum_if_needed(local_count).clamp_min(1)
        job_loss = global_sum / global_count

        # Scheduler fairness is mostly in token allocation. Keep loss weight 1 by default.
        total = total + job_loss
        metrics[f"{job_id}/loss"] = job_loss.detach()
        metrics[f"{job_id}/tokens"] = global_count.detach()

    return total, metrics
```

### RL from external old logprobs

```python
def continuous_rl_loss(model_logprobs, batch, job_specs):
    old_logprobs = batch["rollout_log_probs"]
    action_mask = batch["loss_masks"]
    total = 0.0
    metrics = {}

    for job_id, ranges in batch["job_ranges"].items():
        spec = job_specs[job_id]
        local_losses = []
        local_masks = []

        for start, end in ranges:
            pi_logp = model_logprobs[start:end]
            old_logp = old_logprobs[start:end]
            mask = action_mask[start:end]

            advantages = get_advantages_for_range(batch, job_id, start, end, spec)
            log_ratio = pi_logp - old_logp
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - spec.loss.clip_eps, 1.0 + spec.loss.clip_eps)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages)

            if spec.loss.kl_ref == "base":
                ref_logp = get_or_compute_ref_logprobs(batch, start, end)
                ref_delta = ref_logp - pi_logp
                kl = torch.exp(ref_delta) - ref_delta - 1.0
                token_loss = policy_loss + spec.loss.kl_coef * kl
            elif spec.loss.kl_ref == "provided":
                ref_logp = batch["ref_log_probs"][start:end]
                ref_delta = ref_logp - pi_logp
                kl = torch.exp(ref_delta) - ref_delta - 1.0
                token_loss = policy_loss + spec.loss.kl_coef * kl
            else:
                token_loss = policy_loss

            local_losses.append(token_loss)
            local_masks.append(mask)

        job_loss = masked_mean_concat(local_losses, local_masks)
        total = total + job_loss
        metrics[f"{job_id}/loss"] = job_loss.detach()

    return total, metrics
```

Important:

1. Every RL batch must include `old_logprobs` from the external behavior policy.
2. The trainer computes current policy logprobs with the current adapter slot.
3. Enforce max policy lag before data reaches the trainer.
4. If `kl_ref == "base"`, the trainer may compute reference logprobs using the frozen base. If this is expensive, allow clients to submit `ref_logprobs`.

---

## 14. Training runner

Create `miles/training_engine/runner.py`.

The runner owns Megatron model/optimizer and calls the existing model forward/backward path. It should not own external inference.

```python
class ContinuousTrainingRunner:
    def __init__(self, args, controller, scheduler, trajectory_store, checkpoint_store):
        self.args = args
        self.controller = controller
        self.scheduler = scheduler
        self.trajectory_store = trajectory_store
        self.checkpoint_store = checkpoint_store

        # Reuse existing Miles initialization.
        from miles.backends.megatron_utils.multi_lora import initialize_multi_lora_model_and_optimizer
        self.model, self.optimizer, self.opt_scheduler, self.iteration = (
            initialize_multi_lora_model_and_optimizer(args, role="actor")
        )

        self.pager = AdapterPager(
            args=args,
            model=self.model,
            optimizer=self.optimizer,
            controller=controller,
            scheduler=scheduler,
            checkpoint_store=checkpoint_store,
        )
        self.artifacts = AdapterArtifactPublisher(args=args)

    def run_forever(self):
        while True:
            jobs = ray.get(self.controller.snapshot_jobs.remote())
            self.scheduler.accrue_deficits(jobs)

            selected = self.scheduler.select_training_jobs(jobs)
            if not selected:
                time.sleep(self.args.continuous_idle_sleep_s)
                continue

            self._train_selected(selected, jobs)
```

Train selected jobs:

```python
    def _train_selected(self, selected: list[tuple[str, int]], jobs: dict[str, TrainingJobRuntime]):
        selected_job_ids = [job_id for job_id, _ in selected]

        # Make adapters hot.
        for job_id, _tokens in selected:
            job = jobs[job_id]
            self.pager.ensure_hot(job, jobs)

        # Refresh snapshot because slots may have changed.
        jobs = ray.get(self.controller.snapshot_jobs.remote())

        # Pull examples and attach physical slots.
        examples = []
        for job_id, target_tokens in selected:
            job = jobs[job_id]
            batches = self.trajectory_store.pop_for_job(job_id, target_tokens)
            job_examples = materialize_train_examples(job, batches)
            for ex in job_examples:
                ex.slot = job.slot
            examples.extend(job_examples)

        if not examples:
            return

        loss_types = {ex.loss_type for ex in examples}
        if len(loss_types) != 1:
            # MVP fallback: train homogeneous groups separately.
            for loss_type in sorted(loss_types):
                self._train_examples([e for e in examples if e.loss_type == loss_type], jobs)
        else:
            self._train_examples(examples, jobs)
```

Forward/backward:

```python
    def _train_examples(self, examples: list[TrainExample], jobs: dict[str, TrainingJobRuntime]):
        loss_type = examples[0].loss_type
        batch = pack_examples_by_slot(
            examples,
            max_slots=self.args.multi_lora_n_adapters,
            pad_to_multiple=self.args.data_pad_size_multiplier,
            loss_type=loss_type,
        )

        job_specs = {job_id: jobs[job_id].spec for job_id in batch["job_ranges"]}

        # Mark active for preemption safety.
        ray.get(self.controller.mark_active_step.remote(list(batch["job_ranges"].keys())))

        stats = self._run_megatron_step(batch, job_specs)

        updates = {}
        for job_id, token_count in batch["job_token_counts"].items():
            updates[job_id] = {"trained_tokens": token_count}
        ray.get(self.controller.commit_step.remote(updates))

        # Publish/checkpoint dirty adapters.
        self._maybe_publish_and_checkpoint(list(batch["job_ranges"].keys()))
```

Model step wrapper:

```python
    def _run_megatron_step(self, batch: dict, job_specs: dict[str, TrainingJobSpec]) -> dict:
        from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

        def forward_step(data_iterator, model):
            micro_batch = next(data_iterator)
            set_tokens_per_adapter_slot(model, micro_batch["adapter_token_counts"].to(torch.cuda.current_device()))
            output_tensor = model(
                input_ids=micro_batch["tokens"],
                position_ids=micro_batch.get("position_ids"),
                attention_mask=micro_batch.get("attention_mask"),
                packed_seq_params=micro_batch.get("packed_seq_params"),
            )
            return output_tensor, functools.partial(
                continuous_loss,
                batch=micro_batch,
                job_specs=job_specs,
            )

        data_iterator = OneShotDataIterator(split_into_microbatches(batch, self.args))
        self.optimizer.zero_grad()
        run_existing_megatron_forward_backward(
            args=self.args,
            model=self.model,
            forward_step_func=forward_step,
            data_iterator=data_iterator,
        )
        self.optimizer.step()
        self.opt_scheduler.step(1)
        return {"ok": True}
```

Implementation note: do not hand-roll pipeline parallelism. Reuse the exact existing Miles train loop functions for forward/backward/microbatching wherever possible. Only replace the data iterator and loss function.

---

## 15. API server and client

Create `miles/training_engine/api_server.py` and `miles/training_engine/client.py`.

Endpoints:

```text
POST /v1/training/jobs
GET  /v1/training/jobs/{job_id}
GET  /v1/training/jobs/{job_id}/events
POST /v1/training/jobs/{job_id}/trajectory_batches
POST /v1/training/jobs/{job_id}/cancel
GET  /v1/training/jobs/{job_id}/adapters/latest
GET  /v1/training/jobs/{job_id}/adapters/{version}
```

Server sketch:

```python
@app.post("/v1/training/jobs")
async def create_job(req: TrainingJobSpec):
    job_id = ray.get(controller.submit_job.remote(req))
    # Initialize v0 artifact after slot load or via CPU construction depending on mode.
    return {"job_id": job_id}


@app.post("/v1/training/jobs/{job_id}/trajectory_batches")
async def submit_trajectory_batch(job_id: str, req: ExternalTrajectoryBatch):
    job = ray.get(controller.get_job.remote(job_id))
    validate_trajectory_batch(job, req)
    batch_id, token_count = trajectory_store.put(req)
    ray.get(controller.mark_train_batch_ready.remote(job_id, batch_id, token_count))
    return {"accepted": True, "batch_id": batch_id, "token_count": token_count}
```

Client should be thin and avoid training logic.

---

## 16. Existing controller integration

You have two choices.

### MVP choice: bypass existing `MultiLoRAController` for new engine

The training engine can directly use Megatron-Bridge helpers:

```python
init_adapter_slot(model, slot, rank, alpha)
load_adapter(model, slot, state_dict)
clear_adapter_slot(model, slot)
set_tokens_per_adapter_slot(model, counts)
```

This avoids forcing user jobs into the old `adapter.yaml` state machine. The old controller remains used by legacy Miles flows.

### Compatibility choice: extend existing controller

If you want to keep one source of truth for hot adapter slots, add this method to `miles/ray/multi_lora_controller.py`:

```python
def register_adapter_config(self, config: AdapterConfig) -> dict:
    assert config.rank <= self.max_rank
    if config.name in self.configs:
        raise ValueError(f"adapter already registered: {config.name}")
    if not self.free_slots:
        raise RuntimeError("No free adapter slots")
    slot = min(self.free_slots)
    self.free_slots.remove(slot)
    config = dataclasses.replace(config, slot=slot, state=AdapterState.PENDING)
    self.configs[config.name] = config
    return {"name": config.name, "slot": slot}
```

But do not add complex scheduling to this old controller. Keep scheduling in `TrainingJobController` + `ContinuousTrainingScheduler`.

---

## 17. Disaggregated inference/training contract

This should be the default design direction.

The training engine does not call:

```python
engine.load_lora_adapter_from_tensors(...)
engine.unload_lora_adapter(...)
engine.generate(...)
```

Instead it emits immutable adapter artifacts:

```text
adapter_ready(job_id, version, uri, metadata)
```

External systems decide when to load them.

### Required metadata in every adapter artifact

```json
{
  "job_id": "job_abc",
  "adapter_version": 12,
  "base_model": "qwen3.6-35b-a3b",
  "base_model_revision": "...",
  "tokenizer_hash": "...",
  "lora_config_hash": "...",
  "rank": 16,
  "alpha": 32,
  "target_modules": ["q_proj", "v_proj", "o_proj"],
  "trained_steps": 12,
  "trained_tokens": 98304
}
```

### Required metadata in every external trajectory batch

```json
{
  "job_id": "job_abc",
  "adapter_version": 12,
  "base_model_hash": "...",
  "tokenizer_hash": "...",
  "lora_config_hash": "...",
  "sampling_metadata": {...}
}
```

Reject if hashes mismatch. Reject or downweight if policy lag is too large.

---

## 18. Colocated rollout plugin, later

A colocated rollout plugin can be implemented later using existing Miles SGLang integration.

Rules:

1. Keep it behind a mode flag, e.g. `args.training_engine_rollout_mode = "external" | "colocated"`.
2. In external mode, do not instantiate SGLang engines.
3. In colocated mode, reuse existing `UpdateWeightFromTensor.send_one_multi_lora_adapter(...)`, but change the sync policy from “send every ACTIVE adapter every cycle” to “send dirty adapter versions needed by rollout.”
4. Do not let rollout waiting keep trainer slots hot. A job waiting on rollout should be eligible for preemption/offload.

---

## 19. MoE/expert LoRA

Do not tackle expert LoRA in the first implementation.

The current Megatron-Bridge MultiLoRA path wraps Megatron parallel linears and the standard targets such as:

```text
linear_qkv
linear_proj
linear_fc1
linear_fc2
```

Expert LoRA requires grouping by `(adapter_slot, expert_id)`, interacts with expert parallel all-to-all dispatch, and may require a different grouped-mm layout. The continuous engine should be designed so it can eventually support expert-aware adapter modules, but MVP should stick to the current supported MultiLoRA transform.

---

## 20. CLI/args additions

Add flags in the Miles argument/config layer.

```text
--training-engine                    # enable long-lived continuous engine
--training-engine-mode external_rollout
--continuous-train-token-budget 32768
--continuous-max-adapters-per-step 8
--continuous-base-quantum-tokens 8192
--continuous-idle-sleep-s 0.01
--continuous-enable-preemption true
--continuous-checkpoint-store s3://...
--continuous-artifact-store s3://...
--continuous-api-host 0.0.0.0
--continuous-api-port 8080
```

Existing MultiLoRA flags remain relevant:

```text
--multi-lora
--multi-lora-n-adapters
--lora-rank
--lora-alpha
--target-modules
```

In continuous engine mode, `--multi-lora-n-adapters` means **max hot GPU slots**, not max logical jobs.

---

## 21. Metrics and events

Create `miles/training_engine/events.py` and `metrics.py`.

Emit events:

```text
job.created
job.queued
job.loaded_to_slot
job.train_step_started
job.train_step_finished
job.adapter_published
job.preempted
job.resumed
job.completed
job.failed
```

Metrics:

```text
engine.train_steps_total
engine.train_tokens_total
engine.ready_jobs
engine.hot_slots_used
engine.preemptions_total
engine.slot_load_seconds
engine.slot_save_seconds
engine.adapter_publish_seconds
job.{id}.trained_steps
job.{id}.trained_tokens
job.{id}.ready_train_tokens
job.{id}.policy_lag_rejections
job.{id}.loss
job.{id}.kl
job.{id}.reward_mean
```

Fairness metrics:

```text
job served_tokens / wallclock
job deficit_tokens
max consecutive steps per job
slot residency duration
```

---

## 22. Testing plan

### Unit tests

1. **Packer sorting invariant**
   - Create examples for slots `[2, 0, 2, 1]`.
   - Ensure packed tokens are sorted by slot.
   - Ensure `adapter_token_counts` sums to total tokens.
   - Ensure `job_ranges` correspond to correct token spans.

2. **Policy lag validation**
   - Current adapter version `10`.
   - Accept batch version `10` and `8` with `max_policy_lag=4`.
   - Reject batch version `5`.
   - Reject future batch version `11`.

3. **Scheduler fairness**
   - Two jobs, equal priority, large backlogs.
   - Over many selections, trained token allocation is near equal.
   - With priority 2:1, allocation approximates 2:1.

4. **Preemption victim selection**
   - Do not pick active-step jobs.
   - Do not pick non-preemptible jobs.
   - Prefer idle hot jobs.
   - Respect `min_hot_steps`.

5. **Optimizer state capture/restore**
   - Create toy MultiLoRA model with two slots.
   - Train slot 0 for one step to populate Adam state.
   - Capture state, clear slot, restore into slot 1.
   - Assert restored Adam tensors match.

6. **No optimizer leakage**
   - Train adapter A in slot 0.
   - Preempt/clear slot 0.
   - Load new adapter B into slot 0.
   - Assert B optimizer state starts zero unless restored from B checkpoint.

### Integration tests

1. **Single-job equivalence**
   - Run normal one-job LoRA SFT for `N` steps.
   - Run continuous engine with one job for `N` steps.
   - Compare adapter weights within tolerance.

2. **Two-job packed equivalence**
   - Train two adapters separately for one step each.
   - Train the same two adapters in one continuous packed step.
   - Compare each adapter’s weights to separate baseline.

3. **Preempt/resume equivalence**
   - Train job uninterrupted for `N` steps.
   - Train same job with preemption/offload after step `K`, resume, finish.
   - Compare final adapter weights within tolerance.

4. **External GRPO ingestion**
   - Submit synthetic trajectories with old logprobs/rewards.
   - Ensure trainer computes current logprobs and updates only the correct adapter slot.

5. **Distributed smoke test**
   - Run with TP > 1, PP optional.
   - Ensure `adapter_token_counts` is set on every rank.
   - Ensure adapter export works from correct writer ranks.

### Failure tests

1. Shape mismatch in trajectory batch -> reject.
2. Tokenizer/base hash mismatch -> reject.
3. Batch from stale adapter version -> reject.
4. No free slots and no preemptible victim -> job remains queued; no crash.
5. Artifact publish failure -> job remains hot/dirty and retries.

---

## 23. Implementation phases

### Phase 1: pure external SFT, no preemption

Deliver:

```text
TrainingJobSpec/Runtime schemas
TrainingJobController
DatasetWorker for prompt_completion_jsonl
ContinuousPacker
ContinuousRunner using existing Megatron MultiLoRA path
Adapter artifact export
Basic client/API
```

Limitations:

```text
logical jobs <= hot slots
no optimizer offload
one loss type per step
external rollout not yet enabled
```

Acceptance:

```text
can submit two SFT jobs at arbitrary times
engine batches them by slot when both have data
returns two adapter artifacts
```

### Phase 2: external trajectory ingestion + RL loss

Deliver:

```text
ExternalTrajectoryBatch API
TrajectoryStore
GRPO/PPO loss using old_logprobs
adapter_version validation
max_policy_lag enforcement
```

Acceptance:

```text
external script can generate fake rollout batches and submit them
engine trains adapter and emits next adapter version
```

### Phase 3: preemption/offload of adapter + optimizer state

Deliver:

```text
AdapterPager
optimizer state capture/restore
cold checkpoints
slot victim selection
preempt/resume tests
```

Acceptance:

```text
more logical jobs than hot slots
jobs progress fairly over time
no optimizer leakage across slot reuse
```

### Phase 4: production hardening

Deliver:

```text
persistent job DB
object-store checkpoint/artifact backend
auth/ACL integration
streaming job events
metrics
retryable publishing
fairness/priority controls
```

### Phase 5: optional colocated rollout plugin

Deliver:

```text
mode flag to launch/use SGLang
 dirty-versioned adapter pushes
internal rollout manager
bounded in-flight rollouts per job
```

This must remain optional; external-rollout mode should be the core.

---

## 24. Common pitfalls

1. **Do not keep jobs hot while waiting for external rollout.** If `ready_train_tokens == 0`, the job should be preemptible.
2. **Do not globally average loss across jobs.** Normalize per job, then sum/weight.
3. **Do not preempt during active forward/backward.** Only at training quantum boundaries.
4. **Do not use Python object IDs for durable optimizer checkpoints.** Use stable parameter names.
5. **Do not push every adapter to SGLang in external mode.** Export artifacts only.
6. **Do not assume one physical slot forever.** Logical job identity and slot identity must be separate.
7. **Do not accept RL trajectories without `old_logprobs` and `adapter_version`.** These are required for off-policy/asynchronous RL correctness.
8. **Do not mix arbitrary loss families in MVP.** Group SFT jobs with SFT jobs, GRPO jobs with GRPO jobs. Mixed-loss batching can come later.
9. **Do not rewrite Megatron pipeline parallelism.** Reuse existing Miles forward/backward and just swap data iterator/loss.
10. **Do not support expert/MoE LoRA in MVP.** The existing MultiLoRA layer path is for Megatron parallel linears; expert LoRA needs `(adapter, expert)` routing.

---

## 25. Minimal first PR checklist

A good first coding-agent PR should be small and prove the architecture:

```text
[ ] Add training_engine/schemas.py
[ ] Add training_engine/job_controller.py
[ ] Add training_engine/continuous_packer.py
[ ] Add training_engine/continuous_loss.py with SFT only
[ ] Add training_engine/runner.py that can run one continuous SFT step
[ ] Add local in-memory dataset/trajectory store
[ ] Add tests for packer and scheduler
[ ] Add single-job SFT equivalence smoke test
```

Avoid in the first PR:

```text
[ ] HTTP server
[ ] optimizer state offload
[ ] RL loss
[ ] SGLang integration
[ ] object-store artifact publishing
```

Once the first PR works, add external trajectory ingestion and then preemption/offload.

---

## 26. Suggested local commands for the coding agent

The exact test commands depend on the repo, but use this workflow:

```bash
# inspect branches
git status
git branch --show-current

# find current MultiLoRA implementation
grep -R "class MultiLoRALinear" -n ../Megatron-Bridge/src/megatron/bridge/peft .
grep -R "set_tokens_per_adapter_slot" -n . ../Megatron-Bridge/src/megatron/bridge/peft

# inspect current training forward path
grep -R "adapter_token_counts" -n miles

# inspect optimizer state cleanup
grep -R "zero_optimizer_state_for_adapter" -n miles

# run focused tests after adding unit tests
pytest -q tests/training_engine/test_continuous_packer.py
pytest -q tests/training_engine/test_scheduler.py
pytest -q tests/training_engine/test_optimizer_state.py
```

---

## 27. Summary

Implement the system as a new continuous training service layer around the existing GPU execution path:

```text
keep existing:
  Megatron model construction
  Megatron parallelism
  Megatron-Bridge MultiLoRALinear
  set_tokens_per_adapter_slot
  grouped-mm LoRA execution
  adapter slot init/load/clear/export helpers

add new:
  durable logical job controller
  external trajectory ingestion
  continuous fair scheduler
  adapter hot-slot pager
  per-job loss normalization
  versioned adapter artifact publisher
  preempt/resume checkpointing
```

The key API contract for disaggregated inference/training is:

```text
external inference sends:
  tokenized trajectories + old_logprobs + rewards/advantages + adapter_version

training engine returns:
  immutable adapter artifact version + metadata
```

The coding agent should treat the current Miles/Osmosis MultiLoRA stack as the low-level execution substrate and avoid changing it until the service layer is working. The most important invariant is that every training microbatch is sorted by physical LoRA slot and that `adapter_token_counts` exactly matches that sorted token layout.
