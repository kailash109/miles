# Revised v2 Design: Continuous-Batched MultiLoRA Training Engine on Miles

This document is a revised implementation plan for the `multilora_no_colocate` branch. It incorporates the central `TrainingCoordinator` design and adjusts the system to behave more like a real continuous-batching training service while keeping the implementation minimal and reusing as much Miles/Megatron/Megatron-Bridge code as possible.

The main principle is:

```text
The coordinator decides.
Megatron workers execute.
The coordinator commits.
```

The engine should not run a full scheduler/controller/store loop inside every `MegatronTrainRayActor`. Instead, a single central coordinator owns all global state and emits immutable `TrainStepPlan`s. Every Megatron worker receives the same plan and executes it using the existing Miles/Megatron training stack.

---

## 1. Goals and non-goals

### Goals

Build a long-lived training service that:

1. Accepts LoRA training jobs at arbitrary times.
2. Accepts externally generated SFT examples or RL trajectory batches.
3. Schedules many logical jobs onto a fixed number of GPU-resident MultiLoRA slots.
4. Continuously forms training batches using token budgets, fairness credits, and small batching windows.
5. Trains selected jobs through Miles/Megatron using existing model-parallel infrastructure.
6. Publishes immutable, versioned LoRA adapter artifacts.
7. Keeps inference disaggregated: external systems generate rollouts/logprobs/rewards and later consume adapter artifacts.

### Non-goals for the next implementation patch

Do not implement these yet:

```text
production HTTP auth
external database
full object-store persistence
colocated rollout manager
SGLang/vLLM adapter pushing
per-job optimizer types
packed sequence attention as the default
multi-loss mixing in a single TrainStepPlan
cross-base-model routing
elastic Megatron worker replacement
```

The next milestone should be a central Ray coordinator with in-memory metadata, Ray ObjectRefs for batch payloads, and a fixed already-running Miles/Megatron train group.

---

## 2. Current problem summary

The current prototype has useful pieces, but its biggest architectural issue is that the training engine is instantiated inside every Megatron worker.

Current shape:

```text
RayTrainGroup.run_continuous_sft_engine(...)
  -> broadcasts to every MegatronTrainRayActor
  -> every actor creates a controller/store/scheduler/runner
  -> every actor independently schedules jobs
  -> every actor independently pops data
  -> every actor independently loads/preempts slots
  -> every actor independently commits/publishes
```

This relies on all ranks observing identical inputs and making identical deterministic decisions forever. That is too fragile for a continuous service with arbitrary job arrivals, retries, preemption, publish failures, and actor failures.

A single divergence can break Megatron collectives:

```text
rank 0 plan: slot 0 = job A, slot 1 = job B
rank 1 plan: slot 0 = job A, slot 2 = job C

=> mismatched adapter_token_counts and potentially hanging/corrupt collectives
```

The corrected shape is:

```text
TrainingCoordinator
  -> builds one immutable TrainStepPlan
  -> sends the same plan to every Megatron worker
  -> commits or aborts globally after all workers finish
```

---

## 3. Revised high-level architecture

```text
External clients / rollout systems
  - submit jobs
  - submit SFT examples or RL trajectory batches
  - poll/subscribe for adapter artifacts

        |
        v

TrainingCoordinator Ray actor
  - single source of truth
  - job registry
  - batch metadata and leases
  - fairness scheduler
  - slot table and preemption decisions
  - adapter version state
  - TrainStepPlan creation
  - commit/abort
  - artifact-ready state

        |
        | immutable TrainStepPlan
        v

RayTrainGroup / MegatronTrainRayActor workers
  - existing Miles actors
  - own model shards and optimizer shards
  - execute the exact same TrainStepPlan
  - prepare adapter slots
  - materialize microbatches
  - call existing Miles/Megatron training helpers
  - return WorkerStepResult

        |
        v

TrainingCoordinator commit
  - consume data leases
  - update trained-token accounting
  - advance internal optimizer_step
  - mark pending artifact versions
  - finalize/publish adapter manifests
```

The core worker API becomes:

```python
class MegatronTrainRayActor:
    def prepare_training_engine_worker(self, engine_cfg: dict) -> dict:
        ...

    def execute_train_step_plan(self, plan_ref) -> WorkerStepResult:
        ...
```

The core coordinator API becomes:

```python
class TrainingCoordinator:
    def submit_job(self, spec: TrainingJobSpec) -> CreateJobResponse: ...
    def submit_trajectory_batch(self, batch: ExternalTrajectoryBatch) -> SubmitBatchResponse: ...
    def submit_sft_examples(self, job_id: str, examples: list[TrainExample]) -> SubmitBatchResponse: ...
    def build_next_plan(self) -> TrainStepPlan | None: ...
    def commit_or_abort_plan(self, plan_id: str, results: list[WorkerStepResult]) -> CommitResult: ...
```

The `TrainingCoordinator` is a continuous step scheduler. It does not create a
long-running schedule. It repeatedly creates one-step plans.

Each cycle:

1. Observe current jobs, ready batches, public adapter versions, slot residency,
   fairness deficits, and worker health.
2. Decide the next one-step batch.
3. Lease only the data needed for that step.
4. Assign physical slots for that step.
5. Execute exactly one optimizer step on the Megatron worker group.
6. Commit or abort.
7. Recompute the next step from scratch.

The fact that a job was selected for the previous step gives it no automatic
right to be selected for the next step. Persistent slot residency is only a cache
optimization.


---

## 4. What stays from the current implementation

Keep these abstractions:

### 4.1 Logical job, physical slot, adapter version

```text
job_id             durable user-facing identity
slot               physical MultiLoRA adapter index, can change over time
adapter_version    immutable published adapter artifact version
```

External rollouts must never refer to physical slots. They refer to:

```text
(job_id, adapter_version)
```

The trainer resolves:

```text
job_id -> current hot slot
```

only when building a training plan.

### 4.2 External trajectory contract

Keep `ExternalTrajectoryBatch` as the disaggregated inference/training boundary:

```python
@dataclass
class ExternalTrajectoryBatch:
    job_id: str
    adapter_version: int
    base_model_hash: str | None
    tokenizer_hash: str | None
    lora_config_hash: str | None

    input_ids: Any
    attention_mask: Any
    action_mask: Any
    old_logprobs: Any

    rewards: Any | None = None
    advantages: Any | None = None
    returns: Any | None = None
    group_ids: list[str] | None = None
    ref_logprobs: Any | None = None

    client_batch_id: str | None = None
```

The extra `client_batch_id` makes submissions idempotent. If a client retries after a timeout, the coordinator should return the same accepted `batch_id` rather than double-counting the batch.

### 4.3 `adapter_token_counts`

The existing Megatron-Bridge MultiLoRA layer expects flattened tokens sorted by physical slot:

```text
adapter_token_counts[i] = number of contiguous tokens for physical slot i
```

Keep this as the only model-routing interface. The continuous batcher and materializer must produce it.

### 4.4 Existing Miles/Megatron execution

Do not reimplement model construction, model parallelism, the MultiLoRA forward, or adapter export.

Reuse:

```text
initialize_multi_lora_model_and_optimizer
set_tokens_per_adapter_slot
init_adapter_slot / load_adapter / clear_adapter_slot / expose_adapter_slot
get_forward_backward_func
calculate_log_probs_and_entropy
save_multi_lora_checkpoints
RayTrainGroup / MegatronTrainRayActor
```

The training engine should supply:

```text
which jobs
which slots
which batch data
which loss
```

Miles/Megatron should own:

```text
forward/backward schedule
grad finalization
optimizer step
scheduler step
grad buffer cleanup
model-parallel collectives
```

---

## 5. Continuous batching semantics for training

Training does not have the same safe boundary as inference. Inference can continuously admit decode tokens every step. Training cannot safely preempt halfway through a model-parallel forward/backward/optimizer step.

The training equivalent of continuous batching is:

```text
continuously accept job/data arrivals
accumulate ready data per job
periodically form a token-budgeted TrainStepPlan
run one all-or-nothing Megatron training quantum
commit/abort globally
repeat
```

### 5.1 Dispatch criteria

The coordinator should build and dispatch a plan when any of these are true:

```text
1. ready train tokens fill max_train_tokens_per_step
2. oldest runnable batch has waited longer than max_batch_wait_s
3. a high-priority job is starving
4. workers are idle and there is at least a minimum viable plan
```

Define:

```python
@dataclass(frozen=True)
class BatchingPolicy:
    max_train_tokens_per_step: int = 8192
    max_adapters_per_step: int = 8
    min_tokens_per_job: int = 512
    max_batch_wait_s: float = 0.25
    max_pending_plans: int = 1
    cost_metric: Literal["loss_tokens", "sequence_tokens"] = "loss_tokens"
```

MVP can use `loss_tokens`:

```python
def batch_cost(example_or_batch) -> int:
    return int(action_mask.sum())  # RL
    # or supervised loss-mask tokens for SFT
```

Later, use approximate FLOPs:

```python
cost = sum(seq_len_i * seq_len_i for seq_i in sequences) + lora_rank_cost
```

### 5.2 One active GPU plan

For the first implementation:

```text
max_active_gpu_plans = 1
```

Do not run two Megatron train plans concurrently on the same worker group.

You may prepare the next candidate plan on CPU while the current one runs, but do not commit leases or mutate slot ownership until the active plan finishes.

### 5.3 Token-budgeted fairness

Fairness should be based on trained tokens, not number of batches or number of jobs.

```python
job.deficit_tokens += base_quantum_tokens * job.priority
```

When a job trains:

```python
job.deficit_tokens -= trained_tokens
```

This gives weighted fair sharing over time.

### 5.4 Batching window

A pure greedy scheduler can be too eager under low load or too latent under medium load. Add oldest-ready age:

```python
def should_dispatch(now, ready_tokens, oldest_ready_at):
    if ready_tokens >= max_train_tokens_per_step:
        return True
    if now - oldest_ready_at >= max_batch_wait_s:
        return True
    if any_starving_high_priority_job():
        return True
    return False
```

This is the training-service analog of an inference dynamic batching wait timeout.

---

## 6. Coordinator-owned state model

Avoid a single enum that mixes lifecycle, residency, readiness, and execution. Use orthogonal fields.

```python
class Lifecycle(str, Enum):
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Residency(str, Enum):
    COLD = "cold"
    LOADING = "loading"
    HOT = "hot"
    PREEMPTING = "preempting"

class Readiness(str, Enum):
    EMPTY = "empty"
    READY = "ready"
    LEASED = "leased"

class Execution(str, Enum):
    IDLE = "idle"
    ACTIVE_STEP = "active_step"
```

Runtime:

```python
@dataclass
class TrainingJobRuntime:
    spec: TrainingJobSpec

    lifecycle: Lifecycle = Lifecycle.RUNNING
    residency: Residency = Residency.COLD
    readiness: Readiness = Readiness.EMPTY
    execution: Execution = Execution.IDLE

    slot: int | None = None
    ready_train_tokens: int = 0
    leased_train_tokens: int = 0

    deficit_tokens: int = 0
    consecutive_steps: int = 0
    hot_since_engine_step: int | None = None

    trained_steps: int = 0
    trained_tokens: int = 0

    optimizer_step: int = 0
    latest_materialized_step: int = 0
    latest_published_version: int = 0
    latest_adapter_uri: str | None = None
    pending_publish_steps: set[int] = field(default_factory=set)

    cold_checkpoint_uri: str | None = None
    last_ready_at: float | None = None
    last_error: str | None = None
```

Predicates:

```python
def is_runnable(job: TrainingJobRuntime) -> bool:
    return (
        job.lifecycle == Lifecycle.RUNNING
        and job.readiness == Readiness.READY
        and job.execution == Execution.IDLE
        and job.ready_train_tokens >= job.spec.scheduling.min_tokens_per_train_quantum
    )


def can_preempt(job: TrainingJobRuntime, engine_step: int) -> bool:
    if job.lifecycle != Lifecycle.RUNNING:
        return False
    if job.residency != Residency.HOT or job.slot is None:
        return False
    if job.execution != Execution.IDLE:
        return False
    if not job.spec.scheduling.preemptible:
        return False
    if job.hot_since_engine_step is not None:
        if engine_step - job.hot_since_engine_step < job.spec.scheduling.min_hot_steps:
            return False
    return True
```

---

## 7. Data ingestion, backpressure, and leases

### 7.1 Batch store

Replace destructive FIFO `pop_for_job` with lease semantics.

```python
class BatchState(str, Enum):
    AVAILABLE = "available"
    LEASED = "leased"
    CONSUMED = "consumed"
    REJECTED = "rejected"

@dataclass
class BatchRecord:
    batch_id: str
    job_id: str
    adapter_version: int
    token_count: int
    payload_ref: Any       # Ray ObjectRef in MVP
    state: BatchState
    created_at: float
    client_batch_id: str | None = None
    leased_by_plan_id: str | None = None
```

Coordinator-owned `BatchStore`:

```python
class BatchStore:
    def put(self, job_id: str, payload: TrainPayload, token_count: int,
            adapter_version: int, client_batch_id: str | None) -> str:
        # idempotency
        if client_batch_id and (job_id, client_batch_id) in self.client_id_index:
            return self.client_id_index[(job_id, client_batch_id)]

        payload_ref = ray.put(payload)
        batch_id = make_batch_id(job_id)
        record = BatchRecord(
            batch_id=batch_id,
            job_id=job_id,
            adapter_version=adapter_version,
            token_count=token_count,
            payload_ref=payload_ref,
            state=BatchState.AVAILABLE,
            created_at=time.time(),
            client_batch_id=client_batch_id,
        )
        self.records[batch_id] = record
        self.available_by_job[job_id].append(batch_id)
        if client_batch_id:
            self.client_id_index[(job_id, client_batch_id)] = batch_id
        return batch_id

    def lease_for_plan(self, job_id: str, target_tokens: int, plan_id: str) -> list[BatchLease]:
        leases = []
        tokens = 0
        for batch_id in list(self.available_by_job[job_id]):
            if tokens >= target_tokens:
                break
            record = self.records[batch_id]
            if record.state != BatchState.AVAILABLE:
                continue
            record.state = BatchState.LEASED
            record.leased_by_plan_id = plan_id
            leases.append(BatchLease.from_record(record))
            tokens += record.token_count
        return leases

    def commit_plan(self, plan_id: str) -> None:
        for record in self.records.values():
            if record.leased_by_plan_id == plan_id:
                record.state = BatchState.CONSUMED

    def abort_plan(self, plan_id: str) -> None:
        for record in self.records.values():
            if record.leased_by_plan_id == plan_id:
                record.state = BatchState.AVAILABLE
                record.leased_by_plan_id = None
```

### 7.2 Backpressure

Add queue limits. Without these, external rollout systems can fill Ray object store memory.

```python
@dataclass(frozen=True)
class QueueLimits:
    max_jobs: int = 1024
    max_ready_tokens_per_job: int = 2_000_000
    max_ready_tokens_global: int = 50_000_000
    max_payload_bytes_per_job: int = 16 * 1024**3
    max_batches_per_submit: int = 1024
```

On submit:

```python
def submit_trajectory_batch(batch):
    job = self.jobs[batch.job_id]
    token_count = count_action_tokens(batch.action_mask)

    if job.ready_train_tokens + token_count > self.limits.max_ready_tokens_per_job:
        return SubmitBatchRejected(reason="job_backpressure")

    if self.batch_store.global_ready_tokens + token_count > self.limits.max_ready_tokens_global:
        return SubmitBatchRejected(reason="engine_backpressure")

    batch_id = self.batch_store.put(...)
    job.ready_train_tokens += token_count
    job.readiness = Readiness.READY
    job.last_ready_at = job.last_ready_at or time.time()
    return SubmitBatchAccepted(batch_id=batch_id, token_count=token_count)
```

Backpressure is part of the API. External rollout systems should slow down or stop submitting for that job when the coordinator returns backpressure.

---

## 8. Versioning and artifact semantics

### 8.1 Separate internal optimizer step from public adapter version

Do not expose an adapter version until its artifact is durable.

Use:

```text
optimizer_step               internal train step counter
latest_materialized_step     internal checkpoint exists
latest_published_version     external artifact manifest is READY
```

After training succeeds:

```python
job.optimizer_step += 1
job.pending_publish_steps.add(job.optimizer_step)
```

After rank-local checkpoint/materialization succeeds:

```python
job.latest_materialized_step = step
```

After a `READY` artifact manifest is written:

```python
job.latest_published_version = step
job.latest_adapter_uri = uri
emit AdapterReady(job_id, version=step, uri=uri)
```

External trajectory validation uses `latest_published_version`:

```python
def validate_trajectory_batch(job, batch):
    if batch.adapter_version > job.latest_published_version:
        raise TrajectoryValidationError("future or unpublished adapter_version")

    lag = job.latest_published_version - batch.adapter_version
    if lag > job.spec.budget.max_policy_lag:
        raise TrajectoryValidationError("policy lag exceeds max")
```

### 8.2 Publish v0 at job creation

Disaggregated inference needs a concrete initial artifact.

At job creation:

```python
def submit_job(spec):
    job = TrainingJobRuntime(spec=spec)
    initial_uri = self.initial_artifact_store.create_v0(spec)
    job.latest_published_version = 0
    job.latest_adapter_uri = initial_uri
    self.jobs[spec.job_id] = job
    return CreateJobResponse(job_id=spec.job_id, adapter_version=0, adapter_uri=initial_uri)
```

`v0` can be:

```text
zero LoRA adapter
random-initialized adapter
warm-start adapter copied into this job namespace
metadata-only base/no-adapter artifact if explicitly supported
```

### 8.3 Artifact manifest

Use atomic `WRITING -> READY` manifests.

```python
@dataclass
class AdapterArtifactManifest:
    job_id: str
    version: int
    status: Literal["WRITING", "READY", "FAILED"]
    base_model: str
    tokenizer_hash: str | None
    lora_config_hash: str | None
    files: list[ArtifactFile]
    created_at: float
```

Publish protocol:

```text
1. workers write rank-local or PEFT adapter files to a temporary directory
2. workers return WorkerStepResult with written files
3. coordinator verifies all required files exist
4. coordinator/rank-0 writes manifest.json with status=READY
5. coordinator advances latest_published_version
```

MVP can continue using `save_multi_lora_checkpoints`, but the coordinator should only call `mark_published` after all workers returned success and the final manifest is ready.

---

## 9. TrainStepPlan

Create `miles/training_engine/plan.py`.

```python
@dataclass(frozen=True)
class SlotPreemption:
    job_id: str
    slot: int
    checkpoint_uri: str

@dataclass(frozen=True)
class SlotOnload:
    job_id: str
    slot: int
    source_uri: str | None
    rank: int
    alpha: int
    target_modules: tuple[str, ...]

@dataclass(frozen=True)
class BatchLease:
    batch_id: str
    job_id: str
    token_count: int
    payload_ref: Any
    adapter_version: int

@dataclass(frozen=True)
class TrainStepPlan:
    plan_id: str
    engine_step: int
    loss_type: Literal["sft", "grpo", "ppo", "dpo"]

    selected_jobs: tuple[str, ...]
    target_tokens: dict[str, int]
    loss_weights: dict[str, float]

    job_to_slot: dict[str, int]
    job_to_optimizer_step: dict[str, int]
    job_to_latest_published_version: dict[str, int]

    preemptions: tuple[SlotPreemption, ...]
    onloads: tuple[SlotOnload, ...]
    leases: dict[str, tuple[BatchLease, ...]]

    publish_after_step: tuple[str, ...]
```

Worker result:

```python
@dataclass
class WorkerStepResult:
    plan_id: str
    rank: int
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    written_files: list[str] = field(default_factory=list)
    error: str | None = None
```

The plan is immutable. Workers do not modify global state.

---

## 10. Coordinator planning loop

The coordinator owns planning and commit. It may initially be driven by a simple driver loop. Later it can become an async actor loop.

### 10.1 Build next plan

```python
class TrainingCoordinator:
    def build_next_plan(self) -> TrainStepPlan | None:
        now = time.time()
        self.scheduler.accrue_deficits(self.jobs)

        ready_tokens = self._total_runnable_ready_tokens()
        oldest_ready_at = self._oldest_runnable_ready_at()

        if not self._should_dispatch(now, ready_tokens, oldest_ready_at):
            return None

        selected = self.scheduler.select_training_jobs(
            jobs=self.jobs,
            token_budget=self.batching.max_train_tokens_per_step,
            max_adapters=self.batching.max_adapters_per_step,
        )
        if not selected:
            return None

        loss_type = self._choose_homogeneous_loss_type(selected)
        selected = [item for item in selected if self.jobs[item.job_id].spec.loss.type == loss_type]

        plan_id = new_plan_id()
        preemptions, onloads, job_to_slot = self._assign_slots(selected)

        leases = {}
        for item in selected:
            leases[item.job_id] = tuple(
                self.batch_store.lease_for_plan(
                    item.job_id,
                    target_tokens=item.target_tokens,
                    plan_id=plan_id,
                )
            )

        if not any(leases.values()):
            return None

        plan = TrainStepPlan(
            plan_id=plan_id,
            engine_step=self.engine_step,
            loss_type=loss_type,
            selected_jobs=tuple(item.job_id for item in selected),
            target_tokens={item.job_id: item.target_tokens for item in selected},
            loss_weights={item.job_id: 1.0 for item in selected},
            job_to_slot=job_to_slot,
            job_to_optimizer_step={j: self.jobs[j].optimizer_step for j in job_to_slot},
            job_to_latest_published_version={j: self.jobs[j].latest_published_version for j in job_to_slot},
            preemptions=tuple(preemptions),
            onloads=tuple(onloads),
            leases=leases,
            publish_after_step=tuple(j for j in job_to_slot if self._should_publish(j)),
        )

        self.plan_store[plan_id] = PlanRecord(plan=plan, state="leased")
        self._mark_jobs_leased(plan)
        return plan
```

### 10.2 Slot assignment

Slot policy lives in the coordinator.

```python
def _assign_slots(self, selected) -> tuple[list[SlotPreemption], list[SlotOnload], dict[str, int]]:
    preemptions = []
    onloads = []
    job_to_slot = {}

    for item in selected:
        job = self.jobs[item.job_id]
        if job.residency == Residency.HOT and job.slot is not None:
            job_to_slot[job.spec.job_id] = job.slot
            continue

        slot = self.slot_table.find_free_slot()
        if slot is None:
            victim = self.scheduler.choose_preemption_victim(
                hot_jobs=self._hot_jobs(),
                incoming_job=job,
                engine_step=self.engine_step,
            )
            if victim is None:
                continue
            slot = victim.slot
            preemptions.append(SlotPreemption(
                job_id=victim.spec.job_id,
                slot=slot,
                checkpoint_uri=self._internal_checkpoint_uri(victim),
            ))
            self.slot_table.tentatively_free(slot)

        source_uri = job.cold_checkpoint_uri or job.spec.adapter.adapter_uri
        onloads.append(SlotOnload(
            job_id=job.spec.job_id,
            slot=slot,
            source_uri=source_uri,
            rank=job.spec.adapter.rank,
            alpha=job.spec.adapter.alpha,
            target_modules=tuple(job.spec.adapter.target_modules),
        ))
        job_to_slot[job.spec.job_id] = slot
        self.slot_table.tentatively_bind(slot, job.spec.job_id)

    return preemptions, onloads, job_to_slot
```

### 10.3 Commit or abort

```python
def commit_or_abort_plan(self, plan_id: str, results: list[WorkerStepResult]) -> CommitResult:
    plan_record = self.plan_store[plan_id]
    plan = plan_record.plan

    if not results or not all(r.ok for r in results):
        self.batch_store.abort_plan(plan.plan_id)
        self._release_leased_job_state(plan)
        plan_record.state = "aborted"
        return CommitResult(ok=False, error=self._summarize_errors(results))

    self.batch_store.commit_plan(plan.plan_id)
    self.engine_step += 1

    for job_id in plan.selected_jobs:
        trained_tokens = sum(lease.token_count for lease in plan.leases[job_id])
        job = self.jobs[job_id]
        job.optimizer_step += 1
        job.trained_steps += 1
        job.trained_tokens += trained_tokens
        job.ready_train_tokens = max(0, job.ready_train_tokens - trained_tokens)
        job.deficit_tokens -= trained_tokens
        job.consecutive_steps += 1
        job.execution = Execution.IDLE
        job.readiness = Readiness.READY if job.ready_train_tokens > 0 else Readiness.EMPTY
        job.pending_publish_steps.add(job.optimizer_step)

    self._apply_slot_state_after_success(plan)
    plan_record.state = "committed"
    return CommitResult(ok=True)
```

Artifact publishing can be a second commit step:

```python
def publish_completed_step(self, job_id: str, step: int, worker_files: list[str]) -> None:
    uri = self.artifact_store.finalize_manifest(job_id, step, worker_files)
    job = self.jobs[job_id]
    job.latest_materialized_step = max(job.latest_materialized_step, step)
    job.latest_published_version = max(job.latest_published_version, step)
    job.latest_adapter_uri = uri
    job.pending_publish_steps.discard(step)
```

---

## 11. Ray execution pattern

### 11.1 Driver loop MVP

Use a simple driver loop first:

```python
coordinator = TrainingCoordinator.remote(...)
actor_model = allocate_train_group(...)
await actor_model.init()
await actor_model.prepare_training_engine_workers(engine_cfg)

while True:
    plan = ray.get(coordinator.build_next_plan.remote())
    if plan is None:
        if ray.get(coordinator.all_terminal_or_idle.remote()):
            break
        time.sleep(engine_cfg.idle_sleep_s)
        continue

    plan_ref = ray.put(plan)
    result_refs = actor_model.execute_train_step_plan(plan_ref)

    # Use ray.wait for timeout/health handling, even though all workers are needed.
    ready, not_ready = ray.wait(result_refs, num_returns=len(result_refs), timeout=engine_cfg.step_timeout_s)
    if not_ready:
        results = [WorkerStepResult(plan_id=plan.plan_id, rank=-1, ok=False, error="step_timeout")]
    else:
        results = ray.get(ready)

    ray.get(coordinator.commit_or_abort_plan.remote(plan.plan_id, results))
```

### 11.2 Avoid repeated by-value payloads

Do not pass raw trajectory tensors inside every actor call. The plan should contain `BatchLease` objects with Ray ObjectRefs. Workers do:

```python
payloads = ray.get([lease.payload_ref for lease in plan.leases[job_id]])
```

Later, replace ObjectRefs with object-store URIs for large payloads.

### 11.3 Health checks

Add a lightweight worker health method:

```python
class MegatronTrainRayActor:
    def training_engine_health(self) -> WorkerHealth:
        return WorkerHealth(
            rank=self.rank,
            current_plan_id=self.current_plan_id,
            state=self.engine_state,
        )
```

The coordinator/driver can periodically call this if a plan is stuck.

### 11.4 Placement groups

Keep the whole Megatron training group as one fixed gang-scheduled placement group. Do not create/destroy Ray placement groups per LoRA job. Logical job scheduling happens inside the coordinator, not Ray's resource scheduler.

---

## 12. Worker-side design

### 12.1 Modify `MegatronTrainRayActor`

Replace the current full engine entrypoint with worker preparation and plan execution.

```python
class MegatronTrainRayActor:
    def prepare_training_engine_worker(self, engine_cfg: dict) -> dict:
        self.megatron_plan_executor = MegatronPlanExecutor(
            args=self.args,
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            tokenizer=self.tokenizer,
        )
        return {"rank": self.rank, "ok": True}

    def execute_train_step_plan(self, plan_ref) -> WorkerStepResult:
        plan = ray.get(plan_ref)
        self.current_plan_id = plan.plan_id
        try:
            return self.megatron_plan_executor.execute(plan)
        finally:
            self.current_plan_id = None
```

### 12.2 Modify `RayTrainGroup`

```python
class RayTrainGroup:
    async def prepare_training_engine_workers(self, engine_cfg: dict):
        return await self._broadcast("prepare_training_engine_worker", engine_cfg)

    def execute_train_step_plan(self, plan_ref):
        return [
            actor.execute_train_step_plan.remote(plan_ref)
            for actor in self.actors
        ]
```

### 12.3 `MegatronPlanExecutor`

```python
class MegatronPlanExecutor:
    def __init__(self, args, model, optimizer, opt_param_scheduler, tokenizer=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.slot_executor = AdapterSlotExecutor(args, model, optimizer)
        self.materializer = BatchMaterializer(args, tokenizer)
        self.step_runner = MilesCustomStepRunner(args, model, optimizer, opt_param_scheduler)

    def execute(self, plan: TrainStepPlan) -> WorkerStepResult:
        try:
            self.slot_executor.prepare_slots(plan)
            microbatches = self.materializer.materialize(plan)
            metrics = self.step_runner.train_one_plan(plan, microbatches)
            written_files = self.slot_executor.maybe_write_dirty_adapter_files(plan)
            return WorkerStepResult(
                plan_id=plan.plan_id,
                rank=get_global_rank(),
                ok=True,
                metrics=metrics,
                written_files=written_files,
            )
        except Exception as exc:
            return WorkerStepResult(
                plan_id=plan.plan_id,
                rank=get_global_rank_or_minus_one(),
                ok=False,
                error=repr(exc),
            )
```

### 12.4 `AdapterSlotExecutor`

This replaces policyful `AdapterPager` on workers. It only executes plan ops.

```python
class AdapterSlotExecutor:
    def prepare_slots(self, plan: TrainStepPlan) -> None:
        for op in plan.preemptions:
            self.checkpoint_and_clear(op)
        for op in plan.onloads:
            self.load(op)

    def load(self, op: SlotOnload) -> None:
        init_adapter_slot(self.model, op.slot, rank=op.rank, alpha=op.alpha)
        if op.source_uri:
            state = self.checkpoint_store.load(op.source_uri)
            load_adapter(self.model, op.slot, state["adapter_megatron"])
            restore_optimizer_state_for_adapter(
                self.optimizer,
                self.model,
                op.slot,
                state.get("optimizer", {}),
            )
        self.optimizer.reload_model_params()

    def checkpoint_and_clear(self, op: SlotPreemption) -> None:
        adapter_state = self.checkpoint_store.extract_megatron_adapter_state(self.model, op.slot)
        optimizer_state = capture_optimizer_state_for_adapter(self.optimizer, self.model, op.slot)
        self.checkpoint_store.write_training_checkpoint(
            op.checkpoint_uri,
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )
        clear_adapter_slot(self.model, op.slot)
        zero_optimizer_state_for_adapter(self.optimizer, self.model, op.slot)
        self.optimizer.reload_model_params()
```

---

## 13. Batch materialization and packing

### 13.1 MVP: one sequence per microbatch

Keep the current safe behavior initially:

```text
one sequence per microbatch
normal causal attention
multiple microbatches accumulated into one optimizer step
```

This is not maximum continuous batching throughput, but it is the safest migration.

Call it explicitly:

```python
class MicrobatchPackingMode(str, Enum):
    ONE_SEQUENCE = "one_sequence"
    PACKED_SEQUENCES = "packed_sequences"
```

MVP materializer:

```python
class BatchMaterializer:
    def materialize(self, plan: TrainStepPlan) -> list[dict]:
        examples = []
        for job_id, leases in plan.leases.items():
            slot = plan.job_to_slot[job_id]
            payloads = ray.get([lease.payload_ref for lease in leases])
            for payload in payloads:
                examples.extend(self._payload_to_examples(payload, job_id, slot, plan.loss_type))

        # One example per microbatch for attention correctness.
        return [
            pack_examples_by_slot([ex], max_slots=self.args.multi_lora_n_adapters, loss_type=plan.loss_type)
            for ex in examples
        ]
```

### 13.2 Target: packed sequences

Later, support multiple sequences and multiple slots in one forward.

Layout:

```text
flat tokens:
  slot 0: seq A, seq B
  slot 2: seq C, seq D
  slot 5: seq E

adapter_token_counts:
  [len(A)+len(B), 0, len(C)+len(D), 0, 0, len(E), ...]

cu_seqlens:
  [0, len(A), len(A)+len(B), len(A)+len(B)+len(C), ...]
```

The model input must use Megatron packed sequence or equivalent block-diagonal causal attention. Do not simply concatenate sequences without attention boundaries.

### 13.3 Labels instead of global shifting

Even in one-sequence mode, represent labels explicitly:

```python
batch = {
    "tokens": ...,
    "labels": ...,       # -100 at ignored positions and sequence boundaries
    "loss_masks": ...,
    "action_mask": ...,
}
```

For RL current-policy logprobs, gather by labels:

```python
def gather_current_logprobs(logits, labels, action_mask):
    valid = labels != -100
    out = torch.zeros_like(labels, dtype=torch.float32)
    selected = vocab_parallel_gather_logprobs(logits[valid], labels[valid])
    out[valid] = selected
    return out * action_mask
```

This avoids a later rewrite when packed sequences are introduced.

---

## 14. Integrating with Miles train steps

The current runner hand-rolls too much of the Megatron train step. The revised design should factor a custom-forward train helper from Miles.

Add to `miles/backends/megatron_utils/model.py`:

```python
def train_with_custom_forward_step(
    args,
    model,
    optimizer,
    opt_param_scheduler,
    *,
    forward_step_func,
    data_iterator,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    iteration: int | None = None,
) -> dict:
    """Run one Miles/Megatron train step with a caller-provided forward_step_func.

    This must reuse the normal Miles/Megatron behavior for:
      - config.grad_scale_func
      - config.finalize_model_grads_func
      - get_forward_backward_func
      - optimizer.step and overflow/update_successful handling
      - opt_param_scheduler.step
      - grad buffer cleanup
      - loss reduction conventions
    """
```

Then worker plan execution calls:

```python
stats = train_with_custom_forward_step(
    args=self.args,
    model=self.model,
    optimizer=self.optimizer,
    opt_param_scheduler=self.opt_param_scheduler,
    forward_step_func=self._make_forward_step(plan),
    data_iterator=OneShotDataIterator(microbatches),
    num_microbatches=len(microbatches),
    seq_length=max_seq_len,
    micro_batch_size=1,
)
```

The custom forward step should do only:

```text
set_tokens_per_adapter_slot
model forward
continuous loss
```

Example:

```python
def make_continuous_forward_step(plan, job_specs):
    def forward_step(data_iterator, model):
        micro = next(data_iterator)
        set_tokens_per_adapter_slot(model, micro["adapter_token_counts"].to(device))

        logits = model(
            input_ids=micro["tokens"].to(device).unsqueeze(0),
            position_ids=micro.get("position_ids"),
            attention_mask=micro.get("attention_mask_for_model"),
            packed_seq_params=micro.get("packed_seq_params"),
        )

        def loss_func(output_tensor):
            if plan.loss_type == "sft":
                return continuous_sft_loss(output_tensor, micro, job_specs)
            current_logprobs = gather_current_logprobs(
                output_tensor,
                micro["labels"].to(device),
                micro["action_mask"].to(device),
            )
            return continuous_rl_loss(current_logprobs, micro, job_specs)

        return logits, loss_func
    return forward_step
```

If the Miles helper is hard to factor immediately, isolate current custom logic in `MilesCustomStepRunner`, but do not continue adding optimizer/grad-finalization complexity to the scheduler/runner.

---

## 15. Optimizer state and per-job optimizer config

### 15.1 Slot-independent optimizer-state keys

Current stable names that contain the physical slot are not stable across slot moves:

```text
chunk0.layers.0.linear_qkv.adapters.3.linear_in.weight
```

If a job reloads into slot 7, that key no longer matches.

Use slot-independent keys:

```python
def iter_adapter_named_params_for_slot(model, idx: int):
    for module_prefix, module in iter_named_multi_lora_modules(model):
        adapter = module.adapters[idx]
        for param_name, param in adapter.named_parameters():
            yield f"{module_prefix}.adapter.{param_name}", param
```

Then capture from slot 3 and restore into slot 7 use the same logical keys.

### 15.2 Per-job optimizer config

MVP should restrict optimizer configs:

```python
if spec.optimizer != engine.default_optimizer_spec:
    raise ValueError("per-job optimizer config is not supported in MVP")
```

Do not expose unsupported per-job learning rates/optimizer types as if they work.

Later, implement per-slot parameter groups:

```text
slot 0 param group -> current logical job A optimizer settings
slot 1 param group -> current logical job B optimizer settings
```

When a logical job moves slots, optimizer state and scheduler state move with the logical job.

---

## 16. Loss reduction

The loss functions must be careful about model-parallel groups.

Do not blindly all-reduce scalar per-job numerators/denominators over TP unless verified. TP may already have been handled inside vocab-parallel cross entropy/logprob helpers.

Add a single helper:

```python
def reduce_loss_stats_for_training_tokens(numer: torch.Tensor, denom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-job numerator/denominator over ranks that own distinct data tokens.

    This should usually include DP and CP participants, not necessarily TP ranks
    if token losses are already TP-correct and replicated across TP.
    """
```

Acceptance matrix:

```text
TP=1, DP=1
TP>1, DP=1
TP=1, DP>1
TP>1, DP>1
CP/SP enabled
```

Until verified, keep the first coordinator version SFT-only on TP=1/DP=1 or reuse Miles' existing loss reduction path.

---

## 17. Scheduler details

### 17.1 Accrual

```python
def accrue_deficits(self, jobs):
    for job in jobs.values():
        if is_runnable_or_waiting_with_ready_data(job):
            job.deficit_tokens += self.base_quantum_tokens * job.spec.scheduling.priority
```

### 17.2 Selection

```python
def select_training_jobs(self, jobs, token_budget, max_adapters):
    candidates = [j for j in jobs.values() if is_runnable(j)]
    candidates = [j for j in candidates if j.deficit_tokens >= j.spec.scheduling.min_tokens_per_train_quantum]
    candidates = [j for j in candidates if j.consecutive_steps < j.spec.scheduling.max_consecutive_steps]

    candidates.sort(key=lambda j: (
        -j.deficit_tokens,
        0 if j.residency == Residency.HOT else 1,
        j.consecutive_steps,
        -j.spec.scheduling.priority,
    ))

    selected = []
    remaining = token_budget
    for job in candidates:
        if len(selected) >= max_adapters:
            break
        target = min(
            remaining,
            job.spec.budget.tokens_per_update,
            job.ready_train_tokens,
            job.deficit_tokens,
        )
        if target < job.spec.scheduling.min_tokens_per_train_quantum:
            continue
        selected.append(SelectedJob(job_id=job.spec.job_id, target_tokens=target))
        remaining -= target
    return selected
```

### 17.3 Preemption victim

```python
def choose_preemption_victim(self, hot_jobs, incoming_job, engine_step):
    candidates = [j for j in hot_jobs if can_preempt(j, engine_step)]
    if not candidates:
        return None

    candidates.sort(key=lambda j: (
        j.ready_train_tokens > 0,      # idle jobs first
        j.deficit_tokens,              # less under-served first
        -j.consecutive_steps,          # hoggers first
        estimate_preemption_cost(j),   # cheaper first
        j.spec.scheduling.priority,    # lower priority first
    ))
    return candidates[0]
```

### 17.4 Cold-load penalty

Do not load a cold job for a tiny one-step quantum unless necessary.

Add to the selection score:

```python
score = fairness_score + hot_bonus + starvation_bonus - cold_load_penalty
```

MVP can approximate:

```python
cold_load_penalty = 0 if job.residency == HOT else base_quantum_tokens
```

---

## 18. Ray implementation practices

### 18.1 Central actor should not block ingestion forever

MVP can use a synchronous driver loop. For a service, make the coordinator async or split ingestion from scheduling.

Minimum:

```text
submit_job and submit_batch must be fast
build_next_plan can be called by a driver loop
commit_or_abort_plan is separate
```

Later:

```text
async TrainingCoordinator with concurrency groups:
  ingest: submit_job, submit_batch, get_status
  scheduler: build_plan, commit_plan
  health: worker_heartbeat, metrics
```

### 18.2 Use `ray.wait` for worker results

Even though a Megatron plan needs all workers, use `ray.wait` for timeout and health handling.

```python
ready, not_ready = ray.wait(result_refs, num_returns=len(result_refs), timeout=step_timeout_s)
if not_ready:
    coordinator.abort_plan(...)
    mark_train_group_unhealthy(...)
```

### 18.3 Pass large payloads by ObjectRef

`TrainStepPlan` should contain batch IDs and Ray ObjectRefs, not raw large arrays. Do not duplicate large payloads by value into every actor call.

### 18.4 Fixed placement group

The train group is a fixed gang-scheduled placement group. Do not create a placement group per LoRA job.

---

## 19. API contract

### 19.1 Create job

```python
resp = client.create_lora_training_job(
    base_model="qwen",
    adapter={"rank": 16, "alpha": 32, "target_modules": [...]},
    loss={"type": "grpo", "kl_ref": "base"},
    optimizer={"type": "adamw", "lr": 2e-4},
    budget={"max_steps": 500, "tokens_per_update": 8192, "max_policy_lag": 4},
)
```

Response:

```python
{
    "job_id": "job_abc",
    "adapter_version": 0,
    "adapter_uri": "s3://.../job_abc/v000000/manifest.json",
}
```

### 19.2 Submit RL trajectories

```python
client.submit_trajectory_batch(
    job_id="job_abc",
    adapter_version=0,
    client_batch_id="rollout-worker-3-batch-912",
    input_ids=...,
    attention_mask=...,
    action_mask=...,
    old_logprobs=...,
    rewards=...,
    group_ids=...,
)
```

Response:

```python
{"accepted": True, "batch_id": "batch_xyz", "token_count": 4096}
```

or:

```python
{"accepted": False, "reason": "job_backpressure"}
```

### 19.3 Adapter ready event

```python
{
    "event": "adapter_ready",
    "job_id": "job_abc",
    "adapter_version": 7,
    "adapter_uri": "s3://.../job_abc/v000007/manifest.json",
    "trained_tokens": 57344,
}
```

External inference decides when to load this artifact. The training engine does not push to inference in disaggregated mode.

---

## 20. File-by-file implementation guide

### New files

```text
miles/training_engine/plan.py
miles/training_engine/batch_store.py
miles/training_engine/coordinator.py
miles/training_engine/megatron_executor.py
miles/training_engine/adapter_slot_executor.py
miles/training_engine/batch_materializer.py
miles/training_engine/results.py
```

### Modify `miles/training_engine/schemas.py`

Add:

```text
Lifecycle / Residency / Readiness / Execution
optimizer_step / latest_published_version / pending_publish_steps
client_batch_id on ExternalTrajectoryBatch
```

Update validation to use `latest_published_version`, not internal step.

### Modify `miles/training_engine/scheduler.py`

Keep pure. It should not touch Ray or model state.

Add:

```text
batching window awareness
cold-load penalty
preemption victim selection called only by coordinator
```

### Replace or wrap `trajectory_store.py`

Add `BatchStore` with:

```text
put
lease_for_plan
commit_plan
abort_plan
pending_tokens
backpressure checks
idempotency index
```

### Modify `adapter_pager.py`

Turn policyful pager into worker-local `AdapterSlotExecutor`.

Remove:

```text
ensure_hot(job, all_jobs)
_choose_victim
```

Add:

```text
prepare_slots(plan)
load(SlotOnload)
checkpoint_and_clear(SlotPreemption)
```

### Modify `miles/backends/megatron_utils/multi_lora.py`

Fix optimizer state keys:

```text
bad:  module.adapters.{slot}.param
best: module.adapter.param
```

Add tests for restore into a different slot.

### Modify `runner.py`

Deprecate as global engine loop. Either remove it or reduce it to a single-plan executor. Prefer moving worker logic into `megatron_executor.py`.

### Modify `miles/backends/megatron_utils/actor.py`

Replace full engine method with:

```python
prepare_training_engine_worker(engine_cfg)
execute_train_step_plan(plan_ref)
training_engine_health()
```

### Modify `miles/ray/actor_group.py`

Add broadcast helpers:

```python
prepare_training_engine_workers(engine_cfg)
execute_train_step_plan(plan_ref)
```

### Modify `miles/backends/megatron_utils/model.py`

Factor:

```python
train_with_custom_forward_step(...)
```

Use normal Miles training-step lifecycle.

### Modify `examples/training_engine/engine_driver.py`

Make it own the central loop:

```python
coordinator = TrainingCoordinator.remote(...)
await actor_model.prepare_training_engine_workers(...)

while True:
    plan = ray.get(coordinator.build_next_plan.remote())
    if plan is None: ...
    results = await actor_model.execute_train_step_plan(ray.put(plan))
    ray.get(coordinator.commit_or_abort_plan.remote(plan.plan_id, results))
```

---

## 21. Migration plan

### Phase 1: Central coordinator, same GPU math

Implement:

```text
TrainingCoordinator
TrainStepPlan
BatchStore lease/commit/abort
MegatronPlanExecutor
execute_train_step_plan on workers
```

Keep:

```text
one-example microbatching
SFT-only first if needed
homogeneous optimizer config
in-memory state
```

### Phase 2: Versioning and artifact correctness

Implement:

```text
optimizer_step vs latest_published_version
v0 artifact
artifact manifest WRITING/READY/FAILED
policy-lag validation against published version
```

### Phase 3: Miles train-step integration

Factor and use `train_with_custom_forward_step`.

### Phase 4: Optimizer-state fix

Make optimizer checkpoint keys slot-independent. Add slot-move tests.

### Phase 5: Backpressure and idempotency

Add queue limits and client batch IDs.

### Phase 6: Better batching

Add `PACKED_SEQUENCES` mode with Megatron packed sequence support.

---

## 22. Acceptance tests

### Coordinator unit tests

```text
submit_job publishes or records v0
submit_batch validates adapter version and applies backpressure
batch lease -> commit consumes data
batch lease -> abort releases data
scheduler selects by deficit tokens
slot assignment is central and deterministic
preemption victims are selected centrally
public version does not advance until manifest is READY
```

### Worker/executor tests

```text
all workers receive same plan_id
execute_train_step_plan does not create a controller/store/scheduler
slot onload calls init_adapter_slot/load_adapter
slot preemption captures adapter + optimizer state and clears slot
optimizer state restores into a different slot
```

### Integration tests

```text
TP=1, PP=1, DP=1, two SFT jobs, two slots
TP=2, PP=1, two SFT jobs, two slots
TP=1, PP=1, one slot, two jobs -> forces preemption
failed worker result aborts leases and does not advance versions
publish failure leaves latest_published_version unchanged
client retry with same client_batch_id is idempotent
```

### Continuous batching tests

```text
plan dispatches immediately when token budget is full
plan dispatches after max_batch_wait_s even if not full
small jobs eventually run due to fairness deficit
hot jobs are preferred when fairness is close
cold jobs are not thrashed for tiny quanta
```

---

## 23. Final target shape

```text
client.submit_job / submit_batch
        |
        v
TrainingCoordinator
  - validates versions
  - applies backpressure
  - leases data
  - schedules jobs fairly by trained tokens
  - assigns/preempts slots
  - emits immutable TrainStepPlan
        |
        v
RayTrainGroup.execute_train_step_plan(plan)
  - every Megatron worker executes same immutable plan
  - workers only touch local model/optimizer tensors
  - workers call existing Miles/Megatron train-step helpers
        |
        v
TrainingCoordinator.commit_or_abort
  - consumes or releases data leases
  - advances internal optimizer_step
  - finalizes immutable adapter artifact
  - updates latest_published_version
```

The important design properties are:

```text
single-writer global state
all-or-nothing model-parallel execution
lease-based data safety
token-budgeted continuous batching
backpressure for external producers
artifact-first public versioning
minimal custom Megatron logic
```

This gives a clean path from the current prototype to a real continuous-batched MultiLoRA training service while keeping most of the hard model-parallel work inside existing Miles and Megatron-Bridge code.
