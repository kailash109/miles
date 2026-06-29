# Training Engine Architecture

A visual tour of the **continuous-batched MultiLoRA training engine**: what the
pieces are, **what data flows between them**, and **how a job runs end-to-end**.

The guiding principle is a strict separation of authority:

```text
The coordinator decides.   (single-writer control plane, CPU, torch-free)
Megatron workers execute.  (stateless execution plane, GPU)
The coordinator commits.   (all-or-nothing)
```

One central `TrainingCoordinator` (a Ray actor) owns *all* global state and emits
immutable `TrainStepPlan`s. Every Megatron worker receives the **same** plan,
executes it against its local model/optimizer shard, and returns a
`WorkerStepResult`. Workers never mutate shared state.

> Companion docs: `DESIGN.md` (prose rationale) and
> `revised_v2_multilora_training_engine_design.md` (full design spec). This file
> is the diagram-first view.

---

## 1. The big picture

Four planes cooperate. Clients talk only HTTP; the GPU is reached only through
the coordinator's plans.

```mermaid
flowchart TB
    subgraph CLIENT["Client plane (any host)"]
        C1["ServiceClient / LoraTrainingClient<br/>math_rl_clients.py"]
    end

    subgraph API["HTTP API (api_server.py, FastAPI)"]
        A1["/v1/training/jobs<br/>/sft_examples<br/>/trajectory_batches<br/>/sample • /sync_weights<br/>/v1/stats"]
    end

    subgraph CONTROL["Control plane — CPU, single writer (Ray actor)"]
        CO["TrainingCoordinator<br/>coordinator.py"]
        SC["ContinuousTrainingScheduler<br/>scheduler.py"]
        BS["BatchStore (leases)<br/>batch_store.py"]
        AS["ArtifactStore (manifests)<br/>artifact_store.py"]
        ST["Job registry + slot table<br/>schemas.py"]
        CO --- SC
        CO --- BS
        CO --- AS
        CO --- ST
    end

    subgraph DRIVER["Driver loop (serve_engine.py, background thread)"]
        D1["build -> execute -> commit<br/>+ W&B logging"]
    end

    subgraph EXEC["Execution plane — GPU (Megatron Ray actors, all ranks)"]
        E1["MegatronPlanExecutor<br/>megatron_executor.py"]
        E2["AdapterSlotExecutor<br/>(onload/preempt/export)"]
        E3["BatchMaterializer + ContinuousPacker"]
        E4["MultiLoRA model + optimizer<br/>(N hot slots)"]
        E1 --- E2
        E1 --- E3
        E1 --- E4
    end

    subgraph INFER["Inference plane — optional (online RL)"]
        G1["SglangGenerator -> sglang router"]
        G2["multi_lora_controller"]
    end

    C1 -->|HTTP| A1
    A1 -->|ray.get| CO
    D1 -->|build_next_plan| CO
    D1 -->|"execute_train_step_plan (ray.put plan)"| E1
    E1 -->|WorkerStepResult| D1
    D1 -->|commit_or_abort_plan| CO
    D1 -.->|sync adapters every N steps| G2
    A1 -.->|/sample| G1
    G2 --> G1
```

**Who owns what**

| Plane | Process | Owns | torch? | Ray? |
|---|---|---|---|---|
| Control | 1 Ray actor (CPU) | jobs, leases, slots, versions, scheduling | no | actor only |
| Driver | background thread on the head | the build/execute/commit loop + logging | no | yes |
| Execution | Megatron Ray actors (every GPU rank) | model + optimizer tensors, slot ops | yes | yes |
| Inference | sglang router (optional) | rollout generation for online RL | yes | yes |

---

## 2. Three identities (don't conflate them)

```mermaid
flowchart LR
    J["job_id<br/><i>durable, user-facing</i>"] -->|maps to, can change| S["slot<br/><i>physical MultiLoRA index 0..N-1</i>"]
    J -->|publishes| V["adapter_version<br/><i>immutable artifact, advances only on READY</i>"]
```

- **`job_id`** — the durable handle a client keeps.
- **`slot`** — a physical adapter index in the model. A job is *paged in* to a
  slot when HOT and *paged out* when COLD; its slot can change over its life.
- **`adapter_version`** — a published artifact version. Only advances when a
  durable manifest becomes `READY`. External rollouts reference
  `(job_id, adapter_version)` — **never** a slot.

---

## 3. Orthogonal job state

A job does not have one mixed status enum. It carries **four independent** state
axes (`schemas.py`), and predicates like `is_runnable` / `can_preempt` are
computed over them.

```mermaid
stateDiagram-v2
    direction LR
    state "Lifecycle" as L {
        RUNNING --> COMPLETING
        COMPLETING --> COMPLETED
        RUNNING --> FAILED
        RUNNING --> CANCELLED
    }
    state "Residency (where the weights live)" as R {
        COLD --> LOADING
        LOADING --> HOT
        HOT --> PREEMPTING
        PREEMPTING --> COLD
    }
    state "Readiness (is there data?)" as RD {
        EMPTY --> READY: data submitted
        READY --> LEASED: picked for a plan
        LEASED --> READY: more data left
        LEASED --> EMPTY: drained
    }
    state "Execution (in a step right now?)" as E {
        IDLE --> ACTIVE_STEP: plan leased
        ACTIVE_STEP --> IDLE: commit/abort
    }
```

A job is **runnable** only when all four line up:

```text
is_runnable(job) ==
    Lifecycle.RUNNING
    and Readiness.READY
    and Execution.IDLE
    and ready_train_tokens >= scheduling.min_tokens_per_train_quantum
```

---

## 4. What data is passed around

The system has a small set of typed contracts (`schemas.py`, `plan.py`,
`results.py`). This is the "vocabulary" everything else speaks.

```mermaid
flowchart TB
    subgraph IN["Client -> Coordinator (ingress)"]
        TJS["TrainingJobSpec<br/>adapter, loss, optimizer,<br/>budget, scheduling, output_uri"]
        ETB["ExternalTrajectoryBatch<br/>input_ids, attention_mask, action_mask,<br/>old_logprobs, rewards/advantages, group_ids"]
        TE_IN["TrainExample[] (SFT)<br/>input_ids, loss_mask, labels"]
    end

    subgraph MID["Coordinator -> Workers (the plan)"]
        PLAN["TrainStepPlan (immutable)<br/>selected_jobs, job_to_slot, leases,<br/>onloads, preemptions, job_to_loss,<br/>exports, publish_after_step"]
    end

    subgraph GPU["Inside a worker (materialized)"]
        TE["TrainExample (slot-stamped)"]
        MB["packed microbatch dict<br/>tokens, masks, adapter_token_counts,<br/>job_ranges, advantages/logprobs"]
    end

    subgraph OUT["Workers -> Coordinator (egress)"]
        WSR["WorkerStepResult<br/>ok, metrics: job/train/*, written_files"]
        CR["CommitResult<br/>ok, published[] (adapter_ready)"]
    end

    TJS --> PLAN
    ETB --> PLAN
    TE_IN --> PLAN
    PLAN --> TE --> MB
    MB --> WSR --> CR
```

**Key payloads at a glance**

| Type | Direction | Carries | Defined in |
|---|---|---|---|
| `TrainingJobSpec` | client → coordinator | adapter (rank/alpha/targets), `LossSpec`, `OptimizerSpec`, `BudgetSpec`, `SchedulingSpec` | `schemas.py` |
| `ExternalTrajectoryBatch` | client → coordinator | tokenized RL rollouts + `old_logprobs` + rewards/advantages, `client_batch_id` | `schemas.py` |
| `TrainExample` | internal | one sequence: `input_ids`, masks, `labels`/logprobs, `slot` | `schemas.py` |
| `BatchLease` | store → plan | `payload_ref` (Ray ObjectRef) + `token_count` + version | `plan.py` |
| `TrainStepPlan` | coordinator → workers | the whole step: jobs, slots, leases, slot ops, exports | `plan.py` |
| `WorkerStepResult` | worker → coordinator | `ok`, per-adapter `metrics`, `written_files` | `results.py` |

---

## 5. The central loop (one engine step)

`serve_engine._run_training_loop` drives a build → execute → commit cycle. Data
is **leased**, not consumed, until the plan commits — so a worker failure loses
nothing.

```mermaid
sequenceDiagram
    autonumber
    participant DRV as Driver loop<br/>(serve_engine.py)
    participant CO as Coordinator
    participant SCH as Scheduler
    participant BS as BatchStore
    participant W as Megatron workers<br/>(all ranks)

    DRV->>CO: build_next_plan()
    activate CO
    CO->>SCH: accrue_deficits + select_training_jobs
    SCH-->>CO: SelectedJob[] (job_id, target_tokens)
    CO->>CO: _assign_slots (free slot or preempt victim)
    CO->>BS: lease_for_plan(job, target_tokens, plan_id)
    BS-->>CO: BatchLease[] (AVAILABLE -> LEASED)
    CO-->>DRV: TrainStepPlan (immutable) + mark jobs LEASED
    deactivate CO

    DRV->>W: execute_train_step_plan(ray.put(plan))
    activate W
    W->>W: prepare_slots (preempt/onload)
    W->>W: materialize leases -> packed microbatches
    W->>W: train_with_custom_forward_step (1 optimizer step)
    W->>W: export dirty adapters (if publish)
    W-->>DRV: WorkerStepResult[] (per rank)
    deactivate W

    DRV->>CO: commit_or_abort_plan(plan_id, results)
    activate CO
    alt all results ok
        CO->>BS: commit_plan (LEASED -> CONSUMED)
        CO->>CO: engine_step++, advance optimizer_step/tokens,<br/>apply slot state, finalize READY manifests
        CO-->>DRV: CommitResult(ok, published=[adapter_ready...])
    else any failure / timeout
        CO->>BS: abort_plan (LEASED -> AVAILABLE)
        CO->>CO: release leased jobs (no version advance)
        CO-->>DRV: CommitResult(ok=False, error)
    end
    deactivate CO

    DRV->>DRV: log W&B metrics (incl. train/num_jobs)
    opt online RL, every SYNC_EVERY steps
        DRV->>DRV: sync fresh adapters -> sglang
    end
```

---

## 6. Slots: paging adapters in and out

There are only `N` hot GPU slots (`ENGINE_N_ADAPTERS`, default 16) but possibly
hundreds of jobs. The coordinator pages adapters between **HOT** (resident in a
model slot) and **COLD** (checkpointed to disk).

```mermaid
flowchart LR
    subgraph SLOTS["Model: N physical MultiLoRA slots"]
        S0["slot 0: job-A"]
        S1["slot 1: job-B"]
        S2["slot 2: free"]
        S3["slot 3: job-C"]
    end
    COLD["COLD jobs (checkpoints on disk)<br/>job-D, job-E, ..."]

    COLD -->|"onload: init_adapter_slot, load_adapter, restore optimizer"| S2
    S1 -->|"preempt: checkpoint adapter+optimizer, clear slot"| COLD
```

**Onload vs preempt (executed by `AdapterSlotExecutor` on every rank):**

```mermaid
flowchart TB
    P["TrainStepPlan.preemptions / .onloads"] --> PS["prepare_slots()"]
    PS --> PRE["for each preemption:<br/>capture adapter + optimizer state<br/>write training checkpoint (sync)<br/>snapshot HF-PEFT (async, if inference)<br/>clear_adapter_slot + zero optimizer"]
    PS --> ON["for each onload:<br/>init_adapter_slot(rank, alpha)<br/>load_adapter + restore optimizer state<br/>reload_model_params"]
```

Slot assignment policy lives in `coordinator._assign_slots`: prefer a **free**
slot; otherwise ask the scheduler to `choose_preemption_victim` (idle-first,
least-under-served, longest-hogger, lowest-priority). A job pinned HOT for fewer
than `min_hot_steps` (default 2) cannot be preempted (`can_preempt`).

> Optimizer state survives slot moves because
> `iter_adapter_named_params_for_slot` uses **slot-independent** keys, so Adam
> moments captured from slot 1 restore cleanly into slot 2.

---

## 7. From leased payloads to a packed microbatch

This is the heart of "continuous batching": multiple adapters share one Megatron
step, and tokens must be **sorted by physical slot** so `MultiLoRALinear` can
route contiguous token ranges to the right adapter.

```mermaid
flowchart TB
    L["BatchLease[] (payload_ref per job)"] -->|ray.get| P["payloads<br/>(SFT TrainExample[] or RL ExternalTrajectoryBatch)"]
    P -->|_payload_to_examples + stamp slot| EX["TrainExample[] (slot-stamped)"]
    EX -->|one sequence per microbatch| PACK["pack_examples_by_slot()"]
    PACK --> MB["microbatch dict:<br/>tokens, attention_mask, loss_masks,<br/>adapter_token_counts, job_ranges,<br/>advantages/old_logprobs/rewards"]
```

**The slot-sorted layout invariant** (`continuous_packer.plan_packed_layout`):
flattened tokens are grouped by slot, and `adapter_token_counts[i]` equals the
number of contiguous tokens routed to slot `i`.

```text
flattened tokens:   [ slot 0 tokens | slot 1 tokens | slot 3 tokens | pad ]
adapter_token_counts = [ n0,            n1,           0,  n3+pad ]   (sums to total)
                                                      ^ slot 2 unused -> 0
```

Padding to a multiple is appended to the **last used slot** with
attention/loss masks zeroed, so it never affects gradients while keeping the
per-slot counts summing to the total token count.

> MVP detail: the materializer emits **one sequence per microbatch** (normal
> causal attention). Gradients accumulate across microbatches into disjoint
> per-slot params, producing one optimizer step.

---

## 8. The worker forward/backward + per-adapter metrics

```mermaid
flowchart TB
    MB["microbatches"] --> FS["make_continuous_forward_step()"]
    FS --> M["model(input_ids)<br/>set_tokens_per_adapter_slot(adapter_token_counts)"]
    M --> LF{"loss_type?"}
    LF -->|sft| SFT["continuous_sft_loss"]
    LF -->|grpo/ppo| RL["gather_current_logprobs<br/>-> continuous_rl_loss"]
    SFT --> ACC["per_job_accum<br/>(loss_sum, loss_tokens, n_seq, reward_sum)"]
    RL --> ACC
    ACC -->|DP all-reduce| RED["_reduce_per_job_metrics"]
    RED --> OUT["metrics per job:<br/>job_id/train/loss, reward, tokens,<br/>n_sequences, avg_response_len<br/>+ aggregates: train/total_tokens, etc."]
```

Per-adapter raw sums are accumulated across microbatches on the last pipeline
stage, then data-parallel reduced after the step so per-adapter loss/reward are
correct across DP/CP shards. The driver maps these into W&B under each adapter's
own `{job_id}/...` section, all on a shared `engine/step` x-axis (plus
step-level aggregates like `train/num_jobs`).

---

## 9. End-to-end walkthrough A — an SFT job

```mermaid
sequenceDiagram
    autonumber
    actor U as Client
    participant API as HTTP API
    participant CO as Coordinator
    participant DRV as Driver loop
    participant W as Workers

    U->>API: POST /v1/training/jobs (TrainingJobSpec, loss=sft)
    API->>CO: submit_job(spec)
    CO->>CO: ArtifactStore.create_v0 -> version 0
    CO-->>U: {job_id, adapter_version: 0, adapter_uri}

    U->>API: POST /{job}/sft_examples (input_ids, loss_mask, labels)
    API->>CO: submit_sft_examples -> BatchStore.put (AVAILABLE)
    CO-->>U: {accepted, batch_id, token_count}

    loop driver cycle
        DRV->>CO: build_next_plan() -> plan (job paged HOT, lease)
        DRV->>W: execute_train_step_plan(plan)
        W-->>DRV: WorkerStepResult(ok, metrics, written_files)
        DRV->>CO: commit_or_abort_plan
        CO->>CO: optimizer_step++, and if publish_every then<br/>finalize READY manifest, advance latest_published_version
    end

    U->>API: GET /{job} (poll)
    API-->>U: {trained_steps, last_loss, latest_published_version, latest_adapter_uri}
```

---

## 10. End-to-end walkthrough B — an online RL (GRPO) job

This is the loop in `examples/training_engine/math_rl_clients.py`: **sample →
grade client-side → submit scored rollouts → train**, repeated. Rewards are
computed entirely on the client; the engine only generates and trains.

```mermaid
sequenceDiagram
    autonumber
    actor U as Client (math_rl_clients.py)
    participant API as HTTP API
    participant GEN as SglangGenerator -> router
    participant CO as Coordinator
    participant DRV as Driver loop
    participant W as Workers

    Note over U: register job (loss=grpo) via create_lora_training_client

    loop until trained_steps >= TARGET_STEPS
        U->>API: POST /{job}/sample (prompt_ids, n_samples=8)
        API->>GEN: generate (lora_path = adapter if loaded, else base)
        GEN-->>API: rollouts [{prompt_ids, response_ids, response_logprobs, text}]
        API-->>U: {adapter_version, rollouts}

        U->>U: reward = grade_answer_verl(text, gold)  # client-side
        U->>U: GRPO group-relative advantages from rewards
        U->>API: POST /{job}/trajectory_batches (input_ids, action_mask,<br/>old_logprobs, advantages, adapter_version)
        API->>CO: submit_trajectory_batch -> validate(version, max_policy_lag) -> put
    end

    par training proceeds independently
        DRV->>CO: build_next_plan -> execute -> commit (RL loss)
        CO->>CO: advance version, pending publish
        DRV->>DRV: every SYNC_EVERY steps: sync fresh adapter -> sglang
    end
```

**Why `adapter_version` matters here:** submitted rollouts are validated against
the job's *published* version and rejected if the policy lag exceeds
`max_policy_lag` (default 4) — this bounds off-policyness. The `/sample`
endpoint only routes to a job's adapter once it's actually loaded in sglang;
before that, the fresh (untrained) adapter is equivalent to the base policy, so
it samples the base model.

---

## 11. Scheduling & fairness (deficit-weighted)

Fairness is measured in **trained tokens**. Each runnable/backlogged job accrues
a per-tick credit proportional to its priority; selection spends that credit.

```mermaid
flowchart TB
    A["accrue_deficits()<br/>deficit += base_quantum_tokens * priority<br/>(for jobs RUNNING + READY + have data)"] --> B["select_training_jobs()"]
    B --> F1["filter: is_runnable AND deficit >= min_tokens_per_train_quantum"]
    F1 --> F2["drop jobs at max_consecutive_steps<br/>(unless that would starve the engine)"]
    F2 --> SORT["sort by:<br/>1. deficit minus cold-load penalty (desc)<br/>2. HOT before COLD<br/>3. fewer consecutive_steps<br/>4. higher priority"]
    SORT --> PICK["greedily fill until<br/>max_adapters_per_step or<br/>max_train_tokens_per_step budget"]
    PICK --> OUT["SelectedJob[] with target_tokens"]
```

`target_tokens` per job is `min(remaining_budget, tokens_per_update,
ready_train_tokens, deficit_tokens)`. Cold jobs pay a `base_quantum_tokens`
penalty so the engine doesn't page in a cold adapter just to run a tiny quantum.

**When does a step dispatch at all?** (`_should_dispatch`) — when accumulated
ready tokens hit `max_train_tokens_per_step`, OR the oldest waiting job has
waited `max_batch_wait_s` (default 0.25s), OR workers are otherwise idle.

---

## 12. Data safety: leases, idempotency, backpressure

```mermaid
stateDiagram-v2
    direction LR
    [*] --> AVAILABLE: put()
    AVAILABLE --> LEASED: lease_for_plan()
    LEASED --> CONSUMED: commit_plan()
    LEASED --> AVAILABLE: abort_plan() (worker fail/timeout)
    AVAILABLE --> [*]: drop_job()
```

- **Leases, not pops:** an aborted plan returns its batches to `AVAILABLE`, so a
  worker failure or step timeout never loses data.
- **Idempotency:** `client_batch_id` dedupes retried submissions
  (`client_id_index`).
- **Backpressure:** `QueueLimits` rejects submissions past
  `max_ready_tokens_per_job` (`job_backpressure`) or
  `max_ready_tokens_global` (`engine_backpressure`).
- **Payloads stay out of the coordinator:** only a `payload_ref` (Ray ObjectRef)
  and metadata live in the store; workers `ray.get` the payload at materialize
  time.

---

## 13. Versioning & artifacts

```mermaid
flowchart LR
    V0["submit_job -> v0 manifest<br/>(base / no adapter), READY immediately"] --> STEP["committed + materialized step"]
    STEP -->|publish_every_steps| FIN["ArtifactStore.finalize_manifest<br/>(WRITING -> READY) from written_files"]
    FIN --> ADV["advance latest_published_version<br/>+ latest_adapter_uri"]
    ADV --> EVT["CommitResult.published += adapter_ready event"]
```

Internal `optimizer_step` advances on every committed step; the **public**
`latest_published_version` only advances once a durable manifest is `READY`. RL
submissions are always validated against the *published* version.

---

## 14. Module map (where to look)

```mermaid
flowchart LR
    subgraph CP["Control plane (CPU, torch-free, unit-tested)"]
        schemas["schemas.py<br/>contracts + job state"]
        scheduler["scheduler.py<br/>fairness"]
        batch_store["batch_store.py<br/>leases"]
        artifact_store["artifact_store.py<br/>manifests"]
        coordinator["coordinator.py<br/>the only writer"]
        plan["plan.py / results.py"]
        clientmod["client.py / api_server.py"]
    end
    subgraph EP["Execution plane (GPU, inside Megatron actor)"]
        mexec["megatron_executor.py"]
        slot["adapter_slot_executor.py"]
        mat["batch_materializer.py"]
        packer["continuous_packer.py"]
        loss["continuous_loss.py"]
        ckpt["checkpoint_store.py"]
    end
    subgraph IP["Inference plane (optional)"]
        gen["generation.py"]
        mlc["ray/multi_lora_controller.py"]
    end
    coordinator --> plan --> mexec
    mexec --> slot
    mexec --> mat --> packer
    mexec --> loss
    slot --> ckpt
    clientmod --> coordinator
    gen --> mlc
```

| Concern | File |
|---|---|
| Job state, specs, validation, policies | `schemas.py` |
| Deficit fairness, victim selection | `scheduler.py` |
| Lease / commit / abort, idempotency, backpressure | `batch_store.py` |
| `v0` + atomic WRITING→READY manifests | `artifact_store.py` |
| The single writer; plan build + commit | `coordinator.py` |
| Immutable plan / result types | `plan.py`, `results.py` |
| Thin client + HTTP surface | `client.py`, `api_server.py` |
| Worker step orchestration + metrics | `megatron_executor.py` |
| Slot onload/preempt/export | `adapter_slot_executor.py` |
| Leases → slot-stamped microbatches | `batch_materializer.py` |
| Slot-sorted packing + `adapter_token_counts` | `continuous_packer.py` |
| Per-job SFT / RL loss | `continuous_loss.py` |
| Online-RL generation wrapper | `generation.py` |
| Driver loop + W&B logging | `examples/training_engine/serve_engine.py` |

---

## 15. Running it

- `examples/training_engine/run_engine_serve.sh` — launch the engine service.
- `examples/training_engine/serve_engine.py` — coordinator + workers + HTTP API +
  driver loop (env knobs documented in its module docstring, e.g.
  `ENGINE_N_ADAPTERS`, `ENGINE_ENABLE_GENERATION`, `ENGINE_SYNC_EVERY`).
- `examples/training_engine/math_rl_clients.py` — many-adapter GRPO client over
  three math datasets (the online-RL walkthrough in §10).
- `examples/training_engine/engine_client_demo.py` — minimal client demo.
