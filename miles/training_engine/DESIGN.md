# Continuous-Batched MultiLoRA Training Engine (v2)

This implements `revised_v2_multilora_training_engine_design.md`. The guiding
principle:

```text
The coordinator decides.
Megatron workers execute.
The coordinator commits.
```

A single central `TrainingCoordinator` (Ray actor) owns all global state and
emits immutable `TrainStepPlan`s. Every Megatron worker receives the *same* plan
and executes it with the existing Miles/Megatron stack. There is no
controller/scheduler/store loop inside each worker (the old `runner.py` /
`adapter_pager.py` / `job_controller.py` are gone).

## Two planes

```text
Control plane (CPU, single-writer, unit-tested):
  schemas.py        orthogonal job state, specs, validation, BatchingPolicy/QueueLimits
  scheduler.py      deficit-weighted fairness (select / accrue / preemption victim)
  batch_store.py    lease/commit/abort data leases, idempotency, backpressure
  artifact_store.py v0 + atomic WRITING->READY adapter manifests
  coordinator.py    TrainingCoordinator: the only writer of global state
  plan.py / results.py   immutable plan + worker/commit result types
  client.py / api_server.py   thin client + optional HTTP surface

Execution plane (GPU, inside MegatronTrainRayActor):
  megatron_executor.py     MegatronPlanExecutor.execute(plan) -> WorkerStepResult
  adapter_slot_executor.py prepare_slots(plan): preempt/onload; export dirty adapters
  batch_materializer.py    leased payloads -> slot-stamped microbatches
  continuous_packer.py     slot-sorted layout + adapter_token_counts
  continuous_loss.py       per-job SFT / RL loss; gather_current_logprobs
  checkpoint_store.py      per-rank adapter+optimizer preempt/resume shards
```

## Three identities

```text
job_id           durable user-facing job
slot             physical MultiLoRA adapter index (can change over time)
adapter_version  immutable published artifact version (only advances when READY)
```

External rollouts reference `(job_id, adapter_version)`, never a physical slot.

## Orthogonal job state

Instead of one mixed enum, `TrainingJobRuntime` carries `Lifecycle`,
`Residency`, `Readiness`, and `Execution` independently. `is_runnable` and
`can_preempt` are predicates over these. Internal `optimizer_step` is separate
from public `latest_published_version`.

## One cycle of the central loop (`engine_driver.py`)

```text
1. coordinator.build_next_plan()
     accrue deficits -> should_dispatch? -> scheduler.select_training_jobs
     -> _assign_slots (free slot or central preemption victim)
     -> batch_store.lease_for_plan (data leased, not consumed)
     -> immutable TrainStepPlan (selected jobs, slots, leases, onloads,
        preemptions, per-job LossSpec, export requests)
2. RayTrainGroup.execute_train_step_plan(ray.put(plan))  # same plan to all ranks
     each worker: prepare_slots -> materialize -> train_with_custom_forward_step
     -> export dirty adapters -> WorkerStepResult
3. ray.wait(results, timeout)  # all-or-nothing
4. coordinator.commit_or_abort_plan(plan_id, results)
     all ok  -> commit leases, advance optimizer_step/trained_tokens, apply slot
                state, finalize READY manifests, advance latest_published_version
     any bad -> abort leases (data returned to AVAILABLE), no version advance
```

## Data safety: leases, not pops

`BatchStore` leases batches to a plan. A committed plan consumes them; an aborted
plan (worker failure/timeout) returns them to `AVAILABLE`, so nothing is lost.
`client_batch_id` makes submission idempotent; `QueueLimits` provides
backpressure (`job_backpressure` / `engine_backpressure`).

## Versioning / artifacts

`submit_job` publishes a concrete `v0` (base/no-adapter) manifest immediately.
After a committed+materialized step, the coordinator finalizes a `READY`
manifest from the worker-written files, then advances `latest_published_version`.
`validate_trajectory_batch` checks submitted RL data against the *published*
version and `max_policy_lag`.

## Worker train step

`model.train_with_custom_forward_step` runs the normal Miles/Megatron train
lifecycle (grad scaling, `finalize_model_grads_func`, `optimizer.step` overflow
handling, scheduler step, grad-buffer cleanup) with a caller-supplied
`forward_step_func`. The continuous forward step only sets
`set_tokens_per_adapter_slot`, runs the model, and computes the continuous loss.

MVP uses one sequence per microbatch (safe causal attention); microbatches
accumulate into one optimizer step over disjoint per-slot adapter params. RL
current-policy logprobs come from `gather_current_logprobs` (token-position
aligned, TP-correct), matching the external `old_logprobs` convention.

## Optimizer state across slot moves

`iter_adapter_named_params_for_slot` uses *slot-independent* keys
(`...adapter.<param>`), so Adam state captured from one slot restores into a
different slot when a job migrates.

## What is verified vs. not

- Verified on CPU: schemas/validation, scheduler fairness/selection/preemption,
  batch-store lease/commit/abort/idempotency/backpressure, coordinator
  submit/plan/commit/abort/publish/preemption/completion (see
  `tests/training_engine/`).
- Not yet verified on GPU (needs a cluster run): the exact loss/normalizer
  scaling in the Megatron schedule contract, RL logprob reduction under
  TP>1/DP>1/CP, and slot preempt/onload checkpoint round-trips. These are
  isolated in the execution-plane modules.

## MVP restrictions (per design)

Homogeneous optimizer config across jobs, SFT + GRPO/PPO (DPO stubbed),
one active GPU plan, in-memory state, Ray ObjectRefs for payloads,
one-sequence microbatching.
