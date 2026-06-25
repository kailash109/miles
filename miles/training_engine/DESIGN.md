# Continuous MultiLoRA Training Engine — Design

A long-lived training **service** that accepts LoRA jobs at arbitrary times,
fairly schedules them onto a fixed pool of GPU-resident MultiLoRA slots, trains
them with Miles/Megatron, and publishes immutable, versioned adapter artifacts.
It is a control/scheduling layer built *on top of* the existing
Miles/Megatron-Bridge MultiLoRA execution stack — it does not reimplement model
construction, parallelism, the MultiLoRA forward, or adapter export.

---

## 1. Mental model

### 1.1 Two planes

```
                          ┌──────────────────────────────────────────────┐
                          │            CONTROL PLANE (pure python)         │
   client / external      │  no torch, no CUDA, unit-tested on CPU         │
   rollout system  ─────► │                                                │
                          │  schemas · job_controller · scheduler ·        │
                          │  trajectory_store · dataset_worker ·           │
                          │  continuous_packer.plan_packed_layout ·        │
                          │  client · events · metrics                     │
                          └───────────────────────┬────────────────────────┘
                                                  │ decisions: which jobs,
                                                  │ which slots, how many tokens
                                                  ▼
                          ┌──────────────────────────────────────────────┐
                          │       GPU PROCESS (inside a Miles train actor) │
                          │  imports torch / Megatron                      │
                          │                                                │
                          │  runner · adapter_pager · continuous_loss ·    │
                          │  checkpoint_store · artifact_publisher ·       │
                          │  continuous_packer.pack_examples_by_slot       │
                          │                                                │
                          │  ── reuses ──►  Megatron MultiLoRA model,      │
                          │                 set_tokens_per_adapter_slot,   │
                          │                 get_forward_backward_func,     │
                          │                 save_multi_lora_checkpoints     │
                          └────────────────────────────────────────────────┘
```

The split is deliberate: the control plane is the part with interesting logic
(state machine, fairness, validation, packing layout) and it can be tested
without a GPU. The GPU process is thin glue around Megatron primitives.

### 1.2 Three identities

The single most important abstraction is that these are **separate**:

```
logical job_id        durable, user-facing       "job_alpha"
   │  (mapped at train time by the pager)
   ▼
physical slot         index into MultiLoRALinear.adapters[i]   3
   │  (slot can change over a job's life: preempt → reload elsewhere)
   ▼
adapter_version       immutable published checkpoint           job_alpha/step_12
```

A job migrates across slots (`job_alpha → slot 3 → preempted → slot 7`). All
externally generated RL data is tagged with `(job_id, adapter_version)`, never a
slot — the engine resolves `job_id → current hot slot` only at the moment of
training.

---

## 2. The objects and what their methods actually do

### 2.1 `schemas.py` — contracts (no torch)

**`TrainingJobSpec`** (frozen) is the immutable description a client submits:
`adapter` (rank/alpha/target modules/init), `dataset`, `loss`, `optimizer`,
`budget`, `scheduling`, `output_uri`.

**`TrainingJobRuntime`** is the mutable bookkeeping the engine maintains for a
live job. The fields that drive behavior:

- `slot` — current hot slot, or `None` when cold.
- `ready_train_tokens` — how many trainable (loss/action) tokens are queued.
- `deficit_tokens` — scheduling credit (see scheduler).
- `current_adapter_version` — bumped once per committed step.
- `consecutive_steps` — used to stop one job from hogging the engine.
- `dirty_since_publish` — set on commit, cleared on publish.
- `hot_since_engine_step` — when it became hot (used by preemption to honor a
  minimum residency).

**`ExternalTrajectoryBatch`** is the tokenized RL payload from external
inference: `input_ids / attention_mask / action_mask`, the behavior-policy
`old_logprobs`, plus `rewards / advantages / ref_logprobs / group_ids`, and the
`adapter_version` + compatibility hashes that produced it.

**`TrainExample`** is the *internal* per-sequence unit. Both the SFT dataset
worker and RL ingestion normalize to it, so the packer never needs to know where
data came from.

**`validate_trajectory_batch(job, batch)`** enforces the off-policy contract.
This is the function that makes asynchronous RL safe:

```python
if batch.adapter_version > job.current_adapter_version:
    raise TrajectoryValidationError("future adapter_version")     # came from a version we never published
lag = job.current_adapter_version - batch.adapter_version
if lag > spec.budget.max_policy_lag:
    raise TrajectoryValidationError("policy lag exceeds max")       # too stale to train on
# shapes of input_ids/attention_mask/action_mask must match;
# old_logprobs must match action_mask; action_mask.sum() > 0;
# lora_config_hash must match if provided.
```

`count_action_tokens` (sum of the action mask, works on tensors or nested lists)
is how the engine measures "how much trainable work" a batch carries.

### 2.2 `job_controller.py` — `TrainingJobController`

The durable registry and the **only** place job state transitions legally
happen. It is a plain class (so the state machine is unit-testable) and is
wrapped as a Ray actor at runtime by `make_training_job_controller`. It owns
`jobs: {job_id → TrainingJobRuntime}`, `free_slots`, `hot_job_by_slot`, and an
`engine_step` counter. It **never touches Megatron tensors** — GPU-affecting
transitions are performed by the runner/pager, then *reflected back* here.

Key methods and their effect:

- **`submit_job(spec)`** — validates base-model match + positive rank/alpha,
  registers the runtime, sets `WAITING_FOR_DATA`.
- **`mark_train_batch_ready(job_id, batch_id, token_count)`** — called after
  data is ingested; grows `ready_train_tokens` and flips
  `WAITING_FOR_DATA/COLD_READY/HOT_IDLE → TRAIN_READY`.
- **`reserve_free_slot(job_id)`** — pops the lowest free slot, records
  `hot_job_by_slot`, sets `slot` + `LOADING`; returns `None` if full (the pager
  then preempts).
- **`mark_hot` / `mark_active_step`** — `HOT_IDLE` after the adapter is loaded;
  `ACTIVE_STEP` while a forward/backward is in flight (so the scheduler and
  preemption skip it).
- **`commit_step(updates)`** — the post-training-step bookkeeping. This is where
  fairness "spending" and versioning happen:

  ```python
  self.engine_step += 1
  for job_id, u in updates.items():
      rt = self.jobs[job_id]
      trained = u["trained_tokens"]
      rt.trained_steps += 1
      rt.trained_tokens += trained
      rt.current_adapter_version += 1            # new artifact version exists
      rt.ready_train_tokens = max(0, rt.ready_train_tokens - trained)
      rt.deficit_tokens -= trained               # spend scheduling credit
      rt.consecutive_steps += 1
      rt.dirty_since_publish = True
      rt.state = HOT_IDLE if rt.ready_train_tokens > 0 else WAITING_FOR_DATA
  ```

- **`mark_cold(job_id, ckpt_uri)`** — preemption: frees the slot, clears
  `consecutive_steps`, remembers `cold_checkpoint_uri`, and sets `TRAIN_READY`
  (data left) or `COLD_READY` (drained).
- **`mark_published` / `complete_job` / `budget_exhausted`** — clear the dirty
  flag; finalize (free slot, terminal `COMPLETED`); check `max_steps` /
  `max_train_tokens`.

### 2.3 `scheduler.py` — `ContinuousTrainingScheduler`

Deficit-weighted fair queuing — the same idea as DRR/weighted fair queuing, but
the spent quantity is *trained tokens*. Credit accrues to runnable jobs and is
spent when they train, so a job with a huge backlog can't monopolize the engine.

- **`accrue_deficits(jobs)`** — each round, every runnable job with ready data
  earns `base_quantum_tokens * priority` credit:

  ```python
  if job.state in {TRAIN_READY, HOT_IDLE, COLD_READY} and job.ready_train_tokens > 0:
      job.deficit_tokens += self.base_quantum_tokens * job.spec.scheduling.priority
  ```

  Higher priority ⇒ credit grows faster ⇒ selected more often ⇒ proportionally
  more trained tokens.

- **`select_training_jobs(jobs)`** — returns `[(job_id, target_tokens)]`. It
  filters to jobs that are runnable, not mid-step, and have *both* enough ready
  data and enough credit (`>= min_tokens_per_train_quantum`), then sorts:

  ```python
  candidates.sort(key=lambda j: (
      -j.deficit_tokens,                 # most under-served first (fairness)
      0 if j.slot is not None else 1,    # prefer already-hot jobs (avoid reloads)
      j.consecutive_steps,               # then jobs that have run least recently
      -j.spec.scheduling.priority,
  ))
  ```

  and greedily fills the step up to `max_adapters_per_step` and
  `train_tokens_per_step`, giving each job
  `min(remaining, tokens_per_update, ready_train_tokens, deficit_tokens)` tokens.
  `max_consecutive_steps` excludes a job that has trained too many steps in a row.

- **`choose_preemption_victim(hot_jobs, incoming, engine_step)`** — when no slot
  is free, pick a hot job to evict. It refuses `ACTIVE_STEP` and
  non-preemptible jobs, refuses jobs that haven't met `min_hot_steps` (anti-
  thrash), then prefers **idle** (no ready data), under-credited, long-resident,
  low-priority victims:

  ```python
  candidates.sort(key=lambda j: (
      j.ready_train_tokens > 0,   # False (idle) sorts first
      j.deficit_tokens,           # least credit first
      -j.consecutive_steps,       # longest hogger first
      j.spec.scheduling.priority, # lowest priority first
  ))
  ```

### 2.4 `trajectory_store.py` — `TrajectoryStore` + `ExampleStore`

Two FIFO per-job queues. `TrajectoryStore` holds raw `ExternalTrajectoryBatch`es
(RL); `ExampleStore` holds pre-tokenized `TrainExample`s (SFT). Both expose
`pop_for_job(job_id, target_tokens)` which drains whole items until the trainable
token target is met (never splitting an item), and `pending_tokens` for
accounting. Swapping these for an object store later doesn't touch callers.

### 2.5 `dataset_worker.py` — SFT tokenization

`build_sft_example` turns `{prompt, completion}` into a `TrainExample` with
completion-only supervision. The label/mask construction is the subtle part —
labels are next-token targets, prompt positions and the final position get
`-100` (ignored by cross-entropy), and `loss_mask` mirrors that:

```python
input_ids = prompt_ids + completion_ids        # (+ eos)
loss_mask = [0]*n_prompt + [1]*(n - n_prompt)
labels = input_ids[1:] + [-100]                # next-token targets
labels = [tok if loss_mask[i] and i < n-1 else -100 for i, tok in enumerate(labels)]
loss_mask = [1 if labels[i] != -100 else 0 for i in range(n)]   # keep aligned
```

It takes an injected `encode` callable, so this module stays tokenizer- and
torch-free (the actor supplies its HF tokenizer at runtime).

### 2.6 `continuous_packer.py` — the slot-sort invariant

`MultiLoRALinear` requires the flattened tokens of a microbatch to be **sorted by
physical slot**, with `adapter_token_counts[i]` = number of contiguous tokens for
slot *i*.

- **`plan_packed_layout(examples, max_slots, loss_type)`** (pure, the unit-tested
  core) sorts examples by slot, builds the per-slot token counts and per-job
  token spans, and asserts the invariant:

  ```python
  ordered = sorted(examples, key=lambda ex: (int(ex.slot), -len(ex.input_ids)))
  for ex in ordered:
      n = len(ex.input_ids)
      adapter_token_counts[ex.slot] += n
      job_ranges[ex.job_id].append((cursor, cursor + n))
      job_token_counts[ex.job_id] += sum(ex.loss_mask)   # trainable tokens (accounting)
      cursor += n
  assert sum(adapter_token_counts) == total_tokens
  ```

- **`pack_examples_by_slot(...)`** wraps that into torch tensors (`tokens`,
  `loss_masks`, `adapter_token_counts`, and either `labels` for SFT or
  `rollout_log_probs`/`advantages`/... for RL).

### 2.7 `continuous_loss.py` — per-**job** normalization

Loss is normalized per logical job and then **summed** (not globally averaged).
Because each job's slot is a disjoint set of params, summing per-job means gives
each adapter its own correctly-scaled gradient and stops the largest job from
dominating.

- **`continuous_sft_loss(logits, batch, job_specs)`** — token cross-entropy,
  reduced per job over its `loss_mask`, TP-all-reduced:

  ```python
  token_losses = cross_entropy(logits.reshape(-1, V), labels.clamp_min(0), reduction="none")
  for job_id, ranges in batch["job_ranges"].items():
      s = sum((token_losses[a:b] * loss_masks[a:b]).sum() for a, b in ranges)
      c = sum(loss_masks[a:b].sum() for a, b in ranges)
      total += _tp_all_reduce_sum(s) / _tp_all_reduce_sum(c).clamp_min(1)
  ```

- **`continuous_rl_loss(model_logprobs, batch, job_specs)`** — clipped
  policy-gradient (GRPO/PPO) from external `old_logprobs`:

  ```python
  log_ratio = pi_logp - old_logp
  ratio = torch.exp(log_ratio)
  clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
  policy_loss = -torch.minimum(ratio * adv, clipped * adv)
  # + optional KL to a reference when kl_ref in {base, provided}
  ```

  Its first argument is **per-token current-policy logprobs**, not logits — the
  runner produces those (§2.11).

### 2.8 `adapter_pager.py` — `AdapterPager` (logical job → physical slot)

This performs the actual GPU slot transitions.

- **`ensure_hot(job, all_jobs)`** — if the job has a slot, return it. Otherwise
  reserve a free slot; if none, `choose_preemption_victim`, `preempt` it, and
  reserve the freed slot. Then load the adapter into the slot:

  ```python
  slot = ray.get(self.controller.reserve_free_slot.remote(job.spec.job_id))
  if slot is None:
      self.preempt(self._choose_victim(job, all_jobs))
      slot = ray.get(self.controller.reserve_free_slot.remote(job.spec.job_id))
  self._load_job_into_slot(job, slot)
  ray.get(self.controller.mark_hot.remote(job.spec.job_id, slot))
  ```

- **`_load_job_into_slot`** — `init_adapter_slot(model, slot, rank, alpha)`
  (Megatron-Bridge), then if there is a cold checkpoint or warm-start URI,
  `load_adapter(...)` the weights and `restore_optimizer_state_for_adapter(...)`
  the Adam moments, and `optimizer.reload_model_params()`.
- **`preempt(job)`** — `_save_hot_job_checkpoint` (extract this rank's adapter
  tensors + capture the slot's optimizer state) → `clear_adapter_slot` +
  `zero_optimizer_state_for_adapter` + `reload_model_params` → `mark_cold`.

The capture/restore of optimizer state keyed by **stable parameter names** is the
piece that makes preemption correct (next section).

### 2.9 `multi_lora.py` additions (in `miles/backends/megatron_utils`)

The pager needs to move a slot's Adam state to/from CPU across preemption, and
across *different* physical slots. Python `id(param)` is not stable across slot
moves/process restarts, so the new helpers key by a stable module/param path:

```python
def iter_adapter_named_params_for_slot(model, idx):
    for module_prefix, module in iter_named_multi_lora_modules(model):   # "chunk0.<path>"
        for name, p in module.adapters[idx].named_parameters():
            yield f"{module_prefix}.adapters.{idx}.{name}", p

def capture_optimizer_state_for_adapter(optimizer, model, idx):
    # map id(main_param) -> stable_name, then snapshot Adam state (CPU) under stable_name
def restore_optimizer_state_for_adapter(optimizer, model, idx, captured):
    # map stable_name -> current param, then load Adam state onto the right device
```

These generalize the pre-existing `zero_optimizer_state_for_adapter`.

### 2.10 `checkpoint_store.py` / `artifact_publisher.py`

- **`CheckpointStore`** — internal preempt/resume state, per TP/PP rank
  (`write_/load_training_checkpoint`, `extract_megatron_adapter_state`,
  `load_adapter_init`). This is private engine state, not a user artifact.
- **`AdapterArtifactPublisher.publish(job, model)`** — the user-facing output.
  It reuses Miles' existing exporter rather than re-deriving HF/PEFT conversion:

  ```python
  config = AdapterConfig(name=..., rank=..., alpha=..., dir=Path(output_uri), slot=job.slot)
  save_multi_lora_checkpoints(self.args, model, version, {name: config})  # writes safetensors + adapter_config.json
  self._write_metadata(job, version_dir)   # job_id, version, base_model, lora_config_hash, ... (writer rank only)
  ```

  Output: `{output_uri}/checkpoints/step_{version}/{adapter_model.safetensors,
  adapter_config.json, metadata.json}`.

### 2.11 `runner.py` — `ContinuousTrainingRunner` (the loop)

Owns the training-quantum loop and is the only object that calls Megatron
forward/backward. `_ctl(method, *args)` adapts a plain controller (in-actor) or a
Ray actor handle (`ray.get(getattr(ctl, m).remote(*args))`). It can build its own
model or reuse one a train actor already built.

- **`run_bounded(max_engine_steps)`** — repeatedly `_engine_step()`, stop when all
  jobs are terminal or no one has data.
- **`_engine_step()`** — `snapshot_jobs → accrue_deficits → select_training_jobs`;
  if anything was selected, `_train_selected`.
- **`_train_selected(selected, jobs)`** — make each selected job hot via the pager,
  re-snapshot (slots changed), pull each job's examples and stamp them with the
  job's slot, then train homogeneous loss groups.
- **`_train_examples(examples, jobs)`** — `plan_packed_layout` for ordering +
  accounting, build **one microbatch per example** (one-hot
  `adapter_token_counts`), run the step, `commit_step`, `_maybe_publish`.

  > Why one example per microbatch: packing several sequences into a single
  > `[1, T]` forward without block-diagonal attention would let tokens from
  > different examples/adapters attend to each other. One example per microbatch
  > keeps normal causal attention; grads still accumulate into the disjoint
  > per-slot params across microbatches within one `optimizer.step()`, so it is
  > still "one fused optimizer step over many adapters."

- **`_run_megatron_step(microbatches, job_specs, loss_type)`** — defines the
  Megatron forward step and runs it:

  ```python
  def forward_step(data_iterator, model):
      micro = next(data_iterator)
      set_tokens_per_adapter_slot(model, micro["adapter_token_counts"].to(device))  # route tokens to slots
      tokens = micro["tokens"].to(device).unsqueeze(0)                              # [1, seq]
      logits = model(input_ids=tokens, position_ids=None, attention_mask=None)

      def loss_fn(output):
          if loss_type == "sft":
              return continuous_sft_loss(output, micro, job_specs)
          model_logprobs = self._token_logprobs(output, micro["tokens"].to(device))
          return continuous_rl_loss(model_logprobs, micro, job_specs)
      return logits, loss_fn

  get_forward_backward_func()(forward_step_func=forward_step, ...)   # Miles/Megatron schedule
  self.optimizer.step(); self.opt_scheduler.step(1)
  for chunk in self.model: chunk.zero_grad_buffer()                  # Miles grad hygiene
  ```

- **`_token_logprobs(logits, tokens)`** — the RL bridge. The model returns
  *logits*, but the policy-gradient loss needs the *logprob of the taken token*
  at each position, computed TP-correctly. It shifts (logits[t] predicts token
  t+1) and reuses Miles' vocab-parallel `calculate_log_probs_and_entropy`:

  ```python
  seq_logits = logits.reshape(-1, V).float()              # [T, V]
  log_prob, _ = calculate_log_probs_and_entropy(
      seq_logits[:T-1], tokens[1:T], mpu.get_tensor_model_parallel_group(), ...)
  lp_full = zeros(T); lp_full[1:T] = log_prob.squeeze(-1) # aligned to [T], lp[0]=0
  return lp_full
  ```

  SFT skips this — `cross_entropy(logits, labels)` already does log-softmax +
  gather internally.

### 2.12 `client.py`, `events.py`, `metrics.py`, `args.py`, `api_server.py`

- `build_job_spec(...)` — pure constructor from dict payloads.
- `TrainingClient` — `create_lora_training_job`, `submit_trajectory_batch`
  (validates then stores then marks ready), `cancel_job`.
- `EventLog` / `MetricsRegistry` — in-memory lifecycle events / counters.
- `args.add_training_engine_arguments` — the `--training-engine*` / `--continuous-*`
  flags (not yet wired into Miles' arg parser).
- `api_server.build_app` — optional FastAPI surface (lazy import).

---

## 3. How it plugs into Miles/Megatron (where the engine and the model meet)

The engine runs **inside** a Miles train actor and reuses the actor's model:

```
engine_driver.py  (ray job)
  ├─ create_multi_lora_controller(...)         # legacy hot-slot actor, created EMPTY
  │     (so initialize_multi_lora_model_and_optimizer's load_pending_adapters() works)
  ├─ allocate_train_group(...) -> RayTrainGroup
  │     └─ MegatronTrainRayActor(s)  [one per GPU]
  │           └─ .init():
  │                init(args)                                   # torch.dist + mpu (Miles)
  │                initialize_multi_lora_model_and_optimizer()  # builds MultiLoRA model, N slots
  │                self.model / self.optimizer / self.opt_param_scheduler
  └─ await actor_model.run_continuous_sft_engine(engine_cfg)    # broadcast to every rank
         └─ MegatronTrainRayActor.run_continuous_sft_engine():
              build controller + stores + scheduler (in-process)
              tokenize records with self.tokenizer
              ContinuousTrainingRunner(args, controller, ..., model=self.model, ...)
              runner.run_bounded(max_engine_steps)
```

What is reused unchanged:

| Need | Reused Miles/Megatron piece |
|---|---|
| Build N-slot MultiLoRA model | `initialize_multi_lora_model_and_optimizer` (Megatron-Bridge `MultiLoRA` hook) |
| Allocate the model with `n_adapters` slots | `--multi-lora-n-adapters` + `--multi-lora-dir empty_adapters` (no preloaded adapters) |
| Route tokens to slots in forward | `set_tokens_per_adapter_slot` |
| Slot init/load/clear/expose | `init_adapter_slot` / `load_adapter` / `clear_adapter_slot` / `expose_adapter_slot` |
| Forward/backward schedule | `get_forward_backward_func` |
| TP-correct per-token logprobs | `calculate_log_probs_and_entropy` |
| HF/PEFT adapter export | `save_multi_lora_checkpoints` |
| Process placement / actors | `create_placement_groups`, `allocate_train_group`, `RayTrainGroup`, `MegatronTrainRayActor` |

**SPMD note.** `run_continuous_sft_engine` runs on every rank. Each rank runs an
identical in-process engine over identically-tokenized data, with job identity
keyed by adapter **name** (not a per-rank uuid). So all ranks make the same
scheduling/slot decisions and build the same microbatches, and the Megatron
collectives line up. No cross-process model RPC is needed — engine and model
share the process.

---

## 4. Dataflow: a sample external-rollout RL run with concurrent requests

Two jobs `J_a`, `J_b` train at once; an external inference system generates their
trajectories.

```
 EXTERNAL INFERENCE            TrainingClient / Controller / Store        RUNNER (GPU)
 ───────────────────          ─────────────────────────────────         ────────────
  create J_a, J_b   ─────────► submit_job ×2  → WAITING_FOR_DATA
  load adapter v0   ◄───────── (initial artifact / base)

  sample under J_a@v0
  compute old_logprobs,
  rewards
  submit_trajectory_batch ───► validate_trajectory_batch
   (job=J_a, ver=0, ...)         (lag<=max, shapes, hash)
                                trajectory_store.put
                                mark_train_batch_ready(J_a, n_tok)
                                   → J_a TRAIN_READY, ready_tokens += n_tok
  submit_trajectory_batch ───► ... J_b TRAIN_READY                    (loop running:)
   (job=J_b, ver=0, ...)                                              snapshot_jobs
                                                                      accrue_deficits
                                                                      select_training_jobs
                                                                        → [(J_a,t_a),(J_b,t_b)]
                                                          ◄────────── ensure_hot(J_a)->slot0
                                                                      ensure_hot(J_b)->slot1
                                                                       (preempt victim if full)
                                                                      pop_for_job + materialize
                                                                      plan_packed_layout
                                                                      per-example microbatches:
                                                                        set_tokens_per_adapter_slot
                                                                        logits = model(...)
                                                                        lp = _token_logprobs(logits)
                                                                        continuous_rl_loss(lp,...)
                                                                        forward_backward + step
                                commit_step({J_a,J_b})  ◄──────────── (version 0 -> 1, spend deficit)
                                  current_adapter_version = 1
                                artifact_publisher.publish
                                mark_published(J_a@v1), (J_b@v1)
  poll adapters/latest ◄────── latest_adapter_uri = .../step_1
  load J_a@v1, J_b@v1
  sample under v1  ...  (loop continues; max_policy_lag rejects too-stale batches)
                                complete_job when budget exhausted → free slot
```

Step by step:

1. **Create.** `create_lora_training_job` → `submit_job` → both jobs
   `WAITING_FOR_DATA` at version 0. External inference loads the initial adapter.
2. **Submit (async, repeated).** Each external batch is validated
   (`validate_trajectory_batch`: version not in the future, lag within
   `max_policy_lag`, shapes/hashes), stored, and `mark_train_batch_ready` grows
   `ready_train_tokens` and flips the job to `TRAIN_READY`. Requests for both jobs
   pile up at arbitrary times.
3. **Schedule.** The runner loop credits both runnable jobs (`accrue_deficits`,
   weighted by priority) and `select_training_jobs` returns both, each with a
   token target bounded by deficit / `tokens_per_update` / ready data.
4. **Make hot.** `ensure_hot` reserves slots (`J_a→0`, `J_b→1`); if full, the
   scheduler picks an idle/over-served victim, the pager checkpoints + clears it
   to free a slot.
5. **Pull + materialize.** `pop_for_job` drains each job's queued batches;
   `materialize_rl_examples` turns them into `TrainExample`s stamped with the slot.
6. **Pack + step.** `plan_packed_layout` orders by slot; one microbatch per
   example; `_token_logprobs` produces current-policy logprobs; `continuous_rl_loss`
   computes the clipped PG (+ optional KL) per job, normalized over each job's
   action tokens, summed; `get_forward_backward_func` runs fwd/bwd; `optimizer.step()`
   updates only slots 0 and 1's params.
7. **Commit + publish.** `commit_step` bumps each `current_adapter_version` 0→1
   and spends deficit; `artifact_publisher.publish` writes `step_1/`; external
   inference polls, loads `@v1`, and continues on-policy. Batches that now lag
   past `max_policy_lag` are rejected.
8. **Fairness over time.** Deficits make trained-token allocation track priority;
   `max_consecutive_steps` prevents hogging; jobs waiting on slow rollout (no
   ready data) are not kept hot and are preemptible; each job ends when its
   `budget` is exhausted, freeing its slot for a queued job.

---

## 5. Job state machine

```
                       submit_job
                           │
                           ▼
                    WAITING_FOR_DATA  ◄────────────── commit_step (no data left)
                           │ mark_train_batch_ready                ▲
                           ▼                                       │
                       TRAIN_READY ──── reserve_free_slot ──► LOADING
                           ▲                                       │ mark_hot
            mark_cold      │                                       ▼
         (data left) ──────┤                                   HOT_IDLE ◄── commit_step (data left)
                           │                                       │ select + mark_active_step
                       COLD_READY ◄── mark_cold (drained)          ▼
                           ▲                                   ACTIVE_STEP
                           └───────────── preempt ─────────────────┘
                                                                   │ commit_step
                                                                   ▼
                                          COMPLETED / FAILED / CANCELLED  (terminal)
```

---

## 6. Status: validated vs. unverified

- **Unit-tested on CPU (`tests/training_engine/`, 26 tests):** packer layout
  invariant; scheduler fairness/selection/preemption; controller transitions;
  trajectory validation / policy-lag.
- **Wired, GPU-unverified:** the in-actor Megatron path
  (`run_continuous_sft_engine` → `_run_megatron_step`) for both SFT and RL. SFT
  runs logits→cross-entropy; RL runs `_token_logprobs` → `continuous_rl_loss`.
  These need a cluster run to shake out (optimizer-step return parity with Miles'
  `train_one_step`, arg-parse in `run_engine.sh`).
- **Deferred:** multi-microbatch / pipeline-parallel packing,
  preemption/offload at scale, object-store backends, persistence, the HTTP
  server, and the colocated-rollout plugin.

---

## 7. File map

```
miles/training_engine/
  schemas.py            contracts + validate_trajectory_batch
  job_controller.py     TrainingJobController (state machine)
  scheduler.py          ContinuousTrainingScheduler (deficit fair queuing)
  trajectory_store.py   TrajectoryStore (RL) + ExampleStore (SFT)
  dataset_worker.py     SFT JSONL -> TrainExample
  continuous_packer.py  plan_packed_layout (pure) + pack_examples_by_slot (torch)
  continuous_loss.py    continuous_sft_loss / continuous_rl_loss (per-job)
  checkpoint_store.py   internal preempt/resume state
  artifact_publisher.py versioned HF/PEFT adapter export
  adapter_pager.py      AdapterPager (job -> slot, preemption)
  runner.py             ContinuousTrainingRunner (the quantum loop) + _token_logprobs
  client.py             build_job_spec + TrainingClient
  events.py / metrics.py / args.py / api_server.py

miles/backends/megatron_utils/multi_lora.py   + capture/restore optimizer-state helpers
miles/backends/megatron_utils/actor.py        + run_continuous_sft_engine
miles/ray/actor_group.py                       + run_continuous_sft_engine broadcast

examples/training_engine/
  engine_driver.py      ray-job driver (allocates actor, runs engine)
  run_engine.sh         ray start + ray job submit
  modal_engine.py       Modal app (radixark image, provision + train on H200)
  empty_adapters/       empty --multi-lora-dir (N free slots, no preloaded adapters)

tests/training_engine/  packer / scheduler / controller / validation tests
```
