"""In-memory trajectory/data store for the MVP.

Production should keep batch payloads in an object store and retain only
metadata in memory; the interface here is intentionally small so that swap is
local. No torch/ray imports.
"""

from __future__ import annotations

from .schemas import (
    ExternalTrajectoryBatch,
    TrainExample,
    count_action_tokens,
    make_batch_id,
)


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
        """Pop FIFO batches for a job until ``target_tokens`` is reached.

        Returns whole batches (never splits a batch), so the popped token total
        may exceed ``target_tokens`` by up to one batch.
        """
        out: list[ExternalTrajectoryBatch] = []
        tokens = 0
        ids = self._by_job.get(job_id, [])
        kept: list[str] = []
        for batch_id in ids:
            if tokens >= target_tokens:
                kept.append(batch_id)
                continue
            batch = self._batches.pop(batch_id, None)
            if batch is None:
                continue
            out.append(batch)
            tokens += count_action_tokens(batch.action_mask)
        self._by_job[job_id] = kept
        return out

    def pending_tokens(self, job_id: str) -> int:
        return sum(
            count_action_tokens(self._batches[b].action_mask)
            for b in self._by_job.get(job_id, [])
            if b in self._batches
        )

    def drop_job(self, job_id: str) -> None:
        for batch_id in self._by_job.pop(job_id, []):
            self._batches.pop(batch_id, None)


class ExampleStore:
    """In-memory per-job queue of already-materialized ``TrainExample`` objects.

    Used for SFT jobs whose dataset worker pre-tokenizes examples. Token
    accounting uses loss-mask (trainable) tokens to mirror the trajectory store.
    """

    def __init__(self):
        self._by_job: dict[str, list[TrainExample]] = {}

    def put_many(self, job_id: str, examples: list[TrainExample]) -> int:
        queue = self._by_job.setdefault(job_id, [])
        queue.extend(examples)
        return sum(int(sum(ex.loss_mask)) for ex in examples)

    def pop_for_job(self, job_id: str, target_tokens: int) -> list[TrainExample]:
        out: list[TrainExample] = []
        tokens = 0
        queue = self._by_job.get(job_id, [])
        idx = 0
        while idx < len(queue) and tokens < target_tokens:
            ex = queue[idx]
            out.append(ex)
            tokens += int(sum(ex.loss_mask))
            idx += 1
        self._by_job[job_id] = queue[idx:]
        return out

    def pending_tokens(self, job_id: str) -> int:
        return sum(int(sum(ex.loss_mask)) for ex in self._by_job.get(job_id, []))

    def drop_job(self, job_id: str) -> None:
        self._by_job.pop(job_id, None)
