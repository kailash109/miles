"""Coordinator-owned batch store with lease/commit/abort semantics.

Replaces destructive FIFO popping. Data is *leased* to a plan and only consumed
when that plan commits; an aborted plan releases its leases back to AVAILABLE so
nothing is lost on worker failure. Payloads are kept behind a ``payload_ref``
(``ray.put`` by default, injectable for CPU tests) so the coordinator only holds
metadata.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .plan import BatchLease
from .schemas import make_batch_id


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
    payload_ref: Any
    state: BatchState
    created_at: float
    client_batch_id: str | None = None
    leased_by_plan_id: str | None = None


@dataclass(frozen=True)
class JobQueueView:
    """Cheap, metadata-only snapshot of a job's AVAILABLE batch queue.

    Lets a scheduler reason about *leasable* tokens (records are indivisible:
    ``lease_for_plan`` leases whole records), so it can size grants to what can
    actually be handed out rather than to an idealized token target.
    """

    job_id: str
    available_tokens: int
    available_batches: int
    first_batch_tokens: int | None
    oldest_ready_at: float | None


class BatchStore:
    def __init__(self, put_fn: Callable[[Any], Any] | None = None):
        # put_fn defaults to ray.put (lazy import); tests inject identity.
        self._put_fn = put_fn
        self.records: dict[str, BatchRecord] = {}
        self.available_by_job: dict[str, list[str]] = defaultdict(list)
        self.client_id_index: dict[tuple[str, str], str] = {}
        self.global_ready_tokens: int = 0

    def client_batch_exists(self, job_id: str, client_batch_id: str | None) -> bool:
        return client_batch_id is not None and (job_id, client_batch_id) in self.client_id_index

    def _put(self, payload: Any) -> Any:
        if self._put_fn is not None:
            return self._put_fn(payload)
        import ray

        return ray.put(payload)

    def put(
        self,
        job_id: str,
        payload: Any,
        token_count: int,
        adapter_version: int,
        client_batch_id: str | None = None,
    ) -> str:
        if client_batch_id is not None:
            existing = self.client_id_index.get((job_id, client_batch_id))
            if existing is not None:
                return existing  # idempotent retry

        payload_ref = self._put(payload)
        batch_id = make_batch_id(job_id)
        self.records[batch_id] = BatchRecord(
            batch_id=batch_id,
            job_id=job_id,
            adapter_version=adapter_version,
            token_count=token_count,
            payload_ref=payload_ref,
            state=BatchState.AVAILABLE,
            created_at=time.time(),
            client_batch_id=client_batch_id,
        )
        self.available_by_job[job_id].append(batch_id)
        self.global_ready_tokens += token_count
        if client_batch_id is not None:
            self.client_id_index[(job_id, client_batch_id)] = batch_id
        return batch_id

    def lease_for_plan(self, job_id: str, target_tokens: int, plan_id: str) -> list[BatchLease]:
        leases: list[BatchLease] = []
        tokens = 0
        for batch_id in list(self.available_by_job.get(job_id, [])):
            if tokens >= target_tokens:
                break
            record = self.records[batch_id]
            if record.state != BatchState.AVAILABLE:
                continue
            record.state = BatchState.LEASED
            record.leased_by_plan_id = plan_id
            self.available_by_job[job_id].remove(batch_id)
            self.global_ready_tokens -= record.token_count
            leases.append(
                BatchLease(
                    batch_id=record.batch_id,
                    job_id=record.job_id,
                    token_count=record.token_count,
                    payload_ref=record.payload_ref,
                    adapter_version=record.adapter_version,
                )
            )
            tokens += record.token_count
        return leases

    def commit_plan(self, plan_id: str) -> None:
        for record in self.records.values():
            if record.leased_by_plan_id == plan_id and record.state == BatchState.LEASED:
                record.state = BatchState.CONSUMED

    def abort_plan(self, plan_id: str) -> None:
        for record in self.records.values():
            if record.leased_by_plan_id == plan_id and record.state == BatchState.LEASED:
                record.state = BatchState.AVAILABLE
                record.leased_by_plan_id = None
                self.available_by_job[record.job_id].append(record.batch_id)
                self.global_ready_tokens += record.token_count

    def queue_view(self, job_id: str) -> JobQueueView:
        ids = [
            b
            for b in self.available_by_job.get(job_id, [])
            if self.records[b].state == BatchState.AVAILABLE
        ]
        if not ids:
            return JobQueueView(job_id, 0, 0, None, None)
        return JobQueueView(
            job_id=job_id,
            available_tokens=sum(self.records[b].token_count for b in ids),
            available_batches=len(ids),
            # FIFO order: available_by_job preserves insertion order and
            # lease_for_plan consumes from the front, so ids[0] is the next record.
            first_batch_tokens=self.records[ids[0]].token_count,
            oldest_ready_at=min(self.records[b].created_at for b in ids),
        )

    def pending_tokens(self, job_id: str) -> int:
        return sum(
            self.records[b].token_count
            for b in self.available_by_job.get(job_id, [])
            if self.records[b].state == BatchState.AVAILABLE
        )

    def drop_job(self, job_id: str) -> None:
        for batch_id in self.available_by_job.pop(job_id, []):
            rec = self.records.pop(batch_id, None)
            if rec is not None and rec.state == BatchState.AVAILABLE:
                self.global_ready_tokens -= rec.token_count
