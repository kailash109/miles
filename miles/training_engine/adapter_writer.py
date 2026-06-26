"""Background coalescing writer for inference HF-PEFT adapters.

Eviction snapshots a slot's LoRA weights synchronously (cheap, ~MBs); the slow
serialize + disk/volume write is handed here and runs on one background thread.
Per-job coalescing: if a newer snapshot for the same job arrives before the
previous write flushes, it replaces the pending one, so eviction bursts collapse
to one write per job. ``flush(job_id)`` blocks until that job has no pending
write (used by training onload and the save_weights cold-path so reads never see
a torn/stale dir).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class _Pending:
    out_dir: str
    tensors: dict[str, Any]
    rank: int
    alpha: int
    target_modules: tuple[str, ...]


class AdapterWriter:
    def __init__(self, args):
        self.args = args
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="adapter-writer", daemon=True)
        self._thread.start()

    def submit(
        self,
        job_id: str,
        out_dir: str,
        tensors: dict[str, Any],
        *,
        rank: int,
        alpha: int,
        target_modules: tuple[str, ...],
    ) -> None:
        with self._cv:
            self._pending[job_id] = _Pending(out_dir, tensors, rank, alpha, tuple(target_modules))
            self._cv.notify_all()

    def flush(self, job_id: str, timeout: float = 120.0) -> bool:
        """Block until ``job_id`` has no pending write. Returns False on timeout."""
        with self._cv:
            return self._cv.wait_for(lambda: job_id not in self._pending, timeout=timeout)

    def _run(self) -> None:
        from .adapter_export import write_hf_peft_adapter

        while True:
            with self._cv:
                while not self._stop and not self._pending:
                    self._cv.wait()
                if self._stop and not self._pending:
                    return
                job_id, item = next(iter(self._pending.items()))
            # Write outside the lock; the entry stays in _pending so flush() waits.
            try:
                write_hf_peft_adapter(
                    item.out_dir,
                    item.tensors,
                    args=self.args,
                    rank=item.rank,
                    alpha=item.alpha,
                    target_modules=item.target_modules,
                )
            except Exception as exc:  # noqa: BLE001 - don't kill the writer thread
                print(f"[adapter-writer] write failed for {job_id}: {exc!r}", flush=True)
            finally:
                with self._cv:
                    # Only clear if this exact snapshot is still the pending one
                    # (a newer submit may have replaced it -> leave it for the next pass).
                    if self._pending.get(job_id) is item:
                        del self._pending[job_id]
                    self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
