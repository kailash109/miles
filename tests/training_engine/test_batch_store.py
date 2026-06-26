"""CPU tests for the lease/commit/abort batch store (no torch/ray)."""

from __future__ import annotations

from miles.training_engine.batch_store import BatchState, BatchStore


def _store() -> BatchStore:
    return BatchStore(put_fn=lambda x: x)  # identity ref for tests


def test_put_makes_available_and_tracks_tokens():
    s = _store()
    bid = s.put("j", payload={"x": 1}, token_count=100, adapter_version=0)
    assert s.records[bid].state == BatchState.AVAILABLE
    assert s.global_ready_tokens == 100
    assert s.pending_tokens("j") == 100


def test_idempotent_client_batch_id():
    s = _store()
    a = s.put("j", payload=1, token_count=10, adapter_version=0, client_batch_id="c1")
    b = s.put("j", payload=2, token_count=10, adapter_version=0, client_batch_id="c1")
    assert a == b
    assert s.global_ready_tokens == 10  # second put deduped


def test_lease_commit_consumes():
    s = _store()
    s.put("j", payload=1, token_count=60, adapter_version=0)
    s.put("j", payload=2, token_count=60, adapter_version=0)
    leases = s.lease_for_plan("j", target_tokens=70, plan_id="p1")
    assert len(leases) == 2  # stops once target met (after first crosses 70? 60<70 -> takes 2nd)
    assert s.global_ready_tokens == 0
    s.commit_plan("p1")
    assert all(r.state == BatchState.CONSUMED for r in s.records.values())
    # Nothing left to lease.
    assert s.lease_for_plan("j", target_tokens=100, plan_id="p2") == []


def test_lease_abort_releases():
    s = _store()
    s.put("j", payload=1, token_count=60, adapter_version=0)
    leases = s.lease_for_plan("j", target_tokens=50, plan_id="p1")
    assert len(leases) == 1
    assert s.global_ready_tokens == 0
    s.abort_plan("p1")
    r = next(iter(s.records.values()))
    assert r.state == BatchState.AVAILABLE
    assert r.leased_by_plan_id is None
    assert s.global_ready_tokens == 60
    # Re-leasable after abort.
    assert len(s.lease_for_plan("j", target_tokens=50, plan_id="p2")) == 1


def test_lease_stops_at_target():
    s = _store()
    for _ in range(5):
        s.put("j", payload=1, token_count=100, adapter_version=0)
    leases = s.lease_for_plan("j", target_tokens=250, plan_id="p1")
    # Takes batches until tokens >= 250 -> 3 batches (100,100,100).
    assert sum(l.token_count for l in leases) >= 250
    assert len(leases) == 3
