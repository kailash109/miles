"""Reward-agnostic generation over a running sglang router.

Reuses the exact ``/generate`` HTTP contract miles' rollout path uses
(``input_ids`` + ``sampling_params`` + ``lora_path``), so online-RL generation
in the training engine is a thin wrapper, not a new inference stack. The engine
only *generates*; reward/advantage computation stays on the client (it scores
the returned rollouts and resubmits them as an ``ExternalTrajectoryBatch``).

Mirrors the production rollout path (``miles/rollout/sglang_rollout.py``): one
``/generate`` request per (prompt, sample) sequence, fanned out concurrently with
``asyncio.gather`` over a shared connection-pooled client and bounded by a single
``asyncio.Semaphore`` (so total in-flight requests track sglang's capacity). The
inference engine's continuous batching merges those concurrent requests into
large decode batches across the loaded LoRA adapters -- there is no client-side
prompt stacking, exactly as the rollout path does it.

The public ``generate`` stays synchronous so the FastAPI ``/sample`` handler can
keep holding the (threading) weight-sync read lock across the call; internally it
drives a dedicated event loop running on a background thread, so the async fan-out
never blocks (or is blocked by) the API server's request threads.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

from miles.utils.http_utils import _post

_DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 512}
# Default cap on concurrent /generate requests when the caller doesn't size it
# from server capacity. sglang queues beyond its running budget, so this just
# controls how aggressively we fan out.
_DEFAULT_GEN_CONCURRENCY = int(os.environ.get("ENGINE_GEN_CONCURRENCY", "64"))
# Per-request retry budget. Bounded (vs. the rollout path's larger default)
# because /sample holds the weight-sync read lock for the whole call -- a request
# that retries forever would block weight syncs. The client resubmits on failure.
_DEFAULT_GEN_MAX_RETRIES = int(os.environ.get("ENGINE_GEN_MAX_RETRIES", "10"))


class SglangGenerator:
    """Generates rollouts for tokenized prompts against an sglang router."""

    def __init__(
        self,
        router_url: str,
        *,
        timeout: float = 600.0,
        max_concurrency: int | None = None,
        max_retries: int = _DEFAULT_GEN_MAX_RETRIES,
    ):
        self.router_url = router_url.rstrip("/")
        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency or _DEFAULT_GEN_CONCURRENCY))
        self.max_retries = max(1, int(max_retries))

        # Dedicated event loop on a background thread: the async fan-out + shared
        # httpx client live here, decoupled from FastAPI's request threads (and
        # from the threading ReadWriteLock the /sample handler holds).
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        # Created lazily on the loop (so they bind to the right event loop).
        self._client: Any = None
        self._sem: asyncio.Semaphore | None = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._loop_ready.set)
        self._loop.run_forever()

    async def _ensure_resources(self) -> None:
        # Runs on self._loop, so only the loop thread ever touches these.
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.max_concurrency),
                timeout=httpx.Timeout(self.timeout),
            )
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrency)

    async def _generate_one(
        self, prompt_ids: list[int], params: dict, lora_path: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_ids": list(prompt_ids),
            "sampling_params": dict(params),
            "return_logprob": True,
        }
        if lora_path:
            payload["lora_path"] = lora_path
        assert self._sem is not None
        async with self._sem:
            # Reuse the rollout path's retrying POST (shared pooled client).
            out = await _post(
                self._client, f"{self.router_url}/generate", payload, max_retries=self.max_retries
            )
        meta = out.get("meta_info", {})
        token_logprobs = meta.get("output_token_logprobs") or []
        # sglang returns (logprob, token_id, ...) tuples.
        response_ids = [item[1] for item in token_logprobs]
        response_logprobs = [item[0] for item in token_logprobs]
        return {
            "prompt_ids": list(prompt_ids),
            "response_ids": response_ids,
            "response_logprobs": response_logprobs,
            "text": out.get("text", ""),
        }

    async def _generate_all(
        self, prompts: list[list[int]], params: dict, lora_path: str | None
    ) -> list[dict[str, Any]]:
        await self._ensure_resources()
        # gather preserves input order, so the caller's prompt-major grouping
        # (e.g. GRPO groups per prompt) is retained.
        return await asyncio.gather(*(self._generate_one(p, params, lora_path) for p in prompts))

    def generate(
        self,
        *,
        input_ids_list: list[list[int]],
        sampling_params: dict[str, Any] | None = None,
        lora_path: str | None = None,
        n_samples_per_prompt: int = 1,
    ) -> list[dict[str, Any]]:
        """Return one rollout dict per (prompt, sample), in prompt-major order.

        Each dict carries ``prompt_ids``, ``response_ids``, ``response_logprobs``
        (the behavior-policy logprobs needed by the RL loss), and ``text``.
        """
        params = {**_DEFAULT_SAMPLING, **(sampling_params or {})}
        n = max(1, n_samples_per_prompt)
        # One task per (prompt, sample); the engine fans these out concurrently.
        prompts = [pid for pid in input_ids_list for _ in range(n)]
        if not prompts:
            return []

        self._loop_ready.wait()
        future = asyncio.run_coroutine_threadsafe(
            self._generate_all(prompts, params, lora_path), self._loop
        )
        return future.result()
