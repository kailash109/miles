"""Reward-agnostic generation over a running sglang router.

Reuses the exact ``/generate`` HTTP contract miles' rollout path uses
(``input_ids`` + ``sampling_params`` + ``lora_path``), so online-RL generation
in the training engine is a thin wrapper, not a new inference stack. The engine
only *generates*; reward/advantage computation stays on the client (it scores
the returned rollouts and resubmits them as an ``ExternalTrajectoryBatch``).
"""

from __future__ import annotations

from typing import Any

_DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 512}


class SglangGenerator:
    """Generates rollouts for tokenized prompts against an sglang router."""

    def __init__(self, router_url: str, *, timeout: float = 600.0):
        self.router_url = router_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        input_ids_list: list[list[int]],
        sampling_params: dict[str, Any] | None = None,
        lora_path: str | None = None,
        n_samples_per_prompt: int = 1,
    ) -> list[dict[str, Any]]:
        """Return one rollout dict per (prompt, sample).

        Each dict carries ``prompt_ids``, ``response_ids``, ``response_logprobs``
        (the behavior-policy logprobs needed by the RL loss), and ``text``.
        """
        import requests

        params = {**_DEFAULT_SAMPLING, **(sampling_params or {})}
        rollouts: list[dict[str, Any]] = []
        for prompt_ids in input_ids_list:
            for _ in range(max(1, n_samples_per_prompt)):
                payload: dict[str, Any] = {
                    "input_ids": list(prompt_ids),
                    "sampling_params": dict(params),
                    "return_logprob": True,
                }
                if lora_path:
                    payload["lora_path"] = lora_path
                resp = requests.post(f"{self.router_url}/generate", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                out = resp.json()
                meta = out.get("meta_info", {})
                token_logprobs = meta.get("output_token_logprobs") or []
                # sglang returns (logprob, token_id, ...) tuples.
                response_ids = [item[1] for item in token_logprobs]
                response_logprobs = [item[0] for item in token_logprobs]
                rollouts.append(
                    {
                        "prompt_ids": list(prompt_ids),
                        "response_ids": response_ids,
                        "response_logprobs": response_logprobs,
                        "text": out.get("text", ""),
                    }
                )
        return rollouts
