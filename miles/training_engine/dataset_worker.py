"""Turn SFT datasets (prompt/completion JSONL) into internal TrainExamples.

The worker is tokenizer-pluggable: callers pass an ``encode`` callable so this
module never imports a tokenizer or torch. Output is the same ``TrainExample``
shape used by external RL ingestion, so the packer is source-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .schemas import DatasetSpec, TrainExample, validate_sequence_lengths

# encode(text) -> list[int] token ids
Encoder = Callable[[str], list[int]]


def iter_jsonl(uri_or_path: str) -> Iterator[dict]:
    """Yield JSON objects from a local JSONL file path.

    Only local paths are supported in the MVP; object-store URIs (s3://...)
    should be fetched to a local cache by the caller first.
    """
    path = Path(uri_or_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {uri_or_path}")
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{uri_or_path}:{line_no + 1}: invalid JSON") from exc


def build_sft_example(
    *,
    job_id: str,
    record: dict,
    dataset: DatasetSpec,
    encode: Encoder,
    eos_id: int | None = None,
    max_length: int | None = None,
) -> TrainExample:
    """Build one SFT TrainExample from a prompt/completion record.

    Loss is computed only over completion tokens (prompt tokens are masked).
    ``labels`` are the next-token targets aligned with ``input_ids`` (shifted),
    with prompt positions set to ``-100`` so a standard cross-entropy ignores
    them; ``loss_mask`` mirrors that selection as floats for per-job
    normalization in the loss.
    """
    prompt = str(record[dataset.prompt_key])
    completion_key = dataset.completion_key or "completion"
    completion = str(record[completion_key])

    prompt_ids = encode(prompt)
    completion_ids = encode(completion)
    if eos_id is not None:
        completion_ids = [*completion_ids, eos_id]

    input_ids = [*prompt_ids, *completion_ids]
    if max_length is not None and len(input_ids) > max_length:
        # Right-truncate; keep the whole prompt if possible.
        input_ids = input_ids[:max_length]

    n = len(input_ids)
    n_prompt = min(len(prompt_ids), n)

    attention_mask = [1] * n
    # loss_mask selects completion tokens.
    loss_mask = [0] * n_prompt + [1] * (n - n_prompt)

    # Causal next-token labels: labels[i] is the target for position i. Mask
    # prompt positions and the final position (no next token) with -100.
    labels = input_ids[1:] + [-100]
    labels = [
        (tok if loss_mask[i] == 1 and i < n - 1 else -100)
        for i, tok in enumerate(labels)
    ]
    # Align loss_mask with masked labels.
    loss_mask = [1 if labels[i] != -100 else 0 for i in range(n)]

    example = TrainExample(
        job_id=job_id,
        slot=None,
        loss_type="sft",
        adapter_version=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
    )
    validate_sequence_lengths(example)
    return example


def build_sft_examples(
    *,
    job_id: str,
    records: Iterable[dict],
    dataset: DatasetSpec,
    encode: Encoder,
    eos_id: int | None = None,
    max_length: int | None = None,
) -> list[TrainExample]:
    return [
        build_sft_example(
            job_id=job_id,
            record=record,
            dataset=dataset,
            encode=encode,
            eos_id=eos_id,
            max_length=max_length,
        )
        for record in records
    ]
