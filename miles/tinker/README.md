# Full-parameter Tinker training (experimental)

Run `serve_tinker.py --tinker-full-training` with the usual model/parallelism args,
`--optimizer adam`, `--megatron-to-hf-mode bridge`, and `--tinker-checkpoint-root`.
Trainer and sampler must start from the same HF checkpoint and share checkpoint storage.

Requires BF16, CP=1, separate resident training/inference GPUs, synchronous parameter
gathers, and dynamic batching or `--micro-batch-size 1`. One active model per trainer.

Create with `parameterization: {"type": "full"}` and no `lora_config`. Tinker 0.26.2
requires HTTP model creation; subsequent calls use the normal SDK training client.

- `forward_backward` accumulates gradients; `optim_step` applies the supplied Adam settings.
- Save after `optim_step`. Checkpoints include model/optimizer state, but no pending gradients or RNG state.
- `save_weights_for_sampler` exports full HF weights. The gateway serializes sampling requests.

Validate against a running gateway; the script also demonstrates model creation:

```bash
python tests/manual/tinker/validate_full_training.py \
  --base-url http://localhost:10613 --base-model Qwen/Qwen3-0.6B
```
