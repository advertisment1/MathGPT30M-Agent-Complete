# Architecture ablations: common interface, fair comparisons

This is a **proposed experiment matrix**, not measured results. The reference is the user's *Smol Training Playbook*, architecture chapter: attention (MHA/MQA/GQA), embedding sharing, positional methods, training stability and core components. No result is claimed to have been reproduced from that book.

All models share `GPT(Config).forward(ids, targets=None, caches=None) -> (logits, loss, caches)` and `generate(...)`, the same tokenizer, train/validation split, optimizer, learning-rate schedule, and seed.

| Variant | Change from baseline | Hypothesis | Important confound |
|---|---|---|---|
| gqa_rope_rms_swiglu | 2 KV heads | quality / KV memory compromise | baseline |
| mha_rope_rms_swiglu | 6 KV heads | stronger quality, more KV memory | higher parameter count |
| mqa_rope_rms_swiglu | 1 KV head | lower memory, possible quality loss | lower parameter count |
| gqa_nope | no positional rotation | tests position encoding necessity | extrapolation must be tested separately |
| gqa_layernorm | LayerNorm | stability comparison | small extra norm params |
| gqa_gelu | GELU FFN | compare gating vs conventional FFN | hidden=768 gives roughly matched FFN params vs SwiGLU hidden=512 |
| gqa_untied | untied input/output weights | possible loss improvement | +vocab*dim params, very large change |

## Procedure
1. Prepare FineMath token data once; freeze `train.bin`, `val.bin`, tokenizer, and dataset manifest.
2. Run `python scripts/run_ablations.py --dry-run` to inspect commands.
3. Run `python scripts/run_ablations.py` on GPU. Each variant runs independently; results are written to `runs/ablations/<name>/metrics.jsonl` and `summary.json`.
4. Compare final validation loss, token/s, total parameter count, memory usage, and training variance. Lowest loss alone is **not** enough to pick a winner when parameter counts differ.
5. Repeat the two most promising configurations using seeds 42, 43, 44; evaluate on identical held-out math tasks before choosing.
6. Refit the selected architecture to ~30M params (via dim/layers/hidden); recheck parameter count and smoke tests before long training.

## 48 GPU-hour constraints
The default 120-step grid is a *smoke-scale* architecture screen, not a scientifically conclusive ablation. Benchmark tokens/s first, cap GPU hours, and reserve the bulk for 30M training. Very short runs can rank variants differently from full training. The script currently uses sequential runs and will not enforce a wall-clock budget. Checkpoint/model settings should be recorded for each run.

## Training vs fine-tuning
Architecture screening is not SFT. No SFT, DPO or GRPO is included in this variant runner. Mathematical instruction SFT is a separate optional stage after pretraining.
