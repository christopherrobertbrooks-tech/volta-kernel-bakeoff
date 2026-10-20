# Stage 3 research notes (2026-10-19)

Stage 3 was "our kernels in a real model". Before building anything: what a real model on the V100 spends its time on,
and what llama.cpp already does there. Model: Qwen3.6-35B-A3B (40 layers, 256 experts per layer, 8 used per token,
3 linear-attention layers to each full-attention one), llama.cpp mainline 5e03bdd on the V100.

## What llama.cpp already does at one token (read from `ggml/src/ggml-cuda/ggml-cuda.cu`)

- **All of a layer's experts in one launch** (`mul_mat_id` through MMVQ), on Volta for any batch size
  (`get_mmvq_mmid_max_batch`: "Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID").
- **Gate + up + SwiGLU fused** into one kernel, for the routed experts and the shared expert together
  (`ggml_cuda_should_fuse_mul_mat`, `ggml_cuda_match_shared_expert`). Volta gets it; only Pascal and older are left
  out ("fusion is not universally faster on Pascal").
- **Router softmax + top-k fused** (`topk_moe`), **RMS norm + multiply + RoPE + cache write fused**, and the linear
  attention's state snapshot copies folded into the gated-delta-net kernel (`ggml_cuda_try_gdn_cache_fusion`).
- CUDA graphs: a token's kernels (hundreds; the profile will count them) are launched as one graph.

So the "whole expert in one launch" idea from stage 1 (22 us vs 56 for cuBLAS batched) is already in llama.cpp. Our
remaining kernel-level edge is the per-table speed: 7-27% on most 4-bit tables, 16% on the 8-bit attention table
(`fused_q8` 27.1 us vs `llama_q8_0` 32.1), more on 2-bit.

## What one written token reads (weights, from the GGUF files)

| | GSQ 2-bit | Unsloth UD-Q2_K_XL | Unsloth UD-Q4_K_XL |
| --- | ---: | ---: | ---: |
| attention + linear attention + router | 1.46 GB (60%) | 1.02 GB (59%) | 1.46 GB (53%) |
| output head (248K x 2048) | 0.54 GB (22%) | 0.29 GB (17%) | 0.54 GB (20%) |
| routed experts (8 of 256) | 0.28 GB (12%) | 0.33 GB (19%) | 0.61 GB (22%) |
| shared expert | 0.13 GB | 0.09 GB | 0.13 GB |
| total | 2.42 GB | 1.73 GB | 2.75 GB |

At the measured ~94 tokens a second (both GSQ 2-bit and Unsloth 4-bit) that is 230-260 GB/s of the card's ~900.
If reading weights were all the time, a token would take ~3 ms (330 tokens a second); it takes ~10.6 ms.

**So, before profiling:** at most ~3 of a token's ~10.6 ms are spent reading weights. Even a matrix multiply that read
at the card's full speed would leave most of the time untouched. The other ~7 ms are somewhere else: the linear
attention's recurrent kernel (30 layers), full attention, the many small element-wise steps, routing, or time
between kernels. A profile of 64 written tokens (nsys, CUDA graphs traced per kernel) is queued to run after the
overnight benchmarks: `gsq-gguf/scripts/profile-decode.sh`.

## Options, depending on the profile

1. **Swap our kernels into a private llama.cpp build** for the tables they win (8-bit attention, the output head,
   4-bit / 2-bit experts at one token) and measure the whole model before and after. Same engine, only the kernel
   changes, so the difference is ours. Ceiling: a share of the ~3 ms of weight reading.
2. **Whatever the profile shows as the biggest non-multiply cost** (e.g. the linear-attention kernel or gaps between
   kernels) -- possibly a bigger lever than any multiply kernel, and not one the bake-off has measured.
3. **Not worth it:** a whole new engine in Python around our kernels (stage 3 as first sketched). It would pay
   Python's per-kernel launch cost on every one of those kernels, which CUDA graphs avoid; it could only show the
   kernels work, not that they make a model faster.

Speed the profile can't fix: Strata already writes Qwen3.6 faster than llama.cpp on this card (about 110 tokens a
second at a 256-token prompt in 0.1.45, against llama.cpp's ~93), so llama.cpp is not the fastest engine to beat here.
