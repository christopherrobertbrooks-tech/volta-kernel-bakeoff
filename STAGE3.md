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

## The profiles (2026-10-19 night)

**Writing one token** (Nsight Systems, 64 tokens, CUDA graphs traced per kernel; GSQ 2-bit, Unsloth 4-bit the same
within 2%): llama.cpp runs **~1,526 kernels per token** and the GPU is busy the whole time -- 10.5 ms of kernels for
a 10.4 ms token measured without the profiler (the traced run's "idle" is the profiler's own overhead).

| per written token (GSQ 2-bit) | ms | share | kernels | avg |
| --- | ---: | ---: | ---: | ---: |
| matvec Q8_0 (attention, linear attention, output head) | 4.06 | 38% | 211 | 19 us |
| matvec Q2_0 (routed experts, gate+up fused / down) | 1.57 | 15% | 80 | 20 us |
| norms | 0.83 | 8% | 191 | 4.3 us |
| element-wise (add, mul, glu, scale, ...) | 0.80 | 8% | 282 | 2.8 us |
| copies / set_rows / get_rows | 0.78 | 7% | 151 | 5.2 us |
| converting activations to 8 bits (q8_1) | 0.77 | 7% | 291 | 2.6 us |
| matvec f32 (router, small tables) | 0.77 | 7% | 140 | 5.5 us |
| router top-k, MoE reduction | 0.52 | 5% | 80 | |
| attention + linear attention + conv + rope | 0.64 | 6% | 100 | |

So **about half of a token is ~1,200 small kernels of 3-5 us**, each far too short to use the card. At 32K tokens of
context only flash attention grows (0.30 -> 1.18 ms); writing drops from 96 to 87 tokens a second.

**Reading a 4,096-token prompt:** the 8-bit attention tables are dequantized to fp16 for every chunk and multiplied
by cuBLAS (39% together); **the linear attention's gated-delta-net kernel takes 7 ms per layer per 4,096 tokens (15%,
one kernel per layer -- it walks the tokens one by one)**; the experts' MMQ 10%; large copies 9%.

**Our kernels up close** (Nsight Compute, Qwen3.6 attention table 8192x2048, one token; clocks held at base by the
profiler, so times are longer than in the race): our 8-bit kernel reaches 76% of memory speed (llama.cpp's Q8_0 68%);
our 4-bit 60% (llama.cpp's Q4_0 58%); our 2-bit 37% with arithmetic 60% busy -- unpacking-bound -- against 41% for
llama.cpp's Q2_0, which converts activations to int8 once and uses `dp4a`. On a single 512x2048 expert every kernel is
at 5-19% of memory speed with ~90% of the card idle.

**Drafting** (llama.cpp `--spec-type draft-mtp`, 3 prompts x 2, 256 tokens, temperature 0): GSQ 2-bit 93 -> 128
tokens a second with 1 draft token (79% accepted; the MTP layer is borrowed from Unsloth's file and works as well as
in Unsloth's own, 78%), Unsloth 4-bit 92 -> 122 (126 with 3 draft tokens). Greedy text matches the no-draft text for
the first 150-470 characters, then splits at a near-tie (verifying two tokens at once rounds slightly differently).
**The biggest writing-speed lever on this card is not a kernel: it is drafting, +33-37%.**

## Recommendation for stage 3

1. **Fuse the small steps** (best lever, ~half of a token): e.g. norm + convert-to-8-bit, the element-wise chains
   around the linear attention, router + reduction. Each fused pair removes a 3-5 us kernel; ~1,200 of them per token.
   This is engine work inside a private llama.cpp build, measured end to end.
2. **Prompt reading: the gated-delta-net kernel.** 15% of a long prompt in one serial kernel per layer; a chunked
   (parallel) form of the delta rule exists in the literature (Gated DeltaNet's chunkwise algorithm). llama.cpp's
   kernel (`ggml-cuda/gated_delta_net.cu`) loops over the tokens one at a time, and its launcher says
   `//TODO: Add chunked kernel for even faster pre-fill` -- a known, open gap. Directly useful for Mamu-style
   27K-token prompts. (Our work stays in our own repos; nothing is posted to llama.cpp.)
3. **A Marlin-style prompt kernel for Volta** (added 2026-10-19 after Muse's note; hardest of the list). Prompt
   reading spends 39% of its time dequantizing the 8-bit tables to fp16 and multiplying with cuBLAS. A kernel that
   unpacks inside the tensor-core multiply avoids writing and re-reading the fp16 copy. Marlin (IST-DASLab) does this
   on Ampere+ only: `cp.async` (Ampere), `ldmatrix` (Turing+), m16n8k16 MMA (Ampere). On sm_70 the async copy becomes
   register-staged double buffering, as Volta GEMMs did, and the MMA Volta's m8n8k4 / WMMA 16x16x16. Starting point:
   our `tc_q4` (WMMA; takes 1.6-2.3x cuBLAS's time at 512 tokens, and 1.1-6.5x at 64 (worst on expert-sized tables, where it launches too few blocks to fill the card and its time stays flat from 64 to 512 tokens -- issue #4)). Muse's claim that sm_70
   is "unclaimed by construction" is too strong (llama.cpp's MMQ is tuned for Volta, ExLlama-style kernels run there),
   but a fused dequant + tensor-core prompt kernel for it is not something we've seen.
   **Why `tc_q4` is slow** (Grok's review, 2026-10-20, confirmed in `src/stage2.cuh`): only 64 of its 128 threads
   unpack, one row each, so their loads are not coalesced; it converts one value at a time (`__int2half_rn` +
   `__hmul`) instead of the `0x6400` trick our 4-bit kernel uses; no register double-buffering; 64x64x32 tiles use
   ~10 KB of the 96 KB of shared memory; and on expert-sized tables it launches only 8-80 blocks (issue #4), so the
   rebuild needs split-K or per-shape tiles there, not just 128x128 tiles. Rebuild: all threads, coalesced,
   half2 + `0x6400`, register double-buffering, tile size per shape, split-K for small tables.
   **Go / no-go (issues #1, #5):** with the clock locked and `--check` passing, the rebuilt kernel's speed relative to cuBLAS fp16, weighted by each table's share of real prompt time at 512-4,096 tokens (the big attention and output tables dominate; llama.cpp sends the experts through MMQ, not cuBLAS). Break-even is ~75%: in the 4,096-token GSQ profile cuBLAS took 850 ms and dequantizing to fp16 291 ms, and the fused kernel skips the dequantize, so it matches llama.cpp at 850 / 1,141 = 74.5% of cuBLAS. Under ~70%: drop the Marlin plan; ~85% or more: worth building into llama.cpp. The final verdict should come from an 8-bit version (`tc_q8`): the tables that dominate prompt time are 8-bit in both GSQ's and Unsloth's files. Today's unrebuilt `tc_q4` is at 43-63% per table (#5).
   **Result (2026-10-20, `tc_q4_v2` in `src/stage4_tc.cuh`; 3 clock-locked rounds, `--check` passing, sabotage
   caught):** below the bar. At 512 tokens on the big tables (attention q / out / qkv, MLP up / down) it runs at
   51-62% of cuBLAS's speed, 57% weighted by time (`results/2026-10-20-tc2-default/vs-cublas.txt`), against the
   ~70% drop line. The rebuild did fix what it was aimed at: at 64 tokens it beats cuBLAS on every big table
   (119-147%), and at 128 it is 84-143%. So the kernel is worth having for short prompts and batches up to ~128
   tokens, not for prompt reading at 512+. Eight tile layouts were timed with every split
   (`results/2026-10-20-tc2-tune2`, `TC2_TUNE=1`): bigger 64x64 warp tiles hit 255 registers and spill (12-33% slower at
   512), and the second shared-memory buffer lost every shape at 64-512 tokens, so the default is now one buffer
   (64x64 or 128x128 blocks, 32x32 / 64x32 warps). Per the bar set beforehand, the Marlin-style llama.cpp plan is
   dropped; a hand-written `mma.sync` m8n8k4 version (instead of WMMA) is the one lever left untried, and `tc_q8` was
   not built since `tc_q4` already misses.
   **Why it misses (Nsight Compute, 512 tokens, `results/2026-10-20-ncu-tc_q4_v2`, `bench/ncu-tc2.sh`):** same
   occupancy as cuBLAS (12.5-19%), but its tensor cores are busy 40% of the time against cuBLAS's 73-75%. Shared
   memory is the largest single cause: WMMA's fragment loads from our padded rows (72 halves) still conflict --
   6.3-8.4M conflict replays in 19-21M read wavefronts (cuBLAS: 0.002-0.19M of 6.4-8.7M, swizzled layouts) -- and
   waits on shared memory (short scoreboard + MIO throttle) are 2.5-5x cuBLAS's. Close behind: barrier waits (2x) and
   global-load waits (long scoreboard + LG throttle: each thread loads 4-byte words and their scales one by one). The
   unpack itself is not a limit (ALU 14-26% busy). So no single fix closes the gap: the `mma.sync` version would have
   to swizzle shared memory, widen the weight / scale loads and cut barriers together.
   Tracked in issue #1; the 8-token gap (83 vs 53 us) in #2; fusing the small kernels in #3.
4. **Our 8-bit kernel in place of llama.cpp's Q8_0 at one token**: 76% vs 68% of memory speed on the table that is
   38% of a token -> roughly 4% end to end. Small but cheap to try.
5. Not worth it: our 2-bit kernel (llama.cpp's Q2_0 beats it on big tables; experts are 15% of a token anyway), and a
   Python engine (see below).

## Options considered before the profile

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
