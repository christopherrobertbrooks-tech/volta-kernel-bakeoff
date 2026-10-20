# volta-kernel-bakeoff

Our own quantized matrix-multiply kernels for the **Tesla V100** (sm_70, 2017), raced against **cuBLAS** (NVIDIA's
fp16 tensor-core GEMM) and **llama.cpp's** own CUDA kernels, at the real table shapes of the models we run.

*Work in progress (stage 3 planning: see below). Numbers below are medians over 30 rounds of every contender -- ours,
cuBLAS and llama.cpp -- in one session on 2026-10-20 with the SM clock locked; every round is in
[results/2026-10-20](results/2026-10-20), medians in [medians.csv](results/2026-10-20/medians.csv)
([results/summarize.py](results/summarize.py)).*

## Why the V100 is interesting here

The V100 has fp16 tensor cores but no int8 / int4 ones (Turing added those), so a quantized model's weights must be
unpacked on the ordinary CUDA cores. When a model writes one token at a time, the work is reading the weights from
memory, so the fastest kernel is the one that reads the fewest bytes and unpacks them cheaply where they are used.

## What's in the race

| contender | what it does |
| --- | --- |
| `cublas_fp16` | cuBLAS GemmEx on fp16 weights (tensor cores; 4x the bytes of 4-bit) |
| `dq4+cublas` | unpack 4-bit to fp16, then cuBLAS |
| `fused_q4`, `fused_q4_v2`, `fused_q4_v2w` | ours: one warp per output row, unpacks in registers, fp32 sums; v2 = fp16 bit trick (`0x6400 \| n` is the half 1024 + n), v2w = a whole 32-value group per lane step |
| `fused_q8` | ours, int8 weights |
| `moe_*` | the MoE case: 8-10 experts, one token each, in one launch vs one cuBLAS call each vs cuBLAS batched |
| `q2w_alu_v3` | ours, plain 2-bit (codes -2..1, one fp16 scale per 32 = 2.5 bits per weight) |
| `q2_lut`, `q2w_lut_*` | the "times table": precompute the 16 possible sums per pair of activations, look them up instead of multiplying |
| `tc_q4` | ours, tensor cores for prompts: unpack a tile to shared memory, WMMA 16x16x16 |
| `llama_*` | llama.cpp's kernels via `ggml_mul_mat` (Q4_0, Q8_0, Q2_K, IQ2_XS, IQ2_XXS, F16), [bench/llamacpp_mm.cpp](bench/llamacpp_mm.cpp) |

Shapes: Qwen3.8-Flash-Next Coder experts 640x2560 (10 per token), its attention 12288x2560 and 2560x6144; Qwen3.6-35B-A3B
experts 512x2048 (8 per token) and attention 8192x2048; Qwen3-1.7B MLP 6144x2048 / 2048x6144. Tokens per call: 1 to 512.

## Making it fair

- **Real memory reads:** the V100's L2 is 6 MB, larger than one expert, so every contender cycles through enough copies of
  its weights (>= 96 MB) that each call reads them from HBM, as in a real model.
- **Correct before fast:** `./bakeoff --check` compares every kernel with a CPU double-precision reference of the same
  quantized weights, at 1, 2, 4, 8, 64 and 512 tokens and for the ten-experts-in-one-launch kernel (each expert with
  its own weights and token, so a wrong expert index fails too): 440 checks
  ([check.csv](results/2026-10-20/check.csv); both check files come from a checks-only pass of the runner right after
  the timing session, with the build that added the 512-token checks -- the in-session check in `run.log` was 404 ok
  and 310 sabotage failures without them). `BAKEOFF_SABOTAGE=1` breaks the packed weights on purpose and the check
  must fail: every kernel with packed 2-, 4- or 8-bit weights fails every one of its checks (328 failures,
  [check-sabotage.csv](results/2026-10-20/check-sabotage.csv)); the two cuBLAS fp16 contenders have no packing to break
  and pass, as a control.
- **Same clock for everyone:** the runner locks the V100's SM clock at 1,380 MHz (`nvidia-smi -lgc`), stops if the lock
  doesn't take, and resets it at the end; all 30 rounds read 1,380 MHz at 35-48 C ([gpu.log](results/2026-10-20/gpu.log)).
- Each round takes the median of 5 x 50 calls after warm-up; the table reports the median round. Fastest and slowest
  round differ by 0.9% typically.
- `dq4+cublas` (unpack to fp16, then cuBLAS) cycles its fp16 copies too, but each multiply still reads the copy it has
  just written, which on an expert-sized table is still in L2 -- as it would be in a real model.

## Results (one token, microseconds, lower is better)

| table | cuBLAS fp16 | llama.cpp Q4_0 | **ours 4-bit** | llama.cpp Q2_K | llama.cpp Q2_0 | **ours 2-bit** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next Coder attention 12288x2560 | 101.1 | 32.4 | 30.2 | 31.0 | **18.3** | 23.5 |
| Qwen3.6 attention 8192x2048 | 48.8 | 20.4 | 17.8 | 19.3 | **12.1** | 14.3 |
| Qwen3-1.7B MLP up 6144x2048 | 43.7 | 16.4 | 13.9 | 16.0 | **10.2** | 12.1 |
| Flash-Next Coder expert 640x2560 | 10.7 | 6.8 | 6.3 | 8.9 | 7.5 | **6.2** |
| Qwen3.6 expert 512x2048 | 9.7 | 7.2 | **5.2** | 8.2 | 8.2 | 5.5 |
| Qwen3.6 expert down 2048x512 | 6.9 | 6.6 | 4.7 | 7.8 | 7.3 | **4.0** |

Ours 4-bit = `fused_q4_v2w`, ours 2-bit = `q2w_alu_v3` (one fp16 scale per 32, 2.5 bits per weight). llama.cpp's Q2_0
(one fp16 scale per 64, 2.25 bits) is the same kind of plain 2-bit format. llama.cpp's times include converting the
activations to 8 bits for each multiply, as in a real model. Earlier runs ([2026-10-19](results/2026-10-19), clock
unlocked; [2026-10-20-llama30](results/2026-10-20-llama30)) agree with these medians to 0.09% typically; the
exceptions are `dq4+cublas` on small tables (up to 40% slower now that its copies are cycled) and llama.cpp's
IQ2_XS (~10% faster with the clock locked).

- **4-bit, one token:** ours takes 7-30% less time than llama.cpp's Q4_0 on 7 of the 9 tables, and is 2.5-4% slower
  on the two with 6,144-long rows (Flash-Next attention out, Qwen3-1.7B MLP down).
- **2-bit, one token:** ours takes 12-49% less time than llama.cpp's Q2_K, but Q2_K is a more complex format (a scale
  and offset per 16 weights, 256 per block). Against llama.cpp's own plain 2-bit, **Q2_0, ours loses on the big tables
  (18-29% slower) and wins on the expert-sized ones (17-46% less time)**. Nsight Compute on the Qwen3.6 attention
  table: ours keeps the GPU's arithmetic 60% busy and its memory only 37% -- it is limited by unpacking -- while Q2_0
  converts the activations to 8-bit integers once and multiplies four weights per instruction (`dp4a`).
- **2-bit quality, measured** (Qwen3.6-35B-A3B, WikiText-2, llama.cpp; from the gsq-gguf converter, not yet published): a plain 2-bit format with one
  scale per group, filled with ISTA-DASLab's GSQ weights (about the best such a format can hold), scores perplexity
  7.93; Unsloth's UD-Q2_K_XL (llama.cpp's K and IQ formats, ~2.6 bits on the experts, same file size) 7.26; its 4-bit
  UD-Q4_K_XL 6.73 (lower is better). The plain format's speed comes with a measurable quality cost.
- Ten experts in one launch: 22 us against 56 for cuBLAS batched and 109 for ten separate cuBLAS calls (Qwen3.6's
  eight: 12 us against 31 and 79).
- **Where we lose:** 8 tokens at once (llama.cpp Q4_0 53 us vs our best, `fused_q4_v2`, 83 on the 12288x2560 table;
  `v2w` is the fastest at one token and the slowest at eight), and prompts: our first
  tensor-core kernel takes 1.6-2.3x cuBLAS's time at 512 tokens, and 1.1-6.5x at 64 (worst on expert-sized tables, where it launches too few blocks to fill the card and its time stays flat from 64 to 512 tokens -- issue #4).
- **The times table loses on this card.** Against a badly built multiply kernel it looked 13-17% faster; against a good
  one it is slower everywhere -- the V100 multiplies cheaply, building and reading the table costs more.
- **Found on the way:** a 2-bit kernel that read its activations from shared memory at a 16-float stride per lane was
  slower than 4-bit (bank conflicts); reading them from global memory made it 1.9x faster. Even so, our best 2-bit run
  moves about 420 GB/s of the card's ~900. A single expert is too small to keep the card busy: Nsight Compute shows
  every kernel, ours and llama.cpp's, at 5-19% of memory speed on a 512x2048 expert, with ~90% of the card idle.

## Why llama.cpp's 2-bit can't do every table

llama.cpp's 2-bit formats (Q2_K, IQ2_*) pack 256 weights per block, so a table whose rows are not a multiple of 256
long can't use them -- our harness skips those (`K is not a multiple of 256`). Flash-Next's expert "down" tables have
rows 640 long (2.5 blocks). The published Strata Coder file
([ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF))
shows the consequence: its expert gate/up tables (rows 2,560 long, 10 blocks) are stored at 2-3 bits (IQ2_S, IQ3_XXS,
IQ3_S), while its expert down tables are IQ4_NL (4.5 bits, blocks of 32) in 39 of 48 layers and Q2_0 (2.25 bits, blocks of
64) in the other 9. Formats with 32-, 64- or 128-weight groups -- ours, and ISTA's GSQ (groups of 128; 640 = 5 x 128) --
fit those rows.

## Stage 3: what a whole model spends its time on

Before putting our kernels into a real model, we profiled one: Qwen3.6-35B-A3B in llama.cpp on the V100 (Nsight
Systems for whole tokens, Nsight Compute for single kernels). Full notes and tables: [STAGE3.md](STAGE3.md).

- **llama.cpp already does what stage 1 set out to show.** On Volta it runs all of a layer's experts in one launch and
  fuses gate + up + SwiGLU (routed and shared experts together), the router's softmax + top-k, and norm + RoPE.
- **A written token is ~1,526 kernels, and the GPU is busy the whole time** (10.5 ms of kernels in a 10.4 ms token,
  measured without the profiler). The multiplies are about half: the 8-bit attention and output-head tables 38%, the
  2-bit experts 15%. The other half is ~1,200 small kernels of 3-5 us each (norms, element-wise steps, copies,
  converting activations to 8 bits, routing).
- **The weights are not the bottleneck.** A token reads 2.4 GB, about 3 ms at the card's speed. Halving the experts'
  size (GSQ 2-bit vs Unsloth 4-bit) did not change writing speed (96 vs 95 tokens a second).
- **Reading a long prompt:** the linear attention's gated-delta-net kernel takes 15% of a 4,096-token prompt, one
  serial kernel per layer -- llama.cpp's source marks a chunked version as a TODO.
- **Drafting is the biggest writing-speed lever on this card:** llama.cpp's MTP drafting gives +33-37% (93 -> 128
  tokens a second), more than any kernel here could.

**What we'd do next, in order:**

1. **Fuse the small steps** (about half of a token): norm + 8-bit conversion, the element-wise chains around the linear
   attention, router + reduction -- in a private llama.cpp build, measured end to end.
2. **A chunked gated-delta-net kernel for prompt reading** (Gated DeltaNet's chunkwise form), filling llama.cpp's own
   TODO; it matters most for long prompts (an agent's 27K-token requests).
3. **A Marlin-style prompt kernel for Volta** (hardest of the list): unpack quantized weights inside a tensor-core
   multiply, instead of llama.cpp's dequantize-to-fp16-then-cuBLAS, which is 39% of reading a 4,096-token prompt.
   [Marlin](https://github.com/IST-DASLab/marlin) itself needs Ampere (`cp.async`, `ldmatrix`, the m16n8k16 MMA);
   on the V100 the asynchronous copy would be replaced by register-staged double buffering (as GEMMs did before
   Ampere) and the MMA by Volta's own. Our first tensor-core kernel (`tc_q4`) takes 1.6-2.3x cuBLAS's time at 512 tokens,
   so this starts well behind (go / no-go bar: [STAGE3.md](STAGE3.md), issues #1 and #5); the unpacking itself (the `0x6400` trick, from FasterTransformer) is the part we already have.
4. **Our 8-bit kernel in place of llama.cpp's Q8_0 at one token** (76% vs 68% of memory speed on the table family that
   is 38% of a token): roughly 4% end to end, cheap to try.
5. **Not worth it:** our 2-bit kernel in a model (llama.cpp's Q2_0 is faster on big tables, and experts are 15% of a
   token), and a new engine in Python around our kernels (it would add launch cost to every one of ~1,500 kernels a
   token, which llama.cpp's CUDA graphs avoid).
6. **Still to measure:** Strata, which writes Qwen3.6 faster than llama.cpp on this card (~110 vs ~95 tokens a
   second), profiled the same way -- to see what it does differently.

## Run it

```
make                                   # CUDA 12.9, sm_70 (nvcc; does not run the binary)
python3 bench/test_load_flag.py        # gpu-load sample checks, no GPU
./bakeoff --check                      # correctness (exit 1 on a failure; needs the V100)
./bakeoff > sweep.csv                  # the sweep
bench/overnight.sh                     # check, sabotage check, ROUNDS sweeps, LLAMA_ROUNDS llama.cpp sweeps
```

GitHub Actions (`.github/workflows/ci.yml`) runs the sample checks, ruff, and shellcheck, and compiles with the CUDA 12.9.2 devel image for sm_70. That job does not run the binary.

Hardware: Tesla V100-PCIe-32GB, i7-13700KF, 16 GB RAM, Linux, CUDA 12.9, driver 580.173.02.

Made with AI assistance (Claude); the idea of the bake-off came from an AI research assistant (Muse). MIT licence.
