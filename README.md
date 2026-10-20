# volta-kernel-bakeoff

Our own quantized matrix-multiply kernels for the **Tesla V100** (sm_70, 2017), raced against **cuBLAS** (NVIDIA's
fp16 tensor-core GEMM) and **llama.cpp's** own CUDA kernels, at the real table shapes of the models we run.

*Work in progress. Numbers below are medians over 30 rounds (ours, cuBLAS) and 10 rounds (llama.cpp) run back to back on
2026-10-19; every round is in [results/2026-10-19](results/2026-10-19), medians in
[medians.csv](results/2026-10-19/medians.csv) ([results/summarize.py](results/summarize.py)).*

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
  quantized weights; `BAKEOFF_SABOTAGE=1` breaks the packing on purpose and the check must fail (it does, for every kernel).
- Each round takes the median of 5 x 50 calls after warm-up; the table reports the median round. Fastest and slowest
  round differ by 0.9% typically. The card stayed at 31-45 C and 1,380 MHz.

## Results (one token, microseconds, lower is better)

| table | cuBLAS fp16 | llama.cpp Q4_0 | **ours 4-bit** | llama.cpp Q2_K | llama.cpp Q2_0 | **ours 2-bit** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next Coder attention 12288x2560 | 101.2 | 32.4 | 30.2 | 31.0 | **18.3** | 23.5 |
| Qwen3.6 attention 8192x2048 | 48.8 | 20.3 | 17.8 | 19.3 | **12.1** | 14.3 |
| Qwen3-1.7B MLP up 6144x2048 | 43.6 | 16.4 | 13.9 | 16.0 | **10.2** | 12.2 |
| Flash-Next Coder expert 640x2560 | 10.7 | 6.8 | 6.3 | 8.8 | 7.4 | **6.2** |
| Qwen3.6 expert 512x2048 | 9.7 | 7.1 | **5.2** | 8.2 | 7.5 | 5.5 |
| Qwen3.6 expert down 2048x512 | 6.9 | 6.4 | 4.7 | 7.9 | 7.1 | **4.0** |

Ours 4-bit = `fused_q4_v2w`, ours 2-bit = `q2w_alu_v3` (one fp16 scale per 32, 2.5 bits per weight). llama.cpp's Q2_0
(one fp16 scale per 64, 2.25 bits) is the same kind of plain 2-bit format; its column is 10 rounds from a second run
([results/2026-10-19-llama-q2_0](results/2026-10-19-llama-q2_0); its other columns agree with the first run within
1-3%). llama.cpp's times include converting the activations to 8 bits for each multiply, as in a real model.

- **4-bit, one token:** ours takes 7-27% less time than llama.cpp's Q4_0 on 7 of the 9 tables, and is 3-4% slower
  on the two with 6,144-long rows (Flash-Next attention out, Qwen3-1.7B MLP down).
- **2-bit, one token:** ours takes 12-50% less time than llama.cpp's Q2_K, but Q2_K is a more complex format (a scale
  and offset per 16 weights, 256 per block). Against llama.cpp's own plain 2-bit, **Q2_0, ours loses on the big tables
  (18-28% slower) and wins on the expert-sized ones (17-44% less time)**. Nsight Compute on the Qwen3.6 attention
  table: ours keeps the GPU's arithmetic 60% busy and its memory only 37% -- it is limited by unpacking -- while Q2_0
  converts the activations to 8-bit integers once and multiplies four weights per instruction (`dp4a`).
- **2-bit quality, measured** (Qwen3.6-35B-A3B, WikiText-2, llama.cpp; from the gsq-gguf converter, not yet published): a plain 2-bit format with one
  scale per group, filled with ISTA-DASLab's GSQ weights (about the best such a format can hold), scores perplexity
  7.93; Unsloth's UD-Q2_K_XL (llama.cpp's K and IQ formats, ~2.6 bits on the experts, same file size) 7.26; its 4-bit
  UD-Q4_K_XL 6.73 (lower is better). The plain format's speed comes with a measurable quality cost.
- Ten experts in one launch: 22 us against 56 for cuBLAS batched and 110 for ten separate cuBLAS calls (Qwen3.6's
  eight: 12 us against 31 and 80).
- **Where we lose:** 8 tokens at once (llama.cpp Q4_0 53 us vs our best, `fused_q4_v2`, 83 on the 12288x2560 table;
  `v2w` is the fastest at one token and the slowest at eight), and prompts: our first
  tensor-core kernel is 2-4x slower than cuBLAS.
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

## Run it

```
make                                   # CUDA 12.x, sm_70
./bakeoff --check                      # correctness (exit 1 on a failure)
./bakeoff > sweep.csv                  # the sweep
bench/overnight.sh                     # check, sabotage check, ROUNDS sweeps, LLAMA_ROUNDS llama.cpp sweeps
```

Hardware: Tesla V100-PCIe-32GB, i7-13700KF, 16 GB RAM, Linux, CUDA 12.9, driver 580.173.02.

Made with AI assistance (Claude); the idea of the bake-off came from an AI research assistant (Muse). MIT licence.
