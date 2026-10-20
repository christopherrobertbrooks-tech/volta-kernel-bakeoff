# volta-kernel-bakeoff

Our own quantized matrix-multiply kernels for the **Tesla V100** (sm_70, 2017), raced against **cuBLAS** (NVIDIA's
fp16 tensor-core GEMM) and **llama.cpp's** own CUDA kernels, at the real table shapes of the models we run.

*Work in progress -- results below are from single sweeps; the 30-round run's medians replace them when it's in.*

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
- Median of 5 x 50 calls after warm-up; 30 back-to-back rounds agree to 0.2% (worst 4%). The card stayed at 31-45 C and
  1,380 MHz.

## First results (one token, microseconds, lower is better)

| table | cuBLAS fp16 | llama.cpp Q4_0 | **ours 4-bit** | llama.cpp Q2_K | **ours 2-bit** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next Coder attention 12288x2560 | 101.3 | 32.4 | 30.1 | 31.0 | **23.4** |
| Qwen3.6 attention 8192x2048 | 48.7 | 20.4 | 17.7 | 19.3 | **14.2** |
| Qwen3-1.7B MLP up 6144x2048 | 43.9 | 16.4 | 13.9 | 16.0 | **12.2** |
| Flash-Next Coder expert 640x2560 | 10.9 | 7.5 | 6.5 | 8.8 | **6.2** |

- One token: our 4-bit kernel is 5-15% ahead of llama.cpp's Q4_0 on most tables, our 2-bit up to 30% ahead of its
  fastest 2-bit. Our 2-bit format is simpler (one scale per 32, no codebook), so part of that is less work per weight;
  llama.cpp's IQ2 formats keep more of a model's quality.
- Ten experts in one launch: 22 us against 56 for cuBLAS batched and 109 for ten separate cuBLAS calls.
- **Where we lose:** 8 tokens at once (llama.cpp Q4_0 53 us vs ours 83 on the 12288x2560 table), and prompts: our first
  tensor-core kernel is 2-4x slower than cuBLAS.
- **The times table loses on this card.** Against a badly built multiply kernel it looked 13-17% faster; against a good
  one it is slower everywhere -- the V100 multiplies cheaply, building and reading the table costs more.
- **Found on the way:** a 2-bit kernel that read its activations from shared memory at a 16-float stride per lane was
  slower than 4-bit (bank conflicts); reading them from global memory made it 1.9x faster.

## Run it

```
make                                   # CUDA 12.x, sm_70
./bakeoff --check                      # correctness (exit 1 on a failure)
./bakeoff > sweep.csv                  # the sweep
bench/overnight.sh                     # check, sabotage check, ROUNDS sweeps, LLAMA_ROUNDS llama.cpp sweeps
```

Hardware: Tesla V100-PCIe-32GB, i7-13700KF, 16 GB RAM, Linux, CUDA 12.9, driver 580.173.02.

Made with AI assistance (Claude); the idea of the bake-off came from an AI research assistant (Muse). MIT licence.
