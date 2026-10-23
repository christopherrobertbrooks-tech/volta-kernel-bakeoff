# Issue #3: fusing llama.cpp's small per-token kernels on the V100

**Finished 2026-10-23 (repository archived).** All five changes together: about +6% decode, as an estimate (four measured
together +4.97%, the fifth +1.14%); never measured as one run.

Work in progress on branch `fuse`. The changes live in a **private** llama.cpp copy: worktree `~/llama.cpp-fuse` on the
gateway, branch `fuse`, from b11434 / `5e03bdd87`. They are applied by the scripts in [patches/](patches/), and none of
this is sent upstream. Model: Qwen3.6-35B-A3B, Unsloth UD-Q4_K_XL; decode only.

## Where a written token's time goes

[bench/rank_token_kernels.py](bench/rank_token_kernels.py) on the 2026-10-19 Nsight Systems trace
([results/2026-10-20-issue3-rank](results/2026-10-20-issue3-rank)) shows 1,526 kernels per token. The matrix-vector
multiplies take about half the time. The biggest pieces of the rest that llama.cpp did not already fuse:

- 291 activation conversions to q8_1 (0.80 ms, 7%);
- 63 residual add -> norm pairs;
- small f32 multiply -> sigmoid / top-k;
- the linear attention's state copies.

## How each change is judged

[bench/fuse-ab.sh](bench/fuse-ab.sh) uses one binary with the change switched off by an environment variable, so
nothing else differs between the two sides. Each run does:

- `ROUNDS` rounds (10 to 30 here) of `llama-bench -p 0 -n 128 -r 3`, alternating which side goes first, SM clock locked
  at 1380 MHz;
- greedy text (256 tokens) on both sides, compared byte for byte;
- `llama-perplexity` on 40 chunks of wiki.test.raw on both sides, compared chunk by chunk.

Since 2026-10-21 each run also:

- runs perplexity a second time with 8-row batches (`-ub 8`);
- samples the V100 every 100 ms, and `load_flag.py` flags clock dips;
- logs the llama.cpp commit plus a hash of any uncommitted diff, which it also saves.

[bench/fuse_ab_summary.py](bench/fuse_ab_summary.py) gives each side's median and the per-round paired gain.

Every change so far keeps llama.cpp's arithmetic order, so text and perplexity must come out **identical**, not just
close.

## Results (tokens per second, medians)

| change | patch | off | on | gain | text | perplexity |
| --- | --- | ---: | ---: | ---: | --- | --- |
| residual ADD -> RMS_NORM -> MUL in one kernel (30 per token) | [apply_add_norm_fusion.py](patches/apply_add_norm_fusion.py) | 88.79 | 89.88 | +1.2% | identical | identical (5.8300) |
| + chains of 2-4 adds (routed + shared experts + residual; 40 more per token) | [apply_add_chain_norm_fusion.py](patches/apply_add_chain_norm_fusion.py) | 88.76 | 90.82 | +2.3% (both) | identical | identical |
| reuse a q8_1 conversion within a graph run (90 of 291 per token were repeats) | [apply_q8_cache.py](patches/apply_q8_cache.py) | 90.85 | 92.80 | +2.1% | identical | identical |
| the norm writes its own q8_1 copy (80 more conversions per token skipped) | [apply_q8_fold_norm.py](patches/apply_q8_fold_norm.py) | 92.64 | 93.15 | +0.55% | identical | identical |
| small f32 multiply -> sigmoid in one call (70 per token: ssm_beta, shared-expert gate) | [apply_mmvf_sigmoid.py](patches/apply_mmvf_sigmoid.py) | 94.94 | 95.99 | +1.14% (19/20) | identical | identical |

The first four rows are 10-round medians from separate A/B runs on power saving, each against the previous row's build.
The sigmoid row ([results/2026-10-22-fuse-mmvf-sigmoid](results/2026-10-22-fuse-mmvf-sigmoid)) is 20 rounds in bench
mode with the clock sampled (0 of 40 runs flagged), paired against the four-fusion build: +1.14%, fused faster in 19 of
20 (one round -0.61%). It fuses at one token only, so the `-ub 8` perplexity doesn't exercise it; the greedy text does.

### One run, stock against all four ([results/2026-10-21-fuse-all-vs-stock](results/2026-10-21-fuse-all-vs-stock))

This run tests everything at once: llama.cpp `478e0a253`, nothing uncommitted (the diff hash in the log is that of an
empty diff). One gap Grok found: the script didn't rebuild, and the `bench-*.csv` files report the binary's build as
`bc38f70ad` (llama.cpp records the commit when it configures, so that can be stale). The norm folding was in the binary
(Nsight shows its kernel), but a clean tree alone doesn't prove which binary ran. From 2026-10-22 the script rebuilds
first and logs the binaries' sha256. Stock means the same binary with `GGML_CUDA_NO_ADD_NORM_FUSION=1 GGML_CUDA_NO_Q8_CACHE=1`. It ran 30
rounds, alternating, with the SM clock locked and the V100 sampled every 100 ms; 0 of the 60 bench runs saw a clock
dip.

| rounds | CPU | stock | all four | paired gain (median, range) | fused faster |
| --- | --- | ---: | ---: | --- | --- |
| all 30 | | 90.43 | 94.94 | **+4.97%** (+3.43 to +8.16) | 30 / 30 |
| 1-6 | power saving | 88.90 | 93.30 | +4.94% (+4.80 to +5.21) | 6 / 6 |
| 8-13 | power saving, a 26 GB download running (it started after round 7) | 90.51 | 95.01 | +4.91% (+4.82 to +5.89) | 6 / 6 |
| 15-30 | bench mode (performance governor, C2/C3 sleep off) | 90.44 | 94.98 | +5.02% (+3.43 to +8.16) | 16 / 16 |

Rounds 7 (+4.05%) and 14 (+4.80%) sit on the boundaries: bench mode was switched at 19:02:16, during round 14's stock
run. Groups are by round number (`fuse_ab_summary.py r15-30`).

The paired gain compares each round's two sides with each other, which cancels slow drift in the machine.

- **Greedy text** was byte-identical.
- **Perplexity was identical in all 40 chunks at both batch sizes, to the 4 decimals llama.cpp prints:** 5.8300 with
  llama.cpp's default 512-row batches, and 5.8202 with `-ub 8`. (5.8202 differs from 5.8300 on stock too; small batches
  add up in a different order.) Matching printed values isn't bit identity; the KL-divergence check below goes further.

**Nsight Systems** confirms where the gain comes from. Each side traced `llama-bench -n 16`, which is 16 tokens plus
a warmup token, so 17 decode steps; the counts below are totals / 17. The command is in
[nsys/COMMAND.txt](results/2026-10-21-fuse-all-vs-stock/nsys/COMMAND.txt), and the `.nsys-rep` traces are committed.

| per token | stock | all four |
| --- | ---: | ---: |
| kernels | 1,531 | 1,291 |
| q8_1 conversions | 291 (0.82 ms) | 121 (0.36 ms) |
| fused add/norm kernels | 0 | 81 |
| GPU busy time | 10.84 ms | 10.32 ms (-4.8%) |

The 4.8% less GPU time matches the 4.97% faster decode.

**The CPU matters as much as a fusion.** On power saving the gateway's cores drop into deep sleep (C3 takes ~1 ms to
wake) between llama.cpp's host steps. Keeping them awake made both sides ~1.7% faster, the same lift a running
download gave, and left the fused gain unchanged. From stock on power saving (88.90) to all four in bench mode
(94.98) is +6.8%. `~/bin/bench-mode on|off|status` on the gateway sets it; it does not survive a reboot.

### Checks after Grok's review (2026-10-22)

**Decode unchanged, with five changes on** ([results/2026-10-22-fuse-decode-check](results/2026-10-22-fuse-decode-check)):
private llama.cpp `bcc1ffe7d` (the four fusions, the sigmoid fusion and the #11 fix), clean tree. The script rebuilt
before measuring and logged the binaries' sha256 (`libggml-cuda.so` e88f013c0dfca047). Stock is everything off.

- **Greedy text:** byte-identical.
- **Perplexity:** all 40 chunks the same at `-ub 512` (5.8300) and `-ub 8` (5.8202).
- **KL divergence** (`llama-perplexity --kl-divergence`, 4 chunks, the fused side against the stock side's saved log
  probabilities): mean KLD 0.000000 (`-ub 512`) / 0.000001 (`-ub 8`), maximum 0.000061 / 0.000059, RMS of the probability
  difference 0.001%, same top token 100%. The maximum isn't zero because llama.cpp stores the reference log
  probabilities in 16 bits: **stock against its own saved run gives exactly the same numbers**. The two complete result
  tables (46 lines each: every chunk's perplexity to 6 decimals, every KLD and probability statistic) are
  byte-identical, at both batch sizes. So the fused output matches stock to the full resolution of this method. That is
  as far as these checks go; no logits were hashed.

**The fold at 8-row batches** ([results/2026-10-22-fuse-ub8-nsys](results/2026-10-22-fuse-ub8-nsys), Nsight Systems on
`llama-perplexity -ub 8 --chunks 2` with `--cuda-graph-trace=node`; command and `.nsys-rep` traces committed):

| | stock | everything on |
| --- | ---: | ---: |
| quantize_q8_1 | 42,635 | 20,705 |
| mul_mat_vec_q (same work) | 42,635 | 42,635 |
| add_rms_norm_mul_f32 | 0 | 10,449 (5,160 two-term, 5,160 three-term, 129 plain-norm fold `<1024,0>`) |
| all kernels | 236,364 | 204,114 |

Per chunk perplexity identical (4.0325, 5.7285). Every one of those norm kernels runs at 1-8 rows, where it writes its
q8_1 copy, so the fold runs in 8-row batches too.

**Issue #11** ([results/2026-10-22-q8-cache-streams](results/2026-10-22-q8-cache-streams)): the conversion cache now
works on the main stream only ([apply_q8_cache_mainstream.py](patches/apply_q8_cache_mainstream.py)). The new tests (an
in-place op, branches, two unsynced back-to-back runs; cache on vs off, with and without `GGML_CUDA_GRAPH_OPT=1`) pass,
and each fails on a deliberately broken cache. One limit (Grok): at `GGML_CUDA_GRAPH_OPT=1` these small graphs never
  actually run on a second stream (the outputs are byte-identical at both settings), so the tests don't exercise the
  multi-stream case; the fix covers it by construction, and it costs nothing on the main stream.

### How strong these numbers are

- **The per-row results.** The first four per-row runs are 10-round medians on power saving, with no clock log. Their
  logs name the previous commit, not the patch under test. The patch scripts do apply to stock b11434 and rebuild the
  private branch byte for byte, so what was tested is identifiable. The combined run above logs the commit and a diff
  hash, samples the clock, and has 30 rounds.
- **Outlier rounds.** A few single bench runs come out 1-4 tok/s fast on either side. On power saving that was mostly
  the CPU waking up; in bench mode it still happened twice (rounds 29-30: 98.56 fused, 91.73 stock), cause unknown. The
  paired medians don't depend on them.
- **The perplexity text.** All perplexity checks use 40 chunks of wiki.test.raw at 512 context. Before this run, none
  of them exercised the norm folding.
- **Where the remaining 121 conversions come from.** Nsight gives only the total. The per-input split comes from the
  debug log (`GGML_CUDA_Q8_DEBUG=1`, a build that prints every conversion with its input's name):
  [q8-conversions-by-input.txt](results/2026-10-21-fuse-all-vs-stock/q8-conversions-by-input.txt). Per token, 40 come
  from the routed experts' down input and 40 from the shared expert's, so 80 feed the down projections. 30 are the
  linear attention's gated output, 10 the full attention's, and 1 the final norm.

**A bug the perplexity check caught.** The first version of the conversion cache gained the same +2.3% and produced
identical greedy text, but its perplexity was 5.8296 against 5.8300. The cause was llama.cpp's per-expert fallback in
`mul_mat_id`, which perplexity uses and decoding doesn't. It builds throwaway stack tensors at one address over one
scratch buffer, with different contents each time, and the cache reused conversions across them. The fix: reuse only
tensors that are nodes or leafs of the running graph. The first version's results are kept under
[results/2026-10-20-fuse-q8-cache-v1-bug](results/2026-10-20-fuse-q8-cache-v1-bug) and are not to be quoted.

## Still open

- **Remaining conversions.** 121 per token are not yet removed (291 before; Nsight confirms the total, the debug log
  the split). They are the gated activations that feed the down projections (80), the linear attention's gated output
  (30), the full attention's gated output (10), and the final norm (1).
- **Converting the down projections' inputs inside the multiply: looked at, not worth it.** Those inputs are only 512
  wide, but `mul_mat_vec_q` splits each table over thousands of blocks, and every block would have to convert the whole
  input again (over 1,000 blocks per expert). That likely costs more than the ~3 us launch it saves. Having the
  activation kernel write the q8_1 copy doesn't fit either: each 32-value block of its output is computed by different
  GPU blocks.
- **The rest of the done line.** The whole set still has to be measured stacked with MTP drafting and the 8-bit kernel
  swap, as issue #3 requires.
- **Issue #11.** Fixed on this branch (main stream only); the multi-stream race itself is timing-dependent and not
  reproduced by any test.
- **Next fusion candidates.** The router multiply -> top-k (40 pairs per token, 0.6 ms together) and the linear
  attention's state copies (copy -> gather -> conv, ~0.4 ms).
