# Issue #3: fusing llama.cpp's small per-token kernels on the V100

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

- 10 rounds of `llama-bench -p 0 -n 128 -r 3`, alternating which side goes first, SM clock locked at 1380 MHz;
- greedy text (256 tokens) on both sides, compared byte for byte;
- `llama-perplexity` on 40 chunks of wiki.test.raw on both sides, compared chunk by chunk.

Since 2026-10-21 each run also:

- runs perplexity a second time with 8-row batches (`-ub 8`);
- samples the V100 every 100 ms, and `load_flag.py` flags clock dips;
- logs the llama.cpp commit plus a hash of any uncommitted diff, which it also saves.

[bench/fuse_ab_summary.py](bench/fuse_ab_summary.py) gives each side's median and the per-round paired gain.

Every change so far keeps llama.cpp's arithmetic order, so text and perplexity must come out **identical**, not just
close.

## Results (tokens per second, median of 10 rounds)

| change | patch | off | on | gain | text | perplexity |
| --- | --- | ---: | ---: | ---: | --- | --- |
| residual ADD -> RMS_NORM -> MUL in one kernel (30 per token) | [apply_add_norm_fusion.py](patches/apply_add_norm_fusion.py) | 88.79 | 89.88 | +1.2% | identical | identical (5.8300) |
| + chains of 2-4 adds (routed + shared experts + residual; 40 more per token) | [apply_add_chain_norm_fusion.py](patches/apply_add_chain_norm_fusion.py) | 88.76 | 90.82 | +2.3% (both) | identical | identical |
| reuse a q8_1 conversion within a graph run (90 of 291 per token were repeats) | [apply_q8_cache.py](patches/apply_q8_cache.py) | 90.85 | 92.80 | +2.1% | identical | identical |
| the norm writes its own q8_1 copy (80 more conversions per token skipped) | [apply_q8_fold_norm.py](patches/apply_q8_fold_norm.py) | 92.64 | 93.15 | +0.55% | identical | identical |

The rows above are 10-round medians from four separate A/B runs, each against the previous row's build.

### One run, stock against all four ([results/2026-10-21-fuse-all-vs-stock](results/2026-10-21-fuse-all-vs-stock))

This run tests everything at once (llama.cpp `478e0a253`, nothing uncommitted: the diff hash in the log is that of an
empty diff). Stock means the same binary with `GGML_CUDA_NO_ADD_NORM_FUSION=1 GGML_CUDA_NO_Q8_CACHE=1`. It ran 30
rounds, alternating, with the SM clock locked and the V100 sampled every 100 ms; 0 of the 60 bench runs saw a clock
dip.

| rounds | CPU | stock | all four | paired gain (median, range) | fused faster |
| --- | --- | ---: | ---: | --- | --- |
| all 30 | | 90.43 | 94.94 | **+4.97%** (+3.43 to +8.16) | 30 / 30 |
| 1-6 | power saving | 88.90 | 93.30 | +4.94% (+4.80 to +5.21) | 6 / 6 |
| 7-13 | power saving, a 26 GB download running from 18:45 | 90.49 | 94.95 | +4.90% | 7 / 7 |
| 14-30 | bench mode (performance governor, C2/C3 sleep off) | 90.44 | 94.98 | +5.00% | 17 / 17 |

The paired gain compares each round's two sides with each other, which cancels slow drift in the machine.

- **Greedy text** was byte-identical.
- **Perplexity was identical in all 40 chunks at both batch sizes:** 5.8300 with llama.cpp's default 512-row
  batches, and 5.8202 with `-ub 8`. The 8-row batches exercise the norm folding, which only applies at 1-8 rows.
  (5.8202 differs from 5.8300 on stock too; small batches add up in a different order.)

**Nsight Systems** (16 tokens each way, `nsys/`) confirms where the gain comes from:

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

### How strong these numbers are

- **The per-row results.** The four per-row runs are 10-round medians on power saving, with no clock log. Their logs
  name the previous commit, not the patch under test. The patch scripts do apply to stock b11434 and rebuild the
  private branch byte for byte, so what was tested is identifiable. The combined run above logs the commit and a diff
  hash, samples the clock, and has 30 rounds.
- **Outlier rounds.** A few single bench runs come out 1-4 tok/s fast on either side. On power saving that was mostly
  the CPU waking up; in bench mode it still happened twice (rounds 29-30: 98.56 fused, 91.73 stock), cause unknown. The
  paired medians don't depend on them.
- **The perplexity text.** All perplexity checks use 40 chunks of wiki.test.raw at 512 context. Before this run, none
  of them exercised the norm folding.

**A bug the perplexity check caught.** The first version of the conversion cache gained the same +2.3% and produced
identical greedy text, but its perplexity was 5.8296 against 5.8300. The cause was llama.cpp's per-expert fallback in
`mul_mat_id`, which perplexity uses and decoding doesn't. It builds throwaway stack tensors at one address over one
scratch buffer, with different contents each time, and the cache reused conversions across them. The fix: reuse only
tensors that are nodes or leafs of the running graph. The first version's results are kept under
[results/2026-10-20-fuse-q8-cache-v1-bug](results/2026-10-20-fuse-q8-cache-v1-bug) and are not to be quoted.

## Still open

- **Remaining conversions.** 121 per token are not yet removed (201 distinct before the norm folding; confirmed by Nsight): the gated activations that feed the down
  projections, the linear attention's gated output, and the full attention's gated output.
- **The rest of the done line.** The whole set still has to be measured stacked with MTP drafting and the 8-bit
  kernel swap, as issue #3 requires.
- **A next step for the conversions.** 80 of the remaining 121 feed the down projections, and their inputs are only
  512 wide. The multiply can convert those itself (2 KB per block), which removes the separate launch.
