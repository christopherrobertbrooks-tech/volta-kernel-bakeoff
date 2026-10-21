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

Every change so far keeps llama.cpp's arithmetic order, so text and perplexity must come out **identical**, not just
close.

## Results (tokens per second, median of 10 rounds)

| change | patch | off | on | gain | text | perplexity |
| --- | --- | ---: | ---: | ---: | --- | --- |
| residual ADD -> RMS_NORM -> MUL in one kernel (30 per token) | [apply_add_norm_fusion.py](patches/apply_add_norm_fusion.py) | 88.79 | 89.88 | +1.2% | identical | identical (5.8300) |
| + chains of 2-4 adds (routed + shared experts + residual; 40 more per token) | [apply_add_chain_norm_fusion.py](patches/apply_add_chain_norm_fusion.py) | 88.76 | 90.82 | +2.3% (both) | identical | identical |
| reuse a q8_1 conversion within a graph run (90 of 291 per token were repeats) | [apply_q8_cache.py](patches/apply_q8_cache.py) | 90.85 | 92.80 | +2.1% | identical | identical |
| the norm writes its own q8_1 copy (80 more conversions per token skipped) | [apply_q8_fold_norm.py](patches/apply_q8_fold_norm.py) | 92.64 | 93.15 | +0.55% | identical | identical |

Everything against stock llama.cpp: **88.76 -> 93.15 tokens per second (+4.9%)**, the medians of separate A/B runs
chained together. The last row gained less than the ~2% that 80 skipped conversions suggested: the norm kernel now
does the rounding and block writes itself, which costs back most of the saving. It won 9 of 10 rounds; the tenth
unfused round came out at 93.43.

**A bug the perplexity check caught.** The first version of the conversion cache gained the same +2.3% and produced
identical greedy text, but its perplexity was 5.8296 against 5.8300. The cause was llama.cpp's per-expert fallback in
`mul_mat_id`, which perplexity uses and decoding doesn't. It builds throwaway stack tensors at one address over one
scratch buffer, with different contents each time, and the cache reused conversions across them. The fix: reuse only
tensors that are nodes or leafs of the running graph. The first version's results are kept under
[results/2026-10-20-fuse-q8-cache-v1-bug](results/2026-10-20-fuse-q8-cache-v1-bug) and are not to be quoted.

## Still open

- **Remaining conversions.** 121 per token are not yet removed (201 distinct before the norm folding): the gated activations that feed the down
  projections, the linear attention's gated output, and the full attention's gated output.
- **The rest of the done line.** The whole set still has to be measured stacked with MTP drafting and the 8-bit
  kernel swap, as issue #3 requires.
- **Outliers.** A few single rounds came out 2-4 tok/s high, on both sides. The medians aren't affected, and the
  cause isn't known.
