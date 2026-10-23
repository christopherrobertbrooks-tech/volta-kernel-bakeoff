# Issue #11: q8_1 conversion cache, more than one stream

The fix ([patches/apply_q8_cache_mainstream.py](../../patches/apply_q8_cache_mainstream.py)) makes the cache work on
llama.cpp's main stream only. On any other stream there is no reuse and no norm fold, just the normal conversion. In
the private llama.cpp it is commit `bcc1ffe7d`, on top of the sigmoid fusion `49be8a17c`.

The tests are [bench/q8cache_test.cpp](../../bench/q8cache_test.cpp) and
[bench/q8cache-test.sh](../../bench/q8cache-test.sh). Each case runs with the cache on and with it off
(`GGML_CUDA_NO_Q8_CACHE=1`), at `GGML_CUDA_GRAPH_OPT=0` and `=1` (llama.cpp's concurrent streams), and the outputs
are compared byte for byte:

- **A** an in-place op on a cached activation: the next multiply must not reuse the stale conversion;
- **B** three multiplies of one normed activation, which are branches for the stream optimiser;
- **C** graph B run twice, on input 1 and then input 2, unsynced; the instrument also checks that C differs from B with
  the cache off.

| build | A | B | C | log |
| --- | --- | --- | --- | --- |
| the fix | same | same | same | `fixed.log`, `fixed-again/` (after the sabotage reverts) |
| sabotage: follows in-place results ([sabotage_q8_cache_inplace.py](../../patches/sabotage_q8_cache_inplace.py)) | **differs** | same | same | `sabotage-inplace.log` |
| sabotage: entries and slots kept across runs ([sabotage_q8_cache_noreset.py](../../patches/sabotage_q8_cache_noreset.py)) | same | same | **differs** | `sabotage-noreset.log` |

Each break makes its test fail, at both GRAPH_OPT settings, which also shows the cache is engaged in these graphs.

A first "no reset" sabotage kept the entries but still reset the arena offset. The tests passed it, correctly: the
next run rewrote the very slot each stale entry pointed to, with that run's own data, so no stale value was read. The
break that counts keeps the slots too.

**What these tests can't show.** The real multi-stream race (one stream reading a slot while the next graph run's reset
lets another stream overwrite it) depends on timing, and no test here reproduces it reliably. The fix closes it by
construction: nothing off the main stream touches the arena, and all arena writes and reads on the main stream are
ordered.
