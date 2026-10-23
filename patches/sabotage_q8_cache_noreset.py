# DELIBERATELY BROKEN, for testing only (#11 test C): the q8_1 cache keeps its entries from one graph run to the next,
# so a later run reuses an earlier run's conversion of a tensor at the same address. Apply on top of
# apply_q8_cache_mainstream.py, run bench/q8cache-test.sh (test C must FAIL), then revert the file.


def edit(path, old, new):
    with open(path) as f:
        s = f.read()
    assert s.count(old) == 1, (path, old[:70])
    with open(path, "w") as f:
        f.write(s.replace(old, new))


edit("ggml/src/ggml-cuda/ggml-cuda.cu",
     "    cuda_ctx->q8_entries.clear();\n",
     "    // SABOTAGE: entries kept across graph runs\n")
print("sabotaged (no reset)")
