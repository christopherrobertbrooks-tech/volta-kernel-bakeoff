# DELIBERATELY BROKEN, for testing only (#11 test A): the q8_1 cache follows ANY view_src back to its source, including
# an in-place op's result (same memory, new values), so it reuses a stale conversion. Apply on top of
# apply_q8_cache_mainstream.py, run bench/q8cache-test.sh (test A must FAIL), then revert the file.


def edit(path, old, new):
    with open(path) as f:
        s = f.read()
    assert s.count(old) == 1, (path, old[:70])
    with open(path, "w") as f:
        f.write(s.replace(old, new))


edit("ggml/src/ggml-cuda/common.cuh",
     "        while (t->view_src && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {",
     "        while (t->view_src) {   // SABOTAGE: through in-place results too")
print("sabotaged (in-place)")
