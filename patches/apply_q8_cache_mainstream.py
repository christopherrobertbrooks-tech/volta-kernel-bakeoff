# Issue #11 (on top of apply_q8_fold_norm.py): the q8_1 conversion cache works on the main stream only. Its arena's
# offset resets at the start of every graph run; that is safe while every write and read of the arena is ordered on one
# stream, and only then. With concurrent streams (GGML_CUDA_GRAPH_OPT=1) a copy on another stream could still be read
# when the next run overwrites it -- so on any other stream there is no cache (no reuse, no norm fold): the normal
# conversion path. Decode with the default settings runs on the main stream, so nothing changes there.


def read(path):
    with open(path) as f:
        return f.read()


def write(path, s):
    with open(path, "w") as f:
        f.write(s)


def edit(path, old, new, count=1):
    s = read(path)
    assert s.count(old) == count, (path, old[:70], s.count(old))
    write(path, s.replace(old, new))


CC = "ggml/src/ggml-cuda/common.cuh"
edit(CC, '''    char * q8_find(const ggml_tensor * t, size_t size, cudaStream_t s) {
        if (!q8_in_graph(t)) return nullptr;''', '''    char * q8_find(const ggml_tensor * t, size_t size, cudaStream_t s) {
        if (s != stream(device, 0) || !q8_in_graph(t)) return nullptr;   // main stream only (issue #11)''')
edit(CC, '''    char * q8_add(const ggml_tensor * t, size_t size, cudaStream_t s) {   // a slot for t's q8_1 copy, or nullptr
        if (!q8_in_graph(t)) return nullptr;''', '''    char * q8_add(const ggml_tensor * t, size_t size, cudaStream_t s) {   // a slot for t's q8_1 copy, or nullptr
        if (s != stream(device, 0) || !q8_in_graph(t)) return nullptr;   // main stream only (issue #11)''')
print("patched")
