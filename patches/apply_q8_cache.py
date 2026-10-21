# Issue #3, fusion 2a: reuse an activation's q8_1 conversion within one graph run. In Qwen3.6-35B-A3B 90 of the 291
# conversions per written token are repeats (the MoE input for routed + shared experts; the attention input for
# qkv + gate, q + k + v). The reused bytes are exactly what converting again would produce, so output is unchanged.
# A hit needs the same tensor (or a reshape/view of it), address, shape, strides and stream, in the same graph run,
# and both the tensor and its root must be nodes or leafs of that graph: llama.cpp's per-expert fallback in
# mul_mat_id builds throwaway stack tensors at one address over one scratch buffer with different contents each time
# (the first version reused across them: perplexity 5.8296 vs 5.8300).
# Copies live in a fixed 32 MB device buffer allocated once outside CUDA-graph capture; when it is missing or full
# the normal pool path is used. GGML_CUDA_NO_Q8_CACHE=1 switches the cache off.
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

edit("ggml/src/ggml-cuda/common.cuh", "#include <unordered_map>\n", "#include <unordered_map>\n#include <unordered_set>\n")
edit("ggml/src/ggml-cuda/common.cuh", "    int curr_stream_no = 0;\n", '''    int curr_stream_no = 0;

    // private (volta-kernel-bakeoff #3): activations already converted to q8_1 in this graph run, for reuse
    struct q8_cache_entry {
        const void * root; const void * data; int64_t ne[4]; size_t nb[4]; cudaStream_t stream; char * q8; size_t size;
    };
    char * q8_arena      = nullptr;
    size_t q8_arena_size = 0;
    size_t q8_arena_off  = 0;
    std::vector<q8_cache_entry> q8_entries;
    const ggml_cgraph * q8_graph = nullptr;                  // the graph being run, and its tensors (built on first use)
    std::unordered_set<const void *> q8_graph_tensors;
    bool q8_graph_tensors_built = false;
''')

G = "ggml/src/ggml-cuda/ggml-cuda.cu"
edit(G, '''    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {''', '''    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (q8_arena != nullptr) {
        CUDA_CHECK(cudaFree(q8_arena));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {''')
edit(G, '''static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
''', '''static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    cuda_ctx->q8_arena_off = 0;           // a new graph run: no conversion from an earlier run is reused
    cuda_ctx->q8_entries.clear();
    cuda_ctx->q8_graph = cgraph;
    cuda_ctx->q8_graph_tensors_built = false;
''')

edit("ggml/src/ggml-cuda/mmvq.cu", '''    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }''', '''    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_size     = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * q8_ptr = nullptr;
    bool   q8_hit = false;
    static const bool no_q8_cache = getenv("GGML_CUDA_NO_Q8_CACHE") != nullptr && atoi(getenv("GGML_CUDA_NO_Q8_CACHE"));
    if (!no_q8_cache) {
        const ggml_tensor * root = src1;   // through reshapes / views only: an in-place op's result is not the same data
        while (root->view_src && (root->op == GGML_OP_RESHAPE || root->op == GGML_OP_VIEW)) {
            root = root->view_src;
        }
        if (!ctx.q8_graph_tensors_built && ctx.q8_graph != nullptr) {
            ctx.q8_graph_tensors.clear();
            for (int k = 0; k < ctx.q8_graph->n_nodes; ++k) ctx.q8_graph_tensors.insert(ctx.q8_graph->nodes[k]);
            for (int k = 0; k < ctx.q8_graph->n_leafs; ++k) ctx.q8_graph_tensors.insert(ctx.q8_graph->leafs[k]);
            ctx.q8_graph_tensors_built = true;
        }
        const bool in_graph = ctx.q8_graph_tensors_built && ctx.q8_graph_tensors.count(src1) && ctx.q8_graph_tensors.count(root);
        for (const auto & e : ctx.q8_entries) {
            if (!in_graph) break;
            if (e.root == root && e.data == src1->data && e.stream == stream && e.size == q8_size &&
                e.ne[0] == src1->ne[0] && e.ne[1] == src1->ne[1] && e.ne[2] == src1->ne[2] && e.ne[3] == src1->ne[3] &&
                e.nb[0] == src1->nb[0] && e.nb[1] == src1->nb[1] && e.nb[2] == src1->nb[2] && e.nb[3] == src1->nb[3]) {
                q8_ptr = e.q8;
                q8_hit = true;
                break;
            }
        }
        if (!q8_hit) {
            if (ctx.q8_arena == nullptr) {   // allocate once, never while a CUDA graph is being captured
                cudaStreamCaptureStatus st;
                CUDA_CHECK(cudaStreamIsCapturing(stream, &st));
                if (st == cudaStreamCaptureStatusNone) {
                    ctx.q8_arena_size = 32u << 20;
                    CUDA_CHECK(cudaMalloc(&ctx.q8_arena, ctx.q8_arena_size));
                }
            }
            const size_t off = GGML_PAD(ctx.q8_arena_off, 256);
            if (in_graph && ctx.q8_arena != nullptr && off + q8_size <= ctx.q8_arena_size) {
                q8_ptr = ctx.q8_arena + off;
                ctx.q8_arena_off = off + q8_size;
                ggml_backend_cuda_context::q8_cache_entry e = { root, src1->data, {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]},
                                                               {src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]}, stream, q8_ptr, q8_size };
                ctx.q8_entries.push_back(e);
            }
        }
    }
    if (q8_ptr == nullptr) {
        q8_ptr = src1_q8_1.alloc(q8_size);
    }
    if (!q8_hit) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, q8_ptr, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }''')
edit("ggml/src/ggml-cuda/mmvq.cu", "        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,",
     "        src0->data, src0->type, q8_ptr, ids_d, fusion_local, dst_d, ne00,")
print("patched")
