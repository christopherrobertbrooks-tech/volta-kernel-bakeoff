# Issue #3, fusion 2b (on top of apply_q8_cache.py): the norm kernel writes the q8_1 copy of its output itself and
# registers it in the conversion cache, so the multiplies that read the norm's output skip their own conversion.
# The 8-bit blocks are computed exactly as quantize_q8_1 does (one warp per 32 values, the same lane order, the same
# warp_reduce_max / warp_reduce_sum, d = amax / 127, roundf), so they are bit-identical. Covers the fused add(s) ->
# norm kernel and plain RMS_NORM -> MUL, at 1-8 rows with 1024-multiple widths. GGML_CUDA_NO_Q8_FOLD=1 switches it off.
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

# --- 1. the cache as context helpers (find / add), used by mmvq and the norm ---
CC = "ggml/src/ggml-cuda/common.cuh"
edit(CC, '''    std::unordered_set<const void *> q8_graph_tensors;
    bool q8_graph_tensors_built = false;
''', '''    std::unordered_set<const void *> q8_graph_tensors;
    bool q8_graph_tensors_built = false;

    static const ggml_tensor * q8_root(const ggml_tensor * t) {   // through reshapes / views only
        while (t->view_src && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->view_src;
        }
        return t;
    }
    bool q8_in_graph(const ggml_tensor * t) {   // only the running graph's own tensors (not stack temporaries)
        if (!q8_graph_tensors_built && q8_graph != nullptr) {
            q8_graph_tensors.clear();
            for (int k = 0; k < q8_graph->n_nodes; ++k) q8_graph_tensors.insert(q8_graph->nodes[k]);
            for (int k = 0; k < q8_graph->n_leafs; ++k) q8_graph_tensors.insert(q8_graph->leafs[k]);
            q8_graph_tensors_built = true;
        }
        return q8_graph_tensors_built && q8_graph_tensors.count(t) && q8_graph_tensors.count(q8_root(t));
    }
    char * q8_find(const ggml_tensor * t, size_t size, cudaStream_t s) {
        if (!q8_in_graph(t)) return nullptr;
        const ggml_tensor * root = q8_root(t);
        for (const auto & e : q8_entries) {
            if (e.root == root && e.data == t->data && e.stream == s && e.size == size &&
                e.ne[0] == t->ne[0] && e.ne[1] == t->ne[1] && e.ne[2] == t->ne[2] && e.ne[3] == t->ne[3] &&
                e.nb[0] == t->nb[0] && e.nb[1] == t->nb[1] && e.nb[2] == t->nb[2] && e.nb[3] == t->nb[3]) {
                return e.q8;
            }
        }
        return nullptr;
    }
    char * q8_add(const ggml_tensor * t, size_t size, cudaStream_t s) {   // a slot for t's q8_1 copy, or nullptr
        if (!q8_in_graph(t)) return nullptr;
        if (q8_arena == nullptr) {   // allocate once, never while a CUDA graph is being captured
            cudaStreamCaptureStatus st;
            CUDA_CHECK(cudaStreamIsCapturing(s, &st));
            if (st != cudaStreamCaptureStatusNone) return nullptr;
            q8_arena_size = 32u << 20;
            CUDA_CHECK(cudaMalloc(&q8_arena, q8_arena_size));
        }
        const size_t off = GGML_PAD(q8_arena_off, 256);
        if (off + size > q8_arena_size) return nullptr;
        q8_arena_off = off + size;
        q8_cache_entry e = { q8_root(t), t->data, {t->ne[0], t->ne[1], t->ne[2], t->ne[3]},
                             {t->nb[0], t->nb[1], t->nb[2], t->nb[3]}, s, q8_arena + off, size };
        q8_entries.push_back(e);
        return e.q8;
    }
''')

# --- 2. mmvq uses the helpers ---
MV = "ggml/src/ggml-cuda/mmvq.cu"
s = read(MV)
a = s.index("    if (!no_q8_cache) {\n        const ggml_tensor * root = src1;")
b = s.index("    if (q8_ptr == nullptr) {\n        q8_ptr = src1_q8_1.alloc(q8_size);")
s = s[:a] + '''    if (!no_q8_cache) {
        q8_ptr = ctx.q8_find(src1, q8_size, stream);
        q8_hit = q8_ptr != nullptr;
        if (!q8_hit) {
            q8_ptr = ctx.q8_add(src1, q8_size, stream);
        }
    }
''' + s[b:]
write(MV, s)

# --- 3. the norm kernel writes q8_1 too; n_terms = 0 is plain RMS_NORM -> MUL (no add) ---
N = "ggml/src/ggml-cuda/norm.cu"
s = read(N)
a = s.index("template <int block_size, int n_terms>\nstatic __global__ void add_rms_norm_mul_f32(")
b = s.index("template <int block_size>\nstatic __global__ void rms_norm_back_f32(")
s = s[:a] + r'''template <int block_size, int n_terms>
static __global__ void add_rms_norm_mul_f32(const add_terms t, float * sum, float * dst,
                                            const float * mul, const int ncols, const float eps, char * q8) {
    ggml_cuda_pdl_lc();
    const int     row = blockIdx.x;
    const int     tid = threadIdx.x;
    const int64_t off = (int64_t) row * ncols;
    dst += off;
    const float * x;   // what is normed: the sum written here, or (n_terms == 0) the input
    if constexpr (n_terms > 0) {
        sum += off;
        x = sum;
    } else {
        x = t.p[0] + off;
    }

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        float xi;
        if constexpr (n_terms > 0) {
            xi = t.p[0][off + col];
#pragma unroll
            for (int j = 1; j < n_terms; ++j) {
                xi = xi + t.p[j][off + col];
            }
            sum[col] = xi;
        } else {
            xi = x[col];
        }
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float v = scale * x[col] * mul[col];
        dst[col] = v;
        if (q8 != nullptr) {   // as quantize_q8_1: one warp = one block of 32 values (ncols % block_size == 0)
            float amax = fabsf(v);
            float s    = v;
            amax = warp_reduce_max<QK8_1>(amax);
            s    = warp_reduce_sum<QK8_1>(s);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
            block_q8_1 * y = (block_q8_1 *) q8 + (off + col) / QK8_1;
            y->qs[col % QK8_1] = q;
            if (col % QK8_1 == 0) {
                y->ds = make_half2(d, s);
            }
        }
    }
}

// a q8_1 slot in the conversion cache for the norm's output, when its readers will be mul_mat_vec_q (1-8 rows) and
// the kernel's warps cover whole 32-value blocks (width a multiple of 1024, no row padding needed)
static char * norm_q8_slot(ggml_backend_cuda_context & ctx, const ggml_tensor * out, int nrows, int ncols) {
    static const bool off = (getenv("GGML_CUDA_NO_Q8_FOLD") && atoi(getenv("GGML_CUDA_NO_Q8_FOLD"))) ||
                            (getenv("GGML_CUDA_NO_Q8_CACHE") && atoi(getenv("GGML_CUDA_NO_Q8_CACHE")));
    if (off || nrows > 8 || ncols % 1024 != 0 || ncols % MATRIX_ROW_PADDING != 0 || !ggml_is_contiguous(out)) {
        return nullptr;
    }
    return ctx.q8_add(out, (size_t) nrows * ncols / QK8_1 * sizeof(block_q8_1), ctx.stream());
}

static void launch_add_rms_norm_mul(ggml_backend_cuda_context & ctx, const add_terms & t, int n_terms, float * sum_d,
                                    float * dst_d, const float * w_d, int ncols, int nrows, float eps, char * q8) {
    const dim3 blocks_num(nrows, 1, 1);
    const int  bs = ncols < 1024 ? 256 : 1024;
    const ggml_cuda_kernel_launch_params p{blocks_num, dim3(bs, 1, 1), 32 * sizeof(float), ctx.stream()};
#define ADD_NORM_CASE(NT)                                                                                              \
    case NT:                                                                                                           \
        if (bs == 256) ggml_cuda_kernel_launch(add_rms_norm_mul_f32<256, NT>, p, t, sum_d, dst_d, w_d, ncols, eps, q8);  \
        else ggml_cuda_kernel_launch(add_rms_norm_mul_f32<1024, NT>, p, t, sum_d, dst_d, w_d, ncols, eps, q8);           \
        break;
    switch (n_terms) {
        ADD_NORM_CASE(0)
        ADD_NORM_CASE(2)
        ADD_NORM_CASE(3)
        ADD_NORM_CASE(4)
        ADD_NORM_CASE(5)
        default: GGML_ABORT("unsupported add chain");
    }
#undef ADD_NORM_CASE
}

// adds: the chain's ADD nodes in order (n_adds of them, 1..4); terms = adds[0]->src[0], then each add's src[1]
void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** adds, int n_adds,
                                     ggml_tensor * rms_norm, ggml_tensor * mul_tensor) {
    GGML_ASSERT(1 <= n_adds && n_adds <= 4);
    ggml_tensor * add = adds[n_adds - 1];
    add_terms t{};
    t.p[0] = (const float *) adds[0]->src[0]->data;
    for (int j = 0; j < n_adds; ++j) {
        t.p[j + 1] = (const float *) adds[j]->src[1]->data;
    }
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const int ncols = add->ne[0];
    const int nrows = ggml_nrows(add);
    char * q8 = norm_q8_slot(ctx, mul_tensor, nrows, ncols);
    launch_add_rms_norm_mul(ctx, t, n_adds + 1, (float *) add->data, (float *) mul_tensor->data,
                            (const float *) w->data, ncols, nrows, eps, q8);
}

// plain RMS_NORM -> MUL, taken over from rms_norm_f32<.., true> only when the q8_1 copy can be written (same loop and
// reduction order, so the same result); false = not taken
bool ggml_cuda_op_rms_norm_mul_fold_q8(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_tensor) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];
    if (x->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 || mul_tensor->type != GGML_TYPE_F32 ||
        w->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(mul_tensor) || !ggml_is_contiguous(w) ||
        !ggml_are_same_shape(x, mul_tensor) || w->ne[0] != x->ne[0] || ggml_nrows(w) != 1) {
        return false;
    }
    const int ncols = x->ne[0];
    const int nrows = ggml_nrows(x);
    char * q8 = norm_q8_slot(ctx, mul_tensor, nrows, ncols);
    if (q8 == nullptr) {
        return false;
    }
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    add_terms t{};
    t.p[0] = (const float *) x->data;
    launch_add_rms_norm_mul(ctx, t, 0, nullptr, (float *) mul_tensor->data, (const float *) w->data, ncols, nrows, eps, q8);
    return true;
}

''' + s[b:]
write(N, s)
edit("ggml/src/ggml-cuda/norm.cuh", '''void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** adds, int n_adds,
                                     ggml_tensor * rms_norm, ggml_tensor * mul_tensor);''', '''void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** adds, int n_adds,
                                     ggml_tensor * rms_norm, ggml_tensor * mul_tensor);

bool ggml_cuda_op_rms_norm_mul_fold_q8(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_tensor);''')

# --- 4. plain RMS_NORM -> MUL goes to the folding kernel when it can write the q8_1 copy ---
edit("ggml/src/ggml-cuda/ggml-cuda.cu", '''    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }''', '''    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        if (!ggml_cuda_op_rms_norm_mul_fold_q8(*cuda_ctx, node, cgraph->nodes[i + 1])) {
            ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        }
        return 1;
    }''')
print("patched")
