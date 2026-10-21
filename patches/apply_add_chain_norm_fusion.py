# Issue #3, fusion 1b (on top of apply_add_norm_fusion.py): a chain of residual adds -- e.g. routed experts + shared
# expert + residual, which llama.cpp already merges into one add kernel -- then RMS_NORM -> MUL, in one kernel.
# The terms are summed left to right ((a + b) + c), as llama.cpp's merged add does, so the result is bit-identical.
# GGML_CUDA_NO_ADD_CHAIN_NORM_FUSION=1 switches the chain part off (single adds still fuse).
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

N = "ggml/src/ggml-cuda/norm.cu"
edit(N, '''template <int block_size>
static __global__ void add_rms_norm_mul_f32(const float * a, const float * b, float * sum, float * dst,
                                            const float * mul, const int ncols, const float eps) {
    ggml_cuda_pdl_lc();
    const int     row = blockIdx.x;
    const int     tid = threadIdx.x;
    const int64_t off = (int64_t) row * ncols;
    a += off; b += off; sum += off; dst += off;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a[col] + b[col];''', '''struct add_terms { const float * p[5]; };  // the summed tensors, left to right

template <int block_size, int n_terms>
static __global__ void add_rms_norm_mul_f32(const add_terms t, float * sum, float * dst,
                                            const float * mul, const int ncols, const float eps) {
    ggml_cuda_pdl_lc();
    const int     row = blockIdx.x;
    const int     tid = threadIdx.x;
    const int64_t off = (int64_t) row * ncols;
    sum += off; dst += off;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        float xi = t.p[0][off + col];
#pragma unroll
        for (int j = 1; j < n_terms; ++j) {
            xi = xi + t.p[j][off + col];
        }''')
edit(N, '''void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm,
                                     ggml_tensor * mul_tensor) {
    float eps = 0.0f;''', '''// adds: the chain's ADD nodes in order (n_adds of them, 1..4); terms = adds[0]->src[0], then each add's src[1]
void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** adds, int n_adds,
                                     ggml_tensor * rms_norm, ggml_tensor * mul_tensor) {
    GGML_ASSERT(1 <= n_adds && n_adds <= 4);
    ggml_tensor * add = adds[n_adds - 1];
    add_terms t{};
    t.p[0] = (const float *) adds[0]->src[0]->data;
    for (int j = 0; j < n_adds; ++j) {
        t.p[j + 1] = (const float *) adds[j]->src[1]->data;
    }
    float eps = 0.0f;''')
edit(N, '''    const float * a_d = (const float *) add->src[0]->data;
    const float * b_d = (const float *) add->src[1]->data;
''', '')
edit(N, '''    const dim3 blocks_num(nrows, 1, 1);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params p{blocks_num, block_dims, 32 * sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(add_rms_norm_mul_f32<256>, p, a_d, b_d, sum_d, dst_d, w_d, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params p{blocks_num, block_dims, 32 * sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(add_rms_norm_mul_f32<1024>, p, a_d, b_d, sum_d, dst_d, w_d, ncols, eps);
    }''', '''    const dim3 blocks_num(nrows, 1, 1);
    const int  bs = ncols < 1024 ? 256 : 1024;
    const ggml_cuda_kernel_launch_params p{blocks_num, dim3(bs, 1, 1), 32 * sizeof(float), ctx.stream()};
#define ADD_NORM_CASE(NT)                                                                                         \\
    case NT:                                                                                                      \\
        if (bs == 256) ggml_cuda_kernel_launch(add_rms_norm_mul_f32<256, NT>, p, t, sum_d, dst_d, w_d, ncols, eps); \\
        else ggml_cuda_kernel_launch(add_rms_norm_mul_f32<1024, NT>, p, t, sum_d, dst_d, w_d, ncols, eps);          \\
        break;
    switch (n_adds + 1) {
        ADD_NORM_CASE(2)
        ADD_NORM_CASE(3)
        ADD_NORM_CASE(4)
        ADD_NORM_CASE(5)
        default: GGML_ABORT("unsupported add chain");
    }
#undef ADD_NORM_CASE''')
edit("ggml/src/ggml-cuda/norm.cuh", '''void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm,
                                     ggml_tensor * mul_tensor);''', '''void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** adds, int n_adds,
                                     ggml_tensor * rms_norm, ggml_tensor * mul_tensor);''')

C = "ggml/src/ggml-cuda/ggml-cuda.cu"
a = read(C)
start = a.index("    // residual add, then the next norm: ADD -> RMS_NORM -> MUL")
end = a.index("    // multi-(add or mul)\n")
a = a[:start] + '''    // residual add(s), then the next norm: ADD [-> ADD ...] -> RMS_NORM -> MUL (norm weight) in one kernel; the last
    // ADD's result stays an output (the residual stream reads it again). A chain of 2-4 adds is the merged add llama.cpp
    // would otherwise run (routed experts + shared expert + residual). GGML_CUDA_NO_ADD_NORM_FUSION=1 switches all of
    // this off; GGML_CUDA_NO_ADD_CHAIN_NORM_FUSION=1 only the chains.
    static const bool no_add_norm = getenv("GGML_CUDA_NO_ADD_NORM_FUSION") != nullptr && atoi(getenv("GGML_CUDA_NO_ADD_NORM_FUSION"));
    static const bool no_add_chain_norm = getenv("GGML_CUDA_NO_ADD_CHAIN_NORM_FUSION") != nullptr && atoi(getenv("GGML_CUDA_NO_ADD_CHAIN_NORM_FUSION"));
    if (!no_add_norm && node->op == GGML_OP_ADD) {
        auto plain_f32 = [&](const ggml_tensor * t) {
            return t->type == GGML_TYPE_F32 && ggml_are_same_shape(t, node) && ggml_is_contiguous(t);
        };
        int n_adds = 1;
        while (!no_add_chain_norm && n_adds < 4 && i + n_adds < cgraph->n_nodes &&
               cgraph->nodes[i + n_adds]->op == GGML_OP_ADD && cgraph->nodes[i + n_adds]->src[0] == cgraph->nodes[i + n_adds - 1]) {
            n_adds++;
        }
        // the chain must end in RMS_NORM -> MUL; if the longest one doesn't, shorter ones can't either
        if (i + n_adds + 1 < cgraph->n_nodes) {
            ggml_op ops[6];
            for (int j = 0; j < n_adds; ++j) ops[j] = GGML_OP_ADD;
            ops[n_adds] = GGML_OP_RMS_NORM;
            ops[n_adds + 1] = GGML_OP_MUL;
            const int     outputs[2] = { i + n_adds - 1, i + n_adds + 1 };
            ggml_tensor * last = cgraph->nodes[i + n_adds - 1];
            ggml_tensor * rms  = cgraph->nodes[i + n_adds];
            ggml_tensor * mul  = cgraph->nodes[i + n_adds + 1];
            if (ggml_can_fuse_subgraph(cgraph, i, n_adds + 2, ops, outputs, 2) && rms->src[0] == last &&
                (mul->src[0] == rms || mul->src[1] == rms)) {
                const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
                bool ok = plain_f32(node->src[0]) && plain_f32(rms) && plain_f32(mul) && w->type == GGML_TYPE_F32 &&
                          ggml_is_contiguous(w) && w->ne[0] == node->ne[0] && ggml_nrows(w) == 1;
                ggml_tensor * adds[4];
                for (int j = 0; j < n_adds; ++j) {
                    adds[j] = cgraph->nodes[i + j];
                    ok = ok && plain_f32(adds[j]) && plain_f32(adds[j]->src[1]);
                }
                if (ok) {
                    ggml_cuda_op_add_rms_norm_fused(*cuda_ctx, adds, n_adds, rms, mul);
                    return n_adds + 1;
                }
            }
        }
    }

''' + a[end:]
write(C, a)
print("patched")
