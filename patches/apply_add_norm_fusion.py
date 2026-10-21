# Issue #3, fusion 1: residual ADD -> RMS_NORM -> MUL (norm weight) as one CUDA kernel in a private llama.cpp
# (b11434 / 5e03bdd87). Run from the llama.cpp root. GGML_CUDA_NO_ADD_NORM_FUSION=1 switches it off at run time.
import sys
def edit(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(path, "w").write(s.replace(old, new))

edit("ggml/src/ggml-cuda/norm.cu", "template <int block_size>\nstatic __global__ void rms_norm_back_f32(", r'''// Residual add, then the next norm (ADD -> RMS_NORM -> MUL by the norm weight), in one kernel: s = a + b is written
// out (the residual stream still needs it) and dst = norm(s) * w. The same loop, reduction and expression order as
// rms_norm_f32<block_size, true>, so the result is bit-identical to the three separate kernels.
template <int block_size>
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
        const float xi = a[col] + b[col];
        sum[col] = xi;
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * sum[col] * mul[col];
    }
}

void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm,
                                     ggml_tensor * mul_tensor) {
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const ggml_tensor * w = mul_tensor->src[0] == rms_norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const int ncols = add->ne[0];
    const int nrows = ggml_nrows(add);
    const float * a_d = (const float *) add->src[0]->data;
    const float * b_d = (const float *) add->src[1]->data;
    float * sum_d = (float *) add->data;
    float * dst_d = (float *) mul_tensor->data;
    const float * w_d = (const float *) w->data;
    const dim3 blocks_num(nrows, 1, 1);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params p{blocks_num, block_dims, 32 * sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(add_rms_norm_mul_f32<256>, p, a_d, b_d, sum_d, dst_d, w_d, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params p{blocks_num, block_dims, 32 * sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(add_rms_norm_mul_f32<1024>, p, a_d, b_d, sum_d, dst_d, w_d, ncols, eps);
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(''')

edit("ggml/src/ggml-cuda/norm.cuh", "void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);",
     '''void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm,
                                     ggml_tensor * mul_tensor);''')

edit("ggml/src/ggml-cuda/ggml-cuda.cu", "    // multi-(add or mul)\n", '''    // residual add, then the next norm: ADD -> RMS_NORM -> MUL (norm weight) in one kernel; the ADD's result stays
    // an output (the residual stream reads it again). GGML_CUDA_NO_ADD_NORM_FUSION=1 switches this off.
    static const bool no_add_norm = getenv("GGML_CUDA_NO_ADD_NORM_FUSION") != nullptr && atoi(getenv("GGML_CUDA_NO_ADD_NORM_FUSION"));
    if (!no_add_norm && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
        const ggml_op ops[3]     = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
        const int     outputs[2] = { i, i + 2 };
        ggml_tensor * rms = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        if (ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 2) && rms->src[0] == node &&
            (mul->src[0] == rms || mul->src[1] == rms)) {
            const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
            const ggml_tensor * a = node->src[0];
            const ggml_tensor * b = node->src[1];
            const bool f32 = node->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                             rms->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32;
            const bool plain = ggml_are_same_shape(a, node) && ggml_are_same_shape(b, node) &&
                               ggml_are_same_shape(rms, node) && ggml_are_same_shape(mul, node) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(node) &&
                               ggml_is_contiguous(mul) && ggml_is_contiguous(w) && w->ne[0] == node->ne[0] &&
                               ggml_nrows(w) == 1;
            if (f32 && plain) {
                ggml_cuda_op_add_rms_norm_fused(*cuda_ctx, node, rms, mul);
                return 2;
            }
        }
    }

    // multi-(add or mul)
''')
print("patched")
