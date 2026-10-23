# Issue #3, fusion 3: a small f32/f16 multiply followed by a sigmoid (MUL_MAT [-> RESHAPE] -> UNARY sigmoid) in one
# mul_mat_vec_f call: the epilogue applies llama.cpp's own op_sigmoid formula (1 / (1 + expf(-x)), same -use_fast_math
# build) and writes into the sigmoid's tensor, so the result is bit-identical. In Qwen3.6-35B-A3B: ssm_beta (30 per
# token, through a reshape) and the shared expert's gate (40 per token). One token only (mmvf fuses at ncols_dst = 1).
# GGML_CUDA_NO_MMVF_SIGMOID_FUSION=1 switches it off.


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
edit(CC, '''    const ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_dst = nullptr;
};''', '''    const ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_dst = nullptr;
    bool sigmoid_out = false;   // private (#3): apply sigmoid to the output (mul_mat_vec_f only)
};''')
edit(CC, '''    float * shared_dst = nullptr;
    uint32_t shared_stride_col_dst = 0;
};''', '''    float * shared_dst = nullptr;
    uint32_t shared_stride_col_dst = 0;
    bool sigmoid_out = false;
};''')

MF = "ggml/src/ggml-cuda/mmvf.cu"
edit(MF, '''    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;''',
     '''    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr || fusion.sigmoid_out;''',
     count=2)
edit(MF, '''                default:
                    break;
            }
        }
    }

    dst[tid*stride_col_dst + row] = value;''', '''                default:
                    break;
            }
        }
        if (fusion.sigmoid_out) {   // op_sigmoid in unary.cu
            value = 1.0f / (1.0f + expf(-value));
        }
    }

    dst[tid*stride_col_dst + row] = value;''')
edit(MF, '''        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;''', '''        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
        fusion_local.sigmoid_out = fusion->sigmoid_out;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;''')

G = "ggml/src/ggml-cuda/ggml-cuda.cu"
edit(G, '''    bool fused_mul_mat_vec = false;
''', '''    // private (#3): small f32/f16 multiply -> sigmoid, MUL_MAT [-> RESHAPE] -> UNARY(sigmoid), in one mul_mat_vec_f call
    // that writes the sigmoid's tensor. GGML_CUDA_NO_MMVF_SIGMOID_FUSION=1 switches it off.
    static const bool no_mmvf_sigmoid = getenv("GGML_CUDA_NO_MMVF_SIGMOID_FUSION") != nullptr && atoi(getenv("GGML_CUDA_NO_MMVF_SIGMOID_FUSION"));
    if (!no_mmvf_sigmoid && node->op == GGML_OP_MUL_MAT && ggml_cuda_should_fuse_mul_mat_vec_f(node)) {
        int k = 1;   // index offset of the sigmoid
        if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE && cgraph->nodes[i + 1]->src[0] == node) {
            k = 2;
        }
        ggml_tensor * sig = i + k < cgraph->n_nodes ? cgraph->nodes[i + k] : nullptr;
        if (sig && sig->op == GGML_OP_UNARY && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID &&
            sig->src[0] == cgraph->nodes[i + k - 1]) {
            const ggml_op ops2[2]    = { GGML_OP_MUL_MAT, GGML_OP_UNARY };
            const ggml_op ops3[3]    = { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY };
            const int     outputs[1] = { i + k };
            if (ggml_can_fuse_subgraph(cgraph, i, k + 1, k == 1 ? ops2 : ops3, outputs, 1) && sig->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(sig) && ggml_is_contiguous(node) && ggml_nelements(sig) == ggml_nelements(node) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, k + 1, outputs, 1)) {   // every block reads all of src1
                ggml_tensor out = *node;   // the multiply's own shape and strides, writing into the sigmoid's memory
                out.data = sig->data;
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.sigmoid_out = true;
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, node->src[0], node->src[1], nullptr, &out, &fusion_data);
                return k;
            }
        }
    }

    bool fused_mul_mat_vec = false;
''')
print("patched")
