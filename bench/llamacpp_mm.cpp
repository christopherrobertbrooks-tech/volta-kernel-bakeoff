// llamacpp_mm: llama.cpp's own CUDA matrix-multiply kernels (ggml_mul_mat) timed at the bake-off's shapes on the V100,
// the third contender beside cuBLAS and ours. Same shapes, same token counts, same rule against the 6 MB L2: one graph
// holds R multiplies over R different copies of the weights (>= 96 MB in total), so each reads its weights from HBM.
// Activations are F32 (what llama.cpp feeds its kernels; it converts them to q8_1 itself). Weights are random values
// quantized with llama.cpp's own quantizer (IQ2 types with a flat importance matrix). Q2_0 is plain 2-bit with one
// scale per 64 -- the format GSQ's 2-bit models fit exactly (see the README).
//
//   g++ -O2 -std=c++17 -I<llama.cpp>/ggml/include bench/llamacpp_mm.cpp -L<build>/bin -lggml -lggml-base -lggml-cuda
//   CUDA_VISIBLE_DEVICES=<V100> ./llamacpp_mm > llama.csv
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Shape { const char* name; int N, K; };
static const Shape SHAPES[] = {
    {"flashnext_expert_gate_up", 640, 2560}, {"flashnext_expert_down", 2560, 640},
    {"flashnext_attn_q", 12288, 2560}, {"flashnext_attn_out", 2560, 6144},
    {"qwen36_expert_gate_up", 512, 2048}, {"qwen36_expert_down", 2048, 512},
    {"qwen36_attn_qkv", 8192, 2048}, {"qwen3_1p7b_mlp_up", 6144, 2048}, {"qwen3_1p7b_mlp_down", 2048, 6144},
};
static const int MS[] = {1, 2, 4, 8, 16, 64, 256, 512};
struct Type { ggml_type t; const char* name; };
static const Type TYPES[] = {
    {GGML_TYPE_Q4_0, "llama_q4_0"}, {GGML_TYPE_Q8_0, "llama_q8_0"}, {GGML_TYPE_Q2_K, "llama_q2_k"}, {GGML_TYPE_Q2_0, "llama_q2_0"},
    {GGML_TYPE_IQ2_XS, "llama_iq2_xs"}, {GGML_TYPE_IQ2_XXS, "llama_iq2_xxs"}, {GGML_TYPE_F16, "llama_f16"},
};

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 2; }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(be));
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.02f);
    printf("contender,shape,N,K,M,us,weight_GBps,tflops,copies\n");
    // LLAMA_MM_SHAPE / LLAMA_MM_TYPE / LLAMA_MM_M: one shape, type or token count only (for a profiler)
    const char* only_shape = getenv("LLAMA_MM_SHAPE"); const char* only_type = getenv("LLAMA_MM_TYPE");
    const int only_m = getenv("LLAMA_MM_M") ? atoi(getenv("LLAMA_MM_M")) : 0;
    for (const Type& ty : TYPES) {
        if (only_type && strcmp(ty.name, only_type)) continue;
        ggml_quantize_init(ty.t);
        for (const Shape& sh : SHAPES) {
            if (only_shape && strcmp(sh.name, only_shape)) continue;
            if (sh.K % ggml_blck_size(ty.t) != 0) {   // e.g. Q2_K / IQ2 pack 256 values per block; a 640-wide row can't
                fprintf(stderr, "%s: skipping %s (K %d is not a multiple of %d)\n", ty.name, sh.name, sh.K, (int) ggml_blck_size(ty.t));
                continue;
            }
            // quantize one random matrix with llama.cpp's own quantizer, then upload it R times
            std::vector<float> wf((size_t) sh.N * sh.K);
            for (auto& v : wf) v = nd(rng);
            const size_t row_bytes = ggml_row_size(ty.t, sh.K), wbytes = row_bytes * sh.N;
            std::vector<uint8_t> wq(wbytes);
            std::vector<float> imat(sh.K, 1.f);
            if (ty.t == GGML_TYPE_F16) {
                for (size_t i = 0; i < wf.size(); ++i) reinterpret_cast<ggml_fp16_t*>(wq.data())[i] = ggml_fp32_to_fp16(wf[i]);
            } else {
                ggml_quantize_chunk(ty.t, wf.data(), wq.data(), 0, sh.N, sh.K, ggml_quantize_requires_imatrix(ty.t) ? imat.data() : nullptr);
            }
            const int R = (int) std::min<size_t>(256, std::max<size_t>(1, (96u << 20) / wbytes + 1));
            for (int M : MS) {
                if (only_m && M != only_m) continue;
                ggml_init_params ip = {ggml_tensor_overhead() * (R + 8), nullptr, true};
                ggml_context* wctx = ggml_init(ip);
                std::vector<ggml_tensor*> W(R);
                for (int r = 0; r < R; ++r) W[r] = ggml_new_tensor_2d(wctx, ty.t, sh.K, sh.N);
                ggml_tensor* X = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, sh.K, M);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, be);
                for (int r = 0; r < R; ++r) ggml_backend_tensor_set(W[r], wq.data(), 0, wbytes);
                std::vector<float> xv((size_t) sh.K * M, 0.5f);
                ggml_backend_tensor_set(X, xv.data(), 0, xv.size() * sizeof(float));

                ggml_init_params gp = {ggml_tensor_overhead() * (R + 8) + ggml_graph_overhead_custom(4 * R + 16, false), nullptr, true};
                ggml_context* gctx = ggml_init(gp);
                ggml_cgraph* gf = ggml_new_graph_custom(gctx, 4 * R + 16, false);
                for (int r = 0; r < R; ++r) ggml_build_forward_expand(gf, ggml_mul_mat(gctx, W[r], X));
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "alloc failed\n"); return 2; }
                for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                std::vector<double> t;
                for (int rep = 0; rep < 5; ++rep) {
                    const auto a = std::chrono::steady_clock::now();
                    for (int i = 0; i < 4; ++i) ggml_backend_graph_compute(be, gf);
                    ggml_backend_synchronize(be);
                    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / (4.0 * R));
                }
                std::sort(t.begin(), t.end());
                const double us = t[2];
                printf("%s,%s,%d,%d,%d,%.2f,%.1f,%.3f,%d\n", ty.name, sh.name, sh.N, sh.K, M, us, wbytes / us / 1e3,
                       2.0 * M * sh.N * sh.K / us / 1e6, R);
                fflush(stdout);
                ggml_gallocr_free(ga);
                ggml_free(gctx);
                ggml_backend_buffer_free(buf);
                ggml_free(wctx);
            }
        }
    }
    ggml_backend_free(be);
    return 0;
}
