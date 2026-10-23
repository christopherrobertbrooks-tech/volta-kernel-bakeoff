// q8cache_test: checks for the private llama.cpp's q8_1 conversion cache (issue #11). It writes every output of three
// small graphs to a file; run it with the cache on and with GGML_CUDA_NO_Q8_CACHE=1 and compare the files byte for byte
// (bench/q8cache-test.sh does that, also with GGML_CUDA_GRAPH_OPT=1, llama.cpp's concurrent streams).
//   A  in-place op on a cached activation: y1 = W1 x; x' = scale_inplace(x, 3); y2 = W2 x' -- y2 must not reuse x's
//      conversion (x' is the same memory with different values).
//   B  three multiplies reading one normed activation (branches the concurrent-stream optimiser can spread over
//      streams), summed: y = W1 n + W2 n + W3 n, with n = rms_norm(x) * w.
//   C  graph B run twice back to back, first on input 1 then on input 2, queued without a sync in between; the result
//      kept is the second run's. A cache that reused the first run's conversion would give input 1's answer here.
// Weights: Q4_0, 2048 x 2048, random, quantized with llama.cpp's own quantizer; one token (the mul_mat_vec_q path).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: q8cache_test OUT_FILE\n"); return 2; }
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 2; }
    const int K = 2048, N = 2048;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    ggml_quantize_init(GGML_TYPE_Q4_0);
    const size_t wbytes = ggml_row_size(GGML_TYPE_Q4_0, K) * N;

    ggml_init_params ip = {ggml_tensor_overhead() * 16, nullptr, true};
    ggml_context * wctx = ggml_init(ip);
    ggml_tensor * W[3];
    for (auto & w : W) w = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * nw = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, K);   // the norm weight
    ggml_tensor * xa = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * xb = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, 1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, be);
    std::vector<float> wf((size_t) N * K);
    std::vector<uint8_t> wq(wbytes);
    for (auto & w : W) {
        for (auto & v : wf) v = nd(rng);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, N, K, nullptr);
        ggml_backend_tensor_set(w, wq.data(), 0, wbytes);
    }
    std::vector<float> nwv(K);
    for (auto & v : nwv) v = 1.f + 0.1f * xd(rng);
    ggml_backend_tensor_set(nw, nwv.data(), 0, K * sizeof(float));
    auto rand_x = [&](std::vector<float> & v) { v.resize(K); for (auto & e : v) e = xd(rng); };
    std::vector<float> va, vb1, vb2;
    rand_x(va); rand_x(vb1); rand_x(vb2);

    FILE * out = fopen(argv[1], "wb");
    auto dump = [&](ggml_tensor * t) {
        std::vector<float> y(ggml_nelements(t));
        ggml_backend_tensor_get(t, y.data(), 0, y.size() * sizeof(float));
        fwrite(y.data(), sizeof(float), y.size(), out);
    };

    // A: in-place op on a cached activation
    {
        ggml_init_params gp = {ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
        ggml_context * g = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(g);
        ggml_tensor * y1 = ggml_mul_mat(g, W[0], xa);
        ggml_set_output(y1);
        ggml_build_forward_expand(gf, y1);
        ggml_tensor * xs = ggml_scale_inplace(g, xa, 3.0f);
        ggml_tensor * y2 = ggml_mul_mat(g, W[1], xs);
        ggml_set_output(y2);
        ggml_build_forward_expand(gf, y2);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "alloc A failed\n"); return 2; }
        ggml_backend_tensor_set(xa, va.data(), 0, K * sizeof(float));
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        dump(y1); dump(y2);
        ggml_gallocr_free(ga); ggml_free(g);
    }
    // B and C: three multiplies of one normed activation, summed; once, then twice back to back
    {
        ggml_init_params gp = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
        ggml_context * g = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(g);
        ggml_tensor * n = ggml_mul(g, ggml_rms_norm(g, xb, 1e-6f), nw);
        ggml_tensor * y = ggml_add(g, ggml_add(g, ggml_mul_mat(g, W[0], n), ggml_mul_mat(g, W[1], n)), ggml_mul_mat(g, W[2], n));
        ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "alloc B failed\n"); return 2; }
        ggml_backend_tensor_set(xb, vb1.data(), 0, K * sizeof(float));
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        dump(y);                                                   // B
        ggml_backend_tensor_set_async(be, xb, vb1.data(), 0, K * sizeof(float));
        ggml_backend_graph_compute_async(be, gf);                 // C: two runs queued, no sync between them
        ggml_backend_tensor_set_async(be, xb, vb2.data(), 0, K * sizeof(float));
        ggml_backend_graph_compute_async(be, gf);
        ggml_backend_synchronize(be);
        dump(y);                                                   // C: input 2's result
        ggml_gallocr_free(ga); ggml_free(g);
    }
    fclose(out);
    ggml_backend_buffer_free(buf);
    ggml_free(wctx);
    ggml_backend_free(be);
    printf("q8cache_test: wrote %s\n", argv[1]);
    return 0;
}
