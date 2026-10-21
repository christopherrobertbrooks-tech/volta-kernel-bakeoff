// volta-kernel-bakeoff, stages 1-2 (stage 2: src/stage2.cuh): quantized matrix multiply on a Tesla V100 (sm_70), raced against cuBLAS.
//
//   y[M][N] = x[M][K] . W[N][K]^T        x: fp16 activations (M tokens), W: weights, y: fp32
//
// Weight formats (ours, simple on purpose; the values match GGUF's Q4_0 / Q8_0):
//   q4: two 4-bit codes per byte (value k even in the low nibble, k+1 in the high one), value = (code - 8) * scale
//   q8: one int8 per value, value = code * scale
//   one fp16 scale per 32 consecutive values of a row
//
// Contenders:
//   cublas_fp16   cuBLAS GemmEx on the dequantized fp16 weights (tensor cores; the weights are 4x / 2x bigger)
//   dq4+cublas    dequantize q4 to fp16 every call, then cuBLAS ("unpack, then multiply")
//   fused_q4/q8   our kernel: each warp owns one output row, unpacks in registers, fp32 sums (decode: M <= 8)
//   moe_*         the MoE decode case: E experts, one token each: E cuBLAS calls / cuBLAS batched / our one launch
//
// Timing: the V100's L2 is 6 MB, smaller than one big weight matrix but larger than one MoE expert, so every
// contender cycles through enough copies of its weights (>= 96 MB) that each call reads them from HBM, as it does in
// a real model. Median of 5 batches of 50 calls, after 20 warm-up calls.
//
// Correctness (--check): every contender against a CPU double-precision reference of the same quantized weights, on
// sampled outputs. BAKEOFF_SABOTAGE=1 breaks every packed format so the check must fail (proves it can): q4 swaps the two
// nibbles of every byte (all q4 kernels, dq4+cublas, tc_q4, the MoE one-launch), q8 swaps every pair of int8 codes, q2
// swaps the two codes of every pair; cuBLAS fp16 has no packing and passes as the control.
//
//   nvcc -O3 -arch=sm_70 -lcublas -o bakeoff src/bakeoff.cu
//   ./bakeoff --check            correctness, exit 1 on a failure
//   ./bakeoff > results.csv      the sweep

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int) s_, __FILE__, __LINE__); exit(2); } } while (0)

static const int GROUP = 32;

// ---- kernels --------------------------------------------------------------------------------------------------

// q4 -> fp16, 8 values per thread
__global__ void dequant_q4(const uint32_t* __restrict__ q, const half* __restrict__ s, half* __restrict__ w,
                           int N, int K) {
    const size_t c = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // chunk of 8 values
    const size_t chunks = (size_t) N * K / 8;
    if (c >= chunks) return;
    const uint32_t v = q[c];
    const float sc = __half2float(s[c / 4]);                           // 4 chunks per group of 32
    half out[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int b = (v >> (8 * j)) & 0xFF;
        out[2 * j] = __float2half(((b & 0xF) - 8) * sc);
        out[2 * j + 1] = __float2half(((b >> 4) - 8) * sc);
    }
    reinterpret_cast<uint4*>(w)[c] = *reinterpret_cast<uint4*>(out);
}

// Fused q4 matrix-vector product for M tokens: one warp per output row, each lane 8 values per step.
template <int M>
__global__ void fused_q4(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                         float* __restrict__ y, int N, int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;   // row is per warp (blockDim is a multiple of 32), so a warp leaves whole and the
                            // full-mask shuffles below always see all 32 lanes
    const uint32_t* qr = q + (size_t) row * (K / 8);
    const half* sr = s + (size_t) row * (K / GROUP);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        const uint32_t v = __ldg(qr + c);
        const float sc = __half2float(__ldg(sr + (c >> 2)));
        float w[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = (v >> (8 * j)) & 0xFF;
            w[2 * j] = (float) ((b & 0xF) - 8);
            w[2 * j + 1] = (float) ((b >> 4) - 8);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t) m * K) + c);
            const half2* xh = reinterpret_cast<const half2*>(&xv);
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 xf = __half22float2(xh[j]);
                a += w[2 * j] * xf.x + w[2 * j + 1] * xf.y;
            }
            acc[m] += a * sc;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) y[(size_t) m * N + row] = a;
    }
}

// Fused q8, same shape of work: each lane 8 int8 values (uint2) per step.
template <int M>
__global__ void fused_q8(const uint2* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                         float* __restrict__ y, int N, int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;   // row is per warp (blockDim is a multiple of 32), so a warp leaves whole and the
                            // full-mask shuffles below always see all 32 lanes
    const uint2* qr = q + (size_t) row * (K / 8);
    const half* sr = s + (size_t) row * (K / GROUP);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        const uint2 v = __ldg(qr + c);
        const float sc = __half2float(__ldg(sr + (c >> 2)));
        const int8_t* b = reinterpret_cast<const int8_t*>(&v);
        float w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = (float) b[j];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t) m * K) + c);
            const half2* xh = reinterpret_cast<const half2*>(&xv);
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 xf = __half22float2(xh[j]);
                a += w[2 * j] * xf.x + w[2 * j + 1] * xf.y;
            }
            acc[m] += a * sc;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) y[(size_t) m * N + row] = a;
    }
}

// MoE decode: E experts in one launch (blockIdx.y = expert), one token each, q4 weights.
struct MoePtrs { const uint32_t* q[16]; const half* s[16]; const half* x[16]; float* y[16]; };
__global__ void moe_fused_q4(MoePtrs p, int N, int K) {
    const int e = blockIdx.y;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;   // row is per warp (blockDim is a multiple of 32), so a warp leaves whole and the
                            // full-mask shuffles below always see all 32 lanes
    const uint32_t* qr = p.q[e] + (size_t) row * (K / 8);
    const half* sr = p.s[e] + (size_t) row * (K / GROUP);
    const half* x = p.x[e];
    float acc = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        const uint32_t v = __ldg(qr + c);
        const float sc = __half2float(__ldg(sr + (c >> 2)));
        const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x) + c);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = (v >> (8 * j)) & 0xFF;
            const float2 xf = __half22float2(xh[j]);
            a += ((b & 0xF) - 8) * xf.x + ((b >> 4) - 8) * xf.y;
        }
        acc += a * sc;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) p.y[e][row] = acc;
}

#include "stage2.cuh"
#include "stage3_lut.cuh"
#include "stage4_tc.cuh"
#include "stage5_gap8.cuh"

// ---- host side ------------------------------------------------------------------------------------------------

struct HostQ {             // one quantized matrix, kept on the host for the reference
    int N, K;
    std::vector<int8_t> code;   // [N][K], q4: -8..7, q8: -127..127
    std::vector<half> scale;    // [N][K/32]
};

static HostQ make_weights(int N, int K, int bits, std::mt19937& rng) {
    HostQ h{N, K, std::vector<int8_t>((size_t) N * K), std::vector<half>((size_t) N * K / GROUP)};
    std::uniform_int_distribution<int> c4(-8, 7), c8(-127, 127);
    std::uniform_real_distribution<float> sd(0.004f, 0.02f);
    for (auto& c : h.code) c = (int8_t) (bits == 4 ? c4(rng) : c8(rng));
    for (auto& s : h.scale) s = __float2half(sd(rng) * (bits == 4 ? 1.f : 0.06f));
    return h;
}

static std::vector<uint32_t> pack_q4(const HostQ& h, bool sabotage) {
    std::vector<uint32_t> out((size_t) h.N * h.K / 8, 0);
    for (size_t i = 0; i < h.code.size(); i += 2) {
        uint32_t lo = (uint32_t) (h.code[i] + 8), hi = (uint32_t) (h.code[i + 1] + 8);
        if (sabotage) std::swap(lo, hi);
        const uint32_t byte = lo | (hi << 4);
        out[i / 8] |= byte << (8 * ((i / 2) % 4));
    }
    return out;
}

static double ref_value(const HostQ& h, const std::vector<half>& x, int m, int n) {
    double a = 0;
    for (int k = 0; k < h.K; ++k)
        a += (double) __half2float(x[(size_t) m * h.K + k]) * h.code[(size_t) n * h.K + k] *
             (double) __half2float(h.scale[(size_t) n * h.K / GROUP + k / GROUP]);
    return a;
}

static double ref_scale(const HostQ& h, const std::vector<half>& x, int m, int n) {   // sum of |x*w|, the error yardstick
    double a = 0;
    for (int k = 0; k < h.K; ++k)
        a += std::fabs((double) __half2float(x[(size_t) m * h.K + k]) * h.code[(size_t) n * h.K + k] *
                       (double) __half2float(h.scale[(size_t) n * h.K / GROUP + k / GROUP]));
    return a;
}

template <typename T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }
template <typename T> static T* dupload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size()); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }

static std::vector<half> dequant_host(const HostQ& h) {
    std::vector<half> w((size_t) h.N * h.K);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2half(h.code[i] * __half2float(h.scale[i / GROUP]));
    return w;
}

// y[M][N] = x[M][K] . W[N][K]^T with W fp16 row-major: C(N x M, col-major) = op_T(W as K x N) * X(K x M)
static void cublas_gemm(cublasHandle_t hb, const half* w, const half* x, float* y, int N, int K, int M) {
    const float one = 1.f, zero = 0.f;
    CB(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, CUDA_R_16F, K, x, CUDA_R_16F, K, &zero, y,
                    CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <int M>
static void launch_fused_q4(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    fused_q4<M><<<(N + 3) / 4, 128>>>(q, s, x, y, N, K);
}
template <int M>
static void launch_fused_q8(const uint2* q, const half* s, const half* x, float* y, int N, int K) {
    fused_q8<M><<<(N + 3) / 4, 128>>>(q, s, x, y, N, K);
}
static bool fused_q4_any(int M, const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    switch (M) {
        case 1: launch_fused_q4<1>(q, s, x, y, N, K); return true;
        case 2: launch_fused_q4<2>(q, s, x, y, N, K); return true;
        case 4: launch_fused_q4<4>(q, s, x, y, N, K); return true;
        case 8: launch_fused_q4<8>(q, s, x, y, N, K); return true;
        default: return false;
    }
}
static bool fused_q8_any(int M, const uint2* q, const half* s, const half* x, float* y, int N, int K) {
    switch (M) {
        case 1: launch_fused_q8<1>(q, s, x, y, N, K); return true;
        case 2: launch_fused_q8<2>(q, s, x, y, N, K); return true;
        case 4: launch_fused_q8<4>(q, s, x, y, N, K); return true;
        case 8: launch_fused_q8<8>(q, s, x, y, N, K); return true;
        default: return false;
    }
}

// Median time per call (microseconds) of fn(i), cycling i over the weight copies.
template <typename F> static double time_us(F fn) {
    for (int i = 0; i < 20; ++i) fn(i);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<double> t;
    for (int rep = 0, i = 0; rep < 5; ++rep) {
        CK(cudaEventRecord(a));
        for (int k = 0; k < 50; ++k, ++i) fn(i);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms * 1000.0 / 50);
    }
    CK(cudaGetLastError());
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return t[2];
}

struct Shape { const char* name; int N, K; };
static const Shape SHAPES[] = {
    {"flashnext_expert_gate_up", 640, 2560},    // Strata Coder (Qwen3.8-Flash-Next): 256 experts, 10 per token
    {"flashnext_expert_down", 2560, 640},
    {"flashnext_attn_q", 12288, 2560},
    {"flashnext_attn_out", 2560, 6144},
    {"qwen36_expert_gate_up", 512, 2048},      // Qwen3.6-35B-A3B / Tiel-Coder: 256 experts, 8 per token
    {"qwen36_expert_down", 2048, 512},
    {"qwen36_attn_qkv", 8192, 2048},
    {"qwen3_1p7b_mlp_up", 6144, 2048},         // Qwen3-1.7B, a dense model
    {"qwen3_1p7b_mlp_down", 2048, 6144},
};
static const int MS[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

static int copies_for(size_t bytes) { return (int) std::max<size_t>(1, (96u << 20) / std::max<size_t>(bytes, 1) + 1); }

// ---- correctness ----------------------------------------------------------------------------------------------

// The MoE contenders: E experts, each with its own weights and its own token, in one launch (ours) and in one cuBLAS
// batched call, against the CPU reference per expert. Different weights per expert also catch a wrong expert index.
static int run_check_moe(cublasHandle_t hb, bool sabotage, std::mt19937& rng) {
    int failures = 0;
    for (const Shape& sh : SHAPES) {
        if (!strstr(sh.name, "expert")) continue;
        const int E = strstr(sh.name, "flashnext") ? 10 : 8;
        std::vector<HostQ> h; std::vector<std::vector<half>> x(E);
        std::vector<uint32_t*> dq(E); std::vector<const half*> ds(E); std::vector<half*> dw(E), dxs(E);
        std::uniform_real_distribution<float> xd(-1.f, 1.f);
        for (int e = 0; e < E; ++e) {
            h.push_back(make_weights(sh.N, sh.K, 4, rng));
            dq[e] = dupload(pack_q4(h[e], sabotage)); ds[e] = dupload(h[e].scale); dw[e] = dupload(dequant_host(h[e]));
            x[e].resize(sh.K); for (auto& v : x[e]) v = __float2half(xd(rng));
            dxs[e] = dupload(x[e]);
        }
        float* dy = dalloc<float>((size_t) E * sh.N);
        const float one = 1.f, zero = 0.f;
        std::vector<const void*> a(E), b(E); std::vector<void*> c(E);
        for (int e = 0; e < E; ++e) { a[e] = dw[e]; b[e] = dxs[e]; c[e] = dy + (size_t) e * sh.N; }
        const void** A = (const void**) dupload(a); const void** B = (const void**) dupload(b); void** C = (void**) dupload(c);
        for (int who = 0; who < 2; ++who) {
            CK(cudaMemset(dy, 0, (size_t) E * sh.N * sizeof(float)));
            if (who == 0) {
                MoePtrs p{};
                for (int e = 0; e < E; ++e) { p.q[e] = dq[e]; p.s[e] = ds[e]; p.x[e] = dxs[e]; p.y[e] = dy + (size_t) e * sh.N; }
                moe_fused_q4<<<dim3((sh.N + 3) / 4, E), 128>>>(p, sh.N, sh.K);
            } else {
                CB(cublasGemmBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, 1, sh.K, &one, A, CUDA_R_16F, sh.K, B, CUDA_R_16F,
                                       sh.K, &zero, C, CUDA_R_32F, sh.N, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            CK(cudaDeviceSynchronize());
            std::vector<float> y((size_t) E * sh.N);
            CK(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::uniform_int_distribution<int> nd(0, sh.N - 1);
            double worst = 0; int we = 0, wn = 0;
            for (int e = 0; e < E; ++e)
                for (int t = 0; t < 34; ++t) {
                    const int n = t == 0 ? 0 : t == 1 ? sh.N - 1 : nd(rng);
                    const double err = std::fabs(y[(size_t) e * sh.N + n] - ref_value(h[e], x[e], 0, n)) / (ref_scale(h[e], x[e], 0, n) + 1e-12);
                    if (err > worst) { worst = err; we = e; wn = n; }
                }
            const char* name = who == 0 ? "moe_fused_q4_one_launch" : "moe_cublas_batched";
            const double tol = who == 0 ? 1e-4 : 2e-3;   // cuBLAS uses the fp16-rounded weights
            const bool ok = worst <= tol; failures += !ok;
            printf("check,%s,%s_x%d,1,%.3g,%d,%d,%s\n", name, sh.name, E, worst, we, wn, ok ? "ok" : "FAIL");
        }
        for (int e = 0; e < E; ++e) { CK(cudaFree(dq[e])); CK(cudaFree((void*) ds[e])); CK(cudaFree(dw[e])); CK(cudaFree(dxs[e])); }
        CK(cudaFree((void*) A)); CK(cudaFree((void*) B)); CK(cudaFree(C)); CK(cudaFree(dy));
    }
    return failures;
}

// Qwen3-1.7B's MLP gate + up + SwiGLU (gate and up 6144x2048, stacked): the stacked multiply against the reference
// (fp16 class, 2e-3), and the SwiGLU output against silu(gate) * up computed from that multiply's own result
static int run_check_glu(bool sabotage, std::mt19937& rng) {
    const int N = 6144, K = 2048;
    HostQ h = make_weights(2 * N, K, 4, rng);
    uint32_t* dq = dupload(pack_q4(h, sabotage)); half* ds = dupload(h.scale);
    int failures = 0;
    for (int M : {1, 2, 4, 8}) {
        std::uniform_real_distribution<float> xd(-1.f, 1.f);
        std::vector<half> x((size_t) M * K); for (auto& v : x) v = __float2half(xd(rng));
        std::vector<float> xf(x.size()); for (size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);
        float* dxf = dupload(xf); half* dxh = dalloc<half>(x.size());
        float* dys = dalloc<float>((size_t) M * 2 * N); float* dout = dalloc<float>((size_t) M * N);
        gate_up_glu_fp32in(M, dq, ds, dxf, dxh, dys, dout, N, K);
        CK(cudaDeviceSynchronize());
        std::vector<float> ys((size_t) M * 2 * N), out((size_t) M * N);
        CK(cudaMemcpy(ys.data(), dys, ys.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
        std::uniform_int_distribution<int> nd(0, N - 1), md(0, M - 1);
        double wmm = 0, wglu = 0;
        for (int t = 0; t < 256; ++t) {
            const int m = md(rng), n = nd(rng);
            for (int part = 0; part < 2; ++part) {
                const int row = n + part * N;
                wmm = std::max(wmm, std::fabs(ys[(size_t) m * 2 * N + row] - ref_value(h, x, m, row)) / (ref_scale(h, x, m, row) + 1e-12));
            }
            const double g = ys[(size_t) m * 2 * N + n], u = ys[(size_t) m * 2 * N + N + n];
            const double want = g / (1.0 + std::exp(-g)) * u;
            wglu = std::max(wglu, std::fabs(out[(size_t) m * N + n] - want) / (std::fabs(want) + 1e-6));
        }
        const bool ok = wmm <= 2e-3 && wglu <= 1e-5;
        failures += !ok;
        printf("check,gate_up_glu_fp32in,qwen3_1p7b_mlp_gate_up,%d,%.3g,%.3g,0,%s\n", M, wmm, wglu, ok ? "ok" : "FAIL");
        CK(cudaFree(dxf)); CK(cudaFree(dxh)); CK(cudaFree(dys)); CK(cudaFree(dout));
    }
    CK(cudaFree(dq)); CK(cudaFree(ds));
    return failures;
}

static int run_check(cublasHandle_t hb, bool sabotage) {
    std::mt19937 rng(1234);
    int failures = 0;
    printf("check,contender,shape,M,max_err_rel,worst_m,worst_n,verdict\n");
    for (const Shape& sh : SHAPES) {
        for (int bits : {4, 8}) {
            HostQ h = make_weights(sh.N, sh.K, bits, rng);
            const half* ds = dupload(h.scale);
            uint32_t* dq4 = nullptr; uint2* dq8 = nullptr;
            if (bits == 4) dq4 = dupload(pack_q4(h, sabotage));
            else {
                std::vector<int8_t> c8 = h.code;    // int8 codes go in as they are; sabotage swaps every pair
                if (sabotage) for (size_t i = 0; i + 1 < c8.size(); i += 2) std::swap(c8[i], c8[i + 1]);
                std::vector<uint2> p((size_t) sh.N * sh.K / 8);
                memcpy(p.data(), c8.data(), c8.size());
                dq8 = dupload(p);
            }
            half* dw = dupload(dequant_host(h));    // the fp16 baseline's weights (rounded once to fp16)
            // activations: x in [-1, 1] at every M (512 = the largest batch the sweep times; 64-512 cover every
            // tc_q4_v2 tile / split path); at 1-8 tokens also "bigx" -- 64 neighbouring channels at +-20000..60000,
            // random signs, as a real down_proj input can hold (fp16-fixer saw 43K-80K): it fits fp16 but overflows
            // short fp16 sums -- and "tinyx", x in [-1e-3, 1e-3], where fp16 products lose precision first
            struct XCase { int M, xk; };
            std::vector<XCase> xcases;
            for (int M : {1, 2, 4, 8, 64, 128, 256, 512}) xcases.push_back({M, 0});
            for (int M : {1, 2, 4, 8}) { xcases.push_back({M, 1}); xcases.push_back({M, 2}); }
            for (const XCase& xc : xcases) {
                const int M = xc.M;
                const std::string label = std::string(sh.name) + (xc.xk == 1 ? "+bigx" : xc.xk == 2 ? "+tinyx" : "");
                std::uniform_real_distribution<float> xd(-1.f, 1.f);
                std::vector<half> x((size_t) M * sh.K);
                for (auto& v : x) v = __float2half(xd(rng) * (xc.xk == 2 ? 1e-3f : 1.f));
                if (xc.xk == 1) {
                    std::uniform_real_distribution<float> big(20000.f, 60000.f);
                    const int k0 = sh.K / 3 / 64 * 64;
                    for (int m = 0; m < M; ++m)
                        for (int k = k0; k < std::min(sh.K, k0 + 64); ++k)
                            x[(size_t) m * sh.K + k] = __float2half((rng() & 1 ? 1.f : -1.f) * big(rng));
                }
                const half* dx = dupload(x);
                std::vector<float> xf(x.size());              // the same values in fp32, for the fp32-input paths
                for (size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);
                const float* dxf = dupload(xf);
                half* dxh = dalloc<half>(x.size());
                float* dy = dalloc<float>((size_t) M * sh.N);
                std::vector<std::pair<std::string, int>> runs;
                if (bits == 4) {
                    if (M <= 8) {
                        runs.push_back({"fused_q4", 0}); runs.push_back({"fused_q4_v2", 4}); runs.push_back({"fused_q4_v2w", 5});
                        runs.push_back({"fused_q4_v3", 8}); runs.push_back({"fused_q4_v3f", 9}); runs.push_back({"cvt+fused_q4_v3", 10}); runs.push_back({"fused_q4_v3_fp32in", 11});
                        for (int c = 0; c < fq3::NCFG; ++c) {
                            runs.push_back({std::string("fused_q4_v3[") + fq3::CFGS[c].name + "]", 200 + c});
                            runs.push_back({std::string("fused_q4_v3f[") + fq3::CFGS[c].name + "]", 300 + c});
                        }
                    }
                    if (tc_q4_ok(M, sh.N)) runs.push_back({"tc_q4", 6});
                    if (tc_q4_v2_ok(M, sh.N, sh.K)) {
                        runs.push_back({"tc_q4_v2", 7});
                        for (int c = 0; c < tc2::NCFG; ++c)  // every tile shape and K split the tuning sweep can time
                            for (int si = 0; si <= 4 && (sh.K / tc2::BK) % (1 << si) == 0; ++si)
                                if (tc2::CFGS[c].fits(M, sh.N))
                                    runs.push_back({std::string("tc_q4_v2[") + tc2::CFGS[c].name + "/s" + std::to_string(1 << si) + "]", 100 + 8 * c + si});
                    }
                    runs.push_back({"dq4+cublas", 1});
                }
                else if (M <= 8) runs.push_back({"fused_q8", 2});
                runs.push_back({"cublas_fp16", 3});
                half* dwq = (bits == 4) ? dalloc<half>((size_t) sh.N * sh.K) : nullptr;
                for (auto& r : runs) {
                    CK(cudaMemset(dy, 0, (size_t) M * sh.N * sizeof(float)));
                    if (r.second == 0) fused_q4_any(M, dq4, ds, dx, dy, sh.N, sh.K);
                    else if (r.second == 1) {
                        dequant_q4<<<(int) (((size_t) sh.N * sh.K / 8 + 255) / 256), 256>>>(dq4, ds, dwq, sh.N, sh.K);
                        cublas_gemm(hb, dwq, dx, dy, sh.N, sh.K, M);
                    } else if (r.second == 2) fused_q8_any(M, dq8, ds, dx, dy, sh.N, sh.K);
                    else if (r.second == 4) fused_q4_v2_any(M, false, dq4, ds, dx, dy, sh.N, sh.K);
                    else if (r.second == 5) fused_q4_v2_any(M, true, dq4, ds, dx, dy, sh.N, sh.K);
                    else if (r.second == 6) launch_tc_q4(dq4, ds, dx, dy, sh.N, sh.K, M);
                    else if (r.second == 7) launch_tc_q4_v2(dq4, ds, dx, dy, sh.N, sh.K, M);
                    else if (r.second == 8) fused_q4_v3_any(M, dq4, ds, dx, dy, sh.N, sh.K);
                    else if (r.second == 9) fused_q4_v3_any(M, dq4, ds, dxf, dy, sh.N, sh.K);
                    else if (r.second == 10) { fq3::launch_cvt(dxf, dxh, M * sh.K); fused_q4_v3_any(M, dq4, ds, (const half*) dxh, dy, sh.N, sh.K); }
                    else if (r.second == 11) fused_q4_v3_fp32in(M, dq4, ds, dxf, dxh, dy, sh.N, sh.K);
                    else if (r.second >= 300) fq3::run(r.second - 300, M, dq4, ds, dxf, dy, sh.N, sh.K);
                    else if (r.second >= 200) fq3::run(r.second - 200, M, dq4, ds, dx, dy, sh.N, sh.K);
                    else if (r.second >= 100) tc2::CFGS[(r.second - 100) / 8].run(dq4, ds, dx, dy, sh.N, sh.K, M, 1 << ((r.second - 100) % 8));
                    else cublas_gemm(hb, dw, dx, dy, sh.N, sh.K, M);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> y((size_t) M * sh.N);
                    CK(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    // 256 sampled outputs, always including the first and last row of every token
                    std::uniform_int_distribution<int> nm(0, sh.N - 1), mm(0, M - 1);
                    double worst = 0; int wm = 0, wn = 0;
                    for (int t = 0; t < 256 + 2 * M; ++t) {
                        int m, n;
                        if (t < 2 * M) { m = t / 2; n = (t % 2) ? sh.N - 1 : 0; } else { m = mm(rng); n = nm(rng); }
                        const double err = std::fabs(y[(size_t) m * sh.N + n] - ref_value(h, x, m, n)) /
                                           (ref_scale(h, x, m, n) + 1e-12);
                        if (err > worst) { worst = err; wm = m; wn = n; }
                    }
                    // fp32 sums of fp16 inputs: ~1e-6 of the magnitude; the fp16 baseline rounds the weights once
                    // more (2^-11), and fused_q4_v3 rounds its products and short partial sums to fp16: allow 2e-3
                    // for those, 1e-4 for the kernels that use the exact codes with fp32 sums
                    const double tol = (r.first == "cublas_fp16" || r.first == "dq4+cublas" || r.first.rfind("tc_q4", 0) == 0 ||
                                        r.first.rfind("fused_q4_v3", 0) == 0 || r.first.rfind("cvt+fused_q4_v3", 0) == 0) ? 2e-3 : 1e-4;
                    const bool ok = worst <= tol;
                    failures += !ok;
                    printf("check,%s,%s,%d,%.3g,%d,%d,%s\n", r.first.c_str(), label.c_str(), M, worst, wm, wn,
                           ok ? "ok" : "FAIL");
                }
                CK(cudaFree((void*) dx)); CK(cudaFree(dy)); CK(cudaFree((void*) dxf)); CK(cudaFree(dxh));
                if (dwq) CK(cudaFree(dwq));
            }
            if (dq4) CK(cudaFree(dq4));
            if (dq8) CK(cudaFree(dq8));
            CK(cudaFree((void*) ds)); CK(cudaFree(dw));
        }
        {   // 2-bit, one token: q2_alu and q2_lut against the reference
            HostQ2 h2 = make_q2(sh.N, sh.K, rng);
            std::vector<uint2> pq; std::vector<half> ps; pack_q2(h2, sabotage, pq, ps);
            uint2* dq = dupload(pq); half* dsc = dupload(ps);
            std::uniform_real_distribution<float> xd(-1.f, 1.f);
            std::vector<half> x(sh.K); for (auto& v : x) v = __float2half(xd(rng));
            half* dx = dupload(x); float* dy = dalloc<float>(sh.N);
            for (int lut = 0; lut < 2; ++lut) {
                CK(cudaMemset(dy, 0, sh.N * sizeof(float)));
                launch_q2(lut, dq, dsc, dx, dy, sh.N, sh.K);
                CK(cudaDeviceSynchronize());
                std::vector<float> y(sh.N); CK(cudaMemcpy(y.data(), dy, sh.N * sizeof(float), cudaMemcpyDeviceToHost));
                std::uniform_int_distribution<int> nd(0, sh.N - 1);
                double worst = 0; int wn = 0;
                for (int t = 0; t < 258; ++t) {
                    const int n = t == 0 ? 0 : t == 1 ? sh.N - 1 : nd(rng);
                    double mag; const double r = ref_q2(h2, x, n, &mag);
                    const double err = std::fabs(y[n] - r) / (mag + 1e-12);
                    if (err > worst) { worst = err; wn = n; }
                }
                const bool ok = worst <= 1e-4; failures += !ok;
                printf("check,%s,%s,1,%.3g,0,%d,%s\n", lut ? "q2_lut" : "q2_alu", sh.name, worst, wn, ok ? "ok" : "FAIL");
            }
            {   // warp-per-row 2-bit kernels (row-major layout)
                std::vector<uint32_t> rq; pack_q2_rowmajor(h2, sabotage, rq);
                uint32_t* drq = dupload(rq); half* drs = dupload(h2.scale_rowmajor);
                std::vector<std::pair<std::string, void (*)(const uint32_t*, const half*, const half*, float*, int, int)>> ks;
                ks.push_back({"q2w_alu", launch_q2w_alu}); ks.push_back({"q2w_alu_v2", launch_q2w_alu_v2}); ks.push_back({"q2w_alu_v3", launch_q2w_alu_v3});
                for (const auto& v : LUT_VARIANTS) ks.push_back({v.name, v.fn});
                for (auto& kk : ks) {
                    CK(cudaMemset(dy, 0, sh.N * sizeof(float)));
                    kk.second(drq, drs, dx, dy, sh.N, sh.K);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> y(sh.N); CK(cudaMemcpy(y.data(), dy, sh.N * sizeof(float), cudaMemcpyDeviceToHost));
                    std::uniform_int_distribution<int> nd(0, sh.N - 1);
                    double worst = 0; int wn = 0;
                    for (int t = 0; t < 258; ++t) {
                        const int n = t == 0 ? 0 : t == 1 ? sh.N - 1 : nd(rng);
                        double mag; const double r = ref_q2(h2, x, n, &mag);
                        const double err = std::fabs(y[n] - r) / (mag + 1e-12);
                        if (err > worst) { worst = err; wn = n; }
                    }
                    // a half-precision table rounds each entry once (2^-11): 2e-3, like the fp16 weight baselines
                    const double tol = kk.first.find("_f16") != std::string::npos ? 2e-3 : 1e-4;
                    const bool ok = worst <= tol; failures += !ok;
                    printf("check,%s,%s,1,%.3g,0,%d,%s\n", kk.first.c_str(), sh.name, worst, wn, ok ? "ok" : "FAIL");
                }
                CK(cudaFree(drq)); CK(cudaFree(drs));
            }
            CK(cudaFree(dq)); CK(cudaFree(dsc)); CK(cudaFree(dx)); CK(cudaFree(dy));
        }
    }
    failures += run_check_moe(hb, sabotage, rng);
    failures += run_check_glu(sabotage, rng);
    fprintf(stderr, "%d failure(s)%s\n", failures, sabotage ? " (sabotage mode: failures expected)" : "");
    return failures ? 1 : 0;
}

// ---- the sweep ------------------------------------------------------------------------------------------------

static void run_sweep(cublasHandle_t hb) {
    std::mt19937 rng(42);
    printf("contender,shape,N,K,M,us,weight_GBps,tflops,copies\n");
    const char* only = getenv("BAKEOFF_SHAPE");   // one table only (for a profiler); the MoE part is skipped then
    const int only_m = getenv("BAKEOFF_M") ? atoi(getenv("BAKEOFF_M")) : 0;   // one token count only (for a profiler)
    for (const Shape& sh : SHAPES) {
        if (only && strcmp(sh.name, only)) continue;
        const size_t NK = (size_t) sh.N * sh.K;
        HostQ h4 = make_weights(sh.N, sh.K, 4, rng), h8 = make_weights(sh.N, sh.K, 8, rng);
        const size_t q4b = NK / 2 + NK / GROUP * 2, q8b = NK + NK / GROUP * 2, f16b = NK * 2;
        const int c4 = copies_for(q4b), c8 = copies_for(q8b), cf = copies_for(f16b);
        std::vector<uint32_t*> q4(c4); std::vector<const half*> s4(c4), s8(c8);
        std::vector<uint2*> q8(c8); std::vector<half*> wf(cf);
        const auto p4 = pack_q4(h4, false);
        std::vector<uint2> p8(NK / 8); memcpy(p8.data(), h8.code.data(), NK);
        const auto w16 = dequant_host(h4);
        for (int i = 0; i < c4; ++i) { q4[i] = dupload(p4); s4[i] = dupload(h4.scale); }
        for (int i = 0; i < c8; ++i) { q8[i] = dupload(p8); s8[i] = dupload(h8.scale); }
        for (int i = 0; i < cf; ++i) wf[i] = dupload(w16);
        const int cw = copies_for(f16b);   // dq4+cublas writes and reads its fp16 copy: cycle these too
        std::vector<half*> wtmp(cw); for (auto& w : wtmp) w = dalloc<half>(NK);
        for (int M : MS) {
            if (only_m && M != only_m) continue;
            std::vector<half> x((size_t) M * sh.K, __float2half(0.5f));
            const half* dx = dupload(x);
            const float* dxf = dupload(std::vector<float>((size_t) M * sh.K, 0.5f));   // fp32 activations, as an engine holds them
            half* dxh = dalloc<half>((size_t) M * sh.K);
            float* dy = dalloc<float>((size_t) M * sh.N);
            const double flop = 2.0 * M * NK;
            auto report = [&](const char* who, double us, size_t wbytes, int copies) {
                printf("%s,%s,%d,%d,%d,%.2f,%.1f,%.3f,%d\n", who, sh.name, sh.N, sh.K, M, us, wbytes / us / 1e3,
                       flop / us / 1e6, copies);
                fflush(stdout);
            };
            report("cublas_fp16", time_us([&](int i) { cublas_gemm(hb, wf[i % cf], dx, dy, sh.N, sh.K, M); }), f16b, cf);
            report("dq4+cublas", time_us([&](int i) {
                dequant_q4<<<(int) ((NK / 8 + 255) / 256), 256>>>(q4[i % c4], s4[i % c4], wtmp[i % cw], sh.N, sh.K);
                cublas_gemm(hb, wtmp[i % cw], dx, dy, sh.N, sh.K, M); }), q4b, c4);
            if (tc_q4_ok(M, sh.N))
                report("tc_q4", time_us([&](int i) { launch_tc_q4(q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K, M); }), q4b, c4);
            if (tc_q4_v2_ok(M, sh.N, sh.K))
                report("tc_q4_v2", time_us([&](int i) { launch_tc_q4_v2(q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K, M); }), q4b, c4);
            if (getenv("TC2_TUNE") && tc_q4_v2_ok(M, sh.N, sh.K))   // every tile shape x every valid K split, for tuning
                for (const auto& cf : tc2::CFGS) {
                    if (!cf.fits(M, sh.N)) continue;
                    for (int S = 1; S <= 16; S *= 2) {
                        if ((sh.K / tc2::BK) % S) break;
                        const std::string name = std::string("tc_q4_v2[") + cf.name + "/s" + std::to_string(S) + "]";
                        report(name.c_str(), time_us([&](int i) { cf.run(q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K, M, S); }), q4b, c4);
                    }
                }
            if (M <= 8) {
                report("fused_q4_v2", time_us([&](int i) { fused_q4_v2_any(M, false, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q4_v2w", time_us([&](int i) { fused_q4_v2_any(M, true, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q4_v3", time_us([&](int i) { fused_q4_v3_any(M, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                // like for like with llama.cpp (fp32 activations in, conversion included): fp32 read directly, or a
                // separate fp32 -> fp16 step first; and that step alone
                report("fused_q4_v3f", time_us([&](int i) { fused_q4_v3_any(M, q4[i % c4], s4[i % c4], dxf, dy, sh.N, sh.K); }), q4b, c4);
                report("cvt+fused_q4_v3", time_us([&](int i) { fq3::launch_cvt(dxf, dxh, M * sh.K); fused_q4_v3_any(M, q4[i % c4], s4[i % c4], (const half*) dxh, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q4_v3_fp32in", time_us([&](int i) { fused_q4_v3_fp32in(M, q4[i % c4], s4[i % c4], dxf, dxh, dy, sh.N, sh.K); }), q4b, c4);
                report("cvt_f32_f16", time_us([&](int) { fq3::launch_cvt(dxf, dxh, M * sh.K); }), 0, 1);
                if (getenv("FQ3_TUNE"))   // every variant, for tuning
                    for (int c = 0; c < fq3::NCFG; ++c) {
                        std::string name = std::string("fused_q4_v3[") + fq3::CFGS[c].name + "]";
                        report(name.c_str(), time_us([&](int i) { fq3::run(c, M, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                        name = std::string("fused_q4_v3f[") + fq3::CFGS[c].name + "]";
                        report(name.c_str(), time_us([&](int i) { fq3::run(c, M, q4[i % c4], s4[i % c4], dxf, dy, sh.N, sh.K); }), q4b, c4);
                    }
                report("fused_q4", time_us([&](int i) { fused_q4_any(M, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q8", time_us([&](int i) { fused_q8_any(M, q8[i % c8], s8[i % c8], dx, dy, sh.N, sh.K); }), q8b, c8);
            }
            CK(cudaFree((void*) dx)); CK(cudaFree(dy)); CK(cudaFree((void*) dxf)); CK(cudaFree(dxh));
        }
        {   // 2-bit, one token
            HostQ2 h2 = make_q2(sh.N, sh.K, rng);
            std::vector<uint2> pq; std::vector<half> ps; pack_q2(h2, false, pq, ps);
            const size_t q2b = NK / 4 + NK / GROUP * 2; const int c2 = copies_for(q2b);
            std::vector<uint2*> dq(c2); std::vector<half*> dsc(c2);
            for (int i = 0; i < c2; ++i) { dq[i] = dupload(pq); dsc[i] = dupload(ps); }
            std::vector<half> x(sh.K, __float2half(0.5f)); half* dx = dupload(x); float* dy = dalloc<float>(sh.N);
            for (int lut = 0; lut < 2; ++lut) {
                const double us = time_us([&](int i) { launch_q2(lut, dq[i % c2], dsc[i % c2], dx, dy, sh.N, sh.K); });
                printf("%s,%s,%d,%d,1,%.2f,%.1f,%.3f,%d\n", lut ? "q2_lut" : "q2_alu", sh.name, sh.N, sh.K, us, q2b / us / 1e3,
                       2.0 * NK / us / 1e6, c2);
                fflush(stdout);
            }
            {   // warp-per-row 2-bit kernels
                std::vector<uint32_t> rq; pack_q2_rowmajor(h2, false, rq);
                std::vector<uint32_t*> drq(c2); std::vector<half*> drs(c2);
                for (int i = 0; i < c2; ++i) { drq[i] = dupload(rq); drs[i] = dupload(h2.scale_rowmajor); }
                std::vector<std::pair<const char*, void (*)(const uint32_t*, const half*, const half*, float*, int, int)>> ks;
                ks.push_back({"q2w_alu", launch_q2w_alu}); ks.push_back({"q2w_alu_v2", launch_q2w_alu_v2}); ks.push_back({"q2w_alu_v3", launch_q2w_alu_v3});
                for (const auto& v : LUT_VARIANTS) ks.push_back({v.name, v.fn});
                for (auto& kk : ks) {
                    const double us = time_us([&](int i) { kk.second(drq[i % c2], drs[i % c2], dx, dy, sh.N, sh.K); });
                    printf("%s,%s,%d,%d,1,%.2f,%.1f,%.3f,%d\n", kk.first, sh.name, sh.N, sh.K, us, q2b / us / 1e3,
                           2.0 * NK / us / 1e6, c2);
                    fflush(stdout);
                }
                for (int i = 0; i < c2; ++i) { CK(cudaFree(drq[i])); CK(cudaFree(drs[i])); }
            }
            for (int i = 0; i < c2; ++i) { CK(cudaFree(dq[i])); CK(cudaFree(dsc[i])); }
            CK(cudaFree(dx)); CK(cudaFree(dy));
        }
        for (auto p : q4) CK(cudaFree(p));
        for (auto p : s4) CK(cudaFree((void*) p));
        for (auto p : q8) CK(cudaFree(p));
        for (auto p : s8) CK(cudaFree((void*) p));
        for (auto p : wf) CK(cudaFree(p));
        for (auto w : wtmp) CK(cudaFree(w));
    }

    // gate + up + SwiGLU (Qwen3-1.7B's MLP), timed per (gate, up, glu) like llama.cpp's fused one-token call
    if (!only || !strcmp(only, "qwen3_1p7b_mlp_gate_up")) {
        const int N = 6144, K = 2048;
        const size_t NK2 = (size_t) 2 * N * K, qb = NK2 / 2 + NK2 / GROUP * 2;
        HostQ h = make_weights(2 * N, K, 4, rng);
        const auto p = pack_q4(h, false);
        const int c = copies_for(qb);
        std::vector<uint32_t*> q(c); std::vector<half*> sc(c);
        for (int i = 0; i < c; ++i) { q[i] = dupload(p); sc[i] = dupload(h.scale); }
        for (int M : {1, 2, 4, 8}) {
            if (only_m && M != only_m) continue;
            const float* dxf = dupload(std::vector<float>((size_t) M * K, 0.5f));
            half* dxh = dalloc<half>((size_t) M * K);
            float* dys = dalloc<float>((size_t) M * 2 * N); float* dout = dalloc<float>((size_t) M * N);
            const double us = time_us([&](int i) { gate_up_glu_fp32in(M, q[i % c], sc[i % c], dxf, dxh, dys, dout, N, K); });
            printf("gate_up_glu_fp32in,qwen3_1p7b_mlp_gate_up,%d,%d,%d,%.2f,%.1f,%.3f,%d\n", 2 * N, K, M, us, qb / us / 1e3,
                   2.0 * M * NK2 / us / 1e6, c);
            fflush(stdout);
            CK(cudaFree((void*) dxf)); CK(cudaFree(dxh)); CK(cudaFree(dys)); CK(cudaFree(dout));
        }
        for (int i = 0; i < c; ++i) { CK(cudaFree(q[i])); CK(cudaFree(sc[i])); }
    }
    // MoE decode: E experts, one token each (Strata Coder: 10 of 256; Qwen3.6: 8 of 256)
    for (const Shape& sh : SHAPES) {
        if (only || !strstr(sh.name, "expert")) continue;
        const int E = strstr(sh.name, "flashnext") ? 10 : 8;
        const size_t NK = (size_t) sh.N * sh.K;
        HostQ h4 = make_weights(sh.N, sh.K, 4, rng);
        const auto p4 = pack_q4(h4, false); const auto w16 = dequant_host(h4);
        const size_t q4b = NK / 2 + NK / GROUP * 2, f16b = NK * 2;
        const int c4 = copies_for(q4b * E) * E, cf = copies_for(f16b * E) * E;   // whole expert sets per copy
        std::vector<uint32_t*> q4(c4); std::vector<const half*> s4(c4); std::vector<half*> wf(cf);
        for (int i = 0; i < c4; ++i) { q4[i] = dupload(p4); s4[i] = dupload(h4.scale); }
        for (int i = 0; i < cf; ++i) wf[i] = dupload(w16);
        std::vector<half> x((size_t) E * sh.K, __float2half(0.5f));
        half* dx = dupload(x);
        float* dy = dalloc<float>((size_t) E * sh.N);
        // device pointer arrays for cuBLAS batched, one array per copy set
        const int sets = cf / E;
        std::vector<const void**> A(sets), B(sets); std::vector<void**> C(sets);
        for (int sidx = 0; sidx < sets; ++sidx) {
            std::vector<const void*> a(E), b(E); std::vector<void*> c(E);
            for (int e = 0; e < E; ++e) { a[e] = wf[sidx * E + e]; b[e] = dx + (size_t) e * sh.K; c[e] = dy + (size_t) e * sh.N; }
            A[sidx] = (const void**) dupload(a); B[sidx] = (const void**) dupload(b); C[sidx] = (void**) dupload(c);
        }
        const double flop = 2.0 * E * NK;
        auto report = [&](const char* who, double us, size_t wbytes) {
            printf("%s,%s_x%d,%d,%d,%d,%.2f,%.1f,%.3f,%d\n", who, sh.name, E, sh.N, sh.K, 1, us, wbytes * E / us / 1e3,
                   flop / us / 1e6, E);
            fflush(stdout);
        };
        const float one = 1.f, zero = 0.f;
        report("moe_cublas_loop", time_us([&](int i) {
            for (int e = 0; e < E; ++e) cublas_gemm(hb, wf[((i * E) + e) % cf], dx + (size_t) e * sh.K, dy + (size_t) e * sh.N, sh.N, sh.K, 1); }), f16b);
        report("moe_cublas_batched", time_us([&](int i) {
            CB(cublasGemmBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, 1, sh.K, &one, A[i % sets], CUDA_R_16F, sh.K,
                                   B[i % sets], CUDA_R_16F, sh.K, &zero, C[i % sets], CUDA_R_32F, sh.N, E,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }), f16b);
        report("moe_fused_q4_loop", time_us([&](int i) {
            for (int e = 0; e < E; ++e) { const int j = ((i * E) + e) % c4;
                launch_fused_q4<1>(q4[j], s4[j], dx + (size_t) e * sh.K, dy + (size_t) e * sh.N, sh.N, sh.K); } }), q4b);
        report("moe_fused_q4_one_launch", time_us([&](int i) {
            MoePtrs p{};
            for (int e = 0; e < E; ++e) { const int j = ((i * E) + e) % c4;
                p.q[e] = q4[j]; p.s[e] = s4[j]; p.x[e] = dx + (size_t) e * sh.K; p.y[e] = dy + (size_t) e * sh.N; }
            moe_fused_q4<<<dim3((sh.N + 3) / 4, E), 128>>>(p, sh.N, sh.K); }), q4b);
        for (auto p : q4) CK(cudaFree(p));
        for (auto p : s4) CK(cudaFree((void*) p));
        for (auto p : wf) CK(cudaFree(p));
        for (int sidx = 0; sidx < sets; ++sidx) { CK(cudaFree((void*) A[sidx])); CK(cudaFree((void*) B[sidx])); CK(cudaFree(C[sidx])); }
        CK(cudaFree(dx)); CK(cudaFree(dy));
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device: %s (sm_%d%d), L2 %d MB\n", prop.name, prop.major, prop.minor, prop.l2CacheSize >> 20);
    cublasHandle_t hb; CB(cublasCreate(&hb));
    int rc = 0;
    if (argc > 1 && !strcmp(argv[1], "--check")) {
        const char* sab = getenv("BAKEOFF_SABOTAGE");
        rc = run_check(hb, sab && sab[0] == '1');
    } else {
        run_sweep(hb);
    }
    CB(cublasDestroy(hb));
    return rc;
}
