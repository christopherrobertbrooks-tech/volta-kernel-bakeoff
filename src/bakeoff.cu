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
// sampled outputs; BAKEOFF_SABOTAGE=1 swaps the two nibbles of every q4 byte so the check must fail (proves it can).
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
    if (row >= N) return;
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
    if (row >= N) return;
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
    if (row >= N) return;
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
                std::vector<uint2> p((size_t) sh.N * sh.K / 8);
                memcpy(p.data(), h.code.data(), h.code.size());
                dq8 = dupload(p);
            }
            half* dw = dupload(dequant_host(h));    // the fp16 baseline's weights (rounded once to fp16)
            for (int M : {1, 4, 8, 64}) {
                std::uniform_real_distribution<float> xd(-1.f, 1.f);
                std::vector<half> x((size_t) M * sh.K);
                for (auto& v : x) v = __float2half(xd(rng));
                const half* dx = dupload(x);
                float* dy = dalloc<float>((size_t) M * sh.N);
                std::vector<std::pair<std::string, int>> runs;
                if (bits == 4) {
                    if (M <= 8) { runs.push_back({"fused_q4", 0}); runs.push_back({"fused_q4_v2", 4}); runs.push_back({"fused_q4_v2w", 5}); }
                    if (tc_q4_ok(M, sh.N)) runs.push_back({"tc_q4", 6});
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
                    // more (2^-11): allow 2e-3 for those, 1e-4 for the kernels that use the exact codes
                    const double tol = (r.first == "cublas_fp16" || r.first == "dq4+cublas" || r.first == "tc_q4") ? 2e-3 : 1e-4;
                    const bool ok = worst <= tol;
                    failures += !ok;
                    printf("check,%s,%s,%d,%.3g,%d,%d,%s\n", r.first.c_str(), sh.name, M, worst, wm, wn,
                           ok ? "ok" : "FAIL");
                }
                CK(cudaFree((void*) dx)); CK(cudaFree(dy));
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
            CK(cudaFree(dq)); CK(cudaFree(dsc)); CK(cudaFree(dx)); CK(cudaFree(dy));
        }
    }
    fprintf(stderr, "%d failure(s)%s\n", failures, sabotage ? " (sabotage mode: failures expected)" : "");
    return failures ? 1 : 0;
}

// ---- the sweep ------------------------------------------------------------------------------------------------

static void run_sweep(cublasHandle_t hb) {
    std::mt19937 rng(42);
    printf("contender,shape,N,K,M,us,weight_GBps,tflops,copies\n");
    for (const Shape& sh : SHAPES) {
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
        half* wtmp = dalloc<half>(NK);
        for (int M : MS) {
            std::vector<half> x((size_t) M * sh.K, __float2half(0.5f));
            const half* dx = dupload(x);
            float* dy = dalloc<float>((size_t) M * sh.N);
            const double flop = 2.0 * M * NK;
            auto report = [&](const char* who, double us, size_t wbytes, int copies) {
                printf("%s,%s,%d,%d,%d,%.2f,%.1f,%.3f,%d\n", who, sh.name, sh.N, sh.K, M, us, wbytes / us / 1e3,
                       flop / us / 1e6, copies);
                fflush(stdout);
            };
            report("cublas_fp16", time_us([&](int i) { cublas_gemm(hb, wf[i % cf], dx, dy, sh.N, sh.K, M); }), f16b, cf);
            report("dq4+cublas", time_us([&](int i) {
                dequant_q4<<<(int) ((NK / 8 + 255) / 256), 256>>>(q4[i % c4], s4[i % c4], wtmp, sh.N, sh.K);
                cublas_gemm(hb, wtmp, dx, dy, sh.N, sh.K, M); }), q4b, c4);
            if (tc_q4_ok(M, sh.N))
                report("tc_q4", time_us([&](int i) { launch_tc_q4(q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K, M); }), q4b, c4);
            if (M <= 8) {
                report("fused_q4_v2", time_us([&](int i) { fused_q4_v2_any(M, false, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q4_v2w", time_us([&](int i) { fused_q4_v2_any(M, true, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q4", time_us([&](int i) { fused_q4_any(M, q4[i % c4], s4[i % c4], dx, dy, sh.N, sh.K); }), q4b, c4);
                report("fused_q8", time_us([&](int i) { fused_q8_any(M, q8[i % c8], s8[i % c8], dx, dy, sh.N, sh.K); }), q8b, c8);
            }
            CK(cudaFree((void*) dx)); CK(cudaFree(dy));
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
            for (int i = 0; i < c2; ++i) { CK(cudaFree(dq[i])); CK(cudaFree(dsc[i])); }
            CK(cudaFree(dx)); CK(cudaFree(dy));
        }
        for (auto p : q4) CK(cudaFree(p));
        for (auto p : s4) CK(cudaFree((void*) p));
        for (auto p : q8) CK(cudaFree(p));
        for (auto p : s8) CK(cudaFree((void*) p));
        for (auto p : wf) CK(cudaFree(p));
        CK(cudaFree(wtmp));
    }

    // MoE decode: E experts, one token each (Strata Coder: 10 of 256; Qwen3.6: 8 of 256)
    for (const Shape& sh : SHAPES) {
        if (!strstr(sh.name, "expert")) continue;
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
