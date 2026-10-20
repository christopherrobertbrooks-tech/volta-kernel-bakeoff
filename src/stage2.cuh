// Stage 2 kernels (included by bakeoff.cu after its stage-1 kernels).
//
//   fused_q4_v2   the stage-1 fused q4 kernel with a faster unpack: two codes at a time become fp16 with one mask and
//                 one OR (0x6400 | n is the half 1024 + n), then one half2 subtract / fma gives the exact code - 8
//   fused_q4_v2w  the same unpack, but each lane reads a whole group (32 codes, one uint4) per step
//   tc_q4         prompts (M >= 64): each block unpacks a 64-row x 32-value tile of W into shared memory as fp16 and the
//                 V100's tensor cores (WMMA 16x16x16, fp32 sums) multiply it by a 64-token tile of x
//   q2_alu        2-bit codes {-2,-1,0,1} (u = code + 2), one fp16 scale per 32, stored slice-major ([K/32][N]
//                 uint2, so neighbouring threads read neighbouring rows); one thread per row multiplies as usual
//   q2_lut        the same weights, "times table": per 32-value slice the block precomputes, for each pair of
//                 activations, the 16 possible sums x[2p]*(lo-2) + x[2p+1]*(hi-2); each pair of codes is then ONE
//                 shared-memory lookup instead of two multiply-adds. Same layout and order as q2_alu, so the table is
//                 the only difference.
#include <mma.h>

__device__ __forceinline__ float dot8_q4(uint32_t v, const uint4 xv) {
    const uint32_t MAGIC = 0x64006400u;
    uint32_t a = (v & 0x000F000Fu) | MAGIC;           // codes 0 and 4 as 1024 + n
    uint32_t b = (v & 0x00F000F0u) | MAGIC;           // codes 1 and 5 as 1024 + 16 n
    uint32_t c = ((v >> 8) & 0x000F000Fu) | MAGIC;    // codes 2 and 6
    uint32_t d = ((v >> 8) & 0x00F000F0u) | MAGIC;    // codes 3 and 7
    const half2 k1032 = __float2half2_rn(1032.f), k16 = __float2half2_rn(0.0625f), km72 = __float2half2_rn(-72.f);
    const float2 wa = __half22float2(__hsub2(*reinterpret_cast<half2*>(&a), k1032));
    const float2 wb = __half22float2(__hfma2(*reinterpret_cast<half2*>(&b), k16, km72));
    const float2 wc = __half22float2(__hsub2(*reinterpret_cast<half2*>(&c), k1032));
    const float2 wd = __half22float2(__hfma2(*reinterpret_cast<half2*>(&d), k16, km72));
    const half2* xh = reinterpret_cast<const half2*>(&xv);
    const float2 x01 = __half22float2(xh[0]), x23 = __half22float2(xh[1]), x45 = __half22float2(xh[2]),
                 x67 = __half22float2(xh[3]);
    return wa.x * x01.x + wb.x * x01.y + wc.x * x23.x + wd.x * x23.y +
           wa.y * x45.x + wb.y * x45.y + wc.y * x67.x + wd.y * x67.y;
}

template <int M>
__global__ void fused_q4_v2(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
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
#pragma unroll
        for (int m = 0; m < M; ++m)
            acc[m] += dot8_q4(v, __ldg(reinterpret_cast<const uint4*>(x + (size_t) m * K) + c)) * sc;
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) y[(size_t) m * N + row] = a;
    }
}

template <int M>
__global__ void fused_q4_v2w(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                             float* __restrict__ y, int N, int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint4* qr = reinterpret_cast<const uint4*>(q + (size_t) row * (K / 8));   // one uint4 = one group of 32
    const half* sr = s + (size_t) row * (K / GROUP);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int g = lane; g < K / GROUP; g += 32) {
        const uint4 p = __ldg(qr + g);
        const float sc = __half2float(__ldg(sr + g));
        const uint32_t w[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t) m * K) + g * 4;
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) a += dot8_q4(w[i], __ldg(xr + i));
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

// Tensor-core prompt kernel: block = 4 warps, tile 64 tokens x 64 rows, K step 32 (one scale group).
// Needs M % 64 == 0 and N % 64 == 0 (true for every shape here).
__global__ void tc_q4(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                      float* __restrict__ y, int N, int K) {
    using namespace nvcuda;
    constexpr int BM = 64, BN = 64, BK = 32, LD = BK + 8;
    __shared__ __align__(32) half Xs[BM * LD];
    __shared__ __align__(32) half Ws[BN * LD];
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int warp = threadIdx.x >> 5;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    for (int k0 = 0; k0 < K; k0 += BK) {
        // x tile: 64 tokens x 32 halves = 256 uint4, two per thread
        for (int t = threadIdx.x; t < BM * BK / 8; t += blockDim.x) {
            const int r = t / (BK / 8), cpos = t % (BK / 8);
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (size_t) (m0 + r) * K + k0) + cpos);
            *reinterpret_cast<uint4*>(&Xs[r * LD + cpos * 8]) = v;
        }
        // W tile: 64 rows x one group; threads 0..63 each unpack one row's 32 codes to fp16 (code - 8) * scale
        if (threadIdx.x < BN) {
            const int r = threadIdx.x;
            const uint4 p = __ldg(reinterpret_cast<const uint4*>(q + (size_t) (n0 + r) * (K / 8)) + k0 / BK);
            const half sc = __ldg(s + (size_t) (n0 + r) * (K / GROUP) + k0 / GROUP);
            const uint32_t w[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int b = (w[i] >> (8 * j)) & 0xFF;
                    Ws[r * LD + i * 8 + 2 * j] = __hmul(__int2half_rn((b & 0xF) - 8), sc);
                    Ws[r * LD + i * 8 + 2 * j + 1] = __hmul(__int2half_rn((b >> 4) - 8), sc);
                }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &Xs[(wm + 16 * i) * LD + kk], LD);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], &Ws[(wn + 16 * j) * LD + kk], LD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(y + (size_t) (m0 + wm + 16 * i) * N + n0 + wn + 16 * j, acc[i][j], N,
                                    wmma::mem_row_major);
}

// 2-bit, one token. q: [K/32][N] uint2 (32 codes, value i in bits 2i..2i+1), s: [K/32][N] half.
__global__ void q2_alu(const uint2* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                       float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];                     // the token, as fp32
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = __half2float(x[k]);
    __syncthreads();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int j = 0; j < K / 32; ++j) {
        const uint2 w = __ldg(q + (size_t) j * N + n);
        const float* xj = xs + 32 * j;
        float a = 0.f;
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            const uint32_t word = p < 8 ? w.x : w.y;
            const uint32_t idx = (word >> (4 * (p & 7))) & 0xF;      // two codes
            a += xj[2 * p] * (float) ((int) (idx & 3) - 2) + xj[2 * p + 1] * (float) ((int) (idx >> 2) - 2);
        }
        acc += a * __half2float(__ldg(s + (size_t) j * N + n));
    }
    y[n] = acc;
}

constexpr int LUT_SLICES = 16;                        // 32-value slices per table batch: 16 x 1 KB of shared memory
__global__ void q2_lut(const uint2* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                       float* __restrict__ y, int N, int K) {
    __shared__ float T[LUT_SLICES][16][16];           // [slice][pair][two codes]
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (int j0 = 0; j0 < K / 32; j0 += LUT_SLICES) {
        const int ns = min(LUT_SLICES, K / 32 - j0);
        __syncthreads();                              // the previous batch's lookups are done
        for (int e = threadIdx.x; e < ns * 256; e += blockDim.x) {
            const int sl = e >> 8, p = (e >> 4) & 15, idx = e & 15;
            const int k = 32 * (j0 + sl) + 2 * p;
            T[sl][p][idx] = __half2float(x[k]) * (float) ((idx & 3) - 2) + __half2float(x[k + 1]) * (float) ((idx >> 2) - 2);
        }
        __syncthreads();
        if (n < N) {
            for (int sl = 0; sl < ns; ++sl) {
                const uint2 w = __ldg(q + (size_t) (j0 + sl) * N + n);
                float a = 0.f;
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const uint32_t word = p < 8 ? w.x : w.y;
                    a += T[sl][p][(word >> (4 * (p & 7))) & 0xF];
                }
                acc += a * __half2float(__ldg(s + (size_t) (j0 + sl) * N + n));
            }
        }
    }
    if (n < N) y[n] = acc;
}

// ---- host helpers for stage 2 ---------------------------------------------------------------------------------

template <int M>
static void launch_fused_q4_v2(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    fused_q4_v2<M><<<(N + 3) / 4, 128>>>(q, s, x, y, N, K);
}
template <int M>
static void launch_fused_q4_v2w(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    fused_q4_v2w<M><<<(N + 3) / 4, 128>>>(q, s, x, y, N, K);
}
static bool fused_q4_v2_any(int M, bool wide, const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
#define V2CASE(MM) case MM: if (wide) launch_fused_q4_v2w<MM>(q, s, x, y, N, K); else launch_fused_q4_v2<MM>(q, s, x, y, N, K); return true;
    switch (M) { V2CASE(1) V2CASE(2) V2CASE(4) V2CASE(8) default: return false; }
#undef V2CASE
}
static bool tc_q4_ok(int M, int N) { return M % 64 == 0 && N % 64 == 0; }
static void launch_tc_q4(const uint32_t* q, const half* s, const half* x, float* y, int N, int K, int M) {
    tc_q4<<<dim3(N / 64, M / 64), 128>>>(q, s, x, y, N, K);
}
static void launch_q2(bool lut, const uint2* q, const half* s, const half* x, float* y, int N, int K) {
    if (lut) q2_lut<<<(N + 255) / 256, 256>>>(q, s, x, y, N, K);
    else q2_alu<<<(N + 255) / 256, 256, K * sizeof(float)>>>(q, s, x, y, N, K);
}

// 2-bit weights (codes -2..1), packed slice-major; sabotage reverses the two codes of every pair
struct HostQ2 { int N, K; std::vector<int8_t> code; std::vector<half> scale_rowmajor; };
static HostQ2 make_q2(int N, int K, std::mt19937& rng) {
    HostQ2 h{N, K, std::vector<int8_t>((size_t) N * K), std::vector<half>((size_t) N * K / GROUP)};
    std::uniform_int_distribution<int> c2(-2, 1);
    std::uniform_real_distribution<float> sd(0.01f, 0.04f);
    for (auto& c : h.code) c = (int8_t) c2(rng);
    for (auto& v : h.scale_rowmajor) v = __float2half(sd(rng));
    return h;
}
static void pack_q2(const HostQ2& h, bool sabotage, std::vector<uint2>& q, std::vector<half>& s) {
    const int S = h.K / 32;
    q.assign((size_t) S * h.N, uint2{0, 0}); s.resize((size_t) S * h.N);
    for (int n = 0; n < h.N; ++n)
        for (int j = 0; j < S; ++j) {
            uint32_t w[2] = {0, 0};
            for (int i = 0; i < 32; ++i) {
                int ii = i;
                if (sabotage) ii = i ^ 1;
                const uint32_t u = (uint32_t) (h.code[(size_t) n * h.K + 32 * j + ii] + 2);
                w[i / 16] |= u << (2 * (i % 16));
            }
            q[(size_t) j * h.N + n] = uint2{w[0], w[1]};
            s[(size_t) j * h.N + n] = h.scale_rowmajor[(size_t) n * S + j];
        }
}
static double ref_q2(const HostQ2& h, const std::vector<half>& x, int n, double* mag) {
    double a = 0, m = 0;
    for (int k = 0; k < h.K; ++k) {
        const double t = (double) __half2float(x[k]) * h.code[(size_t) n * h.K + k] *
                         (double) __half2float(h.scale_rowmajor[(size_t) n * h.K / GROUP + k / GROUP]);
        a += t; m += std::fabs(t);
    }
    *mag = m;
    return a;
}
