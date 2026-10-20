// Stage 2 kernels (included by bakeoff.cu after its stage-1 kernels).
//
//   fused_q4_v2   the stage-1 fused q4 kernel with a faster unpack: two codes at a time become fp16 with one mask and
//                 one OR (0x6400 | n is the half 1024 + n), then one half2 subtract / fma gives the exact code - 8
//   fused_q4_v2w  the same unpack, but each lane reads a whole group (32 codes, one uint4) per step
//   tc_q4         prompts (M >= 64, M and N multiples of 64): 128x128 (or 64 along a short side) tile, all threads
//                 unpack q4 with the 0x6400 half2 trick, register-staged K pipeline, WMMA 16x16x16. See below.
//   tc_q4_v1      the original 64x64x32 kernel, kept so a sweep can time it next to tc_q4
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

// Original tensor-core prompt kernel (tc_q4_v1): block = 4 warps, tile 64 tokens x 64 rows, K step 32.
// Needs M % 64 == 0 and N % 64 == 0 (true for every shape here). Kept as the side-by-side fallback.
__global__ void tc_q4_v1(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
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

// 8 packed codes -> 8 fp16 values (code - 8) * scale, in K order. Same 0x6400 construction as dot8_q4:
// 0x6400 is the half 1024, so 0x6400|n is 1024+n (exact for n < 16); subtract 1032 to get n-8. A high nibble sits
// at bit 4, i.e. 16*n, so one half2 fma (multiply by 1/16, add -72) turns it into n-8 as well.
__device__ __forceinline__ half2 tc_u32_h2(uint32_t u) {
    union { uint32_t u; half2 h; } v;
    v.u = u;
    return v.h;
}
__device__ __forceinline__ uint4 tc_dequant8(uint32_t v, half sc, half2 k1032, half2 k16, half2 km72) {
    const uint32_t MAGIC = 0x64006400u;
    const half2 s2 = __half2half2(sc);
    const half2 ha = __hsub2(tc_u32_h2((v & 0x000F000Fu) | MAGIC), k1032);             // codes 0 and 4
    const half2 hb = __hfma2(tc_u32_h2((v & 0x00F000F0u) | MAGIC), k16, km72);         // codes 1 and 5
    const half2 hc = __hsub2(tc_u32_h2(((v >> 8) & 0x000F000Fu) | MAGIC), k1032);      // codes 2 and 6
    const half2 hd = __hfma2(tc_u32_h2(((v >> 8) & 0x00F000F0u) | MAGIC), k16, km72);  // codes 3 and 7
    union { uint4 u; half2 h[4]; } o;
    o.h[0] = __hmul2(__halves2half2(__low2half(ha), __low2half(hb)), s2);
    o.h[1] = __hmul2(__halves2half2(__low2half(hc), __low2half(hd)), s2);
    o.h[2] = __hmul2(__halves2half2(__high2half(ha), __high2half(hb)), s2);
    o.h[3] = __hmul2(__halves2half2(__high2half(hc), __high2half(hd)), s2);
    return o.u;
}

// Rebuilt prompt kernel. One block owns BM tokens by BN rows (128x128 when M and N allow it, 64 on a short side).
// Eight warps on the 128x128 tile, each a 64x32 WMMA tile (4x2 of 16x16). K advances by 64, two scale groups.
//
// sm_70 has no cp.async and no ldmatrix, so the next K tile is loaded into registers while WMMA runs on the current
// shared-memory tile, then written to smem after the readers finish. That is the whole double buffer: one smem stage
// (36,864 bytes for 128x128x64, leading dimension padded by 8) plus the in-flight registers. A second smem stage
// would not fit two blocks in 96 KB. launch_bounds keeps the 256-thread tile at 128 registers (ptxas spills 96
// bytes of addresses across the unrolled HMMA; the prefetched tile stays in registers) so two blocks, 16 warps,
// stay resident.
//
// Weights are [N][K] q4, row-major, so a K=64 slice is 32 contiguous bytes. Two neighbouring threads read one row
// (one uint4 each) and a warp covers 16 rows: 32-byte transactions, every thread unpacking. Activations are 128
// contiguous bytes per row; eight threads read one row. Scales are one fp16 per 32 values, so each uint4 is one scale.
// K % 64 == 0 on every shape in the sweep (the launcher falls back to tc_q4_v1 otherwise).
// MIN_BLOCKS is the occupancy target: 2 for the 256-thread tile (128 regs), 3 for 128 threads, 4 for 64.
template <int BM, int BN, int MIN_BLOCKS>
__global__ void __launch_bounds__((BM / 64) * (BN / 32) * 32, MIN_BLOCKS)
tc_q4_tile(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
           float* __restrict__ y, int N, int K) {
    constexpr int BK = 64;
    constexpr int LD = BK + 8;                                          // pad so consecutive rows don't share banks
    constexpr int THREADS = (BM / 64) * (BN / 32) * 32;
    constexpr int X4 = BK / 8;                                          // uint4s (8 halves) along one X row
    constexpr int W4 = BK / 32;                                         // uint4s (32 codes) along one W row
    constexpr int XPASS = (BM * X4) / THREADS;
    constexpr int WPASS = (BN * W4) / THREADS;
    constexpr int XROWS = THREADS / X4;
    constexpr int WARPS_N = BN / 32;
    constexpr int ACC_M = 4;                                            // 64 / 16
    constexpr int ACC_N = 2;                                            // 32 / 16
    static_assert(XPASS > 0 && WPASS > 0, "tile does not cover the block");
    static_assert(THREADS % X4 == 0 && (BN * W4) % THREADS == 0, "coalesced mapping");

    extern __shared__ __align__(16) char dyn_smem[];
    half* Xs = reinterpret_cast<half*>(dyn_smem);
    half* Ws = Xs + BM * LD;

    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp - warp_m * WARPS_N;
    const int wm = warp_m * 64, wn = warp_n * 32;

    const half2 k1032 = tc_u32_h2(0x64086408u);                         // 1032, 1032
    const half2 k16 = tc_u32_h2(0x2C002C00u);                           // 1/16, 1/16
    const half2 km72 = tc_u32_h2(0xD480D480u);                          // -72, -72

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[ACC_M][ACC_N];
#pragma unroll
    for (int i = 0; i < ACC_M; ++i)
#pragma unroll
        for (int j = 0; j < ACC_N; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    uint4 xreg[XPASS];
    uint4 wreg[WPASS];
    half screg[WPASS];

    // Prologue: first K tile into registers. The loop then stores that tile, prefetches the next one, and only
    // afterwards runs WMMA, so the global load stays in flight across the tensor-core math.
#pragma unroll
    for (int p = 0; p < XPASS; ++p) {
        const int row = p * XROWS + tid / X4;
        const int col4 = tid - (tid / X4) * X4;
        const half* rp = x + ((size_t) (m0 + row) * K);
        xreg[p] = __ldg(reinterpret_cast<const uint4*>(rp) + col4);
    }
#pragma unroll
    for (int p = 0; p < WPASS; ++p) {
        const int idx = p * THREADS + tid;
        const int row = idx / W4;
        const int part = idx - row * W4;
        const uint32_t* rp = q + (size_t) (n0 + row) * (K / 8);
        wreg[p] = __ldg(reinterpret_cast<const uint4*>(rp) + part);
        screg[p] = __ldg(s + (size_t) (n0 + row) * (K / GROUP) + part);
    }
    for (int k0 = 0; k0 < K; k0 += BK) {
        asm volatile("" ::: "memory");                                  // keep the global loads above the smem stores
#pragma unroll
        for (int p = 0; p < XPASS; ++p) {
            const int row = p * XROWS + tid / X4;
            const int col4 = tid - (tid / X4) * X4;
            *reinterpret_cast<uint4*>(&Xs[row * LD + col4 * 8]) = xreg[p];
        }
#pragma unroll
        for (int p = 0; p < WPASS; ++p) {
            const int idx = p * THREADS + tid;
            const int row = idx / W4;
            const int part = idx - row * W4;
            half* dst = Ws + row * LD + part * 32;
            const uint32_t w[4] = {wreg[p].x, wreg[p].y, wreg[p].z, wreg[p].w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
                *reinterpret_cast<uint4*>(dst + 8 * i) = tc_dequant8(w[i], screg[p], k1032, k16, km72);
        }
        __syncthreads();
        // Next K tile into registers before WMMA. smem still holds the tile compute is about to read.
        if (k0 + BK < K) {
            const int nk = k0 + BK;
#pragma unroll
            for (int p = 0; p < XPASS; ++p) {
                const int row = p * XROWS + tid / X4;
                const int col4 = tid - (tid / X4) * X4;
                const half* rp = x + ((size_t) (m0 + row) * K + nk);
                xreg[p] = __ldg(reinterpret_cast<const uint4*>(rp) + col4);
            }
#pragma unroll
            for (int p = 0; p < WPASS; ++p) {
                const int idx = p * THREADS + tid;
                const int row = idx / W4;
                const int part = idx - row * W4;
                const uint32_t* rp = q + (size_t) (n0 + row) * (K / 8) + (nk >> 3);
                wreg[p] = __ldg(reinterpret_cast<const uint4*>(rp) + part);
                screg[p] = __ldg(s + (size_t) (n0 + row) * (K / GROUP) + (nk / GROUP) + part);
            }
            asm volatile("" ::: "memory");
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[ACC_N];
#pragma unroll
            for (int j = 0; j < ACC_N; ++j)
                wmma::load_matrix_sync(fb[j], &Ws[(wn + 16 * j) * LD + kk], LD);
#pragma unroll
            for (int i = 0; i < ACC_M; ++i) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, &Xs[(wm + 16 * i) * LD + kk], LD);
#pragma unroll
                for (int j = 0; j < ACC_N; ++j) wmma::mma_sync(acc[i][j], fa, fb[j], acc[i][j]);
            }
        }
        __syncthreads();                                                // readers done, and the next loads have landed
    }
#pragma unroll
    for (int i = 0; i < ACC_M; ++i)
#pragma unroll
        for (int j = 0; j < ACC_N; ++j)
            wmma::store_matrix_sync(y + (size_t) (m0 + wm + 16 * i) * N + n0 + wn + 16 * j, acc[i][j], N,
                                    wmma::mem_row_major);
}

// 2-bit, one token. q: [K/32][N] uint2 (32 codes, value i in bits 2i..2i+1), s: [K/32][N] half.
__global__ void q2_alu(const uint2* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                       float* __restrict__ y, int N, int K) {
    extern __shared__ __align__(16) char dyn_smem[];  // same dynamic-smem symbol as tc_q4_tile
    float* xs = reinterpret_cast<float*>(dyn_smem);   // the token, as fp32
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
static void launch_tc_q4_v1(const uint32_t* q, const half* s, const half* x, float* y, int N, int K, int M) {
    tc_q4_v1<<<dim3(N / 64, M / 64), 128>>>(q, s, x, y, N, K);
}
// Opt into the full 96 KB (the default cap is 48 KB) and ask for the shared-memory-heavy carveout.
// The 128x128 tile uses ~36 KB, so two blocks can reside; the opt-in is what makes a larger dynamic tile legal.
template <int BM, int BN>
static void launch_tc_q4_tile(const uint32_t* q, const half* s, const half* x, float* y, int N, int K, int M) {
    constexpr int LD = 64 + 8;
    constexpr int THREADS = (BM / 64) * (BN / 32) * 32;
    constexpr int SMEM = (BM + BN) * LD * (int) sizeof(half);
    constexpr int MIN_BLOCKS = THREADS >= 256 ? 2 : THREADS >= 128 ? 3 : 4;
    auto* fn = tc_q4_tile<BM, BN, MIN_BLOCKS>;
    static int ready = 0;
    if (!ready) {
        CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        ready = 1;
    }
    fn<<<dim3(N / BN, M / BM), THREADS, SMEM>>>(q, s, x, y, N, K);
}
static void launch_tc_q4(const uint32_t* q, const half* s, const half* x, float* y, int N, int K, int M) {
    // Every sweep shape has K % 64 == 0. A K that is only a multiple of the group (32) uses the original kernel,
    // which steps K by 32, rather than reading off the end of a 64-wide tile.
    if (K % 64 != 0) {
        launch_tc_q4_v1(q, s, x, y, N, K, M);
        return;
    }
    if (M % 128 == 0 && N % 128 == 0) launch_tc_q4_tile<128, 128>(q, s, x, y, N, K, M);
    else if (N % 128 == 0) launch_tc_q4_tile<64, 128>(q, s, x, y, N, K, M);
    else if (M % 128 == 0) launch_tc_q4_tile<128, 64>(q, s, x, y, N, K, M);
    else launch_tc_q4_tile<64, 64>(q, s, x, y, N, K, M);
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
