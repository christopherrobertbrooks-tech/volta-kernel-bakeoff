// Stage 5: fused_q4_v3, the few-token 4-bit kernel rebuilt for issue #2 (8 tokens: ours 83 us vs llama.cpp's 53).
//
// Nsight Compute (results/2026-10-20-ncu-gap8) showed fused_q4_v2<8> is bound by its own instructions, not memory: 2.1x
// llama.cpp's instruction count on the 12288x2560 table, issue slots 71% busy, DRAM 26%. Per 4-byte word and token it
// converts 8 activations to fp32 and does 8 fp32 FMAs. On expert-sized tables (512 rows) it launches 512 warps for 80
// SMs and waits on memory. This version:
//   - unpacks each byte into one half2 (code 2p, code 2p + 1) in storage order -- PRMT + LOP3 + one HFMA2 per byte,
//     so the activations need no reshuffling -- and multiplies in half2 (two multiply-adds per instruction) straight
//     against the fp16 activations;
//   - sums W words (4 W products per half lane) in fp16, then adds that in fp32 with the group scale. The weights are
//     unpacked as (code - 8) / (32 W), exact, so |w| <= 1/4W and a partial sum of 4 W products stays within max |x|:
//     no overflow for any activation fp16 can hold. (The first version used / 8 and overflowed for |x| above ~16K;
//     the check's "bigx" case caught it.) The 32 W goes back into the fp32 scale;
//   - splits each row across S warps (partial sums added in shared memory) so small tables fill the card;
//   - R rows per warp: each activation load (fp16, 16 bytes per 8 values and token) serves R rows -- at 8 tokens one
//     row per warp spent its time re-reading the activations from L1.
// Precision: each product and partial sum is rounded to fp16 (11 bits), so it is checked in the fp16 class (2e-3,
// like cuBLAS fp16). llama.cpp's path rounds the activations to 8 bits instead.
// Layouts as in stage 1: q [N][K/8] (code i of a word in nibble i, code + 8), s [N][K/32] half, x [M][K] half,
// y [M][N] float. Needs K % (8 W) == 0 (every shape here).

namespace fq3 {
// byte p of v -> half2 ((code 2p - 8) / D, (code 2p+1 - 8) / D), exact for D = 32 or 64: the low half is
// (1024 + lo) / D - 1032 / D, the high half (1024 + 16 hi) / (16 D) - 72 / D; each intermediate fits fp16's 11 bits
template <int D>
__device__ __forceinline__ half2 unpack2(uint32_t v, int p) {
    uint32_t t = __byte_perm(v, 0u, 0x4040u | (p << 8) | p);         // bytes [b, 0, b, 0]
    t = (t & 0x00F0000Fu) | 0x64006400u;                              // halves 1024 + lo, 1024 + 16 hi
    const half2 mul = __halves2half2(__float2half(1.f / D), __float2half(1.f / (16 * D)));
    const half2 add = __halves2half2(__float2half(-1032.f / D), __float2half(-72.f / D));
    return __hfma2(*reinterpret_cast<half2*>(&t), mul, add);
}

template <int M, int S, int W, int R>
__global__ void __launch_bounds__(128) fused_q4_v3(const uint32_t* __restrict__ q, const half* __restrict__ s,
                                                   const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    constexpr int RB = 4 / S * R;                                     // rows per block of 4 warps (R per warp)
    __shared__ float part[S > 1 ? 4 : 1][M * R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * RB + (warp / S) * R, sp = warp % S;
    const bool live = row0 < N;                                       // N % R == 0 for every shape here
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    if (live) {
        for (int c = (sp * 32 + lane) * W; c < K / 8; c += S * 32 * W) {
            half2 w[R][W][4];
            float sc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t* qr = q + (size_t) (row0 + r) * (K / 8) + c;
                uint32_t v[W];
                if (W == 2) { const uint2 p = __ldg(reinterpret_cast<const uint2*>(qr)); v[0] = p.x; v[W - 1] = p.y; }
                else v[0] = __ldg(qr);
                sc[r] = (32.f * W) * __half2float(__ldg(s + (size_t) (row0 + r) * (K / GROUP) + (c >> 2)));   // W words: one group
#pragma unroll
                for (int i = 0; i < W; ++i)
#pragma unroll
                    for (int p = 0; p < 4; ++p) w[r][i][p] = unpack2<32 * W>(v[i], p);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t) m * K) + c;
                half2 a[R];
#pragma unroll
                for (int r = 0; r < R; ++r) a[r] = __float2half2_rn(0.f);
#pragma unroll
                for (int i = 0; i < W; ++i) {
                    const uint4 xv = __ldg(xr + i);                   // one activation load serves R rows
                    const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int p = 0; p < 4; ++p) a[r] = __hfma2(w[r][i][p], xh[p], a[r]);
                }
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][m] += (__low2float(a[r]) + __high2float(a[r])) * sc[r];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float a = acc[r][m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            acc[r][m] = a;
        }
    if (S == 1) {
        if (live && lane == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int m = 0; m < M; ++m) y[(size_t) m * N + row0 + r] = acc[r][m];
        return;
    }
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) part[warp][r * M + m] = acc[r][m];
    __syncthreads();
    if (live && sp == 0 && lane < M * R) {
        float t = 0.f;
#pragma unroll
        for (int j = 0; j < S; ++j) t += part[warp + j][lane];
        y[(size_t) (lane % M) * N + row0 + lane / M] = t;
    }
}

template <int M, int S, int W, int R>
static void launch(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    constexpr int RB = 4 / S * R;
    fused_q4_v3<M, S, W, R><<<(N + RB - 1) / RB, 128>>>(q, s, x, y, N, K);
}

// the variants tried: S warps per row, W words per lane per step, R rows per warp
struct Cfg { const char* name; int S, W, R; };
#define FQ3_CFGS(X) X(1, 1, 1) X(1, 2, 1) X(2, 1, 1) X(2, 2, 1) X(4, 1, 1) X(4, 2, 1) \
                    X(1, 1, 2) X(1, 2, 2) X(2, 1, 2) X(4, 1, 2) X(1, 1, 4) X(2, 1, 4) X(4, 1, 4)
#define FQ3_NAME(S, W, R) {"s" #S "w" #W "r" #R, S, W, R},
static const Cfg CFGS[] = {FQ3_CFGS(FQ3_NAME)};
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);

template <int M>
static void run_m(int cfg, const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    using L = void (*)(const uint32_t*, const half*, const half*, float*, int, int);
#define FQ3_FN(S, W, R) launch<M, S, W, R>,
    static const L fns[] = {FQ3_CFGS(FQ3_FN)};
#undef FQ3_FN
    fns[cfg](q, s, x, y, N, K);
}
static bool run(int cfg, int M, const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    switch (M) {
        case 1: run_m<1>(cfg, q, s, x, y, N, K); return true;
        case 2: run_m<2>(cfg, q, s, x, y, N, K); return true;
        case 4: run_m<4>(cfg, q, s, x, y, N, K); return true;
        case 8: run_m<8>(cfg, q, s, x, y, N, K); return true;
        default: return false;
    }
}
}  // namespace fq3

// The default choice (from the tuning sweep, results/2026-10-20-fq3-tune); FQ3_CFG=<name> overrides it.
//   expert-sized tables (<= 2.6M weights): long rows (K >= 2048) split over 4 warps, short rows one warp each;
//   bigger tables: one token -> 2 warps per row, 2 words per lane; 2-8 tokens -> 4 rows per warp (one activation load
//   serves 4 rows), split over more warps until there are >= 5120 (2 tokens) or 1280 (4-8 tokens) warps.
static int fused_q4_v3_pick(int M, int N, int K) {
    if (const char* c = getenv("FQ3_CFG"))
        for (int i = 0; i < fq3::NCFG; ++i)
            if (!strcmp(c, fq3::CFGS[i].name)) return i;
    auto find = [](int S, int W, int R) {
        for (int i = 0; i < fq3::NCFG; ++i)
            if (fq3::CFGS[i].S == S && fq3::CFGS[i].W == W && fq3::CFGS[i].R == R) return i;
        return 0;
    };
    if ((size_t) N * K <= (size_t) 2048 * 2560 / 2) return K >= 2048 ? find(4, M <= 2 ? 2 : 1, 1) : find(1, 1, 1);
    if (M == 1) return find(2, 2, 1);
    const int target = M == 2 ? 5120 : 1280;
    int S = 1;
    while ((N / 4) * S < target && S < 4) S *= 2;
    return find(S, 1, 4);
}
static bool fused_q4_v3_any(int M, const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    return fq3::run(fused_q4_v3_pick(M, N, K), M, q, s, x, y, N, K);
}
