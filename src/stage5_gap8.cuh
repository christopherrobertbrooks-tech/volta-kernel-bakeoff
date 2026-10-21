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
//   - activations in fp16, or (fused_q4_v3f) in fp32 as a real engine holds them, converted in registers -- the
//     like-for-like comparison with llama.cpp, whose times include converting its fp32 activations to 8 bits;
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

// 8 activations (k = 8 i .. 8 i + 7) of one token as four half2: straight from fp16, or converted from fp32 in registers
__device__ __forceinline__ uint4 load_x8(const half* xr, int i) { return __ldg(reinterpret_cast<const uint4*>(xr) + i); }
__device__ __forceinline__ uint4 load_x8(const float* xr, int i) {
    const float4* p = reinterpret_cast<const float4*>(xr) + 2 * i;
    const float4 a = __ldg(p), b = __ldg(p + 1);
    half2 h[4] = {__floats2half2_rn(a.x, a.y), __floats2half2_rn(a.z, a.w), __floats2half2_rn(b.x, b.y),
                  __floats2half2_rn(b.z, b.w)};
    return *reinterpret_cast<uint4*>(h);
}

// fp32 -> fp16 activations, the separate step a real engine (fp32 activations, as llama.cpp's) would need; n % 4 == 0
__global__ void cvt_f32_f16(const float* __restrict__ x, half* __restrict__ xh, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * i >= n) return;
    const float4 a = __ldg(reinterpret_cast<const float4*>(x) + i);
    half2* o = reinterpret_cast<half2*>(xh) + 2 * i;
    o[0] = __floats2half2_rn(a.x, a.y);
    o[1] = __floats2half2_rn(a.z, a.w);
}
static void launch_cvt(const float* x, half* xh, int n) { cvt_f32_f16<<<(n / 4 + 255) / 256, 256>>>(x, xh, n); }

template <int M, int S, int W, int R, typename XT>
__global__ void __launch_bounds__(128) fused_q4_v3(const uint32_t* __restrict__ q, const half* __restrict__ s,
                                                   const XT* __restrict__ x, float* __restrict__ y, int N, int K) {
    constexpr int RB = 4 / S * R;                                     // rows per block of 4 warps (R per warp)
    __shared__ float part[S > 1 ? 4 : 1][M * R];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * RB + (warp / S) * R, sp = warp % S;
    const bool live = row0 < N;   // rows past N (when N % R != 0): loads clamp to row N - 1, stores are skipped
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
                const int rr = min(row0 + r, N - 1);
                const uint32_t* qr = q + (size_t) rr * (K / 8) + c;
                uint32_t v[W];
                if (W == 2) { const uint2 p = __ldg(reinterpret_cast<const uint2*>(qr)); v[0] = p.x; v[W - 1] = p.y; }
                else v[0] = __ldg(qr);
                sc[r] = (32.f * W) * __half2float(__ldg(s + (size_t) rr * (K / GROUP) + (c >> 2)));   // W words: one group
#pragma unroll
                for (int i = 0; i < W; ++i)
#pragma unroll
                    for (int p = 0; p < 4; ++p) w[r][i][p] = unpack2<32 * W>(v[i], p);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const XT* xr = x + (size_t) m * K;
                half2 a[R];
#pragma unroll
                for (int r = 0; r < R; ++r) a[r] = __float2half2_rn(0.f);
#pragma unroll
                for (int i = 0; i < W; ++i) {
                    const uint4 xv = load_x8(xr, c + i);              // one activation load serves R rows
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
                if (row0 + r < N)
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
    if (live && sp == 0 && lane < M * R && row0 + lane / M < N) {
        float t = 0.f;
#pragma unroll
        for (int j = 0; j < S; ++j) t += part[warp + j][lane];
        y[(size_t) (lane % M) * N + row0 + lane / M] = t;
    }
}

template <typename XT, int M, int S, int W, int R>
static void launch(const uint32_t* q, const half* s, const XT* x, float* y, int N, int K) {
    constexpr int RB = 4 / S * R;
    fused_q4_v3<M, S, W, R, XT><<<(N + RB - 1) / RB, 128>>>(q, s, x, y, N, K);
}

// the variants tried: S warps per row, W words per lane per step, R rows per warp
struct Cfg { const char* name; int S, W, R; };
#define FQ3_CFGS(X) X(1, 1, 1) X(1, 2, 1) X(2, 1, 1) X(2, 2, 1) X(4, 1, 1) X(4, 2, 1) \
                    X(1, 1, 2) X(1, 2, 2) X(2, 1, 2) X(4, 1, 2) X(1, 1, 4) X(2, 1, 4) X(4, 1, 4)
#define FQ3_NAME(S, W, R) {"s" #S "w" #W "r" #R, S, W, R},
static const Cfg CFGS[] = {FQ3_CFGS(FQ3_NAME)};
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);

template <typename XT, int M>
static void run_m(int cfg, const uint32_t* q, const half* s, const XT* x, float* y, int N, int K) {
    using L = void (*)(const uint32_t*, const half*, const XT*, float*, int, int);
#define FQ3_FN(S, W, R) launch<XT, M, S, W, R>,
    static const L fns[] = {FQ3_CFGS(FQ3_FN)};
#undef FQ3_FN
    fns[cfg](q, s, x, y, N, K);
}
template <typename XT>
static bool run(int cfg, int M, const uint32_t* q, const half* s, const XT* x, float* y, int N, int K) {
    switch (M) {
        case 1: run_m<XT, 1>(cfg, q, s, x, y, N, K); return true;
        case 2: run_m<XT, 2>(cfg, q, s, x, y, N, K); return true;
        case 4: run_m<XT, 4>(cfg, q, s, x, y, N, K); return true;
        case 8: run_m<XT, 8>(cfg, q, s, x, y, N, K); return true;
        default: return false;
    }
}
}  // namespace fq3

// The default choice (from the tuning sweep, results/2026-10-20-fq3-tune); FQ3_CFG=<name> overrides it.
//   expert-sized tables (<= 2.6M weights): long rows (K >= 2048) split over 4 warps, short rows one warp each;
//   bigger tables: one token -> 2 warps per row, 2 words per lane; 2-8 tokens -> 4 rows per warp (one activation load
//   serves 4 rows), split over more warps until there are >= 5120 (2 tokens) or 1280 (4-8 tokens) warps.
// FQ3_CFG=<name> forces one variant (for tuning) on every path that picks one; -1 = not set
static int fq3_cfg_override() {
    if (const char* c = getenv("FQ3_CFG"))
        for (int i = 0; i < fq3::NCFG; ++i)
            if (!strcmp(c, fq3::CFGS[i].name)) return i;
    return -1;
}
static int fused_q4_v3_pick(int M, int N, int K) {
    if (const int forced = fq3_cfg_override(); forced >= 0) return forced;
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
template <typename XT>
static bool fused_q4_v3_any(int M, const uint32_t* q, const half* s, const XT* x, float* y, int N, int K) {
    return fq3::run(fused_q4_v3_pick(M, N, K), M, q, s, x, y, N, K);
}

// Like for like with llama.cpp: fp32 activations in, the conversion counted. One rule (results/2026-10-20-fq3f-tune):
// at 1 token, and at 2-4 tokens on expert-sized tables, read the fp32 activations in the kernel (2 rows per warp,
// split over more warps until there are >= 10240 warps on big tables, >= 1280 on expert-sized ones); otherwise
// convert to fp16 in a separate step first (cvt_f32_f16, ~2 us -- mostly the launch) and use fused_q4_v3's rule.
// xh: scratch for the fp16 copy, M * K halves. FQ3_CFG overrides the variant, not the path. An unsupported token
// count (anything but 1, 2, 4, 8) returns false before anything is launched.
static bool fused_q4_v3_fp32in(int M, const uint32_t* q, const half* s, const float* x, half* xh, float* y, int N, int K) {
    if (M != 1 && M != 2 && M != 4 && M != 8) return false;
    const bool small = (size_t) N * K <= (size_t) 2048 * 2560 / 2;
    if (M == 1 || (small && M <= 4)) {
        if (const int forced = fq3_cfg_override(); forced >= 0) return fq3::run(forced, M, q, s, x, y, N, K);
        const int target = small ? 1280 : 10240;
        int S = 1;
        while ((N / 2) * S < target && S < 4) S *= 2;
        int cfg = -1;
        for (int i = 0; i < fq3::NCFG; ++i)
            if (fq3::CFGS[i].S == S && fq3::CFGS[i].W == 1 && fq3::CFGS[i].R == 2) cfg = i;
        if (cfg >= 0) return fq3::run(cfg, M, q, s, x, y, N, K);
    }
    fq3::launch_cvt(x, xh, M * K);
    return fused_q4_v3_any(M, q, s, (const half*) xh, y, N, K);
}

// Gate + up + SwiGLU at a few tokens, as llama.cpp runs it at one token (one fused call, one activation conversion):
// gate and up stacked into one 2N-row table (rows 0..N-1 gate, N..2N-1 up), multiplied in one fused_q4_v3_fp32in
// pass, then out = silu(gate) * up. ystack: M x 2N scratch, out: M x N.
__global__ void swiglu_split(const float* __restrict__ ys, float* __restrict__ out, int N, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int m = i / N, n = i % N;
    const float g = ys[(size_t) m * 2 * N + n], u = ys[(size_t) m * 2 * N + N + n];
    out[i] = g / (1.f + __expf(-g)) * u;
}
static bool gate_up_glu_fp32in(int M, const uint32_t* q, const half* s, const float* x, half* xh, float* ystack,
                               float* out, int N, int K) {
    if (!fused_q4_v3_fp32in(M, q, s, x, xh, ystack, 2 * N, K)) return false;
    swiglu_split<<<(M * N + 255) / 256, 256>>>(ystack, out, N, M * N);
    return true;
}
