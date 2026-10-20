// Stage 4: tc_q4_v2, the tensor-core prompt kernel rebuilt after the Nsight Compute profile (issues #1, #4, #7).
//
// tc_q4 (stage2.cuh) lost to cuBLAS 1.6-2.3x at 512 tokens. The profile showed why: 64 of 128 threads unpacked, one row
// each, so 37% of its global sectors were wasted; its shared memory had 5-6-way bank conflicts; one value at a time was
// converted; nothing overlapped the loads with the tensor cores; and on expert-sized tables 8-80 blocks left most SMs
// idle. This version:
//   - every thread loads and unpacks: a K step of 64 is 8 words (32 bytes, one full sector) per weight row and 128 bytes
//     per token row, split across the threads so each warp reads whole sectors;
//   - unpacks two 4-bit codes per instruction with the fp16 bit trick (0x6400 | n is the half 1024 + n): subtract 1032
//     first (exact for integers), then one multiply by the group scale -- the same single rounding as tc_q4;
//   - the next tile's loads are issued into registers before the current tile's MMAs (Volta has no cp.async) and
//     unpacked into shared memory after them; a second shared-memory buffer (DB) saves a barrier but lost every shape in
//     the tuning sweep, so the default is one buffer;
//   - 128x128 tiles (8 warps, 64x32 each) for big tables, 64x64 (4 warps, 32x32) otherwise; rows padded by 8 halves.
//     64x64 warp tiles (128x128w64) were tried: 255 registers + spills, 12-33% slower at 512 tokens;
//   - split-K for small grids: blockIdx.z takes a slice of K and the partial sums are added into y with atomics.
// Layouts as everywhere here: q [N][K/8] (word w of a row holds codes 8w..8w+7, code + 8 in nibble i), s [N][K/32] half,
// x [M][K] half, y [M][N] float. Needs M % 64 == 0, N % 64 == 0, K % 64 == 0 (every shape here).
#include <mma.h>

namespace tc2 {
constexpr int BK = 64, LD = BK + 8;

// 8 codes of one word -> 8 halves (code - 8) * scale, in k order, as one uint4
__device__ __forceinline__ uint4 unpack8(uint32_t v, half2 s2) {
    const half2 k1032 = __float2half2_rn(1032.f);
    uint32_t r[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t b = v >> (8 * p);
        uint32_t t = (b & 0x0Fu) | ((b & 0xF0u) << 12) | 0x64006400u;   // the halves 1024 + lo, 1024 + hi
        half2 h = __hmul2(__hsub2(*reinterpret_cast<half2*>(&t), k1032), s2);
        r[p] = *reinterpret_cast<uint32_t*>(&h);
    }
    return make_uint4(r[0], r[1], r[2], r[3]);
}

template <int BM, int BN, int WM, int WN, bool DB>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
tc_q4_v2(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x, float* __restrict__ y,
         int N, int K, int k_per_split, bool atomic) {
    using namespace nvcuda;
    constexpr int T = (BM / WM) * (BN / WN) * 32, FM = WM / 16, FN = WN / 16;
    constexpr int XV = BM * BK / 8, WW = BN * BK / 8;               // uint4 of x, words of q per tile
    static_assert(XV % T == 0 && WW % T == 0, "tile not divisible by the thread count");
    constexpr int XPT = XV / T, WPT = WW / T;
    extern __shared__ __align__(128) unsigned char smem[];          // DB: two buffers [X0 | W0 | X1 | W1], else one
    half* const buf = reinterpret_cast<half*>(smem);
    constexpr int STAGE = (BM + BN) * LD;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kb = blockIdx.z * k_per_split, ke = min(K, kb + k_per_split);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = (warp / (BN / WN)) * WM, wn = (warp % (BN / WN)) * WN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    uint4 xr[XPT];
    uint32_t wr[WPT];
    half2 sr[WPT];
    auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < XPT; ++i) {
            const int t = threadIdx.x + i * T, r = t / (BK / 8), c = t % (BK / 8);
            xr[i] = __ldg(reinterpret_cast<const uint4*>(x + (size_t) (m0 + r) * K + k0) + c);
        }
#pragma unroll
        for (int i = 0; i < WPT; ++i) {
            const int t = threadIdx.x + i * T, r = t / (BK / 8), c = t % (BK / 8);
            wr[i] = __ldg(q + (size_t) (n0 + r) * (K / 8) + k0 / 8 + c);
            sr[i] = __half2half2(__ldg(s + (size_t) (n0 + r) * (K / GROUP) + (k0 + 8 * c) / GROUP));
        }
    };
    auto store = [&](int b) {
        half* const Xs = buf + b * STAGE;
        half* const Ws = Xs + BM * LD;
#pragma unroll
        for (int i = 0; i < XPT; ++i) {
            const int t = threadIdx.x + i * T, r = t / (BK / 8), c = t % (BK / 8);
            *reinterpret_cast<uint4*>(&Xs[r * LD + c * 8]) = xr[i];
        }
#pragma unroll
        for (int i = 0; i < WPT; ++i) {
            const int t = threadIdx.x + i * T, r = t / (BK / 8), c = t % (BK / 8);
            *reinterpret_cast<uint4*>(&Ws[r * LD + c * 8]) = unpack8(wr[i], sr[i]);
        }
    };
    load(kb);
    store(0);
    __syncthreads();
    int b = 0;
    for (int k0 = kb; k0 < ke; k0 += BK) {
        const bool more = k0 + BK < ke;
        if (more) load(k0 + BK);                    // the next tile's loads in flight while the tensor cores work
        const half* const Xs = buf + b * STAGE;
        const half* const Ws = Xs + BM * LD;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[FM];
#pragma unroll
            for (int i = 0; i < FM; ++i) wmma::load_matrix_sync(fa[i], &Xs[(wm + 16 * i) * LD + kk], LD);
#pragma unroll
            for (int j = 0; j < FN; ++j) {          // one weight fragment at a time: fewer registers for big warp tiles
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, &Ws[(wn + 16 * j) * LD + kk], LD);
#pragma unroll
                for (int i = 0; i < FM; ++i) wmma::mma_sync(acc[i][j], fa[i], fb, acc[i][j]);
            }
        }
        if (DB) {
            if (more) store(b ^ 1);                 // the other buffer: nobody reads it this step
            b ^= 1;
            __syncthreads();
        } else {
            __syncthreads();                        // one buffer: everyone done reading before it is rewritten
            if (more) store(0);
            __syncthreads();
        }
    }
    if (!atomic) {
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::store_matrix_sync(y + (size_t) (m0 + wm + 16 * i) * N + n0 + wn + 16 * j, acc[i][j], N,
                                        wmma::mem_row_major);
        return;
    }
    // split-K: each warp stages one 16x16 fragment at a time in (now free) shared memory and adds it into y
    float* scratch = reinterpret_cast<float*>(buf) + warp * 256;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                atomicAdd(&y[(size_t) (m0 + wm + 16 * i + e / 16) * N + n0 + wn + 16 * j + e % 16], scratch[e]);
            __syncwarp();
        }
}
}  // namespace tc2

static bool tc_q4_v2_ok(int M, int N, int K) { return M % 64 == 0 && N % 64 == 0 && K % tc2::BK == 0; }

namespace tc2 {
template <int BM, int BN>
static bool fits(int M, int N) { return M % BM == 0 && N % BN == 0; }

template <int BM, int BN, int WM, int WN, bool DB>
static void launch(const uint32_t* q, const half* s, const half* x, float* y, int N, int K, int M, int S) {
    constexpr int T = (BM / WM) * (BN / WN) * 32;
    constexpr int SMEM = (DB ? 2 : 1) * (BM + BN) * LD * (int) sizeof(half);
    static bool set = false;
    if (!set) {
        CK(cudaFuncSetAttribute(tc_q4_v2<BM, BN, WM, WN, DB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        set = true;
    }
    if (S > 1) CK(cudaMemsetAsync(y, 0, sizeof(float) * (size_t) M * N));
    tc_q4_v2<BM, BN, WM, WN, DB><<<dim3(N / BN, M / BM, S), T, SMEM>>>(q, s, x, y, N, K, K / S, S > 1);
}

// the tile shapes tried (block BM x BN, warp WM x WN); name used by the tuning sweep
struct Cfg { const char* name; bool (*fits)(int, int); void (*run)(const uint32_t*, const half*, const half*, float*, int, int, int, int); };
// name: block tile, then w<warp tile> (default 32x32 or 64x32), and "-1" for a single shared-memory buffer
static const Cfg CFGS[] = {
    {"64x64", fits<64, 64>, launch<64, 64, 32, 32, true>},
    {"128x64", fits<128, 64>, launch<128, 64, 64, 32, true>},
    {"64x128", fits<64, 128>, launch<64, 128, 32, 64, true>},
    {"128x128", fits<128, 128>, launch<128, 128, 64, 32, true>},
    {"64x64-1", fits<64, 64>, launch<64, 64, 32, 32, false>},
    {"128x128-1", fits<128, 128>, launch<128, 128, 64, 32, false>},
    {"128x128w64", fits<128, 128>, launch<128, 128, 64, 64, true>},
    {"128x128w64-1", fits<128, 128>, launch<128, 128, 64, 64, false>},
};
constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);
}  // namespace tc2

// The default choice (see tc_q4_v2_pick); TC2_CFG=<name> and TC2_SPLIT=<n> override it (for tuning).
struct Tc2Choice { int cfg, split; };
static Tc2Choice tc_q4_v2_pick(int M, int N, int K) {
    if (const char* c = getenv("TC2_CFG")) {
        for (int i = 0; i < tc2::NCFG; ++i)
            if (!strcmp(c, tc2::CFGS[i].name)) return {i, getenv("TC2_SPLIT") ? atoi(getenv("TC2_SPLIT")) : 1};
    }
    // 128x128 when it alone gives >= 80 blocks; else 64x64 with K split (a power of two dividing the K steps) to >= 160.
    // Both single-buffered: in the tuning sweep (results/2026-10-20-tc2-tune2) the one-buffer versions won every shape at
    // 64-512 tokens -- the second buffer costs more in occupancy than the overlap gains.
    if (M % 128 == 0 && N % 128 == 0 && (M / 128) * (N / 128) >= 80) return {5, 1};
    const int blocks = (M / 64) * (N / 64), steps = K / tc2::BK;
    int S = 1;
    while (blocks * S < 160 && steps % (2 * S) == 0) S *= 2;
    return {4, S};
}

static void launch_tc_q4_v2(const uint32_t* q, const half* s, const half* x, float* y, int N, int K, int M) {
    const Tc2Choice c = tc_q4_v2_pick(M, N, K);
    tc2::CFGS[c.cfg].run(q, s, x, y, N, K, M, c.split);
}
