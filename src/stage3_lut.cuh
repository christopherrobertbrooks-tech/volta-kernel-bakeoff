// Stage 2b (included after stage2.cuh): the 2-bit kernels rebuilt like our fast q4 kernel -- one warp per output row,
// coalesced row-major weights -- so the "times table" is judged inside a good design, not a slow one.
//
// Layout (row-major, like q4): q [N][K/16] uint32 (16 codes, code i in bits 2i..2i+1, u = code + 2), s [N][K/32] half.
//
//   q2w_alu       warp per row, each lane 16 codes per step, two multiply-adds per pair of codes
//   q2w_lut<R,T>  the same, but the block first builds the times table for the token -- for every pair of activations
//                 the 16 possible sums -- in shared memory (T = float or half), then each warp does R rows with ONE
//                 lookup per pair of codes. K is processed in tiles of 2048 activations so the table fits
//                 (2048/2 pairs x 16 = 16384 entries: 64 KB float / 32 KB half). R rows per warp, 8 warps per block:
//                 the table is built once per block and used by 8R rows.

__device__ __forceinline__ float dot16_q2(uint32_t v, const float* xs) {        // xs: 16 activations (fp32, smem)
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += xs[i] * (float) ((int) ((v >> (2 * i)) & 3u) - 2);
    return a;
}

__global__ void q2w_alu(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                        float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];                     // the token in fp32
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = __half2float(x[k]);
    __syncthreads();
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t* qr = q + (size_t) row * (K / 16);
    const half* sr = s + (size_t) row * (K / GROUP);
    float acc = 0.f;
    for (int c = lane; c < K / 16; c += 32)
        acc += dot16_q2(__ldg(qr + c), xs + 16 * c) * __half2float(__ldg(sr + (c >> 1)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

constexpr int LUT_KT = 2048;                          // activations per table tile
template <int R, typename T>
__global__ void q2w_lut(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                        float* __restrict__ y, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* tab = reinterpret_cast<T*>(smem_raw);          // [LUT_KT / 2][16]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int k0 = 0; k0 < K; k0 += LUT_KT) {
        const int kt = min(LUT_KT, K - k0);
        __syncthreads();
        for (int e = threadIdx.x; e < kt * 8; e += blockDim.x) {      // kt/2 pairs x 16 entries
            const int p = e >> 4, idx = e & 15;
            const float x0 = __half2float(x[k0 + 2 * p]), x1 = __half2float(x[k0 + 2 * p + 1]);
            const float v = x0 * (float) ((idx & 3) - 2) + x1 * (float) ((idx >> 2) - 2);
            if constexpr (sizeof(T) == 4) tab[e] = v; else tab[e] = __float2half(v);
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= N) break;
            const uint32_t* qr = q + (size_t) row * (K / 16) + k0 / 16;
            const half* sr = s + (size_t) row * (K / GROUP) + k0 / GROUP;
            float a = 0.f;
            for (int c = lane; c < kt / 16; c += 32) {
                const uint32_t v = __ldg(qr + c);
                const T* tc = tab + (size_t) (8 * c) * 16;               // this chunk's 8 pairs
                float part = 0.f;
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const T t = tc[p * 16 + ((v >> (4 * p)) & 15u)];
                    if constexpr (sizeof(T) == 4) part += t; else part += __half2float(t);
                }
                a += part * __half2float(__ldg(sr + (c >> 1)));
            }
            acc[r] += a;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float a = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}

// ---- host ----
static void pack_q2_rowmajor(const HostQ2& h, bool sabotage, std::vector<uint32_t>& q) {
    q.assign((size_t) h.N * h.K / 16, 0u);
    for (size_t i = 0; i < h.code.size(); ++i) {
        size_t src = i;
        if (sabotage) src = i ^ 1;                     // swap the two codes of every pair
        q[i / 16] |= (uint32_t) (h.code[src] + 2) << (2 * (i % 16));
    }
}
template <int R, typename T>
static void launch_q2w_lut_t(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    const size_t smem = (size_t) (LUT_KT / 2) * 16 * sizeof(T);
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(q2w_lut<R, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem)); attr = true; }
    const int rows_per_block = 8 * R;
    q2w_lut<R, T><<<(N + rows_per_block - 1) / rows_per_block, 256, smem>>>(q, s, x, y, N, K);
}
// the variants the sweep tries: rows per warp x table precision
struct LutVariant { const char* name; void (*fn)(const uint32_t*, const half*, const half*, float*, int, int); };
static const LutVariant LUT_VARIANTS[] = {
    {"q2w_lut_r1_f32", launch_q2w_lut_t<1, float>}, {"q2w_lut_r4_f32", launch_q2w_lut_t<4, float>},
    {"q2w_lut_r8_f32", launch_q2w_lut_t<8, float>}, {"q2w_lut_r16_f32", launch_q2w_lut_t<16, float>},
    {"q2w_lut_r4_f16", launch_q2w_lut_t<4, half>}, {"q2w_lut_r8_f16", launch_q2w_lut_t<8, half>},
    {"q2w_lut_r16_f16", launch_q2w_lut_t<16, half>},
};
static void launch_q2w_alu(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    q2w_alu<<<(N + 7) / 8, 256, K * sizeof(float)>>>(q, s, x, y, N, K);
}

// q2w_alu_v2: the warp-per-row 2-bit kernel with the fp16 bit trick -- (v >> 2i) & 0x00030003 holds codes i and i+8 in
// the two halves; OR 0x6400 makes each half 1024 + u, and one half2 subtract gives the exact code u - 2 for both.
__device__ __forceinline__ float dot16_q2_v2(uint32_t v, const float* xs) {
    const half2 k1026 = __float2half2_rn(1026.f);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t t = ((v >> (2 * i)) & 0x00030003u) | 0x64006400u;
        const float2 w = __half22float2(__hsub2(*reinterpret_cast<half2*>(&t), k1026));
        a += w.x * xs[i] + w.y * xs[i + 8];
    }
    return a;
}
__global__ void q2w_alu_v2(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                           float* __restrict__ y, int N, int K) {
    extern __shared__ float xs[];
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = __half2float(x[k]);
    __syncthreads();
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t* qr = q + (size_t) row * (K / 16);
    const half* sr = s + (size_t) row * (K / GROUP);
    float acc = 0.f;
    for (int c = lane; c < K / 16; c += 32)
        acc += dot16_q2_v2(__ldg(qr + c), xs + 16 * c) * __half2float(__ldg(sr + (c >> 1)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}
static void launch_q2w_alu_v2(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    q2w_alu_v2<<<(N + 7) / 8, 256, K * sizeof(float)>>>(q, s, x, y, N, K);
}

// q2w_alu_v3: like v2, but the activations come straight from global memory as two uint4 loads (16 halves) per step,
// the way the fast q4 kernel reads them. v1/v2 read a shared-memory fp32 copy at a 16-float stride per lane, which puts
// many lanes of a warp on the same shared-memory bank (bank conflicts) -- v2's faster unpack bought almost nothing.
__global__ void q2w_alu_v3(const uint32_t* __restrict__ q, const half* __restrict__ s, const half* __restrict__ x,
                           float* __restrict__ y, int N, int K) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t* qr = q + (size_t) row * (K / 16);
    const half* sr = s + (size_t) row * (K / GROUP);
    const half2 k1026 = __float2half2_rn(1026.f);
    float acc = 0.f;
    for (int c = lane; c < K / 16; c += 32) {
        const uint32_t v = __ldg(qr + c);
        const uint4 xa = __ldg(reinterpret_cast<const uint4*>(x) + 2 * c);       // x[16c .. 16c+7]
        const uint4 xb = __ldg(reinterpret_cast<const uint4*>(x) + 2 * c + 1);   // x[16c+8 .. 16c+15]
        const half* ha = reinterpret_cast<const half*>(&xa);
        const half* hb = reinterpret_cast<const half*>(&xb);
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t t = ((v >> (2 * i)) & 0x00030003u) | 0x64006400u;
            const float2 w = __half22float2(__hsub2(*reinterpret_cast<half2*>(&t), k1026));
            a += w.x * __half2float(ha[i]) + w.y * __half2float(hb[i]);
        }
        acc += a * __half2float(__ldg(sr + (c >> 1)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}
static void launch_q2w_alu_v3(const uint32_t* q, const half* s, const half* x, float* y, int N, int K) {
    q2w_alu_v3<<<(N + 3) / 4, 128>>>(q, s, x, y, N, K);
}
