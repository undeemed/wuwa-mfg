// SPDX-License-Identifier: Apache-2.0
// Operation order adapted from iamwavecut/MLX-DLSS, pinned in the research README.
// Fused inference implementation of the recovered 32-channel half fragment tree.
// No weights, proprietary machine code, or CUDA toolkit headers are included.
__device__ __forceinline__ float half_round(float x) {
    unsigned short h;
    float y;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}
__device__ __forceinline__ float read_half(unsigned short x) {
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(x));
    return y;
}
template<class T> __device__ float read_value(T x) { return half_round((float)x); }
template<> __device__ float read_value<unsigned short>(unsigned short x) { return read_half(x); }
template<class T> __device__ T write_value(float x) { return (T)half_round(x); }
template<> __device__ unsigned short write_value<unsigned short>(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}
template<class T> __device__ void norm32(const T* input, T* output, unsigned rows) {
    const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = thread / 4, lane = thread % 4;
    // All lanes in a four-thread subgroup either participate or exit together.
    if (row >= rows) return;
    const unsigned mask = __activemask();
    float values[8], sums[2];
    for (int parity = 0; parity < 2; ++parity) {
        for (int i = 0; i < 4; ++i)
            values[parity*4+i] = read_value(input[row*32ull + lane*2 + parity + i*8]);
        const float a = values[parity*4], b = values[parity*4+1];
        const float c = values[parity*4+2], d = values[parity*4+3];
        const float first = half_round(b*b + half_round(a*a));
        const float second = half_round(d*d + half_round(c*c));
        float partial = half_round(first + second);
        partial = half_round(partial + __shfl_xor_sync(mask, partial, 2, 4));
        partial = half_round(partial + __shfl_xor_sync(mask, partial, 1, 4));
        sums[parity] = partial;
    }
    float sum = half_round(sums[0] + sums[1]);
    sum = sum < 0.00006198883056640625f ? 0.00006198883056640625f : sum;
    float reciprocal;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(reciprocal) : "f"(sum));
    reciprocal = half_round(reciprocal);
    for (int parity = 0; parity < 2; ++parity)
        for (int i = 0; i < 4; ++i)
            output[row*32ull + lane*2 + parity + i*8] = write_value<T>(values[parity*4+i]*reciprocal);
}
extern "C" __global__ void cosine_norm_f16(const unsigned short* input, unsigned short* output, unsigned rows) {
    norm32(input,output,rows);
}
extern "C" __global__ void cosine_norm_f32(const float* input, float* output, unsigned rows) {
    norm32(input,output,rows);
}
