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
template<class T, bool Publish=false> __device__ void norm32(
    const T* input, T* output, unsigned rows, const T* scale=nullptr,
    unsigned heads=1, unsigned tokens=1, unsigned long long sb=0,
    unsigned long long sh=0, unsigned long long st=0, unsigned long long sc=1) {
    const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = thread / 4, lane = thread % 4;
    // All lanes in a four-thread subgroup either participate or exit together.
    if (row >= rows) return;
    const unsigned head=(row/tokens)%heads;
    const unsigned long long base=Publish ? (row/(heads*tokens))*sb + head*sh + (row%tokens)*st : row*32ull;
    const unsigned mask = __activemask();
    float values[8], sums[2];
    for (int parity = 0; parity < 2; ++parity) {
        for (int i = 0; i < 4; ++i)
            values[parity*4+i] = read_value(input[base + (lane*2 + parity + i*8)*(Publish ? sc : 1)]);
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
    if constexpr (Publish) {
        const float multiplier=scale ? read_value(scale[head]) : 1.f;
        for (int i=0;i<4;++i) {
            float a=half_round(values[i]*reciprocal),b=half_round(values[4+i]*reciprocal);
            if(scale) { a=half_round(a*multiplier);b=half_round(b*multiplier); }
            unsigned short fp8;unsigned packedHalf;
            // PTX packs the second f32 operand into the low byte.
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(fp8) : "f"(b),"f"(a));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packedHalf) : "h"(fp8));
            output[row*32ull+lane*2+i*8]=write_value<T>(read_half((unsigned short)packedHalf));
            output[row*32ull+lane*2+1+i*8]=write_value<T>(read_half((unsigned short)(packedHalf>>16)));
        }
    } else {
        for (int parity = 0; parity < 2; ++parity)
            for (int i = 0; i < 4; ++i)
                output[row*32ull + lane*2 + parity + i*8] = write_value<T>(values[parity*4+i]*reciprocal);
    }
}
extern "C" __global__ void cosine_norm_f16(const unsigned short* input, unsigned short* output, unsigned rows) {
    norm32(input,output,rows);
}
extern "C" __global__ void cosine_norm_f32(const float* input, float* output, unsigned rows) {
    norm32(input,output,rows);
}
extern "C" __global__ void cosine_publish_f16(const unsigned short* input,unsigned short* output,
    unsigned rows,const unsigned short* scale,unsigned heads,unsigned tokens,
    unsigned long long sb,unsigned long long sh,unsigned long long st,unsigned long long sc) {
    norm32<unsigned short,true>(input,output,rows,scale,heads,tokens,sb,sh,st,sc);
}
extern "C" __global__ void cosine_publish_f32(const float* input,float* output,
    unsigned rows,const float* scale,unsigned heads,unsigned tokens,
    unsigned long long sb,unsigned long long sh,unsigned long long st,unsigned long long sc) {
    norm32<float,true>(input,output,rows,scale,heads,tokens,sb,sh,st,sc);
}

__device__ __forceinline__ unsigned short half_bits(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}
struct Pair { float a,b; };
template<class T> __device__ Pair affine_pair(const T* x) {
    float a = half_round(read_value(x[0])*0.044921875f + 1.30078125f);
    float b = half_round(read_value(x[1])*0.044921875f + 1.30078125f);
    a = a < 1.03125f ? 1.03125f : (a > 1.5693359375f ? 1.5693359375f : a);
    b = b < 1.03125f ? 1.03125f : (b > 1.5693359375f ? 1.5693359375f : b);
    const unsigned packed = (unsigned)half_bits(a) | ((unsigned)half_bits(b)<<16);
    const unsigned transformed = (packed<<5) + 0x7ff88000u;
    return {read_half((unsigned short)transformed),read_half((unsigned short)(transformed>>16))};
}
__device__ __forceinline__ float positive_e4m3(float x) {
    const unsigned bits = __float_as_uint(x);
    const float step = x < 0.015625f ? 0.001953125f : __uint_as_float((((bits>>23)&255)-3)<<23);
    return __int2float_rn(__float2int_rn(x/step))*step;
}
template<class T> __device__ void softmax_rows(const T* input, T* output, unsigned rows, unsigned columns) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const unsigned thread = threadIdx.x, lane = thread%32, warp = thread/32;
    const unsigned long long base = (unsigned long long)row*columns;
    float sum = 0;
    for (unsigned pair = thread; pair < columns/2; pair += blockDim.x) {
        const auto value = affine_pair(input+base+pair*2);
        sum += value.a + value.b;
    }
    for (unsigned offset=16; offset; offset/=2) sum += __shfl_down_sync(0xffffffffu,sum,offset);
    __shared__ float partial[8];
    if (!lane) partial[warp] = sum;
    __syncthreads();
    if (!warp) {
        sum = lane < blockDim.x/32 ? partial[lane] : 0;
        for (unsigned offset=16; offset; offset/=2) sum += __shfl_down_sync(0xffffffffu,sum,offset);
        if (!lane) partial[0] = half_round(1.0f/half_round(sum));
    }
    __syncthreads();
    const float reciprocal = partial[0];
    for (unsigned pair = thread; pair < columns/2; pair += blockDim.x) {
        const auto value = affine_pair(input+base+pair*2);
        output[base+pair*2] = write_value<T>(positive_e4m3(half_round(value.a*reciprocal)));
        output[base+pair*2+1] = write_value<T>(positive_e4m3(half_round(value.b*reciprocal)));
    }
}
extern "C" __global__ void bit_affine_softmax_f16(const unsigned short* input, unsigned short* output, unsigned rows, unsigned columns) {
    softmax_rows(input,output,rows,columns);
}
extern "C" __global__ void bit_affine_softmax_f32(const float* input, float* output, unsigned rows, unsigned columns) {
    softmax_rows(input,output,rows,columns);
}

template<class T> __device__ void quadratic_activation(const T* input, T* output, unsigned long long count) {
    const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const float raw=read_value(input[i]);
    const float clamped=raw < -4.f ? -4.f : (raw > 4.f ? 4.f : raw);
    const float magnitude=clamped < 0.f ? -clamped : clamped;
    const float linear=half_round(magnitude*-0.055908203125f+0.447265625f);
    const float gate=half_round(clamped*linear+0.89453125f);
    output[i]=write_value<T>(raw*gate);
}
extern "C" __global__ void quadratic_activation_f16(const unsigned short* input,unsigned short* output,unsigned long long count) {
    quadratic_activation(input,output,count);
}
extern "C" __global__ void quadratic_activation_f32(const float* input,float* output,unsigned long long count) {
    quadratic_activation(input,output,count);
}

struct TensorLayout {
    unsigned long long sizes[8],strides[8];
    unsigned rank;
};
template<class T> __device__ float read_full(T value) { return (float)value; }
template<> __device__ float read_full<unsigned short>(unsigned short value) {return read_half(value);}
template<class T> __device__ void fp8_roundtrip(const T* input,T* output,unsigned long long count,TensorLayout layout) {
    const unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    unsigned long long offset=0,remainder=index;
    if(layout.rank==1)offset=index*layout.strides[0];
    else for(int d=(int)layout.rank-1;d>=0;--d) {
        offset+=(remainder%layout.sizes[d])*layout.strides[d];
        remainder/=layout.sizes[d];
    }
    const float value=read_full(input[offset]);
    unsigned short packed;unsigned decoded;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(value),"f"(value));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(decoded) : "h"(packed));
    output[index]=write_value<T>(read_half((unsigned short)decoded));
}
extern "C" __global__ void fp8_roundtrip_f16(const unsigned short* input,unsigned short* output,unsigned long long count,TensorLayout layout) {
    fp8_roundtrip(input,output,count,layout);
}
extern "C" __global__ void fp8_roundtrip_f32(const float* input,float* output,unsigned long long count,TensorLayout layout) {
    fp8_roundtrip(input,output,count,layout);
}
