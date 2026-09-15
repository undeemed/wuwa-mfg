// SPDX-License-Identifier: Apache-2.0
// Operation order adapted from iamwavecut/MLX-DLSS, pinned in the research README.
// Fused inference implementation of the recovered 32-channel half fragment tree.
// No weights, proprietary machine code, or CUDA toolkit headers are included.
// Output grading below implements our algebraic approximation of observed controls.
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

template<class T> __device__ float grade_read(T x) { return (float)x; }
template<> __device__ float grade_read<unsigned short>(unsigned short x) { return read_half(x); }
template<class T> __device__ T grade_write(float x) { return (T)x; }
template<> __device__ unsigned short grade_write<unsigned short>(float x) { return write_value<unsigned short>(x); }
__device__ float grade_clamp(float x) { return fminf(1.f,fmaxf(0.f,x)); }
template<class T> __device__ void grade_rgb(const T* input,T* output,
    unsigned pixels,unsigned height,unsigned width,
    unsigned long long sb,unsigned long long sc,unsigned long long sh,unsigned long long sw,
    float exposure,float contrast,float saturation) {
    const unsigned pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=pixels)return;
    const unsigned column=pixel%width,row=(pixel/width)%height,batch=pixel/(width*height);
    const unsigned long long base=batch*sb+row*sh+column*sw;
    float rgb[3];
    #pragma unroll
    for(unsigned channel=0;channel<3;++channel) {
        float x=grade_clamp(grade_read(input[base+channel*sc]));
        x=grade_clamp(x*exposure);
        // Compiled with --fmad=false to match the separate Torch operations.
        const float delta=x*x*(3.f-2.f*x)-x;
        rgb[channel]=grade_clamp(x+contrast*delta);
    }
    const float high=fmaxf(rgb[0],fmaxf(rgb[1],rgb[2]));
    const float low=fminf(rgb[0],fminf(rgb[1],rgb[2]));
    const float lightness=(high+low)*.5f;
    #pragma unroll
    for(unsigned channel=0;channel<3;++channel)
        output[(unsigned long long)pixel*3+channel]=grade_write<T>(
            grade_clamp(lightness+saturation*(rgb[channel]-lightness)));
}
extern "C" __global__ void output_grade_f16(const unsigned short* input,unsigned short* output,
    unsigned pixels,unsigned height,unsigned width,
    unsigned long long sb,unsigned long long sc,unsigned long long sh,unsigned long long sw,
    float exposure,float contrast,float saturation) {
    grade_rgb(input,output,pixels,height,width,sb,sc,sh,sw,exposure,contrast,saturation);
}
extern "C" __global__ void output_grade_f32(const float* input,float* output,
    unsigned pixels,unsigned height,unsigned width,
    unsigned long long sb,unsigned long long sc,unsigned long long sh,unsigned long long sw,
    float exposure,float contrast,float saturation) {
    grade_rgb(input,output,pixels,height,width,sb,sc,sh,sw,exposure,contrast,saturation);
}

// Bilinear coefficient interpolation, affine RGB correction and detail compose.
// Finite FP16 operands. The dense 12-channel full-resolution field is not stored.
extern "C" __global__ void affine_compose_f16(
    const unsigned short* source,const unsigned short* detail,const unsigned short* field,
    unsigned short* output,unsigned pixels,unsigned height,unsigned width,
    unsigned long long sb,unsigned long long sc,unsigned long long sy,unsigned long long sx,
    unsigned long long db,unsigned long long dc,unsigned long long dy,unsigned long long dx,
    unsigned long long fb,unsigned long long fc,unsigned long long fy,unsigned long long fx) {
    const unsigned pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=pixels)return;
    const unsigned x=pixel%width,y=(pixel/width)%height,b=pixel/(width*height);
    const unsigned gridH=(height+31)/32,gridW=(width+31)/32;
    const float u=fmaxf(0.f,(x+.5f)*.03125f-.5f),v=fmaxf(0.f,(y+.5f)*.03125f-.5f);
    const unsigned left=(unsigned)u,top=(unsigned)v;
    const unsigned right=left+1<gridW ? left+1 : left,bottom=top+1<gridH ? top+1 : top;
    const float wx=u-left,wy=v-top,ax=1.f-wx,ay=1.f-wy;
    const unsigned long long sourceBase=b*sb+y*sy+x*sx,detailBase=b*db+y*dy+x*dx;
    float color[3];
    #pragma unroll
    for(unsigned c=0;c<3;++c)color[c]=read_half(source[sourceBase+c*sc]);
    #pragma unroll
    for(unsigned row=0;row<3;++row) {
        float affine=0.f;
        #pragma unroll
        for(unsigned column=0;column<4;++column) {
            const unsigned long long base=b*fb+(row*4+column)*fc;
            const float tl=read_half(field[base+top*fy+left*fx]);
            const float tr=read_half(field[base+top*fy+right*fx]);
            const float bl=read_half(field[base+bottom*fy+left*fx]);
            const float br=read_half(field[base+bottom*fy+right*fx]);
            // Explicit RN multiply-adds match the tested CUDA interpolation path.
            const float upper=__fmaf_rn(ax,tl,wx*tr),lower=__fmaf_rn(ax,bl,wx*br);
            const float coefficient=half_round(__fmaf_rn(ay,upper,wy*lower));
            const float term=column<3 ? half_round(coefficient*color[column]) : coefficient;
            affine=column==0 ? term : half_round(affine+term);
        }
        const float residual=read_half(detail[detailBase+row*dc]);
        const float correction=half_round(.25f*half_round(residual+affine));
        output[(unsigned long long)pixel*3+row]=write_value<unsigned short>(
            grade_clamp(half_round(color[row]+correction)));
    }
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

// Recovered PCG/Box-Muller feature construction, using explicit GPU operations.
// This is an arithmetic diagnostic; native sample equality must be measured.
__device__ __forceinline__ unsigned noise_mix(unsigned value) {
    return (value ^ (value >> ((value >> 28) + 4))) * 0x108ef2d9u;
}
__device__ __forceinline__ float noise_uniform(unsigned value) {
    value = noise_mix(value);
    return (float)(((value >> 30) ^ (value >> 8)) + 1u) * 0x1p-24f;
}
__device__ __forceinline__ float noise_radius(float value) {
    float logarithm, radicand, radius;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(logarithm) : "f"(value));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(radicand) : "f"(logarithm),"f"(0.69314718246459960938f));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(radicand) : "f"(radicand),"f"(-2.f));
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(radius) : "f"(radicand));
    return radius;
}
extern "C" __global__ void gaussian_noise_f32(float* output,unsigned width,unsigned height,unsigned frame) {
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=width*height)return;
    const unsigned x=index%width,y=index/width;
    unsigned mixed=noise_mix(y*0xd8163841u ^ x*0x8da6b343u ^ frame*0x9e3779b9u ^ 0x243f6a88u);
    mixed ^= mixed >> 22;
    const float a=noise_uniform(mixed*0xcaa5b80du+0x21dd796bu);
    const float b=noise_uniform(mixed*0x83232c31u+0x3463e0acu);
    const float c=noise_uniform(mixed*0x2c9277b5u+0xac564b05u);
    const float d=noise_uniform(mixed*0xfa6dc5f9u+0x4712a88eu);
    const float ra=noise_radius(a),rb=noise_radius(c);
    const float angle_a=d*6.2831854820251464844f,angle_b=b*6.2831854820251464844f;
    float ca,sa,cb;
    asm("cos.approx.ftz.f32 %0, %1;" : "=f"(ca) : "f"(angle_a));
    asm("sin.approx.ftz.f32 %0, %1;" : "=f"(sa) : "f"(angle_a));
    asm("cos.approx.ftz.f32 %0, %1;" : "=f"(cb) : "f"(angle_b));
    output[index*3]=half_round(rb*ca);
    output[index*3+1]=half_round(rb*sa);
    output[index*3+2]=half_round(ra*cb);
}

// Direct SM89 FP8 tensor-core diagnostic. Layout follows NVIDIA PTX ISA 8.7
// mma.m16n8k32 fragments. Every active warp executes every MMA collectively.
// C is loaded before the first product; K advances in 32-element chunks.
extern "C" __global__ void fp8_mma_f16(
    const unsigned char* a,const unsigned char* b,const unsigned short* seed,
    unsigned short* output,unsigned m,unsigned n,unsigned k,unsigned batches,
    unsigned long long bstride,unsigned long long cstride) {
    const unsigned lane=threadIdx.x&31u,group=lane>>2,part=lane&3u;
    const unsigned tiles_m=(m+15)/16,tiles_n=(n+7)/8;
    const unsigned long long tile=(unsigned long long)blockIdx.x*4+(threadIdx.x>>5);
    const unsigned long long tiles_per_batch=(unsigned long long)tiles_m*tiles_n;
    if(tile>=tiles_per_batch*batches)return; // warp-uniform, including the final CTA
    const unsigned batch=tile/tiles_per_batch;
    const unsigned row=((tile/tiles_n)%tiles_m)*16+group;
    const unsigned column=(tile%tiles_n)*8;
    const unsigned ccol=column+part*2;
    const unsigned long long abase=(unsigned long long)batch*m*k;
    const unsigned long long bbase=(unsigned long long)batch*bstride;
    const unsigned long long cbase=(unsigned long long)batch*cstride;
    unsigned c0=0,c1=0;
    if(seed) {
        if(row<m && ccol<n)c0=seed[cbase+(unsigned long long)row*n+ccol];
        if(row<m && ccol+1<n)c0|=(unsigned)seed[cbase+(unsigned long long)row*n+ccol+1]<<16;
        if(row+8<m && ccol<n)c1=seed[cbase+(unsigned long long)(row+8)*n+ccol];
        if(row+8<m && ccol+1<n)c1|=(unsigned)seed[cbase+(unsigned long long)(row+8)*n+ccol+1]<<16;
    }
    for(unsigned start=0;start<k;start+=32) {
        const unsigned acol=start+part*4;
        unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
        if(row<m) {
            a0=*(const unsigned*)(a+abase+(unsigned long long)row*k+acol);
            a2=*(const unsigned*)(a+abase+(unsigned long long)row*k+acol+16);
        }
        if(row+8<m) {
            a1=*(const unsigned*)(a+abase+(unsigned long long)(row+8)*k+acol);
            a3=*(const unsigned*)(a+abase+(unsigned long long)(row+8)*k+acol+16);
        }
        if(column+group<n) {
            #pragma unroll
            for(unsigned i=0;i<4;++i) {
                b0|=(unsigned)b[bbase+(unsigned long long)(acol+i)*n+column+group]<<(i*8);
                b1|=(unsigned)b[bbase+(unsigned long long)(acol+i+16)*n+column+group]<<(i*8);
            }
        }
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                     "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                     : "+r"(c0),"+r"(c1) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned long long obase=(unsigned long long)batch*m*n;
    if(row<m && ccol<n)output[obase+(unsigned long long)row*n+ccol]=(unsigned short)c0;
    if(row<m && ccol+1<n)output[obase+(unsigned long long)row*n+ccol+1]=(unsigned short)(c0>>16);
    if(row+8<m && ccol<n)output[obase+(unsigned long long)(row+8)*n+ccol]=(unsigned short)c1;
    if(row+8<m && ccol+1<n)output[obase+(unsigned long long)(row+8)*n+ccol+1]=(unsigned short)(c1>>16);
}

extern "C" __global__ void half_mma_f16(
    const unsigned short* a,const unsigned short* b,const unsigned short* seed,
    unsigned short* output,unsigned m,unsigned n,unsigned k,unsigned batches,
    unsigned long long bstride,unsigned long long cstride) {
    const unsigned lane=threadIdx.x&31u,group=lane>>2,part=lane&3u;
    const unsigned tiles_m=(m+15)/16,tiles_n=(n+7)/8;
    const unsigned long long tile=(unsigned long long)blockIdx.x*4+(threadIdx.x>>5);
    const unsigned long long tiles_per_batch=(unsigned long long)tiles_m*tiles_n;
    if(tile>=tiles_per_batch*batches)return;
    const unsigned batch=tile/tiles_per_batch;
    const unsigned row=((tile/tiles_n)%tiles_m)*16+group;
    const unsigned column=(tile%tiles_n)*8,ccol=column+part*2;
    const unsigned long long abase=(unsigned long long)batch*m*k,bbase=(unsigned long long)batch*bstride;
    const unsigned long long cbase=(unsigned long long)batch*cstride;
    unsigned c0=0,c1=0;
    if(seed) {
        if(row<m && ccol<n)c0=seed[cbase+(unsigned long long)row*n+ccol];
        if(row<m && ccol+1<n)c0|=(unsigned)seed[cbase+(unsigned long long)row*n+ccol+1]<<16;
        if(row+8<m && ccol<n)c1=seed[cbase+(unsigned long long)(row+8)*n+ccol];
        if(row+8<m && ccol+1<n)c1|=(unsigned)seed[cbase+(unsigned long long)(row+8)*n+ccol+1]<<16;
    }
    for(unsigned start=0;start<k;start+=16) {
        const unsigned acol=start+part*2;
        unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
        if(row<m) {
            a0=*(const unsigned*)(a+abase+(unsigned long long)row*k+acol);
            a2=*(const unsigned*)(a+abase+(unsigned long long)row*k+acol+8);
        }
        if(row+8<m) {
            a1=*(const unsigned*)(a+abase+(unsigned long long)(row+8)*k+acol);
            a3=*(const unsigned*)(a+abase+(unsigned long long)(row+8)*k+acol+8);
        }
        if(column+group<n) {
            b0=(unsigned)b[bbase+(unsigned long long)acol*n+column+group]
               |((unsigned)b[bbase+(unsigned long long)(acol+1)*n+column+group]<<16);
            b1=(unsigned)b[bbase+(unsigned long long)(acol+8)*n+column+group]
               |((unsigned)b[bbase+(unsigned long long)(acol+9)*n+column+group]<<16);
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                     "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                     : "+r"(c0),"+r"(c1) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    const unsigned long long obase=(unsigned long long)batch*m*n;
    if(row<m && ccol<n)output[obase+(unsigned long long)row*n+ccol]=(unsigned short)c0;
    if(row<m && ccol+1<n)output[obase+(unsigned long long)row*n+ccol+1]=(unsigned short)(c0>>16);
    if(row+8<m && ccol<n)output[obase+(unsigned long long)(row+8)*n+ccol]=(unsigned short)c1;
    if(row+8<m && ccol+1<n)output[obase+(unsigned long long)(row+8)*n+ccol+1]=(unsigned short)(c1>>16);
}
