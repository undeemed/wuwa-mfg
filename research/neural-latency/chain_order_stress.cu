// SPDX-License-Identifier: Apache-2.0
// Original finite ordering litmus: varied block latency and dynamic shared memory.
// Every access stays in a bounded allocation even if producer ordering is broken.
__device__ __forceinline__ void skew(unsigned salt)
{
    if(((blockIdx.x+salt)%7u)==0u)
    {
        const unsigned long long start=clock64();
        while(clock64()-start<20000ull){}
    }
}
extern "C" __global__ void chain_fill(unsigned* output,unsigned count,unsigned seed)
{
    extern __shared__ unsigned temporary[];
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    temporary[threadIdx.x]=(i*1664525u+seed)^0x12345678u;
    __syncthreads();skew(0);
    if(i<count)output[i]=temporary[threadIdx.x];
}
extern "C" __global__ void chain_mix(const unsigned* input,unsigned* output,unsigned count,unsigned salt)
{
    extern __shared__ unsigned temporary[];
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    skew(salt);
    temporary[threadIdx.x]=i<count?input[(i*17u+13u)&(count-1u)]*22695477u+salt:0u;
    __syncthreads();
    if(i<count)output[i]=temporary[threadIdx.x];
}
