// SPDX-License-Identifier: Apache-2.0
// Original bounded integer workload. No model data or native shader code.
extern "C" __global__ void chain_fill(unsigned* output,unsigned count,unsigned seed)
{
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=(i*1664525u+seed)^0x12345678u;
}
extern "C" __global__ void chain_mix(const unsigned* input,unsigned* output,unsigned count,unsigned salt)
{
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=input[(i*17u+13u)&(count-1u)]*22695477u+salt;
}
