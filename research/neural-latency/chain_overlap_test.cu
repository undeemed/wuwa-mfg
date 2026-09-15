// SPDX-License-Identifier: Apache-2.0
// Original bounded rendezvous. A later producer must overlap an earlier waiter.
// A clock deadline prevents an ordering mismatch from hanging the GPU.
extern "C" __global__ void chain_flag_clear(unsigned* flag)
{
    if(threadIdx.x==0)atomicExch(flag,0u);
}
extern "C" __global__ void chain_wait_bounded(unsigned* flag,unsigned* result,unsigned long long cycles)
{
    if(threadIdx.x!=0)return;
    const unsigned long long start=clock64();
    unsigned observed=0;
    do{observed=atomicAdd(flag,0u);}while(!observed && clock64()-start<cycles);
    result[0]=observed;
}
extern "C" __global__ void chain_signal(unsigned* flag)
{
    if(threadIdx.x==0)atomicExch(flag,1u);
}
