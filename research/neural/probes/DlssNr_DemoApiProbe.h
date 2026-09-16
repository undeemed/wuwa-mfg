#pragma once
// SPDX-License-Identifier: GPL-3.0-only
// Original pass-through observer; records public API identifiers and status only.
#include "DlssNr_DemoCommandProbe.h"
#include <unordered_set>

namespace DlssNr::DemoApiProbe
{
inline std::mutex mutex;
inline std::unordered_set<unsigned> resolved;
template<unsigned Id,typename Signature>struct Hook;
template<unsigned Id,typename... Args>struct Hook<Id,NvAPI_Status(__cdecl*)(Args...)>
{
    using Fn=NvAPI_Status(__cdecl*)(Args...);
    inline static Fn original=nullptr;
    static NvAPI_Status __cdecl Call(Args... args)
    {
        const bool selected=DemoCommandProbe::activeFrame.load()!=0;
        if(selected)DemoCommandProbe::Event({{"kind","native_api_enter"},{"id",Id}});
        const auto status=original(args...);
        if(selected)DemoCommandProbe::Event({{"kind","native_api_exit"},{"id",Id},{"status",status}});
        return status;
    }
};
inline void* Resolve(unsigned id,void* function)
{
    if(!DemoCommandProbe::enabled)return function;
    std::lock_guard lock(mutex);
    const bool first=resolved.insert(id).second;
    const char* name="unmapped";void* result=function;bool wrapped=false;
#define NR_API_CASE(ID,NAME) case ID: {name=#NAME;using H=Hook<ID,decltype(&NAME)>; \
    if(function){if(H::original && H::original!=reinterpret_cast<typename H::Fn>(function)) \
        DemoCommandProbe::coverageGap=true;H::original=reinterpret_cast<typename H::Fn>(function); \
        result=reinterpret_cast<void*>(&H::Call);wrapped=true;}break;}
    switch(id)
    {
        NR_API_CASE(0x329fe6e0,NvAPI_D3D12_GetCudaMergedTextureSamplerObject)
        NR_API_CASE(0x0ddac234,NvAPI_D3D12_GetCudaIndependentDescriptorObject)
        NR_API_CASE(0x80403fc9,NvAPI_D3D12_GetCudaTextureObject)
        NR_API_CASE(0x48f5b2ee,NvAPI_D3D12_GetCudaSurfaceObject)
        NR_API_CASE(0x70c07832,NvAPI_D3D12_IsFatbinPTXSupported)
        NR_API_CASE(0xad1a677d,NvAPI_D3D12_CreateCuModule)
        NR_API_CASE(0x7ab88d88,NvAPI_D3D12_EnumFunctionsInModule)
        NR_API_CASE(0xe2436e22,NvAPI_D3D12_CreateCuFunction)
        NR_API_CASE(0x41c65285,NvAPI_D3D12_DestroyCuModule)
        NR_API_CASE(0xdf295ea6,NvAPI_D3D12_DestroyCuFunction)
        NR_API_CASE(0x7fb785ba,NvAPI_D3D12_DestroyCubinComputeShader)
        case 0x24973538:name="NvAPI_D3D12_LaunchCuKernelChain";break;
        case 0x846a9bf0:name="NvAPI_D3D12_LaunchCuKernelChainEx";break;
    }
#undef NR_API_CASE
    if(first)DemoCommandProbe::Event({{"kind","native_api_resolve"},{"id",id},{"name",name},
        {"available",function!=nullptr},{"wrapped",wrapped}});
    return result;
}
}
