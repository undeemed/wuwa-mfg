#pragma once
// SPDX-License-Identifier: GPL-3.0-only
// Original bounded demo observer. Never logs pointers or packed arguments.
#include <d3d12.h>
#include <nvapi.h>
#include <wrl/client.h>
#include <detours/detours.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <atomic>
#include <array>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstring>
#include <stdexcept>

namespace DlssNr::DemoCommandProbe
{
inline bool configured=false,enabled=false,attached=false;
inline std::atomic<bool> coverageGap=false;
inline std::recursive_mutex mutex;
inline std::ofstream output;
inline unsigned events=0,methodLimit=0;
inline std::atomic<unsigned> activeFrame=0,activeChain=0,ownerThread=0;
inline std::atomic<void*> activeCommands=nullptr;
inline thread_local unsigned launchDepth=0,methodDepth=0;
inline std::array<void*,86> targets{};
inline std::unordered_map<void*,unsigned> attachedTargets;
inline size_t retainedBytes=0;
struct OwnedChain
{
    std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS> calls;
    std::vector<std::vector<unsigned char>> arguments;
};
// Deliberately retained through process exit; no asynchronous driver lifetime
// assumption is made in this small observer. GPU resources are not copied.
inline std::vector<std::unique_ptr<OwnedChain>> ownedChains;
inline std::unordered_map<const void*,std::vector<unsigned char>> lastArguments;
inline void Event(nlohmann::json record) noexcept
{
    try
    {
        std::lock_guard lock(mutex);
        if(!enabled)return;
        if(events>=32768){coverageGap=true;return;}
        record["event"]=++events;record["frame"]=activeFrame.load();record["after_chain"]=activeChain.load();
        record["on_evaluation_thread"]=ownerThread.load()==GetCurrentThreadId();
        record["inside_launch"]=launchDepth!=0;output<<record.dump()<<'\n';output.flush();
    }
    catch(...){coverageGap=true;}
}
inline void Configure(const std::filesystem::path& directory)
{
    if(configured)return;configured=true;
    wchar_t path[32768]{};
    if(!GetModuleFileNameW(nullptr,path,(DWORD)std::size(path)) ||
       _wcsicmp(std::filesystem::path(path).filename().c_str(),L"ngx_dlss_demo.exe") ||
       !std::filesystem::exists(directory/L"nr-command-probe.enable"))return;
    const auto file=directory/L"nr-command-probe.jsonl";if(std::filesystem::exists(file))return;
    output.open(file);enabled=bool(output);
    Event({{"kind","configuration"},{"schema",1},{"event_cap",32768},{"owned_byte_cap",64u*1024*1024},
        {"scope","known command-list implementations, selected evaluations, owned packed CPU arguments only"}});
}
inline void Note(unsigned slot,void* commands) noexcept
{
    if(!activeFrame.load())return;
    Event({{"kind","command"},{"slot",slot},{"nested_method",methodDepth!=0},
        {"same_command_list",commands==activeCommands.load()}});
}
template<unsigned Slot,typename Signature>struct Hook;
template<unsigned Slot,typename R,typename C,typename... Args>
struct Hook<Slot,R(STDMETHODCALLTYPE C::*)(Args...)>
{
    using Fn=R(STDMETHODCALLTYPE*)(C*,Args...);
    inline static Fn original=nullptr;
    static R STDMETHODCALLTYPE Call(C* object,Args... args)
    {
        Note(Slot,object);
        struct Depth {Depth(){++methodDepth;}~Depth(){--methodDepth;}} depth;
        return original(object,args...);
    }
};
template<unsigned Slot,typename Signature>
inline void Bind(void** table,const char* name)
{
    if(Slot>=methodLimit)return;
    using H=Hook<Slot,Signature>;
    const auto target=table[Slot];targets[Slot]=target;
    if(attachedTargets.contains(target))
    {
        coverageGap=true;Event({{"kind","hook"},{"slot",Slot},{"method",name},
            {"status","shared implementation omitted"},{"shared_with",attachedTargets.at(target)}});return;
    }
    H::original=reinterpret_cast<typename H::Fn>(target);
    auto result=DetourTransactionBegin();
    if(result==NO_ERROR)
    {
        result=DetourUpdateThread(GetCurrentThread());
        if(result==NO_ERROR)result=DetourAttach(reinterpret_cast<PVOID*>(&H::original),H::Call);
        if(result==NO_ERROR)result=DetourTransactionCommit();else DetourTransactionAbort();
    }
    if(result!=NO_ERROR){coverageGap=true;H::original=nullptr;}
    else attachedTargets[target]=Slot;
    Event({{"kind","hook"},{"slot",Slot},{"method",name},{"status",result}});
}
inline void Attach(ID3D12GraphicsCommandList* commands)
{
    if(!enabled)return;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList10> latest;
    const auto status=commands->QueryInterface(IID_PPV_ARGS(&latest));
    if(FAILED(status))
    {
        coverageGap=true;Event({{"kind","unsupported_interface"},{"interface",10},{"status",status}});return;
    }
    auto** table=*reinterpret_cast<void***>(latest.Get());
    if(attached)
    {
        for(unsigned i=9;i<86;++i)if(i!=26 && i!=80 && targets[i]!=table[i])
        {coverageGap=true;Event({{"kind","changed_implementation"},{"slot",i}});}
        return;
    }
    attached=true;methodLimit=86;
    // Bindings are checked against the installed DirectX SDK's vtable order.
    // Existing research hooks record ResourceBarrier and Barrier separately.
    Bind<9,decltype(&ID3D12GraphicsCommandList10::Close)>(table,"Close");
    Bind<10,decltype(&ID3D12GraphicsCommandList10::Reset)>(table,"Reset");
    Bind<11,decltype(&ID3D12GraphicsCommandList10::ClearState)>(table,"ClearState");
    Bind<12,decltype(&ID3D12GraphicsCommandList10::DrawInstanced)>(table,"DrawInstanced");
    Bind<13,decltype(&ID3D12GraphicsCommandList10::DrawIndexedInstanced)>(table,"DrawIndexedInstanced");
    Bind<14,decltype(&ID3D12GraphicsCommandList10::Dispatch)>(table,"Dispatch");
    Bind<15,decltype(&ID3D12GraphicsCommandList10::CopyBufferRegion)>(table,"CopyBufferRegion");
    Bind<16,decltype(&ID3D12GraphicsCommandList10::CopyTextureRegion)>(table,"CopyTextureRegion");
    Bind<17,decltype(&ID3D12GraphicsCommandList10::CopyResource)>(table,"CopyResource");
    Bind<18,decltype(&ID3D12GraphicsCommandList10::CopyTiles)>(table,"CopyTiles");
    Bind<19,decltype(&ID3D12GraphicsCommandList10::ResolveSubresource)>(table,"ResolveSubresource");
    Bind<20,decltype(&ID3D12GraphicsCommandList10::IASetPrimitiveTopology)>(table,"IASetPrimitiveTopology");
    Bind<21,decltype(&ID3D12GraphicsCommandList10::RSSetViewports)>(table,"RSSetViewports");
    Bind<22,decltype(&ID3D12GraphicsCommandList10::RSSetScissorRects)>(table,"RSSetScissorRects");
    Bind<23,decltype(&ID3D12GraphicsCommandList10::OMSetBlendFactor)>(table,"OMSetBlendFactor");
    Bind<24,decltype(&ID3D12GraphicsCommandList10::OMSetStencilRef)>(table,"OMSetStencilRef");
    Bind<25,decltype(&ID3D12GraphicsCommandList10::SetPipelineState)>(table,"SetPipelineState");
    Bind<27,decltype(&ID3D12GraphicsCommandList10::ExecuteBundle)>(table,"ExecuteBundle");
    Bind<28,decltype(&ID3D12GraphicsCommandList10::SetDescriptorHeaps)>(table,"SetDescriptorHeaps");
    Bind<29,decltype(&ID3D12GraphicsCommandList10::SetComputeRootSignature)>(table,"SetComputeRootSignature");
    Bind<30,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRootSignature)>(table,"SetGraphicsRootSignature");
    Bind<31,decltype(&ID3D12GraphicsCommandList10::SetComputeRootDescriptorTable)>(table,"SetComputeRootDescriptorTable");
    Bind<32,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRootDescriptorTable)>(table,"SetGraphicsRootDescriptorTable");
    Bind<33,decltype(&ID3D12GraphicsCommandList10::SetComputeRoot32BitConstant)>(table,"SetComputeRoot32BitConstant");
    Bind<34,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRoot32BitConstant)>(table,"SetGraphicsRoot32BitConstant");
    Bind<35,decltype(&ID3D12GraphicsCommandList10::SetComputeRoot32BitConstants)>(table,"SetComputeRoot32BitConstants");
    Bind<36,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRoot32BitConstants)>(table,"SetGraphicsRoot32BitConstants");
    Bind<37,decltype(&ID3D12GraphicsCommandList10::SetComputeRootConstantBufferView)>(table,"SetComputeRootConstantBufferView");
    Bind<38,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRootConstantBufferView)>(table,"SetGraphicsRootConstantBufferView");
    Bind<39,decltype(&ID3D12GraphicsCommandList10::SetComputeRootShaderResourceView)>(table,"SetComputeRootShaderResourceView");
    Bind<40,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRootShaderResourceView)>(table,"SetGraphicsRootShaderResourceView");
    Bind<41,decltype(&ID3D12GraphicsCommandList10::SetComputeRootUnorderedAccessView)>(table,"SetComputeRootUnorderedAccessView");
    Bind<42,decltype(&ID3D12GraphicsCommandList10::SetGraphicsRootUnorderedAccessView)>(table,"SetGraphicsRootUnorderedAccessView");
    Bind<43,decltype(&ID3D12GraphicsCommandList10::IASetIndexBuffer)>(table,"IASetIndexBuffer");
    Bind<44,decltype(&ID3D12GraphicsCommandList10::IASetVertexBuffers)>(table,"IASetVertexBuffers");
    Bind<45,decltype(&ID3D12GraphicsCommandList10::SOSetTargets)>(table,"SOSetTargets");
    Bind<46,decltype(&ID3D12GraphicsCommandList10::OMSetRenderTargets)>(table,"OMSetRenderTargets");
    Bind<47,decltype(&ID3D12GraphicsCommandList10::ClearDepthStencilView)>(table,"ClearDepthStencilView");
    Bind<48,decltype(&ID3D12GraphicsCommandList10::ClearRenderTargetView)>(table,"ClearRenderTargetView");
    Bind<49,decltype(&ID3D12GraphicsCommandList10::ClearUnorderedAccessViewUint)>(table,"ClearUnorderedAccessViewUint");
    Bind<50,decltype(&ID3D12GraphicsCommandList10::ClearUnorderedAccessViewFloat)>(table,"ClearUnorderedAccessViewFloat");
    Bind<51,decltype(&ID3D12GraphicsCommandList10::DiscardResource)>(table,"DiscardResource");
    Bind<52,decltype(&ID3D12GraphicsCommandList10::BeginQuery)>(table,"BeginQuery");
    Bind<53,decltype(&ID3D12GraphicsCommandList10::EndQuery)>(table,"EndQuery");
    Bind<54,decltype(&ID3D12GraphicsCommandList10::ResolveQueryData)>(table,"ResolveQueryData");
    Bind<55,decltype(&ID3D12GraphicsCommandList10::SetPredication)>(table,"SetPredication");
    Bind<56,decltype(&ID3D12GraphicsCommandList10::SetMarker)>(table,"SetMarker");
    Bind<57,decltype(&ID3D12GraphicsCommandList10::BeginEvent)>(table,"BeginEvent");
    Bind<58,decltype(&ID3D12GraphicsCommandList10::EndEvent)>(table,"EndEvent");
    Bind<59,decltype(&ID3D12GraphicsCommandList10::ExecuteIndirect)>(table,"ExecuteIndirect");
    Bind<60,decltype(&ID3D12GraphicsCommandList10::AtomicCopyBufferUINT)>(table,"AtomicCopyBufferUINT");
    Bind<61,decltype(&ID3D12GraphicsCommandList10::AtomicCopyBufferUINT64)>(table,"AtomicCopyBufferUINT64");
    Bind<62,decltype(&ID3D12GraphicsCommandList10::OMSetDepthBounds)>(table,"OMSetDepthBounds");
    Bind<63,decltype(&ID3D12GraphicsCommandList10::SetSamplePositions)>(table,"SetSamplePositions");
    Bind<64,decltype(&ID3D12GraphicsCommandList10::ResolveSubresourceRegion)>(table,"ResolveSubresourceRegion");
    Bind<65,decltype(&ID3D12GraphicsCommandList10::SetViewInstanceMask)>(table,"SetViewInstanceMask");
    Bind<66,decltype(&ID3D12GraphicsCommandList10::WriteBufferImmediate)>(table,"WriteBufferImmediate");
    Bind<67,decltype(&ID3D12GraphicsCommandList10::SetProtectedResourceSession)>(table,"SetProtectedResourceSession");
    Bind<68,decltype(&ID3D12GraphicsCommandList10::BeginRenderPass)>(table,"BeginRenderPass");
    Bind<69,decltype(&ID3D12GraphicsCommandList10::EndRenderPass)>(table,"EndRenderPass");
    Bind<70,decltype(&ID3D12GraphicsCommandList10::InitializeMetaCommand)>(table,"InitializeMetaCommand");
    Bind<71,decltype(&ID3D12GraphicsCommandList10::ExecuteMetaCommand)>(table,"ExecuteMetaCommand");
    Bind<72,decltype(&ID3D12GraphicsCommandList10::BuildRaytracingAccelerationStructure)>(table,"BuildRaytracingAccelerationStructure");
    Bind<73,decltype(&ID3D12GraphicsCommandList10::EmitRaytracingAccelerationStructurePostbuildInfo)>(table,"EmitRaytracingAccelerationStructurePostbuildInfo");
    Bind<74,decltype(&ID3D12GraphicsCommandList10::CopyRaytracingAccelerationStructure)>(table,"CopyRaytracingAccelerationStructure");
    Bind<75,decltype(&ID3D12GraphicsCommandList10::SetPipelineState1)>(table,"SetPipelineState1");
    Bind<76,decltype(&ID3D12GraphicsCommandList10::DispatchRays)>(table,"DispatchRays");
    Bind<77,decltype(&ID3D12GraphicsCommandList10::RSSetShadingRate)>(table,"RSSetShadingRate");
    Bind<78,decltype(&ID3D12GraphicsCommandList10::RSSetShadingRateImage)>(table,"RSSetShadingRateImage");
    Bind<79,decltype(&ID3D12GraphicsCommandList10::DispatchMesh)>(table,"DispatchMesh");
    Bind<81,decltype(&ID3D12GraphicsCommandList10::OMSetFrontAndBackStencilRef)>(table,"OMSetFrontAndBackStencilRef");
    Bind<82,decltype(&ID3D12GraphicsCommandList10::RSSetDepthBias)>(table,"RSSetDepthBias");
    Bind<83,decltype(&ID3D12GraphicsCommandList10::IASetIndexBufferStripCutValue)>(table,"IASetIndexBufferStripCutValue");
    Bind<84,decltype(&ID3D12GraphicsCommandList10::SetProgram)>(table,"SetProgram");
    Bind<85,decltype(&ID3D12GraphicsCommandList10::DispatchGraph)>(table,"DispatchGraph");
}
inline void Begin(ID3D12GraphicsCommandList* commands,unsigned frame,bool selected)
{
    if(!enabled)return;Attach(commands);
    if(!selected)return;
    ownerThread=GetCurrentThreadId();activeCommands=commands;activeChain=0;activeFrame=frame;
    Event({{"kind","evaluation_enter"},{"known_interface",10},{"coverage_gap",coverageGap.load()}});
}
inline void End()
{
    if(!activeFrame.load())return;
    Event({{"kind","evaluation_exit"},{"coverage_gap",coverageGap.load()},{"retained_argument_bytes",retainedBytes},
        {"retained_chains",ownedChains.size()},{"method_depth",methodDepth}});
    activeFrame=0;activeCommands=nullptr;
}
struct LaunchScope
{
    bool sampled=false;
    LaunchScope(unsigned chain,unsigned count,bool extended)
    {
        sampled=activeFrame.load()!=0;
        if(sampled){activeChain=chain;Event({{"kind","launch_enter"},{"count",count},{"extended",extended}});}
        ++launchDepth;
    }
    ~LaunchScope(){--launchDepth;if(sampled)Event({{"kind","launch_exit"}});}
};
inline const NVAPI_CU_KERNEL_LAUNCH_PARAMS* CopyArguments(const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels,unsigned count) noexcept
{
    if(!enabled || !activeFrame.load())return kernels;
    try
    {
        std::lock_guard lock(mutex);
        if(!kernels || count<1 || count>512)throw std::runtime_error("unsupported chain size");
        size_t bytes=0;
        for(unsigned i=0;i<count;++i)
        {
            if(!kernels[i].pParams || !kernels[i].paramSize || kernels[i].paramSize>4096)
                throw std::runtime_error("unsupported packed arguments");
            bytes+=kernels[i].paramSize;
        }
        if(retainedBytes+bytes>64u*1024*1024)throw std::runtime_error("argument retention cap");
        auto owned=std::make_unique<OwnedChain>();owned->calls.assign(kernels,kernels+count);owned->arguments.resize(count);
        unsigned reused=0,changed=0;
        for(unsigned i=0;i<count;++i)
        {
            const auto& k=kernels[i];auto& a=owned->arguments[i];a.resize(k.paramSize);memcpy(a.data(),k.pParams,k.paramSize);
            if(memcmp(a.data(),k.pParams,k.paramSize))throw std::runtime_error("argument copy mismatch");
            const auto prior=lastArguments.find(k.pParams);
            if(prior!=lastArguments.end()){++reused;changed+=prior->second!=a;}
            if(lastArguments.size()<2048 || prior!=lastArguments.end())lastArguments[k.pParams]=a;
            owned->calls[i].pParams=a.data();
        }
        const auto result=owned->calls.data();ownedChains.push_back(std::move(owned));retainedBytes+=bytes;
        Event({{"kind","owned_arguments"},{"count",count},{"bytes",bytes},{"copy_exact",true},
            {"reused_source_storage",reused},{"reused_storage_changed",changed}});
        return result;
    }
    catch(const std::exception& e){coverageGap=true;Event({{"kind","arguments_forwarded_original"},{"reason",e.what()}});return kernels;}
    catch(...){coverageGap=true;Event({{"kind","arguments_forwarded_original"},{"reason","unknown copy failure"}});return kernels;}
}
}
