#pragma once
// SPDX-License-Identifier: GPL-3.0-only
// Original demo-only dependency test. No model kernels, tensors or arguments.
#include <d3d12.h>
#include <nvapi.h>
#include <wrl/client.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace DlssNr::DemoChainSelftest
{
using Microsoft::WRL::ComPtr;
struct State
{
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> a,b,readback;
    HANDLE event=nullptr;
    UINT64 fenceValue=0;
    NVDX_ObjectHandle module{},fill{},mix{};
    std::vector<unsigned char> cubin;
};
// Retain resources/module/arguments until process exit, including timeout paths.
inline State* state=nullptr;
inline bool attempted=false;
inline std::filesystem::path directory;
inline void Configure(const std::filesystem::path& path) {directory=path;}
inline void Run(ID3D12GraphicsCommandList* parent,
    decltype(&NvAPI_D3D12_CreateCuModule) createModule,
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction,
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch) noexcept
{
    if(attempted || directory.empty() || !createModule || !createFunction || !launch ||
       !std::filesystem::exists(directory/L"nr-chain-selftest.enable"))return;
    attempted=true;
    nlohmann::json report={{"schema",1},{"complete",false},{"original_workload_only",true},
        {"native_kernels_modified",false},{"target_achieved",false},{"quality_gate_passed",false},
        {"cases",nlohmann::json::array()}};
    const auto output=directory/L"nr-chain-selftest.json";
    if(std::filesystem::exists(output))return;
    const auto save=[&]() {std::ofstream file(output);file<<report.dump(2)<<'\n';};
    const auto require=[&](bool good,const char* where) {if(!good)throw std::runtime_error(where);};
    try
    {
        wchar_t executable[32768]{};
        require(GetModuleFileNameW(nullptr,executable,(DWORD)std::size(executable))!=0,"executable");
        require(!_wcsicmp(std::filesystem::path(executable).filename().c_str(),L"ngx_dlss_demo.exe"),"demo only");
        state=new State;
        const auto cubinPath=directory/L"nr-chain-selftest.cubin";
        const auto bytes=std::filesystem::file_size(cubinPath);
        require(bytes==4456,"unexpected original test cubin size");
        state->cubin.resize(size_t(bytes));std::ifstream blob(cubinPath,std::ios::binary);
        require(bool(blob.read(reinterpret_cast<char*>(state->cubin.data()),bytes)),"read test cubin");
        require(SUCCEEDED(parent->GetDevice(IID_PPV_ARGS(&state->device))),"GetDevice");
        const auto moduleStatus=createModule(state->device.Get(),state->cubin.data(),NvU32(bytes),&state->module);
        report["module_status"]=moduleStatus;require(moduleStatus==NVAPI_OK,"CreateCuModule");
        const auto fillStatus=createFunction(state->device.Get(),state->module,"chain_fill",&state->fill);
        const auto mixStatus=createFunction(state->device.Get(),state->module,"chain_mix",&state->mix);
        report["function_status"]={fillStatus,mixStatus};require(fillStatus==NVAPI_OK && mixStatus==NVAPI_OK,"CreateCuFunction");
        D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        require(SUCCEEDED(state->device->CreateCommandQueue(&q,IID_PPV_ARGS(&state->queue))),"queue");
        require(SUCCEEDED(state->device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&state->allocator))),"allocator");
        require(SUCCEEDED(state->device->CreateCommandList(0,q.Type,state->allocator.Get(),nullptr,IID_PPV_ARGS(&state->commands))),"commands");
        require(SUCCEEDED(state->commands->Close()),"initial close");
        require(SUCCEEDED(state->device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&state->fence))),"fence");
        state->event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(state->event!=nullptr,"event");
        constexpr unsigned capacity=262144;
        const auto buffer=[&](D3D12_HEAP_TYPE type,ComPtr<ID3D12Resource>& result)
        {
            D3D12_HEAP_PROPERTIES heap{};heap.Type=type;
            D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;desc.Width=capacity*4;
            desc.Height=1;desc.DepthOrArraySize=1;desc.MipLevels=1;desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags=type==D3D12_HEAP_TYPE_DEFAULT?D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS:D3D12_RESOURCE_FLAG_NONE;
            require(SUCCEEDED(state->device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
                type==D3D12_HEAP_TYPE_DEFAULT?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,IID_PPV_ARGS(&result))),"buffer");
        };
        for(unsigned count:{1024u,65536u,262144u})for(unsigned kernels:{2u,3u,8u,17u})
        {
            constexpr unsigned seed=74123;
            std::vector<unsigned> expected(count),next(count);
            for(unsigned i=0;i<count;++i)expected[i]=(i*1664525u+seed)^0x12345678u;
            for(unsigned stage=1;stage<kernels;++stage)
            {
                for(unsigned i=0;i<count;++i)next[i]=expected[(i*17u+13u)&(count-1u)]*22695477u+stage*1009u;
                expected.swap(next);
            }
            nlohmann::json row={{"elements",count},{"kernels",kernels}};
            for(bool batched:{false,true})
            {
                // Fresh resources give both modes the same explicit initial
                // states; no assumption about state decay between submissions.
                buffer(D3D12_HEAP_TYPE_DEFAULT,state->a);buffer(D3D12_HEAP_TYPE_DEFAULT,state->b);buffer(D3D12_HEAP_TYPE_READBACK,state->readback);
                require(SUCCEEDED(state->allocator->Reset()) && SUCCEEDED(state->commands->Reset(state->allocator.Get(),nullptr)),"reset");
                // Fully owned argument arrays remain alive until the fence completes.
                auto* args=new std::vector<std::array<unsigned char,24>>(kernels);
                auto* calls=new std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS>(kernels);
                const auto addressA=state->a->GetGPUVirtualAddress(),addressB=state->b->GetGPUVirtualAddress();
                for(unsigned i=0;i<kernels;++i)
                {
                    auto& arg=(*args)[i];auto& call=(*calls)[i];
                    call.hFunction=i?state->mix:state->fill;call.gridDim={(count+255)/256,1,1};call.blockDim={256,1,1};
                    call.pParams=arg.data();call.paramSize=i?24:16;
                    if(!i){memcpy(arg.data(),&addressA,8);memcpy(arg.data()+8,&count,4);memcpy(arg.data()+12,&seed,4);}
                    else
                    {
                        const auto input=i%2?addressA:addressB,outputAddress=i%2?addressB:addressA;const auto salt=i*1009u;
                        memcpy(arg.data(),&input,8);memcpy(arg.data()+8,&outputAddress,8);memcpy(arg.data()+16,&count,4);memcpy(arg.data()+20,&salt,4);
                    }
                }
                std::vector<int> statuses;
                if(batched)statuses.push_back(launch(state->commands.Get(),calls->data(),kernels));
                else for(unsigned i=0;i<kernels;++i)
                {
                    statuses.push_back(launch(state->commands.Get(),calls->data()+i,1));
                    D3D12_RESOURCE_BARRIER u{};u.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;state->commands->ResourceBarrier(1,&u);
                }
                const char* mode=batched?"batched":"serial_barriers";row[mode]["launch_statuses"]=statuses;
                for(auto status:statuses)require(status==NVAPI_OK,"LaunchCuKernelChain");
                auto* finalResource=kernels%2?state->a.Get():state->b.Get();
                D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition={finalResource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
                state->commands->ResourceBarrier(1,&barrier);
                state->commands->CopyBufferRegion(state->readback.Get(),0,finalResource,0,count*4);
                std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter);state->commands->ResourceBarrier(1,&barrier);
                require(SUCCEEDED(state->commands->Close()),"close");ID3D12CommandList* lists[]={state->commands.Get()};state->queue->ExecuteCommandLists(1,lists);
                const auto value=++state->fenceValue;require(SUCCEEDED(state->queue->Signal(state->fence.Get(),value)),"signal");
                if(state->fence->GetCompletedValue()<value)
                {
                    require(SUCCEEDED(state->fence->SetEventOnCompletion(value,state->event)),"event completion");
                    require(WaitForSingleObject(state->event,5000)==WAIT_OBJECT_0,"fence timeout");
                }
                require(state->fence->GetCompletedValue()!=UINT64_MAX && state->fence->GetCompletedValue()>=value,"GPU completed");
                unsigned* mapped=nullptr;D3D12_RANGE range{0,count*4};require(SUCCEEDED(state->readback->Map(0,&range,reinterpret_cast<void**>(&mapped))),"map");
                unsigned mismatches=0;for(unsigned i=0;i<count;++i)mismatches+=mapped[i]!=expected[i];
                D3D12_RANGE written{0,0};state->readback->Unmap(0,&written);
                row[mode]["mismatched_elements"]=mismatches;row[mode]["gpu_completed"]=true;
                // No producer resources or argument storage are released early.
                delete calls;delete args;
            }
            report["cases"].push_back(row);save();
        }
        report["complete"]=true;
    }
    catch(const std::exception& e){report["failure"]=e.what();}
    catch(...){report["failure"]="unknown observer failure";}
    try{save();}catch(...){}
}
}
