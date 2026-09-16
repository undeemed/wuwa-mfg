// SPDX-License-Identifier: Apache-2.0
// Headless D3D12 correctness/timing harness. Never opens a window or game process.
#define NOMINMAX
#include "StudentRuntime.h"
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <nlohmann/json.hpp>
using namespace student;
using Microsoft::WRL::ComPtr;

int wmain(int argc,wchar_t** argv) {
    ComPtr<ID3D12Device> device;
    try {
        if (argc<3) throw std::runtime_error("Usage: StudentProbe model-directory input.bin [output.bin]");
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSettings;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dredSettings)))) {
            dredSettings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
        ComPtr<IDXGIFactory6> factory; check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter; check(factory->EnumAdapterByGpuPreference(0,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter)));
        check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue1> info; DWORD cookie=0;
        if (SUCCEEDED(device.As(&info))) info->RegisterMessageCallback(
            [](D3D12_MESSAGE_CATEGORY,D3D12_MESSAGE_SEVERITY severity,D3D12_MESSAGE_ID,const char* text,void*) {
                if (severity<=D3D12_MESSAGE_SEVERITY_WARNING) std::cerr<<text<<std::endl;
            },D3D12_MESSAGE_CALLBACK_FLAG_NONE,nullptr,&cookie);
        auto begin=std::chrono::steady_clock::now();
        std::cerr<<"Compiling and initializing student graph"<<std::endl;
        Runtime runtime(device.Get(),argv[1]); auto slot=runtime.makeSlot();
        auto compileMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::cerr<<"Ready: "<<compileMs<<" ms; temporary "<<runtime.temporaryBytes()<<" bytes"<<std::endl;
        ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC q{}; q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12CommandAllocator> allocator; check(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> list; check(device->CreateCommandList(0,q.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
        ComPtr<ID3D12Fence> fence; check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr); if (!event) throw std::runtime_error("Event failed");
        uint64_t sequence=0;
        auto execute=[&]() {
            check(list->Close()); ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
            check(queue->Signal(fence.Get(),++sequence)); check(fence->SetEventOnCompletion(sequence,event));
            if (WaitForSingleObject(event,30000)!=WAIT_OBJECT_0) throw std::runtime_error("Headless probe timed out");
            check(device->GetDeviceRemovedReason());
            check(allocator->Reset()); check(list->Reset(allocator.Get(),nullptr));
        };
        auto upload=buffer(device.Get(),Runtime::ImageBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* address{}; D3D12_RANGE empty{0,0}; check(upload->Map(0,&empty,&address));
        std::ifstream input(argv[2],std::ios::binary);
        if (!input.read(static_cast<char*>(address),Runtime::ImageBytes) || input.peek()!=EOF) throw std::runtime_error("Invalid input size");
        upload->Unmap(0,nullptr);
        transition(list.Get(),slot->input.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(slot->input.Get(),0,upload.Get(),0,Runtime::ImageBytes);
        transition(list.Get(),slot->input.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        execute();
        for (int i=0;i<8;++i) runtime.dispatch(list.Get(),*slot); execute();
        const int count=48;
        D3D12_QUERY_HEAP_DESC desc{}; desc.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; desc.Count=count*2;
        ComPtr<ID3D12QueryHeap> queries; check(device->CreateQueryHeap(&desc,IID_PPV_ARGS(&queries)));
        auto timings=buffer(device.Get(),count*2*sizeof(uint64_t),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        for (int i=0;i<count;++i) {
            list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,2*i);
            runtime.dispatch(list.Get(),*slot);
            list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,2*i+1);
        }
        list->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,count*2,timings.Get(),0); execute();
        uint64_t frequency{}; check(queue->GetTimestampFrequency(&frequency));
        uint64_t* data{}; check(timings->Map(0,nullptr,reinterpret_cast<void**>(&data)));
        std::vector<double> ms; for (int i=0;i<count;++i) ms.push_back(double(data[2*i+1]-data[2*i])*1000/frequency);
        timings->Unmap(0,&empty); std::sort(ms.begin(),ms.end());
        auto output=buffer(device.Get(),Runtime::ImageBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        transition(list.Get(),slot->output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(output.Get(),0,slot->output.Get(),0,Runtime::ImageBytes); execute();
        if (argc>3) {
            check(output->Map(0,nullptr,&address)); std::ofstream file(argv[3],std::ios::binary);
            file.write(static_cast<const char*>(address),Runtime::ImageBytes); output->Unmap(0,&empty);
        }
        check(list->Close()); CloseHandle(event);
        if (info) info->UnregisterMessageCallback(cookie);
        std::cout<<nlohmann::json({{"complete",true},{"median_ms",(ms[23]+ms[24])/2},
            {"p95_ms",ms[45]},{"compile_ms",compileMs},{"temporary_bytes",runtime.temporaryBytes()},
            {"persistent_bytes",runtime.persistentBytes()},{"samples",count}}).dump()<<std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<std::endl;
        if (device) std::cerr<<"Device status: 0x"<<std::hex<<uint32_t(device->GetDeviceRemovedReason())<<std::endl;
        ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
        if (device && SUCCEEDED(device.As(&dred))) {
            D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 crumbs{};
            if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&crumbs))) for(auto node=crumbs.pHeadAutoBreadcrumbNode;node;node=node->pNext)
                std::cerr<<"Breadcrumb "<<(node->pCommandListDebugNameA?node->pCommandListDebugNameA:"")<<" "<<(node->pLastBreadcrumbValue?*node->pLastBreadcrumbValue:0)<<" / "<<node->BreadcrumbCount<<std::endl;
            D3D12_DRED_PAGE_FAULT_OUTPUT1 fault{};
            if (SUCCEEDED(dred->GetPageFaultAllocationOutput1(&fault))) std::cerr<<"Page fault VA 0x"<<fault.PageFaultVA<<std::endl;
        }
        return 1;
    } catch (...) { std::cerr<<"Native DirectML graph creation failed"<<std::endl; return 2; }
}
