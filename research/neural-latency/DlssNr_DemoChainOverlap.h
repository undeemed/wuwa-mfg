#pragma once
// SPDX-License-Identifier: GPL-3.0-only
// Original bounded rendezvous experiment; included inside DemoChainSelftest.
inline nlohmann::json Overlap(decltype(&NvAPI_D3D12_CreateCuFunction) createFunction,
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch)
{
    nlohmann::json rows=nlohmann::json::array();
    const auto require=[](bool good,const char* where){if(!good)throw std::runtime_error(where);};
    NVDX_ObjectHandle clear{},wait{},signal{};
    require(createFunction(state->device.Get(),state->module,"chain_flag_clear",&clear)==NVAPI_OK,"clear function");
    require(createFunction(state->device.Get(),state->module,"chain_wait_bounded",&wait)==NVAPI_OK,"wait function");
    require(createFunction(state->device.Get(),state->module,"chain_signal",&signal)==NVAPI_OK,"signal function");
    const auto flag=state->a->GetGPUVirtualAddress(),result=state->b->GetGPUVirtualAddress();
    // Prior selftest iteration restored the UAV states and completed its fence.
    for(unsigned iteration=0;iteration<10;++iteration)for(unsigned mode=0;mode<3;++mode)
    {
        require(SUCCEEDED(state->allocator->Reset()) && SUCCEEDED(state->commands->Reset(state->allocator.Get(),nullptr)),"overlap reset");
        auto* args=new std::array<std::array<unsigned char,24>,3>{};
        auto* calls=new std::array<NVAPI_CU_KERNEL_LAUNCH_PARAMS,3>{};
        const UINT64 cycles=4000000;
        for(unsigned i=0;i<3;++i)
        {
            auto& call=(*calls)[i];call.hFunction=i==0?clear:i==1?wait:signal;
            call.gridDim={1,1,1};call.blockDim={32,1,1};call.pParams=(*args)[i].data();call.paramSize=i==1?24:8;
            memcpy((*args)[i].data(),&flag,8);
            if(i==1){memcpy((*args)[i].data()+8,&result,8);memcpy((*args)[i].data()+16,&cycles,8);}
        }
        std::vector<int> statuses;statuses.push_back(launch(state->commands.Get(),calls->data(),1));
        D3D12_RESOURCE_BARRIER u{};u.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;state->commands->ResourceBarrier(1,&u);
        if(mode==2)statuses.push_back(launch(state->commands.Get(),calls->data()+1,2));
        else
        {
            statuses.push_back(launch(state->commands.Get(),calls->data()+1,1));
            if(mode==0)state->commands->ResourceBarrier(1,&u);
            statuses.push_back(launch(state->commands.Get(),calls->data()+2,1));
        }
        for(auto status:statuses)require(status==NVAPI_OK,"overlap launch");
        state->commands->ResourceBarrier(1,&u);
        D3D12_RESOURCE_BARRIER transition{};transition.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition={state->b.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
        state->commands->ResourceBarrier(1,&transition);
        state->commands->CopyBufferRegion(state->readback.Get(),0,state->b.Get(),0,4);
        std::swap(transition.Transition.StateBefore,transition.Transition.StateAfter);state->commands->ResourceBarrier(1,&transition);
        require(SUCCEEDED(state->commands->Close()),"overlap close");ID3D12CommandList* lists[]={state->commands.Get()};state->queue->ExecuteCommandLists(1,lists);
        const auto value=++state->fenceValue;require(SUCCEEDED(state->queue->Signal(state->fence.Get(),value)),"overlap signal");
        if(state->fence->GetCompletedValue()<value)
        {
            require(SUCCEEDED(state->fence->SetEventOnCompletion(value,state->event)),"overlap event");
            require(WaitForSingleObject(state->event,5000)==WAIT_OBJECT_0,"overlap fence timeout");
        }
        require(state->fence->GetCompletedValue()!=UINT64_MAX && state->fence->GetCompletedValue()>=value,"overlap GPU completion");
        unsigned* mapped=nullptr;D3D12_RANGE range{0,4};require(SUCCEEDED(state->readback->Map(0,&range,reinterpret_cast<void**>(&mapped))),"overlap map");
        const unsigned observed=*mapped;D3D12_RANGE none{0,0};state->readback->Unmap(0,&none);require(observed<=1,"overlap output range");
        rows.push_back({{"iteration",iteration},{"mode",mode==0?"serial_barriers":mode==1?"serial_without_barriers":"batched"},
            {"producer_observed_before_deadline",observed==1},{"deadline_cycles",cycles},{"launch_statuses",statuses},{"gpu_completed",true}});
        delete calls;delete args;
    }
    return rows;
}
