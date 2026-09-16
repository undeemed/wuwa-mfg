// SPDX-License-Identifier: Apache-2.0
#define NOMINMAX
#include "StudentRuntime.h"
#include "StudentGraph.h"
#include <sstream>

namespace student {
void check(HRESULT hr) {
    if (FAILED(hr)) { std::ostringstream out; out<<"Student HRESULT 0x"<<std::hex<<uint32_t(hr); throw std::runtime_error(out.str()); }
}
ComPtr<ID3D12Resource> buffer(ID3D12Device* device,uint64_t size,D3D12_HEAP_TYPE heap,D3D12_RESOURCE_STATES state) {
    ComPtr<ID3D12Resource> result;
    if (!size) return result;
    D3D12_HEAP_PROPERTIES props{}; props.Type=heap;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width=(size+3)&~uint64_t(3); desc.Height=1; desc.DepthOrArraySize=1; desc.MipLevels=1;
    desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (heap==D3D12_HEAP_TYPE_DEFAULT) desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    check(device->CreateCommittedResource(&props,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(&result)));
    return result;
}
void transition(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    if (before==after) return;
    D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; list->ResourceBarrier(1,&b);
}
void barrier(ID3D12GraphicsCommandList* list) {
    D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; list->ResourceBarrier(1,&b);
}
static ComPtr<ID3D12DescriptorHeap> heap(ID3D12Device* device,uint32_t count) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{}; desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors=std::max(1u,count); desc.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> result; check(device->CreateDescriptorHeap(&desc,IID_PPV_ARGS(&result))); return result;
}
static ComPtr<IDMLBindingTable> table(IDMLDevice* device,IDMLDispatchable* target,ID3D12DescriptorHeap* heap,uint32_t count) {
    DML_BINDING_TABLE_DESC desc{target,heap->GetCPUDescriptorHandleForHeapStart(),heap->GetGPUDescriptorHandleForHeapStart(),std::max(1u,count)};
    ComPtr<IDMLBindingTable> result; check(device->CreateBindingTable(&desc,IID_PPV_ARGS(&result))); return result;
}
static DML_BUFFER_BINDING binding(ID3D12Resource* resource) { return {resource,0,resource?resource->GetDesc().Width:0}; }
static void bindStorage(IDMLBindingTable* table,ID3D12Resource* temporary,ID3D12Resource* persistent) {
    auto temp=binding(temporary), persist=binding(persistent);
    DML_BINDING_DESC t{temporary?DML_BINDING_TYPE_BUFFER:DML_BINDING_TYPE_NONE,temporary?&temp:nullptr};
    DML_BINDING_DESC p{persistent?DML_BINDING_TYPE_BUFFER:DML_BINDING_TYPE_NONE,persistent?&persist:nullptr};
    table->BindTemporaryResource(&t); table->BindPersistentResource(&p);
}
struct Runtime::Impl {
    struct Library { HMODULE value{}; ~Library(){ if(value) FreeLibrary(value); } } library;
    ComPtr<ID3D12Device> device;
    ComPtr<IDMLDevice> dml;
    ComPtr<IDMLCommandRecorder> recorder;
    Model model;
    DML_BINDING_PROPERTIES properties{};
    ComPtr<ID3D12Resource> persistent;
};
Runtime::~Runtime()=default;
Runtime::Runtime(ID3D12Device* device,const std::filesystem::path& path):impl(std::make_unique<Impl>()) {
    impl->device=device;
    const auto libraryPath=std::filesystem::absolute(path / "DirectML.dll");
    impl->library.value=LoadLibraryExW(libraryPath.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!impl->library.value) throw std::runtime_error("The private DirectML runtime is missing");
    using CreateDevice=HRESULT(WINAPI*)(ID3D12Device*,DML_CREATE_DEVICE_FLAGS,DML_FEATURE_LEVEL,REFIID,void**);
    auto create=reinterpret_cast<CreateDevice>(GetProcAddress(impl->library.value,"DMLCreateDevice1"));
    if (!create) throw std::runtime_error("DirectMLCreateDevice1 is unavailable");
    check(create(device,std::getenv("STUDENT_TRACE")?DML_CREATE_DEVICE_FLAG_DEBUG:DML_CREATE_DEVICE_FLAG_NONE,DML_FEATURE_LEVEL_6_0,IID_PPV_ARGS(&impl->dml)));
    impl->model=Builder(impl->dml.Get(),path).build();
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"graph compiled"<<std::endl;
    check(impl->dml->CreateCommandRecorder(IID_PPV_ARGS(&impl->recorder)));
    auto& model=impl->model; impl->properties=model.op->GetBindingProperties();
    impl->persistent=buffer(device,impl->properties.PersistentResourceSize);
    IDMLCompiledOperator* operators[]={model.op.Get()};
    ComPtr<IDMLOperatorInitializer> initializer;
    check(impl->dml->CreateOperatorInitializer(1,operators,IID_PPV_ARGS(&initializer)));
    auto props=initializer->GetBindingProperties();
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"initialize: persistent="<<impl->properties.PersistentResourceSize<<" temp="<<props.TemporaryResourceSize<<std::endl;
    auto descriptors=heap(device,props.RequiredDescriptorCount);
    auto bindings=table(impl->dml.Get(),initializer.Get(),descriptors.Get(),props.RequiredDescriptorCount);
    auto temporary=buffer(device,props.TemporaryResourceSize);
    if (temporary) {
        auto value=binding(temporary.Get()); DML_BINDING_DESC desc{DML_BINDING_TYPE_BUFFER,&value};
        bindings->BindTemporaryResource(&desc);
    }
    auto weights=buffer(device,model.data.size(),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
    auto upload=buffer(device,model.data.size(),D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    void* address{}; D3D12_RANGE empty{0,0}; check(upload->Map(0,&empty,&address));
    memcpy(address,model.data.data(),model.data.size()); upload->Unmap(0,nullptr);
    std::vector<DML_BUFFER_BINDING> weightBindings(model.inputs);
    for (auto w:model.weights) weightBindings[w.index]={weights.Get(),w.offset,w.size};
    DML_BUFFER_ARRAY_BINDING array{model.inputs,weightBindings.data()};
    DML_BINDING_DESC inputs{DML_BINDING_TYPE_BUFFER_ARRAY,&array}; bindings->BindInputs(1,&inputs);
    auto persistentBinding=binding(impl->persistent.Get());
    DML_BINDING_DESC output{impl->persistent?DML_BINDING_TYPE_BUFFER:DML_BINDING_TYPE_NONE,impl->persistent?&persistentBinding:nullptr};
    bindings->BindOutputs(1,&output);
    check(impl->dml->GetDeviceRemovedReason());
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"initializer bindings ready"<<std::endl;
    ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC q{}; q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator; check(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list; check(device->CreateCommandList(0,q.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
    list->CopyBufferRegion(weights.Get(),0,upload.Get(),0,model.data.size());
    transition(list.Get(),weights.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[]={descriptors.Get()}; list->SetDescriptorHeaps(1,heaps);
    impl->recorder->RecordDispatch(list.Get(),initializer.Get(),bindings.Get()); barrier(list.Get()); check(list->Close());
    check(impl->dml->GetDeviceRemovedReason());
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"initializer recorded"<<std::endl;
    ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"initializer submitted"<<std::endl;
    ComPtr<ID3D12Fence> fence; check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"fence created"<<std::endl;
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr); if (!event) throw std::runtime_error("Cannot create initialization event");
    check(queue->Signal(fence.Get(),1));
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"fence signaled"<<std::endl;
    check(fence->SetEventOnCompletion(1,event));
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"event registered"<<std::endl;
    // Do not release resources still referenced by the queue, even after a device error.
    DWORD wait;
    do { wait=WaitForSingleObject(event,1000); if (wait==WAIT_TIMEOUT) check(device->GetDeviceRemovedReason()); } while(wait==WAIT_TIMEOUT);
    CloseHandle(event);
    if (wait!=WAIT_OBJECT_0) throw std::runtime_error("Student initialization wait failed");
    check(device->GetDeviceRemovedReason()); model.data.clear(); model.data.shrink_to_fit();
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"initialization finished"<<std::endl;
}
std::unique_ptr<Runtime::Slot> Runtime::makeSlot() {
    auto slot=std::make_unique<Slot>(); auto& p=impl->properties; auto device=impl->device.Get();
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"slot allocating, temp="<<p.TemporaryResourceSize<<std::endl;
    slot->input=buffer(device,ImageBytes); slot->output=buffer(device,ImageBytes);
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"slot image buffers ready"<<std::endl;
    slot->temporary=buffer(device,p.TemporaryResourceSize); slot->heap=heap(device,p.RequiredDescriptorCount);
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"slot scratch and heap ready; descriptors="<<p.RequiredDescriptorCount<<std::endl;
    auto bindings=table(impl->dml.Get(),impl->model.op.Get(),slot->heap.Get(),p.RequiredDescriptorCount);
    if (std::getenv("STUDENT_TRACE")) std::cerr<<"slot table ready"<<std::endl;
    bindStorage(bindings.Get(),slot->temporary.Get(),impl->persistent.Get());
    auto in=binding(slot->input.Get()),out=binding(slot->output.Get());
    std::vector<DML_BINDING_DESC> inputs(impl->model.inputs,{DML_BINDING_TYPE_NONE,nullptr});
    inputs[0]={DML_BINDING_TYPE_BUFFER,&in}; bindings->BindInputs(uint32_t(inputs.size()),inputs.data());
    DML_BINDING_DESC output{DML_BINDING_TYPE_BUFFER,&out}; bindings->BindOutputs(1,&output);
    check(impl->dml->GetDeviceRemovedReason());
    slot->binding=bindings; return slot;
}
void Runtime::dispatch(ID3D12GraphicsCommandList* list,Slot& slot) {
    ComPtr<IDMLBindingTable> bindings; check(slot.binding.As(&bindings));
    ID3D12DescriptorHeap* heaps[]={slot.heap.Get()}; list->SetDescriptorHeaps(1,heaps);
    barrier(list); impl->recorder->RecordDispatch(list,impl->model.op.Get(),bindings.Get()); barrier(list);
    check(impl->dml->GetDeviceRemovedReason());
}
bool Runtime::record(ID3D12GraphicsCommandList* list,Slot& slot,ID3D12Resource* input,ID3D12Resource* output) {
    for (auto resource:{input,output}) {
        if (!resource) return false;
        auto desc=resource->GetDesc();
        if (desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width!=Width || desc.Height!=Height ||
            desc.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT || desc.SampleDesc.Count!=1) return false;
    }
    D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource=input; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION bufferIn{}; bufferIn.pResource=slot.input.Get(); bufferIn.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    bufferIn.PlacedFootprint.Footprint={DXGI_FORMAT_R16G16B16A16_FLOAT,Width,Height,1,Width*8};
    transition(list,input,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list,slot.input.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyTextureRegion(&bufferIn,0,0,0,&src,nullptr);
    transition(list,slot.input.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(list,input,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dispatch(list,slot);
    auto bufferOut=bufferIn; bufferOut.pResource=slot.output.Get();
    auto dst=src; dst.pResource=output;
    transition(list,slot.output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list,output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyTextureRegion(&dst,0,0,0,&bufferOut,nullptr);
    transition(list,output,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(list,slot.output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}
uint64_t Runtime::temporaryBytes() const { return impl->properties.TemporaryResourceSize; }
uint64_t Runtime::persistentBytes() const { return impl->properties.PersistentResourceSize; }
}
