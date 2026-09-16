// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace student {
void check(HRESULT hr);
Microsoft::WRL::ComPtr<ID3D12Resource> buffer(ID3D12Device* device, uint64_t size,
    D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void barrier(ID3D12GraphicsCommandList* list);

class Runtime {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    static constexpr uint32_t Width=1920, Height=1080;
    static constexpr uint64_t ImageBytes=uint64_t(Width)*Height*8;
    struct Slot {
        Microsoft::WRL::ComPtr<ID3D12Resource> input, output, temporary;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
        Microsoft::WRL::ComPtr<IUnknown> binding;
    };
    // Initialization uses a private queue and completes before returning.
    Runtime(ID3D12Device* device, const std::filesystem::path& model);
    ~Runtime();
    std::unique_ptr<Slot> makeSlot();
    void dispatch(ID3D12GraphicsCommandList* list, Slot& slot);
    // Input enters/leaves NON_PIXEL_SHADER_RESOURCE; output enters/leaves UAV.
    bool record(ID3D12GraphicsCommandList* list, Slot& slot, ID3D12Resource* input, ID3D12Resource* output);
    uint64_t temporaryBytes() const;
    uint64_t persistentBytes() const;
};
}
