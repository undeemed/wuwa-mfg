#pragma once
// SPDX-License-Identifier: GPL-3.0-only
// Original demo-only replay of a fixed, private four-frame input sequence.
#include <d3d12.h>
#include <wrl/client.h>
#include <json.hpp>
#include <filesystem>
#include <fstream>
#include <array>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace DlssNr::DemoInputReplay
{
using Json=nlohmann::json;
using Resource=Microsoft::WRL::ComPtr<ID3D12Resource>;
inline std::filesystem::path directory;
inline bool enabled=false;
inline std::array<std::array<Resource,3>,4> uploads;
inline void Configure(const std::filesystem::path& root)
{
    enabled=std::filesystem::exists(root/L"nr-input-replay.enable");
    directory=root/L"nr-input-replay";
}
template<class Shots>
inline bool Apply(ID3D12GraphicsCommandList* commands,ID3D12Device* device,unsigned index,const Json& controls,Shots& shots,Json& result) noexcept
{
    if(!enabled)return true;
    result={{"enabled",true},{"applied",false},{"source_frame",index}};
    try
    {
        const auto require=[](bool good,const char* reason){if(!good)throw std::runtime_error(reason);};
        require(index<4,"frame bound");
        std::ifstream metadataFile(directory/("frame-"+std::to_string(index)+".json"));
        Json metadata;metadataFile>>metadata;
        require(metadata.at("complete").get<bool>() && metadata.at("gpu_completed").get<bool>(),"source not fenced");
        require(metadata.at("controls")==controls,"control mismatch");
        const char* roles[]={"color","depth","motion"};
        // Prepare and validate every upload before recording any GPU copy.
        for(unsigned i=0;i<3;++i)
        {
            auto& shot=shots[i];const auto desc=shot.source->GetDesc();const auto& m=metadata.at("resources").at(roles[i]);
            require(desc.Width==m.at("width").get<UINT64>() && desc.Height==m.at("height").get<UINT>() &&
                unsigned(desc.Format)==m.at("format").get<unsigned>() && shot.rows==m.at("rows").get<UINT>() &&
                shot.rowBytes==m.at("row_bytes").get<UINT64>(),"resource layout mismatch");
            const auto name=std::string(roles[i])+"-"+std::to_string(index)+".raw";
            require(m.at("file")==name,"unexpected input filename");
            const auto file=directory/name;const auto bytes=std::filesystem::file_size(file);
            require(bytes==shot.rowBytes*shot.rows && bytes>0 && bytes<=64ull*1024*1024 && shot.bytes<=64ull*1024*1024,"input byte bound");
            require(!uploads[index][i],"input already replayed");
            std::vector<unsigned char> data(size_t(bytes),0);std::ifstream input(file,std::ios::binary);
            require(bool(input.read(reinterpret_cast<char*>(data.data()),std::streamsize(bytes))),"input read");
            D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;buffer.Width=shot.bytes;
            buffer.Height=1;buffer.DepthOrArraySize=1;buffer.MipLevels=1;buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            require(SUCCEEDED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,IID_PPV_ARGS(&uploads[index][i]))),"create upload");
            void* mapped=nullptr;D3D12_RANGE readRange{0,0};
            require(SUCCEEDED(uploads[index][i]->Map(0,&readRange,&mapped)),"map upload");
            memset(mapped,0,size_t(shot.bytes));
            for(UINT row=0;row<shot.rows;++row)
                memcpy(static_cast<unsigned char*>(mapped)+shot.layout.Offset+size_t(row)*shot.layout.Footprint.RowPitch,
                    data.data()+size_t(row)*size_t(shot.rowBytes),size_t(shot.rowBytes));
            D3D12_RANGE written{0,size_t(shot.bytes)};uploads[index][i]->Unmap(0,&written);
        }
        for(unsigned i=0;i<3;++i)
        {
            auto& shot=shots[i];D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition={shot.source.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST};
            commands->ResourceBarrier(1,&barrier);
            D3D12_TEXTURE_COPY_LOCATION destination{};destination.pResource=shot.source.Get();destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION source{};source.pResource=uploads[index][i].Get();source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;source.PlacedFootprint=shot.layout;
            commands->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
            std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter);commands->ResourceBarrier(1,&barrier);
        }
        result["applied"]=true;result["input_resources"]=3;return true;
    }
    catch(const std::exception& e){result["failure"]=e.what();return false;}
    catch(...){result["failure"]="unknown input replay failure";return false;}
}
}
