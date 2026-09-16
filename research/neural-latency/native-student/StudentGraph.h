// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <optional>
#include <cmath>
#include <iostream>
#include <DirectMLX.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

namespace student {
using dml::Expression;
using Dims = dml::TensorDimensions;
using Microsoft::WRL::ComPtr;
using Json = nlohmann::json;

struct Weight { uint32_t index; uint64_t offset, size; };
struct Model {
    ComPtr<IDMLCompiledOperator> op;
    std::vector<Weight> weights;
    std::vector<char> data;
    uint32_t inputs = 1;
};
inline std::string sha256(const std::vector<char>& bytes) {
    BCRYPT_ALG_HANDLE algorithm{};
    if (BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)
        throw std::runtime_error("SHA256 provider unavailable");
    unsigned char digest[32]{};
    auto status=BCryptHash(algorithm,nullptr,0,reinterpret_cast<PUCHAR>(const_cast<char*>(bytes.data())),
                           static_cast<ULONG>(bytes.size()),digest,sizeof(digest));
    BCryptCloseAlgorithmProvider(algorithm,0);
    if (status<0) throw std::runtime_error("SHA256 failed");
    const char* hex="0123456789abcdef"; std::string text;
    for(auto byte:digest) { text.push_back(hex[byte>>4]); text.push_back(hex[byte&15]); }
    return text;
}

inline Expression reshape(Expression x, Dims shape) {
    return dml::Reinterpret(x, std::move(shape), dml::NullOpt);
}
inline Expression permute(Expression x, const std::vector<uint32_t>& order) {
    auto desc = x.GetOutputDesc();
    Dims strides(desc.sizes.size()), shape;
    uint32_t step = 1;
    for (int i = int(strides.size())-1; i >= 0; --i) {
        strides[i] = step; step *= desc.sizes[i];
    }
    Dims targetStrides;
    for (auto i : order) { shape.push_back(desc.sizes.at(i)); targetStrides.push_back(strides[i]); }
    // Materialize the transpose before any following reshape.
    return dml::Identity(dml::Reinterpret(x, shape, targetStrides));
}
inline Expression slice(Expression x, uint32_t axis, uint32_t start, uint32_t count) {
    auto sizes = x.GetOutputDesc().sizes;
    Dims offsets(sizes.size(), 0); offsets[axis] = start; sizes[axis] = count;
    std::vector<int32_t> strides(sizes.size(), 1);
    return dml::Slice(x, offsets, sizes, strides);
}
inline Expression fp32(Expression x) { return dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32); }
inline Expression fp16(Expression x) { return dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT16); }
inline Expression broadcast(Expression x,const Dims& target) {
    const auto desc=x.GetOutputDesc();
    if (desc.sizes==target) return x;
    if (desc.sizes.size()!=target.size()) throw std::runtime_error("Broadcast rank mismatch");
    Dims strides(target.size()); uint32_t step=1;
    for (int i=int(target.size())-1;i>=0;--i) {
        if (desc.sizes[i]!=1 && desc.sizes[i]!=target[i]) throw std::runtime_error("Broadcast dimension mismatch");
        strides[i]=desc.sizes[i]==target[i]?step:0; step*=desc.sizes[i];
    }
    return dml::Reinterpret(x,target,strides);
}
inline Expression like(Expression x,Expression target) { return broadcast(x,target.GetOutputDesc().sizes); }
inline Expression up(Expression x) { return dml::Upsample2D(x, {2,2}, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR); }
inline Expression join(Expression a, Expression b, uint32_t axis) { return dml::Join({a,b}, axis); }

class Builder {
    dml::Graph graph;
    Json manifest;
    Model model;
    std::map<std::string, Expression> cache;
public:
    Builder(IDMLDevice* device, const std::filesystem::path& directory) : graph(device) {
        std::ifstream file(directory / "model.json"); file >> manifest;
        if (manifest.at("schema") != 1 || manifest.at("architecture") != "region-broad-v1" ||
            manifest.at("width") != 1920 || manifest.at("height") != 1080)
            throw std::runtime_error("Unsupported student model contract");
        std::ifstream weights(directory / "weights.bin", std::ios::binary | std::ios::ate);
        if (!weights || weights.tellg() <= 0 || weights.tellg() > 16*1024*1024)
            throw std::runtime_error("Invalid student weight file");
        model.data.resize(size_t(weights.tellg())); weights.seekg(0);
        if (!weights.read(model.data.data(), model.data.size())) throw std::runtime_error("Truncated weights");
        if (sha256(model.data)!=manifest.at("weights_sha256").get<std::string>())
            throw std::runtime_error("Student weights do not match the manifest; reload after export finishes");
    }
    Expression weight(const std::string& name, Dims shape = {}) {
        auto it = cache.find(name);
        if (it != cache.end()) {
            if (!shape.empty() && it->second.GetOutputDesc().sizes != shape)
                return reshape(it->second, shape);
            return it->second;
        }
        auto row = manifest.at("tensors").at(name);
        auto offset = row.at("offset").get<uint64_t>(), bytes = row.at("bytes").get<uint64_t>();
        if (offset > model.data.size() || bytes > model.data.size()-offset || offset%256)
            throw std::runtime_error("Invalid student tensor bounds");
        if (shape.empty()) { for (auto n : row.at("shape")) shape.push_back(n.get<uint32_t>()); }
        while (shape.size()<4) shape.insert(shape.begin(),1);
        std::string dtype = row.at("dtype");
        auto type = dtype=="float16" ? DML_TENSOR_DATA_TYPE_FLOAT16 :
                    dtype=="float32" ? DML_TENSOR_DATA_TYPE_FLOAT32 :
                    dtype=="uint32" ? DML_TENSOR_DATA_TYPE_UINT32 : DML_TENSOR_DATA_TYPE_UNKNOWN;
        uint64_t count=1; for (auto n:shape) count*=n;
        if (type==DML_TENSOR_DATA_TYPE_UNKNOWN || count*(type==DML_TENSOR_DATA_TYPE_FLOAT16?2:4)!=bytes)
            throw std::runtime_error("Student tensor shape/type mismatch");
        const uint32_t index=model.inputs++;
        auto value=dml::InputTensor(graph,index,dml::TensorDesc(type,DML_TENSOR_FLAG_OWNED_BY_DML,shape));
        model.weights.push_back({index,offset,bytes}); cache.emplace(name,value); return value;
    }
    Expression conv(Expression x, const std::string& name, uint32_t stride=1, uint32_t padding=0, uint32_t groups=1) {
        if (std::getenv("STUDENT_TRACE")) std::cerr<<"conv "<<name<<std::endl;
        auto w=weight(name+".weight"); auto channels=w.GetOutputDesc().sizes[0];
        auto b=weight(name+".bias",{1,channels,1,1});
        return dml::Convolution(x,w,b,DML_CONVOLUTION_MODE_CROSS_CORRELATION,DML_CONVOLUTION_DIRECTION_FORWARD,
                                {stride,stride},{1,1},{padding,padding},{padding,padding},{0,0},groups);
    }
    Expression linear(Expression x,const std::string& name) {
        if (std::getenv("STUDENT_TRACE")) std::cerr<<"linear "<<name<<std::endl;
        auto w=weight(name+".weight"); auto channels=w.GetOutputDesc().sizes[2];
        auto xs=x.GetOutputDesc().sizes, ws=w.GetOutputDesc().sizes;
        w=broadcast(w,{xs[0],xs[1],ws[2],ws[3]});
        auto b=broadcast(weight(name+".bias",{1,1,1,channels}),{xs[0],xs[1],xs[2],channels});
        return dml::Gemm(x,w,b,DML_MATRIX_TRANSFORM_NONE,DML_MATRIX_TRANSFORM_TRANSPOSE);
    }
    Expression norm(Expression x,const std::string& name) {
        if (std::getenv("STUDENT_TRACE")) std::cerr<<"norm "<<name<<std::endl;
        const auto channels=x.GetOutputDesc().sizes.back();
        auto y=fp32(x), w=fp32(weight(name+".weight",{1,1,1,channels})), b=fp32(weight(name+".bias",{1,1,1,channels}));
        return fp16(dml::MeanVarianceNormalization(y,like(w,y),like(b,y),{3},true,true,1e-5f));
    }
    Expression block(Expression x,const std::string& name) {
        auto n=x.GetOutputDesc().sizes[1];
        auto y=conv(x,name+".depthwise",1,1,n);
        y=conv(dml::ActivationGelu(conv(y,name+".expand")),name+".project");
        return x+y*like(weight(name+".scale"),y);
    }
    Expression blocks(Expression x,const std::string& name) { return block(block(x,name+".0"),name+".1"); }
    Expression condition(Expression pooled,const std::string& name) {
        auto x=reshape(pooled,{1,1,1,96});
        x=linear(dml::ActivationGelu(linear(norm(x,name+".norm"),name+".hidden")),name+".project");
        return reshape(x,{1,224,1,1});
    }
    Expression modulate(Expression x,Expression coefficients,uint32_t stage) {
        const uint32_t widths[]={16,32,64}, starts[]={0,32,96};
        auto scale=slice(coefficients,1,starts[stage],widths[stage]);
        auto shift=slice(coefficients,1,starts[stage]+widths[stage],widths[stage]);
        return x*like(scale+1.f,x)+like(shift,x);
    }
    Expression context(Expression source,const std::string& name) {
        auto padded=dml::Padding(source,DML_PADDING_MODE_CONSTANT,0,{0,0,0,0},{0,0,2,0});
        auto tokens=reshape(padded,{1,96,9,4,15,4});
        tokens=reshape(permute(tokens,{0,2,4,3,5,1}),{1,135,16,96});
        auto qkv=linear(norm(tokens,name+".norm"),name+".qkv");
        auto q=slice(qkv,3,0,48), k=slice(qkv,3,48,48), v=slice(qkv,3,96,48);
        auto valid=weight("constant.valid"), denominator=weight("constant.denominator");
        auto rq=dml::Reduce(fp32(q)*like(valid,q),DML_REDUCE_FUNCTION_SUM,{2});
        auto rk=dml::Reduce(fp32(k)*like(valid,k),DML_REDUCE_FUNCTION_SUM,{2});
        auto mq=reshape(rq/like(denominator,rq),{1,1,135,48});
        auto mk=reshape(rk/like(denominator,rk),{1,1,135,48});
        auto scores=dml::Gemm(mq,mk,dml::NullOpt,DML_MATRIX_TRANSFORM_NONE,DML_MATRIX_TRANSFORM_TRANSPOSE)/std::sqrt(48.f);
        auto other=dml::TopK(scores+weight("constant.diagonal"),3,3,DML_AXIS_DIRECTION_DECREASING).index;
        auto routes=join(weight("constant.own"),other,3);
        // Each query region retains all 16 tokens and selects four complete KV regions.
        auto gather=[&](Expression input,uint32_t channels) {
            input=reshape(input,{1,1,135,16*channels});
            return dml::Gather(input,routes,2,2); // [1,135,4,16*channels]
        };
        auto keys=permute(reshape(gather(k,48),{135,64,3,16}),{0,2,1,3});
        auto values=permute(reshape(gather(v,48),{135,64,3,16}),{0,2,1,3});
        auto queries=permute(reshape(q,{135,16,3,16}),{0,2,1,3});
        auto mask=reshape(gather(weight("constant.invalid"),1),{135,1,1,64});
        // Keep attention accumulation and softmax FP32; final output returns to FP16.
        auto logits=dml::Gemm(fp32(queries),fp32(keys),dml::NullOpt,DML_MATRIX_TRANSFORM_NONE,DML_MATRIX_TRANSFORM_TRANSPOSE)*.25f;
        auto probabilities=dml::ActivationSoftmax(logits+like(mask,logits),{3});
        auto result=fp16(dml::Gemm(probabilities,fp32(values)));
        result=reshape(permute(result,{0,2,1,3}),{1,135,16,48});
        auto update=linear(result,name+".output"); update=update*like(weight(name+".scale",{1,1,1,96}),update);
        update=reshape(update,{1,9,15,4,4,96});
        update=reshape(permute(update,{0,5,1,3,2,4}),{1,96,36,60});
        return source+slice(update,2,0,34);
    }
    Model build() {
        auto texture=dml::InputTensor(graph,0,dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT16,{1,1080,1920,4}));
        auto source=permute(slice(texture,3,0,3),{0,3,1,2});
        auto padded=dml::Padding(source,DML_PADDING_MODE_REFLECTION,0,{0,0,0,0},{0,0,8,0});
        // PyTorch pixel_unshuffle is channel, row, column order (CRD).
        auto pixels=dml::SpaceToDepth(padded,4,DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH);
        const std::string first="first.network", correction="refinement";
        auto value=conv(pixels,first+".stem");
        std::vector<Expression> encoded, decoded(3);
        for (int i=0;i<4;++i) {
            value=blocks(value,first+".encoder."+std::to_string(i)); encoded.push_back(value);
            if (i<3) value=conv(value,first+".down."+std::to_string(i),2);
        }
        auto pooled=dml::Reduce(value,DML_REDUCE_FUNCTION_AVERAGE,{2,3});
        auto modulation=condition(pooled,first+".decoder_conditioning");
        value=value*like(dml::ActivationTanh(conv(pooled,first+".context"))*.1f+1.f,value);
        for (int i=2;i>=0;--i) {
            value=conv(up(value),first+".up."+std::to_string(i))+encoded[i];
            value=blocks(modulate(value,modulation,i),first+".decoder."+std::to_string(i)); decoded[i]=value;
        }
        auto residual=dml::DepthToSpace(conv(value,first+".head"),4,DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH);
        auto base=dml::Clip(source+slice(residual,2,0,1080)*.25f,0,1);
        modulation=condition(dml::Reduce(encoded[3],DML_REDUCE_FUNCTION_AVERAGE,{2,3}),correction+".conditioning");
        value=context(conv(encoded[3],correction+".deep.0"),correction+".deep.1");
        for (int i=2;i>=0;--i) {
            auto skip=conv(join(encoded[i],decoded[i],1),correction+".skip."+std::to_string(i));
            value=up(conv(value,correction+".up."+std::to_string(i)))+skip;
            if (i==0) value=value+conv(pixels,correction+".local");
            value=blocks(modulate(value,modulation,i),correction+".blocks."+std::to_string(i));
        }
        residual=dml::DepthToSpace(conv(value,correction+".head"),4,DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH);
        auto output=fp32(dml::Clip(base+slice(residual,2,0,1080)*.25f,0,1));
        auto grade=manifest.at("grade").get<std::vector<float>>();
        if (grade.size()!=3) throw std::runtime_error("Invalid grade");
        output=dml::Clip(output*grade[0],0,1);
        auto delta=output*output*(3.f-2.f*output)-output;
        output=dml::Clip(output+grade[1]*delta,0,1);
        auto lightness=(dml::Reduce(output,DML_REDUCE_FUNCTION_MAX,{1})+dml::Reduce(output,DML_REDUCE_FUNCTION_MIN,{1}))*.5f;
        lightness=like(lightness,output);
        output=fp16(dml::Clip(grade[2]*(output-lightness)+lightness,0,1));
        output=dml::Padding(permute(output,{0,2,3,1}),DML_PADDING_MODE_CONSTANT,1,{0,0,0,0},{0,0,0,1});
        model.op=graph.Compile(DML_EXECUTION_FLAG_NONE,{output},model.inputs);
        return std::move(model);
    }
};
}
