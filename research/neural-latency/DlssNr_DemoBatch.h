#pragma once
// SPDX-License-Identifier: GPL-3.0-only
// Original bounded demo experiment: retain arguments and flush at command boundaries.
#include "DlssNr_DemoCommandProbe.h"
#include <type_traits>
#include <cstdlib>

namespace DlssNr::DemoBatch
{
using LaunchFn=decltype(&NvAPI_D3D12_LaunchCuKernelChain);
struct Chunk
{
    std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS> calls;
    std::vector<std::vector<unsigned char>> arguments;
    Chunk(){calls.reserve(8);arguments.reserve(8);}
};
inline std::recursive_mutex mutex;
inline std::filesystem::path reportPath;
inline bool configured=false,active=false,selected=false,limited=false;
inline std::atomic<bool> wanted=false;
inline thread_local bool draining=false;
inline DWORD owner=0;
inline unsigned frame=0,row=0;
inline ID3D12GraphicsCommandList* recording=nullptr;
inline LaunchFn original=nullptr;
inline std::unique_ptr<Chunk> pending;
inline std::vector<std::unique_ptr<Chunk>> retained;
inline size_t retainedBytes=0;
inline nlohmann::json report;
inline unsigned requested=0,submitted=0,apiCalls=0,fallbackCalls=0;
inline UINT64 totalRequested=0,totalSubmitted=0,totalApiCalls=0;
inline void Save() noexcept
{
    try
    {
        report["requested_kernels"]=totalRequested;report["submitted_kernels"]=totalSubmitted;
        report["real_api_calls"]=totalApiCalls;report["retained_argument_bytes"]=retainedBytes;
        report["retained_chunks"]=retained.size();
        std::ofstream output(reportPath);output<<report.dump(2)<<'\n';
    }catch(...){}
}
[[noreturn]] inline void Fail(const char* reason)
{
    report["failure"]=reason;report["complete_through_last_evaluation"]=false;Save();
    // Only armed in the guarded, separate sample. Stop before additional GPU
    // recording if a deferred call fails; never pretend it executed successfully.
    TerminateProcess(GetCurrentProcess(),0xE0500001u);std::abort();
}
inline void Flush(const char* reason,unsigned slot=0)
{
    if(!pending || pending->calls.empty())return;
    if(GetCurrentThreadId()!=owner)Fail("pending commands reached another thread");
    auto chunk=std::move(pending);const auto count=unsigned(chunk->calls.size());
    auto* calls=chunk->calls.data();retained.push_back(std::move(chunk));
    draining=true;++DemoCommandProbe::launchDepth;
    const auto status=original(recording,calls,count);
    --DemoCommandProbe::launchDepth;draining=false;
    ++apiCalls;++totalApiCalls;
    if(selected)report["sampled_evaluations"][row]["flushes"].push_back({{"reason",reason},{"method_slot",slot},{"count",count},{"status",status}});
    if(status!=NVAPI_OK)Fail("native multi-kernel API call failed");
    submitted+=count;totalSubmitted+=count;
}
inline void Boundary(ID3D12GraphicsCommandList* commands,unsigned slot)
{
    if(!wanted.load() || draining)return;
    std::lock_guard lock(mutex);
    if(!active || !pending)return;
    if(GetCurrentThreadId()!=owner || commands!=recording)Fail("unmodeled command thread or command list");
    Flush("command boundary",slot);
}
inline void Configure(const std::filesystem::path& directory)
{
    if(configured)return;configured=true;
    if(!DemoCommandProbe::enabled || !std::filesystem::exists(directory/L"nr-batch-kernels.enable"))return;
    reportPath=directory/L"nr-batch-report.json";if(std::filesystem::exists(reportPath))return;
    wanted=true;
    report={{"schema",1},{"batch_limit",8},{"retention_limit_bytes",64u*1024*1024},
        {"target_achieved",false},{"quality_gate_passed",false},{"native_runtime_accelerated",false},
        {"complete_through_last_evaluation",false},{"sampled_evaluations",nlohmann::json::array()},
        {"status_scope","Queued requests may return NVAPI_OK before submission; only real flush statuses establish API acceptance."}};
    Save();
}
inline void Begin(ID3D12GraphicsCommandList* commands,unsigned index,bool sample,bool guards)
{
    if(!wanted)return;std::lock_guard lock(mutex);
    if(active || pending)Fail("nested or unfinished evaluation");
    if(!guards || limited){report["guard_disabled"]=true;Save();return;}
    recording=commands;owner=GetCurrentThreadId();frame=index;selected=sample;active=true;
    requested=submitted=apiCalls=fallbackCalls=0;
    if(selected)
    {
        row=unsigned(report["sampled_evaluations"].size());
        report["sampled_evaluations"].push_back({{"frame",frame},{"flushes",nlohmann::json::array()}});
    }
}
inline NvAPI_Status Submit(ID3D12GraphicsCommandList* commands,const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels,unsigned count,LaunchFn call)
{
    if(!wanted.load())return call(commands,kernels,count);
    std::lock_guard lock(mutex);
    if(!active)return call(commands,kernels,count);
    if(GetCurrentThreadId()!=owner || commands!=recording)Fail("unmodeled kernel thread or command list");
    if(original && original!=call)Fail("native launch function changed");
    original=call;requested+=count;totalRequested+=count;
    if(count!=1 || !kernels || !kernels[0].pParams || !kernels[0].paramSize || kernels[0].paramSize>4096 ||
       retainedBytes+kernels[0].paramSize>64u*1024*1024)
    {
        Flush("unsupported request");++fallbackCalls;
        if(retainedBytes>=64u*1024*1024-4096)limited=true;
        draining=true;++DemoCommandProbe::launchDepth;
        const auto status=call(commands,kernels,count);
        --DemoCommandProbe::launchDepth;draining=false;
        ++apiCalls;++totalApiCalls;if(status!=NVAPI_OK)Fail("synchronous fallback failed");
        submitted+=count;totalSubmitted+=count;return status;
    }
    if(pending && pending->calls.size()==8)Flush("batch limit");
    if(!pending)pending=std::make_unique<Chunk>();
    const auto& k=kernels[0];pending->arguments.emplace_back(k.paramSize);
    auto& bytes=pending->arguments.back();memcpy(bytes.data(),k.pParams,k.paramSize);
    if(memcmp(bytes.data(),k.pParams,k.paramSize))Fail("argument copy changed");
    static_assert(std::is_trivially_copyable_v<NVAPI_CU_KERNEL_LAUNCH_PARAMS>);
    pending->calls.emplace_back();memcpy(&pending->calls.back(),&k,sizeof(k));
    pending->calls.back().pParams=bytes.data();retainedBytes+=k.paramSize;
    // This is queue acceptance, not a fabricated claim about an API/GPU result.
    return NVAPI_OK;
}
inline void BeforeExtended()
{
    if(!wanted)return;std::lock_guard lock(mutex);
    if(active){Flush("extended API boundary");report["extended_api_seen"]=true;}
}
inline void End()
{
    if(!wanted)return;std::lock_guard lock(mutex);if(!active)return;
    Flush("evaluation end");if(submitted!=requested)Fail("kernel accounting mismatch");
    if(selected)
    {
        auto& r=report["sampled_evaluations"][row];r["requested_kernels"]=requested;r["submitted_kernels"]=submitted;
        r["real_api_calls"]=apiCalls;r["fallback_calls"]=fallbackCalls;r["complete"]=true;
    }
    active=false;recording=nullptr;report["last_completed_evaluation"]=frame;report["complete_through_last_evaluation"]=true;
    if(selected || frame%64==0)Save();
}
}
