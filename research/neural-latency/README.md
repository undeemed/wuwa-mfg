# Isolated neural-model research

These developer tools investigate the [latency target and current findings](../../docs/neural-latency-research.md).
They are not an installer, and do not provide a 3 ms renderer. The existing
[hidden NVIDIA demo](../../docs/neural-demo-benchmark.md) remains the application
benchmark. Python experiments are for numerical checks and model development.

## Model experiments

Tested with Windows, Python 3.12, PyTorch 2.7.1+cu128 and an RTX 4070 Ti (SM89).
Create a separate environment; PyTorch's CUDA wheel is several GB. No CUDA toolkit
or system driver changes were needed for the local kernel experiment.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install numpy safetensors==0.8.0 pillow
git clone https://github.com/iamwavecut/MLX-DLSS.git MLX-DLSS
git -C MLX-DLSS checkout 0ca2deab092fe6f3e331bf4f616271dbc64521d0
New-Item -ItemType Directory local-weights,results
```

Read the upstream license and recovery notes. Using your own local runtime, run
the pinned source's data extractors; the toolkit does not download or redistribute
weights. Use a path outside a public repository for the two tensor files.

```powershell
.venv\Scripts\python MLX-DLSS\python\mlxdlss\tools\extract_dlssnr_weights.py D:\LocalRuntime\nvngx_dlssnr.dll local-weights\packed.safetensors
.venv\Scripts\python MLX-DLSS\python\mlxdlss\tools\unpack_dlssnr_weights.py local-weights\packed.safetensors local-weights\logical.safetensors
.venv\Scripts\python study_model.py --source MLX-DLSS --weights local-weights\logical.safetensors --output results\audit.json --mode audit
.venv\Scripts\python study_model.py --source MLX-DLSS --weights local-weights\logical.safetensors --output results\rounding.json --mode rounding
.venv\Scripts\python test_fused_norm.py --source MLX-DLSS --output results\fused-norm.json
.venv\Scripts\python study_model.py --source MLX-DLSS --weights local-weights\logical.safetensors --output results\graph.json --mode graph --fused-norm
```

Run GPU experiments sequentially with the demo and game closed. The scripts exit
after each bounded test and release their GPU allocations. They create no window.
`fused_norm.py` compiles the included `.cu` source with the NVRTC DLL already in
the PyTorch environment and loads that kernel into its own process. It does not
modify NVIDIA's DLL or load code into another process.

`study_model.py --mode model --skip 31` reproduces the rejected block-removal
experiment. Only ordinary blocks can be skipped; transition/input/output blocks
are excluded. Its default fixture is a deterministic 320×320 synthetic image.
`--image` accepts a local image but resizes it to that same small smoke-test size.
Neither option constitutes full-resolution or temporal quality validation.

CUDA Graph mode hoists immutable attention-bias permutations out of inference.
That cache is not suitable for training mutable weights. The custom kernel is
inference-only and supports 32-channel FP16/FP32 tensors on Windows SM89. The
tests cover the final partial thread group and noncontiguous layouts. Passing
them does not prove every floating-point boundary case or vendor parity.

## Runtime kernel tracing in the existing demo

[`optiscaler-demo-kernel-probe.patch`](optiscaler-demo-kernel-probe.patch) is an
**incremental research patch** on top of
[`patches/optiscaler-wuwa-compat.patch`](../../patches/optiscaler-wuwa-compat.patch).
The integration is GPL-3.0 under the OptiScaler license. It is not applied by the
normal installer or source-build script.

Use a separate source build produced by `BuildNeural.ps1`, with its pinned
v0.8.4 source and DirectX headers. Apply this incremental patch with `git apply
--check` followed by `git apply`. Rebuild Release x64 using that build's
`local-build.props` through `ForceImportBeforeCppTargets`, and set
`PostBuildEventUseInBuild=false`. This was tested with VS 2022 and Windows SDK
10.0.22621; omitting the DirectX header override caused compilation errors.

Put this research DLL only in the separate demo as `dxgi.dll`, keeping the
original demo DLL backed up. The observer has two gates: process basename must
be `ngx_dlss_demo.exe`, and `nr-kernel-probe.enable` must exist beside the NR DLL.
It records function names, dimensions and status without reading weights, kernel
arguments or GPU buffers. Adding `nr-kernel-timing.enable` also enables fenced
GPU timestamp queries. It does not split a launch chain.

Use the existing guarded demo runner for a 60-second, 60-FPS test. Preserve
`nr-kernel-probe.csv`, `nr-kernel-timing.csv`, the run result and its log before
another run. Then restore the original demo DLL and remove the two marker files.
Do not deploy the trace build to WuWa.

```powershell
python summarize_kernel_trace.py nr-kernel-timing.csv results\kernels.json --width 1920 --height 1080
```

Only supply width/height after checking the model extent in the corresponding
OptiScaler log. The analyzer validates complete successful frames and consistent
launch order, then groups GPU intervals. It discards evaluation 600 by default
because that first timed evaluation allocates the query resources. Architecture
selection remains unknown when NVAPI receives fatbins rather than bare ELF
objects. Instrumented timings are diagnostic, not final speed claims.

No vendor binaries, tensor files, generated GPU objects or raw captures belong
in this directory. See [third-party notices](../../THIRD_PARTY_NOTICES.md).
