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
.venv\Scripts\python test_fused_softmax.py --source MLX-DLSS --output results\fused-softmax.json
.venv\Scripts\python test_fused_gate.py --source MLX-DLSS --output results\fused-gate.json
.venv\Scripts\python test_fused_publish.py --source MLX-DLSS --output results\fused-publish.json
.venv\Scripts\python test_fused_roundtrip.py --output results\fused-roundtrip.json
.venv\Scripts\python study_model.py --source MLX-DLSS --weights local-weights\logical.safetensors --output results\graph.json --mode graph --fused-norm --fused-softmax
.venv\Scripts\python test_batched_ffn.py --source MLX-DLSS --weights local-weights\logical.safetensors --output results\batched-gate.json --fused-gate
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
That cache is not suitable for training mutable weights. The custom kernels are
inference-only on Windows SM89: normalization supports 32-channel FP16/FP32
tensors, and bit-affine softmax supports even row lengths 2–2048. Tests cover the
final partial thread group and noncontiguous layouts. Passing them does not prove
every floating-point boundary case, NaN behavior or vendor parity.

`test_batched_ffn.py` compares independent branch batching with the original
reference and measures full synthetic-model graphs after replay warmup. Its
FP32 tests deliberately expose rounding differences; `all_tested_values_equal`
is therefore false in the retained evidence. Only the tested FP16 path is used
by `compare_capture.py --batched-ffn`; FP32 falls back to the reference.
`--profile` additionally records one eager execution with PyTorch's profiler.
Its ATen and CUDA entries can overlap and must not be summed as disjoint costs.
`--fused-gate` adds the separate activation kernel after the batching comparison.

`FusedNorm.publish` combines strided normalization, per-head scaling and FP8
publication for `[batch,heads,tokens,32]` inference tensors. `FusedNorm.roundtrip`
combines saturation and FP8 conversion for FP16/FP32 tensors of rank at most 8,
without storing an intermediate FP8 tensor. Both use documented SM89 conversion
instructions. The tests compare against the prior operations, including strided
and saturation cases; signed-zero bits are checked in the general conversion.

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

## Matched input/output capture in the existing demo

[`optiscaler-demo-model-capture.patch`](optiscaler-demo-model-capture.patch) is a
separate GPL-3.0 research patch on top of the same R4 compatibility source. Apply
it to a separate source copy with `git apply --check` and `git apply`; use the
Release x64 build procedure and DirectX-header override described above. The
capture and kernel-probe patches have overlapping edit locations. For both at
once, use the combined launch-contract patch below instead of stacking them.
The normal installer applies none of these research patches.

Back up the demo's `dxgi.dll`, then place the capture build there. The process
must be named `ngx_dlss_demo.exe`, and `nr-model-capture.enable` must exist beside
the NR DLL. Preserve the guarded hidden-demo EXE and use its existing bounded
runner. The capture refuses to overwrite an existing `nr-model-capture` folder.
It captures four evaluations starting at the first reset, with a total readback
allocation cap of 512 MiB. Texture readback waits for actual GPU fence completion.
`complete: true`, `gpu_completed: true` and `evaluate_result: 1` in each manifest
are required before using that frame.

After the run, move the capture folder to private storage, disable the marker
and restore the original demo DLL. Keep this build out of WuWa. Raw captures
contain rendered scene data and are deliberately excluded from publication.
The local comparison tool reads the first frame without resizing its color:

```powershell
.venv\Scripts\python compare_capture.py --source MLX-DLSS --weights local-weights\logical.safetensors --capture D:\PrivateCaptures\natural --output results\natural --precision fast
.venv\Scripts\python compare_capture.py --source MLX-DLSS --weights local-weights\logical.safetensors --capture D:\PrivateCaptures\natural --output results\natural-graph --precision fast --batched-ffn --fused-gate --graph
.venv\Scripts\python compare_capture.py --source MLX-DLSS --weights local-weights\logical.safetensors --capture D:\PrivateCaptures\natural --output results\natural-fused --precision fast --batched-ffn --fused-gate --fused-publish --fused-roundtrip --graph --profile
```

`--inspect-only` validates the manifest and textures and writes previews without
running the model. `--precision reference` tests FP32 arithmetic;
`--network-height 1152` tests extra padding; `--noise-frame 1` tests an assumed
noise counter from 0–3. Those switches are diagnostics, not recommended fixes.
Use a fresh output directory for each run. The manifest does not expose the
internal vendor noise counter; a companion launch-contract trace now can.
This tool supports only a first-reset comparison
with preset 0 and RGBA16_FLOAT color/output textures, not temporal evaluation.
It writes numeric metrics plus local previews and a NumPy reconstruction; do not
commit the image data. `quality_gate_passed` remains false because a single frame
cannot establish the required visual and temporal quality.

`--graph` times fixed-input CUDA Graph replay after warmup at the full captured
extent. Its `network_timing_kind` distinguishes GPU intervals from the default
eager wall time. The graph result still excludes D3D12 integration. Both batching
and activation fusion are experimental options, not changes to the installed
NVIDIA runtime.

The two new fusions are opt-in. Compare the resulting local `reconstruction.npy`
against the prior graph on exactly the same captured input. `--profile` requires
`--graph` and records one warmed replay **after** the normal timing samples. Only
CUDA-device events are aggregated in `graph_kernel_profile`; host ATen events
are excluded to avoid double counting. Profile intervals remain instrumented
diagnostics. The full reconstructed graph is still much slower than NVIDIA.

## Combined native launch contract and captures

[`optiscaler-demo-launch-contract.patch`](optiscaler-demo-launch-contract.patch)
combines both research instruments and adds whitelisted scalar metadata. Apply
it directly on the normal R4 compatibility source, **not** on top of either
research patch above. It was built with the same pinned source, DirectX headers
and Release x64 procedure. Build the solution so `SolutionDir` resolves correctly;
building the project alone without that property caused missing-header errors.
The patch follows OptiScaler's GPL-3.0 license.

Only the separate, guarded hidden NVIDIA demo was used. Back up its `dxgi.dll`
before installing this research build there. Preserve the no-show EXE patch and
check that neither WuWa nor another demo process is running. Enable both
`nr-kernel-probe.enable` and `nr-model-capture.enable` beside the NR DLL. Leave
`nr-kernel-timing.enable` disabled for this experiment. Run the existing bounded
runner for 25 seconds at 1920×1080, 60 FPS, Natural style, preset 0 and masking on.
Archive `nr-launch-contract.jsonl`, `nr-kernel-probe.csv`, the completed capture
folder and the run result before another run.

The contract records evaluations 1–4, 64, 128 and 256. It includes launch names,
dimensions, argument sizes, API status and time inside the original native call.
Only the recognized 264-byte preprocessor layout has scalar fields decoded,
using the pinned MLX-DLSS parameter map. It does not dump pointers, arbitrary
packed arguments, GPU allocations or model weights. Launch chains are forwarded
unchanged. Native call durations are CPU submission times, not GPU timing.

A separate marker, `nr-force-sm89.enable`, opts into an architecture experiment:
retain exactly one existing SM89 cubin entry in each recognized module container.
The selector preserves that entry's complete bytes, accepts only zero trailing
padding and otherwise forwards the original module. It makes no on-disk NVIDIA
binary edits. A rejected filtered module is reported without silently retrying
another path. This is not a recommended optimization: the completed experiment
found no useful speedup. The older trial without padding support filtered zero
modules and must not be interpreted as testing architecture selection.

After each trial, restore the original demo DLL and disable all three markers.
Keep raw captures private and do not deploy the research DLL to WuWa. The
analyzer publishes only whitelisted metadata and texture-equality results:

```powershell
python summarize_launch_contract.py D:\PrivateTrials\baseline D:\PrivateTrials\sm89 --output results\native-contract.json
python test_sm89_selection.py
```

The selector test needs `cl` in a Visual Studio x64 developer shell. It extracts
the exact function from the combined patch and checks synthetic valid containers,
byte preservation, zero/nonzero padding, malformed sizes, missing and duplicate
architectures, and disabled operation. It never loads NVIDIA code. The analyzer
validates complete capture fences, corresponding scalar controls, consistent
launch geometry and chain order. It counts each chain's CPU duration only once.
Paired outputs are reported only while inputs and recorded history match from
reset. This limited static-scene check always leaves the full quality gate false.

## Bounded student training probes

`student_probe.py` trains a small residual CNN directly on local matched
color/output textures. It needs PyTorch and NumPy but does not load the recovered
model or its weights. Pixel-unshuffle retains every input pixel. The model is
trained with RGB and gradient losses, then measured as an FP16 CUDA Graph at the
unchanged 1920×1080 input. None of the retained students has passed the quality gate.

For the first probe, train on the left part of the original view and reserve a
separate right-hand region. To reproduce the original loss behavior, use zero
crop border:

```powershell
.venv\Scripts\python student_probe.py --capture D:\PrivateCaptures\natural --output results\student-one --steps 1500 --max-seconds 120 --width 32 --blocks 4 --loss-border 0
```

For the second probe, obtain two extra camera views with the same capture build
and controls. The sample loads `media/sponza.json`; back up its exact bytes before
editing the active `Camera0` target. Keep position `[0,1.8,0]` and up `[0,1,0]`.
Change only target `[1,1.8,0]` to `[-1,1.8,0]` for the second training view, or
`[0,1.8,-1]` for validation. For each, use the existing guarded runner at 1920×1080,
60 FPS, Natural style, preset 0 and automatic masking, for 25 seconds. Archive
the four completed fenced captures before the next run, and restore the original
scene bytes and demo DLL afterward. The existing hidden EXE must remain intact.
All three views are still the same scene.

```powershell
.venv\Scripts\python student_probe.py --capture D:\PrivateCaptures\natural --extra-train-capture D:\PrivateCaptures\west --validation-capture D:\PrivateCaptures\north --output results\student-views --steps 10000 --max-seconds 120 --width 48 --blocks 6 --loss-border 32
```

With a separate validation capture, training uses complete images from the two
training views. Controls and extents must match, and validation must have a
different input hash. The 32-pixel crop border keeps padding artifacts out of
the training loss. The report includes the unchanged-input baseline, holdout
errors, training steps and timing samples. These limited probes provide neither
temporal training nor cross-scene validation. Private checkpoints and NumPy
outputs must remain local; only source and numeric evidence are published.

Two follow-up students add deterministic noise conditioning and wider context:

```powershell
.venv\Scripts\python student_probe.py --capture D:\PrivateCaptures\natural --extra-train-capture D:\PrivateCaptures\west --validation-capture D:\PrivateCaptures\north --noise-source MLX-DLSS --output results\student-noise --width 48 --blocks 6 --steps 10000 --max-seconds 120 --loss-border 32
.venv\Scripts\python student_probe.py --capture D:\PrivateCaptures\natural --extra-train-capture D:\PrivateCaptures\west --validation-capture D:\PrivateCaptures\north --noise-source MLX-DLSS --output results\student-context --width 48 --blocks 6 --steps 10000 --max-seconds 240 --loss-border 96 --patch-size 256 --batch 4 --dilations 1,2,4,8,4,2 --cosine-lr
python analyze_student_error.py --capture D:\PrivateCaptures\north --prediction results\student-views\validation-output.npy --output results\student-spectrum.json
```

Noise is generated once from the pinned reference at counter zero, on the full
image coordinates, then cropped alongside RGB. Its generation and concatenation
are excluded from graph timing. The second configuration changes several
training choices together; do not attribute its result to dilation alone. Both
retained students failed the held-out quality check, and the wider-context model
also missed 3 ms. The frequency diagnostic reads private pixels but writes only
numeric band energies, with a Parseval consistency check. It is not a perceptual
or temporal test. Keep every trained checkpoint and rendered prediction private.

No vendor binaries, tensor files, generated GPU objects or raw captures belong
in this directory. See [third-party notices](../../THIRD_PARTY_NOTICES.md).

## Hierarchical student and additional teacher views

`--architecture hierarchical --whole-frame --batch 1` selects a four-level
encoder/decoder with skip connections and an image-wide context gate. A padded,
reversible pixel-unshuffle retains the source pixels; learned feature maps are
downsampled internally. This is an experimental model, not a claim that internal
compression preserves quality. The first 223,440-parameter model ran in 1.20 ms
but failed the held-out comparison.

```powershell
.venv\Scripts\python student_probe.py --capture D:\PrivateCaptures\natural --extra-train-capture D:\PrivateCaptures\west --validation-capture D:\PrivateCaptures\north --output results\student-hierarchical --architecture hierarchical --width 16 --blocks 2 --whole-frame --batch 1 --loss-border 0 --steps 1500 --max-seconds 240 --cosine-lr
```

`collect_demo_views.py` adds four diagonal training directions and a translated
north-facing validation camera, still within the same Sponza scene. It uses the
existing guarded hidden-demo runner, 20 seconds per view, Natural style, preset
0, masking enabled, and true 1920×1080 model inputs. The original scene bytes,
demo DLL and capture marker are restored in `finally`. The DLL must be a local
build of the published fenced texture-capture patch; supply its verified hash.
Other experimental markers must be disabled. Existing captures are never
overwritten. All pixel data and logs remain in the chosen private output folder.

```powershell
python collect_demo_views.py --demo-dir D:\PrivateDemo\bin\ngx_dlss_demo --output-dir D:\PrivateTrials --capture-dll D:\PrivateBuild\OptiScaler.capture.dll --capture-sha256 YOUR_LOCAL_BUILD_SHA256 --prefix teacher-natural
```

Append each of the four resulting training capture folders with
`--extra-train-capture`. Keep the original north view as `--validation-capture`
and the translated view as `--extra-validation-capture`; neither enters training.
Reports check distinct validation input hashes and record both holdouts.
More views within one scene still do not establish cross-scene or temporal
quality. Publish only the collection manifest and numeric results.

## Native buffer range metadata

[`optiscaler-demo-buffer-ranges.patch`](optiscaler-demo-buffer-ranges.patch)
includes the combined launch-contract/capture patch plus a bounded buffer
registry. Apply it directly to normal R4 source, not over another research patch.
It was built as Release x64 and its patch application/reversal was checked in a
scratch tree. It follows OptiScaler's GPL-3.0 license.

In the guarded hidden demo only, enable `nr-buffer-probe.enable` alongside
`nr-kernel-probe.enable` and `nr-model-capture.enable`. Keep architecture selection
and kernel timing disabled. Run the same 25-second native trial, then archive
`nr-buffer-probe.jsonl` with the other metadata/captures and restore the demo DLL
and markers. The buffer log refuses to overwrite an existing file.

The registry observes default-heap buffers through existing resource creation
hooks, only while direct NR calls are active. It retains at most 128 buffers,
512 MiB each, with a 1 GiB total bound to prevent stale address matches. Only two
known fields in the recognized 264-byte preprocessor arguments are matched to
these ranges: output and weights. Logs contain IDs, offsets, lengths and creation
states, never raw addresses or buffer contents. No new GPU commands are added.

The completed trial matched both pointers on all seven recorded evaluations.
This establishes allocation ownership, **not** tensor layout, current resource
state or intermediate numerical equivalence. Do not infer a safe readback state
from the recorded creation state. The underlying reconstruction remains
numerically different from the vendor. Results are in
[`native-buffer-ranges.json`](../../evidence/neural-model-research/native-buffer-ranges.json).

## Guarded native preprocessor capture

[`optiscaler-demo-pre-tensor.patch`](optiscaler-demo-pre-tensor.patch) extends the
combined research build with legacy/enhanced barrier observation and one optional
allocation-prefix capture. Apply it directly to normal R4 source. It includes the
earlier research patches; do not stack them. Build Release x64 with the existing
procedure. The patch follows OptiScaler's GPL-3.0 license.

For metadata only, enable the same three markers as the buffer-range trial.
The command-list hook observes the implementations presented during feature
creation/evaluation. It records at most 4,096 relevant barrier events. The
`before_chain` field is the most recently entered native call's CPU context;
it does not independently prove the next GPU operation. Resource creation,
split/aliasing transitions, enhanced barriers and implementation gaps inform a
conservative capture guard. This is not a general D3D12 state tracker.

The separate `nr-pre-tensor-capture.enable` marker opts into one **32 MiB** copy
from the known output pointer immediately after the recognized preprocessor on
evaluation 1. It requires the exact packed-argument size, 1920×1080 extent, noise
counter zero, a retained allocation large enough for the prefix, observed
whole-buffer UAV state, and no unsupported barrier/implementation conditions.
The source transitions UAV → COPY_SOURCE → UAV around the copy. Microsoft's
[buffer promotion/decay rules](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12#performance-implications)
allow a promotable BeforeState when a buffer has decayed to COMMON. This research
contract is specific to the observed runtime, not a license to guess arbitrary
resource states.

Readback occurs only after the submitting queue signals its fence and that fence
completes. Reset before submission discards the capture; failed signal/device
removal cannot authorize mapping. The allocation stays retained for process
lifetime. `nr-pre-tensor-capture/metadata.json` identifies completion; its raw
file remains private. An existing capture directory disables a new capture.

Run the existing guarded hidden demo for 25 seconds with Natural style, preset 0,
masking on and 60 FPS. Archive the texture captures, pre-tensor folder and all
metadata, then restore the demo DLL and disable all four experiment markers.
Architecture selection and per-kernel timing remain disabled. Do not install this
build in WuWa. The first two matched frames equaled the earlier baseline byte for
byte; later frames had unmatched inputs and were excluded.

`probe_pre_tensor_layout.py` compares the prefix against the reconstructed
adapter, first block and pooled outputs. The 1152-row hypothesis comes from the
native launch grid; it is not proof of logical extent. The tool searches declared
tile-axis permutations and optionally split channel axes. All 14,040 tested
candidates failed to establish a convincing match. The later lane-bit decoder
below resolves the captured skip prefix; the earlier axis search alone could
not localize an arithmetic error.

```powershell
.venv\Scripts\python probe_pre_tensor_layout.py --source MLX-DLSS --weights D:\PrivateWeights\logical.safetensors --trial D:\PrivateTrials\native-pre-tensor --output D:\PrivateResults\pre-layout --split-channels
```

The tool writes private reconstructed arrays plus numeric metrics. Publish only
metrics, never the prefix or arrays. See
[`native-pre-tensor.json`](../../evidence/neural-model-research/native-pre-tensor.json).

The hierarchical noise-conditioned follow-up uses the six-view width-16 command
above with `--noise-source MLX-DLSS`. It ran in 1.47 ms but worsened both holdouts;
it was rejected. Noise preparation and application integration are excluded from
that timing. Its full numeric record is
[`student-hierarchical-noise.json`](../../evidence/neural-model-research/student-hierarchical-noise.json).

## First-block lane layout and arithmetic

`decode_pre_tensor.py` implements the recovered **4×4 pixels × 32 channels**
layout. Each 512-byte tile contains 32 lanes writing 16 bytes apiece. With
physical offset `p = 16*lane + byte`, the logical index is `channel + 32*x + 128*y`.
Logical bits 0–8 select physical bits **[0, 4, 5, 1, 3, 6, 7, 8, 2]**.
Tiles advance horizontally with a pitch of 480 tiles at width 1920.

Disassembly narrowed the stores to 512-byte tiles. Correlation across 4,096
calibration tiles then assigned each physical slot to exactly one logical slot;
the 512 assignments reduce exactly to the bit permutation above. The decoder
exhaustively checks that the formula is a bijection. It was applied without
refitting to every captured byte on the original and west views. Both correlate
above 0.9989 with the reconstructed first block, with about 53% exact bytes.
This is strong layout evidence, **not numerical equality**. The prefix covers
544 full rows plus 1,024 pixels in each of the following four rows, not the whole
1152-row padded tensor. Pooled output uses a separate pointer and is not decoded.

```powershell
python decode_pre_tensor.py --trial D:\PrivateTrials\native-pre-tensor --reference D:\PrivateResults\pre-layout\block0-private.npy --output D:\PrivateResults\tile-validation.json
```

To reproduce the private binary inspection, obtain NVIDIA's standalone
`cuda_nvdisasm` and `cuda_cuobjdump` Windows packages, version 12.8.90, from the
[CUDA 12.8.1 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
The exact download URLs, sizes and archive hashes are retained in the numeric
evidence. These two packages total about 10.85 MB; a full toolkit installation
was not used. `inspect_pre_kernel.py` verifies the NR DLL, first compressed
fatbinary and extracted SM89 image hashes, finds the exact preprocessor symbol,
and invokes the tools without a window. The generated GPU code must stay private
and outside this repository. The tool does not modify its DLL input.

```powershell
python inspect_pre_kernel.py --dll D:\PrivateRuntime\nvngx_dlssnr.dll --cuobjdump D:\PrivateTools\cuobjdump.exe --nvdisasm D:\PrivateTools\nvdisasm.exe --output D:\PrivateResults\pre-inspection
```

The native first block converts branch inputs to E4M3 before the feed-forward
and QKV projections while retaining separate residual values. The recovered
reference omitted those conversions. `first_block_rounding.py` applies this
change **only to block 0** on one model instance. `probe_pre_tensor_arithmetic.py`
compares the unmodified arithmetic, GPU noise alone, each branch conversion and
both conversions against the same private prefixes:

| Variant | Original view MAE | West view MAE | Exact bytes, original / west |
| --- | ---: | ---: | ---: |
| Reference | 0.007632 | 0.007605 | 53.28% / 53.19% |
| GPU noise | 0.007632 | 0.007605 | 53.28% / 53.19% |
| FP8 feed-forward input + GPU noise | 0.001457 | 0.001431 | 85.98% / 86.28% |
| FP8 QKV input + GPU noise | 0.007621 | 0.007592 | 53.33% / 53.24% |
| Both FP8 branch inputs + GPU noise | 0.000824 | 0.000817 | 91.98% / 92.07% |

```powershell
.venv\Scripts\python test_fused_noise.py --source MLX-DLSS --output D:\PrivateResults\noise-checks.json
.venv\Scripts\python probe_pre_tensor_arithmetic.py --source MLX-DLSS --weights D:\PrivateWeights\logical.safetensors --trial D:\PrivateTrials\native-pre-tensor --trial D:\PrivateTrials\native-pre-tensor-west --output D:\PrivateResults\first-block.json
```

The optional `--first-block-rounding` flag on `compare_capture.py` measures the
same branch correction through the full reconstruction. On matched original-view
input, RGB MAE **worsened from 0.01435 to 0.01611**, despite the intermediate
improvement; high-pass correlation improved from 0.9741 to 0.9810. This exposes
unresolved downstream differences and prevents treating local agreement as an
image-quality pass. Both runs use 1920×1080 input with 1152×1920 padding, the same
controls/noise counter, and the existing reconstruction fusions. GPU graph time
remains around 190 ms, far above the native model's 5.45 ms. No game deployment
or native speedup follows from this experiment.

Source, tool provenance, addresses expressed as constant-buffer offsets, numeric
comparisons and limitations are recorded in
[`native-pre-layout.json`](../../evidence/neural-model-research/native-pre-layout.json)
and [`first-block-rounding.json`](../../evidence/neural-model-research/first-block-rounding.json).
Weights, raw textures/tensors, vendor code and tool binaries are not published.

## Direct tensor-core accumulation

The follow-up isolates another numerical difference: the native FP8 matrix
instructions use FP16 C/D operands and sometimes start with the scaled residual
or attention bias already in C. Adding that value after a separately rounded
matrix product is different. NVIDIA documents the register layout and permitted
types in [PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832);
the internal accumulation order is not fully specified, so instruction names
alone are insufficient evidence of equality.

`FusedNorm.mma` now provides a bounded, direct `m16n8k32` E4M3 operation with
FP16 C/D, plus `m16n8k16` FP16 for the input adapter. It supports matrix batches,
shared or batched B/C, edge tiles and misaligned A views. Inputs must already
have the required dtype; it does not silently quantize weights. The pointwise
and window operations in `mma_first_block.py` retain the recovered model's
schedule. This is a diagnostic implementation, not a tuned production GEMM.

Twenty-six exact arithmetic cases cover layout, tails, batches, strides and
alignment. A cancellation fixture returns 1.5 when C initializes accumulation,
but zero when C is added after a rounded product, confirming the order matters.
All four block-0 FP8 weight matrices were independently checked to round-trip
exactly before they were used. The native-prefix results are:

| Configuration | Original MAE | West MAE | Exact bytes, original / west |
| --- | ---: | ---: | ---: |
| Both branch rounding points, ordinary cuBLAS | 0.000824 | 0.000817 | 91.98% / 92.07% |
| Plus cuBLAS FP16 accumulation | 0.000546 | 0.000547 | 94.30% / 94.34% |
| Direct MMA in FFN, initial residual | 0.000169 | 0.000168 | 98.40% / 98.42% |
| Direct MMA throughout block 0, initial residuals | 0.0000824 | 0.0000804 | 99.17% / 99.19% |
| Plus initial attention bias | 0.0000549 | 0.0000528 | 99.46% / 99.48% |

Each row uses the same captured prefixes and GPU noise formula. The direct-MMA
rows keep cuBLAS FP16 accumulation enabled for the initial adapter; replacing
that adapter with direct FP16 MMA produced exactly the same aggregate metrics.
This still leaves roughly 0.5% differing prefix bytes and does not prove full
model or temporal parity.

```powershell
.venv\Scripts\python test_fp8_mma.py --output D:\PrivateResults\mma-tests.json
.venv\Scripts\python probe_pre_tensor_arithmetic.py --source MLX-DLSS --weights D:\PrivateWeights\logical.safetensors --trial D:\PrivateTrials\native-pre-tensor --trial D:\PrivateTrials\native-pre-tensor-west --output D:\PrivateResults\mma-stem.json --fp16-accumulation --mma
```

Add `--mma-adapter` to the last command, with a fresh output path, to run only the
fully seeded MMA variant with its direct FP16 adapter. The cuBLAS option is a
process-local [PyTorch setting](https://docs.pytorch.org/docs/main/notes/cuda.html#full-fp16-accumulation-in-fp16-gemms),
not a driver-profile change or a game setting.

For full reconstruction, `compare_capture.py --mma-first-block` replaces only
block 0's branches. It intentionally retains that tool's original input adapter,
NumPy noise and downstream implementation, and cannot be combined with
`--first-block-rounding`. On the matched original view, final RGB MAE was
**0.01582**, versus 0.01435 for the earlier reconstruction and 0.01611 for branch
rounding alone. High-pass correlation was 0.98199. Graph time was **193.91 ms**;
this is not native runtime latency. This candidate is not accepted or deployed.

The full settings, per-channel errors, ULP counts, synthetic checks and image
comparison are in [`first-block-mma.json`](../../evidence/neural-model-research/first-block-mma.json).

## Separate pooled features and a native-input diagnostic

The first block's pooled output has now been captured in two hidden demo views.
It is a separate **17,694,720-byte tensor**, addressed by packed argument 248,
with logical shape **576×960×32** for a 1920×1080 image. The existing 32 MiB
skip prefix does not contain this tensor. Its physical layout is two
576×960×16 channel planes, with logical channel bits selecting physical bits
`[0,2,3,1]` inside each plane. It does not use the skip's 512-byte tile layout.
`decode_pre_pool.py` applies that fixed bijection and compares every element,
including the 36 padded rows. Those rows contain features, not all zeros.

The combined `optiscaler-demo-pre-tensor.patch` now supports the separate
`nr-pre-pool-capture.enable` marker. It is mutually exclusive with
`nr-pre-tensor-capture.enable`: enabling both captures neither. The new mode
requires the recognized first preprocessor, successful launch, first reset,
zero noise counter, exact image/pool extents, a retained allocation and the
existing observed UAV-state contract. It copies once, restores UAV state and
maps only after the actual submitting queue's fence completes. Runtime metadata
still says `layout_verified: false`; layout validation is an offline result.

`collect_pre_pool.py` runs the original and west camera views for 25 seconds
each. It requires the exact hidden-executable hash, rejects running games and
conflicting experiments, archives private outputs and restores the demo's DLL,
scene and markers. Use the already configured NVIDIA sample and a locally built
capture DLL; this is not a WuWa install step.

```powershell
python collect_pre_pool.py --demo-dir D:\PrivateDemo\bin\ngx_dlss_demo --output-dir D:\PrivateResults --capture-dll D:\PrivateBuild\OptiScaler.dll --capture-sha256 <verified-build-sha256>
.venv\Scripts\python probe_pre_pool_arithmetic.py --source MLX-DLSS --weights D:\PrivateWeights\logical.safetensors --trial D:\PrivateResults\trials\native-pre-pool-original --trial D:\PrivateResults\trials\native-pre-pool-west --output D:\PrivateResults\pool-arithmetic.json
```

Using the previously tested direct-MMA first block, pooling **unpublished FP16
values** in horizontal pairs gives the closest result in both views. For a 2×2
region with top row `a,c` and bottom row `b,d`, this is
`((a+c)+(b+d))*0.25`, rounded in FP16 at each operation and then published as
E4M3. The old reference uses `(((a+b)+c)+d)*0.25`.

| Pooling order, same direct-MMA block | Original MAE | West MAE | Exact bytes, original / west |
| --- | ---: | ---: | ---: |
| Sequential, vertical first | 0.0001612 | 0.0001652 | 98.81% / 98.83% |
| Paired vertical | 0.0001533 | 0.0001569 | 98.83% / 98.85% |
| Paired horizontal | **0.0001169** | **0.0001192** | **99.07% / 99.09%** |
| Paired diagonal | 0.0001536 | 0.0001570 | 98.83% / 98.85% |

The original reconstruction's pooled tensor had MAE 0.00510 and 60.74% exact
bytes on the original view. The new comparisons use all 17,694,720 elements per
view, without refitting the layout on west. The west input differs from the
older skip-prefix experiment, so it was reconstructed from its new paired
capture rather than reusing an old reference array.

To isolate the pooled path, `compare_capture.py --native-pre-pool <trial>` can
substitute the **exact captured activation** at block 1. It verifies matching
input/output texture hashes and control values, and requires noise frame zero
and network height 1152. This uses information from the native runtime for that
exact frame; it cannot run as an independent model or demonstrate a speedup.

On the identical original image, this substitution still **worsened final RGB
MAE from 0.01435 to 0.01604**. The full-resolution skip and remaining network
retain their previous implementation. Fixing the pooled input alone therefore
does not resolve the final-image mismatch; downstream and/or skip-path errors
remain. The diagnostic graph took 190.66 ms. The real native capture runs stayed
around **5.42 ms model / 5.62–5.63 ms total**, with only two sparse warm readings
per view. No quality acceptance, native speedup or game deployment follows.

Capture contracts, hashes, per-channel errors, padding results, pooling variants
and the matched final-image diagnostic are published in
[`native-pre-pool.json`](../../evidence/neural-model-research/native-pre-pool.json).

## Complete first-block capture and full-network accumulation

The complete full-resolution skip is **70,778,880 bytes**, immediately followed
by the **17,694,720-byte pool** in the observed allocation. The combined capture
patch adds `nr-pre-stem-capture.enable` to copy their contiguous 88,473,600-byte
range once. Exactly one of the prefix, pool or complete-first-block markers must
be enabled. The new mode additionally verifies the 1152×1920 skip extent,
576×960 pool extent, exact pointer separation and retained allocation bounds.
It uses the same submitting-queue fence and restores UAV state after the copy.
This remains a hidden-demo research feature, not a game configuration.

`collect_pre_pool.py --stem` collects both original and west views with this
mode. Its default trial prefix becomes `native-pre-stem`. `decode_pre_stem.py`
separates the payload and applies the independently established skip and pool
layouts. The skip prefixes match the earlier native prefix captures byte for
byte; all decoded values are finite. The new original-view texture differs
slightly from the earlier capture, so all image comparisons below were rerun
against the new paired input and native output.

```powershell
python collect_pre_pool.py --stem --demo-dir D:\PrivateDemo\bin\ngx_dlss_demo --output-dir D:\PrivateResults --capture-dll D:\PrivateBuild\OptiScaler.dll --capture-sha256 <verified-build-sha256>
.venv\Scripts\python probe_pre_tensor_arithmetic.py --whole-skip --fp16-accumulation --mma --mma-adapter --source MLX-DLSS --weights D:\PrivateWeights\logical.safetensors --trial D:\PrivateResults\trials\native-pre-stem-original --trial D:\PrivateResults\trials\native-pre-stem-west --output D:\PrivateResults\whole-skip.json
```

The best direct-MMA first block matches **99.34% / 99.35%** of the complete skip's
bytes in original / west, with MAE **0.000123 / 0.000126**. This comparison covers
every 70,778,880-byte skip, including the bottom rows missing from the old prefix.
It is still not exact arithmetic or final-image quality acceptance.

`compare_capture.py --native-pre-stem <trial>` substitutes both decoded native
first-block outputs: the full-resolution skip and the input to block 1. It
requires matching captured input/output hashes, control values, reset/noise
conditions and dimensions. The resulting program uses known activations from
that exact frame; it cannot run as an independent renderer. The flag cannot be
combined with other first-block substitutions.

An additional `--fp16-accumulation` option tests the process-local cuBLAS setting
throughout the reconstruction. It requires `--precision fast` and records the
actual backend setting. The default explicitly leaves full FP16 accumulation
disabled. This is separate from the direct FP8 MMA diagnostic, and neither flag
changes the native NVIDIA runtime.

All rows use 1920×1080 inputs with the same 1152×1920 padded network extent and
existing pointwise/FFN fusions. Each view's comparisons share identical input,
native target and controls:

| Reconstruction condition | RGB MAE, original | RGB MAE, west | Graph ms, original / west |
| --- | ---: | ---: | ---: |
| Previous implementation | **0.01425** | **0.01647** | 190.58 / 190.61 |
| Exact native first-block outputs | 0.01632 | 0.01757 | 170.10 / 170.09 |
| Full FP16 accumulation | 0.01445 | 0.01663 | 185.16 / 185.23 |
| Exact native first block + full FP16 accumulation | 0.01618 | 0.01745 | 164.70 / 164.73 |

Exact first-block outputs alone do not fix the final image. Remaining network
operations and final composition need validation. Full FP16 accumulation helps
slightly when the native first block is supplied, but worsens the independent
reconstruction. Neither is accepted. The reduced diagnostic graph times partly
come from substituting captured activations instead of calculating the first
block; **they are not deployable speedups**. Even the independent reconstruction
remains around 185 ms, far slower than the native runtime.

The actual hidden-demo captures measured about **5.46 ms model / 5.66 ms total**,
with only two sparse warm readings per view. First-frame barrier logs showed one
global UAV barrier at each of 16 distinct chain positions; these logs did not
identify consecutive duplicate barriers to remove. No barrier-removal patch was
made. Source restoration, capture contracts, complete skip comparisons and all
eight matched image tests are recorded in
[`native-pre-stem.json`](../../evidence/neural-model-research/native-pre-stem.json).

## Output grading changes the earlier image comparisons

The earlier full-image comparisons omitted a non-neutral native output stage.
Their recorded numbers remain valid for that incomplete pipeline, but they
cannot isolate the quality of the network arithmetic. The new experiment below
changes several of those comparisons after applying the observed grading.

`inspect_output_kernels.py` reads the exact SF-v2 DLL and extracts only its two
identified output modules using separately supplied NVIDIA tools. It checks the
DLL, bounded fatbinary payloads and SM89 module hashes. Extraction and disassembly
stay outside the repository. The tool does not modify or launch an application.

The combined demo patch now records selected numeric output-kernel arguments.
Address/texture arguments are recorded only as presence booleans. It retains the
existing demo-only guards and is applied directly to the normal R4 source; do not
stack it with older capture patches. `collect_output_contract.py` performs one
25-second run in the exact verified hidden demo, collecting paired first-reset
textures and argument records. It restores the demo DLL and markers afterward.

For the observed **style 1, preset 0, intensity 1**, the final post-process uses:

| Packed argument offset | Observed value | Derived role |
| --- | ---: | --- |
| 316 / 320 | 0 / 1 | Black / white normalization |
| 324 | approximately -0.1 | Exposure multiplier `2^value` |
| 332 | -0.25 | Blend between the channel value and its smoothstep |
| 336 | approximately -0.1 | HSL saturation multiplier `1 + value` |
| 328 / 340 | 0 / 0 | Neutral gamma / second saturation exponent |
| 344–368 | all 0 | Neutral white-balance and tonal-band controls |

These are observed internal launch parameters, not new user configuration keys
or values fitted to the target image. The optional mask/reference/history fields
tested here are absent; the base/reference input is present. The neural output
kernel separately reports a residual scale of 1/32, consistent with the existing
composition's 1/4 residual relative to the preprocessor's 1/8 RGB scale. That
consistency alone does not prove the complete neural output operation is correct.

`compare_output_grade.py` applies an **algebraic approximation** of the observed
exposure, contrast and saturation to saved reconstructions. It checks each
reconstruction against both hashes of its exact paired input/native output and
against the observed controls. Native HSL round trips, texture arithmetic and
special-function rounding are not reproduced bit for bit. Reduced HSL saturation
is expressed as `L + 0.9*(RGB-L)`, with `L=(max(RGB)+min(RGB))/2`. The result is
rounded to FP16 to match the captured output format.

```powershell
python collect_output_contract.py --demo-dir D:\PrivateDemo\bin\ngx_dlss_demo --output-dir D:\PrivateResults --capture-dll D:\PrivateBuild\OptiScaler.dll --capture-sha256 <verified-build-sha256>
python compare_output_grade.py --contract-trial D:\PrivateResults\trials\native-output-contract --case original D:\PrivateResults\trials\native-output-contract\capture D:\PrivateResults\original-reconstruction --output D:\PrivateResults\output-grade.json
```

Twelve saved full-resolution reconstructions were checked. Four use exactly the
same input and target bytes as the new parameter capture. For the other eight,
reuse of those parameters is an explicit inference from matching runtime,
controls and reset conditions; their image comparisons still use their own
exact paired targets. No network inference or latency benchmark is rerun by this
CPU script.

| Reconstruction | MAE before grading, original / west | MAE after grading, original / west |
| --- | ---: | ---: |
| Previous implementation | 0.01425 / 0.01647 | **0.00794 / 0.00923** |
| Exact native first-block outputs, diagnostic only | 0.01632 / 0.01757 | **0.00524 / 0.00591** |
| Full FP16 accumulation | 0.01445 / 0.01663 | 0.00800 / 0.00879 |
| Exact native first block + full FP16 accumulation | 0.01618 / 0.01745 | 0.00539 / 0.00593 |

On the capture that exactly matches the newly observed contract, the graded
baseline's MAE is **0.00786**. The previous first-block branch-rounding variant
improves that to **0.00496**, direct-MMA variant to **0.00534**, and captured-pool
diagnostic to **0.00525**. Thus the earlier conclusion that these variants worsen
the final image does not hold after accounting for this output stage. The
branch-rounding variant remains closer than the direct-MMA variant on this
particular full-image test; better intermediate agreement is still not proof of
better final output.

The script also checks storage-only rounding, exposure alone, exposure plus
contrast, and FP16 rounding before the full grade. Storage rounding alone barely
changes the errors. Input rounding before grading changes MAE by less than
0.0000002 in these cases. These controls help attribute the improvement to the
grading operation rather than to storage precision.

**No quality or speed target has been met.** All reconstructions still differ
from the native image. The new hidden-demo run measured **5.425 ms model /
5.635 ms total**, from two sparse warm intervals. The CPU grading comparisons do
not improve that timing, and captured-activation variants remain non-deployable.
The next model experiments can keep this known color transform explicit while
testing whether a smaller learned component preserves detail and motion; this
is a research direction, not a demonstrated replacement. Complete numerical
results and contracts are in
[`native-output-grading.json`](../../evidence/neural-model-research/native-output-grading.json).

## Explicit grading in a small student and a fused GPU implementation

`output_grade.py` supplies a differentiable version of the observed exposure,
contrast and saturation. `student_probe.py --output-grade-contract <trial>` wraps
the existing student so it learns the residual **before** that known grading.
The stage is computed in FP32; inference uses an FP16 network and stores the
graded result in FP16. The controls remain fixed to the validated observation.
This is an experiment, not a change to the NVIDIA model or game integration.

`FusedNorm.output_grade` reads strided NCHW RGB directly and writes interleaved
RGB in one CUDA launch. It uses separate arithmetic operations to preserve the
Torch grading result, including its final FP16 conversion. This is our algebraic
approximation of grading, not a claim of bit-exact native color processing.
The function is for finite inference inputs on the tested Windows SM89 setup.

`test_output_grade.py` checks 17 cases with 6,658,722 channel values: planar,
interleaved, sliced, transposed and broadcast inputs; parameter boundaries;
mixed colors containing every finite FP16 value; and a full 1920×1080 image.
All tested FP16/FP32 outputs match Torch exactly. At 1080p, warm CUDA Graph
intervals were **1.316 ms** for the separate Torch operations and **0.0222 ms**
for the fused operation. These measure only grading in this process, not the
native renderer or application latency.

```powershell
.venv\Scripts\python test_output_grade.py --contract-trial D:\PrivateResults\trials\native-output-contract --output D:\PrivateResults\fused-grade.json
```

The first matched training test retains the previous six training views, two
validation views, width 16, two blocks per stage, 1,500 steps, fixed seed,
whole-image batches and cosine learning-rate schedule. Adding explicit grading
**worsens** validation MAE from **0.02285 / 0.02318** to **0.02793 / 0.02790**.
Grading the original input alone is closer at **0.01706 / 0.01963**. Thus the
known color operation improves the initial baseline but does not make this
trained small model accurate. Around 67% of the new north-view squared error
lies at spatial wavelengths of at least 256 pixels, pointing to broad changes
in tone/structure as a remaining problem. This FFT diagnostic does not measure
perceptual or temporal quality.

For those same trained weights, including grading in the complete student graph
takes **2.503 ms** with Torch operations and **1.214 ms** with the fused kernel.
The outputs match exactly. That is a usable optimization for the research
student, but the student fails quality and has no D3D12 integration. It is not a
3 ms replacement for the native renderer.

To improve data coverage, `collect_demo_views.py --views-file <json>` accepts a
bounded camera list. The included
[`translated-training.json`](capture-viewsets/translated-training.json) adds
twelve training views from two translated positions. Collection still uses only
the verified hidden demo and restores the scene, DLL and marker. Private output
inside the repository is rejected. `audit_teacher_views.py` checks restoration
records, runtime hashes, controls, texture sizes, completed fences and separation
from the existing validation images. Only frame zero is used for these student
tests; the other captured frames do not establish temporal training.

All twelve added views passed this audit, giving **18 distinct training inputs**
and the same two validation inputs. The following runs use the same architecture
family, seed, controls, full-image batches, loss and cosine schedule. The
4,500-step runs increase total training updates; the eighteen-view version has
approximately the same updates per image as the original six-view, 1,500-step
run. They are separate runs from the same seeded initialization.

| Training views | Steps | Width | Mean training MAE | North MAE | Shifted north MAE | Full student graph ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1,500 | 16 | Not measured across all six | 0.02793 | 0.02790 | 1.214 |
| 18 | 1,500 | 16 | 0.02747 | 0.01953 | 0.02112 | 1.213 |
| 6 | 4,500 | 16 | 0.00771 | 0.02398 | 0.02688 | 1.216 |
| 18 | 4,500 | 16 | 0.01193 | **0.01558** | **0.01953** | 1.221 |
| 18 | 4,500 | 32 | 0.01036 | 0.01626 | 0.01974 | 1.970 |

More scene coverage improves both validation errors at each training length.
Increasing width from 223,440 to 871,792 parameters improves fitting of the
training views but slightly worsens both validation results. That distinction
matters: a larger model alone is not the next demonstrated quality improvement.
The width-32 model takes 3.247 ms with separate Torch grading and 1.970 ms with
the fused operation, with identical tested output. All reported graph times
include the complete student and grading, but exclude application integration.

Use `--evaluate-all-training` to report every training view after fitting. A
typical graded run adds these options to the existing hierarchical-student
command, with each extra training image supplied through
`--extra-train-capture`:

```powershell
--architecture hierarchical --width 16 --blocks 2 --whole-frame --batch 1 --loss-border 0 --steps 4500 --max-seconds 180 --cosine-lr --evaluate-all-training --output-grade-contract D:\PrivateResults\trials\native-output-contract
```

The models remain rejected for the requested no-quality-loss replacement.
These repeatedly consulted validation cameras are not an independent final
test set, and they do not cover other scenes or motion. The new evidence is in
[`fused-output-grade.json`](../../evidence/neural-model-research/fused-output-grade.json),
[`teacher-translated-views.json`](../../evidence/neural-model-research/teacher-translated-views.json)
and [`student-explicit-grading.json`](../../evidence/neural-model-research/student-explicit-grading.json).

## Learned affine color field with a full-resolution detail branch

The best eighteen-view, width-16 student's remaining squared error is mostly
broad in the two validation views: **79.6% / 82.5%** lies at spatial wavelengths
of at least 64 pixels. The original training view has only 30.9% in those bands.
These periodic-boundary FFT diagnostics motivated a smooth, content-dependent
color correction, but do not establish perceptual or temporal quality.

`student_probe.py --architecture hierarchical-affine` adds a zero-initialized
12-channel head at the existing bottleneck. Each padded 32×32 cell predicts a
3×4 RGB affine correction. Bilinear interpolation supplies coefficients for
every original pixel; the existing learned full-resolution detail branch stays
in place. Before the known output grade, the result is
`clamp(source + 0.25 * (detail + affine_correction), 0, 1)`.
This uses compact transforms rather than a reduced-resolution replacement
image. It follows an idea from [HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf),
but is not an HDRNet reproduction: it uses a two-dimensional field with a detail
decoder, not a bilateral grid with a learned guide.

The new width-16 network has **224,604 parameters**, just 1,164 more than the
previous model. Both runs use the same eighteen training images, two validation
images, controls, seed and 4,500-step schedule. The affine run completed all
steps in 252.1 seconds. Use the previous whole-frame command with
`--architecture hierarchical-affine --max-seconds 450`; the observed output
grading contract is required.

| Model | Mean training MAE | Original view MAE | North MAE | Shifted north MAE | Complete fused graph ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous hierarchical student | 0.01193 | 0.01116 | **0.01558** | **0.01953** | 1.221 |
| Affine field plus detail | 0.01242 | 0.01195 | 0.01678 | 0.01988 | 1.199 |

The affine model is **rejected**: it slightly worsens both validation errors.
These historical timing samples do not establish that its architecture is
faster than the previous model. Both are isolated CUDA Graph measurements,
include grading and all student branches, and exclude D3D12 integration.

`ablate_affine_student.py` reloads the private checkpoint, validates both texture
hashes and controls, and requires exact reproduction of each saved output
before disabling either branch. The following are post-training diagnostics;
disabled branches are still computed, so no speedup is claimed.

| View | Both branches | Affine only | Detail only | Neither: source plus grade |
| --- | ---: | ---: | ---: | ---: |
| Original training view | 0.01195 | 0.01845 | 0.03656 | 0.03891 |
| North validation | 0.01678 | 0.01725 | 0.01963 | 0.01706 |
| Shifted north validation | 0.01988 | 0.02051 | 0.02115 | 0.01963 |

The affine branch explains much of the original view's color change, while
the detail branch further reduces its error. That benefit transfers weakly to
the validation cameras. This test does not support installing the model.

The retained kernel improvement is `FusedNorm.affine_compose`: it interpolates
coefficients and composes RGB directly without allocating a dense twelve-channel
1080p field. The FP16 operation follows the reference's rounding boundaries.
An initial version differed at two channel values in each of the full-size
planar and interleaved tests. Explicit round-to-nearest multiply-adds in the
interpolation fixed those failures. The CUDA operator's interpolation order was
checked against [PyTorch 2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu).

The final **18 finite-input cases**, covering 19,097,025 RGB channel values,
match the Torch reference exactly, including strided and broadcast inputs,
partial cells, zero fields, clipping and full 1080p. The captured graph also
matches. This is tested operator parity, not exhaustive arithmetic or native
NVIDIA image parity. The isolated composition takes **0.0655 ms**, versus
**1.248 ms** for the separate Torch operations. Combining it with the previous
grading fusion reduces this complete affine student's graph from **3.683 ms to
1.199 ms** with identical tested output. It does not accelerate the native
NVIDIA renderer, which remains around 5.4 ms.

Run `test_affine_compose.py --output <private-json>` for the synthetic operation
checks. For branch diagnostics, use
`ablate_affine_student.py --student <private-result-directory> --case <label> <capture-directory> <role> --output <private-json>`;
roles are `primary`, `validation` and `extra-0`. These commands do not launch a
game or the demo. Source, full numeric reports, failed and corrected kernel
checks, and branch metrics are in
[`student-affine-field.json`](../../evidence/neural-model-research/student-affine-field.json).
Weights and images remain private. No new application capture or game change
was needed. The **3 ms at true 1080p with no quality loss** target remains unmet.

## Pretrained block costs, sensitivity and feature distillation

The next experiment starts from the recovered pretrained model rather than a
new small network. It asks which learned blocks could be replaced cheaply
without materially changing the complete image. It takes ideas from
[LLM layer pruning](https://arxiv.org/abs/2403.17887) and
[feature-distilled image-restoration blocks](https://arxiv.org/abs/2605.02794).
Those papers' quality or speed results do not transfer to this renderer.

`summarize_block_costs.py` matches the exact 158-chain native schedule against
the recovered 71-block order, rejecting changed kernel names, order and chain
sizes. This gives an **inferred** correspondence, not a native layer annotation.
All 24 post-warmup frames of the existing timing trace matched. The complete
instrumented intervals have a 6.138 ms median. Blocks 0 and 70 account for about
0.350 and 0.376 ms individually; the eight global blocks together account for
0.874 ms. These are historical instrumented costs, not an uninstrumented
benchmark or removable-latency guarantees. `test_block_costs.py` verifies seven
rejection cases and exclusion of a failed frame from a mixed trace.

**Do not delete native launches using this map.** Chained kernels publish
counters and depend on layouts and downstream waits. Removing their launch can
break synchronization. The ablations below change only the isolated Torch
reconstruction, where same-shape block bypass is explicitly supported.

`probe_block_sensitivity.py` tests all **59 same-shape blocks** on the original
and west views, preserving structural transitions, input/output blocks, 1080p
input, the 1152-row internal layout, noise counter zero and observed grading.
Before testing each view it requires exact reproduction of its earlier saved
ungraded reconstruction. Both checks pass; no captured native intermediate
features are substituted. All 118 single-block tests finish in 96.9 seconds.

The unmodified graded reconstruction has native RGB MAE **0.00794 / 0.00923**.
Every single-block removal changes the image, and none improves native MAE on
both views. Block 43 causes the smallest worst-view change: removing it differs
from the unchanged reconstruction by **0.00290 / 0.00345** RGB MAE, while its
native error becomes **0.00799 / 0.01004**. Its native instrumented block interval
is about 0.0722 ms, so this one block is not a route to the entire required gain.

Joint deletions also fail. The six least-sensitive blocks were selected by
worst-view RGB change in the single-block sweep; the group test recomputes the
complete model instead of adding individual errors.

| Reconstruction variant | Original native MAE | West native MAE | Historical native interval for the corresponding group |
| --- | ---: | ---: | ---: |
| Unmodified | 0.00794 | 0.00923 | Not a deletion |
| Skip blocks 41, 42, 43, 44, 46, 47 | 0.01157 | 0.01532 | 0.436 ms |
| Skip global blocks 31–38 | 0.03720 | 0.04569 | 0.874 ms |

These group medians sum each frame's relevant intervals before taking a median.
They must not be subtracted from the newer uninstrumented native model time.
Group output errors are non-additive, and all candidates are rejected.

`distill_block_projection.py` then replaces only block 43 with a learned affine
512-to-512 feature projection. It fits centered float64 ridge regression on
**8,640 feature tokens from four training cameras**: northeast, northwest,
southeast and southwest. Ridge is fixed at 0.001 times the mean covariance
diagonal, with an unpenalized intercept. Neither evaluation image is included
in this fit. The teacher features are from the reconstruction, not native
intermediate captures. Inference uses FP16 matrix multiplication and bias,
followed by E4M3 publication. The projection has 262,656 parameters versus
1,901,584 stored parameters in the original block.

| Complete-image comparison | Original MAE | West MAE |
| --- | ---: | ---: |
| Deleted block versus unchanged reconstruction | 0.00290 | 0.00345 |
| Fitted projection versus unchanged reconstruction | **0.00223** | **0.00251** |
| Unchanged reconstruction versus native | **0.00794** | **0.00923** |
| Fitted projection versus native | 0.00828 | 0.00958 |

The projection preserves more of the reconstructed teacher's output than
deletion does, but still worsens agreement with native output relative to the
unmodified model. On the original view it is closer to the reconstruction than
deletion while being farther from NVIDIA than deletion. Better feature matching
alone therefore does not guarantee better native output.

On the actual captured feature shape `[1,36,60,512]`, the isolated reconstructed
block takes **0.329 ms** and the projection **0.0358 ms**. Each CUDA Graph matches
its own eager output. This is not a ninefold native speedup: the original native
block's historical instrumented cost is only 0.0722 ms, and no projection was
integrated into the native runtime. The complete reconstruction remains far
slower than NVIDIA, and both versions fail native quality matching.

All application and game files remain unchanged; these tests use existing
private captures and never open a window. Only source, numerical results and
artifact hashes are published in
[`native-block-costs.json`](../../evidence/neural-model-research/native-block-costs.json)
and [`pretrained-block-compression.json`](../../evidence/neural-model-research/pretrained-block-compression.json).
The fitted weights stay private. The two repeatedly consulted evaluation
cameras are research diagnostics, not an independent final quality test.

Reproduction interfaces:

```text
summarize_block_costs.py --trace <private-timing.csv> --output <private-json>
test_block_costs.py --trace <private-timing.csv> --output <private-json>
probe_block_sensitivity.py --source <pinned-reference> --weights <private-weights> --contract-trial <private-trial> --case <label> <capture> <saved-baseline> --output <private-json>
distill_block_projection.py --source <pinned-reference> --weights <private-weights> --contract-trial <private-trial> --train-capture <capture> --case <label> <capture> <saved-baseline> --output-directory <private-directory>
```

Repeat `--case` and `--train-capture` for multiple views. The sweep defaults to
all 59 supported blocks; `--single-blocks none --group <label> <comma-separated-blocks>`
tests a joint removal. Structural transitions and boundary blocks are rejected.
Source commit, weight hash, capture controls and saved baselines are checked
before accepting results. No compressed model is accepted for installation.

## Final-block inputs and FP8 accumulation

The latest diagnostic separates errors accumulated by the reconstructed
decoder from errors in its final block. The new
[`optiscaler-demo-post-inputs.patch`](optiscaler-demo-post-inputs.patch) is an
**add-on to the combined `optiscaler-demo-pre-tensor.patch`**, not a standalone
patch or installer option. It captures two bounded input prefixes after the
first successful neural post block (chain 156), without changing the launch:

| Packed pointer argument | Candidate tensor | Bytes |
| --- | --- | ---: |
| 0 | Half-resolution decoder, 576×960×32 E4M3 | 17,694,720 |
| 8 | Full-resolution skip, 1152×1920×32 E4M3 | 70,778,880 |

The probe requires the observed 1920×1080 first evaluation, noise counter zero,
1920×1152 internal extent, retained buffers with sufficient nonoverlapping
ranges, and observed UAV states with complete barrier coverage. It inserts
bounded copies, restores those states, and reads back only after the actual
submission fence completes. Capture metadata deliberately leaves
`layout_verified` false; range containment alone does not establish semantics.
All existing demo process and marker gates remain in effect.

`collect_pre_pool.py --post-inputs` collected original and west views in two
bounded 25-second runs. It enforces the exact hidden executable before launch.
Both captures completed with four fenced paired image frames; only the first
reset frame is used here. The demo DLL, camera scene and native source files
were restored, and WuWa was not launched or changed. The two sparse model
samples in each run were **5.44 ms**; this instrumented capture is not a speedup.

`decode_post_inputs.py` tests the prior tiled skip mapping and four decoder
layout hypotheses. In the original view, the first **33,554,432 bytes** of the
post-block skip exactly match an earlier first-block capture of the same input
and controls. This check covers that prefix, not all bytes. The decoder's
two-plane, channel-permuted layout correlates with the reconstruction at
**0.98085 / 0.98131**, versus 0.40609 / 0.42650 for unpermuted planes and below
0.053 for the two interleaved alternatives. Those results support the mapping
without proving every layout or arithmetic detail.

`probe_post_block.py` first requires the isolated final block with reconstructed
inputs to reproduce the complete reconstruction's four-channel head exactly.
Both views pass. The original view also exactly reproduces a saved, independently
generated reconstruction with matching input, output and controls. It then
substitutes native inputs to localize the remaining error:

| Final-stage experiment | Original native RGB MAE | West native RGB MAE |
| --- | ---: | ---: |
| Reconstructed inputs and arithmetic | 0.0078644 | 0.0089212 |
| Native skip only | 0.0077571 | 0.0088140 |
| Native decoder only | 0.0015595 | 0.0017197 |
| Both native inputs | 0.0011095 | 0.0012314 |
| Both native inputs, direct FP16 output head | 0.0011096 | 0.0012313 |
| Both native inputs, direct FP8 feed-forward MMA | 0.0002802 | 0.0003137 |
| Same, with residual seeded into the MMA accumulator | 0.0002056 | 0.0002276 |
| Same, with direct attention MMA and seeded attention bias | **0.0001638** | **0.0001792** |
| Reconstructed inputs, all of these final-block MMA changes | 0.0076828 | 0.0087705 |

The last four rows use the direct FP16 output head. Most original image error
comes from the upstream decoder: replacing both inputs lowers it by about 86%.
Within the remaining final-block error, FP8 operand conversion and accumulation
order matter; changing only the output head has negligible effect. The final
block already avoids an extra E4M3 publication before the head in the pinned
reference. There is no additional publication to remove there.

`mma_first_block.py` now accepts `block_index=70` as well as its unchanged
default of zero. Block 70 uses the recovered (-4,-4) window origin, zero padding
after the feed-forward stage and cropping after attention. Its direct-MMA path
uses the existing CUDA implementation; no new native instruction code is
distributed. `test_boundary_mma.py` verifies exact block-0 agreement with the
prior published implementation and exact block-70 chunk invariance in 24
finite-input cases. These checks are not native parity tests.

**Captured native inputs are diagnostic substitutions, not an independently
runnable model.** Without them, the final-block changes improve complete-image
error only modestly. Even the best substitution is not bit-exact, and the
grading approximation, temporal behavior and other scenes remain unvalidated.
No new latency improvement or replacement DLL is claimed. The complete
reconstruction remains far slower than NVIDIA's native implementation, and the
3 ms/no-quality-loss target remains unmet.

Only source and [numeric evidence](../../evidence/neural-model-research/native-post-block.json)
are published. Captured features, textures, weights, vendor binaries and
disassembly stay private. Reproduction interfaces, using fresh private output
directories and the same pinned reference and weights:

```text
collect_pre_pool.py --post-inputs --demo-dir <hidden-demo> --output-dir <private-trials> --capture-dll <private-capture-build> --capture-sha256 <sha256>
probe_post_block.py --source <pinned-reference> --weights <private-weights> --case <label> <trial> <saved-baseline-or-dash> --mma-post --output-directory <private-directory>
test_boundary_mma.py --source <pinned-reference> --weights <private-weights> --output <private-json>
```

Repeat `--case` for multiple views. `--prefix-check-trial` optionally verifies
the matching earlier first-block prefix. The regression test requires Git
history containing commit `3beec244d0aea1a35b744f9210818830801d1a2d`.

## Single-head arithmetic and fused FP8 operands

`probe_single_head.py` extends direct-MMA arithmetic to all known 32-channel,
single-head blocks: **0–4 and 66–70**. The adapter returns FP16 features; its
caller preserves existing E4M3 publication and transition rules. Native skip
and decoder captures are **comparison targets only**. Every candidate computes
its own features from the full 1920×1080 image with a 1920×1152 internal extent.
The unchanged and final-block-only candidates reproduce their previous saved
images exactly in both views. Every selected block is visited exactly once.

| Arithmetic changes | Original native RGB MAE | West native RGB MAE |
| --- | ---: | ---: |
| Unchanged reconstruction | 0.0078644 | 0.0089212 |
| Final block only | 0.0076828 | 0.0087705 |
| Decoder 69 + final block | 0.0076508 | 0.0087265 |
| Decoder 68–69 + final block | 0.0076586 | 0.0087272 |
| Decoder 67–69 + final block | 0.0076611 | 0.0087184 |
| Decoder 66–69 + final block | 0.0076542 | 0.0087144 |
| Encoder 0–4 + final block | 0.0051695 | 0.0058616 |
| All single-head blocks | **0.0051667** | **0.0058308** |

All changed candidates use the direct FP16 output head. Early-stage arithmetic
contributes much more than the last decoder stages to the improvement. Native
decoder feature MAE falls from **2.5563 / 2.8091** to **2.0525 / 2.1907** with all
single-head changes; these feature values are not normalized RGB errors.
Complete-image error falls by about 34%, but no native quality gate is passed.

The new `FusedNorm.pack` writes E4M3 operands directly from strided FP16 input.
With `activate=True`, it fuses the half-rounded quadratic gate and saturating
FP8 conversion, preserving the FP16 product rounding before conversion. It
avoids separate activation and clamped tensors. `test_fused_pack.py` compares
**32 cases and 86,125,878 values** byte for byte, covering every finite FP16
input, signed zero, gate overflow followed by saturation, scalar/empty inputs,
incomplete pairs and thread blocks, sliced/transposed/broadcast layouts, eight
dimensions and representative token chunks. All bytes match.

| Isolated operand operation | Prior graph | Fused graph |
| --- | ---: | ---: |
| Plain packing, 262144×32 | **0.0287 ms** | 0.0439 ms |
| Activation + packing, 262144×128 | 0.8175 ms | **0.2233 ms** |

The prior path already uses the fused activation, followed by Torch clamp and
conversion. Plain packing is slower in this contiguous microbenchmark; it is
not a universal conversion speedup. The full final block also packs strided
Q/K and other operands, giving a different measured result:

| Diagnostic block 70 + direct head | Original view | West view |
| --- | ---: | ---: |
| Existing packing | 25.74 ms | 25.71 ms |
| Fuse activation + packing only | 20.82 ms | 20.85 ms |
| Fuse all operand packing | **19.58 ms** | **19.68 ms** |

All three outputs match exactly on reconstructed features of shape
`[1,1152,1920,32]`. These are diagnostic timings, not the native block's
historical instrumented interval of approximately 0.376 ms.

The gain also carries through **all 71 reconstructed blocks** and the direct
head, with inspection copies disabled:

| Reconstructed network and head | Original view | West view |
| --- | ---: | ---: |
| Direct-MMA single-head candidate, existing packing | 204.24 ms | 204.33 ms |
| Same candidate, fused packing | **180.08 ms** | **180.85 ms** |

Eager and CUDA Graph outputs match exactly, as do fused and unfused complete
candidate images. Thirty timed samples follow warmup for each graph. These
measurements start at prepared features, retain every block, and include the
unused original head as well as the direct head. They exclude feature
preparation, output grading and application integration. The roughly 12% gain
preserves the candidate's tested output; it does not establish native quality
or improve NVIDIA's approximately 5.4 ms runtime. The target remains unmet.

`test_boundary_mma.py` retains its 24 regression/chunk checks and adds 30 packing
comparisons spanning all ten single-head blocks. Gate-only and all packing
fusion match the unfused block output in every added case. The
[arithmetic evidence](../../evidence/neural-model-research/single-head-arithmetic.json)
and [packing evidence](../../evidence/neural-model-research/fused-fp8-packing.json)
contain all comparisons, timing samples and source hashes. Original/west views
remain same-scene first-reset diagnostics, not independent temporal or
perceptual acceptance tests. No app was launched and no game, native DLL,
driver or installer default was changed.

```text
probe_single_head.py --source <pinned-reference> --weights <private-weights> --case <label> <native-post-input-trial> --baseline-results <previous-post-block-results> --output-directory <private-directory>
test_fused_pack.py --source <pinned-reference> --output <private-json>
probe_single_head.py --source <pinned-reference> --weights <private-weights> --case <label> <native-post-input-trial> --baseline-results <previous-post-block-results> --fused-packing all --compare-results <unfused-run> --final-only --time-post --time-model --output-directory <private-directory>
```

Repeat `--case` for multiple views. `--fused-packing` accepts `none`, `gate` and
`all`, defaulting to `none`. `--final-only` keeps the unchanged reference and
all-single-head candidate. The timing flags measure only the reconstructed
scopes above; neither establishes native renderer speed or the 3 ms target.

## Branched blocks and shared attention bias

`inspect_branched_kernels.py` reads the pinned runtime and extracts only the
known SM89 modules for private inspection. The 2/4/8-head chained kernels are
in modules **1/2/3**, respectively, rather than the single-head module 0. Their
selected functions have 2,504 / 2,392 / 2,480 instructions. The module hashes,
tool hashes, symbols and instruction-family counts are recorded; extracted
code and disassembly are never published. NVIDIA's disassembler reports that
automatic dataflow analysis is disabled for these files, so no automatic
dataflow proof is claimed.

Manual inspection of the two-head kernel shows carried FP16 MMA accumulators
across input-channel groups and across the four feed-forward branches, with
FP8 publication between stages. This motivates `MmaBranchedBlock`, which uses
direct FP8 MMA for the dense expansion, grouped contraction, residual-seeded
projection and multi-head attention. The pinned reference supplies weights,
window origins and existing publication rules. The diagnostic supports only
blocks **5–22 and 48–65**, with 2/4/8 heads and 64/128/256 channels. It returns
FP16 features; its caller handles each block's publication/transition contract.
Matching every native operand layout and rounding point remains unproven.

`probe_branched_blocks.py` compares nine candidates on both matched views,
retaining all previously corrected single-head blocks. Native intermediate
features are comparison targets, never substituted inputs. The starting images
must exactly reproduce the previous single-head candidate before any changes:

| Added branched arithmetic | Original native RGB MAE | West native RGB MAE |
| --- | ---: | ---: |
| None: single-head baseline | 0.0051667 | 0.0058308 |
| Feed-forward only, all branched blocks | 0.0050849 | 0.0054542 |
| Attention only, all branched blocks | 0.0052475 | 0.0057919 |
| Both, all branched blocks | 0.0052386 | 0.0054815 |
| Encoder branched blocks only | 0.0052722 | 0.0055029 |
| Decoder branched blocks only | 0.0051439 | 0.0058078 |
| Both, two-head blocks only | **0.0050661** | **0.0053763** |
| Both, four-head blocks only | 0.0052253 | 0.0059461 |
| Both, eight-head blocks only | 0.0051809 | 0.0055892 |

Two-head changes provide the strongest tested result, while applying the
changes everywhere worsens the original view. No candidate passes native
quality. The mixed result prevents treating a generic direct-MMA replacement
as a verified correction to every block. These repeatedly consulted first-reset
views are diagnostic data, not independent perceptual or temporal validation.

A separate implementation improvement removes copies of the attention bias.
`FusedNorm.mma` now accepts a seed matching a trailing output shape, including
the final M×N dimensions. For example, a `[heads,64,64]` bias can repeat across
the leading window dimension without creating `[windows,heads,64,64]` storage.
It does not implement arbitrary broadcasting of singleton dimensions. The CUDA
kernel selects the seed's repeating batch while preserving initial-accumulator
ordering; existing shared and full-size seed behavior remains unchanged.

All 26 existing FP8/FP16 MMA checks pass, including the cancellation case that
distinguishes an initial accumulator from a later addition. The new
`test_mma_seed_broadcast.py` passes **42 mapping tests and five shape rejections**,
covering trailing batch dimensions, matrix tails and strided seeds against
both expanded seeds and exact small-integer arithmetic.

| Isolated attention-score graph | Expanded bias | Shared bias | Bias storage before → after |
| --- | ---: | ---: | ---: |
| 2160 windows, 2 heads | 0.2467 ms | **0.1287 ms** | 35,389,440 → 16,384 bytes |
| 135 windows, 8 heads | 0.0492 ms | **0.0379 ms** | 8,847,360 → 65,536 bytes |

The shared-bias run reproduces **all 18 complete candidate images exactly**.
`test_branched_block.py` also passes 36 unchanged-reference checks and 18
chunking/shared-bias checks spanning all head counts and shifted windows.
On the actual reconstructed block-5 input `[1,288,480,64]`, the complete
diagnostic block improves from **2.5537 to 2.4497 ms** on the original view and
**2.5517 to 2.4467 ms** on the west view. Eager and graph outputs match in both
modes. Thirty samples follow warmup for each timing. This roughly 4% block
improvement is not a native renderer or complete-network speedup.

The [branched arithmetic evidence](../../evidence/neural-model-research/branched-arithmetic.json)
and [shared-bias evidence](../../evidence/neural-model-research/shared-mma-bias.json)
contain the inspection provenance, all candidate results, checks and timing
samples. Only source and numerical evidence are published. No application was
launched and no game, native runtime, driver or installer default changed.
The approximately 5.4 ms native result and unmet 3 ms/no-quality-loss target
remain unchanged.

```text
inspect_branched_kernels.py --dll <private-native-runtime> --cuobjdump <local-tool> --nvdisasm <local-tool> --output-directory <private-directory>
probe_branched_blocks.py --source <pinned-reference> --weights <private-weights> --case <label> <native-post-input-trial> --baseline-results <single-head-results> --output-directory <private-directory>
probe_branched_blocks.py --source <pinned-reference> --weights <private-weights> --case <label> <native-post-input-trial> --baseline-results <single-head-results> --compact-bias --compare-results <expanded-bias-results> --time-block --output-directory <private-directory>
test_branched_block.py --source <pinned-reference> --weights <private-weights> --output <private-json>
test_mma_seed_broadcast.py --output <private-json>
```

Repeat `--case` for multiple views. All outputs must be fresh private paths.
The inspection script requires the exact pinned runtime and separately obtained
NVIDIA tools; no vendor executable, module, weight or disassembly is bundled.

## Global attention in the compact student

The previous compact student used a channel gate driven by an image-wide mean.
Its largest held-out errors were broad tone and structure. This experiment adds
one content-dependent spatial attention layer at the deepest learned feature
level. It tests whether exchanging information between distant regions helps;
it is a newly trained architecture, not a native-kernel substitution.

`attention_student.py` implements per-token LayerNorm, learned Q/K/V, noncausal
scaled dot-product attention and a learned projection with a 0.1 residual scale.
The projection starts at zero. At 1920×1080, reflection padding produces a
34×60 token grid at 1/32 scale, with 96 channels and three 32-channel heads.
Attention receives spatial CNN features without an additional positional
encoding. Original pixels, pixel rearrangement, detail skips and the output
residual remain intact. Only learned features are reduced in spatial size.
The model grows from 223,440 to **260,880 parameters**.

Training uses the same 18 native first-reset views, two held-out views, control
values, fixed output grade and full 1080p inputs as the previous width-16 model.
Both runs use seed 28411, 4,500 AdamW updates, batch one, weight decay 0.0001,
learning rate 0.002 decaying to 0.00002, and pixel L1 plus 0.25 times horizontal
and vertical gradient L1. No validation frame enters training. The added layer
trains in FP32 and runs in FP16. The observed PyTorch 2.7.1 backend is its
efficient-attention operator for both precisions; no FlashAttention-specific
performance claim is made.

| Measurement | Previous compact model | With global attention |
| --- | ---: | ---: |
| Mean training-view RGB MAE | 0.011933 | **0.010733** |
| North validation RGB MAE | **0.015580** | 0.021237 |
| Shifted-north validation RGB MAE | **0.019533** | 0.019931 |
| North validation RMSE | **0.022253** | 0.030373 |
| Shifted-north validation RMSE | 0.029472 | **0.028500** |
| Complete network + grade, fresh graph timing | **1.229 ms** | 1.317 ms |

The new model fits training images better but increases mean absolute error
on both validation views. Some detail and shifted-view tail metrics improve;
there is no consistent quality improvement and **no quality gate is passed**.
The initial post-training graph measured 1.301 ms; a separate checkpoint reload
measured 1.317 ms alongside the prior checkpoint at 1.229 ms. Both timings
include the full FP16 network and fused output grade, with warmup and 30 samples.
They exclude application integration and are not native demo measurements.

`ablate_attention_student.py` reloads both checkpoints and first reproduces all
six saved complete outputs exactly. It then disables only the fitted attention
branch, preserving the surrounding trained weights:

| RGB MAE | Attention enabled | Same weights, attention disabled |
| --- | ---: | ---: |
| Original training view | **0.010317** | 0.027550 |
| North validation view | 0.021237 | **0.016107** |
| Shifted-north validation view | 0.019931 | **0.019096** |

The branch materially affects the image, helping the training view while
hurting both validation MAEs. This is consistent with overfitting; disabling
the branch is a diagnostic, not a separately retrained replacement. Broad
errors remain: wavelengths of at least 64 pixels account for approximately
88.3% and 82.0% of validation squared error. These FFT fractions describe
spatial error scale, not perceptual quality.

`test_attention_student.py` verifies six unchanged-backbone and zero-branch
cases against this repository's earlier implementation, including full 1080p,
padding and batch variations in FP32/FP16. The existing output head is made
nonzero so it cannot hide intermediate errors. A small direct-attention
comparison has maximum error 1.19e-7; distant-token influence and finite,
nonzero gradients are checked. All complete graph outputs match eager output.

The [numerical evidence](../../evidence/neural-model-research/student-global-attention.json)
includes training results, ablation, frequency diagnostics, timing samples,
checks, dataset hashes and source hashes. These repeatedly consulted views are
not an independent final test set. They contain one scene and no temporal
quality validation. This experiment does not establish that global attention
cannot work; this particular trained candidate fails acceptance. More
representative supervision and validation are needed before further architecture
choices can establish preserved quality.

```text
student_probe.py --capture <first-training-capture> --extra-train-capture <next-training-capture> --validation-capture <north-capture> --extra-validation-capture <shifted-north-capture> --output <private-new-directory> --steps 4500 --max-seconds 600 --width 16 --blocks 2 --loss-border 0 --batch 1 --cosine-lr --architecture hierarchical-attention --whole-frame --output-grade-contract <native-output-contract-trial> --evaluate-all-training
test_attention_student.py --output <private-new-json>
ablate_attention_student.py --student <attention-run> --baseline <previous-graded-run> --case <label> <capture> <saved-prediction-basename> --output <private-new-json>
```

Repeat `--extra-train-capture` in the order recorded in the evidence to include
all 18 views; the translated poses are in `capture-viewsets/translated-training.json`.
Repeat `--case` for each compared view. Captures, learned weights and predictions
remain private. No app was opened and no game, native runtime, driver or installer
default changed. NVIDIA's approximately 5.4 ms result and the unmet
3 ms/no-quality-loss target remain unchanged.

## Illumination data and reserved camera views

The attention experiment pointed to generalization problems. The next collection
adds **12 native training views and four reserved validation views** without
changing the model controls, resolution or sample executable. The existing
sample accepts a directional light in its scene file. Three training conditions
vary its irradiance or direction across four existing camera directions.
Validation uses four different camera positions and lighting combinations.
The source scene is restored byte for byte afterward.

`capture-viewsets/illumination.json` records every pose, light and split.
`collect_demo_views.py` accepts the optional `sun` field and retains its exact
hidden-executable hash requirement. The runner checks that hash before each
launch. All **16 captures and 64 fenced frames** completed; only each first-reset
frame is used for fitting or comparison. The collector restores the scene and
DLL, the runner restores its INI, and all captures are archived outside the
repository. Twelve older scene transformations still reproduce their recorded
hashes exactly. No game, driver or native source changed.

The targets are generated by the native model after changing the scene lighting.
This avoids assuming that applying a brightness transform to an old target
would reproduce the native effect on the changed input. At the original camera,
mean input RGB changes from 0.218 to 0.152 under dim lighting and 0.279 under
bright lighting; image variance remains positive.

`audit_teacher_views.py --allow-new-validation` checks the new splits, paired
payloads, controls, runtime hashes and completed fences. The dataset now has
**30 training views and six validation views**, with no input-hash overlap.
`train_student_collection.py` keeps the earlier 18 views in their original order
and appends only collection entries marked `train`. Both older validation views
remain excluded. Training retains seed 28411, 4,500 updates, whole 1920×1080
frames, batch one, the same loss, optimizer and observed output-grade approximation.
The grading parameters are reused under matched controls; this collection does
not recapture the native output-scalar contract under each light.

`evaluate_student_collection.py` compares saved checkpoints without training.
It rejects any validation input present in a model's training hashes and checks
saved predictions when available. Its timings include the complete FP16 network
and fused output grade, with 30 warmed CUDA Graph samples per model.

| Model | Mean MAE, older two views | Mean MAE, new four views | Mean MAE, all six | Full graph |
| --- | ---: | ---: | ---: | ---: |
| Width 16, 18 training views | 0.017556 | 0.031313 | 0.026728 | 1.228 ms |
| Width 16 + attention, 18 views | 0.020584 | 0.031538 | 0.027887 | 1.305 ms |
| Width 16, 30 training views | 0.017823 | **0.025775** | **0.023124** | **1.214 ms** |
| Width 16 + attention, 30 views | 0.020451 | 0.027208 | 0.024956 | 1.304 ms |

The expanded data reduces the plain student's new-view MAE by about **17.7%**
and its six-view mean by 13.5%. Results are mixed per view: the older north and
new neutral view worsen, while three newly lit views improve. Attention still
does not beat the plain model's six-view mean. This is a useful data improvement,
not preserved native quality. The new validation results guide subsequent
architecture choices, so these views are not an independent final test set.

A subsequent width-32 candidate keeps the 30 training views, six validation
views and 4,500 updates unchanged. Its 871,792 parameters reduce mean training
MAE from 0.013834 to 0.009870, but worsen **all four new validation MAEs**:

| Matched 30-view comparison | Width 16 | Width 32 |
| --- | ---: | ---: |
| Mean MAE, older two views | 0.017823 | 0.017752 |
| Mean MAE, new four views | **0.025775** | 0.028888 |
| Mean MAE, all six views | **0.023124** | 0.025176 |
| Complete network + grade | **1.229 ms** | 1.963 ms |

The wider candidate also fails acceptance. This result is consistent with
overfitting despite the larger dataset; more capacity alone did not solve it.
The first four-model comparison reproduces 16 saved validation images exactly;
the width comparison reproduces 12, including six width-16 images that also
match the earlier comparison byte for byte. All six measured graphs match
eager output. Neither the faster candidate nor the better training fit is a
validated native replacement.

The [collection and model evidence](../../evidence/neural-model-research/student-illumination.json)
contains the capture audit, matched training results, every validation score,
timing samples and preserved-file hashes. The native approximately 5.4 ms
baseline and unmet 3 ms/no-quality-loss target remain unchanged.

```text
collect_demo_views.py --demo-dir <configured-hidden-demo> --output-dir <private-output> --capture-dll <private-fenced-capture-build> --capture-sha256 <verified-hash> --prefix teacher-illumination --views-file research/neural-latency/capture-viewsets/illumination.json
audit_teacher_views.py --collection <private-manifest> --baseline-result <prior-18-view-result> --allow-new-validation --output <private-audit>
train_student_collection.py --collection <private-manifest> --audit <private-audit> --baseline-result <prior-18-view-result> --base-trials <private-trials> --grade-contract <private-native-contract> --output <new-private-directory> --architecture hierarchical --width 16
evaluate_student_collection.py --collection <private-manifest> --audit <private-audit> --baseline-result <prior-18-view-result> --base-trials <private-trials> --model <label> <private-student-run> --output-directory <new-private-directory>
```

Repeat `--model` for up to four checkpoints. The training wrapper also accepts
`hierarchical-attention` and width 32. Source and numerical evidence are published;
sample assets, captures, predictions and learned weights remain private. This is
one scene with varied cameras and lighting, with no temporal or cross-scene
acceptance. The native runtime has not been accelerated.

## Exact-zero feasibility for structured sparsity

NVIDIA's sparse matrix instructions require a defined pattern of zeros and
associated metadata; support for the instruction does not make dense weights
sparse. See the [PTX sparse-MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-sparse-mma).

`inspect_weight_sparsity.py` reads the pinned logical weight file without changing
it. Across **360 weight tensors and 143,831,616 values**, only **7,234,859 values
(5.03%)** are exactly zero. No selected tensor has even half its values zero,
and none passes a complete 2:4 check along either of its final two logical axes.
At their existing shapes, these tensors cannot be wholly converted to 2:4
storage by merely rearranging their values. Additional zeroing would change
the model and require retraining and quality validation.

The audit records zero counts and group histograms, not weights. Its logical
axis checks do not prove a native operand mapping or compatible accumulation
order, and no sparse kernel or performance benchmark is claimed. This rules
out an automatic conversion of these tensors, not all possible kernel or
trained-sparsity improvements.
The [sparsity evidence](../../evidence/neural-model-research/weight-sparsity.json)
contains per-tensor counts, logical-axis histograms and source hashes. Run
`inspect_weight_sparsity.py --weights <private-pinned-weights> --output <private-json>`
to reproduce the audit locally.

## Unseen photo content and contained background launches

The existing sample can render a small emissive image plane, allowing different
content to reach the native model without installing another application.
`make_demo_image_scene.py` writes a **1,072-byte mesh** from numeric geometry and
a material/scene description. Its default asymmetric texture checks image
orientation and coverage. `test_demo_image_scene.py` validates ten file chunks,
seven stream references, four triangles and two exact PNG round trips. Both
fixture captures have the expected corner colors and orientation at 1920×1080.

`collect_demo_image.py` stages only these generated assets inside the sample's
media mount, runs the existing fenced capture build, then restores the scene,
DLL and INI and archives its assets/captures privately. The first prototype used
an unsupported absolute filesystem path in the sample's virtual filesystem;
it produced no model frames and exposed an error dialog that the main-window
hide patch did not contain. That failed trial is retained in the evidence.

The trial runner now always uses `tools/isolated_demo_process.py`: it requires
the same exact hidden-executable hash and starts the sample on a private Windows
desktop that is never activated. It neither requests desktop-switch access nor
calls `SwitchDesktop`. This follows Microsoft's
[process desktop interface](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/ns-processthreadsapi-startupinfow)
and [desktop creation API](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-createdesktopw).
A deliberate unsupported-file test kept its error dialog on that inactive
desktop through four observations, then restored all sample files. Normal
trials verified hidden demo windows and an unchanged input desktop.

An original-scene repeat reproduces the historical input and native output
hashes exactly. The preceding control is also retained: its input MAE is
0.00000055 and native-output MAE is 0.002443 against the historical capture.
Therefore the evidence supports a matching repeat, not universal deterministic
captures. The log's `0x087A0001` presentation status means the window is occluded,
as defined by [Microsoft](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/dxgi-status).
The sample continues rendering and completing the captured GPU work.

`prepare_demo_photo_cases.py` selects three licensed photographs before inference,
records attribution/source hashes, and center-crops/resamples them to 1920×1080
without changing their aspect ratio. All originals, prepared textures, native
captures and student predictions remain private. Sources and terms are listed
in [third-party notices](../../THIRD_PARTY_NOTICES.md#private-photo-validation-sources).
No photo is included in training or used to fit another model in this experiment.

`evaluate_photo_students.py` reloads the three frozen 30-view students, checks
that each new captured input is absent from their training hashes, verifies
the runtime/controls and all capture fences, then evaluates the complete models.
All three models have larger MAE than the fixed-grade diagnostic on every photo:

| Candidate | Portrait MAE | Landscape MAE | Cat MAE | Mean MAE | Full graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed output grade only | 0.037861 | 0.013430 | 0.027504 | 0.026265 | Not timed |
| Width 16 | 0.067996 | 0.047196 | 0.035881 | 0.050357 | 1.223 ms |
| Width 16 + attention | 0.068771 | 0.064812 | 0.045958 | 0.059847 | 1.301 ms |
| Width 32 | 0.070023 | 0.057092 | 0.033331 | 0.053482 | 1.953 ms |

The complete FP16-network-plus-grade graphs match eager output and have 30 timing
samples each. They exclude D3D12 integration. Native photo trials provide one
retained sparse timing sample each, **5.37–5.41 ms** for the model; these are not
a latency distribution or a native optimization result. The fixed grade omits
the learned effect and is not an accepted replacement.

These findings expose poor transfer beyond the training scene. They do not
separate architecture limits from inadequate data or training. Static photos
are not representative game sequences, and sample rendering/exposure puts
**9.3–27.1% of input channel values** at or above one. This includes values
above one, so the original description of that entire fraction as clipped was
too strong. Comparisons use the actual rendered
input and its native target, not the original photograph. Temporal quality and
perceptual acceptance remain untested. The three cases now inform research and
must not be advertised as a future independent final test set.

The [numeric evidence](../../evidence/neural-model-research/photo-validation.json)
contains source provenance, seven successful captures/28 fenced frames,
fixture checks, both original-scene controls, the deliberate failure check,
all model comparisons and preserved-file hashes. No candidate is installed;
the 3 ms/no-quality-loss objective remains unmet.

```text
make_demo_image_scene.py --output <new-private-scene> [--texture <local-image>]
collect_demo_image.py --demo-dir <configured-hidden-demo> --output-dir <private-trials-root> --scene-dir <private-scene> --capture-dll <private-fenced-build> --capture-sha256 <verified-hash> --label <new-label>
prepare_demo_photo_cases.py --output <new-private-photo-directory>
evaluate_photo_students.py --base <private-trials-root> --photos <private-photo-directory> --plain <width-16-run> --attention <attention-run> --wide <width-32-run> --output <new-private-comparison-directory>
```

## Calibrated photo training extension

The default image-plane emission overexposes the test pattern. Its captured RGB
values are exactly one for **29.01%** of channels and above one for **1.69%**.
Reducing emission to 0.25 lowers the exactly-one fraction to **10.26%**. At
emission **0.1**, the same pattern has no values at or above one and a maximum
of **0.8911**. Only the material emission changes; texture, geometry, camera,
native model controls and full 1920×1080 model extent remain identical.
This calibrates renderer input, not model speed or source-image fidelity.

`make_demo_image_scene.py --emittance 0.1` supports this adjustment. Checks
preserve the default asset bytes, verify that three valid emission changes
alter only the material, and reject eight invalid inputs before creating files.
Captured inputs can still have small negative or above-one filtering overshoots.
Among seven calibrated photos, none has a value exactly one; one channel value
in the lake image exceeds one. The other six photo captures stay below one.

`collect_demo_photo_training.py` records its selection before fitting, prepares
four new training photo identities, and recaptures the three earlier validation
photos at emission 0.1. It retains their original-emission captures as validation
too. Author credits, original dimensions, transformations and source hashes are
recorded; [source terms](../../THIRD_PARTY_NOTICES.md#private-photo-validation-sources)
apply separately from the code license. No images or weights are distributed.

The extension adds four frames to the existing 30 training views. All twelve
validation frames are excluded from fitting: six Sponza views and three other
photo identities at two emissions. Source identity, input hashes, model controls,
capture fences, staged asset hashes and restoration are checked before training
or evaluation. The renderer stays on its private inactive desktop with the exact
hidden executable hash required before each launch.

Pass `--photo-collection <private-extension-manifest>` to
`train_student_collection.py` or `evaluate_student_collection.py` alongside
their existing illumination arguments. The trainer keeps width 16, 4,500 steps,
seed, optimizer, learning-rate schedule, whole-frame loss and output grade fixed.
Mixed-content reporting no longer assumes every capture belongs to one scene.
The evaluation preserves per-group scores and checks saved outputs against
fresh inference. Full student graph timing still excludes D3D12 integration.

```text
collect_demo_photo_training.py --demo-dir <configured-hidden-demo> --base <private-trials-root> --validation-photos <previous-private-photo-directory> --capture-dll <private-fenced-build> --capture-sha256 <verified-hash> --output <new-private-extension-directory>
```

The bounded 34-frame run completed in **85.8 seconds**. Adding four photo
identities improves all six photo validation MAEs, but worsens five of six
Sponza validation MAEs. It does not satisfy the quality requirement.

| Validation group | Previous 30-frame student MAE | Mixed 34-frame student MAE | Change |
| --- | ---: | ---: | --- |
| Six Sponza views | **0.023124** | 0.028038 | 21.2% worse |
| Three photos, emission 0.1 | 0.032443 | **0.026108** | 19.5% lower |
| Three photos, original emission | 0.050357 | **0.028470** | 43.5% lower |
| All twelve validation frames | 0.032262 | **0.027663** | 14.3% lower |

The complete graph remains approximately **1.2 ms**: 1.229 ms for the previous
checkpoint and 1.217 ms for the mixed-data checkpoint, each with 30 samples and
exact eager/graph agreement. The architecture is identical; the small timing
difference is not evidence of an optimization. All twelve new saved validation
outputs and the six previously saved Sponza outputs reproduce exactly.

The mixed student still has higher MAE than the fixed-grade diagnostic on five
of six photo cases. Neither pixel improvement nor the fixed grade establishes
preserved neural detail. The tradeoff shows that data affects transfer; it does
not establish that simply collecting more photos will solve it. Balanced
training, capacity and supervision remain open questions. This is one small,
static-content experiment, with no temporal or in-game acceptance.

[Numerical evidence](../../evidence/neural-model-research/photo-training-extension.json)
includes both calibration captures, seven new photo captures, all 36 new fenced
frames, original/source hashes, matched training settings, all validation scores,
timing samples and preserved game/demo files. **No replacement is installed,
the native model remains about 5.4 ms, and the 3 ms/no-quality-loss goal is unmet.**

## Optimization controls and decoder execution

The mixed-data student's error increased on its training scenes as well as
validation scenes. Two controlled runs therefore keep the 34-frame collection,
width 16, loss, seed and 4,500 additional steps fixed, but use a learning rate
of **0.0002 → 0.000002**, one tenth of the previous schedule. One starts from
random weights; the other initializes from the earlier 30-view checkpoint.
The latter starts a fresh optimizer and retains all thirty old native targets
alongside the four new photo targets. Its total history is **9,000 steps**,
including the original 4,500; it is not a compute-matched from-scratch comparison.

| Model | Mean training MAE | Six Sponza validation MAE | Six photo validation MAE |
| --- | ---: | ---: | ---: |
| Previous 30-frame model | 0.013834 | **0.023124** | 0.041400 |
| Mixed 34 frames, original learning rate | 0.028178 | 0.028038 | 0.027289 |
| Mixed 34 frames, lower learning rate | 0.033189 | 0.029980 | **0.025402** |
| Warm start, lower learning rate | **0.013030** | 0.023212 | 0.035708 |

Training means cover different frame sets in the first row. Warm starting
recovers most scene accuracy and fits the mixed training set better, but loses
much of the photo improvement. None passes the native quality requirement.
These outcomes do not establish a universal learning rate or a capacity limit.
Keeping old examples during adaptation follows the general replay idea studied
in [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682).
This supervised image experiment does not implement that paper's reinforcement
learning algorithm or inherit its results.

`train_student_collection.py` and `student_probe.py` now accept `--initial-lr`,
`--final-lr` and `--initialize-from <private-student-run>`. Initialization checks
architecture, controls, input dimensions and source training hashes. Those hashes
must be a subset of the current training inputs and disjoint from validation.
Only weights are loaded; source hashes and prior training steps are recorded.
Defaults preserve the original training behavior.

The separate execution experiment moves decoder 1×1 projections before nearest
2× feature upsampling, reducing the number of projected positions by four.
An original CUDA kernel combines nearest upsampling with skip addition. It
preserves the FP16 addition rounding and handles strided NCHW inputs. Fifteen
operator cases, including batches and three memory layouts, match Torch bits;
six unsupported input cases are rejected.

Although the projection move is algebraically equivalent, the wider model's
final projection changes GPU rounding. On the first diagnostic image, 0.0199%
of that projection's values change, with a maximum difference of 0.0004883.
Moving all three projections changes each of twelve final width-32 images,
with maximum RGB difference 0.0009766. The fusion itself matches the reordered
Torch path bit-for-bit; the projection change causes the discrepancy.

Keeping the wider model's final projection in its original position restores
exact final output on all twelve checked images. Three width-16 checkpoints
can move all three projections and retain exact output on all twelve images
each. The final controlled run reports:

| Checkpoint | Original full graph | Selected execution path | Full-image comparison |
| --- | ---: | ---: | --- |
| Width 16, original 30 frames | 1.227 ms | **1.139 ms** | 12/12 bit-identical |
| Width 16, lower learning rate | 1.221 ms | **1.146 ms** | 12/12 bit-identical |
| Width 16, warm start | 1.229 ms | **1.143 ms** | 12/12 bit-identical |
| Width 32, original 30 frames | 2.000 ms | **1.928 ms** | 12/12 bit-identical; final projection unchanged |

Every graph includes the full 1920×1080 student and output grade, has thirty
timing samples and matches eager execution. Earlier runs and the failed
all-projection width-32 path are retained in the evidence. Isolated stage
timings varied substantially and are not used as the speedup claim.
These measured student gains are **not** native NVIDIA runtime improvements.

Decoder changes stay disabled by default. Research code can set
`network.reorder_decoder = True`, `network.fused_decoder_backend = kernel`,
and `network.decoder_reorder_stages = (0, 1, 2)` for the tested width-16 path
or `(1, 2)` for the tested width-32 path. Other checkpoints, dimensions and
devices require fresh comparison; these are measured cases, not a universal
equivalence guarantee. No model or execution change is installed in the game.

```text
test_decoder_execution.py --base <private-trials-root> --photos <private-photo-extension-manifest> --model <label> <private-student-run> [--model <label> <another-run>] --output <new-private-report.json>
```

[Complete evidence](../../evidence/neural-model-research/optimization-and-decoder.json)
records the two training runs, all twelve validation comparisons, three decoder
experiments including the rounding investigation, source hashes and preserved
runtime files. No sample or game launch was needed. Native latency remains
about **5.4 ms**; image quality and application integration still prevent the
**3 ms with no quality loss** objective from being considered achieved.

## Native intermediate feature supervision

Matching final RGB alone still leaves a quality gap. This experiment adds a
training loss against captured native decoder features, using a learned 1×1
projection from the small model's last decoder. The idea of intermediate hints
and a projection between differently sized representations comes from
[FitNets](https://arxiv.org/abs/1412.6550). This is a joint auxiliary-loss
experiment, not its full training procedure; classification results do not prove
renderer fidelity. [LIT](https://arxiv.org/abs/1810.01937) is a related research
direction, but its block training method is not implemented here.

Seven photo feature captures reuse the already selected private image scenes.
Every captured RGB input **and** native output matches the earlier photo pair
byte-for-byte. All seven launches use the existing sample, the exact hidden
executable hash and an inactive private desktop. The original scene, runtime
DLL and INI are restored after each capture. Nothing is launched in WuWa.

Before fitting, the captured decoder and skip features are used to replay only
the reconstructed final block. Across the two earlier views and seven photos,
RGB MAE against native output ranges from **0.000131 to 0.000363**. Both earlier
replay outputs reproduce exactly. This supports using the decoded features as
training targets, but does not prove their layout or arithmetic matches native
execution exactly. This diagnostic requires native activations and is **not**
an independent replacement or a speedup result.

The private target preparation crops valid decoder rows to 540×960×32 and
averages 2×2 cells into 270×480×32 auxiliary targets. Channel normalization is
fitted on six training images only: the two earlier views and four training
photos. Three different photo identities provide validation hints. Both RGB
hashes must match the current training or validation split. The original full
1920×1080 RGB inputs and output targets are unchanged; pooling applies only to
the auxiliary features.

The width-16 model uses the same 34 RGB training images, twelve validation
images, seed, 4,500 steps and learning-rate schedule as the mixed-data baseline.
Its loss adds **0.01 × normalized feature MSE** when the sampled training image
has a feature target. A zero-initialized 544-parameter projection is trained
alongside the model, then removed from inference and saved separately in the
private lab. Only six of the 34 training images have these hints.

| Training | Six scene validation views, mean RGB MAE | Six photo validation cases, mean RGB MAE |
| --- | ---: | ---: |
| Original 30-frame baseline | 0.023124 | 0.041400 |
| Mixed 34-frame RGB baseline | 0.028038 | 0.027289 |
| Mixed data with native feature hints | **0.024409** | **0.031071** |
| Earlier warm-start control | 0.023212 | 0.035708 |

Relative to the mixed RGB baseline, hints lower mean scene error by **12.94%**
but increase mean photo error by **13.86%**. Five of six scene cases improve;
three of six photo cases improve and three worsen. Mean training RGB error
falls from 0.028178 to 0.022573. Final normalized feature MSE is 0.4992 on the
six training targets and 0.5120 on the three validation targets. Better feature
fit does not establish preserved output quality, and this one loss weight does
not establish the limits of intermediate supervision.

The inference model still has 223,440 parameters. Its complete 1080p Torch
network and grading graph measures **1.241 ms median**, with thirty samples and
exact agreement with eager execution. It loads without the auxiliary head or
native feature files. All 42 available prior validation outputs reproduce
exactly in the four-model comparison. This excludes application integration;
native latency remains about **5.4 ms**, and the full quality goal is unmet.

`test_feature_hint.py` confirms the auxiliary gradient reaches the student,
rejects all three validation targets from the training loss, checks that targets
are constants, and verifies bit-identical RGB before/after hook removal and a
strict inference checkpoint reload. These are implementation checks, not a
perceptual or temporal quality pass.

```text
collect_feature_image.py --demo-dir <existing-hidden-demo> --base <private-trials-root> --scene-dir <private-image-scene> --capture-dll <checked-private-build> --capture-sha256 <expected-hash> --prefix <fresh-label>
audit_feature_teacher.py --source <pinned-private-reference> --weights <private-weights> --case <label> <private-trial> [--case <label> <another-trial>] --output <fresh-private-directory>
prepare_feature_targets.py --case <label> train <private-trial> --case <label> validation <another-trial> --output <fresh-private-directory>
```

Add `--feature-targets <private-target-manifest> --feature-weight 0.01` to the
existing collection training command to enable hints. They are disabled by
default. `collect_pre_pool.py --single-view` supports the prepared photo scene;
its default two-view collection remains unchanged.

[Source provenance and complete numerical evidence](../../evidence/neural-model-research/native-feature-hints.json)
retain capture checks, both oracle audits, normalization statistics, the training
run, validation regressions and source tests. Weights, raw features and images
remain private. All data are first-reset static frames from the sample; this
does not validate temporal stability or game integration.

## Feature capacity, staged training and fused output

The 16→32 training projection could limit the previous feature-hint experiment.
An analysis of the same private targets separates that limit from prediction
error. Principal components and a shared position template are fitted on the
**six training images only**. Validation remains separate. All projection
results below use the native targets themselves: they are optimistic feature
bounds, not independently predicted images or quality results.

| Normalized feature MSE | Six training images | Three validation images |
| --- | ---: | ---: |
| Best rank-16 projection fitted on training targets | 0.003320 | 0.002401 |
| Oracle projection into the learned head's affine subspace | 0.029774 | 0.020972 |
| Actual jointly trained student | 0.499229 | 0.512021 |
| Position template averaged from training targets | 0.809037 | 0.585340 |
| Oracle per-image channel means | 0.873279 | 0.364070 |

Sixteen principal components retain **99.668% of training variance**. The
actual prediction error is much larger than either projection bound, so simply
widening the projection is not justified by these measurements. This does not
rule out limits in the backbone, training data, feature alignment or objective.
The shared position template also does not explain most training variation.
Per-image channel means use unavailable native information and are not proposed
as an inference method. Projection vectors and templates remain private.

The next experiment separates feature pretraining from RGB fitting, following
the staged-training idea in [FitNets, section 2.3](https://arxiv.org/html/1412.6550v4#S2.SS3).
It uses our renderer losses and architecture, not the paper's classification
objective. First, 1,500 steps train the backbone and auxiliary projection on six
feature targets. The RGB head receives no gradient and remains exactly zero.
The auxiliary projection is then removed, and a separate 4,500-step RGB run
starts from those backbone weights with a fresh optimizer and no feature loss.
That second stage uses all 34 training images and the same twelve validation
images, seed and original learning-rate schedule as the mixed RGB baseline.

Feature pretraining takes 18.59 seconds and lowers mean training feature MSE
from 1.0000 to 0.6163; validation MSE worsens from 0.4197 to 0.4716. The later RGB
stage takes 85.24 seconds and reaches mean training RGB MAE **0.012689**, versus
0.028178 for the mixed RGB baseline. Better training fit still fails to transfer
consistently:

| Training | Six scene validation views, mean RGB MAE | Six photo validation cases, mean RGB MAE |
| --- | ---: | ---: |
| Mixed RGB baseline | 0.028038 | 0.027289 |
| Joint RGB and feature hints | 0.024409 | 0.031071 |
| Feature pretraining, then RGB | **0.023961** | **0.033654** |

Against the mixed RGB baseline, five scene cases improve and one worsens;
**all six photo cases worsen**. Group mean changes are −14.54% and +23.33%.
This candidate is rejected for deployment. It also has **6,000 total training
steps**, so the comparison with a 4,500-step baseline is not compute-matched.
All 42 previously saved validation outputs reproduce exactly in the comparison.

```text
analyze_feature_capacity.py --targets <private-target-manifest> --student <private-joint-hint-run> --output <fresh-private-directory>
pretrain_feature_student.py --targets <private-target-manifest> --baseline-result <private-mixed-RGB-result.json> --base <private-trials-root> --output <fresh-private-directory> --steps 1500
```

For the second stage, use the existing collection training command with
`--initialize-from <private-feature-pretraining-directory>` and omit feature
targets. The pretraining utility only handles the checked six-training,
three-validation, width-16 configuration. It does not launch an application.

The independent kernel change reads the quarter-resolution, 48-channel RGB
head directly. One CUDA operation performs pixel-shuffle indexing, residual
scaling/addition, clamping and color grading. It avoids materializing the dense
residual and separate intermediate images, following the general goal of
reducing global-memory work described in NVIDIA's
[CUDA memory optimization guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations).
The measured benefit below comes from this experiment, not from the guide.

The kernel explicitly preserves both FP16 residual roundings before the FP32
grade. Fourteen finite-input tests include padded/cropped extents, three memory
layouts, batches, broadcast strides, boundary controls and every finite half
pattern in selected source/head combinations. Eight invalid-input cases are
rejected. All checked outputs match the prior fused-grade path bit-for-bit.

| Checkpoint | Existing full graph | Output fusion only | Earlier decoder fusion only | Both fusions |
| --- | ---: | ---: | ---: | ---: |
| Mixed RGB, width 16 | 1.236 ms | 1.132 ms | 1.150 ms | **1.055 ms** |
| Joint feature hints, width 16 | 1.233 ms | 1.140 ms | 1.149 ms | **1.052 ms** |
| Staged training, width 16 | 1.244 ms | 1.141 ms | 1.148 ms | **1.053 ms** |
| Original 30 frames, width 32 | 2.016 ms | 1.929 ms | 1.935 ms | **1.843 ms** |

All four modes produce bit-identical RGB on all twelve validation images for
each checkpoint: **48 complete model/image comparisons**. Each graph has thirty
timing samples and matches eager execution bitwise. Width 32 retains its last
decoder projection in the original position, as required by the earlier rounding
test. These are complete 1080p student-plus-grade graphs; application integration
is excluded. The new output fusion is disabled by default. Research callers can
set `model.fused_student_output = True` after assigning the inference backend;
the decoder controls remain separate. The affine student is not supported by
this output fusion.

```text
test_student_output.py --base <private-trials-root> --photos <private-photo-manifest> --model <label> <private-student-run> [--model <label> <another-run>] --output <fresh-private-report.json>
```

[Complete numerical evidence](../../evidence/neural-model-research/staged-features-and-output-fusion.json)
includes the feature bounds, failed pretraining/generalization results, source
provenance, kernel tests and all timing samples. No sample or game launch was
needed, and no game, driver, native model or normal runtime file changed.
**Native latency remains about 5.4 ms; the 3 ms/no-quality-loss goal remains unmet.**

## Broader image training with a fixed validation split

The earlier student fitted its training images more closely while losing
accuracy on different photographs. Its training set contained only four photo
identities. The next experiment adds twelve training sources covering interiors,
portraits, vegetation, animals, food, landscapes and a night city. Four additional
identities are reserved for validation before any capture or fitting. The
[source catalogue](image-collections/diverse-extension.json) pins all sixteen
files, credits, licenses and crop positions. None overlaps the seven earlier
photo identities by source hash.

`collect_diverse_images.py prepare` downloads bounded source files, verifies
their pinned sizes and SHA-1 values, converts them to RGB and fits a 1920×1080
crop without aspect distortion. It records SHA-256 hashes for the originals,
prepared textures and generated scene manifests. Requests are paced; an explicit
`--resume-preparation` revalidates completed files before continuing an interrupted
preparation. Unrecorded partial artifacts cause an error for inspection.

Collection uses the existing capture build and image-plane helper. Every launch
checks the exact hidden sample executable and runs on an inactive Windows
desktop. The original scene, DLL and INI are restored after each bounded trial.
Four completed, fenced frame captures are checked per image; only the first
reset frame enters fitting or evaluation. The sample's earlier DLSS processing
affects the rendered texture before the neural model receives its actual
1920×1080 input. These are static plane examples, not game or temporal validation.

```text
collect_diverse_images.py prepare --previous-photos <private-photo-manifest> --output <fresh-private-image-directory>
collect_diverse_images.py collect --previous-photos <private-photo-manifest> --output <prepared-private-image-directory> --demo-dir <existing-hidden-demo> --base <private-trials-root> --capture-dll <checked-local-capture-dll> --capture-sha256 <expected-sha256>
```

The existing training and comparison commands accept
`--image-collection <private-image-directory/manifest.json>` alongside
`--photo-collection`. This produces **46 training frames and 16 validation
frames**: all older examples are retained in their original splits. No validation
pixels enter fitting. The new run keeps width 16, 223,440 parameters, seed 28411,
4,500 full-frame steps, the original learning-rate schedule and the RGB/detail
loss. It starts from scratch without feature supervision. The number of training
steps is held constant, so each image receives fewer exposures on average.

`test_student_output.py` accepts the same image-manifest option alongside
`--photos`. It rechecks the existing output and decoder fusions on all sixteen
validation inputs and the supplied checkpoints. This adds coverage; it introduces
no new kernel or application integration.

All sixteen collections completed: **64 fenced frames and 64 observations on
inactive desktops**, with normal sample files restored each time. A source-host
rate limit interrupted preparation after ten downloads; explicit resume checked
those files and continued the unchanged selection with paced requests. No sample
launch was retried. Training completed all 4,500 steps in 99.54 seconds.

| Validation group | Earlier 34-frame RGB model | Earlier staged model | New 46-frame RGB model |
| --- | ---: | ---: | ---: |
| Six original scene views | 0.028038 | 0.023961 | 0.029773 |
| Six older photo cases | 0.027289 | 0.033654 | 0.024322 |
| Four new photo cases | 0.028078 | 0.029687 | 0.021973 |

Values are mean absolute RGB error against native output, lower is better.
Compared with the earlier RGB model, the new model improves all four new photos
and four of six older photo cases. Their group means fall **21.74%** and
**10.87%** respectively. Only two of six scene views improve, and that group's
mean rises **6.19%**. More data helps photo generalization but does not remove the
quality tradeoff. The staged comparison model includes 1,500 extra feature
pretraining steps, so it is not compute-matched to either RGB-only model.
All forty previously saved validation outputs reproduce exactly.

The existing execution fusions remain bit-exact on all sixteen inputs for each
of the three checkpoints: **48 complete model/image comparisons** across four
execution modes. Fourteen operator tests and eight invalid-input guards pass;
all CUDA Graph outputs match eager execution. Each graph uses thirty timing
samples. With both fusions, the new model measures **1.060 ms**, compared with
1.288 ms for its existing graph path in this run. This verifies earlier
optimizations on the new checkpoint; it is not a new native-runtime speedup.
The full 1080p student and grade are included, and D3D12 integration is excluded.

[Numerical evidence](../../evidence/neural-model-research/diverse-image-extension.json)
contains source identities, capture/restoration proofs, split checks, training,
per-image comparisons and all timing samples. No model is accepted or installed.
The native runtime and game remain unchanged, and the **3 ms/no-quality-loss
goal remains unmet**.

## Paired gradients and residual-scale fusion

`analyze_student_gradients.py` checks the fixed 46-frame training set and computes
FP32 RGB/detail-loss gradients at the existing checkpoint trained on 46 frames. It performs
no optimizer update, verifies the weights and parameter gradient fields remain
unchanged, and excludes all sixteen validation cases. Across the thirty scene
and sixteen photo examples, **288 of 480 cross-domain pairs** have negative
gradient cosine. The cosine between the two mean gradients is **−0.929**.
Within-scene and within-photo negative counts are 221/435 and 51/120. This shows
opposition at this checkpoint, but does not prove harmful interference or the
curvature conditions discussed by the research paper.

The controlled training experiment is inspired by
[PCGrad, Algorithm 1](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf).
For two conflicting gradients, each is projected against the original opposite
gradient before averaging. Otherwise their ordinary average is used. Here the
two groups are content domains sharing the same renderer objective, rather than
different prediction tasks. All model parameters are shared.

`train_student_collection.py --paired-gradient mean` is the matched control;
`--paired-gradient pcgrad` enables projection. Both require the full image
extension and use the same model, seed, learning-rate schedule and 4,500 optimizer
steps. Each update samples one of the thirty scene frames and one of the sixteen
photos, computes their gradients in separate passes, and averages them with equal
domain weights. Thus each run processes **9,000 examples**; neither is
compute-matched to the earlier 4,500-example, uniformly sampled model. The
gradient combiner adds no inference operation. Eight small algebra cases and five
invalid-input guards check the two-task rule and ordinary mean-gradient control.

```text
analyze_student_gradients.py --base <private-trials-root> --photos <private-photo-manifest> --images <private-image-manifest> --model <private-46-frame-run> --output <fresh-private-report.json>
test_paired_gradient.py --output <fresh-private-report.json>
```

Separately, `residual_scale_add` combines each student residual block's
per-channel multiplication and skip addition. It explicitly rounds the FP16
product before adding, preserving the previous two-operation arithmetic.
`ResidualBlock.fused_residual_backend` defaults to `None`; the optional backend
is inference-only. Twenty operator cases cover several shapes and layouts,
broadcast strides and every finite half pattern in selected operand combinations,
including products that overflow. Seven invalid-input guards are checked.
The first scalar version was slower despite being exact. The final implementation
uses aligned pairs for contiguous channel-last tensors with 16, 32, 64, 96, 128 or
192 channels, retaining the scalar fallback for other layouts. Explicit
`mul.rn.f16x2` and `add.rn.f16x2` keep multiplication and addition separate;
the [PTX instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions)
documents these operations. Each specialized channel count has a finite-pattern
test, and a misaligned view checks fallback behavior.

`test_student_output.py --residual-fusion` adds residual-only and all-fusion modes
to its earlier four modes. The full-image comparison includes the original
output and decoder fusions and retains the existing width-32 decoder fallback.
This optimization concerns experimental student graphs, not NVIDIA's native
runtime or the installed game.

Both paired training runs completed 4,500 updates: 198.40 seconds for ordinary
averaging and 197.31 seconds for projection. Mean training RGB error fell to
0.020323 and 0.017220 respectively, from the earlier model's 0.029834. The first
losses match exactly. Conflict was observed on 2,321 ordinary-control updates and
2,079 projection-run updates.

| Validation group | Earlier uniform training | Paired mean control | Paired PCGrad |
| --- | ---: | ---: | ---: |
| Six scene views | 0.029773 | 0.026924 | 0.026135 |
| Six older photo cases | 0.024322 | 0.024670 | 0.024762 |
| Four new photo cases | 0.021973 | 0.023107 | 0.023897 |

Values are mean absolute RGB error, lower is better. Relative to the matched
control, projection improves the scene mean by 2.93%, but worsens the photo
means by 0.37% and 3.42%. Three new photos improve individually; the library
regresses enough to worsen that group's average. Better training fit and less
gradient conflict have not established native quality. Both candidates remain
rejected for installation. All 48 saved validation outputs reproduce exactly.

All six execution modes match bit-for-bit across sixteen validation images for
each of four checkpoints, including width 32: **64 complete model/image checks**.
The first generic residual fusion added 2–4% to the previously optimized graph
time. After introducing the contiguous paired path, single-replay measurements
improved, but showed timing variation. A separate comparison alternates the old
and new graph order over thirty pairs after forty warmup replays per mode. Each
interval averages ten consecutive replays:

| Checkpoint | Earlier fusions | With residual fusion | Reduction |
| --- | ---: | ---: | ---: |
| Earlier 46-frame model | 1.180 ms | 1.123 ms | 4.82% |
| Paired mean control | 1.190 ms | 1.136 ms | 4.60% |
| Paired PCGrad | 1.197 ms | 1.158 ms | 3.21% |
| Earlier width-32 model | 2.091 ms | 1.956 ms | 6.45% |

These are full 1080p student-plus-grade graph measurements, excluding application
integration. They are interval averages, not individual-frame tail latency.
Other desktop GPU work remains uncontrolled; timings from different protocols
should not be compared directly. The optional fusion remains disabled by default.

```text
benchmark_student_residual.py --capture <private-validation-capture> --model <name> <private-model-directory> [--model <name> <another-model-directory>] --output <fresh-private-report.json>
```

[Complete numerical evidence](../../evidence/neural-model-research/paired-gradients-and-residual-fusion.json)
retains the training diagnostic, failed quality outcomes, initial slower kernel,
final exactness tests and both timing protocols. No sample or game launch was
needed; normal runtime files and game/driver settings are unchanged. **Native
quality and the full 3 ms target remain unachieved.**
