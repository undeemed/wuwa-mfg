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
