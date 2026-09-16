# Neural latency research: model and kernel experiments

**Target: 3 ms at a 1920×1080 model extent without reducing image quality. This
target has not been achieved.** The current NVIDIA runtime remains about 5.4–6.1 ms in
the separate NVIDIA DLSS Sample. None of the experiments below replaces the
working WuWa installation or changes the installer defaults.

Current work focuses on training a smaller model to reproduce the original
output. Earlier kernel experiments remain documented below. Reduced resolution, skipped
frames, weaker blending and a different visual style do not satisfy the target.

The current [student integration](student-hotswap.md) runs the broadly trained
student in DirectX 12, with live NVIDIA/student selection and weight reload.
It passed 56 image parity checks and 16 hidden-demo switch/reload checks. The
integrated student measured about 3.35–3.78 ms total in that smoke test; live WuWa
performance and final temporal validation remain unmeasured. This is distinct
from the earlier 2.23 ms CUDA-only timing. The integration is installed locally
with a backup; NVIDIA's own runtime binary is preserved.

An earlier [model experiment](../research/neural-latency/README.md#region-routed-context-with-a-matched-dense-control)
adds a region-attention branch inspired by MoBA and BiFormer, with matched dense
and routed training runs. The routed complete model takes **2.463 ms** versus
**2.592 ms** for dense attention in the same timing comparison. It improves the
scene validation group, but worsens both photo groups: overall RGB error is
**4.86% higher** than the frozen first model. FP32 checks confirm that this is not
explained by FP16 rounding. Neither candidate passes the strict quality gate.
The routed model remains the working research candidate after direct image
comparison; a metric regression alone does not quantify perceived quality loss.
The quality gap and complete-pass 3 ms target remain unresolved; these timings
exclude D3D12 integration, and the native runtime is unchanged.

A subsequent [export and reload check](../research/neural-latency/README.md#how-the-routed-student-was-built-and-exported)
preserves the routed model's output bit for bit on all 16 validation images,
including a fresh process that does not import its original model classes.
In this separate paired timing run, the existing optimized path takes **2.090 ms**
and the standard-operator export **3.722 ms**. Export succeeds, but preserving the
existing optimizations remains necessary for deployment. The package is a private
PyTorch artifact, not a native runtime DLL, and no game or sample was launched.

The preceding [sampling experiment](../research/neural-latency/README.md#training-coverage-and-cluster-sampling)
also failed validation. The 62 training captures still come from one 3D scene and
16 photos. Reweighting them or adding selective attention has not resolved the
generalization gap; broader teacher-labeled training content remains the next
data direction. No game installation or new kernel change was made.

The earlier [native batching investigation](../research/neural-latency/README.md#rejected-native-batching-and-verified-overlap-differences)
rejected an eight-kernel batcher after GPU errors and no completed image readback.
It preserved observed barriers and received successful API results, which proved
insufficient. A bounded original producer/consumer test then showed an overlap
difference: a later producer satisfied an earlier waiter in 10/10 individual-call
trials, but 0/10 batched trials. Static native polling loops make changed scheduling
a plausible explanation, not a proven native dependency cycle. The pass-through
API observer still produced four byte-identical reference frames. Generic batching
is retired, normal files are restored, and no native speedup is claimed.

## Measurements from the actual runtime

The [hidden NVIDIA demo](neural-demo-benchmark.md) ran the SF-v2 runtime, Natural
style, one pass, output-relative scale 1.0, at 1920×1080. The physical GPU was an
RTX 4070 Ti with driver 616.92. The demo had no visible main window during both
trace tests. Test processes were stopped after their bounded runs.

A demo-only observer records the NR DLL's NVAPI module/function resolution and
launches. It forwards each complete launch chain unchanged. The first test was
metadata-only; the second added D3D12 timestamp pairs around each chain. Readback
and reuse are protected by a fence signalled after the actual submitting queue's
`ExecuteCommandLists`. Failed signals quarantine the query slot. Discarded,
unsubmitted command lists are handled separately.

The 60-second timing test captured **24 complete evaluations after warmup**, each
containing **158 successful, single-kernel chains**. Their summed GPU intervals
had a median of **6.138 ms** (5.662–6.277 ms). OptiScaler's separate sparse timing
reported 6.16 ms for the model and 6.36 ms for the total pass. Instrumentation
overhead and interference from other GPU work were not subtracted. This is a cost
breakdown, not evidence that instrumentation accelerated the model.

| Kernel family | Launches per evaluation | Median aggregate interval |
| --- | ---: | ---: |
| Fused 8-head / 256-channel window blocks | 12 | 0.847 ms |
| Split 16-head QKV | 16 | 0.472 ms |
| Fused 4-head / 128-channel window blocks | 8 | 0.376 ms |
| Full-resolution output block | 1 | 0.376 ms |
| Full-resolution input block | 1 | 0.350 ms |
| Split 16-head feed-forward | 15 | 0.312 ms |

Individual families varied between samples; for example, the first row ranged
from 0.630 to 0.994 ms. Medians of families need not add to the median of complete
evaluations. [Machine-readable results](../evidence/neural-kernel-timing.json)
include all families and ranges.

FP8-named fused kernels were **actually invoked**, rather than merely found in
the binary. The runtime submitted fatbins instead of bare ELF objects to NVAPI,
so this observer does **not** establish which embedded SM architecture the driver
selected. `sm=0` in raw metadata means unknown. The physical device is SM89.

The cost is spread across the transformer. Eliminating only the two
full-resolution boundary blocks would not supply the roughly 3 ms saving, and
removing them would also change the model. Generic FP8 toggles and substituting
standard FlashAttention are not established shortcuts.

## Native launch arguments and architecture selection

A combined, demo-only capture and launch observer now records a limited set of
scalar arguments alongside the fenced input/output captures. Each of seven
observed evaluations again contained 158 successful single-kernel chains, all
using packed arguments through `LaunchCuKernelChain`. The baseline's time inside
those native calls totalled 0.131–0.165 ms of CPU time per observed evaluation.
These are submission durations, **not GPU execution intervals**; removing them
would not imply the same reduction in the neural pass.

For the four captured frames, the preprocessor's extent was 1920×1080, its noise
counter advanced through 0, 1, 2 and 3, and style was 1/128 with automatic masking
enabled. Tone, structure, masking, depth direction and extent agreed with the
public control values captured for the same evaluations. This confirms the
first-frame noise and style assumptions on these runs. It does not establish
that every inferred argument offset or reconstructed operator is correct.

All nine real module submissions contained an SM89 cubin for the RTX 4070 Ti.
An explicit architecture experiment selected only that existing cubin from each
fatbin, without changing its instructions or weights. The first attempt made no
change: its strict size check rejected allocations with trailing padding. The
corrected selector accepts only zero padding and rejects malformed headers,
unknown trailing data and missing or duplicate SM89 entries. All nine filtered
modules were then accepted by the driver.

| Demo-only test | Modules actually filtered | Sparse model median | Surrounding GPU work |
| --- | ---: | ---: | ---: |
| Original module submissions | 0/9 | 5.415 ms | 0.20 ms |
| Initial selector, rejected by its size guard | 0/9 | 5.440 ms | 0.20 ms |
| Corrected selector, SM89 entry only | 9/9 | 5.430 ms | 0.20 ms |

Each bounded 25-second hidden run retained only two sparse samples after the
approximate warmup. This experiment found **no useful latency reduction** and
was not installed in WuWa. It does not prove which architecture the original
driver path selected. The first two captured frames had identical inputs,
recorded controls/scalars and output bytes between baseline and the corrected
selector. Inputs diverged on frame three; later outputs are excluded from the
paired quality comparison. Two matching frames do not satisfy the quality gate.

The model remains the dominant measured GPU cost. A simple architecture
selection override did not approach 3 ms. The combined probe, selector tests,
sanitized results and reproduction instructions are published in
[`research/neural-latency`](../research/neural-latency/README.md) and
[`native-launch-contract.json`](../evidence/neural-model-research/native-launch-contract.json).

## Recovering an editable model

[MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS/tree/0ca2deab092fe6f3e331bf4f616271dbc64521d0)
provides an independent reconstruction and weight tools. The local study pins
commit `0ca2deab092fe6f3e331bf4f616271dbc64521d0` and uses its Apache-2.0 source.
It does not bundle or distribute NVIDIA weights.

The extractor read the local SF-v2 DLL as data. All 153 packed tensors decoded
into **649 logical tensors**, with no unsupported or opaque tensors. Every shape
matched the pinned source specification. The total tensor element count was
145,755,123; this includes weights and auxiliary tensors, not a claim about
trainable parameter count. The reconstructed graph has 71 blocks.

The DLL SHA-256 was
`6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`.
The local logical tensor file SHA-256 was
`fa6ebc71bc6f91347d51b3368b0ef6e952b6157b6d9a164e7db82cffb88d3143`.
The extraction result establishes layout compatibility, **not** matching image
quality. Upstream's reconstruction has documented residual differences and
control limitations. The matched comparisons below also show residual errors.
Captured vendor outputs remain the authoritative training targets; this
reconstruction is not yet a validated replacement or equivalent teacher.

## Experiments completed in the reconstruction

These are development experiments in isolated Python processes. They are **not
game or NVIDIA-DLL performance results**. GPU event intervals can include idle
gaps while Python enqueues work. Small synthetic inputs only establish a
numerical smoke test, not visual equivalence on gameplay.

1. **Native FP8 conversion for FP16 activations.** The reconstruction's FP16 path
   originally simulated E4M3 conversion with many tensor operations. Native CUDA
   conversion matched values for all 63,488 finite FP16 bit patterns. The only
   bit-pattern difference was negative zero. A `[1,1088,1920,32]` conversion
   measured 14.63 ms versus 1.60 ms. This tensor shape is an operation benchmark,
   not a 1080p end-to-end result. The complete 320×320 synthetic model head was
   numerically identical in the measured comparison.
2. **CUDA Graph replay and constant hoisting.** The first capture failed because
   the reference created a GPU index tensor from a CPU list inside each bias
   permutation. Caching immutable inference weights' permutations before capture
   fixed this. A 320×320 synthetic replay measured 99.90 ms versus 1,173.97 ms for
   the original eager execution in that run, with identical model-head output.
   This makes model research more practical; it remains much slower than NVIDIA.
3. **Fused cosine normalization.** A CUDA kernel reproduces the reference's
   32-channel half-precision fragment reduction, rounding and reciprocal square
   root. It compiles locally for SM89 through PyTorch's bundled NVRTC. Twelve
   test cases covered FP16/FP32, zeros, small values and large norms, with 4,099
   rows each; all matched the reference values. At `[262144,32]`, the isolated
   operation measured 2.568 ms versus 0.0414 ms. This is an operation speedup
   against eager PyTorch, not a speedup against NVIDIA's fused attention.
   With this kernel inside CUDA Graph replay, the complete 320×320 synthetic
   model measured **53.04 ms**, with identical model-head output. The earlier
   replay without this fusion was 99.90 ms; these were separate process runs.
4. **Fused bit-affine softmax.** A second custom CUDA kernel preserves the
   recovered packed-half transformation, reduction, reciprocal rounding and
   E4M3 conversion. It matched 12,182,044 tested values across 92 cases: FP16/FP32,
   even row lengths from 2 to 2048, five input scales and noncontiguous layouts.
   The isolated `[32768,64]` operation measured **0.0332 ms versus 0.6669 ms** for
   the reference. With normalization and softmax fused, the complete 320×320
   synthetic CUDA Graph measured **42.73 ms**, with identical model-head output.
   This improves the research implementation; NVIDIA already uses fused kernels,
   and this is not a measured improvement to the vendor runtime.
5. **Removing a transformer block without training.** Skipping global block 31
   changed the synthetic model head: MAE 0.1945, RMSE 0.3452 and maximum absolute
   error 3.042. These are raw head units, not display RGB metrics. This candidate
   was rejected. Deleting blocks alone is not a quality-preserving model.
6. **Batch independent feed-forward branches.** Eager profiling found 6,270
   matrix-multiply calls in the small reference fixture. A candidate batches
   independent branch products while keeping the original order of additions
   across heads. In one warmed, same-process comparison, the 320×320 CUDA Graph
   fell from **42.57 ms to 18.30 ms**, with identical model-head output. Tested
   FP16 operations matched; FP32 cases had differences, so the comparison tool
   keeps the original implementation for FP32. On the real 1080p captured input,
   the batched FP16 reconstruction matched all **6,220,800 RGB values** from the
   previous reconstruction. This does not remove its existing vendor mismatch.
7. **Fuse the quadratic activation.** A local CUDA kernel preserves the
   reference's half rounding, clamping and quadratic gate. All **8,640,830**
   tested values matched, including every finite FP16 input, FP32 cases and
   noncontiguous tensors. The isolated operation measured **0.0399 ms versus
   0.9605 ms**. Combined with branch batching, the 320×320 graph reached
   **10.86 ms**, with identical head output. At the actual 1920×1080 input,
   padded to 1088×1920, the complete graph still took **226.92 ms**. Its RGB
   output again matched the earlier reconstruction exactly. This full-resolution
   result is far slower than NVIDIA and is not an installable improvement.
8. **Fuse cosine publication and read strided views directly.** Normalization,
   per-head scaling, half rounding and E4M3 conversion now share a CUDA kernel.
   It reads Q/K views without a separate contiguous copy. All 848,000 tested
   FP16/FP32 values matched the reference, including strided channels and large
   scales. A warmed `[256,8,64,32]` graph interval fell from **0.0788 to 0.0225 ms**
   against the previous fused normalization followed by separate scaling and
   casts. Full 1080p reconstruction measured **207.59 ms** (a later run measured
   207.33 ms), with all 6,220,800 RGB values **bit-for-bit identical** to the
   226.92 ms reconstruction. These full-model measurements are separate runs.
9. **Fuse general E4M3 publication.** Profiling that reconstructed graph found
   4,164 CUDA kernels, including repeated clamp and copy/conversion passes.
   Direct saturating conversion removes the intermediate FP8 allocation and
   supports strided inputs. All 11,042,218 tested values matched, including every
   finite FP16 value, FP32 random inputs, saturation, zero strides, scalar/empty
   tensors and nonfinite cases. Every non-NaN output bit matched the previous
   clamp-and-cast path. A `[1,1088,1920,32]` conversion fell from **1.523 to
   0.582 ms**. With both new fusions, the complete reconstructed graph measured
   **178.20 ms**, again preserving all final RGB bits. Its instrumented replay
   contained 3,196 CUDA kernels. This remains far slower than the native runtime.

These kernels use documented SM89 FP8 conversion instructions from
[NVIDIA's PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt).
The conversion fusions preserve the tested reconstruction; they do not resolve
its existing difference from NVIDIA. The latest graph profile still spends
substantial time in conversion, additions, matrix multiplication and activation.
Published evidence includes the isolated tests, full-output equality and the
instrumented kernel families in
[`fused-publication-full1080.json`](../evidence/neural-model-research/fused-publication-full1080.json).

All retained numerical records are in
[`evidence/neural-model-research`](../evidence/neural-model-research).
Source and reproduction instructions are in
[`research/neural-latency`](../research/neural-latency).

## Matched 1080p vendor comparisons

A separate demo-only capture patch records four consecutive evaluations starting
at the first reset. Color, depth and motion are copied before evaluation; output
is copied afterward. The readbacks are mapped only after a fence signalled on
the actual submitting queue completes. Copies restore each resource's expected
state. Recorded controls include style, preset, masking, guide extents and reset.

Two 25-second hidden-demo runs completed all four fenced captures each, with
Natural style, preset 0 and a true 1920×1080 model extent. The first used automatic
masking; the second disabled it only to diagnose reconstruction differences.
Their sparse model intervals were approximately 5.46 ms, with about 0.20 ms
around inference. These capture runs do not demonstrate a speedup over earlier
runs. Each comparison used the exact color and vendor output from its own first
reset frame. Raw textures and previews remain local and are not redistributed.

| Reconstruction diagnostic | Mean absolute RGB error | RGB RMSE |
| --- | ---: | ---: |
| FP16, masking enabled, noise counter 0 | 0.01515 | 0.02237 |
| FP32 reference precision | 0.01545 | 0.02248 |
| FP32 with padded network height 1152 | 0.01479 | 0.02414 |
| Masking disabled in both vendor and reconstruction | 0.01551 | 0.02291 |
| Noise counter 1 | 0.01564 | 0.02309 |
| Noise counter 2 | 0.01542 | 0.02255 |
| Noise counter 3 | 0.01547 | 0.02282 |

Errors are in normalized RGB units, not a perceptual quality percentage. The
first row has 33.01 dB PSNR, 0.99776 RGB correlation and a maximum channel error
of 0.16626. High correlation does not establish equivalence. FP32 did not remove
the gap; larger padding lowered mean absolute error slightly while worsening
RMSE and tail errors. Disabling masking and varying the noise counter also did
not close it. None of these diagnostics was adopted as a fix.

The original visible extent is preserved in every comparison. Default network
padding is 1088×1920; the 1152-height variant adds padding without downscaling.
The original capture manifests do not contain the internal noise counter. The
later combined trace above observes counter 0 on its first reset frame, supporting
the comparison's default rather than the tested counter offsets. The local weight file also differs from the hash
used for upstream's reported golden comparisons. These remain possible sources
of disagreement, alongside incomplete reconstruction behavior.

The full-resolution Python runs took roughly 0.9–1.2 seconds of inference wall
time. They are quality diagnostics, not an optimized runtime or a route already
meeting the latency target. Only the first reset frame has been compared; motion
and temporal quality remain unvalidated. Numeric records, capture hashes and
controls are in
[`matched-vendor-capture.json`](../evidence/neural-model-research/matched-vendor-capture.json).

The native launch grids motivated a second padding check using the same FP16
arithmetic, batched feed-forward layers and fused activation as the 1088-height
reference. Grid dimensions alone do not prove the effective tensor extent;
kernels may guard extra threads. With 1152 network rows, RGB MAE fell from
0.01515 to 0.01435, but RMSE worsened from 0.02237 to 0.02367 and the 99th-percentile
channel error rose from 0.06934 to 0.08044. The visible input stayed 1920×1080.
This did not resolve parity and was not adopted. Its reconstructed graph took
243.4 ms; this is not the native NVIDIA pass. See
[`native-geometry-diagnostic.json`](../evidence/neural-model-research/native-geometry-diagnostic.json).

## First trained student experiments

Two small convolutional students were trained directly against the captured
NVIDIA RGB output. A reversible pixel-unshuffle packs each 4×4 input neighborhood
into channels; the input image is not resized. A depthwise/pointwise residual
network predicts a full-resolution RGB correction through pixel-shuffle. Training
uses pixel and gradient losses. These are initial architecture experiments, not
distilled replacements for individual transformer blocks.

| Candidate | Training | Held-out RGB MAE | Unchanged-input MAE on the same holdout | PyTorch median at 1920×1080 |
| --- | --- | ---: | ---: | ---: |
| 32 channels, 4 blocks, 21,328 parameters | 1,500 steps; left part of one view | 0.05611 | 0.01986 | 0.94 ms |
| 48 channels, 6 blocks, 64,032 parameters | 10,000 steps; two complete views | 0.02249 | 0.01996 | 2.29 ms |

**Both candidates were rejected.** On their respective holdouts, even the
unchanged source was closer to NVIDIA than the student prediction. The first
used a spatially separated region; the second reserved a third camera view for
evaluation. The larger student's held-out maximum channel error was 0.19958.
Its higher training capacity and additional views did not establish preserved
quality. A fast kernel interval cannot compensate for that failure.

The extra views were collected by changing only the demo scene's camera target,
then restoring its original bytes after the run. At position `[0,1.8,0]`, the
original target was `[1,1.8,0]`, the second training target `[-1,1.8,0]`, and the
held-out target `[0,1.8,-1]`. Each hidden run completed four fenced captures.
All are views of **one Sponza scene**, not a diverse scene dataset. No temporal
frames, disocclusions, faces or game UI were used to establish student quality.

The two training runs took approximately 6.8 and 74.1 seconds. The second excluded
a 32-pixel border from crop losses so padded patch edges were not treated as
valid interior predictions. Timings above are 30 warmed FP16 CUDA Graph samples
of the student, including its pixel rearrangement and RGB correction, but
excluding D3D12 integration. They are not measurements in the application and do
not satisfy the target. The small networks, source and numeric failures are
documented; private trained weights and rendered images are not distributed.

See [`student-probes.json`](../evidence/neural-model-research/student-probes.json),
[`student-capture-views.json`](../evidence/neural-model-research/student-capture-views.json)
and [`batched-ffn-full1080.json`](../evidence/neural-model-research/batched-ffn-full1080.json).

## Noise and context experiments in the student

The RGB-only student omitted the three deterministic noise channels used by the
reconstructed model. A new candidate concatenates those channels at their
absolute image coordinates before pixel-unshuffle. The first-reset noise counter
is zero, supported by the native trace. This is a conditioning experiment; the
noise feature values themselves have not been captured from the native kernel.
All runtime controls remain fixed across training and held-out views.

The original 48-channel student's held-out error was also analyzed by spatial
frequency. Approximately **76% of its squared RGB error lies at wavelengths of
64 pixels or larger**; the largest band includes image-wide color differences.
This does not prove why it failed, but it motivates testing wider image context.
The FFT diagnostic checks Parseval energy consistency and reports its periodic
image-border limitation. It is not a perceptual-quality metric.

| Candidate | Receptive field | Held-out RGB MAE | Held-out RGB RMSE | 1080p graph median |
| --- | ---: | ---: | ---: | ---: |
| Previous RGB-only student | 52×52 pixels | 0.02249 | 0.03273 | 2.29 ms |
| Add deterministic noise channels | 52×52 pixels | 0.02407 | 0.03515 | 2.32 ms |
| Noise, wider dilated context and changed training schedule | 172×172 pixels | 0.02476 | 0.03661 | 5.72 ms |

**Both new students were rejected.** The noise-conditioned model improved the
reported training-view MAE from 0.02041 to 0.01535 while worsening the held-out
view. The wider-context candidate also failed on quality and exceeded the timing
target. Because its crop size, batch size, loss border and learning-rate schedule
changed together with dilation, this experiment does not isolate dilation as the
cause. It establishes that this particular combined candidate is unsuitable.

Each candidate has 66,336 parameters and trained for 10,000 steps on the same two
camera views; the third view stayed held out. Training took 80.5 and 57.9 seconds,
respectively. The first used 128-pixel crops, batch 8, border 32 and constant
learning rate 0.002. The second used 256-pixel crops, batch 4, border 96,
dilations `[1,2,4,8,4,2]` and cosine decay to 0.00002. Input image resolution was
not reduced. Timings exclude noise precomputation/input concatenation and D3D12
integration, and do not establish application latency. No temporal or cross-scene
quality has been validated.

The data still cover only one scene. The next model work needs richer teacher
examples and a way to retain broad context efficiently, with held-out image and
temporal checks. Adding noise alone or increasing dilation did not solve it.
Source, training metrics and the frequency diagnostic are in
[`student-noise-context.json`](../evidence/neural-model-research/student-noise-context.json)
and [`student-error-spectrum.json`](../evidence/neural-model-research/student-error-spectrum.json).

## Hierarchical models with more camera views

A four-level convolutional encoder/decoder now tests learned multiscale features,
skip connections and an image-wide context gate. The RGB input is padded and
reversibly rearranged into channels; internal feature maps use learned
downsampling. Every candidate still consumes true 1920×1080 input. Preserving the
input extent does not guarantee that the learned representation preserves quality.

Five additional views were collected in the existing hidden NVIDIA sample: four
diagonal training directions and one translated north-facing validation view.
All 20 frame captures completed behind their GPU fences, with finite, nonconstant
RGB. Training uses only the first reset frame from each view. The scene and demo
DLL were restored afterward. The original north-facing view remained held out;
the translated view was also excluded from training. This is still **one scene**.

| Hierarchical candidate | Training views | Original north MAE | Translated north MAE | 1080p graph median |
| --- | ---: | ---: | ---: | ---: |
| Width 16; 223,440 parameters | 2 | 0.02619 | Not evaluated | 1.20 ms |
| Width 16; 223,440 parameters | 6 | 0.02285 | 0.02318 | 1.19 ms |
| Width 32; 871,792 parameters | 6 | 0.02322 | 0.02375 | 1.94 ms |
| Unchanged input baseline | — | 0.01996 | 0.02187 | — |

**All three models failed quality validation.** Additional views reduced the
first candidate's original-north MAE by about 13%, but both holdouts still favored
the unchanged input. Increasing width improved the reported training-view fit
while slightly worsening both held-out errors. The experiment does not establish
that more parameters solve the quality gap.

All three runs used 1,500 whole-frame steps, batch 1, two residual blocks per
stage, no noise channels, zero loss border, and the same cosine learning-rate
schedule. Training took 21.5, 21.5 and 35.9 seconds. Timing covers 30 warmed FP16
CUDA Graph samples, with graph/eager output equality, but excludes application
integration. No temporal or cross-scene quality claim is supported. These models
were not installed in WuWa or substituted into the NVIDIA sample.

Source and reproduction flags are in the
[research README](../research/neural-latency/README.md#hierarchical-student-and-additional-teacher-views).
Numeric records are in
[`student-hierarchical.json`](../evidence/neural-model-research/student-hierarchical.json)
and [`student-additional-views.json`](../evidence/neural-model-research/student-additional-views.json).

## Native tensor allocation mapping

A demo-only metadata probe matched the preprocessor's known output and weight
addresses to two default-heap buffers created during NR calls. The output address
was 388,096 bytes into a 212,519,936-byte buffer; the weight address was at the
start of a 147,719,680-byte buffer. Both matches held on all seven recorded
evaluations. The registry retains a bounded set of resources to prevent matching
recycled addresses; public records contain IDs and offsets, not raw addresses.

This confirms which allocations contain the two known pointers. It does **not**
establish tensor layout, current resource states or numerical parity. The new
probe reads no GPU buffer contents and adds no GPU commands. The existing fenced
input/output texture capture remained active. Any intermediate capture needs a
separately established state and synchronization contract.

The 25-second trial retained two sparse GPU intervals after warmup: median
**5.435 ms model + 0.200 ms surrounding = 5.635 ms total**, at true 1920×1080.
This is diagnostic evidence, not a speedup. The research build was confined to
the hidden sample and removed afterward; normal R4 source and binaries were
restored. The combined patch was built and checked by applying and reversing it
in a scratch tree. See
[`native-buffer-ranges.json`](../evidence/neural-model-research/native-buffer-ranges.json)
and the [probe instructions](../research/neural-latency/README.md#native-buffer-range-metadata).

## Native first-preprocessor snapshot

The next probe observed the runtime's actual barriers. The weight buffer moved
COMMON → COPY_DEST → UAV during initialization; the scratch buffer moved
COMMON → UAV. Both legacy and enhanced barrier hooks attached successfully.
Observed relevant runtime calls used legacy transitions and global UAV barriers,
with no reported implementation gaps. The event log was bounded at 4,096 entries.

A separately enabled capture then copied a 32 MiB prefix starting at the known
preprocessor output pointer after its first successful launch. The guard checked
the observed state contract, extent, counter, allocation range and hook coverage.
The source state was restored, and the copy was read only after its actual queue
fence completed. Microsoft documents the
[buffer state promotion/decay rules](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12#performance-implications)
used by this restricted capture path. These rules do not make arbitrary resource
states interchangeable.

The snapshot completed. Against the earlier launch-contract baseline, the first
two frames had identical color/depth/motion inputs, controls, recorded scalar
values and output bytes. Later inputs differed and were excluded. A closer-in-time
barrier-only run also had different first inputs, so it cannot support a paired
output claim. The capture run's two sparse post-warmup model samples were both
5.45 ms; this is not a speedup.

The first investigation treated the bytes as an **allocation prefix**. A search
against the reconstructed adapter, first block and pooled output tested 1,080
simple tiled layouts, then 14,040 layouts including split channel axes. The best
fresh-sample correlation in the larger search was only about 0.149, far short of
establishing a match. The later lane-bit decoder below resolves the captured
skip prefix. No numerical equivalence follows from successful readback alone.

This provides a concrete intermediate artifact for investigating the native
kernel's stores and resolving the reconstruction mismatch. Raw bytes and
reconstructed arrays remain private. The combined source patch, reproduction
instructions and numeric evidence are in the
[research README](../research/neural-latency/README.md#guarded-native-preprocessor-capture)
and [`native-pre-tensor.json`](../evidence/neural-model-research/native-pre-tensor.json).

A parallel model experiment added the reconstructed deterministic noise channels
to the six-view width-16 hierarchical student. It used the same 1,500-step
schedule, producing 224,208 parameters and a 1.466 ms graph median. Original-north
MAE worsened to 0.02566 and translated-north MAE to 0.02580, compared with 0.02285
and 0.02318 without noise. **This candidate was rejected.** Its noise values are
not independently proven sample-exact against the native preprocessor, and timing
excludes their preparation and D3D12 integration. See
[`student-hierarchical-noise.json`](../evidence/neural-model-research/student-hierarchical-noise.json).

## First-block layout and missing branch rounding

Private inspection of the SM89 preprocessor revealed 512-byte tiles containing
4×4 pixels and 32 channels. A fixed nine-bit address permutation, validated on
the original view and a separate west view without refitting, decodes every byte
of the captured 32 MiB skip prefix. Both views correlate above 0.9989 with the
reconstruction, but only about 53% of the original reference values match exactly.
This establishes a usable intermediate comparison, not model equivalence.

The next discrepancy was two missing FP8 conversions: the native code rounds
the feed-forward and QKV branch inputs while retaining the unrounded residual
operands. Adding both conversions to reconstructed block 0 reduced prefix MAE
about ninefold, from 0.00763 to 0.000824 on the original view and from 0.00761 to
0.000817 on the west view. Exact bytes rose to about 92%. A CUDA implementation
of the recovered noise formula made essentially no difference by itself, so
approximate noise math does not explain this first-block error.

**The full-image quality check still fails.** With only block 0 changed, matched
original-view RGB MAE rose from 0.01435 to 0.01611, although high-pass correlation
improved from 0.9741 to 0.9810. Better agreement at one intermediate stage can
expose errors elsewhere in an inexact reconstruction. Neither that local result
nor its high correlation proves unchanged final quality. Reconstruction graph
time remains around 190 ms; native model time remains 5.45 ms in the latest
sparse demo samples. The 3 ms target has not been reached.

The correction stays optional and confined to offline research. The published
[decoder, inspection and arithmetic tools](../research/neural-latency/README.md#first-block-lane-layout-and-arithmetic)
and numeric evidence document both the improvement and the failed image test.
The next native checks should isolate downstream publication/accumulation and
the separate pooled output, rather than train another student against an
unverified reconstruction.

The subsequent accumulation test raised first-block exact agreement to
**99.46–99.48%** on both captured prefixes. The key changes were direct FP8
tensor-core instructions with FP16 C/D, initial residuals in the projection
accumulators, and initial attention bias in the QK product. A direct FP16 adapter
gave the same aggregate results as cuBLAS with FP16 accumulation. Twenty-six
synthetic cases independently test matrix layout, tails, batching, strides,
alignment and an accumulator-order cancellation case.

This remains an intermediate result. The full reconstruction with only block 0's
branches replaced produced RGB MAE 0.01582 and took 193.91 ms in a CUDA graph.
It preserves the comparison tool's original input adapter/noise/downstream code;
it is not the same configuration as the isolated best-prefix test. RGB error
remains worse than the earlier 0.01435 baseline, and neither reconstruction
timing nor prefix equality establishes a native speedup. See the
[direct-MMA experiment](../research/neural-latency/README.md#direct-tensor-core-accumulation)
and its numeric evidence. The working game/runtime remains unchanged.

## Separate native pooled features

Two additional hidden-demo captures cover the complete first pooled tensor,
including padded rows. Its layout differs from the full-resolution skip:
two planes of 16 channels, rather than 512-byte spatial tiles. A fixed decoder
now supplies an actual native intermediate target for subsequent model work.

Direct tensor-core arithmetic plus horizontal-pair FP16 averaging matches
99.07–99.09% of its bytes across the two views, with mean absolute error around
0.00012. Padding and remaining arithmetic differences are included in those
figures. This improves the reference for intermediate-feature distillation; it
does not establish an accurate or faster replacement model.

An additional diagnostic fed the exact native pooled tensor into the original
reconstruction. Final RGB error still increased from 0.01435 to 0.01604 on the
same captured image. Its other blocks and full-resolution skip were unchanged,
so correcting this pooled path alone is insufficient. Capturing subsequent
native intermediates and the complete skip can separate those remaining errors.
The real native model still measured about 5.42 ms; the 3 ms quality-preserving
target remains unmet. See the [pooling experiment](../research/neural-latency/README.md#separate-pooled-features-and-a-native-input-diagnostic)
and [numeric evidence](../evidence/neural-model-research/native-pre-pool.json).

## Complete first-block substitution

The complete skip and pool were subsequently captured together in two hidden
demo views. The reconstructed first block matches about 99.34–99.35% of the full
skip bytes, now including previously uncaptured rows. Substituting both exact
native outputs still worsens final RGB error in both views. This moves the next
numerical investigation to the remaining network and output composition; it
does not establish that the first block alone was the cause.

Enabling full FP16 cuBLAS accumulation throughout the reconstruction reduced its
graph time from about 190.6 to 185.2 ms but slightly worsened image error in both
views. With a captured native first block it improved image error slightly,
still far short of matching the native result. Neither candidate is accepted.
The native runtime remained around 5.46 ms for the model and 5.66 ms total.

The [complete-capture experiment](../research/neural-latency/README.md#complete-first-block-capture-and-full-network-accumulation)
documents the capture guards, decoder, paired comparisons and limitations.
Its lower timings when using stored native activations are diagnostic only;
they cannot be reported as a usable renderer speedup.

## Correcting the native output stage

Later inspection found non-neutral exposure, contrast and saturation in the
native post-process. The earlier full-image comparisons above omitted this
stage. Applying an algebraic approximation of the **observed**, unfitted
parameters reduces the baseline reconstruction's mean absolute RGB error from
0.01425 / 0.01647 to **0.00794 / 0.00923** in original / west views.

This also changes the earlier arithmetic comparisons. After grading, the exact
native first-block diagnostic improves those errors to **0.00524 / 0.00591**.
On another precisely matched capture, the branch-rounding variant reaches
**0.00496**, compared with **0.00786** for its graded baseline. The earlier
reports that these variants worsened the final image describe the incomplete
composition pipeline and must not be generalized to the corrected comparison.

These are CPU comparisons of saved reconstructions, not latency improvements.
Every candidate still differs from native output; captured-activation variants
cannot run independently, and the grading implementation is not bit exact.
The new hidden-demo run stayed at **5.425 ms model / 5.635 ms total** from two
sparse warm readings. The 3 ms target remains unmet.

Model research can now account explicitly for the known color transform while
working on the learned detail changes. Whether that permits a smaller model
without losing quality is still untested. See the
[output-grading experiment](../research/neural-latency/README.md#output-grading-changes-the-earlier-image-comparisons)
and [complete numeric evidence](../evidence/neural-model-research/native-output-grading.json).

## Student grading fusion and expanded training coverage

The observed color operations now have a differentiable training implementation
and a fused CUDA inference implementation. The latter matches our Torch
approximation exactly in 17 finite-input cases and the tested complete student
outputs. Its isolated 1080p graph interval is 0.0222 ms versus 1.316 ms for the
separate Torch operations. This does not accelerate NVIDIA's existing kernel.

Adding the known grading to the small six-view student did not improve its
quality. Twelve additional hidden-demo camera captures expanded training to
eighteen views, while preserving the same two validation images. At 4,500
training steps, this reduced validation MAE from **0.02398 / 0.02688** with six
views to **0.01558 / 0.01953** with eighteen. Increasing the student's width then
improved training error but slightly worsened both validation errors.

The complete width-16 student and grading take about **1.22 ms** in Torch CUDA
Graph timing; width 32 takes **1.97 ms**. Neither is a validated renderer:
quality still differs, application integration is absent, and this is one
scene without temporal validation. Repeatedly using these validation cameras
also means they cannot serve as the independent final quality test. No student
was installed in WuWa. The native runtime remains around 5.4 ms.

The [full experiment and numeric evidence](../research/neural-latency/README.md#explicit-grading-in-a-small-student-and-a-fused-gpu-implementation)
record the data split, exact-output checks, training ablations and timing scope.
The next model work needs better generalization and detail preservation, with
broader data and independent sequences; simply adding width is not supported
by this comparison.

## Affine color field and fused composition

The next model experiment added a learned smooth RGB transform alongside the
existing full-resolution detail branch. It was motivated by the previous
model's error spectrum: about 80–83% of validation squared error was at spatial
wavelengths of at least 64 pixels. This is a diagnostic of broad image changes,
not a perceptual quality measurement.

At the same eighteen training views and 4,500 steps, the new model slightly
worsened both validation MAEs to **0.01678 / 0.01988**, from **0.01558 / 0.01953**.
Post-training branch tests show that the affine branch handles much of the
original training view's color change, but that benefit transfers weakly to the
validation views. It remains rejected for deployment.

A new fused kernel interpolates the small coefficient field and composes it
with each original RGB pixel and the full-resolution detail residual. It
matches its Torch reference in 18 finite-input tests and takes **0.0655 ms**
instead of **1.248 ms** for that isolated operation. Together with the existing
grading fusion, the complete affine student takes **1.199 ms** instead of
**3.683 ms**, with identical tested output. These times exclude application
integration and do not improve NVIDIA's native 5.4 ms result.

The [full experiment](../research/neural-latency/README.md#learned-affine-color-field-with-a-full-resolution-detail-branch)
and [numeric evidence](../evidence/neural-model-research/student-affine-field.json)
include the rejected initial interpolation rounding, corrected kernel tests,
training report and branch ablations. This is a reusable research optimization,
not a validated lower-latency replacement model. No game files were changed.

## Pretrained block sensitivity and a fitted replacement

An exact schedule matcher now relates the existing 158-chain native timing
trace to the recovered 71-block model. The correspondence is inferred from
kernel names and order. All 24 complete timing frames match; parser checks reject
changed schedules, invalid intervals and incomplete or failed frames. These
instrumented costs guide research, but native chained launches cannot simply be
deleted because they publish synchronization counters and depend on layouts.

The isolated reconstruction was tested with each of its 59 same-shape blocks
removed, on two matched views. Both unmodified outputs first reproduced earlier
saved reconstructions exactly. **None of the 59 removals improved native RGB
error on both views** across the 118 view tests. The least-sensitive block was 43; the six
least-sensitive blocks removed together raised native MAE from 0.00794 / 0.00923
to 0.01157 / 0.01532. Removing all eight global blocks was worse still, at
0.03720 / 0.04569. Their historical combined native interval is only 0.874 ms,
and that interval is not a guaranteed saving from any valid compressed graph.

A follow-up fit replaced block 43 with an affine feature projection, trained
on four different camera views. It preserved more of the reconstructed
teacher's complete image than simple deletion: RGB change fell from
0.00290 / 0.00345 to 0.00223 / 0.00251. But its errors against native output,
**0.00828 / 0.00958**, remained worse than the unmodified reconstruction.
The candidate is rejected. Improving agreement with an imperfect teacher is
insufficient to establish the requested native quality.

The projection takes 0.0358 ms in an isolated Torch CUDA Graph, versus 0.329 ms
for that reconstructed block. This is not a native renderer speedup: the native
block was already about 0.0722 ms in the instrumented trace. No application was
launched and no game or native DLL was changed. The [full experiment](../research/neural-latency/README.md#pretrained-block-costs-sensitivity-and-feature-distillation)
includes all removals, group checks, fitted projection results, source and
numeric evidence. The 3 ms/no-quality-loss target remains unmet.

## Locating final-block reconstruction error

Two hidden demo captures now include bounded native decoder and skip inputs to
the last neural block. The capture build leaves native launches intact and
requires known extents, nonoverlapping buffer ranges, observed UAV states and
completed submission fences. Original files were restored after collection;
WuWa was not launched or changed.

Replacing both reconstructed inputs with the native captures lowers final RGB
error from **0.007864 / 0.008921** to **0.001110 / 0.001231** on the original and
west views. This indicates that most of the mismatch accumulates upstream.
Direct FP8 MMA, operand rounding, and seeding residuals and attention bias into
the accumulators further lower the diagnostic error to **0.000164 / 0.000179**.
Changing only the final FP16 head has negligible effect.

These low errors depend on captured native intermediate features. With all
inputs computed by the reconstruction itself, the same final-block changes
reach only **0.007683 / 0.008770**. No independently runnable replacement has
achieved native quality, and no native latency gain was measured. Both capture
runs still reported sparse model samples of 5.44 ms. The [full diagnostic](../research/neural-latency/README.md#final-block-inputs-and-fp8-accumulation)
includes the layout checks, output comparisons, regression tests and source.

This distinguishes two requirements: a smaller model must retain the learned
effect, while faster kernels must preserve the intended arithmetic. The first
small models meet the timing budget but fail quality; the reconstructed model
still needs numerical corrections before its output is a trustworthy teacher.
Neither requirement has been solved by the new final-block result.

## Single-head corrections and measured packing optimization

Direct-MMA arithmetic now covers the known single-head blocks 0–4 and 66–70.
Every activation is computed from the input image; native captures serve only
as comparison targets. Correcting the early stages helps much more than
correcting only the last decoder stages. With all changes, native RGB error
falls from **0.007864 / 0.008921** to **0.005167 / 0.005831**. This improves the
reconstruction but does not establish native quality.

A fused activation/FP8 packing kernel preserves every tested output byte in
32 cases, including all finite FP16 input values. Activation plus conversion
falls from 0.818 ms to 0.223 ms on a representative chunk. Plain conversion is
slower in its contiguous microbenchmark, so that result is not generalized to
every operand. In the complete diagnostic final block, fusing all packing is
faster than fusing activation alone, with identical output.

Across all 71 reconstructed blocks and the direct head, packing fusion reduces
GPU graph time from **204.2–204.3 ms to 180.1–180.9 ms**, with identical eager and
graph outputs. These times start at prepared features and exclude output
grading and application integration. They remain far slower than the native
runtime's approximately **5.4 ms**. Neither native quality nor 3 ms is achieved.

The [full experiment](../research/neural-latency/README.md#single-head-arithmetic-and-fused-fp8-operands)
documents the arithmetic progression, packing tradeoff, complete-image checks,
timing scopes and source. No app, game, native DLL, driver or installer setting
was changed.

## Branched arithmetic and shared attention bias

Private inspection located the native 2/4/8-head kernels and confirmed carried
FP16 MMA accumulators in the two-head feed-forward path. A diagnostic adapter
tests that accumulation pattern in the 36 branched window blocks. Results are
mixed: two-head changes reduce native RGB MAE from 0.005167 / 0.005831 to
**0.005066 / 0.005376**, but changing every branched block worsens the original
view. No native quality gate is passed and no model replacement is accepted.

A separate kernel change shares each attention-bias matrix across windows.
It preserves initial-accumulator ordering and every tested output. The larger
attention-score graph falls from **0.247 ms to 0.129 ms**; the complete diagnostic
two-head block improves by about **4%**, from 2.55 ms to 2.45 ms. All 18 complete
candidate images remain identical after this optimization. These are research
implementation timings, not native renderer gains; NVIDIA's approximately
5.4 ms result remains unchanged.

The [full experiment](../research/neural-latency/README.md#branched-blocks-and-shared-attention-bias)
includes inspection provenance, the nine-candidate comparison, seed mapping
tests, regression checks and explicit timing scopes. No application, game,
native DLL, driver or installer setting was changed.

## Compact student with global attention

A new student adds spatial self-attention to the previous model's deepest
learned features. It retains full 1920×1080 inputs and detail paths. With the
same 18 training views and 4,500 updates, mean training error improves from
0.011933 to **0.010733**, but validation MAE worsens from 0.015580 / 0.019533 to
**0.021237 / 0.019931**. Some other metrics improve; overall quality acceptance
still fails. Disabling the fitted branch improves both validation MAEs while
worsening the original training view, consistent with overfitting.

The complete FP16 network plus output grade measures **1.317 ms**, versus
1.229 ms for the previous checkpoint in the same comparison. Reloaded images
and graph outputs match their saved/eager counterparts exactly. These are
offline Torch timings, not an application result. No validated replacement is
available: inference speed is already adequate for these small candidates,
but retaining the native effect across views remains unresolved.

The [experiment and evidence](../research/neural-latency/README.md#global-attention-in-the-compact-student)
document the matched training setup, branch ablation, implementation checks
and limitations. No application, game, native DLL, driver or installer setting
was changed.

## Expanded lighting data and sparsity feasibility

Sixteen additional hidden-sample captures provide 12 new training views and
four reserved validation views under varied sunlight. All 64 captured frames
completed their GPU fences; the first-reset images extend the dataset to
30 training and six validation views. Original scene, DLL and INI bytes were
restored, and no game or driver changed.

At the same 4,500 training steps, the width-16 student's mean pixel error on
the four new views falls from **0.031313 to 0.025775**, about **17.7%**, while
full-network-plus-grade timing remains approximately **1.2 ms**. Individual
views have mixed results, and attention still does not improve the combined
validation score. A width-32 candidate fits training better and runs at
**1.96 ms**, but worsens all four new validation MAEs. No quality gate is passed.

The [experiment and evidence](../research/neural-latency/README.md#illumination-data-and-reserved-camera-views)
include split checks, all checkpoint comparisons and timings. The additional
validation views now inform model choices; they are not a final independent
test. One scene and first-reset frames cannot establish temporal or cross-scene
quality preservation.

A read-only kernel-feasibility check also rules out automatic conversion of
the inspected weight tensors to 2:4 storage at their existing shapes: only
**5.03%** of 143.8 million inspected weight values are zero, and none of the
360 tensors passes the format checks. The [sparsity audit](../research/neural-latency/README.md#exact-zero-feasibility-for-structured-sparsity)
documents its scope and the NVIDIA format reference. No sparse kernel was
installed or benchmarked; pruning would require changing the model and
validating its quality.

## Testing content beyond the training scene

A new, tiny image-plane scene lets the existing sample generate native targets
for unfamiliar content. Three reserved photographs were tested against the
frozen 30-view students, with no retraining. Every student has worse pixel MAE
than the fixed-grade diagnostic on every photo. The plain student's mean MAE
is **0.050357**, versus **0.026265** for the grade-only diagnostic, while its
complete Torch graph runs at **1.223 ms**. Speed is available; preserving the
learned effect remains unresolved. A color grade is not a substitute for that
effect, and neither result passes acceptance.

The [experiment](../research/neural-latency/README.md#unseen-photo-content-and-contained-background-launches)
records attribution, private-data preparation, matched native inputs/targets,
all three model comparisons and limitations. These are static rendered photos;
sample exposure pushes some channels to or above one, and there is no temporal or game-quality
acceptance. The new cases now inform research rather than remaining a final
independent test set.

The same work fixes a background-launch gap: hiding the sample's main window
did not contain load-error dialogs. The runner now places the verified sample
on a private Windows desktop that is never activated. A deliberate load failure
verified dialog containment and file restoration. An original-scene repeat
matches the historical input/output hashes exactly; an earlier slightly
different capture is retained as well. The native runtime, game and driver
remain unchanged, and the approximately **5.4 ms** native baseline still exceeds
the target.

## Calibrated input and mixed photo training

Adding four different training photographs improves all six photo validation
MAEs, but worsens five of six Sponza validation MAEs. At the same width, seed,
4,500-step budget and optimizer, average photo MAE falls **34.1%**, while average
Sponza MAE rises **21.2%**. The complete student graph remains about **1.2 ms**.
This is useful evidence that the training data affects generalization; it is
not an accepted model or proof that additional photos alone solve quality.

Before collection, emission was calibrated using the same generated pattern.
At emission 0.1 its captured maximum falls below one; no texture, geometry,
camera or model control changes. Seven new photos use that setting. Their
inputs have no exactly-one channels, although the lake has one above-one value
and small negative filtering overshoots remain. The earlier description of
all `>=1` values as clipped was too broad. Captured renderer input, rather than
the original photograph, remains the reference for every comparison.

Four new photo identities enter training. Three previous identities remain
excluded at both original and calibrated emission, alongside all six Sponza
validation views. The [experiment and reproduction commands](../research/neural-latency/README.md#calibrated-photo-training-extension)
and [complete numeric evidence](../evidence/neural-model-research/photo-training-extension.json)
record the improvements and regressions, all 36 new fenced frames, source
attribution, output reproduction and preserved files. No native kernel change,
game modification or model installation occurred. The native **5.4 ms** baseline
and the unmet **3 ms with no quality drop** objective remain unchanged.

## Training stability and decoder work

Lowering the mixed-data learning rate alone improves photo agreement but further
hurts scene agreement. Initializing from the earlier 30-view student recovers
most scene accuracy, yet sacrifices much of the photo gain. Both complete
4,500 additional steps; the warm start has 9,000 steps of total training history.
All previous validation identities remain excluded. This isolates a training
tradeoff, not a successful native-quality model.

A separate decoder change reduces width-16 full-graph time from roughly
**1.23 ms to 1.14 ms**, with bit-identical outputs across three checkpoints and
twelve images each. It moves pointwise projections ahead of nearest feature
upsampling and fuses the upsampling/skip addition. The width-32 model requires
keeping its final projection in the original position to preserve output;
that path reduces **2.00 ms to 1.93 ms**, with twelve matching images. Moving
all its projections is faster but changes rounding and is not treated as exact.

The [implementation and results](../research/neural-latency/README.md#optimization-controls-and-decoder-execution)
and [numeric evidence](../evidence/neural-model-research/optimization-and-decoder.json)
retain both failed and successful execution comparisons. These optimizations
stay disabled by default and apply only to experimental student execution.
They do not improve native NVIDIA latency, establish native image quality or
include D3D12 integration. The full goal remains unmet.

## Feature capacity and final-output fusion

An analysis of the native feature targets finds that sixteen principal
components retain 99.668% of training variance. The previous student's feature
prediction error is much larger than this projection limit. A shared position
template also leaves substantial error. These are diagnostic bounds using
native targets, not deployable predictors or proof of native quality.

A two-stage experiment pretrains against feature targets, then fits RGB with a
fresh optimizer. It substantially improves training fit but still worsens all
six held-out photo cases against the mixed RGB baseline. The extra pretraining
steps and all regressions are documented; the model is rejected for deployment.

A separate CUDA fusion combines final pixel rearrangement, residual composition
and grading. Together with the earlier decoder fusion, complete 1080p student
graphs fall from about **1.24 to 1.05 ms** for width 16 and **2.02 to 1.84 ms**
for the checked width-32 model. All 48 model/image comparisons are bit-identical.
Fourteen operator cases, eight input guards and thirty samples per graph are
recorded. Fusion remains disabled by default and excludes app integration.

The [implementation and full results](../research/neural-latency/README.md#feature-capacity-staged-training-and-fused-output)
and [numeric evidence](../evidence/neural-model-research/staged-features-and-output-fusion.json)
preserve both the speedup and quality failures. Native NVIDIA execution remains
about 5.4 ms, and no game or normal runtime changes were made.

## Broader native-supervised image data

The earlier training set contained only four photo identities. Twelve new
training sources and four reserved validation sources now cover a wider range
of interiors, portraits, vegetation, animals and landscapes. Their identities,
licenses, hashes and crop positions were fixed before captures and fitting.
The complete set contains **46 training frames and 16 validation frames**.
All older examples retain their original splits.

Sixteen bounded sample runs produced 64 completed fenced captures. Every launch
verified the exact hidden executable and ran on an inactive desktop; all 64
desktop observations passed. The original scene, DLL and INI were restored after
each run. Only the first-reset frame enters this experiment. Photo planes pass
through the sample's earlier DLSS processing before the neural model receives
its true 1920×1080 input; they do not establish gameplay or temporal quality.

The new student keeps the original architecture, width 16, 223,440 parameters,
seed and 4,500-step RGB training budget. No feature pretraining or auxiliary loss
is used. Relative to the earlier 34-frame RGB model:

| Reserved group | Earlier mean RGB MAE | New mean RGB MAE | Change |
| --- | ---: | ---: | ---: |
| Six original scene views | 0.028038 | 0.029773 | 6.19% worse |
| Six older photo cases | 0.027289 | 0.024322 | 10.87% better |
| Four new photo cases | 0.028078 | 0.021973 | 21.74% better |

All four new photos improve, but four original scene views regress. This supports
expanding content coverage as a useful training direction; it does not establish
preserved quality. The fixed step budget also gives each image fewer exposures
on average, and one seed cannot establish a general model or data limit.

The earlier output and decoder fusions remain bit-exact across all sixteen
validation inputs and three checkpoints. The new model's complete student-plus-
grade graph measures **1.060 ms** with both fusions, excluding D3D12 integration.
No new kernel or native-runtime acceleration is claimed. The model is not
accepted or installed. [Method and commands](../research/neural-latency/README.md#broader-image-training-with-a-fixed-validation-split)
and [numerical evidence](../evidence/neural-model-research/diverse-image-extension.json)
record the data, training, comparison, exactness checks and timing samples.

## Gradient interference and residual fusion

A diagnostic on all 46 training frames found negative gradient alignment in
288/480 scene/photo pairs. The two mean gradients had cosine −0.929. This is an
observation at one checkpoint, not proof that gradient interference caused the
quality tradeoff. No validation pixels or weight updates entered the diagnostic.

A controlled two-domain projection experiment then compared ordinary gradient
averaging with PCGrad. Both models use the same architecture, seed, 4,500 updates
and 9,000 sampled training examples. Each update pairs one scene frame with one
photo. The projection model improves training fit and reduces scene validation
MAE by 2.93% relative to the matched control, but photo-group errors rise by 0.37%
and 3.42%. Neither model passes the quality gate. The older uniformly sampled
model used only 4,500 examples, so gains over it cannot be attributed solely to
the projection method.

The independent kernel experiment combines per-channel residual scaling and
addition while preserving both FP16 roundings. A generic scalar version was
exact but slower. A specialized path for aligned, contiguous channel-last pairs
removed most indexing overhead. Twenty operator cases and seven invalid-input
guards pass, including boundary patterns, specialized widths and a misaligned
fallback. All six execution modes are bit-identical on sixteen images for four
checkpoints: **64 complete model/image checks**.

Alternating the previous and new graph paths over thirty pairs, with ten replays
per interval, measures **3.21–4.82% less GPU time** for the width-16 students and
**6.45% less** for the tested width-32 model. Complete student-plus-grade interval
averages are about 1.12–1.16 ms and 1.96 ms respectively. This excludes D3D12
integration; it does not accelerate NVIDIA's native runtime. The fusion remains
an optional research path, disabled by default.

[Methods and commands](../research/neural-latency/README.md#paired-gradients-and-residual-scale-fusion)
and [numerical evidence](../evidence/neural-model-research/paired-gradients-and-residual-fusion.json)
include the rejected scalar timing, both training controls, exactness checks and
timing variation. The game, native model, normal runtime files and driver settings
remain unchanged.

## Direct decoder conditioning

Training-only diagnostics found that a native-target-derived constant RGB
correction explains only about 1–4% of the models' errors. The ordinary model's
context gate is also largely unsaturated. This does not support treating the
problem as a simple global color bias or a stuck gate.

A new branch instead supplies pooled image information directly to the decoder
features at three scales. Its feature-wise scale/shift projection starts at zero,
and insertion tests reproduce the earlier checkpoint exactly on the checked
FP32/FP16 inputs. The branch adds 31,232 parameters, bringing the width-16 model
to 254,672. It retains the same data, seed and 4,500-step ordinary training
schedule; no validation pixels enter fitting.

Against the matched ordinary baseline, scene validation MAE falls **4.54%** and
older-photo MAE falls **12.16%**, with all six older photo cases improving. The
newer-photo group is mixed and worsens **0.61%** on average. The model therefore
remains below native quality and is not installed.

All existing fusion modes remain bit-identical across 48 complete model/image
checks. The candidate's fully optimized 1080p student-plus-grade graph measures
**1.0407 ms** in the alternating interval benchmark, excluding application
integration. This revalidates existing kernels on the new architecture; it does
not accelerate NVIDIA's native runtime.

[Method and commands](../research/neural-latency/README.md#direct-image-conditioning-of-decoder-features)
and [numerical evidence](../evidence/neural-model-research/decoder-conditioning.json)
record the diagnostics, training, regressions and timing scope. Game, driver and
normal runtime state remain unchanged.

## Capacity test and conditioning fusion

Doubling the conditioned model's width increases its parameters from 254,672 to
995,696. With the same data, seed and 4,500 updates, older-photo mean error falls
2.36% and newer-photo error falls 0.51%, but scene error rises 1.20%. Mean training
error also rises slightly. Extra width alone does not resolve the quality gap.

A new optional kernel combines the decoder skip and conditioning operations while
preserving all four FP16 rounding steps. Forty-two operator cases and 48 complete
model/image comparisons pass bit-exact checks across eight execution modes.
In alternating full-1080p graph measurements, it reduces the smaller conditioned
model from **1.0389 to 0.9958 ms** and the larger one from **1.7895 to 1.7350 ms**.
These are improvements to the experimental students, excluding application
integration; the native runtime is unchanged and neither student meets its quality.

[Method, commands and limits](../research/neural-latency/README.md#conditioned-capacity-and-decoder-fusion)
and [numeric evidence](../evidence/neural-model-research/conditioning-capacity-and-fusion.json)
include the mixed quality results, unchanged timing control and exactness scope.
The new fusion remains disabled by default. Game and driver state are unchanged.

## Spatial context through a small latent array

A training-only diagnostic finds that region-specific RGB offsets explain much
more error than one image-wide offset. These are target-derived oracle values,
not usable runtime corrections. They motivate a spatial-context experiment but
do not prove that missing context is the cause of the quality gap.

The new branch uses 32 learned slots to exchange information between spatial CNN
features, with xy coordinates and read/process/write attention inspired by
[Perceiver IO](https://arxiv.org/abs/2107.14795). It starts neutral and adds 153,120
parameters. With the same data and 4,500 updates, newer-photo error falls 3.93%,
but scene and older-photo error rise 2.25% and 4.93%. Temporarily removing the
trained branch confirms it changes the features and modestly helps training fit.
The model still fails native-quality acceptance.

Its optimized full-1080p graph takes **1.0977 ms**, excluding application
integration. Existing kernels remain bit-exact across 48 complete model/image
checks and eight execution modes; no new kernel or native-runtime acceleration
is claimed. The [method and commands](../research/neural-latency/README.md#spatial-error-and-latent-context-exchange)
and [numeric evidence](../evidence/neural-model-research/latent-context-exchange.json)
include all comparisons and their limits. Game and driver state remain unchanged.

## Paired conditioned training and native batching coverage

The conditioned model still has some opposing scene/photo gradients. Two matched
4,500-update runs compare ordinary paired averaging with PCGrad, using the same
architecture and 9,000 sampled examples each. Both improve scene error relative
to the ordinary conditioned baseline, but worsen both photo groups. Projection
reduces the mean control's large older-photo regression while worsening scene
and newer-photo errors. Neither run passes the quality requirement.

The strongest regression occurs on the brighter landscape case. All sixteen
training photo identities use emission 0.1, while validation includes 1.0 as well.
This is a concrete coverage gap for a future training-data experiment, not proof
that brightness is the sole cause. Existing kernels remain bit-exact across 64
full-image comparisons and eight modes. The two optimized graphs take about
1.00 ms, excluding application integration; the native runtime is unchanged.

A separate read-only audit of existing native traces finds sixteen logged global
UAV barriers among 158 successful single-kernel calls in six uncensored sampled
frames. The marker cannot distinguish barriers inside a launch from those
between launches, and the observer covers only selected resources. This trace
therefore does not establish executable batches or a GPU speedup. The
[method and results](../research/neural-latency/README.md#paired-training-of-the-conditioned-model)
and [numeric evidence](../evidence/neural-model-research/paired-conditioning-and-launch-audit.json)
record both investigations and the next evidence needed. No application was
launched or game/driver state changed.

## Brighter training coverage and native launch ordering

A matched data experiment addresses the missing high-emission training photos.
Sixteen new native captures reuse existing training identities; validation
identities remain excluded. Both models use identical 4,500-update paired-mean
schedules. One repeats the dim captures in its additional photo slots, while the
other substitutes the brighter captures. Brighter data reduces average native
target error by 4.37% on scene views, 24.24% on older photo cases and 12.50% on
newer photos. Twelve of sixteen individual cases improve; four regress.

The full 1080p student-plus-grade graph remains around 1.004 ms, excluding
application integration. Existing fusions are bit-exact on all 64 model/image
pairs across eight modes. This is a useful data improvement, but it does not
establish unchanged perceptual or temporal quality, and no model is installed.

A new native observer resolves the earlier barrier ambiguity. Seven complete
sampled evaluations each have 158 successful launch scopes and twenty intercepted
UAV barriers outside the native launch API: nineteen between launches and one
after the last. The first-reset frame's inputs and output are byte-identical to
the capture control. Later frames have different inputs and cannot test output
equivalence. Other command coverage, argument lifetimes and multi-kernel ordering
still need verification before batching. No native acceleration is claimed.

The [data method and results](../research/neural-latency/README.md#matched-native-brightness-training),
[observer procedure](../research/neural-latency/README.md#native-launch-scopes-and-complete-intercepted-barrier-calls)
and [numeric evidence](../evidence/neural-model-research/brightness-training-and-launch-order.json)
document both experiments. All eighteen sample launches used the inactive private
desktop; game, driver and normal runtime state remain unchanged. The full goal
of 3 ms at 1920×1080 with unchanged native quality remains unmet.

## Output-clamp training and a native dependency test

The brighter-data model clips about 13.8% of channels on the brighter training
images. Most clipped channels have an output-space gradient pointing back into
range, but they account for only 8.05% of RGB error. A training-only
straight-through gradient preserves the exact forward clamp and inference path.
Against the ordinary control, it improves scene validation error 1.63% but
worsens older-photo error 19.19% and newer-photo error 4.53%. Eight individual
cases improve and eight regress. The candidate is rejected; the optional training
flag stays off by default.

Its full-1080p student-plus-grade graph remains around 1.001 ms, excluding
application integration. All 48 model/image comparisons retain bit-exact results
across the existing eight execution modes. Fast inference is already possible
for these small models; preserving the native effect remains the unresolved part.

A separate test submits original integer kernels through NVIDIA's native
multi-kernel launch API, using fresh buffers and a separate CPU reference. All
24 tested mode/size/length combinations finish with zero mismatched elements.
This supports dependency ordering for those finite workloads on this GPU and
driver. It does not prove that the native model's calls can be merged safely or
that merging them would meet 3 ms. No native kernels were modified or accelerated.

The [training method](../research/neural-latency/README.md#training-gradients-through-the-output-clamp),
[native selftest](../research/neural-latency/README.md#native-multi-kernel-dependency-selftest)
and [numeric evidence](../evidence/neural-model-research/clamp-training-and-chain-selftest.json)
document the completed experiments. The single sample launch used the inactive
private desktop; WuWa and the normal runtime remain unchanged. The full target
is still unmet.

## Spectral training and controlled native argument forwarding

Broad spatial frequencies account for roughly 64–69% of squared training error
in the brighter-data model. A training-only spectral loss, calibrated without
validation data, worsens all three validation groups by 13.87%, 28.85% and 10.87%.
It is rejected and remains off by default. Its graph still takes about 1.003 ms;
the unresolved problem is faithful output, not student inference speed.

The native observer now traces 75 command-list methods alongside both barrier
methods. It finds a descriptor-heap change after launch 1 and confirms extensive
reuse of packed argument storage: 144 reused buffers change within the first
evaluation. Deferred calls therefore need their own argument storage and must
preserve the command boundaries.

A new fixed-input replay test removes differences in the demo's rendered inputs
and prior history. From reset, all four captured frames have identical inputs
across three modes. Both command tracing with original arguments and tracing with
copied arguments reproduce the native reference output byte-for-byte on every
frame. This is a controlled correctness result, not a batching or speed result.
No real-model kernel calls have been merged yet.

The [spectral experiment](../research/neural-latency/README.md#training-against-spectral-errors),
[native method and replay procedure](../research/neural-latency/README.md#native-command-coverage-argument-ownership-and-fixed-input-replay)
and [numeric evidence](../evidence/neural-model-research/spectral-training-and-native-input-replay.json)
document the completed work. Five hidden sample launches used the inactive
private desktop. WuWa, driver settings and the normal runtime remain unchanged;
the full 3 ms and unchanged-quality goal remains unmet.

## Papers and what can transfer

The joint [native feature supervision experiment](../research/neural-latency/README.md#native-intermediate-feature-supervision)
adds training-only targets from the native decoder. Against the mixed RGB
baseline, it reduces mean scene validation error by 12.94% but increases photo
error by 13.86%. The inference model stays near 1.24 ms in a complete Torch
graph; its extra training projection is removed. It still fails native quality.
The [numeric evidence](../evidence/neural-model-research/native-feature-hints.json)
includes all regressions, capture checks and split/gradient/inference tests.
Seven new sample captures ran on an inactive private desktop and restored the
original files. WuWa and the normal runtime remain unchanged.

These papers provide research ideas. Their reported speedups are on other models
and hardware; none establishes the target for this runtime.

| Work | Relevant idea | Constraint in this renderer |
| --- | --- | --- |
| [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf) | Improve GPU work partitioning and avoid materializing intermediate attention matrices. | The recovered attention uses cosine normalization, explicit rounding and an approximate softmax. A standard attention replacement changes those operations; custom fusion must preserve them. |
| [SageAttention2++](https://arxiv.org/html/2505.21136v3) | Use faster FP8 matrix instructions with FP16 accumulators and manage numerical range. | The inspected native code already uses this instruction family. The paper supports investigating accumulation and data movement, but its speedup over another attention implementation cannot be applied to this renderer. Our direct-MMA result improves a numerical reference, not native latency. |
| [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) | Calibrate activation/weight scaling before lower-precision execution. | This runtime already invokes FP8-named kernels. LLM INT8 results do not imply a further lossless gain, and calibration must include renderer activations and controls. |
| [Knowledge distillation](https://arxiv.org/abs/1503.02531) | Train a smaller student against the larger model's behavior. | A renderer needs matched pixels, detail and temporal consistency, not just matching classification probabilities. A trustworthy teacher and held-out sequences are prerequisites. |
| [MPRNet](https://arxiv.org/html/2102.02808v1) | Refine an image through supervised residual stages that retain access to the original input and exchange features. | The RGB cascade barely improves validation error. A subsequent shared-feature decoder improves training fit but worsens validation. Neither implements MPRNet or establishes fidelity from its restoration results. |
| [Tiny Recursive Models](https://arxiv.org/html/2510.04871v1) | Revisit a predicted answer with a small network and retained latent state. | Puzzle accuracy and parameter efficiency do not imply low image-processing latency. Reviewed as motivation for refinement; no TRM or recursive inference implementation is included, and every additional pass must fit the full latency budget. |
| [Gradient Surgery / PCGrad](https://arxiv.org/abs/2001.06782) | Adjust conflicting training gradients without adding inference work. | Our matched two-domain test improves scene error but worsens photo error. Negative alignment alone does not prove the paper's full conditions or guarantee renderer quality. |
| [FiLM](https://arxiv.org/abs/1709.07871) | Condition feature channels using learned affine transformations. | Our image-conditioned decoder improves scene and older-photo errors but slightly worsens the newer-photo mean. Visual-reasoning performance does not establish native renderer quality. |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | Exchange spatial information through a compact latent array and output queries. | Our small CNN branch improves newer-photo error but worsens the other groups. Its efficient attention interface does not establish equivalent renderer quality. |
| [FitNets](https://arxiv.org/abs/1412.6550) | Use intermediate teacher features and a learned projection to guide a smaller student. | Our joint auxiliary loss improves scene errors but worsens photo errors. It does not implement the full FitNets procedure or establish renderer quality from classification results. |
| [LIT](https://arxiv.org/abs/1810.01937) | Train shallower blocks with intermediate teacher inputs and targets. | This may avoid unstable student inputs during block training, but it is not implemented here. A deployable replacement must run without captured native activations. |
| [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682) | Retain prior examples when adapting a model to new data. | Its reinforcement-learning results do not establish pixel fidelity. Our warm-start experiment retains old native targets, but still trades photo accuracy against scene accuracy; it is not an implementation of CLEAR. |
| [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) | Rank layer-removal sensitivity, then fine-tune to repair changes. | Its LLM question-answering results do not establish pixel fidelity. Our 59-block sweep found no single removal improving both native-view errors; the first fitted feature projection also failed native quality. |
| [EfficientViT](https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf) | Hardware-friendly multiscale operators for dense, high-resolution prediction. | Replacing attention with linear attention changes the learned function; it is a student architecture to train and validate, not an interchangeable kernel. |
| [TinyVLA](https://arxiv.org/html/2409.12514v3) | A compact backbone and task-specific decoder can reduce inference cost. | Fast robot action prediction does not require reproducing every image pixel. Borrow compact architecture design and task-specific training, not its quality claims. |
| [V-JEPA 2](https://arxiv.org/html/2506.09985v1) | Predict compact representations; curate and weight training data using feature clusters and target distributions. | Semantic latent accuracy does not establish pixel fidelity. Our small TRAIN-only cluster-sampling test worsens validation; it does not implement the paper's retrieval pipeline or transfer its benchmark gains. |
| [ToCa](https://arxiv.org/html/2410.05317v1) | Selectively reuse features based on redundancy and error sensitivity. | Its reuse is across diffusion steps. This effect already uses one pass; across-frame reuse adds motion, disocclusion and noise-state problems. It must not simply retain an old rendered image. |
| [Edge-Efficient Image Restoration](https://arxiv.org/abs/2605.02794) | Train replacement blocks against intermediate features, select combinations, then fine-tune the whole model. | Its transformer/SSM experiments use other restoration tasks and hardware. Here, the initial small CNNs fail the held-out image test; replacement blocks would need to retain learned context and be validated against actual vendor output. |
| [Simple Baselines for Image Restoration / NAFNet](https://arxiv.org/abs/2204.04676) | Simple multiplicative gates can replace some expensive nonlinear components in a trained restoration architecture. | This suggests a student design; substituting gates into the trained vendor graph would change its function. No NAFNet replacement has been validated here. |
| [Real Image Denoising with Knowledge Distillation for High-Performance Mobile NPUs](https://arxiv.org/abs/2605.03680) | Choose operators for the target device and expand training context while distilling a smaller network. | Its mobile-NPU denoising metrics do not prove renderer quality on Ada GPUs. Our wider-context candidate still failed, so the paper's success cannot be transferred without evidence. |
| [Deep Bilateral Learning / HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf) | Predict compact, content-dependent color transforms and apply them to full-resolution pixels. | Its pointwise transform assumptions limit newly created detail. This suggests separating color processing from learned detail, not replacing the complete neural effect with a color filter. Our fixed-grading student test is not an HDRNet implementation. |

## Quality and performance acceptance

The unchanged NVIDIA result is the baseline. A candidate must be tested on the
same model input, control values, noise/frame index, exposure and temporal
history. Required cases include motion, disocclusion, fine foliage, skin, UI,
bright effects, dark scenes and transitions. Same-frame comparisons and temporal
sequences are needed; FPS or a high global pixel correlation alone is inadequate.

For an arithmetic-preserving kernel, first compare intermediate tensors and final
outputs, including rounding boundaries and noncontiguous inputs. For a smaller
student, compare RGB error, high-frequency structure, perceptual metrics and
temporal errors on held-out sequences, followed by direct visual comparison.
A synthetic head match is not a pass for this gate.

Timing must include the full NR pass at a **true 1920×1080 model extent**, with
GPU warmup, repeated samples and separate median/tail results. Instrumented
kernel timings locate costs; final claims require an uninstrumented comparison
in the existing hidden demo. A single isolated-kernel speedup or reduced network
resolution cannot be reported as meeting 3 ms.

The [broader-data warm start](broad-student-training.md) completed 6,000 updates
on 262 training frames and evaluated 56 development images. It reduces overall
RGB error 12.42% against the earlier routed checkpoint at essentially unchanged
standalone inference cost: 2.230 versus 2.242 ms in the same run. The older
six-photo group regresses 10.77%, and visual differences remain. This candidate
is not installed or accepted as native-equivalent.

Remaining work: improve the student's fidelity and extend matched captures to
varied scenes and temporal sequences. For model compression, train against actual vendor
outputs and reserve separate sequences for validation. The smaller model must
preserve fine detail and temporal stability, then meet the latency target in the
existing hidden demo. There is currently no trained, validated replacement to
install in WuWa.
