# Neural latency research: model and kernel experiments

**Target: 3 ms at a 1920×1080 model extent without reducing image quality. This
target has not been achieved.** The current NVIDIA runtime remains about 5.4–6.1 ms in
the separate NVIDIA DLSS Sample. None of the experiments below replaces the
working WuWa installation or changes the installer defaults.

Two routes are being investigated: execute the existing mathematics faster, and
train a smaller network to reproduce its output. Reduced resolution, skipped
frames, weaker blending and a different visual style do not satisfy the target.

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

## Papers and what can transfer

These papers provide research ideas. Their reported speedups are on other models
and hardware; none establishes the target for this runtime.

| Work | Relevant idea | Constraint in this renderer |
| --- | --- | --- |
| [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf) | Improve GPU work partitioning and avoid materializing intermediate attention matrices. | The recovered attention uses cosine normalization, explicit rounding and an approximate softmax. A standard attention replacement changes those operations; custom fusion must preserve them. |
| [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) | Calibrate activation/weight scaling before lower-precision execution. | This runtime already invokes FP8-named kernels. LLM INT8 results do not imply a further lossless gain, and calibration must include renderer activations and controls. |
| [Knowledge distillation](https://arxiv.org/abs/1503.02531) | Train a smaller student against the larger model's behavior. | A renderer needs matched pixels, detail and temporal consistency, not just matching classification probabilities. A trustworthy teacher and held-out sequences are prerequisites. |
| [EfficientViT](https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.pdf) | Hardware-friendly multiscale operators for dense, high-resolution prediction. | Replacing attention with linear attention changes the learned function; it is a student architecture to train and validate, not an interchangeable kernel. |
| [TinyVLA](https://arxiv.org/html/2409.12514v3) | A compact backbone and task-specific decoder can reduce inference cost. | Fast robot action prediction does not require reproducing every image pixel. Borrow compact architecture design and task-specific training, not its quality claims. |
| [V-JEPA 2](https://arxiv.org/html/2506.09985v1) | Predict compact latent representations and learn useful temporal structure. | Semantic latent accuracy does not establish correct fine texture or UI edges. A latent predictor could assist a student, but requires pixel and temporal losses and refresh on disocclusions. |
| [ToCa](https://arxiv.org/html/2410.05317v1) | Selectively reuse features based on redundancy and error sensitivity. | Its reuse is across diffusion steps. This effect already uses one pass; across-frame reuse adds motion, disocclusion and noise-state problems. It must not simply retain an old rendered image. |
| [Edge-Efficient Image Restoration](https://arxiv.org/abs/2605.02794) | Train replacement blocks against intermediate features, select combinations, then fine-tune the whole model. | Its transformer/SSM experiments use other restoration tasks and hardware. Here, the initial small CNNs fail the held-out image test; replacement blocks would need to retain learned context and be validated against actual vendor output. |
| [Simple Baselines for Image Restoration / NAFNet](https://arxiv.org/abs/2204.04676) | Simple multiplicative gates can replace some expensive nonlinear components in a trained restoration architecture. | This suggests a student design; substituting gates into the trained vendor graph would change its function. No NAFNet replacement has been validated here. |
| [Real Image Denoising with Knowledge Distillation for High-Performance Mobile NPUs](https://arxiv.org/abs/2605.03680) | Choose operators for the target device and expand training context while distilling a smaller network. | Its mobile-NPU denoising metrics do not prove renderer quality on Ada GPUs. Our wider-context candidate still failed, so the paper's success cannot be transferred without evidence. |

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

Remaining work: extend matched captures to varied scenes and temporal sequences,
resolve reconstruction differences, and build compatible fused attention and
feed-forward operations. For model compression, train against actual vendor
outputs and reserve separate sequences for validation. The smaller model must
preserve fine detail and temporal stability, then meet the latency target in the
existing hidden demo. There is currently no trained, validated replacement to
install in WuWa.
