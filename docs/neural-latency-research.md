# Neural latency research: model and kernel experiments

**Target: 3 ms at a 1920×1080 model extent without reducing image quality. This
target has not been achieved.** The current NVIDIA runtime remains about 6 ms in
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
The internal vendor noise counter has not been captured, so a reset frame is not
proof of a matched noise state. The local weight file also differs from the hash
used for upstream's reported golden comparisons. These remain possible sources
of disagreement, alongside incomplete reconstruction behavior.

The full-resolution Python runs took roughly 0.9–1.2 seconds of inference wall
time. They are quality diagnostics, not an optimized runtime or a route already
meeting the latency target. Only the first reset frame has been compared; motion
and temporal quality remain unvalidated. Numeric records, capture hashes and
controls are in
[`matched-vendor-capture.json`](../evidence/neural-model-research/matched-vendor-capture.json).

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
