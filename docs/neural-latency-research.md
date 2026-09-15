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
control limitations. Vendor comparison on the intended style and temporal inputs
is still required before it can serve as a trustworthy replacement or teacher.

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
4. **Removing a transformer block without training.** Skipping global block 31
   changed the synthetic model head: MAE 0.1945, RMSE 0.3452 and maximum absolute
   error 3.042. These are raw head units, not display RGB metrics. This candidate
   was rejected. Deleting blocks alone is not a quality-preserving model.

All retained numerical records are in
[`evidence/neural-model-research`](../evidence/neural-model-research).
Source and reproduction instructions are in
[`research/neural-latency`](../research/neural-latency).

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

Remaining work: establish matched vendor/reconstruction captures, test the fused
kernel on broader real inputs, build compatible fused attention/feed-forward
operations, and evaluate trained smaller models against that quality baseline.
There is currently no validated replacement to install in WuWa.
