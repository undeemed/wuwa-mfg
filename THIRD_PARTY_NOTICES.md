# Third-party notices

## RTXMFG

Upstream: https://github.com/dashdogy/RTX40MFG-Unlock

Tag: `v1.3.3`; commit: `e13a9841733b0ae43b7215e8c51fe0eb3897816f`.

Copyright (c) 2026 Michael Robles. MIT license, reproduced in [licenses/RTXMFG-MIT.txt](licenses/RTXMFG-MIT.txt).

The installer downloads the original release and modifies its wrapper-preparation predicate. The source-level diff, identifiers and exact-build binary patch are derived from this work. The project neither redistributes the upstream DLL in its setup ZIP nor claims it as original work. Its license notice accompanies the patch scripts.

## OptiScaler Neural Rendering

The source patch `patches/optiscaler-wuwa-compat.patch` modifies
[wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass),
tag `v0.8.4`, commit `8802b2b470db0462fa1ed03a125e793a7c06d735`.
It includes the WuWa compatibility changes and regression tests. These OptiScaler-derived
changes are **GPL-3.0**, not covered by the toolkit's MIT license.
See [the complete license](licenses/OptiScaler-GPL-3.0.txt) and upstream's
[credits and component notices](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/tree/v0.8.4/Licenses).
The build script fetches the exact upstream source and submodules, applies the public patch,
and compiles locally. Retain upstream notices in any redistribution and comply with each component's license.

OptiScaler is built on [OptiScaler/OptiScaler](https://github.com/OptiScaler/OptiScaler),
with Neural Rendering work from [Dagherbou/OptiScaler](https://github.com/Dagherbou/OptiScaler)
and colour-processing work from [RenoDX](https://github.com/clshortfuse/renodx).
This toolkit adds WuWa integration; it does not claim authorship of those projects or NVIDIA's model.

`nvngx_dlssnr.dll`, NVIDIA models, game DLLs, and compiled OptiScaler/backend DLLs are
**not distributed** in this repository or its setup archive. The NR runtime is supplied
separately by the user. A hash identifies the tested file; it grants no redistribution rights.
The source build uses [Microsoft DirectX-Headers](https://github.com/microsoft/DirectX-Headers)
tag `v1.619.5`, commit `ee479f0bd5f7b884f202bcf0c3f076cc050dd256`, under its upstream MIT license.

## NVIDIA NVAPI interface

Upstream: https://github.com/NVIDIA/nvapi

Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. MIT license, reproduced in [licenses/NVAPI-MIT.txt](licenses/NVAPI-MIT.txt).

`wuwa_mfg/windows.py` contains ctypes declarations and interface IDs based on the public NVAPI headers. It loads the user's installed `nvapi64.dll`; no NVIDIA binaries are distributed.

## Python

The launcher downloads the official CPython 3.12.10 Windows embeddable runtime from python.org. Its PSF and third-party license text is included inside the downloaded distribution as `LICENSE.txt`. Source and licensing: https://www.python.org/downloads/release/python-31210/ and https://docs.python.org/3/license.html.

The runtime is not bundled in this repository or its source/script release.

## Research references

The numerical experiments in `research/neural-latency` use
[MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS) commit
`0ca2deab092fe6f3e331bf4f616271dbc64521d0`. The CUDA normalization, bit-affine
softmax, quadratic activation, cosine publication, Gaussian noise, first-block
branch rounding and batched feed-forward implementations adapt the
operation order and rounding specified by its PyTorch
reference. These model experiment files are provided under Apache-2.0; retain
the [license](licenses/MLX-DLSS-Apache-2.0.txt) and this attribution. The upstream
repository and weights are not vendored here. The demo probe and model-capture
patches, including the combined native launch-contract probe and optional SM89
selector, buffer/barrier metadata probe, fenced preprocessor-prefix and separate
pooled-output, complete-first-block and final-block-input captures, and output-argument observation,
extend the GPL-3.0 OptiScaler
integration and follow that license instead. The camera-view, pooled-feature and
output-contract collection utilities are original MIT-licensed orchestration;
they do not include sample assets.

The CUDA experiment invokes the user's separately installed PyTorch, NVRTC and
CUDA driver. Their binaries and generated GPU code are not distributed here.
Private native inspection uses NVIDIA's separately downloaded `cuobjdump` and
`nvdisasm` tools. Only our inspection/decoder source, derived layout and arithmetic
findings, tool provenance and numeric results are published; no NVIDIA tools,
disassembly, CUDA modules, model weights or captured tensors are included.
The output-grading comparison is an original Apache-2.0 algebraic implementation
of the observed color operations. It does not contain NVIDIA instruction code
and does not claim bit-exact reproduction of NVIDIA's output stage.
Its differentiable student wrapper and fused CUDA version use the same license.
The camera-collection audit and numeric camera-pose list contain no sample
assets or captured images. HDRNet is cited as a research direction; no HDRNet
source, weights or image dataset is included. The affine-field student,
composition kernel and branch diagnostics are original Apache-2.0 experiments,
not an HDRNet reproduction. Interpolation behavior was checked against
[PyTorch 2.7.1's CUDA bilinear operator](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu);
no PyTorch operator source is vendored in this repository.
The native block-cost mapper, schedule checks, pretrained block-sensitivity
probe and affine feature-distillation experiment are original Apache-2.0
research utilities. They invoke the same attributed MLX-DLSS reconstruction and
its private logical weights. Neither the original weights nor fitted derivative
weights are distributed. The layer-pruning and restoration-distillation papers
are cited for research ideas; their source and data are not copied.
The direct FP8/FP16 MMA diagnostic follows the public NVIDIA PTX ISA fragment
layout and instruction interface. Its block-0 and block-70 scheduling adapts the same pinned
MLX-DLSS reference and retains the research files' Apache-2.0 notices.
The final-block decoder, substitution probe and boundary-block regression checks
use that attribution and license. They contain no captured features or vendor
instruction code. The regression reference is the earlier Apache-2.0 diagnostic
from this repository's own Git history.
The single-head arithmetic extension, FP8 operand packing kernel and associated
checks retain the same Apache-2.0 attribution. The fused gate follows the pinned
reference's operation order and explicit half rounding. No vendor instructions,
learned weights, captured features or generated CUDA modules are included.
The branched 2/4/8-head diagnostic likewise adapts the attributed reference's
graph and window mapping under Apache-2.0. Its private native-inspection utility
and shared-bias MMA extension contain no vendor instruction code. Only derived
findings, numeric instruction counts, hashes and experiment results are published.
Research papers are linked in the latency investigation; their reported results
are not presented as performance claims for this toolkit.
The compact student's global-attention branch, training extension and diagnostics
are original Apache-2.0 research code using the separately installed PyTorch
scaled-dot-product attention API. They are not an implementation copied from
the cited research papers. No trained student weights or teacher captures are
distributed.
The illumination view list, scene-configuration extension and collection
orchestration are original MIT-licensed code and numeric parameters. They use
the separately obtained sample's scene interface; no NVIDIA sample source,
models or textures are distributed. The student comparison and read-only
weight-sparsity audit are original Apache-2.0 research utilities. The latter
exports only aggregate counts and histograms, never weight values, and links
the public NVIDIA PTX specification for the sparse-matrix format.

FirstEverTech/RTX4000-MFG-Unlock inspired the native NVIDIA profile experiment and is linked for attribution. Its repository content and binary artifacts are not copied into this project.

## Private photo validation sources

The image-plane writer, fixture checks, downloader and isolated-desktop helper
are original MIT-licensed utilities; the student evaluator retains Apache-2.0.
The mesh is generated from numeric geometry. No sample assets or NVIDIA source
are copied into this repository. Pillow and NumPy are separately installed
dependencies used for private image preparation and validation.

The research downloads these images solely as private renderer inputs. It
center-crops/resamples each to 1920×1080, then captures the sample's rendered
input and native neural output. No original image, prepared texture, output,
prediction or learned weights are distributed with this project.

| Source | Author | Terms |
| --- | --- | --- |
| [Brent W. Jett official portrait, S92-47144](https://commons.wikimedia.org/wiki/File:Brent_W._Jett_-_Official_Astronaut_Portrait.jpg) | NASA | Public domain in the United States; see [NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/). No endorsement is implied. |
| [Fronalpstock big](https://commons.wikimedia.org/wiki/File:Fronalpstock_big.jpg) | Hannes Röst | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Private crop/resampling and neural transformations are recorded; no adaptations are distributed. |
| [3 year old calico cat](https://commons.wikimedia.org/wiki/File:3_year_old_calico_cat.jpg) | Babelball | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). |

The numerical evidence retains these sources, author credits, transformation
description, original dimensions and file hashes. Photo terms are separate
from the licenses on this repository's research code.

The native-feature capture wrapper is original MIT code; the target preparation,
oracle audit, auxiliary projection and associated tests are original Apache-2.0
research code. They use the previously documented private reference and captures.
No native weights, activation arrays, reconstructed images or vendor code are
distributed. [FitNets](https://arxiv.org/abs/1412.6550) and
[LIT](https://arxiv.org/abs/1810.01937) are credited as research sources; their
implementations are not copied, and the joint auxiliary experiment does not
implement either full method.

The subsequent feature-capacity analysis, feature pretraining utility, fused
student-output kernel and associated tests are original Apache-2.0 research
code. Staged pretraining is inspired by FitNets section 2.3 but uses a different
architecture and renderer losses. PCA vectors, position templates, projections,
activation targets and model checkpoints remain private. The implementation
uses separately installed NumPy, PyTorch and CUDA interfaces; no vendor kernel
code or learned weights are distributed.

The later decoder upsampling/skip-add kernel, execution diagnostics and training
initialization controls are original Apache-2.0 research code. The collection
wrapper retains its MIT license. They use separately installed PyTorch/CUDA
interfaces and contain no vendor kernel instructions or learned weights. The
replay paper is credited for its research idea; its implementation is not copied.

The subsequent training extension uses four different photo identities. The
three sources above remain excluded from training, including their lower-emission
variants. Original and transformed training images and all weights remain private.
The collector and split audit are original MIT-licensed research utilities.

| Additional training source | Author | Terms |
| --- | --- | --- |
| [Cat on snow](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg) | Von.grzanka | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Private crop/resampling and neural transformations; no adaptations distributed. |
| [Golden Gate Bridge](https://commons.wikimedia.org/wiki/File:GoldenGateBridge-001.jpg) | Rich Niewiroski Jr. | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). Private crop/resampling and neural transformations recorded. |
| [Moraine Lake](https://commons.wikimedia.org/wiki/File:Moraine_Lake_17092005.jpg) | Gorgo | Released into the public domain by the author. |
| [Tracy Caldwell Dyson in the ISS Cupola](https://commons.wikimedia.org/wiki/File:Tracy_Caldwell_Dyson_in_Cupola_ISS.jpg) | NASA/Tracy Caldwell Dyson | Public domain in the United States; [NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/). No endorsement is implied. |

The next private extension adds sixteen different sources. The
[fixed source catalogue](research/neural-latency/image-collections/diverse-extension.json)
records each author, source page, original-file URL, license and license link,
dimensions, SHA-1 and crop position. Training sources are Martin Eklund's forest
waterfall, Basile Morin's palace interior, Beijing Drive Culture Media's concert
hall, Benh LIEU SONG's cloisters, Paulo Barcellos Jr.'s night city, Don McCulley's
sunflower, Dirk Vorderstraße's retriever, Wikigab's dunes, the NASA portraits of
Jessica Watkins and Michael Foale, Brian Prechtel's strawberries and NASA's
Namibia view. Validation sources are Diliff's Trinity College library, Diego
Delso's Key West beach, the NASA portrait of Chris Cassidy and Joe Mania's dunes.
The catalogue retains the concert-hall author's original Chinese credit.

These sources carry the separately recorded CC0, CC BY, CC BY-SA or public-domain
terms. NASA material implies no endorsement. Preparation converts to RGB and
fits a 1920×1080 crop without changing aspect ratio; the three portrait crops
are biased upward. Rendering and neural processing add further transformations.
All original files, transformed images, captures and learned weights stay
private. Only source metadata, original MIT collector code and numerical
experiment results are published. These image terms do not change the licenses
of the research code.

The training-pair validator, gradient diagnostic, two-domain gradient combiner,
algebra tests and residual-scale fusion are original Apache-2.0 research code.
The projection rule is credited to Yu et al.,
[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782),
Algorithm 1. No upstream implementation is copied. Domain-specific renderer
results are our own experiments, not claims transferred from that paper.
Gradients, model checkpoints, captures and predictions remain private; only
source, scalar diagnostics and numerical evidence are distributed.

The context diagnostic, decoder-conditioning module and its mechanical tests
are original Apache-2.0 code. The feature-wise affine-conditioning idea is
credited to Perez et al., [FiLM: Visual Reasoning with a General Conditioning
Layer](https://arxiv.org/abs/1709.07871). No upstream implementation is copied.
The renderer uses pooled image features rather than question embeddings; the
paper's visual-reasoning results are not renderer quality claims. All learned
weights, captured activations, original images and predictions remain private.

The decoder-conditioning fusion and its operator tests are original Apache-2.0
code. The scalar and packed half arithmetic follow NVIDIA's published
[PTX instruction semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions).
No vendor implementation, binary or disassembly is distributed. The associated
capacity experiment uses the existing original image-conditioned architecture;
its learned weights and captured data remain private.

The spatial-error diagnostic, latent-context branch, mechanical tests and
training-branch ablation are original Apache-2.0 code. The latent
read/process/write interface is credited to Jaegle et al.,
[Perceiver IO](https://arxiv.org/abs/2107.14795). No upstream implementation or
trained weights are copied. The small CNN adaptation and measured results are
our own experiment; the paper's benchmarks are not renderer quality claims.
All private captured images, predictions, activations and trained weights remain
outside this repository.

The conditioned-model gradient-routing test and native launch-batching trace
audit are original Apache-2.0 code. The training extension uses the previously
credited PCGrad implementation without changing its projection formula. The
native audit reads existing private metadata and the repository's original
observer patch; it distributes no vendor code, arguments, addresses or binaries.

The training-brightness collector/audit and launch-order collector/analyzer are
original Apache-2.0 code. The launch-order C++ patch is an original GPL-3.0
extension to the existing OptiScaler research patch. It contains no NVIDIA
implementation or binary payload. Brighter training captures reuse the sixteen
previously credited image identities; all source images, derived scene assets,
native captures and trained weights remain private. No new image sources were
downloaded for this experiment.

The clipping diagnostic, straight-through clamp and its mechanical test, original
integer CUDA workload, NVRTC compiler utility and hidden-demo selftest runner are
original Apache-2.0 research code. The OptiScaler C++ selftest header and attachment
patch are original GPL-3.0-only extensions. They use separately installed NVIDIA
and PyTorch interfaces; no vendor implementation or generated binary is included.
The straight-through idea is credited to Bengio, Léonard and Courville,
[Estimating or Propagating Gradients Through Stochastic Neurons for Conditional
Computation](https://arxiv.org/abs/1308.3432), with the limitations discussed by
Yin et al., [Understanding Straight-Through Estimator in Training Activation
Quantized Neural Nets](https://arxiv.org/abs/1903.05662). No paper implementation is
copied. Training weights, images, activations and raw gradients stay private.

The spectral-error diagnostic, auxiliary loss and its tests are original
Apache-2.0 research code, inspired by Jiang et al.,
[Focal Frequency Loss for Image Reconstruction and Synthesis](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.pdf),
equations 7–10. No paper implementation is copied or benchmark gain transferred
to this renderer. Captures, spectral arrays, gradients and trained weights remain
private. The command-list observer and input-replay C++ extensions are original
GPL-3.0-only code using separately installed DirectX/NVIDIA interfaces and
Microsoft Detours. Their Python collectors and analyzer are original
Apache-2.0 code. No vendor implementation, binary, packed argument data or GPU
address is distributed.

The rejected native batcher, pass-through API observer and bounded overlap
selftest helpers/patches are original GPL-3.0-only extensions. The integer stress
and producer/consumer CUDA workloads, compiler/runner extensions, synchronization
inspection utility and numeric analyzer are original Apache-2.0 code. NVIDIA
NVAPI and DirectX supply external interface declarations; no vendor implementation
is copied. Private kernel extraction is read-only and only numeric instruction
counts are published. No vendor disassembly, binary, weights, captured pixels,
packed arguments or GPU addresses are distributed.

The frozen-base progressive student, trainer, mechanical checks and evaluator are
original Apache-2.0 research code. The experiment is inspired by residual stages
and original-input access in Zamir et al.,
[Multi-Stage Progressive Image Restoration](https://arxiv.org/html/2102.02808v1).
It does not implement MPRNet or copy its code, weights or figures. Jolicoeur-Martineau's
[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/html/2510.04871v1)
was reviewed for iterative refinement ideas; no TRM implementation or transferred
benchmark result is claimed. The existing training images, teacher captures and
both student checkpoints remain private.

The shared-feature correction decoder, trainer, mechanical test and precision audit, optional
student-feature collection, and evaluator/diagnostic extensions are original
Apache-2.0 code. Cross-stage feature exchange is credited to MPRNet section 3.2;
this frozen-base design does not implement its complete architecture or training
procedure. No external model implementation, weights, captured features or images
are included. Existing inference kernels are reused without changes.

The training-coverage audit, validated cluster sampler, tests and optional
trainer extension are original Apache-2.0 code. Data-curation discussion in
[V-JEPA 2](https://arxiv.org/html/2506.09985v1) motivated examining feature clusters
and sampling coverage. The experiment does not copy the paper's code, use its
weights or implement its retrieval pipeline. Private descriptors, trained weights
and input/output pixels are not distributed.

The region-context branch, independent attention/gradient tests, matched comparison,
and process-local Python platform fallback are original Apache-2.0 research code.
The branch is inspired by block selection in
[MoBA](https://github.com/MoonshotAI/MoBA) and region routing in
[BiFormer: Vision Transformer with Bi-Level Routing Attention](https://arxiv.org/pdf/2303.08810).
It implements neither paper's complete model and copies no external implementation,
weights or figures. The papers' benchmark gains are not transferred to this renderer.
PyTorch supplies the existing attention and gather operators. No new CUDA kernel,
vendor modification, captured image, feature tensor or trained weight is included.
