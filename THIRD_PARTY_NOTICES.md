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
softmax, quadratic activation and batched feed-forward implementations adapt the
operation order and rounding specified by its PyTorch
reference. These model experiment files are provided under Apache-2.0; retain
the [license](licenses/MLX-DLSS-Apache-2.0.txt) and this attribution. The upstream
repository and weights are not vendored here. The demo probe and model-capture
patches extend the GPL-3.0 OptiScaler integration and follow that license instead.

The CUDA experiment invokes the user's separately installed PyTorch, NVRTC and
CUDA driver. Their binaries and generated GPU code are not distributed here.
Research papers are linked in the latency investigation; their reported results
are not presented as performance claims for this toolkit.

FirstEverTech/RTX4000-MFG-Unlock inspired the native NVIDIA profile experiment and is linked for attribution. Its repository content and binary artifacts are not copied into this project.
