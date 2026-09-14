# Third-party notices

## RTXMFG

Upstream: https://github.com/dashdogy/RTX40MFG-Unlock

Tag: `v1.3.3`; commit: `e13a9841733b0ae43b7215e8c51fe0eb3897816f`.

Copyright (c) 2026 Michael Robles. MIT license, reproduced in [licenses/RTXMFG-MIT.txt](licenses/RTXMFG-MIT.txt).

The installer downloads the original release and modifies its wrapper-preparation predicate. The source-level diff, identifiers and exact-build binary patch are derived from this work. The project neither redistributes the upstream DLL in its setup ZIP nor claims it as original work. Its license notice accompanies the patch scripts.

## NVIDIA NVAPI

Upstream: https://github.com/NVIDIA/nvapi

Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. MIT license, reproduced in [licenses/NVAPI-MIT.txt](licenses/NVAPI-MIT.txt).

`wuwa_mfg/windows.py` contains ctypes declarations and interface IDs based on the public NVAPI headers. It loads the user's installed `nvapi64.dll`; no NVIDIA binaries are distributed.

## Python

The launcher downloads the official CPython 3.12.10 Windows embeddable runtime from python.org. Its PSF and third-party license text is included inside the downloaded distribution as `LICENSE.txt`. Source and licensing: https://www.python.org/downloads/release/python-31210/ and https://docs.python.org/3/license.html.

The runtime is not bundled in this repository or its source/script release.

## Research references

FirstEverTech/RTX4000-MFG-Unlock inspired the native NVIDIA profile experiment and is linked for attribution. Its repository content and binary artifacts are not copied into this project.
