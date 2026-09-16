# NVIDIA neural runtime performance investigation

[← Neural research](README.md)

Offline inspection of the tested SF-v2 `nvngx_dlssnr.dll` did not establish a
faster replacement for the RTX 4070 Ti. Output-relative model resolution remains
the measured performance control. No neural-runtime binary edit was installed during this
investigation, and no debugger or kernel profiler was attached to WuWa.

## Exact runtime and method

The input was 165,830,144 bytes, SHA-256
`6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`.
The analysis read PE sections and parameter strings, decompressed CUDA fatbin
payloads with Zstandard, checked embedded ELF architecture metadata, and traced
RIP-relative parameter references in the native `.text` section with Capstone.
The DLL was read as data, never loaded by the inspection scripts.

The retained [offline inspector](../../tools/inspect_nr_runtime.py) reproduces the
architecture inventory and checks the two reset instructions for this exact hash.
Run it in a separate analysis environment with `zstandard==0.25.0` and
`capstone==5.0.6` installed:

```powershell
python tools/inspect_nr_runtime.py "D:\LocalRuntime\nvngx_dlssnr.dll"
```

It prints aggregate metadata only. These packages are not game dependencies.

There were **15 CUDA fatbins**, containing **60 compiled CUDA objects**: 15 each
for SM75, SM86, **SM89 (Ada)** and SM120. Eight additional PTX entries targeted
SM120. This package uses the newer CUDA ELF layout: the SM identifier occupies
bits 8–15, rather than the low byte. The decoded identifier matched each fatbin
entry's architecture field. See [LLVM's CUDA ELF definitions](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/BinaryFormat/ELF.h).

FP8-named neural kernels are present in the SM89 objects. This establishes that
Ada-targeted compiled code exists; it **does not identify which kernels ran in
WuWa**, their precision throughout the graph, or their individual GPU cost.
Static strings alone cannot establish active dispatch or an optimization benefit.

## The internal scaling parameter is overwritten

At native RVA `0x17FF2`, the runtime references `DLSSNR.ScalingRatio`, reads it,
then stores float **1.0** over the destination at RVA `0x18006`. The evaluation
path references the same key at RVA `0x1A956` and again stores 1.0 at `0x1A96A`.
These offsets apply only to the hash above. They are findings, not patch recipes.

Consequently, setting that ordinary parameter is not evidence of internal model
downscaling. Removing the overwrite has not been validated and could leave model
dimensions, resources and private callback state inconsistent. Our working scale
instead resizes the model's actual feature and resources through OptiScaler's
existing creation path. Independent [private-contract experiments](https://github.com/kibblerz/DLSS5-Reshade-AIO/blob/main/lab/PRIVATE-CONTRACT-FINDINGS.md)
also found that nominal scaling/preset changes did not establish a working private
upscaler; those tests were not reproduced here and used their own runtime inputs.

## Why the historical FP4 hybrid was not installed

The historical upstream hybrid at commit
`d2b65cda8978ea72fa06e8a72125692a4893316d` replaces selected feed-forward work,
uses a [checksum-pinned SM120 asset](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/d2b65cda8978ea72fa06e8a72125692a4893316d/OptiScaler/dlssnr/DlssNrHybridAssets.h),
and has an exact source-module predicate in
[DlssNrNative.cpp](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/d2b65cda8978ea72fa06e8a72125692a4893316d/OptiScaler/dlssnr/DlssNrNative.cpp).
This is not a drop-in Ada implementation. A display-name spoof does not provide
Blackwell instructions or convert an SM120 kernel into SM89 code.

Its published [RTX 5090 comparison](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/d2b65cda8978ea72fa06e8a72125692a4893316d/docs/HYBRID-V072-VALIDATION.md)
reported 55.16 versus 54.57 average rendered FPS for the candidate and original
FP8 path. The author explicitly treated that small difference as unproven from
single sequential runs. It was whole-game performance, not isolated model timing.

Further internal optimization would require identifying costly dispatches in an
isolated harness, implementing compatible SM89 kernels or a different model, and
checking output correctness, GPU resource lifetimes and repeatable performance.
This remains research work; no faster NVIDIA runtime is claimed by this toolkit.

An [official NVIDIA demo setup](benchmark.md) now provides a separate
application for testing the same pre-SR path at a 1920×1080 model extent. Feature
creation and evaluation succeeded; a controlled speedup has not been measured.

The later [live kernel investigation](findings.md) now identifies
158 actual CUDA launches per evaluation in that demo and measures their costs.
It also includes an editable model and tested custom CUDA normalization kernel.
These development results do not establish a faster NVIDIA runtime or a 3 ms,
quality-preserving replacement.
