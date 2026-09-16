# Startup timing fix

[← Documentation](README.md)

## What failed

With the original RTXMFG v1.3.3, WuWa discovered the Streamline wrapper before RTXMFG had selected the Ada GPU backend. The wrapper's startup maximum appeared to be cached as one generated frame. The mod prepared the wrapper only when the first NGX FG Create identified Ada, which was too late for that cached value. Higher requests returned result `38` or remained limited to 2x.

GPU display-name aliases exposed the native multiplier UI but did not fix the backend. A forced global native 3x override without RTXMFG produced a device hang on the test PC. Changing the initialization order with the existing RTXMFG provider/temporal patches present was followed by successful output.

## Exact binary edit

| Item | Value |
| --- | --- |
| Upstream | `dashdogy/RTX40MFG-Unlock`, tag `v1.3.3` |
| Tag commit | `e13a9841733b0ae43b7215e8c51fe0eb3897816f` |
| Original DLL SHA-256 | `1c0c561f1819b2f37c7f7f2528e9488290967408e51da0573b7b642ba8cfc56e` |
| Patched DLL SHA-256 | `46e931fdc265fcd1875f6c113b689a159466061c13cef239f57ca386d66f148f` |
| Instruction RVA | `0x1C97F` |
| File offset | `0x1BD7F` |
| Before | `E8 5C 48 01 00 84 C0 75 47` |
| After | `E8 FC 4B 01 00 D1 E8 74 47` |

Before: call `IsAda()`, test AL, prepare the wrapper if true.

After: call the existing `Selected()` helper, shift EAX right once, prepare if zero. Enum states Unknown (0) and Ada (1) pass; Ampere (2), Conflict (3), and other 32-bit states do not. Preparation while the family is Unknown is why the installer independently restricts physical hardware; this predicate is **not a general hardware safety check**.

Only five instruction bytes and three PE-checksum bytes differ. File size, exports, embedded assets and other code remain identical. `wuwa_mfg/patch.py` reproduces the exact verified hash without loading or executing the DLL. A source-level equivalent is provided in [`patches/early-wrapper-preparation.patch`](../patches/early-wrapper-preparation.patch); it is not a claim of a byte-identical full rebuild. Upstream build prerequisites still apply.

Unique pattern matching, expected-byte checks, provider identity validation, count/index handling, the D157 temporal correction and memory-protection restoration remain in the upstream mod. No NVIDIA disk file was modified in the working experiment or by this installer.

## Observed order after the patch

1. The active OTA wrapper maximum was prepared before the first FG Create.
2. Startup status reported a maximum of five generated frames while the GPU family was still Unknown.
3. The physical Ada GPU was verified, then the provider and temporal patches were prepared before FG use.
4. Active gameplay samples reported accepted 3x requests and three frames presented.
5. Removing the global forced-count override and starting a fresh game process allowed native 6x requests and six frames presented.

This supports an initialization timing diagnosis. It does not establish that every piece of the tested configuration is independently necessary. In particular, the global FG DLL override and display-name aliases were retained in the successful configuration.

## Profile control

The initial working 3x run used a driver count override of **2 generated frames**. Merely changing it to zero while that game process was alive allowed 6x requests but still reported three frames presented. After a full game restart, native 6x requests reported six frames presented.

The installer reproduces the final configuration: global FG DLL override 1, global count 0/application-controlled, and no user override for either setting in WuWa's own profile. It does not set the driver's explicit FG On-mode, preset, dynamic target, or SR overrides.

## Integrity and limits

The NVIDIA wrapper and provider cache paths and hashes are pinned in `wuwa_mfg/windows.py`. The installer checks the physical PCI device ID instead of trusting its marketing name and refuses newer cache directories. Runtime verification remains necessary: the existence of a cache file does not prove the game selected it.

The verification tool reads only files written by RTXMFG and enumerates WuWa process IDs. It rejects stopped-process status, stale heartbeats, FG-off samples, request errors, stale state samples, and mismatched requested/presented counts. It does not measure input latency, inspect individual synthesized images, or prove unique displayed frames. For a fixed-count comparison, select a fixed multiplier in the game; dynamic output can vary.
