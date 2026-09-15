# Neural rendering with RTXMFG

The optional neural engine uses a locally patched **OptiScaler NR v0.8.4** alongside
this project's RTXMFG unlock. RTXMFG keeps control of 3x/4x/5x/6x; OptiScaler runs
one neural pass before DLSS Super Resolution. The changes are source patches, with
regression tests and a build script. This is experimental community integration,
not an official NVIDIA DLSS 5 release or support commitment.

## What has been tested

On the same RTX 4070 Ti / driver 616.92 configuration as MFG, the neural model has
run alongside runtime-confirmed 6x. A previous revision GPU-crashed during a
settings/window transition. The current patch fixes a reproduced descriptor-slot
lifetime defect, and an initial game run continued with NR enabled. **Long-term
and transition stability are not established.** Read [the crash evidence](neural-crashes.md).

The guided installer starts NR **off**. Enable it separately after verifying your
MFG baseline. Existing manual OptiScaler installations are not automatically adopted.

## Build once, then use guided setup

Requirements: the supported MFG installation, Git for Windows, Visual Studio 2022
with Desktop development with C++ and a Windows SDK, and your own compatible
`nvngx_dlssnr.dll`. The script does not install these prerequisites.

1. Download/extract the complete toolkit release or clone this repository.
2. Obtain the NR runtime according to [upstream's setup information](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/v0.8.4/INSTALL-DLSSNR.md).
   This integration accepts only the tested SF-v2 file identified below. No NVIDIA
   runtime or model is bundled here.
3. From PowerShell in the toolkit folder, run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\BuildNeural.ps1 -NrRuntime "D:\Downloads\nvngx_dlssnr.dll"
   ```

   This downloads pinned source/dependencies, applies the patch, runs the isolated
   tests, compiles Release x64 and creates `.build\bundle`. Build output stays local.
   Use `-WorkDirectory "D:\Builds\WuWa-NR"` for another **new** directory.
   Existing directories are preserved; a failed build is not automatically deleted.
4. Close WuWa. Open **Setup.cmd → 5**, select that bundle folder and review the
   experimental setup prompt. Normal administrator access is needed for the game folder.
5. Launch and use **Setup.cmd → 2** to check that MFG still reports the selected
   multiplier. Close WuWa, then choose **6** to enable NR on the next launch.
6. **F8** toggles NR in a running game; no numpad is needed. The binding was verified
   in startup logs, but input/menu attachment has been inconsistent in WuWa. If it
   does not respond, use **7** with the game closed to disable NR reliably on restart.

**Insert** is OptiScaler's menu key and **Backspace** is RTXMFG's. A missing menu
does not prove either backend failed. Use logs and fresh frame counts. Menu changes
can save the INI; the managed installer will then stop rather than overwrite it.
Keep a copy and review it before restoring the recorded version.

## Model resolution relative to output

The new DX12 option makes a scale of 1.0 use the game's **output dimensions**, even
though the neural pass runs before SR. The previous input-relative setting of 0.5
processed 1281 × 721 from a 2562 × 1442 input. It did not mean half of a 4K output.

| Output scale | Model pixels for 3840 × 2160 output | Share of full-resolution pixels |
| --- | --- | --- |
| 1.0 / 100% | 3840 × 2160 | 100% |
| 0.75 / 75% | 2880 × 1620 | 56.25% |
| 0.5 / 50% | 1920 × 1080 | 25% |
| 0.333333 / about 33% | 1280 × 720 | about 11.1% |

The supplied profile selects 100% output resolution with NR initially disabled.
Full 4K contains almost nine times as many pixels as the earlier 1281 × 721 test.
That test took roughly 4 ms per neural pass; **this does not predict full-resolution
latency or guarantee smooth gameplay**. Frame generation does not remove neural
inference cost or make input responsiveness scale with the displayed FPS counter.

If full resolution feels slow, close WuWa and select **Setup.cmd → 10**, then enter
`0.75`, `0.5`, or another fraction between 0.25 and 1.0. Command-line equivalent:

```powershell
python setup.py neural-scale --scale 0.75
```

This changes both axes by the same fraction of the current game output, preserving
its proportions within pixel rounding. It follows output resolution changes rather
than baking in 4K dimensions. DLSS input resolution changes do not change the model's
output-relative size. If output dimensions are unavailable, it falls back to the
active input and logs that basis. Oversized dimensions are uniformly limited to
D3D12's texture dimension limit. This is **manual proportional scaling**, not an
automatic latency controller; pixel count scales quadratically, not linearly.

For a manual installation, the relevant settings are:

```ini
[FrameGen]
External=true
[DlssNr]
Enabled=true
ToggleKey=0x77
RunBeforeSR=true
Passes=1
WorkingScaleRelativeToOutput=true
WorkingScale=1.0
```

Use the complete [profile](../neural/OptiScaler.ini), not only this excerpt.
`WorkingScaleRelativeToOutput` is added by this patch and applies to the DX12 NR
route. The model's edit is returned to the pre-SR colour input before DLSS upscales
the game image. Full model resolution does not make the game itself render at native
resolution or change the placement to a post-SR neural pass.

## Installed files and recovery

The add-on installs `dxgi.dll`, `OptiScaler.ini`, the user-supplied `nvngx_dlssnr.dll`,
and the nine pinned support DLLs listed in [`neural.py`](../wuwa_mfg/neural.py).
It retains the existing patched `winmm.dll` and RTXMFG configuration. Its external-FG
mode avoids taking over RTXMFG's Streamline/FG path. No additional driver settings
or GPU aliases are changed by the neural installer.

The journal is `%ProgramData%\WuWaMFG\neural.json`, with configuration backups
beside it. **7** disables NR at the next launch while retaining the compatibility
loader; **8** removes the managed add-on after hash checks. Restore NR before using
the main MFG restore action. Unknown existing injectors, different backends, changed
files and redirected destinations stop setup/restore rather than being overwritten.
Identical pre-existing support DLLs are retained on uninstall.

A local bundle's manifest checks build identity and file integrity, not an external
signature or independent code attestation. Build from reviewed source on a trusted
machine. Do not install somebody else's unexplained `bundle.json`/DLL collection.

## Exact inputs and reproducibility

| Input | Pin |
| --- | --- |
| OptiScaler NR source | v0.8.4, `8802b2b470db0462fa1ed03a125e793a7c06d735` |
| OptiScaler standard package SHA-256 | `8789912859882e66b3f3a1aa768db947da779dfd65225df69ea919052e73a2e4` |
| NVIDIA NR runtime | 310.8.2, SF-v2, 165830144 bytes |
| NR runtime SHA-256 | `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927` |
| DirectX-Headers | v1.619.5, `ee479f0bd5f7b884f202bcf0c3f076cc050dd256` |

The build applies [`optiscaler-wuwa-compat.patch`](../patches/optiscaler-wuwa-compat.patch).
Submodule revisions come from the pinned upstream commit. New headers are supplied
through a local MSBuild include override; the system SDK is not overwritten.
DLL hashes from local builds are not expected to be identical across toolchains.
The bundle records its own DLL hashes and the exact public patch hash.

Local validation used VS 2022 17.11.5, Windows SDK 10.0.22621, Release x64. The build
completed with upstream warnings. The source/helper tests do not execute the proprietary
NR model; game testing and visual inspection remain separate.

## Check the right evidence

`Setup.cmd → 9` shows recent NR startup/timing lines only when the log names a
currently running game PID. Startup `Enabled` is not the current F8 toggle state.
Look for `DLSS-NR resolution`, `feature created`, recent `DLSS-NR elapsed`, and any
descriptor-backpressure skips. Use MFG Verify for **fresh, FG-active** requested and
presented counts. A menu or unfocused window may legitimately stop FG.

If a transition crashes, disable NR at the next launch and compare the same action
with MFG unchanged. Record the exact build, model dimensions, active counts and
transition. Share sanitized findings, not raw crash dumps or account-bearing logs.
