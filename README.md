# WuWa Experience Toolkit — DLSS MFG + Neural Rendering

**Get more from Wuthering Waves on older-generation RTX hardware: RTXMFG
3x/4x/5x/6x unlock via GPU-name spoofing and targeted tweaks, with an optional
neural rendering engine and controls for quality versus performance.**

The project remains `wuwa-mfg`. It now contains both the original MFG installer and
the source changes, build tools, setup and crash investigation for running
OptiScaler Neural Rendering alongside it. The validated hardware is **RTX 4070 Ti
with NVIDIA driver 616.92**; support for all older RTX cards is a goal, not a claim.

| Feature | Status | Start here |
| --- | --- | --- |
| RTXMFG 3x/4x/5x/6x | 3x and 6x verified in runtime; 4x and 5x tester-reported | [Guided MFG setup](#install-the-wuwa-dlss-mfg-unlock) |
| Optional neural engine + MFG | Model runs alongside actual 6x; transition stability experimental | [Neural setup and source build](docs/neural-rendering.md) |
| Model resolution control | DX12 scale relative to game output: full resolution or proportional fractions | [Resolution settings](docs/neural-rendering.md#model-resolution-relative-to-output) |
| Live NVIDIA / student switching | Local 1080p student integration; engine swap and weight reload tested in the hidden demo | [Student setup and controls](docs/student-hotswap.md) |
| Backups, restore and verification | Separate controls for MFG and NR, with hash checks | [Recovery](#undo) |

[Download toolkit](https://github.com/undeemed/wuwa-mfg/releases/latest) ·
[Neural rendering](docs/neural-rendering.md) · [Crash findings](docs/neural-crashes.md) ·
[Neural latency research](docs/neural-latency-research.md) ·
[All investigation work](docs/investigation-history.md) · [Validation](docs/validation.md)

## DLSS Multi Frame Generation unlock

The working setup combines three parts:

- **GPU-name spoofing:** present the GPU's Windows display name as an RTX 5080 to expose WuWa's native multiplier choices.
- **RTXMFG binary tweak:** patch the wrapper startup timing so the higher frame-generation capacity is prepared before WuWa caches it.
- **NVIDIA profile tweaks:** enable the FG DLL override and clear the forced frame count, allowing WuWa's selected multiplier to apply.

**Choose 3x, 4x, 5x or 6x in the game.** RTXMFG runs in Follow game mode, so control stays with WuWa. Setup handles the binary patch automatically and offers GPU-name spoofing during installation.

Tested on an **RTX 4070 Ti with driver 616.92**: 3x and 6x were confirmed in runtime telemetry; 4x and 5x were reported working by the tester. The installer is experimental and restricted to the [validated configuration](#supported-configuration).

[Download WuWa DLSS MFG Unlock](https://github.com/undeemed/wuwa-mfg/releases/latest) · [Installation guide](#install-the-wuwa-dlss-mfg-unlock) · [How the RTXMFG patch works](docs/technical.md) · [Compatibility and test results](docs/validation.md) · [Troubleshooting](docs/troubleshooting.md)

<a id="quick-start"></a>

## Install the WuWa DLSS MFG unlock

1. Download **WuWa-Experience-Toolkit-0.2.0.zip** from Releases and extract the entire folder somewhere you can keep it.
2. Close Wuthering Waves. Double-click **Setup.cmd**, choose **Install**, and approve the normal Windows administrator prompt.
3. Setup finds Steam libraries automatically. For another launcher, paste the WuWa folder when prompted. It must resolve to `Client/Binaries/Win64/Client-Win64-Shipping.exe`.
4. If WuWa currently offers only an on/off FG switch, choose **yes** for GPU-name spoofing (the optional RTX 5080 display-name aliases). The tested setup used this to expose the native multiplier menu. Review the target and type `INSTALL`.
5. Restart Windows if you applied the aliases; otherwise launch a fresh game process. Enable DLSS Frame Generation and select the multiplier in WuWa's graphics settings.
6. Choose **Verify** in Setup.cmd, then return to focused gameplay. It samples fresh runtime status for 30 seconds and reports requested/presented counts. Check that the image and motion also look normal.

No Python installation or NVIDIA Profile Inspector download is required. Setup downloads a checksum-pinned, official Python runtime into its own `.runtime` folder. This repository/release contains **source and scripts**; setup generates the patched mod DLL locally from the verified upstream download.

<a id="how-setup-applies-the-patch"></a>

## How setup applies the RTXMFG binary patch

1. Download and verify the original **RTXMFG v1.3.3** archive and DLL against their pinned SHA-256 hashes.
2. The [installer](wuwa_mfg/cli.py) automatically calls `patch_dll(download_dll())`. The [Python patcher](wuwa_mfg/patch.py) applies the wrapper startup timing change and updates the PE checksum.
3. Verify the modified DLL against the known patched SHA-256, then [install those patched bytes](wuwa_mfg/core.py) as `Client/Binaries/Win64/winmm.dll`.

The patcher from the published **v0.1.0 setup ZIP** was tested against a fresh upstream download: its output was **byte-for-byte identical to the DLL in the working WuWa test installation**. See [the exact changes and hashes](docs/technical.md#exact-binary-edit).

The separate [`patches/early-wrapper-preparation.patch`](patches/early-wrapper-preparation.patch) file expresses the same change in C++ for developers rebuilding upstream source. Setup uses the Python binary patcher above; users running **Setup.cmd → Install** do not need to apply the source `.patch` file manually.

<a id="supported-configuration-in-010"></a>

## Supported configuration

| Component | Validated configuration |
| --- | --- |
| OS | 64-bit Windows |
| Physical GPU | RTX 4070 Ti, PCI `10DE:2782`, one discrete GPU |
| NVIDIA driver | 616.92 / Windows version `32.0.16.1692` |
| Game | Wuthering Waves, Steam build tested September 14, 2026 |
| Upstream mod | RTXMFG v1.3.3, exact SHA-256 pinned |
| Active NVIDIA wrapper | 2.14.0.0, cache version `134656` |
| Active NVIDIA provider | 310.9.0.0, cache version `20318464` |

Setup refuses unvalidated GPU/driver/cache combinations and unknown proxy DLL conflicts. It does not install or downgrade a driver, supply NVIDIA cache files, or assume a renamed GPU is supported. Other RTX 40-series cards and games need separate validation before this installer supports them. Driver, game, and NVIDIA OTA updates can change behavior.

This is a third-party game mod, **not official DLSS 5 or an anti-cheat-approved feature**. Account safety has not been established. The installer does not alter anti-cheat, kernel drivers, hardware IDs, or Windows security settings; RTXMFG still modifies the game's graphics pipeline in memory.

## What MFG setup changes

- Installs one patched `winmm.dll` beside WuWa's shipping EXE.
- Sets `RTXMFG-Universal.json` to **Follow game**, Preset B. It does not force a multiplier or invent a dynamic FPS target. The saved `multiplier: 4` field is ignored while Follow game is enabled.
- Enables NVIDIA's global FG DLL override (`0x10E41E03 = 1`) and sets its generated-frame count override to application control (`0x104D6667 = 0`). It removes user overrides for those same two settings in WuWa's driver profile so the game inherits them. **These global settings can also affect other compatible games.** Other profile settings are preserved.
- Optionally spoofs the GPU name by changing three Windows description strings to `NVIDIA GeForce RTX 5080`, exposing WuWa's native menu. This affects names shown system-wide; the physical GPU stays a 4070 Ti. Setup never restarts Windows automatically.

Follow game accepts supported choices made by WuWa. Dynamic mode is followed **only if the game requests it**. This is not unlimited frame generation; the tested capacity is five generated frames plus one rendered frame, or 6x total.

## Wuthering Waves DLSS MFG FAQ

### Can I use DLSS Multi Frame Generation on an RTX 40-series GPU?

This project's working WuWa configuration was tested on an **RTX 4070 Ti with NVIDIA driver 616.92**. Runtime captures confirmed 3x and 6x; the tester also reported 4x and 5x working. The installer currently accepts only the [validated GPU, driver and NVIDIA cache builds](#supported-configuration). Other RTX 40-series cards need separate validation.

### Why does Wuthering Waves only show Frame Generation on/off or stay at 2x?

In the tested configuration, **RTX 5080 GPU-name spoofing** exposed WuWa's native multiplier choices. The RTXMFG startup patch addressed the cached 2x limit, while NVIDIA profile tweaks removed a forced frame count. Restart Windows after applying the name spoof, or fully restart the game after changing driver overrides. See [2x/3x troubleshooting](docs/troubleshooting.md#counts-are-still-2x3x-after-selecting-6x).

### Does the installer automatically apply the custom RTXMFG patch?

Yes. Setup downloads the pinned upstream DLL, applies this project's binary patch, verifies the modified file's SHA-256, and installs it as `winmm.dll`. The separate C++ `.patch` file is for source builds. See [the automatic patching steps](#how-setup-applies-the-rtxmfg-binary-patch).

### Can I switch between 3x, 4x, 5x and 6x or use Dynamic MFG?

Select the multiplier in WuWa's graphics settings. **Follow game** leaves control with the game; dynamic mode is followed only when WuWa requests it. A 6x multiplier means up to five generated frames plus one rendered frame, not six times the input responsiveness or a guaranteed sixfold FPS increase. Use the installer's **Verify** action to check runtime counts, and check visible motion separately.

## Add the neural engine

Use the [neural setup guide](docs/neural-rendering.md) to build the pinned OptiScaler
source with this project's compatibility patch. **Setup.cmd → 5** installs the local
bundle; **6/7** enable or disable NR at the next launch, **8** restores the add-on,
**9** reads its status and **10** changes its output resolution scale. **F8** is the
neural toggle, with restart-based controls available if WuWa does not receive it.

NR starts off. The profile uses one pre-SR pass and 100% of the game output for the
model. At 4K that is 3840 × 2160. Lower the scale to 75% or 50% if delay is excessive;
both dimensions follow the output proportionally. Full-resolution model quality and
performance require game testing, and the game still uses its chosen DLSS render scale.

The compatibility patch keeps RTXMFG in charge of FG, fixes an observed stale-module
reference, and guards a reproduced GPU descriptor/constant-slot lifetime defect.
The [crash report](docs/neural-crashes.md) distinguishes those findings from the GPU
fault whose exact shader/resource remains unidentified. NR is **experimental**, with
no claim that every settings/window transition is stable.

This is community neural rendering integration, not an official or bundled “leaked
DLSS 5” release. The proprietary NR runtime must be supplied separately. Source,
patches and build/setup scripts are included; NVIDIA and compiled model DLLs are not.

<a id="undo"></a>

## Uninstall and restore the previous configuration

Close WuWa. If the managed neural add-on is installed, use **Setup.cmd → 8** first. Then open **Setup.cmd → 3**, review the MFG target, and type `RESTORE`. To keep the add-on installed but disable NR, use **7** instead.

Original files, the two profile settings at both scopes, and any changed descriptions are saved **before changes begin** in `%ProgramData%\WuWaMFG\state.json`. Restore keeps the backup. Restart Windows after restoring descriptions. If a file or setting changed since installation, rollback stops rather than overwriting that change; see [recovery instructions](docs/troubleshooting.md).

An existing manually installed copy of this exact patch is left in place. Use Verify for that installation; this installer cannot reconstruct its earlier originals.

## Development

Python 3.10+ standard library only. The Windows integration uses NVAPI DRS and registry APIs; tests use a simulated backend and temporary game folders.

```powershell
python -m unittest discover -s tests -v
python setup.py doctor --game "D:\Games\Wuthering Waves"
python setup.py verify --game "D:\Games\Wuthering Waves" --seconds 30
```

To reproduce the binary edit without installing anything:

```powershell
python setup.py patch --input original-v1.3.3.dll --output patched-winmm.dll
```

The input and output are checked against known SHA-256 digests. The output must be a new file. Set `RTXMFG_TEST_ARCHIVE` to the original v1.3.3 ZIP to include the exact-release integration test. CI downloads the pinned upstream archive and runs that test on Windows and Linux.

## Credits and license

- [dashdogy/RTX40MFG-Unlock](https://github.com/dashdogy/RTX40MFG-Unlock/tree/v1.3.3), by Michael Robles: RTXMFG itself, its loader, wrapper/provider patches, configuration and telemetry. MIT licensed. This project adds a small startup timing change and packaging; it does not claim authorship of RTXMFG.
- [FirstEverTech/RTX4000-MFG-Unlock](https://github.com/FirstEverTech/RTX4000-MFG-Unlock): research that motivated testing NVIDIA's global profile controls. Its native override alone did not produce the successful WuWa result reported here.
- [OptiScaler Neural Rendering](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass): the neural engine, model integration and upstream GPU lifetime framework. The complete WuWa compatibility diff is published here with GPL-3.0 attribution.
- [NVIDIA NVAPI](https://github.com/NVIDIA/nvapi): DRS interface definitions.

[MIT license](LICENSE) for the toolkit Python/setup code and RTXMFG changes. The OptiScaler-derived source patch and included regression changes are [GPL-3.0](licenses/OptiScaler-GPL-3.0.txt). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` for upstream notices. NVIDIA, Kuro Games and RTXMFG's maintainers do not endorse this project.
