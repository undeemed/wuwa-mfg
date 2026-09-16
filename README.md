<div align="center">

# WuWa Experience Toolkit

**DLSS Multi Frame Generation and optional neural rendering for Wuthering Waves.**

[![Release](https://img.shields.io/github/v/release/undeemed/wuwa-mfg)](https://github.com/undeemed/wuwa-mfg/releases/latest)
[![Tests](https://github.com/undeemed/wuwa-mfg/actions/workflows/test.yml/badge.svg)](https://github.com/undeemed/wuwa-mfg/actions/workflows/test.yml)
![Platform: Windows x64](https://img.shields.io/badge/platform-Windows_x64-0078D4)

[**Download**](https://github.com/undeemed/wuwa-mfg/releases/latest) ·
[Quick start](#quick-start) · [Documentation](docs/README.md) ·
[Report an issue](https://github.com/undeemed/wuwa-mfg/issues/new/choose)

</div>

Unlock **RTXMFG 3x / 4x / 5x / 6x** through GPU-name spoofing, an automatic
RTXMFG startup patch, and NVIDIA profile tweaks. Choose your multiplier in WuWa's
graphics settings; the toolkit handles setup, runtime verification, and restore.

## Features

- **In-game MFG control.** Follow game mode keeps the multiplier under WuWa's control.
- **Guided setup.** Finds Steam installs, checks compatibility, patches RTXMFG locally,
  and backs up changes. No separate Python or Profile Inspector installation.
- **Optional neural rendering.** Run NVIDIA's neural engine alongside MFG, with an
  effect toggle and adjustable processing resolution.
- **Reversible changes.** Separate install, diagnostics, and restore actions for
  frame generation and the neural add-on.

<a id="supported-configuration"></a>
<a id="supported-configuration-in-010"></a>

## Compatibility

The validated setup is **Windows x64 · RTX 4070 Ti · NVIDIA driver 616.92 · WuWa
on Steam**. Runtime readings confirmed 3x and 6x; 4x and 5x were tester-reported.

Setup also checks exact NVIDIA cache builds and refuses unvalidated combinations.
Other RTX cards are not currently supported by this installer.
[Full requirements and test results →](docs/installation.md#requirements)

> [!IMPORTANT]
> This is an experimental third-party mod. Anti-cheat approval and account safety
> are not established. Setup changes two global NVIDIA FG settings, which can
> affect other games. Review [what changes](docs/installation.md#what-setup-changes).

<a id="install-the-wuwa-dlss-mfg-unlock"></a>

## Quick start

1. [Download the latest toolkit ZIP](https://github.com/undeemed/wuwa-mfg/releases/latest)
   and extract the entire folder.
2. Close WuWa. Run **Setup.cmd → 1: Install MFG unlock**, approve the Windows prompt,
   and review the detected game folder.
3. Choose the offered **RTX 5080 name spoof** if WuWa only shows FG on/off. Type
   `INSTALL` to apply. **Restart Windows if you enabled the spoof.**
4. Launch WuWa and select a frame-generation multiplier in its graphics settings.
   Use **Setup.cmd → 2: Verify** while playing to check actual runtime counts.

[Detailed installation](docs/installation.md) ·
[Setup menu reference](docs/installation.md#setup-menu) ·
[Troubleshooting](docs/troubleshooting.md)

## Neural rendering

The optional **OptiScaler NR** integration adds NVIDIA's neural effect alongside
MFG. It requires a local source build and a separately supplied compatible NVIDIA
runtime. Guided setup starts the effect **off**; **F8** toggles it after enabling.
Resolution and transition stability are covered in the [neural setup guide](docs/neural-rendering.md).

The custom student model is **research only**. Its first WuWa trial showed
washed-out output, weak dark-scene accuracy, and reported stutters. The test setup
has returned to NVIDIA. [Research status and live-switch controls →](research/neural/README.md)

<a id="how-setup-applies-the-patch"></a>
<a id="how-setup-applies-the-rtxmfg-binary-patch"></a>

## FAQ

**Does setup actually apply the custom patch?** Yes. It downloads the pinned
RTXMFG v1.3.3 DLL, applies the binary edit, verifies its hash, and installs it as
`winmm.dll`. No manual source patch is needed. [How it works →](docs/technical.md)

**Does 6x mean six times the responsiveness?** No. It means up to five generated
frames per rendered frame. FPS and input responsiveness are different; verify
runtime counts and judge motion separately.

**Is this official DLSS 5?** No. This is a community toolkit. NVIDIA runtimes,
model weights, and compiled OptiScaler DLLs are not bundled.

<a id="undo"></a>
<a id="uninstall-and-restore-the-previous-configuration"></a>

## Uninstall

Close WuWa, run **[Uninstall.cmd](Uninstall.cmd)**, review the target, and type
`UNINSTALL`. It restores the managed neural add-on first, then MFG, saved NVIDIA
settings, and any GPU-name aliases. Backups are retained.

For the separate student integration, follow
[its restore procedure](research/neural/integration.md#build-and-install-locally) first.
[Restore details →](docs/installation.md#restore)

## Documentation

| I want to… | Read |
| --- | --- |
| Set up MFG or undo changes | [Installation](docs/installation.md) |
| Add the NVIDIA neural effect | [Neural rendering](docs/neural-rendering.md) |
| Fix a problem | [Troubleshooting](docs/troubleshooting.md) |
| Understand compatibility and measured results | [Validation](docs/validation.md) |
| Find source, tools, or experiments | [Repository map](docs/repository-layout.md) · [Research index](research/README.md) |

<a id="development"></a>

## Contributing

Bug reports, reproducible compatibility reports, and focused improvements are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and
reporting guidelines. [All documentation →](docs/README.md)

## Credits and license

Built on [RTXMFG](https://github.com/dashdogy/RTX40MFG-Unlock),
[OptiScaler Neural Rendering](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass),
and [NVIDIA NVAPI](https://github.com/NVIDIA/nvapi), with driver-profile research
from [FirstEverTech](https://github.com/FirstEverTech/RTX4000-MFG-Unlock).

Toolkit/setup code and RTXMFG changes: [MIT](LICENSE). OptiScaler-derived patches:
[GPL-3.0](licenses/OptiScaler-GPL-3.0.txt). Research files carry their own notices;
see [third-party attribution](THIRD_PARTY_NOTICES.md) for details.
