# WuWa MFG

**A reproducible RTXMFG startup patch and guided Windows installer for Wuthering Waves.**

The tested setup lets WuWa select its own frame-generation multiplier through **6x**, using Follow game mode. Runtime telemetry reported matching 3x and 6x output on an RTX 4070 Ti, with normal visible gameplay reported by the tester. This is an **experimental, narrowly validated release**.

[Download the setup ZIP](https://github.com/undeemed/wuwa-mfg/releases/latest) · [How the patch works](docs/technical.md) · [Test results](docs/validation.md) · [Troubleshooting](docs/troubleshooting.md)

## Quick start

1. Download **WuWa-MFG-0.1.0.zip** from Releases and extract the entire folder somewhere you can keep it.
2. Close Wuthering Waves. Double-click **Setup.cmd**, choose **Install**, and approve the normal Windows administrator prompt.
3. Setup finds Steam libraries automatically. For another launcher, paste the WuWa folder when prompted. It must resolve to `Client/Binaries/Win64/Client-Win64-Shipping.exe`.
4. If WuWa currently offers only an on/off FG switch, choose **yes** for the optional GPU display-name aliases. The tested setup used these aliases to expose the native multiplier menu. Review the target and type `INSTALL`.
5. Restart Windows if you applied the aliases; otherwise launch a fresh game process. Enable DLSS Frame Generation and select the multiplier in WuWa's graphics settings.
6. Choose **Verify** in Setup.cmd, then return to focused gameplay. It samples fresh runtime status for 30 seconds and reports requested/presented counts. Check that the image and motion also look normal.

No Python installation or NVIDIA Profile Inspector download is required. Setup downloads a checksum-pinned, official Python runtime into its own `.runtime` folder, then downloads and verifies the original upstream RTXMFG archive. The tested DLL is reproduced locally. This repository/release contains **source and scripts, with no mod, game, or NVIDIA binaries bundled**.

## Supported configuration in 0.1.0

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

## What setup changes

- Installs one patched `winmm.dll` beside WuWa's shipping EXE.
- Sets `RTXMFG-Universal.json` to **Follow game**, Preset B. It does not force a multiplier or invent a dynamic FPS target. The saved `multiplier: 4` field is ignored while Follow game is enabled.
- Enables NVIDIA's global FG DLL override (`0x10E41E03 = 1`) and sets its generated-frame count override to application control (`0x104D6667 = 0`). It removes user overrides for those same two settings in WuWa's driver profile so the game inherits them. **These global settings can also affect other compatible games.** Other profile settings are preserved.
- Optionally changes three Windows GPU description strings to `NVIDIA GeForce RTX 5080` to expose WuWa's native menu. This affects names shown system-wide; the physical GPU stays a 4070 Ti. Setup never restarts Windows automatically.

Follow game accepts supported choices made by WuWa. Dynamic mode is followed **only if the game requests it**. This is not unlimited frame generation; the tested capacity is five generated frames plus one rendered frame, or 6x total.

## Undo

Close WuWa, open **Setup.cmd → Restore**, review the target, and type `RESTORE`.

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
- [NVIDIA NVAPI](https://github.com/NVIDIA/nvapi): DRS interface definitions.

[MIT license](LICENSE) for this project's source. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` for upstream notices. NVIDIA, Kuro Games and RTXMFG's maintainers do not endorse this project.
