# Installation and restore

[← Documentation](README.md) · [Troubleshooting](troubleshooting.md)

## Requirements

The installer accepts the following validated configuration. Other combinations
need separate testing before support can be added.

| Component | Validated configuration |
| --- | --- |
| OS | 64-bit Windows |
| Physical GPU | RTX 4070 Ti, PCI `10DE:2782`, one discrete GPU |
| NVIDIA driver | 616.92 / Windows `32.0.16.1692` |
| Game | Wuthering Waves, Steam build tested September 14, 2026 |
| Upstream mod | RTXMFG v1.3.3, exact SHA-256 pinned |
| NVIDIA wrapper | 2.14.0.0, cache version `134656` |
| NVIDIA provider | 310.9.0.0, cache version `20318464` |

Setup checks the physical GPU, driver, cache hashes, and existing proxy DLLs.
It does not install drivers or NVIDIA cache files. Driver, game, and NVIDIA OTA
updates can change compatibility. See [validation](validation.md) for the evidence.

You need internet access to GitHub and python.org on first use. Setup downloads a
checksum-pinned official Python runtime into its own `.runtime` folder; no system
Python or NVIDIA Profile Inspector installation is needed.

## Install MFG

1. Download **WuWa-Experience-Toolkit-0.2.0.zip** from the
   [latest release](https://github.com/undeemed/wuwa-mfg/releases/latest).
   Extract the whole folder and keep it for verification and recovery.
2. Close WuWa. Run **Setup.cmd**, choose **1**, and approve the normal Windows
   administrator prompt.
3. Check the detected Steam installation. For another launcher, paste the game
   folder when prompted; it must resolve to `Client/Binaries/Win64/Client-Win64-Shipping.exe`.
   Folder detection does not establish compatibility with an untested build.
4. If the game only offers FG on/off, choose **yes** for the optional RTX 5080
   display-name aliases. The validated setup used them to expose multiplier choices.
5. Review the target and type `INSTALL`. Setup downloads RTXMFG, patches it locally,
   checks the resulting hash, and saves backups before applying changes.
6. Restart Windows if you applied display-name aliases. Otherwise, start a fresh
   game process. Enable DLSS Frame Generation and select the multiplier in WuWa.

The release contains source and scripts. You do not need to apply a `.patch` file
manually. [Exact patch procedure and hashes →](technical.md#exact-binary-edit)

## Verify

Open **Setup.cmd → 2**, then return to focused gameplay with FG enabled. The check
samples fresh RTXMFG status for 30 seconds and compares requested and presented
frame counts. Confirm that the picture and motion also look normal.

An FPS counter or a working menu alone does not establish the multiplier. Runtime
verification also does not measure input latency or prove that every displayed
frame is unique. **6x** means up to five generated frames plus one rendered frame.

## What setup changes

| Change | Scope |
| --- | --- |
| Patched `winmm.dll` | Beside the game's shipping EXE |
| `RTXMFG-Universal.json` | Follow game mode, Preset B; no forced multiplier |
| NVIDIA FG DLL override | Global `0x10E41E03 = 1` |
| NVIDIA generated-frame count override | Global `0x104D6667 = 0`, application-controlled |
| Matching WuWa profile entries | Removes user overrides for those same two settings so the game inherits them |
| Optional RTX 5080 aliases | Three Windows description strings; names change system-wide |

The **global NVIDIA settings can affect other compatible games**. Other profile
settings are preserved. The name aliases do not change the physical GPU or hardware
IDs. Setup does not restart Windows automatically.

The saved `multiplier: 4` field is ignored in Follow game mode. Dynamic output is
followed only if WuWa requests it; this does not provide unlimited frame generation.

This is a third-party graphics mod. Anti-cheat approval and account safety are
not established. Setup does not modify anti-cheat, kernel drivers, hardware IDs,
or Windows security settings; RTXMFG modifies the game's graphics pipeline in memory.

## Setup menu

Run `Setup.cmd` again whenever you need one of these actions.

| Option | Action | Close WuWa first? |
| --- | --- | --- |
| 1 | Install MFG unlock | Yes |
| 2 | Verify runtime frame counts | No; use focused gameplay |
| 3 | Restore MFG setup | Yes |
| 4 | Diagnose compatibility | No; read only |
| 5 | Install a locally built neural add-on | Yes |
| 6 / 7 | Enable / disable neural rendering for the next launch | Yes |
| 8 | Restore the neural add-on | Yes |
| 9 | Read neural status | No; read only |
| 10 | Set neural model resolution scale | Yes |

For the optional neural engine, follow [its build and setup guide](neural-rendering.md).
For the separate research student, use [the research integration guide](../research/neural/integration.md).

## Restore

1. Close WuWa.
2. If you installed the separate student integration, restore it using
   [its own backup procedure](../research/neural/integration.md#build-and-install-locally).
3. If the managed neural add-on is installed, use **Setup.cmd → 8** first.
   To keep it installed but turn its effect off, use **7** instead.
4. Restore MFG with **Setup.cmd → 3**, review the target, and type `RESTORE`.
5. Restart Windows if GPU description strings were restored.

Original MFG files, both scopes of the two profile settings, and changed descriptions
are saved in `%ProgramData%\WuWaMFG\state.json` before changes begin. Restore keeps
the backup. If a file or setting changed after installation, rollback stops rather
than overwriting it. See [recovery instructions](troubleshooting.md#rollback-stops-because-something-changed).

A manually installed copy of this exact patch is left in place. Use Verify for
that installation; the installer cannot reconstruct its earlier originals.
