# Troubleshooting and recovery

## Only an on/off switch appears

The tested PC needed the optional RTX 5080 display-name aliases, followed by a Windows restart, before WuWa exposed its multiplier choices. The aliases change description strings only; they do not provide MFG by themselves. If you skipped them, restore the managed installation first, archive its restored backup folder, then reinstall and select the aliases. Avoid applying unrelated GPU spoofing tools.

## Backspace does nothing

The RTXMFG v1.3.3 overlay did not reliably attach during the working test. This setup uses WuWa's **native graphics settings**, and Verify reads the backend's status files. The presence of an overlay is not required to establish matching runtime counts.

## Counts are still 2x/3x after selecting 6x

Fully close and restart WuWa after installation or changing driver overrides. The tested process retained an older forced 3x count until it restarted. Use Setup → Diagnose to check the two profile entries; run Verify while the game is focused in gameplay with FG on. FPS alone is insufficient evidence.

## Unsupported hardware, driver, or NVIDIA cache

Version 0.1.0 deliberately accepts the tested physical GPU, driver and cache hashes only. It does not download NVIDIA files or replace a driver to manufacture a matching environment. Do not remove validation checks simply to get past this message. A maintainer can evaluate a new combination using a reproducible report and then add support.

## Existing proxy DLL

Setup stops if another proxy such as `dxgi.dll`, `version.dll`, `d3d12.dll` or an unknown `winmm.dll` is present. Use the other mod's documented removal/restore process first. This setup does not rename or chain unknown injectors.

An original RTXMFG v1.3.3 `winmm.dll` is backed up and restored. A manually installed copy of the exact working patch is detected and left unchanged; select Verify instead.

## Hang, black screen or artifacts

Close WuWa and use Restore. A crash during testing is a compatibility failure, even if requested/presented counts agree. Do not disable anti-cheat or security features to continue testing. Provide a concise issue report with the selected mode, driver, physical GPU and whether the fault persists after restoring.

## Rollback stops because something changed

The rollback compares current files, the two NVIDIA settings and any alias records against both the saved originals and the installed values. It stops before overwriting a later edit. Keep `%ProgramData%\WuWaMFG\state.json`; this contains the recovery data.

1. Keep a separate copy of the file or setting named in the error.
2. Reconcile that item with the installed or original state recorded in the backup, or ask for help with that specific conflict. Do not delete the backup to suppress the error.
3. Run Restore again. An interrupted rollback can be retried; items already restored are accepted.

Restore does not require the original driver/cache versions to remain installed. If a driver update moves the GPU registry slot, alias restoration stops rather than writing descriptions into a different device. Resolve that device mapping separately.

After successful restoration, the backup is retained with status `restored`. To do a new installation, move the entire `%ProgramData%\WuWaMFG` folder to an archival location first. It must say `restored`; archiving an active backup would discard managed rollback for the current setup.

## Downloads or Windows prompts

Run Setup.cmd from an extracted folder, not inside a ZIP. Internet access to python.org and GitHub is required on first use. Python is unpacked locally; no Python PATH entry is installed. Setup.cmd's execution-policy flag applies to that PowerShell process only and does not change the saved system policy. Install/Restore request UAC; Diagnose/Verify do not.

If a checksum fails, stop. For a damaged Python download, remove only the setup folder's `.runtime` cache and retry. Do not disable checksum verification. This project does not advise bypassing antivirus or anti-cheat alerts.
