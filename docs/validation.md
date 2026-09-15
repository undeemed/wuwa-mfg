# Validation record

One RTX 4070 Ti 12 GB, NVIDIA driver 616.92, Wuthering Waves on Windows, September 14, 2026.

| Run | Runtime evidence | Visual report |
| --- | --- | --- |
| Patched RTXMFG, 3x requested | 53 FG-active samples: accepted/applied, result 0, multiplier 3, frames presented 3, bridge ready | Normal image and smooth motion |
| Follow game, global count 0, fresh game process | 67 FG-active samples: accepted/applied, result 0, native request 6, frames presented 6, bridge ready | Tester reported 4x/5x/6x working smoothly after raising graphics settings |

The 3x capture covered roughly 81 seconds between the first and last active samples. The later capture contained 180 total samples; 67 were FG-active and the remaining samples were excluded. The later runtime capture observed 6x only. **4x and 5x were user-reported; they were not individually captured.**

These are status observations plus human visual reports, not an independent frame uniqueness analysis, latency benchmark, exhaustive game-compatibility matrix or long-term stability test. No anti-cheat/account-safety qualification was performed.

## Installer validation

- Reproduced the exact tested patched DLL from the checksum-pinned upstream ZIP.
- Verified the only binary differences are five code bytes and three checksum bytes.
- Original and modified predicate fragments were tested during the investigation in an isolated local process against Unknown, Ada, Ampere, Conflict and invalid states; all ten results matched the expected truth tables.
- Automated installer tests use temporary folders and a simulated Windows backend. They cover clean setup, preservation of an existing official mod/config, restore, interrupted installs, post-save failures, partial registry failures, existing-mod conflicts, later user edits and stale telemetry.
- Read-only Windows integration checks recognized the live working installation and read matching 6x telemetry.
- All 28 tests passed locally on Windows, including ctypes DRS translation, global/local override ownership and a predefined-setting conflict that must stop before saving.

The packaged installer was not run to overwrite the tester's already-working game or change its live profiles. Windows/NVIDIA mutation paths are therefore validated by API inspection and simulated failure tests, while the installed DLL and final settings have the separate gameplay evidence above.

Sanitized aggregate data is in [`evidence/session-summary.json`](../evidence/session-summary.json). Raw game logs, screenshots, user paths, machine identifiers and original registry/profile backups are intentionally not published.

## Version 0.2.0 neural extension

The full suite now contains **43 passing tests**, including the exact upstream
RTXMFG binary test and 15 neural install/config/rollback tests. Neural mutation
tests use synthetic bundles and temporary game folders. Packaging the actual
locally built OptiScaler DLL with the pinned upstream backends and NR runtime also
passed bundle validation; no proprietary file is included in the public release.

The source patch builds in Release x64. Its descriptor regression rejects the old
allocator as a negative control; the guarded test and production GPU-lifetime
regression pass on WARP. Output-extent tests pass for 4K, ultrawide, changing DLSS
input size, invalid inputs and uniform limits. These tests do not establish neural
visual quality, latency or stability in WuWa.

Revision 2 ran NR at 1281×721 alongside actual 6x before a transition GPU crash.
Revision 3 adds the reproduced descriptor-lifetime guard; its initial game run
showed actual 6x and roughly 4 ms NR timing with no new crash during observation.
A following full-output run created the model at 3840×2160 and produced 28
accepted, FG-active, matching 6x observations with distinct heartbeats out of 30
polls. Recent total NR intervals were 21.42 and 24.22 ms, with a GPU memory snapshot
of 11402/12282 MiB used. Two transition/off or mismatched polls were excluded.
The user has not yet assessed responsiveness or confirmed exact-trigger repetition;
this is an initial runtime result, not a long-term stability or input-latency test. See [the complete findings](neural-crashes.md) and [work history](investigation-history.md).
