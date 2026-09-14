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
