# Contributing

Keep changes scoped to reproducible WuWa setup, diagnostics and the startup timing fix. This project does not accept anti-cheat evasion, kernel-driver modifications, blanket removal of hardware checks, or unverified compatibility claims.

Run `python -m unittest discover -s tests -v`. For exact binary reproduction, set `RTXMFG_TEST_ARCHIVE` to the original v1.3.3 release ZIP. No real game/profile/registry mutations are made by the tests.

For a new GPU or driver combination, first document physical PCI ID, driver and NVIDIA wrapper/provider versions/hashes. Distinguish runtime requested/presented counts from visible image quality and independently captured modes from user reports. Include negative results. Do not generalize one successful session to an entire GPU generation.

Preserve upstream attribution, rollback coverage, whole-file digest checks and the narrow modification scope. A source rebuild must describe its toolchain and dependencies rather than claiming the release's binary hash without reproducing it.

Issue reports should omit account names, serials, user directory paths and full registry backups. Start with the issue template; do not upload the game's binaries or NVIDIA files.
