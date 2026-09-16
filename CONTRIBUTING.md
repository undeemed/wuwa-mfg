# Contributing

[Project home](README.md) · [Repository map](docs/repository-layout.md) · [Tool index](tools/README.md)

Contributions should stay focused on reproducible WuWa graphics setup, MFG,
optional neural rendering, diagnostics, and recovery. Include unsuccessful
results when they help establish compatibility limits.

## Development setup

The installer uses **Python 3.10+ and the standard library**. Windows integration
uses NVAPI DRS and registry APIs; installer tests use simulated backends and
temporary game folders.

From the repository root:

```powershell
python -m unittest discover -s tests -v
python tools/check_docs.py
```

Set `RTXMFG_TEST_ARCHIVE` to the original RTXMFG v1.3.3 release ZIP to include the
exact-binary integration test. CI downloads that pinned archive and tests on
Windows and Linux. The installer tests do not modify a real game, profile, or registry.

Read-only checks for an installed game:

```powershell
python setup.py doctor --game "D:\Games\Wuthering Waves"
python setup.py verify --game "D:\Games\Wuthering Waves" --seconds 30
```

To reproduce the binary edit without installing it:

```powershell
python setup.py patch --input original-v1.3.3.dll --output patched-winmm.dll
```

Input and output are checked against known SHA-256 hashes. The output must be a
new file. See [the implementation](docs/technical.md) for the exact edit.

## Compatibility reports

Use the [issue template](https://github.com/undeemed/wuwa-mfg/issues/new/choose).
For a new configuration, include the physical GPU PCI ID, driver version, and
NVIDIA wrapper/provider versions and hashes. Distinguish runtime requested and
presented counts, observed picture quality, and tester reports.

Omit account names, serials, personal directory paths, and full registry backups.
Do not upload game binaries, NVIDIA files, model weights, or raw private captures.

## Code changes

Preserve upstream attribution, rollback coverage, whole-file digest checks, and
the narrow modification scope. A source rebuild must describe its toolchain and
dependencies; do not claim a release binary's hash without reproducing it.

For OptiScaler compatibility changes, update
`patches/optiscaler-wuwa-compat.patch` against the pinned v0.8.4 commit. Keep its
GPL-3.0 licensing, run the descriptor-slot negative control and guarded WARP tests,
the production GPU lifetime regression, and the output-extent test. Build Release
x64 using [the neural build instructions](docs/neural-rendering.md).

Record real gameplay separately. An isolated lifetime test does not prove that
a GPU hang is fixed, and numerical model parity does not prove visual equivalence.
See [crash findings](docs/neural-crashes.md) and [research status](research/neural/README.md).

The project does not accept anti-cheat evasion, kernel-driver modifications,
blanket removal of hardware checks, or unverified compatibility claims.

## Documentation and layout

Keep the front-page README focused on installation and use. Put user guides in
`docs/`, build helpers in `tools/`, and research reports beside their code. Update
the relevant index when adding a file and run `python tools/check_docs.py` after
moving one. Preserve historical evidence bytes and document path migrations in
the [repository map](docs/repository-layout.md).
