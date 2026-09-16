# Changelog

## 0.2.1 — Uninstall and repository cleanup

- Adds `Uninstall.cmd`, Setup option U, and `python setup.py uninstall` to restore
  managed neural rendering followed by MFG, saved NVIDIA settings, and GPU-name aliases.
- Checks both components before restoration, preserves recovery records, and
  supports retrying an interrupted restore. Manual installations are left unchanged.
- Shortens the README, adds task-based documentation indexes, and groups neural
  research, tools, probes, and evidence without changing the MFG runtime patch.
- Includes the previously published experimental student integration and its
  quality limitations. NVIDIA remains the default neural engine.

## 0.2.0 — WuWa Experience Toolkit

- Retains the RTXMFG 3x/4x/5x/6x unlock, GPU-name spoofing and scoped driver tweaks.
- Adds an optional neural engine using a pinned OptiScaler NR v0.8.4 source build.
- Publishes external-FG compatibility, common-module retention and GPU-aware
  descriptor/constant-slot ownership changes with reproducible regression tests.
- Adds output-relative DX12 model resolution: 100% uses the game output dimensions;
  lower fractions uniformly scale both axes. Full-resolution performance remains
  subject to game validation.
- Adds separate NR install, enable, disable, scale, status and restore actions.
  F8 is the configured toggle; NR starts disabled in guided setup.
- Documents the crash evidence, unresolved attribution, successful tests and
  unsuccessful experiments. NR transition stability remains experimental.
- Source/script distribution only; NVIDIA/model/compiled OptiScaler binaries are
  not included. The OptiScaler source patch is GPL-3.0; toolkit code remains MIT.

## 0.1.0 — DLSS MFG unlock

- Guided setup and rollback for the tested RTX 4070 Ti / driver 616.92 combination.
- Exact-build RTXMFG v1.3.3 wrapper-preparation patch and source equivalent.
- Follow game configuration, NVIDIA profile controls and optional GPU display aliases.
- Fresh-process, FG-active runtime verification and sanitized MFG validation record.
