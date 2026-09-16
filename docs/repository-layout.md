# Repository layout

[← Documentation](README.md) · [Research](../research/README.md) · [Tools](../tools/README.md)

Start at `Setup.cmd` to install, `docs/` to use the toolkit, or `research/` to study
the experiments. User guides stay directly under `docs/`; research reports live
next to their code. No runtime files or learned weights are included.

## Current layout

```text
wuwa-toolkit/
├── README.md                    Project overview and quick start
├── Setup.cmd                    Windows setup entry point
├── Uninstall.cmd                Restore all managed add-ons
├── Launch.ps1                   Interactive setup menu
├── setup.py                     Command-line entry point
├── CONTRIBUTING.md              Development and test instructions
├── CHANGELOG.md
├── docs/                        User guides and implementation references
│   ├── README.md                Documentation index
│   ├── installation.md          Requirements, setup, menu, and restore
│   ├── neural-rendering.md      Optional NVIDIA neural engine
│   ├── troubleshooting.md
│   ├── validation.md
│   ├── technical.md             RTXMFG patch details
│   ├── neural-crashes.md
│   ├── investigation-history.md
│   └── repository-layout.md     This map
├── wuwa_mfg/                    Installer implementation
├── neural/                      Default OptiScaler configuration
├── patches/                     Three game-integration source patches
├── tools/                       Build scripts and diagnostic helpers
│   ├── README.md                Tool index
│   ├── BuildNeural.ps1
│   ├── BuildStudentNeural.ps1
│   └── *.py
├── tests/                       Installer and rollback tests
├── evidence/                    MFG and neural gameplay summaries
├── research/
│   ├── README.md                Research index
│   ├── neural/
│   │   ├── README.md            Task-based code and report index
│   │   ├── findings.md          Current status and limitations
│   │   ├── training.md
│   │   ├── integration.md       Student engine and live switching
│   │   ├── benchmark.md
│   │   ├── runtime.md
│   │   ├── EXPERIMENTS.md       Full experiment notebook
│   │   ├── *.py, *.cu           Related research modules and tests
│   │   ├── backend/             DirectML runtime and standalone probe
│   │   ├── probes/              Experimental OptiScaler patches and headers
│   │   ├── data/                Capture definitions and source attribution
│   │   └── evidence/            Sanitized numeric experiment records
│   └── windows-wmi/             WMI report, control source, and evidence
├── licenses/                    Upstream license texts
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
└── .github/                     CI and issue template
```

The related Python research modules remain together: they share imports and
direct script entry points. The short [research index](../research/neural/README.md)
provides routes by task, so browsing all of them is optional.

## Before this cleanup

The original tree had **300 tracked files**, excluding local caches and outputs:

```text
wuwa-mfg/
├── Setup.cmd, Launch.ps1, setup.py
├── BuildNeural.ps1, BuildStudentNeural.ps1
├── README.md, CONTRIBUTING.md, CHANGELOG.md
├── docs/                        11 mixed user guides and research reports
├── wuwa_mfg/                    7 installer modules
├── neural/                      1 configuration
├── tools/                       10 helpers
├── patches/                     3 integration patches
├── tests/                       5 test files
├── evidence/                    77 numeric records
│   └── neural-model-research/   72 of those records
├── research/
│   ├── neural-latency/          165 scripts, probes, data, and notebook files
│   └── windows-wmi/             2 files
└── licenses/                    4 upstream license texts
```

The long front page duplicated setup details. Research reports were mixed into
user documentation, most result files lived elsewhere, and the research README
opened directly into the full experiment log.

## Path changes

| Previous location | Current location |
| --- | --- |
| `BuildNeural.ps1`, `BuildStudentNeural.ps1` | `tools/` |
| `docs/neural-latency-research.md` | [research/neural/findings.md](../research/neural/findings.md) |
| `docs/broad-student-training.md` | [research/neural/training.md](../research/neural/training.md) |
| `docs/student-hotswap.md` | [research/neural/integration.md](../research/neural/integration.md) |
| `docs/neural-demo-benchmark.md` | [research/neural/benchmark.md](../research/neural/benchmark.md) |
| `docs/nvidia-runtime-investigation.md` | [research/neural/runtime.md](../research/neural/runtime.md) |
| `research/neural-latency/README.md` | [research/neural/EXPERIMENTS.md](../research/neural/EXPERIMENTS.md) |
| `research/neural-latency/*.py`, `*.cu` | `research/neural/`, with filenames preserved |
| `research/neural-latency/native-student/` | `research/neural/backend/` |
| Loose neural probe patches and `DlssNr_Demo*.h` headers | `research/neural/probes/` |
| Neural capture/source JSON definitions | `research/neural/data/`, with filenames preserved |
| `evidence/neural-model-research/` | `research/neural/evidence/` |
| `evidence/neural-demo-summary.json`, `neural-kernel-timing.json` | `research/neural/evidence/` |
| `evidence/neural-wmi-hang.json` | `research/windows-wmi/evidence.json` |

Historical evidence retains its original bytes, source paths, and hashes.
Commands that read source from an older Git commit retain that commit's paths.
Existing v0.2.0 release archives retain their original structure; these paths
describe the current source checkout.
