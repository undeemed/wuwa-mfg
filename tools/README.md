# Developer tools

[← Repository map](../docs/repository-layout.md)

For ordinary installation, run **Setup.cmd** from the project root. Run the
commands below from that same root when building or investigating the toolkit.

| Tool | Purpose | Instructions |
| --- | --- | --- |
| [BuildNeural.ps1](BuildNeural.ps1) | Build the NVIDIA NR compatibility integration | [Neural setup](../docs/neural-rendering.md) |
| [make_neural_bundle.py](make_neural_bundle.py) | Validate and package a local NR build | [Neural setup](../docs/neural-rendering.md) |
| [check_docs.py](check_docs.py) | Check local documentation links and anchors | [Contributing](../CONTRIBUTING.md#documentation-and-layout) |
| [BuildStudentNeural.ps1](BuildStudentNeural.ps1) | Build the optional research integration | [Student integration](../research/neural/integration.md) |
| [prepare_directml.py](prepare_directml.py) | Fetch and verify pinned DirectML dependencies | [Student integration](../research/neural/integration.md) |
| [install_student_neural.py](install_student_neural.py) | Install or restore a local student bundle | [Student integration](../research/neural/integration.md) |
| [set_neural_engine.py](set_neural_engine.py) | Request a live engine switch or weight reload | [Live controls](../research/neural/integration.md#controls) |
| [inspect_nr_runtime.py](inspect_nr_runtime.py) | Inspect a separately supplied runtime | [Runtime findings](../research/neural/runtime.md) |
| [run_neural_demo_trial.py](run_neural_demo_trial.py) | Run the isolated application benchmark | [Benchmark](../research/neural/benchmark.md) |
| [prepare_hidden_demo.py](prepare_hidden_demo.py), [isolated_demo_process.py](isolated_demo_process.py) | Prepare and contain hidden demo launches | [Benchmark](../research/neural/benchmark.md) |
| [test_capture_ready.py](test_capture_ready.py), [test_neural_hotswap.py](test_neural_hotswap.py) | Check capture readiness and live switching | [Research index](../research/neural/README.md) |

The build scripts moved here from the root. Use `tools/BuildNeural.ps1` and
`tools/BuildStudentNeural.ps1` with a current checkout. The v0.2.0 release ZIP
retains its original root-level build-script paths.
