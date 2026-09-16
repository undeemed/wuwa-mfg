# Neural rendering research

[← Research](../README.md) · [Use the NVIDIA engine](../../docs/neural-rendering.md)

**Current status:** the test setup uses NVIDIA's neural engine. The student is
saved for further work after the first live WuWa comparison reported washed-out
output, poor dark-scene accuracy, and stutters. The target remains **1920×1080 in
3 ms without losing image quality**; it has not been met.

The 56-image numerical check validates our DirectML port against our PyTorch
model. It does not establish NVIDIA-equivalent quality or smooth gameplay.

## Start with a question

| Question | Read |
| --- | --- |
| What works, and what is still unresolved? | [Findings](findings.md) |
| How was the student trained? | [Training method and data](training.md) |
| How does live engine switching work? | [Integration and controls](integration.md) |
| How are timings measured outside WuWa? | [Hidden demo benchmark](benchmark.md) |
| What was found inside the NVIDIA runtime? | [Runtime investigation](runtime.md) |
| Where are the older experiments and commands? | [Experiment notebook](EXPERIMENTS.md) |
| Where are the recorded numbers? | [Evidence index](evidence/README.md) |

## Find the code

| Task | Entry points |
| --- | --- |
| Collect and prepare broader data | [collect_broad_sources.py](collect_broad_sources.py), [prepare_broad_sources.py](prepare_broad_sources.py) |
| Generate reference outputs | [compact_teacher_capture.py](compact_teacher_capture.py), [collect_demo_image.py](collect_demo_image.py) |
| Train and compare the current student | [train_broad_student.py](train_broad_student.py), [evaluate_broad_student.py](evaluate_broad_student.py) |
| Read the model architecture | [student_probe.py](student_probe.py), [shared_feature_student.py](shared_feature_student.py), [region_context_student.py](region_context_student.py) |
| Export and check DirectML | [export_dml_student.py](export_dml_student.py), [validate_dml_student.py](validate_dml_student.py) |
| Inspect the native backend | [StudentRuntime.cpp](backend/StudentRuntime.cpp), [StudentGraph.h](backend/StudentGraph.h) |
| Find capture settings and source attribution | [data/](data/) |
| Find experimental OptiScaler probes | [probes/](probes/) |

Python modules stay at this level so their sibling imports and command-line entry
points remain straightforward. `test_*.py` files beside them cover the research
experiments; the root [tests/](../../tests/) directory covers the installer.

## Run an experiment

Use the environment and commands in the [notebook](EXPERIMENTS.md#model-experiments).
Run GPU experiments sequentially, with WuWa and the demo closed. Demo launches use
the [private-desktop helper](../../tools/isolated_demo_process.py).

Keep weights, captures, and generated outputs outside the public repository.
The [integration guide](integration.md) covers local export, build, installation,
live switching, and restore. Check each source file's license notice before reuse.
