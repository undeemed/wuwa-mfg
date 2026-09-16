# Neural research evidence

[← Neural research](../README.md)

These are immutable numeric records from individual experiments. Their recorded
source paths and hashes refer to the layout at the time of the run. The September
2026 documentation cleanup moved files without rewriting these records.

## Recent work

| Record | What it establishes |
| --- | --- |
| [broad-data-v1.json](broad-data-v1.json) | Broader-data training and development-set results |
| [student-hotswap-v1.json](student-hotswap-v1.json) | DirectML/PyTorch parity and hidden-demo switching; not gameplay quality |
| [region-context.json](region-context.json) | Region attention compared with its dense control |
| [region-export.json](region-export.json) | Export/reload parity and separate inference timings |
| [cluster-sampling.json](cluster-sampling.json) | Sampling coverage and its validation limitations |

## Earlier measurements

- [neural-demo-summary.json](neural-demo-summary.json): the separate application benchmark.
- [neural-kernel-timing.json](neural-kernel-timing.json): observed native kernel timings.
- [Experiment notebook](../EXPERIMENTS.md): interpretation and links for the remaining records.

For the visible gameplay rejection and return to NVIDIA, see the
[current findings](../findings.md) and [integration assessment](../integration.md).
Those later observations do not retroactively turn the earlier numerical tests
into image-quality or frame-pacing validation.

Old `evidence/neural-model-research/<file>` paths now resolve to this directory.
Old `research/neural-latency/` source paths are mapped in the
[repository layout](../../../docs/repository-layout.md).
