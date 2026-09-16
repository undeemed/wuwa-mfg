# Broader training for the fast neural student

[← Neural research](README.md)

This experiment continues the existing **381,600-parameter region-routed
student**. It does not change the inference architecture, replace NVIDIA model
weights, or install a model in WuWa. The normal neural runtime remains separate.

The purpose is to test whether more varied native-teacher examples improve
fidelity while retaining the student's low standalone inference cost. The
[protocol](data/broad-data-protocol.json) fixes the training
budget and separates development images from future gameplay acceptance tests.

## Completed development result

All 240 new reference captures passed their four-frame resource checks and
restored the sample. The fixed **6,000 updates / 12,000 example presentations**
completed in **321.31 seconds**. All 262 training pairs were sampled, and the
frozen first stage stayed identical. No development image entered optimization.

The new checkpoint reduces mean RGB absolute error **12.42%** across the 56
development images relative to the previous routed checkpoint. **46 improve and
10 regress.** This is an error reduction, not a percentage estimate of visual
quality or gameplay reliability.

| Development group | Images | Previous RGB MAE | New RGB MAE | Error change |
| --- | ---: | ---: | ---: | ---: |
| All | 56 | 0.023801 | 0.020845 | −12.42% |
| Existing scene views | 6 | 0.025626 | 0.025578 | −0.18% |
| Earlier photo group | 6 | 0.020807 | 0.023048 | **+10.77%** |
| Previous diverse photos | 4 | 0.022438 | 0.020264 | −9.69% |
| New author-separated photos | 40 | 0.024113 | 0.019863 | −17.63% |

The earlier 16-image set as a whole regresses **1.21%**. Its largest regression
is the bright landscape case: MAE rises from 0.027992 to 0.041925. A focused
rerun reproduces its recorded metrics and confirms visible color/detail
differences. This regression is retained, not removed from the average.

In the same alternating full-model benchmark, the previous and new routed
models take **2.242 ms and 2.230 ms** median, respectively, at true 1920×1080.
Their interval p95 values are 2.277 and 2.264 ms. Treat this as essentially the
same inference cost; the architecture did not change. The frozen first model
alone takes 1.098 ms with mean MAE 0.021464 on these 56 images. All 168
model/image checks match the existing fused and unfused outputs bit for bit.

Six preselected visual comparisons still show differences from native texture,
shading, highlights and color. The additional worst-regression inspection was
chosen after scoring and is labeled accordingly. Content bucket names are
search categories, not certified coverage; a portrait's center crop, for
example, may omit the face. No perceptual-equivalence, motion or complete-game
latency acceptance is claimed. **The candidate remains uninstalled.**

[Full numeric evidence](evidence/broad-data-v1.json)
includes every development score, training sampling count, alternating timing
interval, capture-speed pilot and preservation check. Source images, capture
pixels and learned weights remain private.

## Data and fitting

- Retain the previous 62 training frames: 30 scene views and 32 photographs,
  including the existing brightness variants.
- Add 200 high-resolution, licensed source photographs rendered through the
  native teacher at a true **1920×1080** model extent.
- Reserve 40 new photographs by author and retain all 16 earlier development
  images. None of these 56 images enters optimization.
- Warm-start the existing routed checkpoint. Freeze its first 254,672
  parameters and train the existing 126,928-parameter correction stage.
- Perform 6,000 AdamW updates, each using one previous example and one new
  example. The previous example is sampled equally between scene and photo
  groups, giving expected group weights of 25%, 25% and 50%.
- Keep FP32 fitting with TF32 disabled, the original image/gradient loss, and
  the fixed cosine learning-rate schedule from 0.0002 to 0.000002.

Source selection uses 20 content buckets, exact content hashes, perceptual
deduplication and visual screening. Author grouping reduces overlap between
the new training and development partitions. It does not establish independent
gameplay coverage. These are still-image development examples, not 240 gameplay
trials or evidence of a 99% gameplay pass rate.

The [source attribution catalogue](data/broad-source-attribution.json)
records the 240 source pages, authors, licenses, hashes and fixed partitions.
It contains no source images or teacher outputs.

## Faster reference collection

The opt-in `--capture-only` mode ends a hidden demo trial after four complete,
GPU-fenced frames are saved. It verifies metadata, extents and payload sizes
before stopping the owned process. The collector then restores the sample,
checks all 16 resource hashes, and applies lossless Windows file compression.

A repeat capture pilot reproduced every resource hash from the earlier capture.
Its demo run took 4.046 seconds; the complete capture command took 9.616 seconds,
including startup, restoration and compression. The earlier collector waited
for its 20-second capture window. This reduces collection waiting time; it is
**not a model inference latency result**. Ordinary latency benchmarking retains
its warmup and timing collection, with capture-only disabled by default.

Every demo launch still checks the pinned hidden executable and runs on an
inactive private desktop. The tool verifies that the input desktop did not
change and that the demo windows remain hidden. No game process is opened.

## Reproduction

These are research tools for an existing local lab with the previously audited
teacher capture build, hidden demo and student checkpoints. The public
repository does not distribute those binaries, checkpoints or captured pixels.

```text
prepare_broad_sources.py --lab <private-lab> --output <private-lab>/broad-sources-v1
collect_broad_sources.py --sources <private-lab>/broad-sources-v1 --base <private-demo-root> --demo <hidden-demo-directory> --capture-dll <verified-local-capture-build> --capture-sha256 <verified-hash> --output <private-lab>/broad-teacher-v1.json --capture-only
train_broad_student.py --base <private-demo-root> --first <frozen-first-model> --candidate <previous-routed-model> --sources <prepared-manifest.json> --teacher <teacher-manifest.json> --review <completed-visual-review.json> --output <fresh-private-model-directory>
evaluate_broad_student.py --base <private-demo-root> --lab <private-lab> --photos <old-photo-training-manifest.json> --images <old-diverse-photo-manifest.json> --first <frozen-first-model> --previous <previous-routed-model> --previous-evaluation <previous-evaluation/result.json> --candidate <new-private-model-directory> --sources <prepared-manifest.json> --teacher <teacher-manifest.json> --output <fresh-private-comparison-directory>
```

Keep the game closed during GPU collection, fitting and measurement. A private
`pause-image-collection.enable` file in the demo trial root prevents the next
capture launch. Each completed capture can be resumed from its manifest; an
incomplete trial requires inspection rather than automatic reuse.

The visual-review record must identify the exact source-manifest hash, mark
every selected source approved, contain no rejected entries, and set `complete`
only after actual inspection. The trainer checks that record and all original
input/target hashes. Its host cache stores the original FP16 texture values
exactly and recomputes frozen features during fitting to bound GPU memory.

The evaluator compares the frozen first model, previous routed checkpoint and
new checkpoint on the same 56 images. It rechecks the earlier outputs and
scores, proves the existing fusions match unfused output bits, and alternates
complete-model CUDA Graph timings. It reports regressions as well as gains.

## What remains before deployment

Standalone model timing excludes texture conversion, application
synchronization and composition. Ten-replay interval p95 is not frame p95.
Still-image error metrics cannot establish perceptual equivalence or temporal
stability. The target remains a complete effect below 3 ms at true 1080p with
acceptable quality in representative gameplay; this training run alone cannot
establish that target.

The separate final protocol calls for 300 independently selected ten-second
gameplay sequences after freezing the candidate. No such final trials have
been collected for this experiment. Native runtime integration and real motion
testing remain required before installing a replacement.
