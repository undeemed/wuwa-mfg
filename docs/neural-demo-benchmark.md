# Separate neural benchmark application

Use the official **NVIDIA NGX DLSS Sample** as a small graphical test application.
It exercises the same OptiScaler pre-SR neural path without launching or modifying
WuWa. The sample is a rasterized Sponza scene, not a frame-generation benchmark.

## Reproduce the setup

Download `ngx_dlss_demo_windows.zip` from the official
[DLSS SDK v310.9.1 release](https://github.com/NVIDIA/DLSS/releases/tag/v310.9.1).
The tested archive is 84,484,111 bytes, SHA-256
`21c0511c6b45e80c9d0338f504f4f986d261d52050f7fe2af480e3c42581fc35`.
The demo executable is 1,505,280 bytes. Extracted scene assets and libraries take
more space, and the separate neural runtime must also be supplied locally.

1. Extract the archive into a new test directory, outside any game installation.
2. Find `DLSS_Sample_App/bin/ngx_dlss_demo/ngx_dlss_demo.exe`.
3. Place your locally built revision-4 OptiScaler `dxgi.dll`, its `OptiScaler.ini`,
   and your locally obtained `nvngx_dlssnr.dll` beside that executable. Keep the
   sample's own `nvngx_dlss.dll`. No RTXMFG DLL is needed for this NR test.
4. Change the following entries in the **test copy** of `OptiScaler.ini`:

```ini
[ProcessFilter]
TargetProcessName=ngx_dlss_demo.exe

[FrameGen]
Enabled=false
FGInput=nofg
FGOutput=nofg
External=true

[DlssNr]
Enabled=true
RunBeforeSR=true
FinishedPicture=false
DeferredDLSS=false
Passes=1
Preset=0
Style=1
WorkingScaleRelativeToOutput=true
WorkingScale=1.0
AutoCapture=false
ToggleKey=0x77

[Menu]
ShortcutKey=0x76
```

Merge these entries into the existing sections; do not append duplicate sections.
F7 is the menu binding and F8 toggles the effect. Start in the executable's
directory:

```powershell
.\ngx_dlss_demo.exe -d3d12 -width 1920 -height 1080
```

### Actual background operation

The sample calls `glfwShowWindow` after initialization and focuses its window.
Windows hidden-startup flags alone were insufficient: the tester still saw it
open in front. Do not treat `SW_HIDE` or `Start-Process -WindowStyle Hidden` alone
as a verified solution for this executable.

With the demo closed, apply the optional
[background preparation script](../tools/prepare_hidden_demo.py) to the separate
test executable:

```powershell
python tools/prepare_hidden_demo.py "D:\NR-Demo\DLSS_Sample_App\bin\ngx_dlss_demo\ngx_dlss_demo.exe"
```

It accepts only executable SHA-256
`d28d05c82fc776189c6049649f67b34b1ed2b48f76b2dcfac4fc3c3c4366af35`, saves an
exact `.visible-original` backup and makes the demo's void `glfwShowWindow`
function return immediately. The one-byte change is at RVA `0x82EC0`; it leaves
the renderer and neural runtime unchanged. Result SHA-256:
`f262742631d02da935649322f220f0490b114287c98a1d75d60e04557f98a9c6`.
The equivalent source edit is an immediate return in that GLFW function. Donut
already creates the window with `GLFW_VISIBLE=false`, and its render loop uses
window dimensions rather than visibility to decide whether to render.

During the patched test, Windows reported no visible main window
(`MainWindowHandle=0`), while fresh 1920×1080 NR timing records continued. To
restore the interactive demo, close it and copy the `.visible-original` backup
over `ngx_dlss_demo.exe`. No patched executable is distributed by this repository.

With the sample's initial DLSS Performance mode, render input is 960×540 and
display output is 1920×1080. Output-relative scale **1.0** therefore creates a
**1920×1080 neural model**. This matches the model extent of WuWa at 4K with scale
0.5, but not its render input, full composition cost, scene content, or FG load.

## What has been verified

On the tested RTX 4070 Ti, the demo loaded the same revision-4 OptiScaler build
and SF-v2 runtime used by WuWa. Its log recorded successful creation of feature
18 at 1920×1080 through the direct compatibility runtime, and a successful neural
evaluation with one model pass. These records establish initialization and an
evaluation; they do **not** establish continuous rendering or a 3 ms result.

The tester confirmed the scene was visible and closed WuWa. Fresh recurring
`DLSS-NR elapsed` records then became available. The first contended observation
with both applications running is excluded from the comparisons below.

## Initial comparisons

These short trials used the same executable, runtime, model extent and initial
scene. Summaries retain records first observed after the 15-second polling mark
(polling runs every five seconds), using the existing sparse GPU timing records.
This is an approximate warmup cutoff, not a full per-frame profile.

| Trial | Median total NR | Median model | Outcome |
| --- | ---: | ---: | --- |
| Natural style, preset 0, automatic mask on | 6.105 ms | 5.880 ms | Completed 45-second observation |
| Automatic mask off | 6.100 ms | 5.900 ms | Device loss during observation |
| Preset 1 | 6.090 ms | 5.745 ms | Device loss during observation |
| Standard style | 6.050 ms | 5.845 ms | Completed 45-second observation; appearance differs |
| Full root-state restoration options | 6.600 ms | 6.260 ms | Completed 60 seconds, but restoration errors were logged; rejected |
| Baseline with a 60 FPS demo cap | 6.520 ms | 6.315 ms | Completed 90-second observation; reduced GPU load, not model cost |
| Windows hidden-startup flag, 60 FPS cap | 6.410 ms | 6.210 ms | Still opened in front according to tester; startup flag alone rejected |

The ordinary settings did not demonstrate a meaningful speedup, much less a
1920×1080 model in 3 ms. The model accounts for about 5.9 ms of a typical 6.1 ms
interval; host-side GPU work accounts for about 0.2 ms. Removing that entire
surrounding cost would still fall well short of the target. No settings from
these experiments were copied back to WuWa.

Two saved demo dumps were inspected offline. Windows reported `0x80000003`, and
the immediate return address was in the demo at RVA `0xD6855`. Disassembly places
it immediately after a `DebugBreak` reached when `GetDeviceRemovedReason` fails,
matching the sample's included NVRHI source. Thus the breakpoint reports a lost
D3D12 device; it does not identify the underlying faulty shader, resource or
driver operation. Device loss occurred in an initial baseline as well, so it
cannot be attributed to the mask toggle alone. No debugger was attached to WuWa.

The restoration experiment emitted missing-original-function errors for graphics
root CBV and descriptor-table restoration. A run surviving briefly with those
errors is not evidence that the integration is fixed. That option was reverted.

Aggregate records are in [neural-demo-summary.json](../evidence/neural-demo-summary.json).

At a 60 FPS demo cap, observed GPU utilization was about 53–55% rather than the
uncapped 100%. This makes the test application less demanding overall, while the
individual neural pass still costs about 6–6.5 ms. The 90-second run had no device
loss; that is a short observation, not a demonstrated fix for the earlier faults.
The subsequent executable-level background change also continued rendering and
reporting timings with no visible main window reported by Windows.

## Bounded trial runner

The [source runner](../tools/run_neural_demo_trial.py) changes only the configured
demo's INI, starts that demo hidden, records local logs/timings, stops its own process,
and restores the original INI in a `finally` block. It refuses to start while
WuWa or another instance of this demo is running. Use a new label for each trial:

```powershell
python tools/run_neural_demo_trial.py baseline --demo-dir "D:\NR-Demo\DLSS_Sample_App\bin\ngx_dlss_demo" --output-dir "D:\NR-Trials"
```

Options include `--mask false`, `--preset 1`, `--style 0`, `--fps 60` and
`--seconds 90`. Keep raw logs and machine-specific output local. The source is
provided; NVIDIA binaries and assets are obtained separately from their owners.
The runner requests hidden startup, but the executable-level preparation above
is also required for this demo. Successful feature creation alone is insufficient
if an application pauses while hidden. Require fresh timing records.

## Compare changes

Keep the scene, model extent, style, pass count and camera fixed. Let compilation
and model initialization finish. Collect multiple fresh `DLSS-NR elapsed` records
from the demo's `OptiScaler.log`, and report total NR time separately from model
time. These are sparse GPU intervals, not a per-frame trace or input latency.

Compare baseline, candidate, then baseline again under the same GPU load. A clean
run requires the game and other GPU workloads to be stopped by the user. While
they remain active, label observations as contended and do not promise the same
timing in WuWa. Check the picture and motion as well as timing: reducing model
resolution, skipping frames, disabling the effect, or changing its appearance is
not an equivalent full-1080p optimization.

A successful demo optimization would still need verification in WuWa. Shared
runtime/host improvements may carry over; game-specific scheduling, SR, FG and
resource-transition behavior may not. No 1920×1080-at-3-ms improvement has yet
been established.

## Earlier harness experiment

A separate NR-only harness was also built from
[DLSS5-Reshade-AIO's lab](https://github.com/kibblerz/DLSS5-Reshade-AIO/tree/09301f5528e619e8b9ec17c257d167e2985f53b0/lab).
A 320×180 compatibility run created/evaluated the runtime successfully, but its
short, contended timings were unsuitable for benchmarking. Optional kernel
instrumentation compiled but was not validated or used to identify active
kernels. The existing NVIDIA demo was selected at the tester's request instead.
Neither harness binaries nor NVIDIA runtime/model assets are distributed here.
