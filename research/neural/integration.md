# Live NVIDIA / student switching

[← Neural research](README.md)

The optional DirectX 12 integration can switch between the original NVIDIA neural
runtime and the locally trained 381,600-parameter student during a session. The
student runs on the same D3D12 command list as OptiScaler's existing encode/blend
path. There is no Python process, CPU image round trip, or additional game hook.

This is an experimental local student, **not an official NVIDIA DLSS model**.
Source, export/build tools and numerical evidence are published. Learned weights,
vendor runtimes, and private captures are not included in this repository.

**Live WuWa quality trial: not accepted.** The first visible gameplay comparison
reported washed-out output, a substantial quality gap from NVIDIA, and occasional
stutters. Numerical port parity and the hidden-demo test below did not establish
equivalent perceptual quality. Keep NVIDIA selected for normal use; the student
remains a research candidate.

The test setup has returned to NVIDIA with `Engine=0`, saved as the default.
The student is retained for targeted dark-scene training and validation.

Sparse gameplay log samples were about 3.86–3.92 ms total with the student and
6.03–6.43 ms with NVIDIA at a 1920×1080 model extent. These are not frame-time
percentiles or a controlled comparison and cannot diagnose the reported stutters.
The colour transform and final composition path are shared by both engines; the
cause of the visible difference still requires matched gameplay inputs. The
student's static-image training and lack of temporal inputs are limitations,
not a confirmed explanation of this particular scene.

## Controls

- **F7:** open OptiScaler. In the neural input controls, use **Neural engine** to
  select **NVIDIA** or **Student (1080p)**. The status line says which is active.
- **F8:** toggle the whole neural effect using the existing binding.
- **Reload student weights:** load an updated local export without restarting.
  NVIDIA runs while the new graph initializes. An incomplete/corrupt export leaves
  NVIDIA active and explains the failure in the status line.
- Save the OptiScaler configuration if you want a menu selection to persist.
  `Engine=0` means NVIDIA; `Engine=1` requests the student.

The first installation replaces the OptiScaler DLL and therefore requires a closed
game. Subsequent engine switches and compatible weight reloads do not.

The trained model currently supports **1920×1080**, one **Natural** pass, preset 0,
intensity/structure/tone 1, skin structure -1, and auto mask enabled. For a 3840×2160
game output, select **50%**, with output-relative scaling enabled. For 1920×1080
output, select 100%. Unsupported sizes or model controls use NVIDIA instead of
silently running a different model. The existing final strength/colour/blend
controls apply to either engine. This student has no temporal history or motion
inputs; its region routing remains data-dependent on the current frame.

An optional file control uses the same selection path, with no key injection:

```powershell
python tools/set_neural_engine.py student --game 'D:\Games\Wuthering Waves'
python tools/set_neural_engine.py nvidia --game 'D:\Games\Wuthering Waves'
python tools/set_neural_engine.py reload --game 'D:\Games\Wuthering Waves'
```

The request is consumed on the next active neural frame. If the effect is off,
enable it with F8 first. This does not change saved preferences automatically.

## Build and install locally

1. Prepare the existing compatibility source and working NVIDIA NR installation
   using [tools/BuildNeural.ps1 and the neural setup](../../docs/neural-rendering.md). Keep its
   original bundle for verification and recovery.
2. Export a compatible **local** trained checkpoint. The exporter uses the existing
   model definitions and requires the research PyTorch environment:

   ```powershell
   python research/neural/export_dml_student.py `
     --first C:\Private\first-student --candidate C:\Private\broad-student `
     --output C:\Private\student-export
   ```

3. Build the integration with Visual Studio 2022 C++ tools, Windows SDK, Git and
   Python available. This applies the incremental student patch to the prepared
   source. Its sibling `directx-headers` dependency remains required:

   ```powershell
   .\tools\BuildStudentNeural.ps1 -CompatibilitySource C:\Private\nr-build\compat-source `
     -StudentModelDirectory C:\Private\student-export `
     -WorkDirectory C:\Private\student-build
   ```

   DirectML 1.15.4 and the DirectMLX header are fetched from Microsoft and verified
   against pinned hashes. Only the exact Windows x64 runtime is extracted.

4. With WuWa closed, install the local bundle over the working NR setup:

   ```powershell
   python tools/install_student_neural.py --game 'D:\Games\Wuthering Waves' `
     --dll C:\Private\student-build\bundle\OptiScaler.dll `
     --model C:\Private\student-build\bundle\neural-student `
     --existing-bundle C:\Private\nr-build\bundle `
     --backup C:\Private\student-backup
   ```

   This selects the student and 50% output-relative scale for the tested **4K**
   setup. It preserves F7/F8 and all unrelated settings. Adjust the scale in F7
   for a different output resolution. The installer verifies the original NR/MFG
   setup, records hashes and backs up every replaced file. It changes only
   `dxgi.dll`, three NR configuration entries, and `neural-student/`.

5. To restore the prior installation, close WuWa and run:

   ```powershell
   python tools/install_student_neural.py --restore --backup C:\Private\student-backup
   ```

   Restore refuses to overwrite files changed since installation; preserve those
   edits first. Restore the student addition before using the older NR restore
   workflow, whose stored DLL/config hashes belong to the prior installation.

To update weights later, write a complete matching `weights.bin` and `model.json`
into `neural-student/`, then use **Reload student weights**. The running engine
keeps its already-loaded weights until reload; the loader verifies the new weight
hash before creating a GPU graph. Keep the private model package out of commits
and releases.

## Validation and limits

The [recorded test](evidence/student-hotswap-v1.json) used
an RTX 4070 Ti with driver 616.92:

- All 56 development images passed DirectML-versus-PyTorch port parity. Mean
  absolute pixel error averaged **0.0000443** on a 0–1 scale; the largest per-image
  99th-percentile error was 0.001221. This measures the port of our student, not
  its similarity to NVIDIA or its generalization to new gameplay.
- One inactive-desktop demo session confirmed **16** switch/reload actions,
  including reload, deliberate missing-weight fallback, and recovery. The demo
  and any dialogs remained off the input desktop. Original demo files were restored.
- Student intervals measured **3.35–3.78 ms total**, including **3.15–3.58 ms**
  for the model/texture transfers. One NVIDIA interval in that same session was
  5.73 ms total. These are sparse smoke-test intervals, not a controlled speedup
  claim. Earlier **2.23 ms CUDA** results use a different backend and exclude the
  game integration.
- The hidden sample was occluded and returned `DXGI_STATUS_OCCLUDED`; it continued
  rendering and producing GPU timing. This test does not establish visible WuWa
  gameplay performance, image stability during movement, or MFG interaction.
- Both engines remain resident for switching. Up to three protected student
  dispatch slots add roughly 0.6 GB of GPU storage. Allocation failure uses the
  NVIDIA fallback. The first visible WuWa trial failed the user's quality
  assessment; the separate final temporal test remains outstanding.

The loader compiles/initializes on a private worker and queue, then dispatches on
the game's existing D3D12 command list. GPU fences and recording-reset tracking
govern reuse and retirement of descriptors, tensor buffers and old models.
Switching back resets NVIDIA history. No game list is closed/submitted by this
backend, and no unrelated queue or anti-cheat behavior is modified.

The initial installed revision prepared only its first dispatch slot on the
worker, then allocated the other two when rendering first used them. A separate
local follow-up prepares all three before publishing the student as ready and
builds successfully. It has not been installed or tested in visible gameplay;
the cause of the reported stutters is still unconfirmed.

Implementation: [incremental OptiScaler patch](../../patches/optiscaler-neural-student.patch),
[original native backend](backend/),
[headless parity checker](validate_dml_student.py), and
[hidden live-switch test](../../tools/test_neural_hotswap.py).
DirectML graph construction follows Microsoft's
[DirectMLX interface](https://learn.microsoft.com/en-us/windows/ai/directml/dml-directmlx)
and [CompileGraph API](https://learn.microsoft.com/en-us/windows/win32/api/directml/nf-directml-idmldevice1-compilegraph).
