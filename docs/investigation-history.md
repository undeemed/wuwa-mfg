# Investigation history and work inventory

This records the work behind the toolkit, including failed experiments. The scope
is the tested WuWa installation on an RTX 4070 Ti, September 14–15, 2026. A high FPS
counter or visible menu was never sufficient evidence of the actual FG multiplier.

| Stage | Observation | Retained result |
| --- | --- | --- |
| Native game | Normal unmodified 2x loaded | Baseline |
| NeuralScreen desktop capture | Perceived lag despite high FPS; lowest settings still reported about 45 FPS | Not part of the game installer; in-engine NR investigated instead |
| RTXMFG presets and DLL names | Trials included version, dxgi, d3d12, winmm, dinput8 and xinput1_3; menus varied, higher requests were rejected or hung | Do not repeat filename roulette; final MFG loader is winmm.dll |
| Driver SR/FG overrides | Removing forced counts, SR overrides and preset tests did not alone solve higher FG | Only the final documented two FG controls are installed |
| RTXMFG v1.3.2 menu | Menu attached and exposed debug route information, but available maximum remained 2x | Menu attachment and provider capacity are separate |
| GPU display-name spoof | RTX 5080 aliases exposed native multiplier choices | Optional, backed up, requires restart |
| Native Inspector-style override | Higher native override alone GPU-hung on this PC | Not presented as a working standalone solution |
| RTXMFG v1.3.3 startup analysis | Wrapper capacity preparation occurred too late for WuWa's cached limit | Narrow exact-build patch; source equivalent and offline byte tests published |
| Working fixed 3x | Accepted/applied and three frames presented; normal picture/smooth motion | Runtime-confirmed |
| Follow game | Fresh process after removing fixed driver count reported actual 6x; user tested 4/5/6 | Game owns supported multiplier; 4/5 remain user-reported |
| OptiScaler NR v0.8.3 | NR ran, but selected 6x fell back to actual 2x, user reported 42 FPS | Not shipped as compatible |
| OptiScaler v0.7.7 external FG | GPU hang; invalid GPU write with no allocation/shader attribution | Retired; old heuristic lifetime concerns were not proof of cause |
| Stock NR v0.8.4, NR off | Streamline hook/detach errors and GPU hang | Build-specific external-FG compatibility required |
| Compatibility revision 1 | CPU crash referenced an unloaded Streamline common module | Module lifetime fix in revision 2 |
| Revision 2, NR off | Extended run and actual 6x samples | External-FG baseline improved |
| Revision 2, NR on | NR at 1281×721 and actual 6x; later transition MMU write fault | Crash analyzed offline; attribution limits documented |
| Revision 3 | Reproduced and guarded reuse of live descriptor/constant slots; first game run with NR and actual 6x | Guard included, further transition testing needed |
| Output-resolution revision | Model size follows output, rather than pre-SR input; 100%/75%/50% preserve output proportions | Full 3840×2160 model and actual 6x observed; roughly 21–24 ms NR intervals, user responsiveness/visual assessment pending |

## Where the work lives

| Artifact | Published location |
| --- | --- |
| Exact RTXMFG edit, pins, profile semantics | [Technical analysis](technical.md), `wuwa_mfg/patch.py`, `patches/early-wrapper-preparation.patch` |
| Guided MFG setup, rollback and diagnostics | `Setup.cmd`, `Launch.ps1`, `wuwa_mfg/` |
| Original successful MFG evidence | [Validation](validation.md), `evidence/session-summary.json` |
| All retained OptiScaler compatibility/source changes | `patches/optiscaler-wuwa-compat.patch` against the pinned source |
| Neural source build and local bundle creation | `BuildNeural.ps1`, `tools/make_neural_bundle.py` |
| Neural profile, optional install/disable/scale/restore | `neural/OptiScaler.ini`, `wuwa_mfg/neural.py`, `wuwa_mfg/neural_cli.py` |
| Crash diagnosis, proof and limits | [Crash investigation](neural-crashes.md) |
| Reproduction/tests | `tests/` and regression sources inside the OptiScaler patch |
| Attribution/licensing | `THIRD_PARTY_NOTICES.md`, `licenses/` |

Transient experiments are described rather than distributed as a collection of
obsolete injectors. Proprietary game/NVIDIA binaries, raw dumps, machine identifiers,
local screenshots and private recovery records stay out of the public repository.
All retained custom code needed to reproduce the published setup is included or
expressed as a patch against an exact public upstream commit.
