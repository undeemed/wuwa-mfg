# Neural crash investigation and compatibility patch

These findings come from one RTX 4070 Ti / driver 616.92 WuWa installation on
September 15, 2026. Separate a demonstrated code defect from a plausible explanation
of a game crash. The published source contains both the fixes and regression tests;
raw crash dumps, registry exports and personal logs are not public artifacts.

## CPU crash: stale Streamline DLL reference

The first external-FG compatibility build crashed reading an address inside a
previously unloaded `sl.common` image. The read address was at module-relative
offset `0xC1560`; the failing instruction was in the NVIDIA `sl.dlss_g` wrapper at
offset `0x4F685`. The unloaded image's size, checksum and timestamp matched the
Streamline common module. This supports a stale module reference, rather than
treating every failure as a generic GPU hang.

Revision 2 retained the real common DLL for the external-FG path using the existing
library-lifetime hook. It did not add an anti-cheat bypass or new injection mechanism.
A following NR-off run continued for roughly 55 minutes; its sampled active states
reported actual 6x. Retention trades a bounded loaded module for avoiding premature
unload. It is not proof that every provider lifetime is correct.

## GPU crash: what the dump establishes

With revision 2 and NR on, the model ran at **1281 × 721**, one pre-SR pass. Recent
GPU intervals were about 3.90–3.91 ms total and 3.35–3.37 ms in the model. RTXMFG
accepted 6x and reported six frames presented during gameplay.

The later Aftermath dump reported `Error_DMA_PageFault`, an invalid GPU **write**
at virtual address `0x33B96C000`, MMU fault and graphics-engine reset. The tester
identified opening settings/changing graphics/switching windows as the trigger
category, and the crash context recorded an unfocused game. The dump contained no
shader PC or resource mapping identifying the offending allocation. Its final
FG-off state does not invalidate the preceding FG-active 6x samples.

**Identified:** the fault type, GPU write address, transition context and active
graphics stack. **Not identified from that dump:** the exact shader/resource that
wrote the invalid address. The descriptor defect below is a concrete candidate;
it is not a proven one-to-one attribution of this crash.

## Reproduced defect: live descriptor/constant slots reused

The v0.8.4 NR shader helper allocated descriptor tables and upload constants from
96 slots and advanced the index modulo 96. It did not wait for ownership of that
slot to end. With outstanding command recordings/submissions, the next wrap could
overwrite data still referenced by GPU work. A settings/resolution transition or
GPU backlog can expose this class of lifetime bug.

Revision 3 replaces that ring with `DispatchSlots<96>`. The existing GPU-lifetime
tracker marks a slot reusable only after recorded uses are no longer replayable
and their submitted GPU fences have completed. Completion is signalled through an
atomic ticket owned by the slot allocation. No new queue wait or interception is
introduced. If all slots remain busy, the dispatch is skipped and logged instead
of overwriting live bindings. Encode failure propagates before dependent reads,
restoring the colour resource state and keeping the original game colour.

The isolated WARP regression exercises production lifetime tracking:

| Scenario | Result |
| --- | --- |
| Old unchecked wrapping allocator, 97th allocation | Reused a live slot: negative control fails as expected |
| Guarded allocator, 97th allocation | Refused; existing bindings retained |
| Unsubmitted recording, repeated collection | Slot remains owned |
| Submitted command list reset while queue is GPU-blocked | Slot remains owned until fence completion |
| Completed but replayable command list | Slot remains owned until recording reset/destruction |
| Completed and reset recording | Slots become reusable |

Both the guarded regression and the existing upstream GPU-lifetime smoke test
passed. An initial revision-3 game run created the NR model, reported recent
roughly 4 ms NR intervals and actual 6x, and continued without a new crash during
observation. A later 20-second capture had FG off and is **not** counted as 6x
validation. The tester has not yet confirmed repetition of the exact crash trigger.

The guard fixes the reproduced slot reuse. It cannot by itself rule out other
resource-state, model, driver, or transition bugs. NR remains experimental.

## What the complete source patch includes

1. Opt-in `[FrameGen] External=true` compatibility: let RTXMFG own external FG;
   avoid duplicate Streamline/FG interception and inappropriate NVAPI shutdown.
2. Preserve the loaded Streamline common module for the external-FG lifetime.
3. Ensure NR's direct compatibility runtime remains reachable with external FG.
4. GPU-aware ownership of NR descriptor/upload-constant slots and safe skip propagation.
5. DX12 output-relative model sizing, config persistence, a menu control and extent tests.

The patch includes tests for module lifetime, GPU dispatch slots and model extents.
Use [BuildNeural.ps1](../BuildNeural.ps1) to apply it to the exact upstream commit.
The output-size tests cover 4K, ultrawide, DLSS input changes, invalid sizes/scales
and uniform texture-limit handling. They verify dimensions, not model quality or
performance at those dimensions.

## Practical prevention while testing

Use the patched build and keep one neural pass. Set graphics/output resolution
before enabling NR; disable NR and restart before further graphics experiments if
the toggle/menu is unreliable. Compare repeatable transitions with NR disabled,
keeping the MFG setup unchanged. Lower the model's output scale if inference cost
causes excessive delay. These reduce variables; they are not a guarantee against
device-hung errors. No TDR-delay tweak, anti-cheat disable, driver patch or additional
proxy chain is part of this fix.
