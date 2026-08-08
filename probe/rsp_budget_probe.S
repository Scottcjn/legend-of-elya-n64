## Minimal rspq overlay, used ONLY to measure the DMEM/IMEM budget an overlay
## actually gets once rspq's own resident state is accounted for.
#include <rsp_queue.inc>

    .data
    RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand BudgetProbeCmd, 4
    RSPQ_EndOverlayHeader

    RSPQ_BeginSavedState
probe_state:  .long 0
    RSPQ_EndSavedState

    .text
BudgetProbeCmd:
    jr ra
    nop
