# MoEBoost Workshop Architecture: Workshop Manager, People Busyness Catalog & Heterogeneous Workstations

```text
                  ┌────────────────────────────────────────────────────────┐
                  │              PEOPLE BUSYNESS CATALOG                   │
                  │                                                        │
                  │  [Hot People]        ◄──►      [Warm People]           │
                  │  (Pinned in RAM Shelf)         (Standby in RAM)        │
                  │         ▲                             ▲                │
                  │         │                             │ (Promote /     │
                  │         ▼                             ▼  Demote)       │
                  │                 [Cold People]                          │
                  │                 (Archived on SSD Storage Shelf)        │
                  └────────────────────────────────────────────────────────┘
                                               │
                                 Token Task Arrival
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │     THE WORKSHOP MANAGER        │
                              │     (`WorkshopManager`)         │
                              │                                 │
                              │ Inspects Slot Capacity, Queue,  │
                              │ and Busyness Catalog to order   │
                              │ EXACTLY where the expert works  │
                              └────────────────┬────────────────┘
                                               │
                 ┌─────────────────────────────┴─────────────────────────────┐
                 ▼                                                           ▼
   ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
   │ HIGH-TECH TOOL WORKSTATION (VRAM)       │   │ HAND-LABOR WORKSTATION (RAM)            │
   │                                         │   │                                         │
   │ * Manager assigns VIP/Hot or free slots │   │ * Manager diverts here during tool rush │
   │ * Specialized Power Tools (CUDA Cores)  │   │ * Manual Labor by Hand (CPU AVX / MKL)  │
   │ * Super Fast, but LIMITED SLOTS         │   │ * Slower per task, but MANY SLOTS       │
   │ * People queue if slots are busy        │   │ * Zero waiting line! Overflow spillway  │
   │ * Steps out of slot when done (transient│   │ * Eliminates PCIe equipment-moving lag  │
   └─────────────────────────────────────────┘   └─────────────────────────────────────────┘
```

### Roles and Responsibilities

1. **The Workshop Manager (`WorkshopManager`)**:
   - The central dispatch supervisor. The expert never guesses where to work; the **Manager issues a concrete directive** (`ManagerDirective`):
     - `"Manager: You are a HOT expert with high busyness. Slot granted at Tool Workstation."`
     - `"Manager: Tool Workstation is full/busy. Don't wait in line; do the job by hand in RAM!"`
     - Orders cold expert weights brought from the SSD storage shelf into RAM before work begins (`requires_ssd_to_ram_staging=True`).

2. **SSD Is Pure Storage Only (Never a Workstation)**:
   - The SSD is an archival cabinet/shelf in the back room. No computation or "working" ever happens on the SSD.
   - For a cold expert stored on SSD to do work, its weights **MUST first be brought into RAM**.

3. **The Two Computing Workstations**:
   - **High-Tech Tool Workstation (VRAM / GPU)**: Ultra-fast specialized power tools, limited slots (`max_vram_slots`).
   - **Hand-Labor Workstation (RAM / CPU)**: Manual labor by hand, slower per token, but **many slots** (parallel CPU threads with zero waiting line).

### Dynamic In-vs-Go Auto-Adjustment (1:1 up to 50:50)

The Workshop Manager automatically adjusts the in-vs-go eviction ratio in real-time based on active queue pressure:
- **Depth <= 1 (Serial Token Stream)**: **1:1 Strict Ratio** (1 out / 1 in). Retains 100% of the active hot set with zero premature displacement.
- **Depth <= 4 (Mild Token Bursts)**: **10:90 Ratio** (10% breathing buffer). Eliminates slot thrashing.
- **Depth <= 8 (Moderate Surge)**: **25:75 Ratio** (25% chunk buffer).
- **Depth > 8 (Heavy Surges / Topic Switch)**: **50:50 Ratio** (Half-table flush). Purges stale context rapidly so incoming waves sit immediately without thread stalls.

2. **High-Tech Tool Workstation (`WorkstationType.TOOL_WORKSTATION_VRAM`)**:
   - **The Tools**: Dedicated Tensor Cores, CUDA kernels, and GDDR5/GDDR6 VRAM.
   - **The Bottleneck**: Limited physical slots (bounded VRAM).
   - **Behavior**: Fast execution. When transient mode is enabled, the person finishes reading/computing and immediately steps out of the tool slot (auto-discard/demote), leaving room for the next person.

3. **Hand-Labor Workstation (`WorkstationType.HAND_WORKSTATION_RAM`)**:
   - **The Process**: When the Tool Workstation has a long line or slots are full, instead of waiting in line or spending valuable time hauling heavy equipment across the PCIe bus, the person does the calculation **by hand in RAM** using multi-threaded CPU workers.
   - **The Advantage**: Slower per token calculation than a GPU core, but with virtually unlimited chairs (multi-core host parallelism), completely eliminating GPU pipeline stalls!

## Promotion and demotion

- RAM experts promote to VRAM after `MOE_VRAM_PROMOTE_HITS` observations.
- A VRAM-resident expert keeps its RAM backing. RAM eviction skips those experts and evicts eligible cold/low-frequency RAM entries first.
- Hot SSD experts can stage directly to VRAM through a temporary host buffer when the 20%+ RAM reserve leaves enough temporary transfer headroom.
- If a VRAM expert without a RAM copy is displaced, a demotion callback first tries to evict cold/low-use RAM entries, then reads the expert into RAM. If safe RAM capacity is genuinely insufficient, it remains SSD-backed and is reported as `SSD_overflow`.
- Active VRAM leases and RAM entries are pinned. A per-expert cooldown prevents immediate promotion/eviction loops.
- Prefetch and promotion are bounded by live RAM/VRAM headroom; failure degrades to SSD paging rather than breaking the reserve.

## Important backend limitation

The current llama.cpp integration does not expose expert-level per-token tensor routing hooks. The cache and API can stage and report expert residency, but llama.cpp chat generation does not yet consume these staged tensors. This update does not claim a live inference speedup. Actual acceleration requires an expert-aware inference backend adapter.

## Validation

The package test suite covers direct SSD-to-VRAM staging, VRAM-to-RAM demotion with cold RAM eviction, hot RAM backing protection, active VRAM pinning, cooldown, headroom enforcement, VRAM execution queueing, and direct RAM spillway fallback.
