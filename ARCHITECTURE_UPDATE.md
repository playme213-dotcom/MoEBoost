# MoEBoost: Unlimited Shared Queue & Dynamic Spillway Architecture

## 1. Executive Architecture Summary
MoEBoost implements an on-demand, hierarchical expert scheduler for mixture-of-experts (MoE) LLMs that separates storage from execution workstations:
- **Archival Tier (NVMe SSD)**: 0% compute. Acts purely as immutable weight storage.
- **High-Tech Tool Workstation (VRAM)**: Dedicated fast execution engine utilizing CUDA device buffers for hot experts.
- **Hand-Labor Spillway Workstation (RAM)**: Dynamic spillway capacity capped at **50% of host RAM** (reserving 35% for OS headroom).
- **Unlimited Shared Queue Line**: A single unified line feeds both workstations with zero artificial queue limits, eliminating head-of-line blocking.

```
                           [ Inbound Expert Tasks ]
                                       │
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │     UNLIMITED UNIFIED SHARED QUEUE (Zero Artificial Limit)    │
       │            [Job #4] ➔ [Job #3] ➔ [Job #2] ➔ [Job #1]          │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                      [ Dynamic Work-Stealing Pull ]
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
   [ VRAM Tool Workstation ]                     [ RAM Spillway Workstation ]
   • Primary Consumer                            • Overflow Spillway Consumer
   • Fast hardware power tools                   • 4,906 Chairs (50% RAM Reserve)
   • Pulls 1st whenever chairs free              • Pulls immediately when VRAM is busy
```

---

## 2. Workstation Invariants & Math

### Workstation Chairs
- **VRAM Chairs**:
  $$\text{VRAM Chairs} = \frac{\text{Allocatable VRAM (80\% of 3,072 MB)}}{\text{Average Expert Size (2.5 MB)}} \approx \mathbf{443\text{ chairs}}$$
- **RAM Chairs (50% Backup Rule)**:
  $$\text{RAM Chairs} = \frac{50\% \text{ System RAM (11.45 GB)}}{\text{Average Expert Size (2.5 MB)}} = \mathbf{4,906\text{ chairs}}$$

### Queue Discipline
- **Unified Deque**: Shared across both workstations. No artificial caps (`None`).
- **Llama.cpp Offload**: Strict `gpu_layers = 0` invariant to prevent monolithic VRAM exhaustion.
- **Host Reserve**: Minimum 35% RAM headroom enforced at all times.

---

## 3. Telemetry & Visualizer Deck
- **Monitor View**: Real-time KPI cards tracking VRAM Chairs (In / Max), VRAM Stepped Out (Go), Unified Shared Queue (Line), RAM Spillway Drainage, and physical memory.
- **Neural Workstation Tab**: Live synaptic canvas visualizer with animated flow particles showing expert dispatch from SSD ➔ Manager ➔ Workstation chairs.
