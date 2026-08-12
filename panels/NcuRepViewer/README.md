### ncu-rep Viewer

This Python Panel is a viewer for NVIDIA Nsight Compute profiler reports (.ncu-rep files) that teams have logged as experiment assets. It turns those raw GPU-kernel profiles into an interactive, Nsight-style dashboard right inside the Comet UI.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/NcuRepViewer/ncu-rep-viewer.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


```python
from comet_ml import start
          
experiment = start(
  api_key="YOUR-API-KEY",
  project_name="ncu-rep-viewer",
  workspace="YOUR-WORKSPACE"
)

# examples:
experiment.log_asset("manual_nvtx.ncu-rep")
experiment.log_asset("CuVectorAddDrv.ncu-rep")
experiment.log_asset("mergeSort.ncu-rep")

experiment.end()
```

The flow:

1. Discovers reports — scans the experiments currently selected in your Comet project and collects every .ncu-rep asset attached to them.
2. Downloads on demand — when you pick a report from the sidebar dropdown, it streams that asset from Comet to local disk (cached, so re-selecting is instant) and extracts the per-kernel profiling data.
3. Lets you drill in — you choose a specific kernel launch and can filter its metrics.

What it presents, across three tabs:

- Summary — a table of all kernels in the report (duration, grid/block size, memory & compute throughput, occupancy, detected issues), plus a comparison chart when there's more than one kernel.
- Details — the Nsight-style analysis sections for the selected kernel:
  - a device header (GPU name, compute capability, SM count, memory, cache, clocks),
  - Speed of Light (compute vs memory throughput, bottleneck call),
  - Roofline (achieved performance vs arithmetic intensity),
  - Compute and Memory workload analysis, plus detailed per-unit memory tables (L1/TEX, L2, DRAM, shared),
  - scheduler / warp-stall, instruction, launch, and occupancy statistics,
  - NVTX ranges active at launch, and
  - analysis & recommendations surfaced from Nsight's own rules.
- Raw Metrics — the complete searchable metric table for the kernel.

A design principle worth noting: the richer sections (Roofline, memory tables) need reports captured with a fuller metric set. When a given report wasn't profiled that deeply, those sections show a short "needs --set full" note instead of failing — so the panel works gracefully on both lightweight and detailed captures.
