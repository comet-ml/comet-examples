%pip install ncu-report
%pip install streamlit-ncu-rep-viewer==0.1.1

from comet_ml import API
import os
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(layout="wide")

api = API()

# Where downloaded .ncu-rep assets are cached on the local (panel) filesystem.
ASSET_DIR = Path(tempfile.gettempdir()) / "ncu_rep_assets"


@st.cache_data(persist="disk", show_spinner="Downloading report…")
def download_asset(experiment_id, asset_id, file_name, chunk_size=1024 * 1024):
    """Stream an asset from Comet to a local file and return its path.

    Comet's Python API has no direct download-to-path for generic assets
    (only download_model / download_tensorflow_folder), so we fetch the asset
    ourselves — ncu-report needs a real file on disk. We stream it in chunks
    (return_type="response", stream=True) to avoid holding a large report fully
    in memory. Writes to a temp file first so a failed download never leaves a
    truncated .ncu-rep behind for the cache to reuse.
    """
    experiment = experiment_map[experiment_id]
    response = experiment.get_asset(asset_id, return_type="response", stream=True)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / f"{asset_id}_{file_name}"
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fd.write(chunk)
    tmp.replace(path)
    return str(path)


def get_asset_list(_experiment, experiment_id, asset_type):
    return _experiment.get_asset_list() #asset_type=asset_type)

def get_all_rep_data(_experiments, experiment_ids):
    data = set()
    # First, get a selection from asset names:
    bar = st.progress(0, "Loading ncu-rep list...")
    for i, experiment in enumerate(_experiments):
        for asset in get_asset_list(experiment, experiment.id, None):
            bar.progress(i/len(_experiments), "Loading ncu-rep list...")
            if asset["fileName"].endswith(".ncu-rep"):
                data.add((experiment.id, experiment.name, asset["fileName"], asset["assetId"], asset["step"], ))
    bar.empty()
    return data


@st.cache_data(show_spinner="Loading report…")
def load_kernels(path):
    result = subprocess.run(
        [sys.executable, "-m", "streamlit_ncu_rep_viewer.extract", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_val(kernel, name, default=None):
    entry = kernel["metrics"].get(name)
    return entry["value"] if entry else default


def strip_markup(text):
    text = re.sub(r"@url:([^:]+):[^@]+@", r"\1", text)
    text = re.sub(r"@section:[^:]+:([^@]+)@", r"\1", text)
    return text


def pretty(name):
    """Turn a metric leaf like 'sm__pipe_alu_cycles_active' into 'Pipe Alu'."""
    core = name.split("__", 1)[-1].split(".", 1)[0]
    core = re.sub(r"_cycles_active$|_active$|_pct_.*$", "", core)
    return core.replace("_", " ").title()


def nvtx_ranges(kernel):
    """Active NVTX ranges for a kernel, flattened outer→inner across domains."""
    out = []
    for dom in kernel.get("nvtx", []):
        out.extend(dom.get("push_pop", []))
        out.extend(dom.get("start_end", []))
    return out


def nvtx_breadcrumb(kernel):
    """Compact 'outer › inner' breadcrumb of active NVTX ranges."""
    return " › ".join(nvtx_ranges(kernel))


def hbar(labels, values, unit="%", color="#636efa", height=None, xmax=None,
         highlight_max=False):
    """Horizontal bar chart, sorted descending, Nsight-style."""
    pairs = sorted(zip(labels, values), key=lambda p: p[1])
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = color
    if highlight_max and values:
        top = max(values)
        colors = ["#ef553b" if v == top else color for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1f}{unit}" for v in values], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=height or max(180, len(labels) * 26 + 60),
        xaxis=dict(title=f"% of Peak" if unit == "%" else "", range=[0, xmax or (max(values) * 1.15 if values else 1)]),
        margin=dict(t=10, b=30, l=10, r=40),
    )
    return fig


# ── device / session header ─────────────────────────────────────────────────

def _dev(kernel, metric):
    entry = kernel["metrics"].get(metric)
    return entry["value"] if entry else None


def _fmt_bytes(v):
    if not isinstance(v, (int, float)) or v <= 0:
        return None
    n = float(v)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024


def _fmt_int(v):
    return f"{int(v):,}" if isinstance(v, (int, float)) else None


def _fmt_bits(v):
    return f"{int(v)}-bit" if isinstance(v, (int, float)) else None


def _fmt_khz_ghz(v):
    return f"{v / 1e6:.3f} GHz" if isinstance(v, (int, float)) and v else None


def _fmt_hz_ghz(v):
    return f"{v / 1e9:.3f} GHz" if isinstance(v, (int, float)) and v else None


def section_device_header(kernel):
    """Report-wide GPU/session summary line plus an expandable attribute panel.

    The device__attribute_* metrics are present in every capture (default set
    included), so this renders whenever the report carries device info.
    """
    name = _dev(kernel, "device__attribute_display_name")
    cc_maj = _dev(kernel, "device__attribute_compute_capability_major")
    cc_min = _dev(kernel, "device__attribute_compute_capability_minor")
    cc = (f"{int(cc_maj)}.{int(cc_min)}"
          if isinstance(cc_maj, (int, float)) and isinstance(cc_min, (int, float)) else None)
    sm = _dev(kernel, "device__attribute_multiprocessor_count")
    total_mem = _dev(kernel, "device__attribute_total_memory")

    if name is None and cc is None:
        return  # no device info in this report

    bits = []
    if name:
        bits.append(f"**{name}**")
    if cc:
        bits.append(f"CC {cc}")
    if isinstance(sm, (int, float)):
        bits.append(f"{int(sm)} SMs")
    if _fmt_bytes(total_mem):
        bits.append(_fmt_bytes(total_mem))
    st.caption("🖥️  " + "  ·  ".join(bits))

    compute = [
        ("Compute Capability", cc),
        ("SM Count", _fmt_int(sm)),
        ("Warp Size", _fmt_int(_dev(kernel, "device__attribute_warp_size"))),
        ("Max Warps / SM", _fmt_int(_dev(kernel, "device__attribute_max_warps_per_multiprocessor"))),
        ("Max Threads / SM", _fmt_int(_dev(kernel, "device__attribute_max_threads_per_multiprocessor"))),
        ("Max Blocks / SM", _fmt_int(_dev(kernel, "device__attribute_max_blocks_per_multiprocessor"))),
        ("Schedulers / SM", _fmt_int(_dev(kernel, "device__attribute_num_schedulers_per_multiprocessor"))),
        ("Registers / SM", _fmt_int(_dev(kernel, "device__attribute_max_registers_per_multiprocessor"))),
        ("Shared Mem / SM", _fmt_bytes(_dev(kernel, "device__attribute_max_shared_memory_per_multiprocessor"))),
    ]
    memclk = [
        ("Device Memory", _fmt_bytes(total_mem)),
        ("L2 Cache", _fmt_bytes(_dev(kernel, "device__attribute_l2_cache_size"))),
        ("Memory Bus Width", _fmt_bits(_dev(kernel, "device__attribute_global_memory_bus_width"))),
        ("Max SM Clock", _fmt_khz_ghz(_dev(kernel, "device__attribute_max_gpu_frequency_khz"))),
        ("Max Memory Clock", _fmt_khz_ghz(_dev(kernel, "device__attribute_max_mem_frequency_khz"))),
        ("Achieved SM Clock", _fmt_hz_ghz(_dev(kernel, "gpc__cycles_elapsed.avg.per_second"))),
        ("Achieved Memory Clock", _fmt_hz_ghz(_dev(kernel, "dram__cycles_elapsed.avg.per_second"))),
    ]

    with st.expander("Device Attributes", expanded=False):
        cols = st.columns(2)
        for col, header, spec in ((cols[0], "Compute", compute),
                                  (cols[1], "Memory & Clocks", memclk)):
            rows = [{"Attribute": lbl, "Value": val} for lbl, val in spec if val is not None]
            if rows:
                col.markdown(f"**{header}**")
                col.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── pages ─────────────────────────────────────────────────────────────────────

def page_summary(kernels):
    rows = []
    for kernel in kernels:
        duration_us = (get_val(kernel, "gpu__time_duration.sum", 0) or 0) / 1000
        rows.append({
            "Kernel": kernel["name"],
            "Duration (µs)": round(duration_us, 3),
            "Grid": int(get_val(kernel, "launch__grid_size", 0) or 0),
            "Block": int(get_val(kernel, "launch__block_size", 0) or 0),
            "Waves/SM": round(get_val(kernel, "launch__waves_per_multiprocessor", 0) or 0, 2),
            "Memory Throughput (%)": round(get_val(kernel, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 0) or 0, 1),
            "SM Throughput (%)": round(get_val(kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed", 0) or 0, 1),
            "Occupancy (%)": round(get_val(kernel, "sm__warps_active.avg.pct_of_peak_sustained_active", 0) or 0, 1),
            "NVTX": nvtx_breadcrumb(kernel) or "—",
            "Issues": ", ".join(r["name"] for r in kernel["rules"]) or "—",
        })

    # Drop the NVTX column entirely if no kernel has NVTX data.
    if all(r["NVTX"] == "—" for r in rows):
        for r in rows:
            del r["NVTX"]

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if len(rows) > 1:
        names = [r["Kernel"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Memory Throughput (%)", x=names, y=[r["Memory Throughput (%)"] for r in rows]))
        fig.add_trace(go.Bar(name="SM Throughput (%)", x=names, y=[r["SM Throughput (%)"] for r in rows]))
        fig.add_trace(go.Bar(name="Occupancy (%)", x=names, y=[r["Occupancy (%)"] for r in rows]))
        fig.update_layout(barmode="group", xaxis_tickangle=-30, height=280,
                          yaxis_title="% of Peak", legend_title="Metric",
                          margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")


def section_speed_of_light(kernel):
    with st.expander("GPU Speed Of Light Throughput", expanded=True):
        sm = get_val(kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed", 0) or 0
        mem = get_val(kernel, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 0) or 0
        duration_us = (get_val(kernel, "gpu__time_duration.sum", 0) or 0) / 1000
        cycles = get_val(kernel, "gpc__cycles_elapsed.max", 0) or 0
        sm_hz = get_val(kernel, "gpc__cycles_elapsed.avg.per_second", 0) or 0
        dram_hz = get_val(kernel, "dram__cycles_elapsed.avg.per_second", 0) or 0

        cols = st.columns(2)
        cols[0].metric("Compute (SM) Throughput", f"{sm:.1f} %")
        cols[1].metric("Memory Throughput", f"{mem:.1f} %")

        fig = go.Figure(go.Bar(
            x=[sm, mem], y=["Compute (SM)", "Memory"], orientation="h",
            marker_color=["#ef553b" if sm >= mem else "#636efa",
                          "#ef553b" if mem > sm else "#636efa"],
            text=[f"{sm:.1f}%", f"{mem:.1f}%"], textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(height=160, xaxis=dict(range=[0, 110], title="% of Peak"),
                          yaxis=dict(autorange="reversed"), margin=dict(t=10, b=30, l=10, r=40))
        st.plotly_chart(fig, width="stretch")

        cols = st.columns(4)
        cols[0].metric("Duration (µs)", round(duration_us, 3))
        cols[1].metric("Elapsed Cycles", f"{int(cycles):,}")
        cols[2].metric("SM Frequency (GHz)", round(sm_hz / 1e9, 2))
        cols[3].metric("DRAM Frequency (GHz)", round(dram_hz / 1e9, 2))

        bottleneck = "Memory-bound" if mem > sm else "Compute-bound"
        st.caption(f"Higher of the two throughputs dominates → **{bottleneck}** "
                   f"(SM {sm:.1f}% vs Memory {mem:.1f}%).")


# Floating-point instruction groups → FLOPs per thread-instruction. FMA counts
# as 2 (a multiply and an add).
_FP_OPS = {
    "FP32": {"fadd": 1, "fmul": 1, "ffma": 2},
    "FP64": {"dadd": 1, "dmul": 1, "dfma": 2},
    "FP16": {"hadd": 1, "hmul": 1, "hfma": 2},
}
# Metric whose % of peak approximates each precision's compute ceiling, used to
# estimate peak FLOP/s when the report has no explicit peak metric.
_FP_PIPE_UTIL = {
    "FP32": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "FP64": "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "FP16": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
}


def section_roofline(kernel):
    """Floating-point roofline: achieved performance vs arithmetic intensity.

    Needs a report captured with FP instruction counts (the Roofline or full
    metric set). Ceilings are derived from the achieved value and its % of peak
    (peak DRAM bandwidth) and from FP-pipe utilization (compute peak) — an
    estimate used when the report carries no explicit peak metrics.
    """
    duration_s = (get_val(kernel, "gpu__time_duration.sum", 0) or 0) / 1e9
    dram_bytes = get_val(kernel, "dram__bytes.sum")

    flops = {}
    for prec, ops in _FP_OPS.items():
        total, found = 0.0, False
        for op, weight in ops.items():
            v = get_val(kernel, f"sm__sass_thread_inst_executed_op_{op}_pred_on.sum")
            if isinstance(v, (int, float)):
                total += weight * v
                found = True
        if found and total > 0:
            flops[prec] = total

    with st.expander("Roofline", expanded=False):
        if not flops:
            st.info("Roofline needs FP instruction counts — profile with the "
                    "Roofline or full metric set (`--set full`).")
            return
        if not (duration_s > 0 and isinstance(dram_bytes, (int, float)) and dram_bytes > 0):
            st.info("Roofline needs kernel duration and DRAM bytes, "
                    "which are not present in this report.")
            return

        # Peak DRAM bandwidth from achieved bytes/s and its % of peak.
        achieved_bw = dram_bytes / duration_s
        dram_pct = get_val(kernel, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
        peak_bw = (achieved_bw / (dram_pct / 100)
                   if isinstance(dram_pct, (int, float)) and dram_pct > 0 else None)

        # Per-precision achieved point (arithmetic intensity, FLOP/s) and an
        # estimated compute ceiling from FP-pipe utilization.
        points, peak_compute = {}, {}
        for prec, f in flops.items():
            points[prec] = (f / dram_bytes, f / duration_s)  # (FLOP/byte, FLOP/s)
            util = get_val(kernel, _FP_PIPE_UTIL[prec])
            if isinstance(util, (int, float)) and util > 0:
                peak_compute[prec] = (f / duration_s) / (util / 100)

        dominant = max(flops, key=flops.get)
        ceil = peak_compute.get(dominant)
        colors = {"FP32": "#636efa", "FP64": "#ef553b", "FP16": "#00cc96"}

        ais = [p[0] for p in points.values()]
        xmin, xmax = min(ais) / 10, max(ais) * 10
        n = 60
        xs = [xmin * (xmax / xmin) ** (i / (n - 1)) for i in range(n)]

        fig = go.Figure()
        if peak_bw and ceil:
            ys = [min(ceil, peak_bw * x) / 1e9 for x in xs]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     name=f"Roofline ({dominant})",
                                     line=dict(color="#888", width=2)))
        elif peak_bw:
            fig.add_trace(go.Scatter(x=xs, y=[peak_bw * x / 1e9 for x in xs],
                                     mode="lines", name="Memory ceiling",
                                     line=dict(color="#888", width=2)))
        elif ceil:
            fig.add_trace(go.Scatter(x=[xmin, xmax], y=[ceil / 1e9, ceil / 1e9],
                                     mode="lines", name=f"Compute peak ({dominant})",
                                     line=dict(color="#888", width=2, dash="dash")))

        for prec, (ai, perf) in points.items():
            fig.add_trace(go.Scatter(
                x=[ai], y=[perf / 1e9], mode="markers+text", name=prec,
                marker=dict(size=12, color=colors.get(prec, "#ab63fa")),
                text=[prec], textposition="top center",
            ))

        fig.update_layout(
            height=380,
            xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log"),
            yaxis=dict(title="Performance (GFLOP/s)", type="log"),
            margin=dict(t=20, b=40, l=10, r=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, width="stretch")

        ai_d, perf_d = points[dominant]
        line = f"**{dominant}**: {perf_d / 1e9:,.1f} GFLOP/s at {ai_d:.3f} FLOP/byte."
        if peak_bw:
            line += f"  Peak DRAM BW ≈ {peak_bw / 1e9:,.0f} GB/s."
            if ceil:
                bound = "memory-bound" if ai_d < ceil / peak_bw else "compute-bound"
                line += f"  Ridge at {ceil / peak_bw:.2f} FLOP/byte → **{bound}**."
        st.caption(line + "  Peaks are estimated from % of peak and FP-pipe "
                   "utilization when the report has no explicit peak metrics.")


def _breakdown_contributors(kernel, top_metric, n=8):
    """Return (labels, values) of the top-n sub-metrics of a breakdown: metric."""
    entry = kernel["metrics"].get(f"breakdown:{top_metric}")
    if not entry:
        return None
    subs = str(entry["value"]).split(",")
    rows = []
    for sub in subs:
        v = get_val(kernel, sub)
        if isinstance(v, (int, float)) and v > 0:
            rows.append((pretty(sub), v))
    rows.sort(key=lambda p: -p[1])
    rows = rows[:n]
    return ([r[0] for r in rows], [r[1] for r in rows]) if rows else None


def section_compute(kernel):
    with st.expander("Compute Workload Analysis", expanded=False):
        pipes = {}
        for name, entry in kernel["metrics"].items():
            if (name.startswith("sm__pipe_") or name.startswith("sm__inst_executed_pipe_")) \
                    and name.endswith(".avg.pct_of_peak_sustained_elapsed"):
                v = entry["value"]
                if isinstance(v, (int, float)):
                    pipes[pretty(name)] = v
        if not pipes:
            st.info("No pipeline utilization metrics in this report.")
            return
        st.plotly_chart(hbar(list(pipes.keys()), list(pipes.values()),
                             highlight_max=True, xmax=110), width="stretch")
        top = max(pipes, key=pipes.get)
        st.caption(f"Busiest pipeline: **{top}** at {pipes[top]:.1f}% of peak (red).")


def section_memory(kernel):
    with st.expander("Memory Workload Analysis", expanded=False):
        levels = {
            "L1/TEX Cache": get_val(kernel, "l1tex__throughput.avg.pct_of_peak_sustained_active"),
            "L2 Cache": get_val(kernel, "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
            "DRAM": get_val(kernel, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        }
        levels = {k: v for k, v in levels.items() if isinstance(v, (int, float))}
        if levels:
            fig = go.Figure(go.Bar(
                x=list(levels.keys()), y=list(levels.values()), marker_color="#00cc96",
                text=[f"{v:.1f}%" for v in levels.values()], textposition="outside",
            ))
            fig.update_layout(yaxis=dict(title="% of Peak", range=[0, 110]),
                              height=260, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")

        contrib = _breakdown_contributors(kernel, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
        if contrib:
            st.markdown("**Top memory-unit contributors**")
            st.plotly_chart(hbar(*contrib, highlight_max=True, xmax=110),
                            width="stretch")


# Per-unit memory tables, mirroring Nsight's Memory Workload Analysis tables.
# Each row is (label, metric name); rows whose metric is absent are dropped, so
# a default-set report shows just the throughput lines and a full capture fills
# in requests / sectors / hit rates / bytes.
_MEM_TABLES = {
    "L1 / TEX Cache": [
        ("Sector Hit Rate", "l1tex__t_sector_hit_rate.pct"),
        ("Requests", "l1tex__t_requests_pipe_lsu.sum"),
        ("Sectors", "l1tex__t_sectors_pipe_lsu.sum"),
        ("Wavefronts", "l1tex__data_pipe_lsu_wavefronts.sum"),
        ("Bytes", "l1tex__t_bytes_pipe_lsu.sum"),
        ("Throughput", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ],
    "L2 Cache": [
        ("Sector Hit Rate", "lts__t_sector_hit_rate.pct"),
        ("Requests", "lts__t_requests.sum"),
        ("Sectors", "lts__t_sectors.sum"),
        ("Bytes", "lts__t_bytes.sum"),
        ("Throughput", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("Fill: Device Sectors", "lts__d_sectors_fill_device.sum"),
        ("Fill: Sysmem Sectors", "lts__d_sectors_fill_sysmem.sum"),
    ],
    "Device Memory (DRAM)": [
        ("Bytes", "dram__bytes.sum"),
        ("Bytes Read", "dram__bytes_read.sum"),
        ("Bytes Written", "dram__bytes_write.sum"),
        ("Sectors", "dram__sectors.sum"),
        ("Throughput", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ],
    "Shared Memory": [
        ("Load Instructions", "smsp__inst_executed_op_shared_ld.sum"),
        ("Store Instructions", "smsp__inst_executed_op_shared_st.sum"),
        ("Bank Conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
        ("Wavefronts", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"),
    ],
}


def _metric_row(kernel, label, metric):
    """One {Metric, Value, Unit} row, or None if the metric isn't present."""
    entry = kernel["metrics"].get(metric)
    if not entry:
        return None
    val, unit = entry["value"], entry["unit"] or ""
    if isinstance(val, bool):
        text = str(val)
    elif isinstance(val, (int, float)):
        if unit == "%":
            text = f"{val:.2f}"
        elif float(val).is_integer():
            text = f"{int(val):,}"
        else:
            text = f"{val:,.2f}"
    else:
        text = str(val)
    return {"Metric": label, "Value": text, "Unit": unit}


def section_memory_tables(kernel):
    with st.expander("Memory Tables", expanded=False):
        cols = st.columns(2)
        shown = 0
        for title, spec in _MEM_TABLES.items():
            rows = [r for r in (_metric_row(kernel, lbl, m) for lbl, m in spec) if r]
            if not rows:
                continue
            with cols[shown % 2]:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            shown += 1
        if not shown:
            st.info("Detailed memory tables need a report captured with the "
                    "Memory Workload Analysis tables (`--set full`).")


def section_instruction(kernel):
    with st.expander("Instruction Statistics", expanded=False):
        pipes = {}
        for name, entry in kernel["metrics"].items():
            if name.startswith("sm__inst_executed_pipe_") \
                    and name.endswith(".avg.pct_of_peak_sustained_elapsed"):
                v = entry["value"]
                if isinstance(v, (int, float)) and v > 0:
                    pipes[pretty(name).replace("Inst Executed ", "")] = v
        issued = get_val(kernel, "sm__inst_executed.avg.pct_of_peak_sustained_elapsed")
        if isinstance(issued, (int, float)):
            st.metric("Executed IPC (% of peak)", f"{issued:.1f} %")
        if pipes:
            st.plotly_chart(hbar(list(pipes.keys()), list(pipes.values()),
                                 highlight_max=True, xmax=110), width="stretch")
        else:
            st.info("No per-pipe instruction metrics in this report.")


def section_scheduler(kernel):
    with st.expander("Scheduler / Warp State Statistics", expanded=False):
        issue = get_val(kernel, "sm__issue_active.avg.pct_of_peak_sustained_elapsed")
        active_warps = get_val(kernel, "sm__warps_active.avg.per_cycle_active")
        occ = get_val(kernel, "sm__warps_active.avg.pct_of_peak_sustained_active")
        cols = st.columns(3)
        if isinstance(issue, (int, float)):
            cols[0].metric("Issue Active (% of peak)", f"{issue:.1f} %")
        if isinstance(active_warps, (int, float)):
            cols[1].metric("Active Warps / Cycle", round(active_warps, 1))
        if isinstance(occ, (int, float)):
            cols[2].metric("Achieved Occupancy (%)", round(occ, 1))

        stall_prefix = "smsp__pcsamp_warps_issue_stalled_"
        rows = []
        for name, entry in kernel["metrics"].items():
            if name.startswith(stall_prefix) and not name.endswith("_not_issued"):
                val = entry["value"] or 0
                if isinstance(val, (int, float)) and val > 0:
                    rows.append((name.replace(stall_prefix, "").replace("_", " ").title(), val))
        if rows:
            st.markdown("**Warp Stall Reasons** (PC-sampled)")
            labels = [r[0] for r in rows]
            values = [r[1] for r in rows]
            st.plotly_chart(hbar(labels, values, unit="", color="#ab63fa",
                                 highlight_max=True), width="stretch")
        else:
            st.info("No PC-sampling stall data in this report.")


def section_nvtx(kernel):
    if not kernel.get("nvtx"):
        return
    with st.expander("NVTX Ranges", expanded=True):
        st.caption("Ranges active on the call stack when this kernel launched.")
        for dom in kernel["nvtx"]:
            ranges = list(dom.get("push_pop", [])) + list(dom.get("start_end", []))
            if not ranges:
                continue
            crumb = "  ›  ".join(f"`{r}`" for r in ranges)
            st.markdown(f"**{dom['domain']}** — {crumb}")


def page_details(kernel):
    section_nvtx(kernel)
    section_speed_of_light(kernel)
    section_roofline(kernel)
    section_compute(kernel)
    section_memory(kernel)
    section_memory_tables(kernel)
    section_scheduler(kernel)
    section_instruction(kernel)

    with st.expander("Launch Statistics", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Grid Size", int(get_val(kernel, "launch__grid_size", 0) or 0))
        cols[1].metric("Block Size", int(get_val(kernel, "launch__block_size", 0) or 0))
        cols[2].metric("Threads", int(get_val(kernel, "launch__thread_count", 0) or 0))
        cols[3].metric("Waves / SM", round(get_val(kernel, "launch__waves_per_multiprocessor", 0) or 0, 2))

        cols2 = st.columns(4)
        cols2[0].metric("Registers / Thread", int(get_val(kernel, "launch__registers_per_thread", 0) or 0))
        cols2[1].metric("Shared Mem Dynamic (B)", int(get_val(kernel, "launch__shared_mem_per_block_dynamic", 0) or 0))
        cols2[2].metric("Shared Mem Static (B)", int(get_val(kernel, "launch__shared_mem_per_block_static", 0) or 0))
        cols2[3].metric("Duration (µs)", round((get_val(kernel, "gpu__time_duration.sum", 0) or 0) / 1000, 3))

    with st.expander("Occupancy", expanded=True):
        achieved = get_val(kernel, "sm__warps_active.avg.pct_of_peak_sustained_active", 0) or 0
        active_warps = get_val(kernel, "sm__warps_active.avg.per_cycle_active", 0) or 0
        max_warps = get_val(kernel, "smsp__maximum_warps_avg_per_active_cycle", 0) or 0

        cols = st.columns(3)
        cols[0].metric("Achieved Occupancy (%)", round(achieved, 1))
        cols[1].metric("Active Warps / Cycle", round(active_warps, 1))
        cols[2].metric("Max Warps / Cycle", int(max_warps))

        limiters = {
            "Registers":  int(get_val(kernel, "launch__occupancy_limit_registers", 0) or 0),
            "Warps":      int(get_val(kernel, "launch__occupancy_limit_warps", 0) or 0),
            "Shared Mem": int(get_val(kernel, "launch__occupancy_limit_shared_mem", 0) or 0),
            "Blocks":     int(get_val(kernel, "launch__occupancy_limit_blocks", 0) or 0),
        }
        min_val = min(limiters.values())
        fig = go.Figure(go.Bar(
            x=list(limiters.keys()), y=list(limiters.values()),
            marker_color=["#ef553b" if v == min_val else "#636efa" for v in limiters.values()],
            text=list(limiters.values()), textposition="outside",
        ))
        fig.update_layout(title="Occupancy Limiters (red = binding constraint)",
                          yaxis_title="Blocks", height=240, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Analysis & Recommendations", expanded=True):
        if kernel["rules"]:
            for rule in kernel["rules"]:
                msg = rule.get("message", {})
                title = msg.get("title", rule["name"])
                body = strip_markup(msg.get("message", ""))
                icon = "⚠️" if msg.get("type", 0) == 3 else "ℹ️"
                st.warning(f"**{icon} {title}**\n\n{body}")
        else:
            st.success("No issues detected.")


def page_raw(kernel, search):
    rows = [
        {"Metric": name, "Value": str(entry["value"]), "Unit": entry["unit"]}
        for name, entry in sorted(kernel["metrics"].items())
        if not search or search.lower() in name.lower()
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=400)


# ── main ─────────────────────────────────────────────────────────────────────

experiments = api.get_panel_experiments()
experiment_map = {exp.id: exp for exp in experiments}
experiment_ids = sorted([exp.id for exp in experiments])
data = sorted(get_all_rep_data(experiments, experiment_ids))

with st.sidebar:
    st.title("ncu-rep viewer")

    if not data:
        st.warning("No .ncu-rep assets found in the selected experiments.")
        st.stop()

    option = st.selectbox(
        "ncu-rep file:",
        data,
        format_func=lambda item: "%s: %s" % (item[1], item[2]),
    )
    exp_id, exp_name, file_name, asset_id, step = option

    # Comet has no direct asset download-to-file; fetch bytes and save locally.
    selected_file = download_asset(exp_id, asset_id, file_name)

    kernels = load_kernels(str(selected_file))
    st.caption(f"{len(kernels)} kernel launch(es)")

kernel_names = [f"[{k['range']}.{k['action']}] {k['name']}" for k in kernels]
col1, col2 = st.columns([1, 2])
selected_kernel = col1.selectbox("Kernel", kernel_names)
search = col2.text_input("Filter metrics", placeholder="e.g. dram, occupancy, launch")
kernel = kernels[kernel_names.index(selected_kernel)]

section_device_header(kernel)

crumb = nvtx_breadcrumb(kernel)
if crumb:
    st.caption(f"🏷️ NVTX:  {crumb}")

tab_summary, tab_details, tab_raw = st.tabs(["Summary", "Details", "Raw Metrics"])

with tab_summary:
    page_summary(kernels)
with tab_details:
    page_details(kernel)
with tab_raw:
    page_raw(kernel, search)
