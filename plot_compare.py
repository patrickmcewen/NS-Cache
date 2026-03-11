"""
plot_compare.py  –  cross-file comparison plots for NS-Cache sweep results

Five plot types (all saved under config/plots/compare/):

  1. pua_vs_not_pua/<cell>_<cfg>.png
       HitLatency vs CacheArea pareto curve for PUA and not-PUA variants of
       the same (cell, cfg) pair.

  2. cells_area/<cfg>_[pua|not_pua].png
       HitLatency vs CacheArea pareto curves for every cell config at a given
       capacity/cfg, for a chosen PUA mode.

  3. cells_energy/<cfg>_[pua|not_pua].png
       Same but HitDynamicEnergy vs HitLatency.

  4. cells_area/<cfg>_[pua|not_pua]_filtered.png
       Same as (2) using *_filtered_pareto.csv files (mat-size filtered pareto
       fronts produced by make_filtered_pareto.py).

  5. cells_energy/<cfg>_[pua|not_pua]_filtered.png
       Same as (3) using *_filtered_pareto.csv files.

  6. cells_leakage/<cfg>_[pua|not_pua].png
       HitLatency vs Leakage_mW pareto curves for every cell config.

  7. cells_leakage/<cfg>_[pua|not_pua]_filtered.png
       Same as (6) using *_filtered_pareto.csv files.

Usage:
    python plot_compare.py <csv_dir>

<csv_dir> should contain files named like:
    gcDRAM1_GainCell_L2Cache_512KB_pareto.csv
    gcDRAM1_GainCell_L2Cache_512KB_not_pua_pareto.csv
    gcDRAM1_GainCell_L2Cache_512KB_filtered_pareto.csv      (from make_filtered_pareto.py)
    gcDRAM1_GainCell_L2Cache_512KB_not_pua_filtered_pareto.csv
"""

import sys
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

# ── Columns used across plots ──────────────────────────────────────────────────
HIT_LAT   = "HitLatency_ns"
WRITE_LAT = "WriteLatency_ns"
AREA      = "CacheArea_mm2"
HIT_ENRG  = "HitDynamicEnergy_nJ"
WRITE_ENRG = "WriteDynamicEnergy_nJ"
LEAKAGE   = "Leakage_mW"

PARETO_METRICS = [HIT_LAT, WRITE_LAT, AREA, HIT_ENRG, WRITE_ENRG, LEAKAGE]

COL_LABEL = {
    HIT_LAT:    "Hit Latency (ns)",
    WRITE_LAT:  "Write Latency (ns)",
    AREA:       "Cache Area (mm²)",
    HIT_ENRG:   "Hit Dynamic Energy (pJ)",
    WRITE_ENRG: "Write Dynamic Energy (pJ)",
    LEAKAGE:    "Leakage (mW)",
}

# Multiply column values by this factor before plotting.
COL_SCALE = {
    HIT_ENRG:   1e3,   # nJ → pJ
    WRITE_ENRG: 1e3,   # nJ → pJ
}

# Colours for up to 7 cells; extend if needed.
CELL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]

CACHE_TYPE_TO_WORD_SIZE = {
    "L1": "64b",
    "L2": "64b",
    "L3": "512b",
}

def cfg_name_to_label(cfg_name):
    # e.g. "GainCell_L2Cache_512KB" → "L2 Cache, 512KB"
    m = re.search(r"(L\d)Cache_(\S+)", cfg_name)
    if m:
        return f"{m.group(1)} Cache, {m.group(2)}, {CACHE_TYPE_TO_WORD_SIZE[m.group(1)]} word"
    return cfg_name



# ── Helpers ────────────────────────────────────────────────────────────────────

def load_cell_info(nsc_root):
    """Return (cell_types, cell_areas) dicts keyed by cell name (e.g. 'gcDRAM1').
    cell_types maps to '2T' or '3T'; cell_areas maps to float F²."""
    cell_dir = os.path.join(nsc_root, "config", "gc_cfg", "cells")
    cell_types = {}
    cell_areas = {}
    for cell_file in glob.glob(os.path.join(cell_dir, "*.cell")):
        cell_name = os.path.splitext(os.path.basename(cell_file))[0]
        with open(cell_file) as f:
            for line in f:
                m = re.search(r"GainCellType:\s*(\w+)", line)
                if m:
                    cell_types[cell_name] = m.group(1)
                m = re.search(r"CellArea \(F\^2\):\s*([\d.eE+\-]+)", line)
                if m:
                    cell_areas[cell_name] = float(m.group(1))
    return cell_types, cell_areas


def parse_filename(path, suffix="_pareto"):
    """Return (cell, cfg, is_pua) from a CSV filename ending in <suffix>."""
    base = os.path.splitext(os.path.basename(path))[0]
    assert base.endswith(suffix), f"Expected {suffix}.csv file, got: {path}"
    base = base[: -len(suffix)]
    if base.endswith("_not_pua"):
        is_pua = False
        base = base[: -len("_not_pua")]
    else:
        is_pua = True
    idx = base.index("_")
    cell = base[:idx]
    cfg  = base[idx + 1:]
    return cell, cfg, is_pua


def load_df(path):
    df = pd.read_csv(path)
    for col in PARETO_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=PARETO_METRICS, inplace=True)
    df = df[(df[PARETO_METRICS] > 0).all(axis=1)]
    return df


def pareto_front_2d(df, xcol, ycol):
    """Return the 2-D pareto-optimal subset of df (both axes minimised)."""
    mask = paretoset(df[[xcol, ycol]], sense=["min", "min"])
    front = df[mask].sort_values(xcol)
    return front



def _scaled(df, col):
    return df[col] * COL_SCALE.get(col, 1)


def _lower_convex_hull(lx, ly):
    """Lower convex hull indices of points in log-log space, sorted by lx."""
    hull = []
    for i in range(len(lx)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (lx[a]-lx[o])*(ly[i]-ly[o]) - (ly[a]-ly[o])*(lx[i]-lx[o])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def draw_pareto_curve(ax, front, xcol, ycol, color, label, all_points=None, convex_hull=False):
    """Background scatter of all_points, then pareto front scatter + line.
    If convex_hull=True, draws the lower convex hull in log-log space instead
    of connecting all pareto points directly."""
    if all_points is not None:
        ax.scatter(_scaled(all_points, xcol), _scaled(all_points, ycol),
                   color=color, s=10, zorder=2, alpha=0.25, linewidths=0)
    x = _scaled(front, xcol).to_numpy()
    y = _scaled(front, ycol).to_numpy()
    ax.scatter(x, y, color=color, s=30, zorder=4, alpha=0.9)
    if convex_hull:
        lx, ly = np.log10(x), np.log10(y)
        idx = _lower_convex_hull(lx, ly)
        hlx, hly = lx[idx], ly[idx]
        fine_lx = np.linspace(hlx.min(), hlx.max(), 300)
        fine_ly = np.interp(fine_lx, hlx, hly)
        ax.plot(10**fine_lx, 10**fine_ly, color=color, linewidth=1.5, label=label)
    else:
        ax.plot(x, y, color=color, linewidth=1.5, label=label)


def _set_ticks(ax, n=7, xlim=None, ylim=None):
    """Set n evenly-spaced ticks (logspace or linspace) on each axis."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:.2g}")

    for maxis, is_log, lim in [
        (ax.xaxis, ax.get_xscale() == "log", xlim),
        (ax.yaxis, ax.get_yscale() == "log", ylim),
    ]:
        lo, hi = lim if lim is not None else maxis.get_view_interval()
        ticks = np.logspace(np.log10(lo), np.log10(hi), n) if is_log else np.linspace(lo, hi, n)
        maxis.set_major_locator(FixedLocator(ticks))
        maxis.set_minor_locator(NullLocator())
        maxis.set_major_formatter(fmt)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="x", rotation=30)


def finish_axis(ax, xcol, ycol, title, log_x=True, log_y=True, xlim=None, ylim=None):
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(COL_LABEL.get(xcol, xcol), fontsize=13)
    ax.set_ylabel(COL_LABEL.get(ycol, ycol), fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=11)
    _set_ticks(ax, xlim=xlim, ylim=ylim)


# ── Plot 1: PUA vs not-PUA for each (cell, cfg) ───────────────────────────────

def plot_pua_vs_not_pua(groups, out_dir, cell_types, cell_areas=None, convex_hull=False):
    """
    groups: dict keyed by (cell, cfg) → {True: df_pua, False: df_not_pua}
    """
    pua_dir = os.path.join(out_dir, "pua_vs_not_pua")
    os.makedirs(pua_dir, exist_ok=True)

    for (cell, cfg), variants in sorted(groups.items()):
        if True not in variants or False not in variants:
            continue  # need both variants for this plot

        fig, ax = plt.subplots(figsize=(7, 5))

        for is_pua, color, label in [
            (True,  "#1f77b4", "PuA"),
            (False, "#ff7f0e", "not PUA"),
        ]:
            df = variants[is_pua]
            front = pareto_front_2d(df, AREA, HIT_LAT)
            draw_pareto_curve(ax, front, AREA, HIT_LAT, color, label, all_points=df, convex_hull=convex_hull)

        ctype = cell_types.get(cell)
        area_f2 = cell_areas.get(cell) if cell_areas else None
        area_str = f", {area_f2:g} F²" if area_f2 is not None else ""
        finish_axis(ax, AREA, HIT_LAT, f"Cell {cell[-1]} ({ctype}{area_str}) — {cfg_name_to_label(cfg)}\n Effect of Peripherals Under Array Design")
        plt.tight_layout()

        fname = f"{cell}_{cfg}.png"
        out_path = os.path.join(pua_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Plot 2 & 3: cells compared at fixed capacity / PUA mode ───────────────────

def plot_cells_at_cfg(groups, out_dir, xcol, ycol, subdir, fname_suffix="", title_note="", cell_types=None, cell_areas=None, convex_hull=False):
    """
    For every (cfg, is_pua) combination, produce one plot with one pareto curve
    per cell.
    groups: dict keyed by (cell, cfg) → {True: df, False: df}
    cell_types: optional dict mapping cell name → '2T'/'3T' for legend labels
    """
    plot_dir = os.path.join(out_dir, subdir)
    os.makedirs(plot_dir, exist_ok=True)

    from collections import defaultdict
    cfg_pua_cells = defaultdict(list)
    for (cell, cfg), variants in groups.items():
        for is_pua, df in variants.items():
            cfg_pua_cells[(cfg, is_pua)].append((cell, df))

    for (cfg, is_pua), cell_dfs in sorted(cfg_pua_cells.items()):
        if len(cell_dfs) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        pua_label = "Peripherals Under Array" if is_pua else "No Peripherals Under Array"

        all_x, all_y = [], []
        for i, (cell, df) in enumerate(sorted(cell_dfs)):
            color = CELL_COLORS[i % len(CELL_COLORS)]
            front = pareto_front_2d(df, xcol, ycol)
            ctype = cell_types.get(cell) if cell_types else None
            area_f2 = cell_areas.get(cell) if cell_areas else None
            area_str = f", {area_f2:g} F²" if area_f2 is not None else ""
            label = f"Cell {cell[-1]} ({ctype}{area_str})" if ctype else cell
            draw_pareto_curve(ax, front, xcol, ycol, color, label, all_points=df, convex_hull=convex_hull)
            all_x.append(_scaled(front, xcol).to_numpy())
            all_y.append(_scaled(front, ycol).to_numpy())

        xs = np.concatenate(all_x)
        ys = np.concatenate(all_y)
        pad = 0.08  # 8% padding in log space
        xlim = (10 ** (np.log10(xs.min()) - pad), 10 ** (np.log10(xs.max()) + pad))
        ylim = (10 ** (np.log10(ys.min()) - pad), 10 ** (np.log10(ys.max()) + pad))

        title = f"{cfg_name_to_label(cfg)} — {pua_label}\n"
        if title_note:
            title += f" ({title_note})"
        finish_axis(ax, xcol, ycol, title, xlim=xlim, ylim=ylim)
        plt.tight_layout()

        fname = f"{cfg}_{pua_label}{fname_suffix}.png"
        out_path = os.path.join(plot_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(csv_dir, convex_hull=False):
    assert os.path.isdir(csv_dir), f"Not a directory: {csv_dir}"

    nsc_root = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(nsc_root, "config", "plots", "compare")
    os.makedirs(out_dir, exist_ok=True)

    # Use the pre-computed pareto CSV files.
    all_csvs = sorted([p for p in glob.glob(os.path.join(csv_dir, "*_pareto.csv")) if not p.endswith("_filtered_pareto.csv") and not (p.find("no_set") != -1)])

    # groups[(cell, cfg)][is_pua] = DataFrame
    groups = {}
    for path in all_csvs:
        try:
            cell, cfg, is_pua = parse_filename(path)
        except ValueError:
            print(f"Skipping (cannot parse): {os.path.basename(path)}")
            continue
        key = (cell, cfg)
        if key not in groups:
            groups[key] = {}
        assert is_pua not in groups[key], \
            f"Duplicate entry for ({cell}, {cfg}, pua={is_pua})"
        df = load_df(path)
        print(f"Loaded {len(df):4d} rows  [{('PUA' if is_pua else 'not_pua'):7s}]  {cell}  {cfg}")
        groups[key][is_pua] = df

    assert groups, "No valid CSV files found."

    cell_types, cell_areas = load_cell_info(nsc_root)

    # Plot 1
    plot_pua_vs_not_pua(groups, out_dir, cell_types, cell_areas, convex_hull=convex_hull)

    # Plot 2: area vs hit latency, one curve per cell
    plot_cells_at_cfg(groups, out_dir, AREA, HIT_LAT, "cells_area",
                      cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

    # Plot 3: hit energy vs hit latency, one curve per cell
    plot_cells_at_cfg(groups, out_dir, HIT_ENRG, HIT_LAT, "cells_energy",
                      cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

    # Plot 6: leakage power vs hit latency, one curve per cell
    plot_cells_at_cfg(groups, out_dir, LEAKAGE, HIT_LAT, "cells_leakage",
                      cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

    # Plot 7: write latency vs area, one curve per cell
    plot_cells_at_cfg(groups, out_dir, AREA, WRITE_LAT, "cells_write_area",
                      cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

    # Plot 8: write latency vs write energy, one curve per cell
    plot_cells_at_cfg(groups, out_dir, WRITE_ENRG, WRITE_LAT, "cells_write_energy",
                      cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

    # ── Filtered plots (plots 4 & 5) ─────────────────────────────────────────
    # Requires *_filtered_pareto.csv files produced by make_filtered_pareto.py.
    filtered_csvs = sorted([p for p in glob.glob(os.path.join(csv_dir, "*_filtered_pareto.csv")) if not (p.find("no_set") != -1)])

    groups_filtered = {}
    for path in filtered_csvs:
        try:
            cell, cfg, is_pua = parse_filename(path, suffix="_filtered_pareto")
        except (ValueError, AssertionError):
            print(f"Skipping (cannot parse): {os.path.basename(path)}")
            continue
        key = (cell, cfg)
        if key not in groups_filtered:
            groups_filtered[key] = {}
        assert is_pua not in groups_filtered[key], \
            f"Duplicate entry for ({cell}, {cfg}, pua={is_pua})"
        df = load_df(path)
        print(f"Loaded {len(df):4d} rows  [{('PUA' if is_pua else 'not_pua'):7s}]  {cell}  {cfg}  (filtered)")
        groups_filtered[key][is_pua] = df

    if groups_filtered:
        # Plot 4: area vs hit latency, mat-size filtered
        plot_cells_at_cfg(groups_filtered, out_dir, AREA, HIT_LAT, "cells_area",
                          fname_suffix="_filtered", title_note="mat dimension <= 128 for 2T, <= 1024 for 3T",
                          cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

        # Plot 5: hit energy vs hit latency, mat-size filtered
        plot_cells_at_cfg(groups_filtered, out_dir, HIT_ENRG, HIT_LAT, "cells_energy",
                          fname_suffix="_filtered", title_note="mat dimension <= 128 for 2T, <= 1024 for 3T",
                          cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

        # Plot 7: leakage power vs hit latency, mat-size filtered
        plot_cells_at_cfg(groups_filtered, out_dir, LEAKAGE, HIT_LAT, "cells_leakage",
                          fname_suffix="_filtered", title_note="mat dimension <= 128 for 2T, <= 1024 for 3T",
                          cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

        # Plot 8: write latency vs area, mat-size filtered
        plot_cells_at_cfg(groups_filtered, out_dir, AREA, WRITE_LAT, "cells_write_area",
                          fname_suffix="_filtered", title_note="mat dimension <= 128 for 2T, <= 1024 for 3T",
                          cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)

        # Plot 9: write latency vs write energy, mat-size filtered
        plot_cells_at_cfg(groups_filtered, out_dir, WRITE_ENRG, WRITE_LAT, "cells_write_energy",
                          fname_suffix="_filtered", title_note="mat dimension <= 128 for 2T, <= 1024 for 3T",
                          cell_types=cell_types, cell_areas=cell_areas, convex_hull=convex_hull)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir")
    parser.add_argument("--convex-hull", action="store_true",
                        help="Draw lower convex hull in log-log space instead of connecting all pareto points")
    args = parser.parse_args()
    main(args.csv_dir, convex_hull=args.convex_hull)
