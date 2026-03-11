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
AREA      = "CacheArea_mm2"
HIT_ENRG  = "HitDynamicEnergy_nJ"

PARETO_METRICS = [HIT_LAT, AREA, HIT_ENRG]

# Colours for up to 7 cells; extend if needed.
CELL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_cell_types(nsc_root):
    """Return dict mapping cell name (e.g. 'gcDRAM1') to '2T' or '3T'."""
    cell_dir = os.path.join(nsc_root, "config", "gc_cfg", "cells")
    cell_types = {}
    for cell_file in glob.glob(os.path.join(cell_dir, "*.cell")):
        cell_name = os.path.splitext(os.path.basename(cell_file))[0]
        with open(cell_file) as f:
            for line in f:
                m = re.search(r"GainCellType:\s*(\w+)", line)
                if m:
                    cell_types[cell_name] = m.group(1)
                    break
    return cell_types


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
    return df


def pareto_front_2d(df, xcol, ycol):
    """Return the 2-D pareto-optimal subset of df (both axes minimised)."""
    mask = paretoset(df[[xcol, ycol]], sense=["min", "min"])
    front = df[mask].sort_values(xcol)
    return front


def _lower_convex_hull(lx, ly):
    """Lower convex hull of points in log-log space, sorted by lx.
    Returns hull vertex indices. The lower hull is the boundary closest
    to the origin (bottom-left), since it passes below all other points."""
    hull = []
    for i in range(len(lx)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (lx[a]-lx[o])*(ly[i]-ly[o]) - (ly[a]-ly[o])*(lx[i]-lx[o])
            if cross <= 0:   # right turn or collinear: prune
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def draw_pareto_curve(ax, front, xcol, ycol, color, label):
    """Scatter + lower convex hull boundary in log-log space."""
    x = front[xcol].to_numpy()
    y = front[ycol].to_numpy()
    ax.scatter(x, y, color=color, s=25, zorder=4, alpha=0.8)

    lx = np.log10(x)
    ly = np.log10(y)
    idx = _lower_convex_hull(lx, ly)
    hlx, hly = lx[idx], ly[idx]
    # Interpolate finely in log-log space so segments look smooth on a log axis.
    fine_lx = np.linspace(hlx.min(), hlx.max(), 300)
    fine_ly = np.interp(fine_lx, hlx, hly)
    ax.plot(10**fine_lx, 10**fine_ly, color=color, linewidth=1.5, label=label)


def finish_axis(ax, xcol, ycol, title, log_x=True, log_y=True):
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xcol, fontsize=13)
    ax.set_ylabel(ycol, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=11)


# ── Plot 1: PUA vs not-PUA for each (cell, cfg) ───────────────────────────────

def plot_pua_vs_not_pua(groups, out_dir):
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
            (True,  "#1f77b4", "PUA"),
            (False, "#ff7f0e", "not PUA"),
        ]:
            front = pareto_front_2d(variants[is_pua], AREA, HIT_LAT)
            draw_pareto_curve(ax, front, AREA, HIT_LAT, color, label)

        finish_axis(ax, AREA, HIT_LAT, f"{cell} — {cfg}\nPUA vs not-PUA")
        plt.tight_layout()

        fname = f"{cell}_{cfg}.png"
        out_path = os.path.join(pua_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Plot 2 & 3: cells compared at fixed capacity / PUA mode ───────────────────

def plot_cells_at_cfg(groups, out_dir, xcol, ycol, subdir, fname_suffix="", title_note="", cell_types=None):
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
        pua_label = "PUA" if is_pua else "not_PUA"

        for i, (cell, df) in enumerate(sorted(cell_dfs)):
            color = CELL_COLORS[i % len(CELL_COLORS)]
            front = pareto_front_2d(df, xcol, ycol)
            ctype = cell_types.get(cell) if cell_types else None
            label = f"{cell} ({ctype})" if ctype else cell
            draw_pareto_curve(ax, front, xcol, ycol, color, label)

        title = f"{cfg} — {pua_label}\nPareto by cell"
        if title_note:
            title += f" ({title_note})"
        finish_axis(ax, xcol, ycol, title)
        plt.tight_layout()

        fname = f"{cfg}_{pua_label}{fname_suffix}.png"
        out_path = os.path.join(plot_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(csv_dir):
    assert os.path.isdir(csv_dir), f"Not a directory: {csv_dir}"

    nsc_root = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(nsc_root, "config", "plots", "compare")
    os.makedirs(out_dir, exist_ok=True)

    # Use the pre-computed pareto CSV files.
    all_csvs = sorted(glob.glob(os.path.join(csv_dir, "*_pareto.csv")))

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

    cell_types = load_cell_types(nsc_root)

    # Plot 1
    plot_pua_vs_not_pua(groups, out_dir)

    # Plot 2: area vs hit latency, one curve per cell
    plot_cells_at_cfg(groups, out_dir, AREA, HIT_LAT, "cells_area",
                      cell_types=cell_types)

    # Plot 3: hit energy vs hit latency, one curve per cell
    plot_cells_at_cfg(groups, out_dir, HIT_ENRG, HIT_LAT, "cells_energy",
                      cell_types=cell_types)

    # ── Filtered plots (plots 4 & 5) ─────────────────────────────────────────
    # Requires *_filtered_pareto.csv files produced by make_filtered_pareto.py.
    filtered_csvs = sorted(glob.glob(os.path.join(csv_dir, "*_filtered_pareto.csv")))

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
                          fname_suffix="_filtered", title_note="mat-size filtered",
                          cell_types=cell_types)

        # Plot 5: hit energy vs hit latency, mat-size filtered
        plot_cells_at_cfg(groups_filtered, out_dir, HIT_ENRG, HIT_LAT, "cells_energy",
                          fname_suffix="_filtered", title_note="mat-size filtered",
                          cell_types=cell_types)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python plot_compare.py <csv_dir>"
    main(sys.argv[1])
