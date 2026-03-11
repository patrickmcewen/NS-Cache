"""
make_filtered_pareto.py  –  apply mat-dimension filter then compute pareto set

For each raw (non-pareto) CSV in <csv_dir>, restricts designs by mat size based
on the cell's transistor count, computes the 3-objective pareto front, and saves
the result as <base>_filtered_pareto.csv in the same directory.

Mat-size limits (rows and columns):
    2T gain cells  →  ≤ 128
    3T gain cells  →  ≤ 1024

Usage:
    python make_filtered_pareto.py <csv_dir>
"""

import sys
import os
import re
import glob

import pandas as pd
from paretoset import paretoset

HIT_LAT  = "HitLatency_ns"
AREA     = "CacheArea_mm2"
HIT_ENRG = "HitDynamicEnergy_nJ"
WRITE_LAT = "WriteLatency_ns"
LEAKAGE = "Leakage_mW"

PARETO_METRICS = [HIT_LAT, AREA, HIT_ENRG, WRITE_LAT, LEAKAGE]

MAT_LIMITS = {"2T": 128, "3T": 1024}


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


def cell_name_from_path(path):
    """Return the leading cell name token (e.g. 'gcDRAM1') from a CSV path."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base[: base.index("_")]


def filter_by_mat_size(df, cell_type):
    limit = MAT_LIMITS[cell_type]
    return df[(df["Data_MatRows"] <= limit) & (df["Data_MatCols"] <= limit)]


def main(csv_dir):
    assert os.path.isdir(csv_dir), f"Not a directory: {csv_dir}"

    nsc_root = os.path.dirname(os.path.abspath(__file__))
    cell_types = load_cell_types(nsc_root)
    print(f"Cell types: {cell_types}")

    raw_csvs = sorted(
        p for p in glob.glob(os.path.join(csv_dir, "*.csv"))
        if not os.path.basename(p).endswith("_pareto.csv")
        and not os.path.basename(p).endswith("_filtered_pareto.csv")
    )
    assert raw_csvs, f"No raw CSV files found in {csv_dir}"

    for path in raw_csvs:
        cell = cell_name_from_path(path)
        ctype = cell_types.get(cell)
        assert ctype is not None, \
            f"No cell type found for '{cell}' — check config/gc_cfg/cells/"

        df = pd.read_csv(path)
        for col in PARETO_METRICS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=PARETO_METRICS, inplace=True)

        df = filter_by_mat_size(df, ctype)

        mask = paretoset(df[PARETO_METRICS], sense=["min"] * len(PARETO_METRICS))
        pareto = df[mask]

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(csv_dir, f"{base}_filtered_pareto.csv")
        pareto.to_csv(out_path, index=False)
        print(f"  {cell} ({ctype}, limit={MAT_LIMITS[ctype]}): "
              f"{len(df)} filtered → {len(pareto)} pareto  →  {os.path.basename(out_path)}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python make_filtered_pareto.py <csv_dir>"
    main(sys.argv[1])
