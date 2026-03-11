#!/usr/bin/env python3
"""For each cell file, copy the sweep configs, patch the cell path, and run run_and_plot.py."""

import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT          = os.path.dirname(os.path.abspath(__file__))
CELLS_DIR     = os.path.join(ROOT, "config", "gc_cfg", "cells")
BASE_CFG_DIR  = os.path.join(ROOT, "config", "gc_cfg", "sweep_configs_no_set")
RUN_AND_PLOT  = os.path.join(ROOT, "run_and_plot.py")
PYTHON        = sys.executable

CELL_LINE_RE  = re.compile(r"^-MemoryCellInputFile:.*$", re.MULTILINE)


def run_cell(cell_path):
    cell_name = os.path.splitext(os.path.basename(cell_path))[0]
    out_dir   = os.path.join(ROOT, "config", "gc_cfg", cell_name)
    os.makedirs(out_dir, exist_ok=True)

    cell_rel  = os.path.relpath(cell_path, ROOT)

    base_cfgs = [f for f in os.listdir(BASE_CFG_DIR) if f.endswith(".cfg")]
    assert base_cfgs, f"No .cfg files found in {BASE_CFG_DIR}"

    for cfg_name in base_cfgs:
        src = os.path.join(BASE_CFG_DIR, cfg_name)
        dst = os.path.join(out_dir, f"{cell_name}_{cfg_name}")

        with open(src) as f:
            content = f.read()

        patched = CELL_LINE_RE.sub(f"-MemoryCellInputFile: {cell_rel}", content)
        assert patched != content, f"No -MemoryCellInputFile line found in {src}"

        with open(dst, "w") as f:
            f.write(patched)

        print(f"[{cell_name}] Running {cfg_name} ...")
        subprocess.run([PYTHON, RUN_AND_PLOT, dst], check=True)

    return cell_name


def main():
    cell_files = sorted(
        os.path.join(CELLS_DIR, f)
        for f in os.listdir(CELLS_DIR)
        if f.endswith(".cell")
    )
    assert cell_files, f"No .cell files found in {CELLS_DIR}"

    with ProcessPoolExecutor(max_workers=len(cell_files)) as ex:
        futures = {ex.submit(run_cell, c): c for c in cell_files}
        for fut in as_completed(futures):
            cell_path = futures[fut]
            try:
                name = fut.result()
                print(f"[DONE] {name}")
            except Exception as e:
                print(f"[FAIL] {cell_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
