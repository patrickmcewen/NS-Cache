#!/usr/bin/env python3
"""Run NS-Cache on a .cfg file, then plot the Pareto front of the output CSV."""

import sys
import os
import subprocess

NSC      = os.path.join(os.path.dirname(__file__), "nsc")
PLOT     = os.path.join(os.path.dirname(__file__), "plot_pareto.py")
PYTHON   = sys.executable


def main(cfg_path):
    assert cfg_path.endswith(".cfg"), f"Expected a .cfg file, got: {cfg_path}"
    assert os.path.isfile(cfg_path), f"File not found: {cfg_path}"

    print(f"=== Running NS-Cache: {cfg_path} ===")
    subprocess.run([NSC, cfg_path], check=True)

    nsc_root = os.path.dirname(os.path.abspath(__file__))
    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
    csv_path = os.path.join(nsc_root, "config", "csv", cfg_name + ".csv")
    assert os.path.isfile(csv_path), f"Expected output CSV not found: {csv_path}"

    print(f"\n=== Plotting Pareto front: {csv_path} ===")
    subprocess.run([PYTHON, PLOT, csv_path], check=True)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python run_and_plot.py <path/to/config.cfg>"
    main(sys.argv[1])
