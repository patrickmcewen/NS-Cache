#!/usr/bin/env python3
"""Generate NS-Cache .cell config files from memory_cell_data.csv.

Unit conventions:
  - Gate width for factor=1 transistor: 60 nm (user-specified)
  - AccessCMOSWidth in F where F = 60 nm / BASE_WIDTH_F (= 45.8 nm from template)
  - CurrentOn/Off in A/m: derived from drive currents divided by actual width in meters
  - CellArea in F^2: converted from nm^2 using featureSize
  - RetentionTime in us: converted from seconds
  - DRAMCellCapacitance in F: used directly from Csn_Add column
"""

import csv
import os

GATE_WIDTH_NM = 60.0          # actual width for factor=1 transistor (nm)
BASE_WIDTH_F  = 1         # AccessCMOSWidth (F) for factor=1 (from template)
FEATURE_SIZE_NM = GATE_WIDTH_NM  # ~45.8 nm
CELL_ASPECT_RATIO_2T = 1
CELL_ASPECT_RATIO_3T = 0.67
MIN_SENSE_MV = 10
TEMPERATURE_K = 300

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "memory_cell_data.csv")


def fmt(v):
    """Format a float: use scientific notation for small values, else plain float."""
    if abs(v) < 1e-2 and v != 0:
        return f"{v:.3e}"
    return f"{v:.4g}"


def write_cell(path, row, index):
    config    = row["Config"].strip()          # "2T" or "3T"
    wfactor   = float(row["wfactor"])
    rfactor   = float(row["rfactor"])
    r1factor  = float(row["r1factor (select)"])
    csn_add   = float(row["Csn_Add"])          # F
    write_uA  = float(row["Write Drive (uA)"])
    off_A     = float(row["Off Current (A)"])
    area_nm2  = float(row["Cell Area 3.0 (nm^2)"])
    retention_s = float(row["Retention (s)"])

    assert config in ("2T", "3T"), f"Unknown config: {config}"
    assert wfactor > 0 and rfactor > 0, "Factors must be positive"

    gate_width_m = GATE_WIDTH_NM * 1e-9  # 60 nm in meters

    # On-current (A/m): write drive / actual write transistor width
    on_current_Am = (write_uA * 1e-6) / (wfactor * gate_width_m)

    # Off-current (A/m): CSV off current is for a factor=1 (60 nm) transistor
    off_current_Am = off_A / gate_width_m

    # Transistor widths in F
    w_width_F  = wfactor  * BASE_WIDTH_F
    r_width_F  = rfactor  * BASE_WIDTH_F
    r1_width_F = r1factor * BASE_WIDTH_F

    # Cell area: convert nm^2 -> F^2
    cell_area_F2 = area_nm2 / (FEATURE_SIZE_NM ** 2)

    # Retention: seconds -> microseconds
    retention_us = retention_s * 1e6

    lines = [
        f"-MemCellType: gcDRAM",
        f"",
        f"",
        f"-CellArea (F^2): {fmt(cell_area_F2)}",
        f"-CellAspectRatio: {CELL_ASPECT_RATIO_2T if config == '2T' else CELL_ASPECT_RATIO_3T}",
        f"",
        f"-ReadMode: voltage",
        f"",
        f"-AccessType: CMOS",
        f"-AccessCMOSWidth (F): {fmt(w_width_F)}",
        f"-AccessCMOSWidth1 (F): {fmt(r_width_F)}",
    ]

    if config == "3T":
        lines.append(f"-AccessCMOSWidth2 (F): {fmt(r1_width_F)}")

    lines += [
        f"",
        f"-DRAMCellCapacitance (F): {fmt(csn_add)}",
        f"-ResetVoltage (V): vdd",
        f"-SetVoltage (V): vdd",
        f"",
        f"-MinSenseVoltage (mV): {MIN_SENSE_MV}",
        f"",
        f"-RetentionTime (us): {fmt(retention_us)}",
        f"-Temperature (K): {TEMPERATURE_K}",
        f"",
        f"-CurrentOnNmos (A/m): {fmt(on_current_Am)}",
        f"-CurrentOffNmos (A/m): {fmt(off_current_Am)}",
        f"-CurrentOnNmos1 (A/m): {fmt(on_current_Am)}",
        f"-CurrentOffNmos1 (A/m): {fmt(off_current_Am)}",
    ]

    if config == "3T":
        lines += [
            f"-CurrentOnNmos2 (A/m): {fmt(on_current_Am)}",
            f"-CurrentOffNmos2 (A/m): {fmt(off_current_Am)}",
        ]

    lines += [
        f"",
        f"-GainCellType: {config}",
        f"",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Wrote {os.path.basename(path)}  "
          f"[{config} w={wfactor} r={rfactor} r1={r1factor}  "
          f"on={fmt(on_current_Am)} A/m  off={fmt(off_current_Am)} A/m  "
          f"area={fmt(cell_area_F2)} F^2  ret={fmt(retention_us)} us]")


def main():
    print(f"Feature size: {FEATURE_SIZE_NM:.2f} nm  "
          f"(gate width {GATE_WIDTH_NM} nm / base width {BASE_WIDTH_F} F)")

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["Config"].strip()]

    for i, row in enumerate(rows, start=1):
        out_path = os.path.join(SCRIPT_DIR, f"gcDRAMPlanar{i}.cell")
        write_cell(out_path, row, i)

    print(f"\nGenerated {len(rows)} cell config(s).")


if __name__ == "__main__":
    main()
