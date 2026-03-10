import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from paretoset import paretoset

# ── Configurable Pareto metrics (all minimized) ───────────────────────────────
PARETO_METRICS = [
    "HitLatency_ns",
    "WriteLatency_ns",
    "CacheArea_um2",
    "Leakage_mW",
]

# ── Plot axes ─────────────────────────────────────────────────────────────────
X_COL    = "Data_MatRows"
Y_COL    = "Data_MatCols"
Z_COLS   = ["HitLatency_ns",     "WriteLatency_ns",    "CacheArea_um2"]
Z_LABELS = ["Hit Latency (ns)",  "Write Latency (ns)", "Cache Area (µm²)"]

# ── Mat-level plot axes ────────────────────────────────────────────────────────
MAT_Z_COLS   = ["Data_ReadLatency_ns",   "Data_WriteLatency_ns",   "Data_MatArea_um2"]
MAT_Z_LABELS = ["Mat Read Latency (ns)", "Mat Write Latency (ns)", "Mat Area (µm²)"]


def log10_ticks(data):
    """Ticks at integer powers of 10 plus actual min/max, labelled as values."""
    vmin, vmax = data.min(), data.max()
    lo = int(np.floor(np.log10(vmin)))
    hi = int(np.ceil(np.log10(vmax)))
    # Integer power-of-10 positions
    power_ticks = np.arange(lo, hi + 1, dtype=float)
    # Add actual min and max positions (in log10 space)
    ticks = np.unique(np.concatenate([power_ticks, [np.log10(vmin), np.log10(vmax)]]))

    def fmt(v):
        # Format actual value: use scientific if very large/small, else decimal
        if v == 0:
            return "0"
        if abs(v) >= 1e4 or abs(v) < 1e-2:
            return f"{v:.2e}"
        return f"{v:.3g}"

    labels = []
    for t in ticks:
        v = 10 ** t
        # Show as 10^x for exact integer powers, else show actual value
        if abs(t - round(t)) < 1e-9:
            labels.append(f"$10^{{{int(round(t))}}}$")
        else:
            labels.append(fmt(v))
    return ticks, labels


def pow2_ticks(data):
    """Actual power-of-2 values present in the data, for use as axis tick positions and labels."""
    unique = np.unique(data.astype(int))
    powers = unique[unique > 0]
    logs = np.log2(powers)
    exact = powers[np.abs(logs - np.round(logs)) < 1e-6]
    values = exact.astype(float)
    labels = [str(int(v)) for v in exact]
    return values, labels


def set_pow2_axis(ax, xi, yi):
    """Apply power-of-2 ticks with actual value labels on a log-scale 2D axes."""
    xvals, xlabels = pow2_ticks(xi)
    yvals, ylabels = pow2_ticks(yi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator(xvals))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(xlabels))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.FixedLocator(yvals))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(ylabels))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def main(csv_path):
    nsc_root  = os.path.dirname(os.path.abspath(__file__))
    cfg_name  = os.path.splitext(os.path.basename(csv_path))[0]
    plots_dir = os.path.join(nsc_root, "config", "plots", cfg_name)
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in PARETO_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=PARETO_METRICS, inplace=True)
    print(f"Loaded {len(df)} rows from {csv_path}")

    mask = paretoset(df[PARETO_METRICS], sense=["min"] * len(PARETO_METRICS))
    pareto = df[mask]
    print(f"Pareto front: {len(pareto)} designs")

    pareto_csv = os.path.join(nsc_root, "config", "csv", cfg_name + "_pareto.csv")
    pareto.to_csv(pareto_csv, index=False)
    print(f"Saved: {pareto_csv}")

    fig = plt.figure(figsize=(18, 6))

    for idx, (z_col, z_label) in enumerate(zip(Z_COLS, Z_LABELS)):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        x = pareto[X_COL].to_numpy(dtype=float)
        y = pareto[Y_COL].to_numpy(dtype=float)
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        x, y, z = x[valid], y[valid], z[valid]

        lx, ly, lz = np.log10(x), np.log10(y), np.log10(z)
        sc = ax.scatter(lx, ly, lz, c=lz, cmap="viridis", s=20, alpha=0.7)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = log10_ticks(z)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        # Colorbar: show actual values as 10^x ticks
        cb_ticks, cb_labels = log10_ticks(z)
        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(cb_labels)

    fig.suptitle(f"Pareto Front — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "pareto_3d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # ── 2D plots ──────────────────────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = pareto[X_COL].to_numpy(dtype=float)
    y = pareto[Y_COL].to_numpy(dtype=float)

    for ax2, z_col, z_label in zip(axes, Z_COLS, Z_LABELS):
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        xi, yi, zi = x[valid], y[valid], z[valid]

        sc = ax2.scatter(xi, yi, c=np.log10(zi), cmap="viridis", s=40, alpha=0.8)

        cb_ticks, cb_labels = log10_ticks(zi)
        cb = fig2.colorbar(sc, ax=ax2, label=z_label)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(cb_labels)

        set_pow2_axis(ax2, xi, yi)
        ax2.tick_params(axis="both", labelsize=8)
        ax2.set_xlabel(X_COL,   fontsize=9)
        ax2.set_ylabel(Y_COL,   fontsize=9)
        ax2.set_title(z_label,  fontsize=10)
        ax2.grid(True, which="major", linestyle="--", alpha=0.4)

    fig2.suptitle(f"Pareto Front — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path2 = os.path.join(plots_dir, "pareto_2d.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")

    # ── Mat-level 3D plots ────────────────────────────────────────────────────
    fig3 = plt.figure(figsize=(18, 6))

    for idx, (z_col, z_label) in enumerate(zip(MAT_Z_COLS, MAT_Z_LABELS)):
        ax = fig3.add_subplot(1, 3, idx + 1, projection="3d")

        x = pareto[X_COL].to_numpy(dtype=float)
        y = pareto[Y_COL].to_numpy(dtype=float)
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        x, y, z = x[valid], y[valid], z[valid]

        lx, ly, lz = np.log10(x), np.log10(y), np.log10(z)
        sc = ax.scatter(lx, ly, lz, c=lz, cmap="viridis", s=20, alpha=0.7)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = log10_ticks(z)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        cb_ticks, cb_labels = log10_ticks(z)
        cb = fig3.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(cb_labels)

    fig3.suptitle(f"Pareto Front (Mat Level) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path3 = os.path.join(plots_dir, "pareto_mat_3d.png")
    plt.savefig(out_path3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path3}")

    # ── Mat-level 2D plots ────────────────────────────────────────────────────
    fig4, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = pareto[X_COL].to_numpy(dtype=float)
    y = pareto[Y_COL].to_numpy(dtype=float)

    for ax4, z_col, z_label in zip(axes, MAT_Z_COLS, MAT_Z_LABELS):
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        xi, yi, zi = x[valid], y[valid], z[valid]

        sc = ax4.scatter(xi, yi, c=np.log10(zi), cmap="viridis", s=40, alpha=0.8)

        cb_ticks, cb_labels = log10_ticks(zi)
        cb = fig4.colorbar(sc, ax=ax4, label=z_label)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(cb_labels)

        set_pow2_axis(ax4, xi, yi)
        ax4.tick_params(axis="both", labelsize=8)
        ax4.set_xlabel(X_COL,   fontsize=9)
        ax4.set_ylabel(Y_COL,   fontsize=9)
        ax4.set_title(z_label,  fontsize=10)
        ax4.grid(True, which="major", linestyle="--", alpha=0.4)

    fig4.suptitle(f"Pareto Front (Mat Level) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path4 = os.path.join(plots_dir, "pareto_mat_2d.png")
    plt.savefig(out_path4, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path4}")
    #plt.show()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python plot_pareto.py <path/to/results.csv>"
    main(sys.argv[1])
