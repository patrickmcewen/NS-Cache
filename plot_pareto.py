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
    "CacheArea_mm2",
    "Leakage_mW",
    "HitDynamicEnergy_nJ",
    "WriteDynamicEnergy_nJ"
]

# ── Plot axes ─────────────────────────────────────────────────────────────────
X_COL    = "Data_MatRows"
Y_COL    = "Data_MatCols"
Z_COLS   = ["HitLatency_ns",     "WriteLatency_ns",    "CacheArea_mm2"]
Z_LABELS = ["Hit Latency (ns)",  "Write Latency (ns)", "Cache Area (mm²)"]

# ── Mat-level plot axes ────────────────────────────────────────────────────────
MAT_Z_COLS   = ["Data_ReadLatency_ns",   "Data_WriteLatency_ns",   "Data_MatArea_mm2"]
MAT_Z_LABELS = ["Mat Read Latency (ns)", "Mat Write Latency (ns)", "Mat Area (mm²)"]

# ── Cache-level energy/power plot axes ────────────────────────────────────────
CACHE_ENERGY_Z_COLS   = ["HitDynamicEnergy_nJ",    "WriteDynamicEnergy_nJ",    "Leakage_mW"]
CACHE_ENERGY_Z_LABELS = ["Hit Dynamic Energy (nJ)", "Write Dynamic Energy (nJ)", "Cache Leakage Power (mW)"]

# ── Mat-level energy/power plot axes ──────────────────────────────────────────
MAT_ENERGY_Z_COLS   = ["Data_ReadDynamicEnergy_pJ",   "Data_WriteDynamicEnergy_pJ",   "Data_Leakage_mW"]
MAT_ENERGY_Z_LABELS = ["Mat Read Energy (pJ)",         "Mat Write Energy (pJ)",         "Mat Leakage Power (mW)"]

# ── Refresh energy + cache availability plot axes ─────────────────────────────
REFRESH_Z_COLS     = ["RefreshDynamicEnergy_nJ",      "CacheAvailability_pct"]
REFRESH_Z_LABELS   = ["Refresh Dynamic Energy (nJ)",  "Cache Availability (%)"]
REFRESH_MINIMIZE   = [True,                            False]   # availability is maximized
REFRESH_LOG_SCALE  = [True,                            False]   # availability is linear (percentage)


def uniform_ticks(data, log_scale, n=5):
    """n evenly spaced ticks across the data range, in the plot coordinate space."""
    if log_scale:
        lmin, lmax = np.log10(data.min()), np.log10(data.max())
        positions = np.linspace(lmin, lmax, n)
        labels = []
        for p in positions:
            v = 10 ** p
            if abs(v) >= 1e4 or abs(v) < 1e-2:
                labels.append(f"{v:.2e}")
            else:
                labels.append(f"{v:.3g}")
    else:
        positions = np.linspace(data.min(), data.max(), n)
        labels = [f"{v:.3g}" for v in positions]
    return positions, labels


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
        sc = ax.scatter(lx, ly, lz, c=lz, cmap="viridis_r", s=20, alpha=0.7)
        bi = np.argmin(lz)
        ax.scatter([lx[bi]], [ly[bi]], [lz[bi]], c="red", marker="*", s=120, zorder=5)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = uniform_ticks(z, log_scale=True)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(zticks); cb.set_ticklabels(zlabels)

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

        ticks, labels = uniform_ticks(zi, log_scale=True)
        sc = ax2.scatter(xi, yi, c=np.log10(zi), cmap="viridis_r", s=40, alpha=0.8)
        bi = np.argmin(zi)
        ax2.scatter([xi[bi]], [yi[bi]], c="red", marker="*", s=200, zorder=5)

        cb = fig2.colorbar(sc, ax=ax2, label=z_label)
        cb.set_ticks(ticks); cb.set_ticklabels(labels)

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
        sc = ax.scatter(lx, ly, lz, c=lz, cmap="viridis_r", s=20, alpha=0.7)
        bi = np.argmin(lz)
        ax.scatter([lx[bi]], [ly[bi]], [lz[bi]], c="red", marker="*", s=120, zorder=5)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = uniform_ticks(z, log_scale=True)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        cb = fig3.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(zticks); cb.set_ticklabels(zlabels)

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

        ticks, labels = uniform_ticks(zi, log_scale=True)
        sc = ax4.scatter(xi, yi, c=np.log10(zi), cmap="viridis_r", s=40, alpha=0.8)
        bi = np.argmin(zi)
        ax4.scatter([xi[bi]], [yi[bi]], c="red", marker="*", s=200, zorder=5)

        cb = fig4.colorbar(sc, ax=ax4, label=z_label)
        cb.set_ticks(ticks); cb.set_ticklabels(labels)

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

    # ── Cache-level energy/power 3D plots ─────────────────────────────────────
    fig5 = plt.figure(figsize=(18, 6))

    for idx, (z_col, z_label) in enumerate(zip(CACHE_ENERGY_Z_COLS, CACHE_ENERGY_Z_LABELS)):
        ax = fig5.add_subplot(1, 3, idx + 1, projection="3d")

        x = pareto[X_COL].to_numpy(dtype=float)
        y = pareto[Y_COL].to_numpy(dtype=float)
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        x, y, z = x[valid], y[valid], z[valid]

        lx, ly, lz = np.log10(x), np.log10(y), np.log10(z)
        sc = ax.scatter(lx, ly, lz, c=lz, cmap="viridis_r", s=20, alpha=0.7)
        bi = np.argmin(lz)
        ax.scatter([lx[bi]], [ly[bi]], [lz[bi]], c="red", marker="*", s=120, zorder=5)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = uniform_ticks(z, log_scale=True)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        cb = fig5.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(zticks); cb.set_ticklabels(zlabels)

    fig5.suptitle(f"Pareto Front (Cache Energy/Power) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path5 = os.path.join(plots_dir, "pareto_cache_energy_3d.png")
    plt.savefig(out_path5, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path5}")

    # ── Cache-level energy/power 2D plots ─────────────────────────────────────
    fig6, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = pareto[X_COL].to_numpy(dtype=float)
    y = pareto[Y_COL].to_numpy(dtype=float)

    for ax6, z_col, z_label in zip(axes, CACHE_ENERGY_Z_COLS, CACHE_ENERGY_Z_LABELS):
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        xi, yi, zi = x[valid], y[valid], z[valid]

        ticks, labels = uniform_ticks(zi, log_scale=True)
        sc = ax6.scatter(xi, yi, c=np.log10(zi), cmap="viridis_r", s=40, alpha=0.8)
        bi = np.argmin(zi)
        ax6.scatter([xi[bi]], [yi[bi]], c="red", marker="*", s=200, zorder=5)

        cb = fig6.colorbar(sc, ax=ax6, label=z_label)
        cb.set_ticks(ticks); cb.set_ticklabels(labels)

        set_pow2_axis(ax6, xi, yi)
        ax6.tick_params(axis="both", labelsize=8)
        ax6.set_xlabel(X_COL,   fontsize=9)
        ax6.set_ylabel(Y_COL,   fontsize=9)
        ax6.set_title(z_label,  fontsize=10)
        ax6.grid(True, which="major", linestyle="--", alpha=0.4)

    fig6.suptitle(f"Pareto Front (Cache Energy/Power) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path6 = os.path.join(plots_dir, "pareto_cache_energy_2d.png")
    plt.savefig(out_path6, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path6}")

    # ── Mat-level energy/power 3D plots ───────────────────────────────────────
    fig7 = plt.figure(figsize=(18, 6))

    for idx, (z_col, z_label) in enumerate(zip(MAT_ENERGY_Z_COLS, MAT_ENERGY_Z_LABELS)):
        ax = fig7.add_subplot(1, 3, idx + 1, projection="3d")

        x = pareto[X_COL].to_numpy(dtype=float)
        y = pareto[Y_COL].to_numpy(dtype=float)
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        x, y, z = x[valid], y[valid], z[valid]

        lx, ly, lz = np.log10(x), np.log10(y), np.log10(z)
        sc = ax.scatter(lx, ly, lz, c=lz, cmap="viridis_r", s=20, alpha=0.7)
        bi = np.argmin(lz)
        ax.scatter([lx[bi]], [ly[bi]], [lz[bi]], c="red", marker="*", s=120, zorder=5)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = uniform_ticks(z, log_scale=True)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        cb = fig7.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(zticks); cb.set_ticklabels(zlabels)

    fig7.suptitle(f"Pareto Front (Mat Energy/Power) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path7 = os.path.join(plots_dir, "pareto_mat_energy_3d.png")
    plt.savefig(out_path7, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path7}")

    # ── Mat-level energy/power 2D plots ───────────────────────────────────────
    fig8, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = pareto[X_COL].to_numpy(dtype=float)
    y = pareto[Y_COL].to_numpy(dtype=float)

    for ax8, z_col, z_label in zip(axes, MAT_ENERGY_Z_COLS, MAT_ENERGY_Z_LABELS):
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        xi, yi, zi = x[valid], y[valid], z[valid]

        ticks, labels = uniform_ticks(zi, log_scale=True)
        sc = ax8.scatter(xi, yi, c=np.log10(zi), cmap="viridis_r", s=40, alpha=0.8)
        bi = np.argmin(zi)
        ax8.scatter([xi[bi]], [yi[bi]], c="red", marker="*", s=200, zorder=5)

        cb = fig8.colorbar(sc, ax=ax8, label=z_label)
        cb.set_ticks(ticks); cb.set_ticklabels(labels)

        set_pow2_axis(ax8, xi, yi)
        ax8.tick_params(axis="both", labelsize=8)
        ax8.set_xlabel(X_COL,   fontsize=9)
        ax8.set_ylabel(Y_COL,   fontsize=9)
        ax8.set_title(z_label,  fontsize=10)
        ax8.grid(True, which="major", linestyle="--", alpha=0.4)

    fig8.suptitle(f"Pareto Front (Mat Energy/Power) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path8 = os.path.join(plots_dir, "pareto_mat_energy_2d.png")
    plt.savefig(out_path8, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path8}")

    # ── Refresh energy / cache availability 3D plots ──────────────────────────
    fig9 = plt.figure(figsize=(12, 6))

    for idx, (z_col, z_label, minimize, log_scale) in enumerate(
            zip(REFRESH_Z_COLS, REFRESH_Z_LABELS, REFRESH_MINIMIZE, REFRESH_LOG_SCALE)):
        ax = fig9.add_subplot(1, 2, idx + 1, projection="3d")

        x = pareto[X_COL].to_numpy(dtype=float)
        y = pareto[Y_COL].to_numpy(dtype=float)
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        x, y, z = x[valid], y[valid], z[valid]

        lx, ly = np.log10(x), np.log10(y)
        zplot  = np.log10(z) if log_scale else z
        cmap   = "viridis_r" if minimize else "viridis"
        sc = ax.scatter(lx, ly, zplot, c=zplot, cmap=cmap, s=20, alpha=0.7)
        bi = np.argmin(zplot) if minimize else np.argmax(zplot)
        ax.scatter([lx[bi]], [ly[bi]], [zplot[bi]], c="red", marker="*", s=120, zorder=5)

        xvals, xlabels = pow2_ticks(x)
        yvals, ylabels = pow2_ticks(y)
        zticks, zlabels = uniform_ticks(z, log_scale=log_scale)
        ax.set_xticks(np.log10(xvals)); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.log10(yvals)); ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);          ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        cb = fig9.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(zticks); cb.set_ticklabels(zlabels)

    fig9.suptitle(f"Pareto Front (Refresh / Availability) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path9 = os.path.join(plots_dir, "pareto_refresh_avail_3d.png")
    plt.savefig(out_path9, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path9}")

    # ── Refresh energy / cache availability 2D plots ──────────────────────────
    fig10, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = pareto[X_COL].to_numpy(dtype=float)
    y = pareto[Y_COL].to_numpy(dtype=float)

    for ax10, z_col, z_label, minimize, log_scale in zip(
            axes, REFRESH_Z_COLS, REFRESH_Z_LABELS, REFRESH_MINIMIZE, REFRESH_LOG_SCALE):
        z = pareto[z_col].to_numpy(dtype=float)
        valid = (x > 0) & (y > 0) & (z > 0)
        xi, yi, zi = x[valid], y[valid], z[valid]

        ticks, labels = uniform_ticks(zi, log_scale=log_scale)
        cmap = "viridis_r" if minimize else "viridis"
        cval = np.log10(zi) if log_scale else zi
        sc = ax10.scatter(xi, yi, c=cval, cmap=cmap, s=40, alpha=0.8)
        bi = np.argmin(zi) if minimize else np.argmax(zi)
        ax10.scatter([xi[bi]], [yi[bi]], c="red", marker="*", s=200, zorder=5)

        cb = fig10.colorbar(sc, ax=ax10, label=z_label)
        cb.set_ticks(ticks); cb.set_ticklabels(labels)

        set_pow2_axis(ax10, xi, yi)
        ax10.tick_params(axis="both", labelsize=8)
        ax10.set_xlabel(X_COL,   fontsize=9)
        ax10.set_ylabel(Y_COL,   fontsize=9)
        ax10.set_title(z_label,  fontsize=10)
        ax10.grid(True, which="major", linestyle="--", alpha=0.4)

    fig10.suptitle(f"Pareto Front (Refresh / Availability) — {cfg_name}", fontsize=13)
    plt.tight_layout()
    out_path10 = os.path.join(plots_dir, "pareto_refresh_avail_2d.png")
    plt.savefig(out_path10, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path10}")
    #plt.show()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python plot_pareto.py <path/to/results.csv>"
    main(sys.argv[1])
