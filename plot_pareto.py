import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def log10_ticks(data):
    """Ticks at integer powers of 10, labelled as 10^x."""
    lo = int(np.floor(np.log10(data.min())))
    hi = int(np.ceil(np.log10(data.max())))
    ticks = np.arange(lo, hi + 1, dtype=float)
    labels = [f"$10^{{{int(t)}}}$" for t in ticks]
    return ticks, labels


def pow2_ticks(data):
    """Ticks at powers of 2 present in the data."""
    unique = np.unique(data.astype(int))
    powers = unique[unique > 0]
    logs = np.log2(powers)
    # Keep only exact powers of 2
    exact = powers[np.abs(logs - np.round(logs)) < 1e-6]
    ticks = np.log10(exact.astype(float))
    labels = [str(int(v)) for v in exact]
    return ticks, labels


def main(csv_path):
    df = pd.read_csv(csv_path)
    for col in PARETO_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=PARETO_METRICS, inplace=True)
    print(f"Loaded {len(df)} rows from {csv_path}")

    mask = paretoset(df[PARETO_METRICS], sense=["min"] * len(PARETO_METRICS))
    pareto = df[mask]
    print(f"Pareto front: {len(pareto)} designs")

    pareto_csv = csv_path.replace(".csv", "_pareto.csv")
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

        xticks, xlabels = pow2_ticks(x)
        yticks, ylabels = pow2_ticks(y)
        zticks, zlabels = log10_ticks(z)
        ax.set_xticks(xticks);  ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(yticks);  ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_zticks(zticks);  ax.set_zticklabels(zlabels, fontsize=7)

        ax.set_xlabel(X_COL,   fontsize=8, labelpad=8)
        ax.set_ylabel(Y_COL,   fontsize=8, labelpad=8)
        ax.set_zlabel(z_label, fontsize=8, labelpad=8)
        ax.set_title(z_label,  fontsize=10)

        # Colorbar: show actual values as 10^x ticks
        cb_ticks, cb_labels = log10_ticks(z)
        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=z_label)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels(cb_labels)

    fig.suptitle(f"Pareto Front — {csv_path.split('/')[-1]}", fontsize=13)
    plt.tight_layout()
    out_path = csv_path.replace(".csv", "_pareto_3d.png")
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

        xticks, xlabels = pow2_ticks(xi)
        yticks, ylabels = pow2_ticks(yi)
        ax2.set_xticks(np.power(10.0, xticks))
        ax2.set_xticklabels(xlabels, fontsize=8)
        ax2.set_yticks(np.power(10.0, yticks))
        ax2.set_yticklabels(ylabels, fontsize=8)
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        ax2.set_xlabel(X_COL,   fontsize=9)
        ax2.set_ylabel(Y_COL,   fontsize=9)
        ax2.set_title(z_label,  fontsize=10)
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)

    fig2.suptitle(f"Pareto Front — {csv_path.split('/')[-1]}", fontsize=13)
    plt.tight_layout()
    out_path2 = csv_path.replace(".csv", "_pareto_2d.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.show()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python plot_pareto.py <path/to/results.csv>"
    main(sys.argv[1])
