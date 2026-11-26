"""
Q3 Part C - automated visualisation

reads   results/metrics.csv
writes  figures/
          1_time_vs_p.png        (i)   execution time T_P vs p, one line per n
          2_speedup_vs_p.png     (ii)  speedup vs p + ideal linear speedup (dashed)
          3_efficiency_vs_p.png  (iii) efficiency vs p, one line per n
          4_overhead_vs_p.png    (iv)  total overhead T_o vs p
          5_dashboard.png        (v)   all four as a 2x2 dashboard

    python src/plot.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # files only, no window
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# fixed colour + marker per problem size (colour-blind checked; the marker
# shape is a second cue so the lines are never told apart by colour alone)
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
MARKERS = ["o", "s", "^", "D", "v"]
REF = "#555555"                          # reference lines (ideal, E = 1, 0)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "legend.frameon": False,
})


def n_label(n):
    n = int(n)
    exp = len(str(n)) - 1
    return f"n = $10^{{{exp}}}$" if n == 10 ** exp else f"n = {n:,}"


def _lines_per_n(ax, m, col):
    for i, (n, g) in enumerate(m.groupby("n")):
        ax.plot(g.p, g[col], color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)], label=n_label(n))
    ax.set_xticks(sorted(m.p.unique()))
    ax.set_xlabel("Number of worker processes p")


def plot_time(ax, m):
    _lines_per_n(ax, m, "T_P")
    ax.set_yscale("log")
    ax.set_ylabel("Parallel execution time T_P (s, log scale)")
    ax.set_title("(i) Execution time vs p")
    ax.legend()


def plot_speedup(ax, m):
    _lines_per_n(ax, m, "speedup")
    ps = sorted(m.p.unique())
    ax.plot(ps, ps, ls="--", color=REF, lw=1.5, marker="", label="ideal linear (S = p)")
    ax.set_ylabel("Speedup S = T_S / T_P (dimensionless)")
    ax.set_title("(ii) Speedup vs p")
    ax.legend()


def plot_efficiency(ax, m):
    _lines_per_n(ax, m, "efficiency")
    ax.axhline(1.0, color=REF, ls=":", lw=1.2, label="ideal (E = 1)")
    ax.set_ylim(0, max(1.1, m.efficiency.max() * 1.1))
    ax.set_ylabel("Efficiency E = S / p (dimensionless)")
    ax.set_title("(iii) Efficiency vs p")
    ax.legend()


def plot_overhead(ax, m):
    _lines_per_n(ax, m, "overhead")
    ax.axhline(0.0, color=REF, ls=":", lw=1.2)
    ax.set_ylabel("Total overhead T_o = p*T_P - T_S (s)")
    ax.set_title("(iv) Total overhead vs p")
    ax.legend()


PLOTS = [
    ("1_time_vs_p.png", plot_time),
    ("2_speedup_vs_p.png", plot_speedup),
    ("3_efficiency_vs_p.png", plot_efficiency),
    ("4_overhead_vs_p.png", plot_overhead),
]


def main():
    m = pd.read_csv(RESULTS / "metrics.csv").sort_values(["n", "p"])
    FIGS.mkdir(exist_ok=True)
    for fname, draw in PLOTS:
        fig, ax = plt.subplots(figsize=(7, 5))
        draw(ax, m)
        fig.tight_layout()
        fig.savefig(FIGS / fname, dpi=130)
        plt.close(fig)
        print(f"saved figures/{fname}")
    dashboard(m)


def dashboard(m):
    """(v) all four plots as subplots of one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, (_, draw) in zip(axes.flat, PLOTS):
        draw(ax, m)
    fig.suptitle("Monte-Carlo pi: serial vs multiprocessing performance "
                 "(medians of 5 runs)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGS / "5_dashboard.png", dpi=130)
    plt.close(fig)
    print("saved figures/5_dashboard.png")


if __name__ == "__main__":
    main()
