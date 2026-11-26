"""
Q3 Part B - metrics engine

reads   results/raw_timings.csv
writes  results/metrics.csv   (one row per (n, p))

T_S = median of the serial runs for that n
T_P = median of the parallel runs for that (n, p)

    python src/metrics.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ------------------------- the 5 metrics -------------------------
def speedup(t_s, t_p):
    """S = T_S / T_P"""
    return t_s / t_p


def efficiency(s, p):
    """E = S / p"""
    return s / p


def cost(p, t_p):
    """C = p * T_P   (processor-time product)"""
    return p * t_p


def total_overhead(p, t_p, t_s):
    """T_o = p * T_P - T_S   (all time spent by the p processes that the
    serial program does not need)"""
    return p * t_p - t_s


def serial_fraction(s, p):
    """Experimentally determined serial fraction (Amdahl / Karp-Flatt)
        f = (1/S - 1/p) / (1 - 1/p)
    Not defined for p = 1 (division by zero) -> NaN."""
    if p == 1:
        return float("nan")
    return (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
# ------------------------------------------------------------------


def compute(raw):
    """raw_timings DataFrame -> metrics DataFrame (one row per (n, p))."""
    med = raw.groupby(["workload", "n", "p"], as_index=False)["time_sec"].median()
    t_serial = med[med.workload == "serial"].set_index("n")["time_sec"]
    par = med[med.workload == "parallel"].sort_values(["n", "p"])

    rows = []
    for r in par.itertuples(index=False):
        n, p, t_p = int(r.n), int(r.p), float(r.time_sec)
        t_s = float(t_serial[n])
        s = speedup(t_s, t_p)
        rows.append({
            "n": n,
            "p": p,
            "T_S": t_s,
            "T_P": t_p,
            "speedup": s,
            "efficiency": efficiency(s, p),
            "cost": cost(p, t_p),
            "overhead": total_overhead(p, t_p, t_s),
            "serial_fraction": serial_fraction(s, p),
        })
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(RESULTS / "raw_timings.csv")
    m = compute(raw)
    m.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.6f")
    print(m)
    print(f"\nsaved {len(m)} rows -> results/metrics.csv")
    return m


if __name__ == "__main__":
    main()
