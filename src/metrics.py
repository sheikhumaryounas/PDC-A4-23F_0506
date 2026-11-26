"""
Q3 Part B - metrics engine

reads   results/raw_timings.csv
writes  results/metrics.csv   (one row per (n, p))

T_S = median of the serial runs for that n
T_P = median of the parallel runs for that (n, p)

    python src/metrics.py
"""
from pathlib import Path

import numpy as np
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


def format_table(m):
    """Fixed-width text table, one block of rows per problem size n."""
    head = (f"{'n':>13} {'p':>3} {'T_S (s)':>10} {'T_P (s)':>10} {'Speedup':>8} "
            f"{'Effic.':>7} {'Cost (s)':>10} {'T_o (s)':>10} {'f (Amdahl)':>11}")
    line = "-" * len(head)
    out = [line, head, line]
    last_n = None
    for r in m.itertuples(index=False):
        if last_n is not None and r.n != last_n:
            out.append(line)
        last_n = r.n
        f = "-" if pd.isna(r.serial_fraction) else f"{r.serial_fraction:.4f}"
        out.append(f"{r.n:>13,} {r.p:>3} {r.T_S:>10.4f} {r.T_P:>10.4f} {r.speedup:>8.3f} "
                   f"{r.efficiency:>7.3f} {r.cost:>10.4f} {r.overhead:>10.4f} {f:>11}")
    out.append(line)
    return "\n".join(out)


# ------------------- bonus: iso-efficiency estimate -------------------
E_TARGET = 0.6


def isoefficiency(m, target=E_TARGET):
    """How large must n be so that E >= target for each p?

    E = T_S / (p*T_P) = T_S / (T_S + T_o), so
        E >= target   <=>   T_o <= k * T_S   with  k = 1/target - 1
    Straight-line models fitted to the measurements:
        T_S(n)    ~ c0 + c1*n        (serial work grows linearly with n)
        T_o(n, p) ~ a_p + b_p*n      (fixed cost per parallel job + part that
                                      grows with n, e.g. memory-bandwidth sharing)
    Solve  a_p + b_p*n <= k*(c0 + c1*n)  ->  n >= (a_p - k*c0) / (k*c1 - b_p).
    The fits use weights 1/y so the small sizes are not drowned by n = 10^8.
    If b_p >= k*c1 the overhead grows as fast as the useful work and no n is
    big enough. E_inf = c1 / (c1 + b_p) is the efficiency for n -> infinity.
    Returns (c0, c1, [(p, n_min, a_p, b_p, E_inf), ...]).
    """
    k = 1.0 / target - 1.0
    ts = m.drop_duplicates("n")
    c1, c0 = np.polyfit(ts.n, ts.T_S, 1, w=1.0 / ts.T_S)

    res = []
    for p, g in m[m.p > 1].groupby("p"):
        if g.n.nunique() < 2:
            continue
        to = g.overhead.clip(lower=1e-6)        # ~0 or slightly negative -> ~0
        b, a = np.polyfit(g.n, to, 1, w=1.0 / to)
        e_inf = c1 / (c1 + b) if (c1 + b) > 0 else float("nan")
        denom = k * c1 - b
        n_min = float("inf") if denom <= 0 else max((a - k * c0) / denom, 0.0)
        res.append((int(p), n_min, a, b, e_inf))
    return c0, c1, res


def format_isoefficiency(m, target=E_TARGET):
    c0, c1, res = isoefficiency(m, target)
    out = [f"\nBonus - problem size needed to keep E >= {target} as p doubles",
           f"  model:  T_S(n) ~ {c0:.3e} + {c1:.3e}*n s,   T_o(n,p) ~ a_p + b_p*n s"]

    # what the measurements themselves say (smallest measured n with E >= target)
    for p in sorted(m.p.unique()):
        if p == 1:
            continue
        ok = m[(m.p == p) & (m.efficiency >= target)]
        seen = f"{ok.n.min():,}" if len(ok) else "none of the measured sizes"
        out.append(f"  p={p}: measured E >= {target} first at n = {seen}")

    prev = None
    for p, n_min, a, b, e_inf in res:
        if np.isinf(n_min):
            txt = f"not reachable - E_inf = {e_inf:.2f} < {target}, overhead grows as fast as the work"
        elif n_min == 0:
            txt = "fixed overhead is negligible, every n works"
        else:
            txt = f"n >= {n_min:,.0f}"
            if prev and np.isfinite(prev):
                txt += f"   ({n_min / prev:.1f}x the size needed at p={p // 2})"
        out.append(f"  p={p}: a_p={a:.2e} s, b_p={b:.2e} s/sample, E_inf={e_inf:.2f}  ->  {txt}")
        prev = n_min if n_min > 0 else None
    return "\n".join(out)
# -----------------------------------------------------------------------


def main():
    raw = pd.read_csv(RESULTS / "raw_timings.csv")
    m = compute(raw)
    m.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.6f")
    print("\nPerformance metrics (T_S and T_P are medians of the runs)")
    print(format_table(m))
    print(format_isoefficiency(m))
    print(f"\nsaved {len(m)} rows -> results/metrics.csv")
    return m


if __name__ == "__main__":
    main()
