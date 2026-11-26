"""
Q3 - Parallel Performance Analyzer (driver)

    python src/main.py

Times the serial version (T_S) and the parallel version (T_P) for
n = 10^6, 10^7, 10^8 and p = 1, 2, 4, 8. Every configuration is run 5 times
and every single run is written to results/raw_timings.csv
(columns: workload, n, p, run, time_sec). Serial rows have p = 1.

Options (mainly for quick tests):
    python src/main.py --runs 2 --sizes 1000000 10000000 --procs 1 2
"""
import argparse
import csv
import os
import time
from pathlib import Path

from pi_serial import count_hits
from pi_parallel import estimate_pi_parallel, start_pool

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DEFAULT_SIZES = [10**6, 10**7, 10**8]
DEFAULT_PROCS = [1, 2, 4, 8]
DEFAULT_RUNS = 5


def time_serial(n, run):
    t0 = time.perf_counter()
    hits = count_hits(n, seed=run)
    dt = time.perf_counter() - t0
    print(f"  serial    n={n:>12,}        run {run}  {dt:8.4f} s   pi~{4 * hits / n:.5f}")
    return ("serial", n, 1, run, dt)


def time_parallel(pool, n, p, run):
    t0 = time.perf_counter()
    est = estimate_pi_parallel(pool, n, p, seed=1000 * p + run)
    dt = time.perf_counter() - t0
    print(f"  parallel  n={n:>12,}  p={p}  run {run}  {dt:8.4f} s   pi~{est:.5f}")
    return ("parallel", n, p, run, dt)


def benchmark(sizes, procs, runs):
    """Runs are done in rounds: in each round every serial n and every (n, p)
    is timed once. If the CPU clock changes during the benchmark (turbo,
    thermal throttling, power plan) the change hits all configurations the
    same way instead of only the ones that happened to run last.
    The pools that are not in use just block on their task queue (no CPU)."""
    pools, startup = {}, []
    for p in procs:
        pools[p], t_up = start_pool(p)
        startup.append((p, t_up))
        print(f"  [pool with {p} worker(s) started in {t_up:.3f} s]")

    count_hits(10**6, seed=0)          # warm up the main process as well
    rows = []
    try:
        for run in range(1, runs + 1):
            print(f"-- round {run}/{runs}")
            for n in sizes:
                rows.append(time_serial(n, run))
                for p in procs:
                    rows.append(time_parallel(pools[p], n, p, run))
    finally:
        for pool in pools.values():
            pool.terminate()
            pool.join()
    return rows, startup


def save_raw(rows, path):
    rows = sorted(rows, key=lambda r: (r[0] != "serial", r[2], r[1], r[3]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "n", "p", "run", "time_sec"])
        for workload, n, p, run, t in rows:
            w.writerow([workload, n, p, run, f"{t:.6f}"])


def save_startup(startup, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "pool_startup_sec"])
        for p, t in startup:
            w.writerow([p, f"{t:.4f}"])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Serial vs parallel Monte-Carlo pi benchmark")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--procs", type=int, nargs="+", default=DEFAULT_PROCS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    print(f"Monte-Carlo pi | sizes={args.sizes} procs={args.procs} runs={args.runs} "
          f"| logical CPUs={os.cpu_count()}")

    t0 = time.perf_counter()
    rows, startup = benchmark(args.sizes, args.procs, args.runs)
    save_raw(rows, RESULTS / "raw_timings.csv")
    save_startup(startup, RESULTS / "pool_startup.csv")
    print(f"\nsaved {len(rows)} rows -> results/raw_timings.csv "
          f"(benchmark took {time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
