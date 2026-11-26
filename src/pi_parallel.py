"""
Q3 Part A - parallel Monte-Carlo pi with Python multiprocessing

Decomposition (data parallel, like an MPI scatter + reduce):
  1. split the n samples into p parts whose sizes differ by at most 1
  2. every worker process runs the SAME kernel as the serial program
     (pi_serial.count_hits) on its part, with its own random stream
  3. the parent adds the p hit counts together (reduction) -> pi

Independent random streams: numpy SeedSequence(seed).spawn(p) gives p child
seeds that are statistically independent, so two workers never generate the
same sequence of points.

Pool start-up: on Windows multiprocessing uses the "spawn" start method, so
each worker is a fresh Python interpreter that has to import numpy again
(hundreds of ms). The pool is therefore created ONCE per p, before timing
starts - the same way mpiexec starts MPI ranks once - and T_P only measures
distribute -> compute -> reduce. The start-up cost is measured separately
by start_pool() and reported as an extra overhead in the README.
"""
import time
from multiprocessing import Pool

import numpy as np

from pi_serial import count_hits


def _worker(task):
    """Runs inside a worker process: task = (number of samples, SeedSequence)."""
    n_part, seed = task
    return count_hits(n_part, seed)


def split_work(n, p):
    """Split n samples into p chunk sizes that differ by at most 1."""
    base, extra = divmod(n, p)
    return [base + (1 if i < extra else 0) for i in range(p)]


def parallel_hits(pool, n, p, seed=12345):
    """Total hits for n samples using p workers of an already running pool."""
    parts = split_work(n, p)
    streams = np.random.SeedSequence(seed).spawn(p)
    # chunksize=1 -> exactly one task per worker, no extra batching
    partial = pool.map(_worker, zip(parts, streams), chunksize=1)
    return sum(partial)


def estimate_pi_parallel(pool, n, p, seed=12345):
    return 4.0 * parallel_hits(pool, n, p, seed) / n


def start_pool(p):
    """Start p workers and wait until every one of them has imported numpy and
    run the kernel once. Returns (pool, start-up seconds)."""
    t0 = time.perf_counter()
    pool = Pool(processes=p)
    pool.map(_worker, [(1000, i) for i in range(p)], chunksize=1)   # warm-up
    return pool, time.perf_counter() - t0


if __name__ == "__main__":
    # quick check:  python src/pi_parallel.py
    n = 10**7
    for p in (1, 2, 4, 8):
        pool, t_up = start_pool(p)
        with pool:
            t0 = time.perf_counter()
            est = estimate_pi_parallel(pool, n, p)
            dt = time.perf_counter() - t0
        print(f"p={p}  n={n:,}  pi~{est:.6f}  T_P={dt:.4f} s  (pool start-up {t_up:.2f} s)")
