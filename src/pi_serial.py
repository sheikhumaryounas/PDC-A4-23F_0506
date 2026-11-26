"""
Q3 Part A - serial Monte-Carlo estimation of pi  (gives T_S)

Throw n random points (x, y) in the unit square and count the ones that land
inside the quarter circle x^2 + y^2 <= 1.  Then  pi ~= 4 * hits / n.

How this was made as fast as possible on ONE core (measured on my laptop,
n = 10^7, median of 5):
  naive numpy float64, x*x + y*y temporaries ........ ~0.11 s
  this version ....................................... ~0.036 s  (~3x faster)

  - numpy vectorised kernel, no python loop over single points
  - float32 random numbers: half the memory traffic of float64 and more than
    enough precision for a hit/miss test
  - x and y are drawn with ONE generator call into one buffer of 2*m values
  - fixed-size blocks of 64K points: the 512 KB buffer stays inside the L2
    cache, which was faster than 1M or 2M point blocks, and memory use is
    constant even for n = 10^8
  - every array is preallocated, all ops are in-place (out=...), so no new
    allocations inside the loop
"""
import time

import numpy as np

BLOCK = 1 << 16          # 65,536 points per block


def count_hits(n, seed=None):
    """Return how many of n random points fall inside the quarter circle."""
    rng = np.random.default_rng(seed)
    bs = max(1, min(BLOCK, n))
    buf = np.empty(2 * bs, dtype=np.float32)     # first half x, second half y
    mask = np.empty(bs, dtype=np.bool_)

    hits = 0
    done = 0
    while done < n:
        m = min(bs, n - done)
        v = buf[: 2 * m]
        rng.random(out=v, dtype=np.float32)      # x_0..x_m-1, y_0..y_m-1
        np.multiply(v, v, out=v)                 # square everything
        xs = v[:m]
        np.add(xs, v[m:], out=xs)                # x^2 + y^2
        np.less_equal(xs, 1.0, out=mask[:m])
        hits += int(np.count_nonzero(mask[:m]))
        done += m
    return hits


def estimate_pi(n, seed=None):
    return 4.0 * count_hits(n, seed) / n


if __name__ == "__main__":
    # quick check:  python src/pi_serial.py
    for n in (10**6, 10**7, 10**8):
        t = time.perf_counter()
        est = estimate_pi(n, seed=1)
        t = time.perf_counter() - t
        print(f"n={n:>12,}  pi~{est:.6f}  |err|={abs(est - np.pi):.2e}  time={t:.4f} s")
