import numpy as np, pandas as pd, sys
print(pd.__version__, np.__version__)
def ksum(M):
    """Kahan sums down axis 0 (rows in order), NaN skipped; returns (sum, count) like pandas groupby sum/count"""
    s = np.zeros(M.shape[1]); c = np.zeros(M.shape[1]); n = np.zeros(M.shape[1], np.int64)
    for row in M:
        ok = ~np.isnan(row)
        y = row - c
        t = s + y
        cn = (t - s) - y
        cn = np.where(cn != cn, 0.0, cn)
        s = np.where(ok, t, s); c = np.where(ok, cn, c); n += ok
    return s, n
rng = np.random.default_rng(0)
bad = 0
for trial in range(3000):
    k = rng.integers(1, 7); nb = rng.integers(1, 50)
    M = rng.random((k, nb)) * rng.choice([1, 1e-3, 1e3, 0.37])
    M[rng.random((k, nb)) < 0.25] = np.nan
    df = pd.DataFrame({"g": np.tile(np.arange(nb), k), "v": M.ravel()})
    g = df.groupby("g").v.agg(["sum", "count"])
    s, n = ksum(M)
    if not (np.array_equal(g["sum"].to_numpy(), s) and np.array_equal(g["count"].to_numpy(), n)):
        bad += 1
    # mean
    gm = df.groupby("g").v.mean().to_numpy()
    mm = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    if not np.array_equal(gm, mm, equal_nan=True):
        bad += 1000
print("bad", bad)
