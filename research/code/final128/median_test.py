import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
print(pd.__version__, np.__version__)
rng = np.random.default_rng(1)
def roll_med(x, w=9, minp=3):
    h = w // 2
    xp = np.r_[np.full(h, np.nan), x, np.full(h, np.nan)]
    W = sliding_window_view(xp, w)
    cnt = np.isfinite(W).sum(1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmedian(W, 1)
    return np.where(cnt >= minp, m, np.nan)
bad_r = bad_g = bad_t = 0
for t in range(3000):
    nb = rng.integers(1, 60)
    x = rng.random(nb) * rng.choice([1, 0.01, 100, 3.3])
    if rng.random() < .5: x = np.round(x * 10) / 10
    x[rng.random(nb) < 0.3] = np.nan
    r1 = pd.Series(x).rolling(9, center=True, min_periods=3).median().to_numpy()
    r2 = roll_med(x)
    if not np.array_equal(r1, r2, equal_nan=True): bad_r += 1
    y = x[np.isfinite(x)]
    if len(y):
        g = pd.DataFrame({"d": 0, "v": y}).groupby("d").v.median().iat[0]
        if not (g == np.median(y)): bad_g += 1
        tr = pd.DataFrame({"d": 0, "v": y}).groupby("d").v.transform("median").iat[0]
        if not (tr == np.median(y)): bad_t += 1
print("rolling bad", bad_r, "group median bad", bad_g, "transform median bad", bad_t)
