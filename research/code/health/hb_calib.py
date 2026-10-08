"""Note 38: rule-limit calibration on real windows (presumed-healthy vs weak positives)."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H, health_core as hc, pandas as pd, numpy as np

def degraded(R):
    f = R[R.window.eq('full')].pivot_table(index=['DeviceId', 'detector'], columns='period',
                                            values=['n_on', 'share'])
    f.columns = [f'{a}_{b}' for a, b in f.columns]
    f = f.dropna()
    ratio = f.share_stg / f.share_dec.clip(lower=1e-6)
    d = f[(f.n_on_dec >= 500) & (f.n_on_stg > 0) & (ratio < 0.2)]
    return set(d.index), ratio

def table(R, W):
    s = R[R.period.eq('stg')].copy()
    s = hb_rescore(s)
    s = s.merge(W, on=['DeviceId', 'detector'], how='left')
    return s

def hb_rescore(s):
    out = []
    for w, g in s.groupby('window'):
        st = H.EVAL_WIN['stg'][w][0]
        out.append(H.rescore(g, st).assign(window=w))
    return pd.concat(out, ignore_index=True)

def summary(s, deg):
    s['wl_degraded'] = [(d, k) in deg for d, k in zip(s.DeviceId, s.detector)]
    s['flag'] = s.status.isin(['bad', 'suspect'])
    s['badf'] = s.status.eq('bad')
    s['ned'] = s.status.eq('not_enough_data')
    groups = {'presumed healthy': s.presumed_healthy.eq(True),
              'dead (print, listed)': s.wl_dead_print.eq(True),
              'dead since Dec 2024': s.wl_dead_since_dec.eq(True),
              'card both dead': s.wl_card_dead.eq(True),
              'card erratic': s.wl_card_erratic.eq(True),
              'dq health fail': s.wl_dq_health.eq(True),
              'label-check health': s.wl_lc_health.eq(True),
              'degraded vs Dec 2024': s.wl_degraded}
    rows = []
    for g, m in groups.items():
        for w, x in s[m].groupby('window'):
            rows.append(dict(group=g, window=w, n=len(x), flag=x.flag.mean(), bad=x.badf.mean(), ned=x.ned.mean()))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    R = pd.read_parquet(H.HB / 'real_stats.parquet')
    folds = pd.read_csv(H.DCW / 'folds_v4.csv'); R = R[R.DeviceId.isin(folds.DeviceId.str.lower())]
    W = pd.read_parquet(H.HB / 'weak_labels.parquet')
    deg, _ = degraded(R)
    s = table(R, W)
    t = summary(s, deg)
    print(t.pivot_table(index='group', columns='window', values='flag').round(3)[['2h_a','2h_b','2h_c','6h_a','6h_b','24h_a','24h_b','full']])
    print(t.pivot_table(index='group', columns='window', values='n').iloc[:, :1])
    ph = s[s.presumed_healthy.eq(True)]
    sc = [c for c in s if c.startswith('s_')]
    print(ph.groupby('window')[sc].agg(lambda x: (x > 0).mean()).round(4))
    s.to_parquet(H.HB / 'rules_real.parquet')
