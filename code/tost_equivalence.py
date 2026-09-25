"""Two one-sided tests (TOST) for equivalence of the three human-factor
representations in Section VIII-L, computed from data/e1_representation.csv.

Equivalence margin (smallest difference of interest), fixed by rule rather than
by inspecting the result: half of the gain that human-factor adjustment
achieves over severity-only on the same metric and seeds.
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats as SS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
if not os.path.exists(os.path.join(DATA, 'e1_representation.csv')):
    raise SystemExit('data/e1_representation.csv not found. Run exp_extra.py first, or run reproduce_all.py.')
e1 = pd.read_csv(os.path.join(DATA, 'e1_representation.csv'))

def tost(d, margin):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p_lo = 1 - SS.t.cdf((m + margin) / se, n - 1)   # H0: diff <= -margin
    p_hi = SS.t.cdf((m - margin) / se, n - 1)       # H0: diff >= +margin
    ci90 = SS.t.interval(0.90, n - 1, loc=m, scale=se)
    return dict(n=n, mean_diff=float(m), margin=float(margin),
                p_tost=float(max(p_lo, p_hi)), ci90=[float(ci90[0]), float(ci90[1])],
                equivalent=bool(max(p_lo, p_hi) < 0.05))

out = {}
for metric in ['auc', 'band_auc']:
    piv = e1.pivot(index='seed', columns='model', values=metric)
    gain = (piv['M4 ABW agent'] - piv['M1 severity-only']).mean()
    margin = 0.5 * gain
    out[metric] = dict(
        severity_gain=float(gain),
        abw_vs_learned=tost(piv['M4 ABW agent'] - piv['M3 learned feature fusion'], margin),
        abw_vs_posthoc=tost(piv['M4 ABW agent'] - piv['M2 post-hoc adjustment'], margin))

with open(os.path.join(DATA, 'e1_equivalence.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
