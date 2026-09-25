"""Per-class detail for the confounding experiment (Section VIII-M, Table 13,
Fig. 12). Writes data/e2_confounding_detail.csv and the 'e2_detail' entry of
data/results.json. Uses run_confounded() from exp_extra.py."""
import os, json, importlib.util, sys, io, contextlib
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
import abw_sim2 as A
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '..', 'data')
if not os.path.exists(os.path.join(D, 'results.json')):
    raise SystemExit('data/results.json not found. Run run_all2.py first, '
                     'or run reproduce_all.py to execute every step in order.')
src = open(os.path.join(HERE, 'exp_extra.py')).read()
ns = dict(np=np, A=A)
exec('def run_confounded' + src.split('def run_confounded')[1].split('\ndef cross_conf')[0], ns)
run_confounded = ns['run_confounded']
rows = []
for spill in [0.0, 0.10, 0.20, 0.30, 0.50]:
    for s_ in range(1, 31):
        d, _ = run_confounded(s_, spill)
        cm = np.array(A.attribution(d)['confusion'])
        rec = 100 * np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        rows.append(dict(spill=spill, seed=s_, acc=float(np.trace(cm) / cm.sum()),
                         macro_f1=A.attribution(d)['macro_f1'],
                         r_b=float(np.corrcoef(d['B_obs'], d['theta'])[0, 1]),
                         auc=A.auc(d['y'], d['R_obs']),
                         cross=100.0 * (cm[1, 3] + cm[3, 1]) / max(cm[1].sum() + cm[3].sum(), 1),
                         wl_to_balanced=100 * cm[1, 2] / max(cm[1].sum(), 1),
                         rec_negligible=rec[0], rec_workload=rec[1], rec_balanced=rec[2], rec_behavior=rec[3]))
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, 'e2_confounding_detail.csv'), index=False)
det = []
for sp, v in df.groupby('spill'):
    r = {c: float(v[c].mean()) for c in df.columns if c not in ('seed',)}
    r['spill'] = float(sp); r['acc_sd'] = float(v.acc.std()); r['rec_workload_sd'] = float(v.rec_workload.std())
    det.append(r)
p = os.path.join(D, 'results.json'); j = json.load(open(p)); j['e2_detail'] = det
json.dump(j, open(p, 'w'), indent=1); print('e2_detail written')
