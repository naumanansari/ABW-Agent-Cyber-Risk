"""
Revision analyses (all on the reference seeds 1-30 unless stated).

R1  Effect size and 95% CI for the within-band paired difference; SD of the
    marginal human-factor-only AUC.
R2  Attribution against a majority-class baseline, with Cohen's kappa.
R3  Interventional reference class: driver defined by the reduction in the
    true hazard when B or W is set to zero (additive, threshold, reversed).
R4  M5 gradient boosting on raw telemetry; held-out comparison with M1-M4 on
    the matched, threshold, and saturating processes.
R5  Attribution recovered from the fitted logistic model (M3).
R6  Weight sweep with the behavior proxy rescaled by its expected scale 0.84.
R7  Dependent-severity generator.
R8  Independent corroborating signals (O, X) and recurrence propensity.
R9  Seed-42 confusion cells for Fig. 5.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, cohen_kappa_score, f1_score
import abw_sim2 as A

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); out = lambda n: os.path.join(D, n)
SEEDS = list(range(1, 31))
R = {}
msd = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
ci95 = lambda v: [float(np.mean(v) - 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))),
                  float(np.mean(v) + 1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))]
def band_auc(y, s, score):
    return float(np.mean([r['auc_abw'] for r in
                          A.stratified_auc(dict(s=s, y=y, R_obs=score, B_obs=np.zeros_like(s),
                                                W_obs=np.zeros_like(s)), n_bins=10)]))
def macro_f1(t, p):
    return float(f1_score(t, p, labels=[0, 1, 2, 3], average='macro', zero_division=0))

# ---- R1
st = pd.read_csv(out('severity_stratified.csv'))
ps = st.groupby('seed')[['auc_abw', 'auc_sev']].mean()
diff = (ps.auc_abw - ps.auc_sev).values
hf = [A.auc(A.run_simulation(s)['y'], A.WB_DEFAULT * A.run_simulation(s)['B_obs']
            + A.WW_DEFAULT * A.run_simulation(s)['W_obs']) for s in SEEDS]
R['r1'] = dict(band_diff=msd(diff), band_diff_ci95=ci95(diff),
               wins=int((diff > 0).sum()), hf_marginal_auc=msd(hf))

# ---- R2, R3, R5, R6 on reference and alternative processes
rows = []
for dgp in ['additive', 'threshold', 'reversed_weights']:
    for s_ in SEEDS:
        d = A.run_simulation(s_, dgp=dgp)
        Bt, Wt, s = d['B_true'], d['W_true'], d['s']
        h = A.hazard(dgp, s, Bt, Wt)
        rb = (h - A.hazard(dgp, s, np.zeros_like(Bt), Wt)) / s
        rw = (h - A.hazard(dgp, s, Bt, np.zeros_like(Wt))) / s
        ref_int = A.driver_class(rb, rw)
        row = dict(dgp=dgp, seed=s_, f1_interventional=macro_f1(ref_int, d['drv_obs']),
                   agree_def_int=float((ref_int == d['drv_true']).mean()))
        if dgp == 'additive':
            t = d['drv_true']
            row.update(acc=float((t == d['drv_obs']).mean()),
                       majority=float(np.bincount(t, minlength=4).max() / len(t)),
                       kappa=float(cohen_kappa_score(t, d['drv_obs'])),
                       f1_def=macro_f1(t, d['drv_obs']))
            # R5 attribution from the fitted logistic model
            n = len(d['y']); rng = np.random.default_rng(1000 + s_)
            perm = rng.permutation(n); tr = perm[:n // 2]
            lr = LogisticRegression(max_iter=1000).fit(np.column_stack([s, d['B_obs'], d['W_obs']])[tr], d['y'][tr])
            bb, bw = lr.coef_[0][1], lr.coef_[0][2]
            wb_fit = bb / (bb + bw)
            lr_cls = A.driver_class(wb_fit * d['B_obs'], (1 - wb_fit) * d['W_obs'])
            row.update(m3_wb=float(wb_fit), m3_f1_def=macro_f1(t, lr_cls),
                       m3_f1_int=macro_f1(ref_int, lr_cls), m3_acc=float((t == lr_cls).mean()))
            # R6 weight sweep with rescaled behavior proxy
            grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
            best = lambda B: float(grid[int(np.argmax([A.auc(d['y'], A.adjust_additive(s, B, d['W_obs'], w)) for w in grid]))])
            row.update(best_wb_raw=best(d['B_obs']), best_wb_rescaled=best(A.clip01(d['B_obs'] / A.B_OBS_SCALE)))
            row.update(e_zero=float((d['E_obs'] == 0).mean()))
        rows.append(row)
df = pd.DataFrame(rows); df.to_csv(out('rev_attribution.csv'), index=False)
add = df[df.dgp == 'additive']
R['r2'] = dict(accuracy=msd(add.acc), majority_baseline=msd(add.majority), kappa=msd(add.kappa))
R['r3'] = {g: dict(f1_interventional=msd(v.f1_interventional), agree_def_int=msd(v.agree_def_int))
           for g, v in df.groupby('dgp')}
R['r5'] = dict(m3_wb=msd(add.m3_wb), m3_f1_def=msd(add.m3_f1_def), m3_f1_int=msd(add.m3_f1_int),
               m3_acc=msd(add.m3_acc), abw_f1_def=msd(add.f1_def))
R['r6'] = dict(best_wb_raw=msd(add.best_wb_raw), best_wb_rescaled=msd(add.best_wb_rescaled))
R['e_zero_reference'] = msd(add.e_zero)

# ---- R4 gradient boosting, held-out half, three processes
rows = []
for dgp in ['additive', 'threshold', 'saturating']:
    for s_ in SEEDS:
        d = A.run_simulation(s_, dgp=dgp)
        n = len(d['y']); rng = np.random.default_rng(1000 + s_)
        perm = rng.permutation(n); tr, te = perm[:n // 2], perm[n // 2:]
        y, s = d['y'], d['s']
        raw = np.column_stack([s, d['L_obs'], d['C_obs'], d['E_obs'], d['rho_obs'], d['O_obs'], d['X_obs']])
        gbm = HistGradientBoostingClassifier(random_state=0).fit(raw[tr], y[tr])
        lr = LogisticRegression(max_iter=1000).fit(np.column_stack([s, d['B_obs'], d['W_obs']])[tr], y[tr])
        scores = {'M1 severity-only': s, 'M2 post-hoc adjustment': s * (1 + 0.25 * (d['B_obs'] + d['W_obs'])),
                  'M3 learned feature fusion': lr.predict_proba(np.column_stack([s, d['B_obs'], d['W_obs']]))[:, 1],
                  'M4 ABW agent': d['R_obs'], 'M5 gradient boosting': gbm.predict_proba(raw)[:, 1]}
        for nm, sc in scores.items():
            cal = LogisticRegression(max_iter=1000).fit(sc[tr].reshape(-1, 1), y[tr])
            p = cal.predict_proba(sc[te].reshape(-1, 1))[:, 1]
            rows.append(dict(dgp=dgp, seed=s_, model=nm, auc_test=roc_auc_score(y[te], sc[te]),
                             band_test=band_auc(y[te], s[te], sc[te]), brier_test=brier_score_loss(y[te], p)))
g4 = pd.DataFrame(rows); g4.to_csv(out('rev_gbm.csv'), index=False)
R['r4'] = {f'{g}|{m}': {c: msd(v[c]) for c in ['auc_test', 'band_test', 'brier_test']}
           for (g, m), v in g4.groupby(['dgp', 'model'])}

# ---- R7 dependent severity, R8 independent corroboration
rows = []
for name, kw in [('dependent_severity', dict(dependent_severity=True)),
                 ('independent_corroboration', dict(independent_corroboration=True))]:
    for s_ in SEEDS:
        d = A.run_simulation(s_, **kw)
        strat = A.stratified_auc(d, n_bins=10)
        rows.append(dict(variant=name, seed=s_, auc=A.auc(d['y'], d['R_obs']), auc_sev=A.auc(d['y'], d['s']),
                         strat_abw=float(np.mean([r['auc_abw'] for r in strat])),
                         strat_sev=float(np.mean([r['auc_sev'] for r in strat])),
                         acc=float((d['drv_true'] == d['drv_obs']).mean()),
                         f1=macro_f1(d['drv_true'], d['drv_obs']),
                         kappa=float(cohen_kappa_score(d['drv_true'], d['drv_obs'])),
                         r_s_rho=float(np.corrcoef(d['s'], d['rho_true'])[0, 1]),
                         r_s_theta=float(np.corrcoef(d['s'], d['theta'])[0, 1]),
                         e_zero=float((d['E_obs'] == 0).mean()), esc=float(d['y'].mean())))
g7 = pd.DataFrame(rows); g7.to_csv(out('rev_variants.csv'), index=False)
R['r7_r8'] = {v: {c: msd(x[c]) for c in x.columns if c not in ('variant', 'seed')}
              for v, x in g7.groupby('variant')}
R['r7_r8']['dependent_severity']['band_diff_ci95'] = ci95(
    (g7[g7.variant == 'dependent_severity'].strat_abw - g7[g7.variant == 'dependent_severity'].strat_sev).values)

# ---- R9
cm = np.array(A.attribution(A.run_simulation(42))['confusion'])
R['r9'] = dict(behavior_to_workload=int(cm[3, 1]), workload_to_behavior=int(cm[1, 3]),
               balanced_to_workload_pct=float(100 * cm[2, 1] / cm[2].sum()),
               behavior_to_negligible_pct=float(100 * cm[3, 0] / cm[3].sum()))

json.dump(R, open(out('revision_results.json'), 'w'), indent=1)
print(json.dumps(R, indent=1))
