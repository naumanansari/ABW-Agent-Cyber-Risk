import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_fscore_support
import abw_sim2 as A

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
os.makedirs(DATA, exist_ok=True)
out = lambda n: os.path.join(DATA, n)

SEEDS30 = list(range(1, 31))
SEEDS15 = list(range(1, 16))
R = {}

# ---------------------------------------------------------------- 1. main run
d = A.run_simulation(42)
R['main'] = dict(
    n=int(len(d['y'])), esc_rate=float(d['y'].mean()),
    auc_abw=float(A.auc(d['y'], d['R_obs'])),
    auc_sev=float(A.auc(d['y'], d['s'])),
    auc_hf=float(A.auc(d['y'], A.WB_DEFAULT * d['B_obs'] + A.WW_DEFAULT * d['W_obs'])),
    r_behavior=float(np.corrcoef(d['B_obs'], d['theta'])[0, 1]),
    r_workload=float(np.corrcoef(d['W_obs'], d['rho_true'])[0, 1]))
pd.DataFrame({k: d[k] for k in ['s', 'theta', 'rho_true', 'B_true', 'W_true',
                                'B_obs', 'W_obs', 'R_obs', 'y', 'drv_true', 'drv_obs']}
             ).to_csv(out('main_run_raw.csv'), index=False)

# ---------------------------------------------------------------- 2. threshold sweep
rows = []
for t in np.linspace(d['R_obs'].min(), d['R_obs'].max(), 200):
    pred = (d['R_obs'] >= t).astype(int)
    p, r_, f, _ = precision_recall_fscore_support(d['y'], pred, average='binary',
                                                  zero_division=0)
    rows.append(dict(tau=float(t), precision=p, recall=r_, f1=f,
                     esc_rate=float(pred.mean())))
ts = pd.DataFrame(rows); ts.to_csv(out('threshold_sweep.csv'), index=False)
best = ts.loc[ts.f1.idxmax()]
dflt = ts.iloc[(ts.tau - A.TAU_DEFAULT).abs().idxmin()]
R['operating_points'] = dict(
    default=dict(tau=A.TAU_DEFAULT, precision=float(dflt.precision),
                 recall=float(dflt.recall), f1=float(dflt.f1),
                 esc_rate=float(dflt.esc_rate)),
    best_f1=dict(tau=float(best.tau), precision=float(best.precision),
                 recall=float(best.recall), f1=float(best.f1),
                 esc_rate=float(best.esc_rate)))

# ---------------------------------------------------------------- 3. weight sweep
ws = [dict(wb=float(w), auc=float(A.auc(d['y'], A.adjust_additive(d['s'], d['B_obs'], d['W_obs'], w))))
      for w in np.round(np.arange(0.1, 0.91, 0.05), 2)]
pd.DataFrame(ws).to_csv(out('weight_sweep.csv'), index=False)
R['weight_sweep'] = dict(full=ws, peak=max(ws, key=lambda x: x['auc']),
                         at_default=[x for x in ws if abs(x['wb'] - 0.60) < 1e-9][0])

# ---------------------------------------------------------------- 4. severity-stratified
strat_rows = []
for s_ in SEEDS30:
    dd = A.run_simulation(s_)
    for r_ in A.stratified_auc(dd, n_bins=10):
        r_['seed'] = s_; strat_rows.append(r_)
st = pd.DataFrame(strat_rows); st.to_csv(out('severity_stratified.csv'), index=False)
g = st.groupby('band')[['auc_abw', 'auc_sev', 'auc_hf', 'rate']].agg(['mean', 'std'])
R['stratified'] = dict(
    per_band=[dict(band=int(b),
                   auc_abw=float(g.loc[b, ('auc_abw', 'mean')]),
                   auc_abw_sd=float(g.loc[b, ('auc_abw', 'std')]),
                   auc_sev=float(g.loc[b, ('auc_sev', 'mean')]),
                   auc_sev_sd=float(g.loc[b, ('auc_sev', 'std')]),
                   auc_hf=float(g.loc[b, ('auc_hf', 'mean')]),
                   rate=float(g.loc[b, ('rate', 'mean')])) for b in g.index],
    pooled=dict(auc_abw=float(st.auc_abw.mean()), auc_abw_sd=float(st.groupby('seed').auc_abw.mean().std()),
                auc_sev=float(st.auc_sev.mean()), auc_sev_sd=float(st.groupby('seed').auc_sev.mean().std()),
                auc_hf=float(st.auc_hf.mean()), auc_hf_sd=float(st.groupby('seed').auc_hf.mean().std())))
per_seed = st.groupby('seed')[['auc_abw', 'auc_sev']].mean()
from scipy import stats as SS
tt = SS.ttest_rel(per_seed.auc_abw, per_seed.auc_sev)
R['stratified']['paired_test'] = dict(t=float(tt.statistic), p=float(tt.pvalue),
                                      wins=int((per_seed.auc_abw > per_seed.auc_sev).sum()),
                                      n=len(per_seed))

# ---------------------------------------------------------------- 5. driver attribution
att_rows = []
for s_ in SEEDS30:
    dd = A.run_simulation(s_)
    flag = dd['R_obs'] >= A.TAU_DEFAULT
    a_all, a_esc = A.attribution(dd), A.attribution(dd, flag)
    att_rows.append(dict(seed=s_, acc_all=a_all['accuracy'], f1_all=a_all['macro_f1'],
                         acc_esc=a_esc['accuracy'], f1_esc=a_esc['macro_f1'],
                         **{f'f1_{k}': v for k, v in a_all['per_class_f1'].items()}))
at = pd.DataFrame(att_rows); at.to_csv(out('attribution.csv'), index=False)
R['attribution'] = dict(
    {c: dict(mean=float(at[c].mean()), sd=float(at[c].std())) for c in at.columns if c != 'seed'},
    confusion_seed42=A.attribution(d)['confusion'],
    support_seed42=A.attribution(d)['support'], classes=A.DRIVERS)

# ---------------------------------------------------------------- 6. cost analysis
RATIOS = [1, 2, 3, 5, 8, 12, 20]
cost_rows = []
for s_ in SEEDS30:
    dd = A.run_simulation(s_)
    for r_ in A.cost_analysis(dd, RATIOS, rng_seed=s_):
        r_['seed'] = s_; cost_rows.append(r_)
ct = pd.DataFrame(cost_rows); ct.to_csv(out('cost_analysis.csv'), index=False)
piv = ct.groupby(['ratio', 'model']).cost.mean().unstack()
R['cost'] = dict(ratios=RATIOS, table=[
    dict(ratio=float(r),
         abw=float(piv.loc[r, 'ABW']), sev=float(piv.loc[r, 'Severity-only']),
         esc_all=float(piv.loc[r, 'Escalate all']), esc_none=float(piv.loc[r, 'Escalate none']),
         gain_vs_sev=float(100 * (piv.loc[r, 'Severity-only'] - piv.loc[r, 'ABW']) / piv.loc[r, 'Severity-only']),
         gain_vs_default=float(100 * (min(piv.loc[r, 'Escalate all'], piv.loc[r, 'Escalate none'])
                                      - piv.loc[r, 'ABW']) / min(piv.loc[r, 'Escalate all'], piv.loc[r, 'Escalate none'])))
    for r in RATIOS])

# ---------------------------------------------------------------- 7. calibration
cal = A.calibration(d, rng_seed=42)
pd.DataFrame(cal['reliability_curve']).to_csv(out('reliability_curve.csv'), index=False)
cal_ms = {k: [] for k in ['ABW', 'Severity-only', 'No-skill (base rate)']}
for s_ in SEEDS30:
    c = A.calibration(A.run_simulation(s_), rng_seed=s_)
    for k in cal_ms: cal_ms[k].append(c[k]['brier'])
R['calibration'] = dict(
    seed42={k: v for k, v in cal.items() if k != 'reliability_curve'},
    brier_multiseed={k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
                     for k, v in cal_ms.items()})

# ---------------------------------------------------------------- 8. multi-seed robustness
ms = []
for s_ in SEEDS30:
    dd = A.run_simulation(s_)
    pred = (dd['R_obs'] >= A.TAU_DEFAULT).astype(int)
    p, r_, f, _ = precision_recall_fscore_support(dd['y'], pred, average='binary', zero_division=0)
    ms.append(dict(seed=s_, auc=A.auc(dd['y'], dd['R_obs']),
                   auc_sev=A.auc(dd['y'], dd['s']), precision=p, recall=r_, f1=f,
                   r_behavior=np.corrcoef(dd['B_obs'], dd['theta'])[0, 1],
                   r_workload=np.corrcoef(dd['W_obs'], dd['rho_true'])[0, 1],
                   esc_true=dd['y'].mean(), esc_flag=pred.mean()))
mdf = pd.DataFrame(ms); mdf.to_csv(out('multiseed.csv'), index=False)
R['multiseed'] = {c: dict(mean=float(mdf[c].mean()), sd=float(mdf[c].std()))
                  for c in mdf.columns if c != 'seed'}

# ---------------------------------------------------------------- 9. ablation
VAR = {'Additive fusion (wb=0.6)': lambda x: A.adjust_additive(x['s'], x['B_obs'], x['W_obs'], 0.6),
       'Equal-weight fusion (wb=0.5)': lambda x: A.adjust_additive(x['s'], x['B_obs'], x['W_obs'], 0.5),
       'Multiplicative fusion (HEART-style)': lambda x: A.adjust_multiplicative(x['s'], x['B_obs'], x['W_obs'], 0.6),
       'Behavior-only (wb=1.0)': lambda x: A.adjust_additive(x['s'], x['B_obs'], x['W_obs'], 1.0),
       'Workload-only (wb=0.0)': lambda x: A.adjust_additive(x['s'], x['B_obs'], x['W_obs'], 0.0),
       'Severity-only (no adjustment)': lambda x: x['s']}
ab = []
for s_ in SEEDS30:
    dd = A.run_simulation(s_)
    for k, fn in VAR.items():
        ab.append(dict(seed=s_, variant=k, auc=A.auc(dd['y'], fn(dd)),
                       auc_strat=float(np.mean([r['auc_abw'] for r in []]) if False else np.nan)))
abd = pd.DataFrame(ab); abd.to_csv(out('ablation.csv'), index=False)
gp = abd.pivot(index='seed', columns='variant', values='auc')
R['ablation'] = {k: dict(mean=float(gp[k].mean()), sd=float(gp[k].std())) for k in VAR}
t2 = SS.ttest_rel(gp['Additive fusion (wb=0.6)'], gp['Multiplicative fusion (HEART-style)'])
R['ablation_test'] = dict(add_vs_mult_wins=int((gp['Additive fusion (wb=0.6)'] >
                                                gp['Multiplicative fusion (HEART-style)']).sum()),
                          t=float(t2.statistic), p=float(t2.pvalue),
                          mean_diff=float((gp['Additive fusion (wb=0.6)'] -
                                           gp['Multiplicative fusion (HEART-style)']).mean()))

# ---------------------------------------------------------------- 10. sensitivity
tick, nois = [], []
for n_ in [5, 10, 20, 30, 50]:
    for s_ in SEEDS30:
        dd = A.run_simulation(s_, tickets=n_)
        tick.append(dict(tickets=n_, seed=s_, r_behavior=np.corrcoef(dd['B_obs'], dd['theta'])[0, 1],
                         auc=A.auc(dd['y'], dd['R_obs']),
                         attr=A.attribution(dd)['accuracy']))
for nz in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for s_ in SEEDS30:
        dd = A.run_simulation(s_, noise_scale=nz)
        nois.append(dict(noise=nz, seed=s_, r_workload=np.corrcoef(dd['W_obs'], dd['rho_true'])[0, 1],
                         auc=A.auc(dd['y'], dd['R_obs']),
                         attr=A.attribution(dd)['accuracy']))
tdf, ndf = pd.DataFrame(tick), pd.DataFrame(nois)
tdf.to_csv(out('ticket_sensitivity.csv'), index=False)
ndf.to_csv(out('noise_sensitivity.csv'), index=False)
R['ticket_sensitivity'] = {str(k): dict(r=float(v.r_behavior.mean()), auc=float(v.auc.mean()),
                                        attr=float(v.attr.mean()), r_sd=float(v.r_behavior.std()),
                                        auc_sd=float(v.auc.std()))
                           for k, v in tdf.groupby('tickets')}
R['noise_sensitivity'] = {str(k): dict(r=float(v.r_workload.mean()), auc=float(v.auc.mean()),
                                       attr=float(v.attr.mean()), r_sd=float(v.r_workload.std()),
                                       auc_sd=float(v.auc.std()))
                          for k, v in ndf.groupby('noise')}

# ---------------------------------------------------------------- 11. cross-population
POPS = {'Base (design default)': dict(),
        'Healthy org (low skill deficit)': dict(skill_a=2, skill_b=10),
        'Distressed org (wide skill spread)': dict(skill_a=4, skill_b=4),
        'High-overload org (40%)': dict(overload_p=0.40),
        'Low-overload org (5%)': dict(overload_p=0.05)}
cp = []
for k, kw in POPS.items():
    for s_ in SEEDS30:
        dd = A.run_simulation(s_, **kw)
        sweep = [(w, A.auc(dd['y'], A.adjust_additive(dd['s'], dd['B_obs'], dd['W_obs'], w)))
                 for w in np.round(np.arange(0.1, 0.91, 0.05), 2)]
        strat = A.stratified_auc(dd, n_bins=10)
        cp.append(dict(population=k, seed=s_, auc=A.auc(dd['y'], dd['R_obs']),
                       auc_sev=A.auc(dd['y'], dd['s']),
                       strat_abw=float(np.mean([r['auc_abw'] for r in strat])),
                       strat_sev=float(np.mean([r['auc_sev'] for r in strat])),
                       best_wb=float(max(sweep, key=lambda x: x[1])[0]),
                       attr=A.attribution(dd)['accuracy'], esc=dd['y'].mean()))
cpd = pd.DataFrame(cp); cpd.to_csv(out('cross_population.csv'), index=False)
R['cross_population'] = {k: {c: dict(mean=float(v[c].mean()), sd=float(v[c].std()))
                             for c in ['auc', 'auc_sev', 'strat_abw', 'strat_sev', 'best_wb', 'attr', 'esc']}
                         for k, v in cpd.groupby('population')}

# ---------------------------------------------------------------- 12. misspecification
DGPS = ['additive', 'multiplicative', 'saturating', 'threshold',
        'reversed_weights', 'behavior_only']
mis = []
for g_ in DGPS:
    for s_ in SEEDS30:
        dd = A.run_simulation(s_, dgp=g_)
        strat = A.stratified_auc(dd, n_bins=10)
        mis.append(dict(dgp=g_, seed=s_, auc=A.auc(dd['y'], dd['R_obs']),
                        auc_sev=A.auc(dd['y'], dd['s']),
                        strat_abw=float(np.mean([r['auc_abw'] for r in strat])),
                        strat_sev=float(np.mean([r['auc_sev'] for r in strat])),
                        esc=dd['y'].mean()))
mid = pd.DataFrame(mis); mid.to_csv(out('misspecification.csv'), index=False)
R['misspecification'] = {k: {c: dict(mean=float(v[c].mean()), sd=float(v[c].std()))
                             for c in ['auc', 'auc_sev', 'strat_abw', 'strat_sev', 'esc']}
                         for k, v in mid.groupby('dgp')}

with open(out('results.json'), 'w') as f:
    json.dump(R, f, indent=1)
print('OK. datasets:', len([x for x in os.listdir(DATA) if x.endswith('.csv')]))
print('main AUC %.4f  sev %.4f' % (R['main']['auc_abw'], R['main']['auc_sev']))
print('stratified pooled: abw %.4f sev %.4f hf %.4f' %
      (R['stratified']['pooled']['auc_abw'], R['stratified']['pooled']['auc_sev'],
       R['stratified']['pooled']['auc_hf']))
print('attribution acc %.4f  macroF1 %.4f' %
      (R['attribution']['acc_all']['mean'], R['attribution']['f1_all']['mean']))
