"""
Two experiments the reviewer identified as decisive.

E1 - Representation comparison. Does agent-level separation buy anything a
     post-hoc scalar adjustment or a learned feature-fusion model cannot?
E2 - Confounded behaviour/workload. The simulation generates theta and rho
     independently; in reality overload produces behaviour-like symptoms.
     Does attribution survive when workload contaminates the behaviour channel?
     And can load-conditioning the behaviour channel repair it?
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats as SS
import abw_sim2 as A

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
out = lambda n: os.path.join(D, n)
if not os.path.exists(os.path.join(D, 'results.json')):
    raise SystemExit('data/results.json not found. Run run_all2.py first, '
                     'or run reproduce_all.py to execute every step in order.')
SEEDS = list(range(1, 31))
R = {}

# =====================================================================
# E1  representation comparison
# =====================================================================
rows = []
for s_ in SEEDS:
    d = A.run_simulation(s_)
    n = len(d['y']); rng = np.random.default_rng(1000 + s_)
    perm = rng.permutation(n); tr, te = perm[:n // 2], perm[n // 2:]
    y_tr, y_te = d['y'][tr], d['y'][te]
    s, B, W = d['s'], d['B_obs'], d['W_obs']

    # M1 severity only
    m1 = s
    # M2 post-hoc scalar adjustment: one blended human-factor correction
    H = 0.5 * (B + W)
    m2 = s * (1 + 0.5 * H)
    # M3 learned feature fusion on the SAME inputs, fitted on the training half
    X = np.column_stack([s, B, W])
    lr = LogisticRegression(max_iter=1000).fit(X[tr], y_tr)
    m3 = lr.predict_proba(X)[:, 1]
    # M4 ABW agent, deterministic, nothing fitted
    m4 = d['R_obs']

    def band(sc):
        return float(np.mean([r['auc_abw'] for r in
                              A.stratified_auc({**d, 'R_obs': sc}, n_bins=10)]))

    for nm, sc in [('M1 severity-only', m1), ('M2 post-hoc adjustment', m2),
                   ('M3 learned feature fusion', m3), ('M4 ABW agent', m4)]:
        cal = LogisticRegression(max_iter=1000).fit(sc[tr].reshape(-1, 1), y_tr)
        pte = cal.predict_proba(sc[te].reshape(-1, 1))[:, 1]
        rows.append(dict(seed=s_, model=nm,
                         auc=roc_auc_score(d['y'], sc),
                         auc_test=roc_auc_score(y_te, sc[te]),
                         band_auc=band(sc),
                         brier=brier_score_loss(y_te, pte),
                         attribution=(A.attribution(d)['accuracy']
                                      if nm == 'M4 ABW agent' else np.nan)))
e1 = pd.DataFrame(rows); e1.to_csv(out('e1_representation.csv'), index=False)
g = e1.groupby('model')[['auc', 'auc_test', 'band_auc', 'brier']].agg(['mean', 'std'])
R['e1'] = {m: {c: dict(mean=float(g.loc[m, (c, 'mean')]), sd=float(g.loc[m, (c, 'std')]))
               for c in ['auc', 'auc_test', 'band_auc', 'brier']} for m in g.index}
piv = e1.pivot(index='seed', columns='model', values='auc')
t34 = SS.ttest_rel(piv['M4 ABW agent'], piv['M3 learned feature fusion'])
t24 = SS.ttest_rel(piv['M4 ABW agent'], piv['M2 post-hoc adjustment'])
R['e1_tests'] = dict(
    abw_vs_learned=dict(diff=float((piv['M4 ABW agent'] - piv['M3 learned feature fusion']).mean()),
                        t=float(t34.statistic), p=float(t34.pvalue)),
    abw_vs_posthoc=dict(diff=float((piv['M4 ABW agent'] - piv['M2 post-hoc adjustment']).mean()),
                        t=float(t24.statistic), p=float(t24.pvalue)))

print('=== E1 representation comparison (30 seeds) ===')
for m in g.index:
    print(f"  {m:28s} AUC {g.loc[m,('auc','mean')]:.4f}  band {g.loc[m,('band_auc','mean')]:.4f}"
          f"  Brier {g.loc[m,('brier','mean')]:.4f}")
print('  ABW - learned fusion : %+.4f AUC (p=%.3g)' % (R['e1_tests']['abw_vs_learned']['diff'],
                                                       R['e1_tests']['abw_vs_learned']['p']))
print('  ABW - post-hoc scalar: %+.4f AUC (p=%.3g)' % (R['e1_tests']['abw_vs_posthoc']['diff'],
                                                       R['e1_tests']['abw_vs_posthoc']['p']))

# =====================================================================
# E2  confounded behaviour / workload
# =====================================================================
def run_confounded(seed, spill, tickets=10):
    """Overload contaminates the observed behaviour channel: an administrator
    at high load looks behaviourally worse at identical true skill."""
    rng = np.random.default_rng(seed)
    theta_admin = rng.beta(A.SKILL_A, A.SKILL_B, size=A.N_ADMINS)
    idx = rng.integers(0, A.N_ADMINS, size=A.N_INSTANCES)
    theta = theta_admin[idx]
    over = rng.random(A.N_INSTANCES) < A.OVERLOAD_P
    rho_t = np.clip(np.where(over, rng.normal(A.RHO_OVER_MU, A.RHO_OVER_SD, A.N_INSTANCES),
                             rng.normal(A.RHO_NORM_MU, A.RHO_NORM_SD, A.N_INSTANCES)),
                    A.RHO_LO, A.RHO_HI)
    O_t = A.clip01(A.OAFTER_SCALE * (rho_t - 1)); X_t = A.clip01(A.XCTX_SCALE * (rho_t - 1))
    B_t, W_t = theta.copy(), A.workload_score(rho_t, O_t, X_t)
    s = rng.uniform(A.SEV_LO, A.SEV_HI, A.N_INSTANCES)
    h = A.hazard('additive', s, B_t, W_t)
    h0 = np.quantile(h, 0.70)
    pr = 1 / (1 + np.exp(-(A.LOGIT_SLOPE * (h - h0) + rng.normal(0, A.LOGIT_NOISE, A.N_INSTANCES))))
    y = (rng.random(A.N_INSTANCES) < pr).astype(int)

    # contamination: load inflates the observed late / incomplete rates
    load_excess = np.clip(rho_t - 1, 0, None)
    pL = A.clip01(theta + spill * load_excess)
    pC = A.clip01(A.BEHAVIOR_C_SCALE * theta + spill * load_excess)
    L_o = rng.binomial(tickets, pL) / tickets
    C_o = rng.binomial(tickets, pC) / tickets
    E_o = A.clip01(rng.poisson(np.clip(theta, 0, 1) * A.BEHAVIOR_E_SCALE) / A.BEHAVIOR_E_NORM)
    B_o = A.behavior_score(L_o, C_o, E_o)

    rho_o = rho_t + rng.normal(0, A.NOISE_RHO, A.N_INSTANCES)
    O_o = A.clip01(O_t + rng.normal(0, A.NOISE_O, A.N_INSTANCES))
    X_o = A.clip01(X_t + rng.normal(0, A.NOISE_X, A.N_INSTANCES))
    W_o = A.workload_score(rho_o, O_o, X_o)

    # mitigation: condition the behaviour channel on observed load
    B_adj = A.clip01(B_o - 0.40 * spill * np.clip(rho_o - 1, 0, None))

    mk = lambda b: dict(s=s, theta=theta, rho_true=rho_t, B_true=B_t, W_true=W_t,
                        B_obs=b, W_obs=W_o, R_obs=A.adjust_additive(s, b, W_o), y=y,
                        drv_true=A.driver_class(A.WB_DEFAULT * B_t, A.WW_DEFAULT * W_t),
                        drv_obs=A.driver_class(A.WB_DEFAULT * b, A.WW_DEFAULT * W_o))
    return mk(B_o), mk(B_adj)


def cross_conf(d):
    cm = np.array(A.attribution(d)['confusion'])
    den = cm[1, :].sum() + cm[3, :].sum()
    return 100.0 * (cm[1, 3] + cm[3, 1]) / max(den, 1)


rows = []
for spill in [0.0, 0.10, 0.20, 0.30, 0.50]:
    for s_ in SEEDS:
        dn, da = run_confounded(s_, spill)
        rows.append(dict(spill=spill, seed=s_, variant='uncorrected',
                         attr=A.attribution(dn)['accuracy'], cross=cross_conf(dn),
                         auc=A.auc(dn['y'], dn['R_obs']),
                         r_b=np.corrcoef(dn['B_obs'], dn['theta'])[0, 1]))
        rows.append(dict(spill=spill, seed=s_, variant='load-conditioned',
                         attr=A.attribution(da)['accuracy'], cross=cross_conf(da),
                         auc=A.auc(da['y'], da['R_obs']),
                         r_b=np.corrcoef(da['B_obs'], da['theta'])[0, 1]))
e2 = pd.DataFrame(rows); e2.to_csv(out('e2_confounding.csv'), index=False)
gg = e2.groupby(['spill', 'variant'])[['attr', 'cross', 'auc', 'r_b']].agg(['mean', 'std'])
R['e2'] = [dict(spill=float(sp), variant=v,
                attr=float(gg.loc[(sp, v), ('attr', 'mean')]),
                attr_sd=float(gg.loc[(sp, v), ('attr', 'std')]),
                cross=float(gg.loc[(sp, v), ('cross', 'mean')]),
                cross_sd=float(gg.loc[(sp, v), ('cross', 'std')]),
                auc=float(gg.loc[(sp, v), ('auc', 'mean')]),
                r_b=float(gg.loc[(sp, v), ('r_b', 'mean')]))
           for sp, v in gg.index]

print('\n=== E2 confounded behaviour/workload (30 seeds per setting) ===')
print('  spill  variant            attr-acc   cross-confusion%   AUC     r(B,theta)')
for r_ in R['e2']:
    print(f"  {r_['spill']:.2f}   {r_['variant']:18s} {r_['attr']:.3f}      "
          f"{r_['cross']:6.2f}          {r_['auc']:.3f}   {r_['r_b']:.3f}")

j = json.load(open(out('results.json')))
j.update(R)
json.dump(j, open(out('results.json'), 'w'), indent=1)
print('\nsaved to results.json')
