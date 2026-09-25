"""
ABW Agent - extended simulation core.

Every generative constant used anywhere in the evaluation is declared here.
No constant is defined at a call site.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------- constants
SKILL_A, SKILL_B = 2.0, 6.0          # theta_a ~ Beta(2,6)
OVERLOAD_P = 0.20                    # P(overload episode)
RHO_NORM_MU, RHO_NORM_SD = 1.00, 0.15
RHO_OVER_MU, RHO_OVER_SD = 1.60, 0.30
RHO_LO, RHO_HI = 0.20, 3.00

OAFTER_SCALE = 0.60                  # O_true = clip(0.6 (rho-1))
XCTX_SCALE = 0.50                    # X_true = clip(0.5 (rho-1))

B_W_LATE, B_W_INCOMPLETE, B_W_REOPEN = 0.40, 0.35, 0.25
W_W_LOAD, W_W_AFTER, W_W_SWITCH = 0.50, 0.30, 0.20

BEHAVIOR_C_SCALE = 0.90              # C ~ Binomial(n, 0.9 theta)/n
BEHAVIOR_E_SCALE = 1.50              # reopen count ~ Poisson(1.5 theta)
BEHAVIOR_E_NORM = 3.00               # E = clip(count / 3)

NOISE_RHO, NOISE_O, NOISE_X = 0.10, 0.08, 0.08

SEV_LO, SEV_HI = 0.20, 0.90          # s_r ~ Uniform

# revision variants (off by default; the reference run draws none of these)
DEPSEV_RHO, DEPSEV_THETA, DEPSEV_NOISE = 0.15, 0.15, 0.05   # dependent-severity generator
CORROB_SD = 0.15                     # independent noise on true O and X
RECUR_A, RECUR_B = 2.0, 6.0          # per-administrator recurrence propensity lambda_a
B_OBS_SCALE = 0.84                   # E[B_obs] = 0.84 theta under the reference generator
LOGIT_SLOPE, LOGIT_MID, LOGIT_NOISE = 8.0, 0.85, 0.15

WB_DEFAULT, WW_DEFAULT = 0.60, 0.40
TAU_DEFAULT = 0.77

N_ADMINS, N_INSTANCES, N_TICKETS = 300, 3000, 10

# driver-attribution class boundaries
ATTR_FLOOR = 0.10                    # below this total adjustment -> "negligible"
ATTR_HI, ATTR_LO = 0.65, 0.35        # behaviour share cut-points
DRIVERS = ['negligible', 'workload', 'balanced', 'behavior']


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def behavior_score(L, C, E, wb1=B_W_LATE, wb2=B_W_INCOMPLETE, wb3=B_W_REOPEN):
    return clip01(wb1 * L + wb2 * C + wb3 * E)


def workload_score(rho, O, X):
    return clip01(W_W_LOAD * (rho - 1.0) + W_W_AFTER * O + W_W_SWITCH * X)


def adjust_additive(s, B, W, wb=WB_DEFAULT):
    return s * (1.0 + wb * B + (1.0 - wb) * W)


def adjust_multiplicative(s, B, W, wb=WB_DEFAULT):
    return s * (1.0 + wb * B) * (1.0 + (1.0 - wb) * W)


# ---------------------------------------------------------------- ground-truth DGPs
def hazard(dgp, s, B, W):
    """Latent hazard used to draw the escalation label. 'additive' is the
    specification the scorer assumes; the rest are misspecifications."""
    if dgp == 'additive':
        return s * (1.0 + WB_DEFAULT * B + WW_DEFAULT * W)
    if dgp == 'multiplicative':
        return s * (1.0 + WB_DEFAULT * B) * (1.0 + WW_DEFAULT * W)
    if dgp == 'saturating':
        # human-factor effect saturates: no unbounded compounding
        return s * (1.0 + np.tanh(1.6 * (WB_DEFAULT * B + WW_DEFAULT * W)))
    if dgp == 'threshold':
        # human factors only bite once either signal passes a floor
        eff = np.where(B > 0.35, B - 0.35, 0.0) + np.where(W > 0.35, W - 0.35, 0.0)
        return s * (1.0 + 1.5 * eff)
    if dgp == 'reversed_weights':
        # workload matters far more than behaviour - opposite of design prior
        return s * (1.0 + 0.25 * B + 0.75 * W)
    if dgp == 'behavior_only':
        return s * (1.0 + 1.0 * B)
    raise ValueError(dgp)


def driver_class(cb, cw):
    """4-class attribution label from behaviour / workload contributions."""
    tot = cb + cw
    out = np.full(len(tot), 0, dtype=int)          # negligible
    live = tot >= ATTR_FLOOR
    share = np.divide(cb, np.where(tot > 0, tot, 1.0))
    out[live & (share < ATTR_LO)] = 1              # workload
    out[live & (share >= ATTR_LO) & (share <= ATTR_HI)] = 2   # balanced
    out[live & (share > ATTR_HI)] = 3              # behavior
    return out


# ---------------------------------------------------------------- simulation
def run_simulation(seed=42, n_admins=N_ADMINS, n_instances=N_INSTANCES,
                   tickets=N_TICKETS, skill_a=SKILL_A, skill_b=SKILL_B,
                   overload_p=OVERLOAD_P, noise_scale=1.0, dgp='additive',
                   dependent_severity=False, independent_corroboration=False):
    rng = np.random.default_rng(seed)

    theta_admin = rng.beta(skill_a, skill_b, size=n_admins)
    idx = rng.integers(0, n_admins, size=n_instances)
    theta = theta_admin[idx]

    over = rng.random(n_instances) < overload_p
    rho_t = np.where(over,
                     rng.normal(RHO_OVER_MU, RHO_OVER_SD, n_instances),
                     rng.normal(RHO_NORM_MU, RHO_NORM_SD, n_instances))
    rho_t = np.clip(rho_t, RHO_LO, RHO_HI)

    if independent_corroboration:
        O_t = clip01(OAFTER_SCALE * (rho_t - 1.0) + rng.normal(0, CORROB_SD, n_instances))
        X_t = clip01(XCTX_SCALE * (rho_t - 1.0) + rng.normal(0, CORROB_SD, n_instances))
        lam = rng.beta(RECUR_A, RECUR_B, size=n_admins)[idx]
    else:
        O_t = clip01(OAFTER_SCALE * (rho_t - 1.0))
        X_t = clip01(XCTX_SCALE * (rho_t - 1.0))
        lam = None

    B_t = theta.copy()
    W_t = workload_score(rho_t, O_t, X_t)
    s = rng.uniform(SEV_LO, SEV_HI, n_instances)
    if dependent_severity:
        s = np.clip(s + DEPSEV_RHO * (rho_t - 1.0) + DEPSEV_THETA * theta
                    + rng.normal(0, DEPSEV_NOISE, n_instances), SEV_LO, SEV_HI)

    h = hazard(dgp, s, B_t, W_t)
    # centre each DGP on the same operating point so escalation base rates stay comparable
    h_mid = np.quantile(h, 1.0 - 0.30)
    pr = 1.0 / (1.0 + np.exp(-(LOGIT_SLOPE * (h - h_mid)
                               + rng.normal(0.0, LOGIT_NOISE, n_instances))))
    y = (rng.random(n_instances) < pr).astype(int)

    # ---- noisy observation
    L_o = rng.binomial(tickets, np.clip(theta, 0, 1)) / tickets
    C_o = rng.binomial(tickets, np.clip(BEHAVIOR_C_SCALE * theta, 0, 1)) / tickets
    e_rate = np.clip(theta, 0, 1) if lam is None else 0.5 * np.clip(theta, 0, 1) + 0.5 * lam
    E_o = clip01(rng.poisson(e_rate * BEHAVIOR_E_SCALE) / BEHAVIOR_E_NORM)
    B_o = behavior_score(L_o, C_o, E_o)

    rho_o = rho_t + rng.normal(0, NOISE_RHO * noise_scale, n_instances)
    O_o = clip01(O_t + rng.normal(0, NOISE_O * noise_scale, n_instances))
    X_o = clip01(X_t + rng.normal(0, NOISE_X * noise_scale, n_instances))
    W_o = workload_score(rho_o, O_o, X_o)

    R_o = adjust_additive(s, B_o, W_o)

    return dict(seed=seed, s=s, theta=theta, rho_true=rho_t,
                B_true=B_t, W_true=W_t, B_obs=B_o, W_obs=W_o,
                R_obs=R_o, y=y, hazard=h,
                L_obs=L_o, C_obs=C_o, E_obs=E_o, rho_obs=rho_o, O_obs=O_o, X_obs=X_o,
                drv_true=driver_class(WB_DEFAULT * B_t, WW_DEFAULT * W_t),
                drv_obs=driver_class(WB_DEFAULT * B_o, WW_DEFAULT * W_o))


# ---------------------------------------------------------------- metrics
def auc(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def stratified_auc(d, n_bins=5):
    """AUC inside narrow severity bands, where severity-only is uninformative
    by construction. Returns per-band AUC for the fused score and for severity."""
    s, y = d['s'], d['y']
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (s >= lo) & (s <= hi) if i == n_bins - 1 else (s >= lo) & (s < hi)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        rows.append(dict(band=i + 1, lo=float(lo), hi=float(hi), n=int(m.sum()),
                         rate=float(y[m].mean()),
                         auc_abw=auc(y[m], d['R_obs'][m]),
                         auc_sev=auc(y[m], s[m]),
                         auc_hf=auc(y[m], WB_DEFAULT * d['B_obs'][m]
                                    + WW_DEFAULT * d['W_obs'][m])))
    return rows


def attribution(d, subset=None):
    t, o = d['drv_true'], d['drv_obs']
    if subset is not None:
        t, o = t[subset], o[subset]
    k = len(DRIVERS)
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(t, o):
        cm[a, b] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    f1s = []
    for c in range(k):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return dict(accuracy=acc, macro_f1=float(np.mean(f1s)),
                per_class_f1={DRIVERS[i]: float(f1s[i]) for i in range(k)},
                confusion=cm.tolist(), support=cm.sum(axis=1).tolist())


def cost_analysis(d, ratios, rng_seed=0):
    """Threshold chosen on a training half, expected cost evaluated on the
    held-out half, for each cost ratio C_FN / C_FP."""
    rng = np.random.default_rng(rng_seed)
    n = len(d['y'])
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    y_tr, y_te = d['y'][tr], d['y'][te]
    out = []
    for name, sc in [('ABW', d['R_obs']), ('Severity-only', d['s'])]:
        s_tr, s_te = sc[tr], sc[te]
        grid = np.quantile(s_tr, np.linspace(0.01, 0.99, 99))
        for r in ratios:
            costs = [((y_tr == 1) & (s_tr < g)).sum() * r
                     + ((y_tr == 0) & (s_tr >= g)).sum() for g in grid]
            g = grid[int(np.argmin(costs))]
            c_te = (((y_te == 1) & (s_te < g)).sum() * r
                    + ((y_te == 0) & (s_te >= g)).sum()) / len(te)
            out.append(dict(model=name, ratio=float(r), tau=float(g), cost=float(c_te)))
    # reference strategies
    for r in ratios:
        out.append(dict(model='Escalate all', ratio=float(r),
                        tau=float('nan'), cost=float((y_te == 0).sum() / len(te))))
        out.append(dict(model='Escalate none', ratio=float(r), tau=float('nan'),
                        cost=float((y_te == 1).sum() * r / len(te))))
    return out


def calibration(d, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(d['y']); perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    res = {}
    for name, sc in [('ABW', d['R_obs']), ('Severity-only', d['s'])]:
        m = LogisticRegression().fit(sc[tr].reshape(-1, 1), d['y'][tr])
        p = m.predict_proba(sc[te].reshape(-1, 1))[:, 1]
        res[name] = dict(brier=float(brier_score_loss(d['y'][te], p)),
                         slope=float(m.coef_[0][0]),
                         auc=float(auc(d['y'][te], p)))
    base = d['y'][tr].mean()
    res['No-skill (base rate)'] = dict(
        brier=float(np.mean((d['y'][te] - base) ** 2)), slope=0.0, auc=0.5)
    # reliability curve for the fused score
    m = LogisticRegression().fit(d['R_obs'][tr].reshape(-1, 1), d['y'][tr])
    p = m.predict_proba(d['R_obs'][te].reshape(-1, 1))[:, 1]
    q = np.quantile(p, np.linspace(0, 1, 11))
    curve = []
    for i in range(10):
        msk = (p >= q[i]) & (p < q[i + 1]) if i < 9 else (p >= q[i])
        if msk.sum() >= 20:
            curve.append(dict(pred=float(p[msk].mean()),
                              obs=float(d['y'][te][msk].mean()), n=int(msk.sum())))
    res['reliability_curve'] = curve
    return res
