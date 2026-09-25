import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); F = os.path.join(HERE, '..', 'figures')
if not os.path.exists(os.path.join(D, 'results.json')):
    raise SystemExit('data/results.json not found. Run run_all2.py first, '
                     'or run reproduce_all.py to execute every step in order.')
os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.axisbelow': True})
R = json.load(open(os.path.join(D, 'results.json')))
csv = lambda n: pd.read_csv(os.path.join(D, n))
save = lambda n: (plt.savefig(os.path.join(F, n)), plt.close())

C_ABW, C_SEV, C_HF, C_3 = '#1f4e79', '#c00000', '#2e8b57', '#d99000'

# ---- Fig. 3 : severity-stratified  (KEY)
pb = R['stratified']['per_band']; b = [x['band'] for x in pb]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2),
                       gridspec_kw={'width_ratios': [2.5, 1]})
w = .27; x = np.arange(len(b))
ax[0].bar(x - w, [p['auc_abw'] for p in pb], w, yerr=[p['auc_abw_sd'] for p in pb],
          color=C_ABW, capsize=2, label='ABW fused score')
ax[0].bar(x, [p['auc_hf'] for p in pb], w, color=C_HF, label='Human-factor signal only')
ax[0].bar(x + w, [p['auc_sev'] for p in pb], w, yerr=[p['auc_sev_sd'] for p in pb],
          color=C_SEV, capsize=2, label='Severity-only')
ax[0].axhline(.5, color='k', lw=.8, ls=':'); ax[0].set_ylim(.45, .80)
ax[0].set_xticks(x); ax[0].set_xticklabels(b)
ax[0].set_xlabel('Severity decile band (narrow $s_r$ range)')
ax[0].set_ylabel('AUC-ROC within band')
ax[0].set_title('Discrimination within narrow severity bands', fontsize=9)
ax[0].legend(fontsize=7, ncol=3, loc='upper center', framealpha=.95, borderpad=.3)
p_ = R['stratified']['pooled']
names = ['ABW', 'Human\nfactor', 'Severity\nonly']
vals = [p_['auc_abw'], p_['auc_hf'], p_['auc_sev']]
errs = [p_['auc_abw_sd'], p_['auc_hf_sd'], p_['auc_sev_sd']]
ax[1].bar(names, vals, yerr=errs, color=[C_ABW, C_HF, C_SEV], capsize=3, width=.55)
ax[1].tick_params(axis='x', labelsize=7.5)
ax[1].axhline(.5, color='k', lw=.8, ls=':')
for i, v in enumerate(vals): ax[1].text(i, v + .012, f'{v:.3f}', ha='center', fontsize=8)
ax[1].set_ylim(.45, .72); ax[1].set_ylabel('Pooled within-band AUC')
ax[1].set_title('Pooled (30 seeds)', fontsize=9)
plt.tight_layout()
save('fig3_severity.png')

# ---- Fig. 4 : attribution confusion matrix
cm = np.array(R['attribution']['confusion_seed42'], float)
lab = ['Negligible', 'Workload-\ndominant', 'Balanced', 'Behavior-\ndominant']
row = cm / cm.sum(1, keepdims=True)
plt.figure(figsize=(4.3, 3.7))
plt.imshow(row, cmap='Blues', vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{row[i,j]*100:.0f}%\n({int(cm[i,j])})', ha='center', va='center',
                 fontsize=7.5, color='white' if row[i, j] > .55 else '#222')
plt.xticks(range(4), lab, fontsize=7.5); plt.yticks(range(4), lab, fontsize=7.5)
plt.xlabel('Driver identified by the agent'); plt.ylabel('True dominant driver')
plt.title('Driver attribution (seed 42; row-normalised)', fontsize=9)
plt.colorbar(fraction=.046).set_label('Row share', fontsize=8)
plt.grid(False); save('fig4_attribution.png')

# ---- Fig. 5 : misspecification
ms = R['misspecification']
ks = ['additive', 'multiplicative', 'saturating', 'threshold', 'reversed_weights',
      'behavior_only']
nm = ['Additive\n(matched)', 'Multiplic.\n+interaction', 'Saturating', 'Threshold',
      'Reversed\nweights', 'Behavior\nonly']
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
x = np.arange(len(ks)); w = .36
ax[0].bar(x - w / 2, [ms[k]['auc']['mean'] for k in ks], w,
          yerr=[ms[k]['auc']['sd'] for k in ks], color=C_ABW, capsize=2, label='ABW')
ax[0].bar(x + w / 2, [ms[k]['auc_sev']['mean'] for k in ks], w,
          yerr=[ms[k]['auc_sev']['sd'] for k in ks], color=C_SEV, capsize=2,
          label='Severity-only')
ax[0].set_xticks(x); ax[0].set_xticklabels(nm, fontsize=7); ax[0].set_ylim(.80, .92)
ax[0].set_ylabel('Marginal AUC-ROC'); ax[0].legend(fontsize=7)
ax[0].set_title('Marginal', fontsize=9)
ax[1].bar(x - w / 2, [ms[k]['strat_abw']['mean'] for k in ks], w,
          yerr=[ms[k]['strat_abw']['sd'] for k in ks], color=C_ABW, capsize=2, label='ABW')
ax[1].bar(x + w / 2, [ms[k]['strat_sev']['mean'] for k in ks], w,
          yerr=[ms[k]['strat_sev']['sd'] for k in ks], color=C_SEV, capsize=2,
          label='Severity-only')
ax[1].axhline(.5, color='k', lw=.8, ls=':')
ax[1].set_xticks(x); ax[1].set_xticklabels(nm, fontsize=7); ax[1].set_ylim(.48, .70)
ax[1].set_ylabel('Within-severity-band AUC'); ax[1].legend(fontsize=7)
ax[1].set_title('Within severity bands', fontsize=9)
fig.suptitle('Robustness to ground-truth misspecification (30 seeds per condition)',
             fontsize=9.5, y=1.02)
save('fig5_misspecification.png')

print('written fig3-Fig7')
