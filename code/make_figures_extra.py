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
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'axes.axisbelow': True})
R = json.load(open(os.path.join(D, 'results.json')))
if 'e1' not in R or 'e2_detail' not in R:
    raise SystemExit('Experiment results missing. Run exp_extra.py and exp_confounding_detail.py first, or run reproduce_all.py.')
C_ABW, C_SEV, C_HF, C_3 = '#1f4e79', '#c00000', '#2e8b57', '#d99000'

# ---------------- Fig 11 : representation comparison
e1 = R['e1']
names = ['M1 severity-only', 'M2 post-hoc adjustment', 'M3 learned feature fusion', 'M4 ABW agent']
short = ['Severity\nonly', 'Post-hoc\nadjustment', 'Learned\nfusion', 'ABW\nagent']
mar = [e1[n]['auc']['mean'] for n in names]; mar_sd = [e1[n]['auc']['sd'] for n in names]
bnd = [e1[n]['band_auc']['mean'] for n in names]; bnd_sd = [e1[n]['band_auc']['sd'] for n in names]
col = [C_SEV, C_HF, C_HF, C_ABW]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
x = np.arange(4)
ax[0].bar(x, mar, yerr=mar_sd, color=col, capsize=3, width=.6)
for i, v in enumerate(mar): ax[0].text(i, v + .0018, f'{v:.4f}', ha='center', fontsize=7.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=8)
ax[0].set_ylim(.860, .888); ax[0].set_ylabel('Marginal AUC-ROC')
ax[0].set_title('Marginal discrimination', fontsize=9)
ax[1].bar(x, bnd, yerr=bnd_sd, color=col, capsize=3, width=.6)
for i, v in enumerate(bnd): ax[1].text(i, v + .004, f'{v:.4f}', ha='center', fontsize=7.5)
ax[1].axhline(.5, color='k', lw=.8, ls=':')
ax[1].set_xticks(x); ax[1].set_xticklabels(short, fontsize=8)
ax[1].set_ylim(.48, .68); ax[1].set_ylabel('Within-severity-band AUC')
ax[1].set_title('Within severity bands', fontsize=9)
fig.suptitle('Three representations of the same two signals are statistically indistinguishable',
             fontsize=9.5, y=1.03)
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig6_representation.png')); plt.close()

# ---------------- Fig 12 : confounding
d2 = R['e2_detail']
sp = [r['spill'] for r in d2]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
a0 = ax[0]
a0.plot(sp, [r['rec_workload'] for r in d2], 'o-', color=C_SEV, lw=2,
        label='Workload-dominant recall')
a0.plot(sp, [r['wl_to_balanced'] for r in d2], 's--', color=C_3,
        label='True workload \u2192 labelled "balanced"')
a0.plot(sp, [r['cross'] for r in d2], '^-', color=C_ABW,
        label='Behavior/workload cross-confusion')
a0.set_xlabel('Contamination of the behavior channel by load')
a0.set_ylabel('Percent of true workload-dominant cases')
a0.set_ylim(-4, 100); a0.legend(fontsize=7, loc='center left')
a0.set_title('Where the attribution fails', fontsize=9)

a1 = ax[1]
a1.plot(sp, [r['macro_f1'] for r in d2], 'o-', color=C_SEV, lw=2, label='Macro-F1 (attribution)')
a1.plot(sp, [r['acc'] for r in d2], 's-', color=C_3, label='Overall accuracy (attribution)')
a1.plot(sp, [r['auc'] for r in d2], '^--', color=C_ABW, label='AUC-ROC (ranking)')
a1.plot(sp, [r['r_b'] for r in d2], 'v:', color='#777', label=r'$r(B_{obs},\theta)$')
a1.set_xlabel('Contamination of the behavior channel by load')
a1.set_ylabel('Value'); a1.set_ylim(.50, .95); a1.legend(fontsize=7, loc='lower left')
a1.set_title('What the aggregate metrics show', fontsize=9)
fig.suptitle('Confounding degrades attribution while leaving ranking and aggregate accuracy intact',
             fontsize=9.5, y=1.03)
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig7_confounding.png')); plt.close()
print('written fig6, fig7')
