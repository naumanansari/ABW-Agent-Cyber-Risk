"""Run the full pipeline in the required order.

Usage (from the code/ folder):  python reproduce_all.py

Each step reads files written by the previous one, so the scripts must run in
this order. This regenerates every file in ../data and Figs. 3-9 in ../figures.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['run_all2.py', 'exp_extra.py', 'exp_confounding_detail.py',
         'tost_equivalence.py', 'exp_revision.py',
         'make_figures2.py', 'make_figures_extra.py']

for i, s in enumerate(STEPS, 1):
    print(f'[{i}/{len(STEPS)}] {s} ...', flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE,
                       stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        sys.exit(f'Stopped: {s} failed (exit code {r.returncode}).')
print('Done. All data files and Figs. 3-9 regenerated.')
