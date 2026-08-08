"""
Per-basin LATCHED classification on the microcin J25 (ring-closure, HLDA CV) landscape.

For each manually defined basin bounding box:
  1. Collect the frames inside the box and keep the N with the highest TRAM weight.
  2. Write them to a DCD and run LATCHED (VMD).
  3. Report the TRAM-weighted pre-lasso / pre-tadpole split of the basin.

Requires hlda_cv.py to have been run first, and VMD on PATH.
"""

import os, glob, subprocess, pickle
from collections import defaultdict
import numpy as np
import mdtraj as md
from natsort import natsorted

from latched_split import TCL_TEMPLATE

# User parameters
pwd = ''
LASSO   = 'microcinJ25'
OUT_DIR = 'basins_latched'
VMD_BIN = 'vmd'

CLUSTER_NUMBER = 400
TRAM_LAG       = 150

# LATCHED parameters for microcin J25 (VMD 1-indexed)
RING_SIZE  = 8     # residues 1-8
TAIL_START = 14
N_RESIDUES = 21

N_FRAMES = 1000    # max frames per basin, ranked by TRAM weight

# Basin bounding boxes: (cv1_lo, cv1_hi, cv2_lo, cv2_hi)
# CV1 = Gly1 N - Glu8 CD distance (A);  CV2 = HLDA CV
BASIN_BOXES = [
    (3.5, 4.0, -15, -10),
    (5.0, 6.0, -15, -10),
    (3.5, 4.0, -25, -20),
    (3.5, 4.0, -30, -25),
    (5.5, 6.0, -25, -20),
    (5.5, 6.0, -30, -25),
]

os.makedirs(OUT_DIR, exist_ok=True)
PRMTOP = f'{pwd}{LASSO}/unbiased/{LASSO}_nowat.prmtop'

# TRAM weights
tram_obj = pickle.load(open(
    f'{pwd}{LASSO}/tram_files/{LASSO}_ther_obj_cluster_{CLUSTER_NUMBER}_lag_{TRAM_LAG}_msm_obj.pkl', 'rb'))
stat_dis = tram_obj.stationary_distribution_full_state
dtrajs = pickle.load(open(
    f'{pwd}{LASSO}/tram_files/{LASSO}_dtrajs_cluster_{CLUSTER_NUMBER}_lag_{TRAM_LAG}.pkl', 'rb'))
txx_dtrajs      = np.concatenate(dtrajs)
unique_clusters = np.unique(txx_dtrajs)
n_per_cluster   = np.array([np.sum(txx_dtrajs == c) for c in unique_clusters])
tram_weights = np.array([stat_dis[txx_dtrajs[i]] / n_per_cluster[txx_dtrajs[i]]
                         for i in range(len(txx_dtrajs))])

# CV1 = ring-closure distance (A), CV2 = HLDA CV
with open(f'{pwd}{LASSO}/features/{LASSO}_biased_ring_close_dist_ca_n.pickle', 'rb') as f:
    biased_rc = pickle.load(f)
with open(f'{pwd}{LASSO}/features/{LASSO}_unbiased_ring_close_dist_ca_n.pickle', 'rb') as f:
    unbiased_rc = pickle.load(f)
cv1 = np.concatenate(biased_rc + unbiased_rc) * 10.0

with open(f'{LASSO}_biased_hlda_cv_dahora2024.pickle', 'rb') as f:
    biased_hlda = pickle.load(f)
with open(f'{LASSO}_unbiased_hlda_cv_dahora2024.pickle', 'rb') as f:
    unbiased_hlda = pickle.load(f)
cv2 = np.concatenate(biased_hlda + unbiased_hlda)

assert len(cv1) == len(tram_weights) == len(cv2), 'Frame count mismatch'

# Global frame index -> (trajectory index, frame index)
biased_trajs = natsorted(glob.glob(f'{pwd}{LASSO}/biased/{LASSO}_biased_traj_win_*_rep_*.dcd'))
biased_trajs = [t for t in biased_trajs if not t.endswith('_wat.dcd')]
unbiased_trajs = natsorted(glob.glob(f'{pwd}{LASSO}/unbiased/{LASSO}_unbiased_*_traj_*.dcd'))
unbiased_trajs = [t for t in unbiased_trajs if not t.endswith('_wat.dcd')]
all_trajs = biased_trajs + unbiased_trajs

traj_lengths  = [len(arr) for arr in biased_rc] + [len(arr) for arr in unbiased_rc]
cumul_lengths = np.cumsum([0] + traj_lengths)


def global_to_traj_frame(global_idx):
    ti = int(np.searchsorted(cumul_lengths, global_idx + 1) - 1)
    return ti, int(global_idx - cumul_lengths[ti])


def parse_latched_output(txt_path):
    is_prelasso = []
    with open(txt_path) as f:
        for line in f:
            if 'Pre-Lasso' in line:      is_prelasso.append(True)
            elif 'Pre-Tadpole' in line:  is_prelasso.append(False)
    return np.array(is_prelasso, dtype=bool)


results = []

for bid, (x_lo, x_hi, y_lo, y_hi) in enumerate(BASIN_BOXES, start=1):
    print(f'\nBasin {bid}: CV1 [{x_lo}, {x_hi}] A  CV2 [{y_lo}, {y_hi}]')

    in_box = (cv1 >= x_lo) & (cv1 <= x_hi) & (cv2 >= y_lo) & (cv2 <= y_hi)
    all_in_box = np.where(in_box)[0]
    if len(all_in_box) == 0:
        print('  No frames in box, skipping.')
        results.append({'basin': bid, 'n_frames': 0,
                        'prelasso_pct': np.nan, 'pretadpole_pct': np.nan})
        continue

    order = np.argsort(tram_weights[all_in_box])[::-1]
    selected = all_in_box[order[:N_FRAMES]]
    n_sel = len(selected)
    print(f'  Frames in box: {len(all_in_box)}, selected: {n_sel}')

    dcd_path = os.path.join(OUT_DIR, f'basin_{bid}_frames.dcd')
    if not os.path.exists(dcd_path):
        traj_to_local = defaultdict(list)
        for gidx in selected:
            ti, fi = global_to_traj_frame(gidx)
            traj_to_local[ti].append(fi)
        frames = []
        for ti in sorted(traj_to_local.keys()):
            t = md.load(all_trajs[ti], top=PRMTOP)
            for fi in traj_to_local[ti]:
                frames.append(t[fi])
        md.join(frames).save_dcd(dcd_path)

    out_prefix = os.path.join(OUT_DIR, f'basin_{bid}_latched_output')
    tcl_path = os.path.join(OUT_DIR, f'basin_{bid}_latched.tcl')
    with open(tcl_path, 'w') as f:
        f.write(TCL_TEMPLATE.format(
            TOPOLOGY   = PRMTOP,
            TRAJECTORY = dcd_path,
            OUTPUT     = out_prefix,
            RING_SIZE  = RING_SIZE,
            TAIL_START = TAIL_START,
            TAIL_END   = N_RESIDUES,
        ))

    txt_path = out_prefix + '.txt'
    if not os.path.exists(txt_path):
        log_path = os.path.join(OUT_DIR, f'basin_{bid}_vmd.log')
        with open(log_path, 'w') as lf:
            ret = subprocess.run([VMD_BIN, '-dispdev', 'text', '-e', tcl_path],
                                 stdout=lf, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            print(f'  WARNING: VMD non-zero exit, check {log_path}')

    is_prelasso = parse_latched_output(txt_path)
    if len(is_prelasso) != n_sel:
        print(f'  WARNING: LATCHED output has {len(is_prelasso)} lines, expected {n_sel}')

    w_sel = tram_weights[selected[:len(is_prelasso)]]
    w_total = w_sel.sum()
    prelasso_pct   = w_sel[is_prelasso].sum()  / w_total * 100 if w_total > 0 else np.nan
    pretadpole_pct = w_sel[~is_prelasso].sum() / w_total * 100 if w_total > 0 else np.nan
    print(f'  Pre-lasso {prelasso_pct:.1f} %, pre-tadpole {pretadpole_pct:.1f} % (TRAM-weighted)')

    results.append({'basin': bid, 'n_frames': n_sel,
                    'prelasso_pct': prelasso_pct, 'pretadpole_pct': pretadpole_pct,
                    'n_prelasso_raw': int(is_prelasso.sum()),
                    'n_pretadpole_raw': int((~is_prelasso).sum())})

print(f'\n{"Basin":>6}  {"N frames":>9}  {"Pre-lasso %":>12}  {"Pre-tadpole %":>14}')
for r in results:
    pl = f'{r["prelasso_pct"]:.1f}' if not np.isnan(r.get('prelasso_pct', np.nan)) else 'N/A'
    pt = f'{r["pretadpole_pct"]:.1f}' if not np.isnan(r.get('pretadpole_pct', np.nan)) else 'N/A'
    print(f'{r["basin"]:>6}  {r["n_frames"]:>9}  {pl:>12}  {pt:>14}')
