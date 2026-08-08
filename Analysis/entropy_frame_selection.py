"""
Write cpptraj input files that extract the pre-folded and unfolded ensembles fed to PARENT.

Pre-folded state: Q >= 0.8 and ring-closure distance <= 6 A.
Unfolded state:   Q <= 0.1 and ring-closure distance >= 6 A.
The unfolded set is down-sampled to the size of the pre-folded set so both ensembles
enter the entropy calculation with the same number of frames.
"""

import glob
import os
import pickle
import random
from natsort import natsorted

random.seed(42)

# User parameters
pwd = ''

lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin', 'caulonodin-V', 'caulosegnin-I', 'caulosegnin-II', 'chaxapeptin', 'citrocin',
              'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I', 'xanthomonin-II']

# State definition
Q_FOLDED = 0.8
Q_UNFOLDED = 0.1
RC_CUTOFF = 0.6   # nm (6 A)


def write_cpptraj_in(path, prmtop, traj_pairs, traj_files, out_xtc):
    with open(path, 'w') as f:
        f.write('parm ' + prmtop + ' \n')
        for traj_idx, frame_idx in traj_pairs:
            f.write('trajin ' + traj_files[traj_idx] + ' ' + str(frame_idx + 1) + ' ' + str(frame_idx + 1) + '\n')
        f.write('trajout ' + out_xtc + ' \n')
        f.write('run \n')
        f.write('exit \n')


for lasso in lasso_name:
    print(lasso)

    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_biased_q_list.pickle"), 'rb') as f:
        biased_q = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_unbiased_q_list.pickle"), 'rb') as f:
        unbiased_q = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_biased_ring_close_dist_ca_n.pickle"), 'rb') as f:
        biased_rc = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_unbiased_ring_close_dist_ca_n.pickle"), 'rb') as f:
        unbiased_rc = pickle.load(f)

    assert len(biased_q) == len(biased_rc)
    assert len(unbiased_q) == len(unbiased_rc)

    folded = {'biased': [], 'unbiased': []}
    unfolded = {'biased': [], 'unbiased': []}
    for tag, q_list, rc_list in (('biased', biased_q, biased_rc), ('unbiased', unbiased_q, unbiased_rc)):
        for traj in range(len(q_list)):
            for frame in range(len(q_list[traj])):
                q, rc = q_list[traj][frame], rc_list[traj][frame]
                if q <= Q_UNFOLDED and rc >= RC_CUTOFF:
                    unfolded[tag].append([traj, frame])
                elif q >= Q_FOLDED and rc <= RC_CUTOFF:
                    folded[tag].append([traj, frame])

    print('folded frames:', 'biased:', len(folded['biased']), ' unbiased:', len(folded['unbiased']))
    print('unfolded frames:', 'biased:', len(unfolded['biased']), ' unbiased:', len(unfolded['unbiased']))

    biased_trajs = natsorted(glob.glob(pwd + lasso + '/biased/' + lasso + '_biased_traj_win_*_rep_*.dcd'))
    unbiased_trajs = natsorted(glob.glob(pwd + lasso + '/unbiased/' + lasso + '_unbiased_*_traj_*.dcd'))
    out_dir = os.path.join(pwd, 'PARENT_entropy')
    os.makedirs(out_dir, exist_ok=True)

    # Pre-folded ensemble: all qualifying frames from both the biased and unbiased data.
    with open(os.path.join(out_dir, f'{lasso}_select_frame_folded_NC08_RC_CAN_06_for_entropy.in'), 'w') as f:
        f.write('parm ' + pwd + lasso + '/biased/' + lasso + '_nowat.prmtop \n')
        for traj_idx, frame_idx in folded['biased']:
            f.write('trajin ' + biased_trajs[traj_idx] + ' ' + str(frame_idx + 1) + ' ' + str(frame_idx + 1) + '\n')
        for traj_idx, frame_idx in folded['unbiased']:
            f.write('trajin ' + unbiased_trajs[traj_idx] + ' ' + str(frame_idx + 1) + ' ' + str(frame_idx + 1) + '\n')
        f.write('trajout ' + lasso + '_select_frame_folded_NC08_RC_CAN_06_for_entropy.xtc \n')
        f.write('run \n')
        f.write('exit \n')

    # Unfolded ensemble: unbiased frames only, matched in size to the pre-folded set.
    n_folded = len(folded['biased']) + len(folded['unbiased'])
    unfolded_sample = random.sample(unfolded['unbiased'], n_folded)
    write_cpptraj_in(
        os.path.join(out_dir, f'{lasso}_select_frame_unfolded_NC01_RC_CAN_06_for_entropy.in'),
        pwd + lasso + '/unbiased/' + lasso + '_nowat.prmtop',
        unfolded_sample,
        unbiased_trajs,
        lasso + '_select_frame_unfolded_NC01_RC_CAN_06_for_entropy.xtc',
    )
