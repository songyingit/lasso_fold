"""
Calculate the features needed for TRAM analysis from biased and unbiased MD trajectories.

Three features are written per trajectory:
  * fraction of native contacts (Q)
  * pairwise CA-CA residue distances (tICA input)
  * ring-closure distance: N of residue 1 to the side-chain carboxylate carbon
    of the acceptor residue (CD of Glu, CG of Asp)
"""

import pickle
import mdtraj as md
import numpy as np
from tqdm import tqdm
from itertools import combinations
import os
import glob
import re
from natsort import natsorted


def q_func(traj, native):
    BETA_CONST = 50
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45
    heavy = native.topology.select_atom_indices('heavy')
    heavy_pairs = np.array(
        [(i,j) for (i,j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - native.topology.atom(j).residue.index) > 3])
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    r = md.compute_distances(traj, native_contacts)
    r0 = md.compute_distances(native[0], native_contacts)
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q


def ring_closure_dist(traj, acceptor):
    """Distance (nm) between N of residue 1 and the carboxylate carbon of the acceptor residue."""
    end_res_type = re.findall(r'[A-Za-z]+', acceptor)[0]
    end_res_idx = int(re.findall(r'\d+', acceptor)[0]) - 1

    if end_res_type == 'Asp':
        carbon_atom_name = 'CG'
    elif end_res_type == 'Glu':
        carbon_atom_name = 'CD'
    else:
        raise ValueError(f"Unexpected acceptor residue type: {end_res_type}. Expected Asp or Glu.")

    n_atom_idx = traj.topology.select("resid 0 and name N")
    carbon_atom_idx = traj.topology.select(f"resid {end_res_idx} and name {carbon_atom_name}")
    if len(n_atom_idx) == 0 or len(carbon_atom_idx) == 0:
        raise ValueError(f"Could not find ring-closure atoms for acceptor {acceptor}")

    atom_pairs = np.array([[n_atom_idx[0], carbon_atom_idx[0]]])
    return md.compute_distances(traj, atom_pairs)[:, 0]


def cal_features(lasso, traj_files, prmtop_file, native, acceptor):
    # xanthomonin-II and subterisin have a disordered C-terminal extension that is
    # excluded from the native contact map.
    q_resid_range = {'xanthomonin-II': 'resid 0 to 13', 'subterisin': 'resid 0 to 16'}
    sel = q_resid_range.get(lasso)
    native_q = native.atom_slice(native.topology.select(sel)) if sel else native

    q_list = []
    all_pairwise_features = []
    ring_close_dist = []
    for traj in tqdm(traj_files):
        t = md.load(traj, top=prmtop_file)
        t_q = t.atom_slice(t.topology.select(sel)) if sel else t
        q_list.append(q_func(t_q, native_q))
        dist = md.compute_contacts(t, contacts='all', scheme='ca')
        all_pairwise_features.append(dist[0])
        ring_close_dist.append(ring_closure_dist(t, acceptor))

    return q_list, all_pairwise_features, ring_close_dist


if __name__=='__main__':

    pwd = ''
    lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin','caulonodin-V','caulosegnin-I','caulosegnin-II', 'chaxapeptin', 'citrocin',
                  'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I', 'xanthomonin-II']
    # ring-closure acceptor residue (isopeptide bond partner of the N-terminus)
    ring_end = ['Glu8', 'Asp9', 'Asp8', 'Glu9', 'Asp9', 'Glu9', 'Glu8', 'Glu9', 'Asp8', 'Glu8',
                'Glu8', 'Glu8', 'Glu9', 'Glu9', 'Asp9', 'Asp9', 'Glu8', 'Glu8', 'Glu7', 'Glu7']

    for l, lasso in enumerate(lasso_name):
        print('Calculating features for ' + lasso)
        os.makedirs(pwd + lasso + '/features/', exist_ok=True)
        native = md.load(pwd + lasso + '/reference/' + lasso + '-nobond-min.ncrst', top = pwd + lasso + '/reference/' + lasso + '-nobond-HMR.prmtop')

        biased_trajs = natsorted(glob.glob(pwd + lasso + '/biased/' + lasso + '_biased_traj_win_*_rep_*.dcd'))
        biased_trajs = [traj for traj in biased_trajs if not traj.endswith('_wat.dcd')]
        biased_prmtop = pwd + lasso + '/biased/' + lasso + '_nowat.prmtop'
        biased_q_list, biased_all_pairwise_features, biased_ring_close_dist = cal_features(
            lasso, biased_trajs, biased_prmtop, native, ring_end[l])
        with open(pwd + lasso + '/features/' + lasso + '_biased_q_list.pickle', 'wb') as f:
            pickle.dump(biased_q_list, f)
        with open(pwd + lasso + '/features/' + lasso + '_biased_all_pairwise_features.pickle', 'wb') as f:
            pickle.dump(biased_all_pairwise_features, f)
        with open(pwd + lasso + '/features/' + lasso + '_biased_ring_close_dist_ca_n.pickle', 'wb') as f:
            pickle.dump(biased_ring_close_dist, f)

        unbiased_trajs = natsorted(glob.glob(pwd + lasso + '/unbiased/' + lasso + '_unbiased_*_traj_*.dcd'))
        unbiased_trajs = [traj for traj in unbiased_trajs if not traj.endswith('_wat.dcd')]
        unbiased_prmtop = pwd + lasso + '/unbiased/' + lasso + '_nowat.prmtop'
        unbiased_q_list, unbiased_all_pairwise_features, unbiased_ring_close_dist = cal_features(
            lasso, unbiased_trajs, unbiased_prmtop, native, ring_end[l])
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_q_list.pickle', 'wb') as f:
            pickle.dump(unbiased_q_list, f)
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_all_pairwise_features.pickle', 'wb') as f:
            pickle.dump(unbiased_all_pairwise_features, f)
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_ring_close_dist_ca_n.pickle', 'wb') as f:
            pickle.dump(unbiased_ring_close_dist, f)
