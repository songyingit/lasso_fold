"""
Calculate the features needed for TRAM analysis from biased and unbiased MD trajectories.

"""

import pickle
import mdtraj as md
import numpy as np
from tqdm import tqdm
from itertools import combinations
import os
import glob
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

def cal_features(lasso, traj_files, prmtop_file, native, ring_size):
    q_list = []
    all_pairwise_features = []
    ring_close_dist = []
    for i, traj in tqdm(enumerate(traj_files)):
        t = md.load(traj, top = prmtop_file)
        if lasso != 'xanthomonin-II' and  lasso != 'subterisin':
            q = q_func(t, native)
        elif lasso == 'xanthomonin-II':
            t = t.atom_slice(t.topology.select('resid 0 to 13'))
            native = native.atom_slice(native.topology.select('resid 0 to 13'))
            q = q_func(t, native)
        elif lasso == 'subterisin':
            t = t.atom_slice(t.topology.select('resid 0 to 16'))
            native = native.atom_slice(native.topology.select('resid 0 to 16'))
            q = q_func(t, native)
        q_list.append(q)
        dist = md.compute_contacts(t, contacts = 'all', scheme = 'ca')
        all_pairwise_features.append(dist[0])
        ring_close_dist.append(dist[0][:, ring_size - 4])

    return q_list, all_pairwise_features, ring_close_dist

if __name__=='__main__':

    pwd = '/home/xuenan/storage/Research/Lasso/analysis_new/analysis_lasso_new/TRAM/writing_papers/data/'
    lasso_name = ['caulosegnin-II', 'astexin-1', 'benenodin-1', 'brevunsin', 'caulonodin-V', 'caulosegnin-I','acinetodin', 'capistruin', 'microcinJ25', 'chaxapeptin','citrocin','klebsidin', 'rubrivinodin', 'sphaericin', 'sphingopyxin-I', 'streptomonomicin', 'subterisin', 'ubonodin', 'xanthomonin-I', 'xanthomonin-II'] 
    ring_len = [9, 9, 8, 9, 9, 8, 8, 9, 8, 8, 8, 8, 9, 9, 9, 9, 8, 8, 7, 7]

    for l, lasso in enumerate(lasso_name):
        print('Calculating features for ' + lasso)
        os.makedirs(pwd + lasso + '/features/', exist_ok=True)

        biased_trajs = natsorted(glob.glob(pwd + lasso + '/biased/' + lasso + '_biased_traj_win_*_rep_*.dcd'))
        biased_trajs = [traj for traj in biased_trajs if not traj.endswith('_wat.dcd')]
        print(len(biased_trajs))
        biased_prmtop = pwd + lasso + '/biased/' + lasso + '_nowat.prmtop'
        native = md.load(pwd + lasso + '/reference/' + lasso + '-nobond-min.ncrst', top = pwd + lasso + '/reference/' + lasso + '-nobond-HMR.prmtop') 
        biased_q_list, biased_all_pairwise_features, biased_ring_close_dist = cal_features(lasso, biased_trajs, biased_prmtop, native, ring_len[l])
        print(len(biased_q_list), biased_q_list[0].shape)
        print(len(biased_all_pairwise_features), biased_all_pairwise_features[0].shape)
        print(len(biased_ring_close_dist), biased_ring_close_dist[0].shape)
        with open(pwd + lasso + '/features/' + lasso + '_biased_q_list.pickle', 'wb') as f:
            pickle.dump(biased_q_list, f)
        with open(pwd + lasso + '/features/' + lasso + '_biased_all_pairwise_features.pickle', 'wb') as f:
            pickle.dump(biased_all_pairwise_features, f)
        with open(pwd + lasso + '/features/' + lasso + '_biased_ring_close_dist.pickle', 'wb') as f:
            pickle.dump(biased_ring_close_dist, f)

        unbiased_trajs = natsorted(glob.glob(pwd + lasso + '/unbiased/' + lasso + '_unbiased_*_traj_*.dcd'))
        unbiased_trajs = [traj for traj in unbiased_trajs if not traj.endswith('_wat.dcd')]
        print(len(unbiased_trajs))
        unbiased_prmtop = pwd + lasso + '/unbiased/' + lasso + '_nowat.prmtop'
        native = md.load(pwd + lasso + '/reference/' + lasso + '-nobond-min.ncrst', top = pwd + lasso + '/reference/' + lasso + '-nobond-HMR.prmtop') 
        unbiased_q_list, unbiased_all_pairwise_features, unbiased_ring_close_dist = cal_features(lasso, unbiased_trajs, unbiased_prmtop, native, ring_len[l])
        print(len(unbiased_q_list), unbiased_q_list[0].shape)
        print(len(unbiased_all_pairwise_features), unbiased_all_pairwise_features[0].shape)
        print(len(unbiased_ring_close_dist), unbiased_ring_close_dist[0].shape)
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_q_list.pickle', 'wb') as f:
            pickle.dump(unbiased_q_list, f)
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_all_pairwise_features.pickle', 'wb') as f:
            pickle.dump(unbiased_all_pairwise_features, f)
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_ring_close_dist.pickle', 'wb') as f:
            pickle.dump(unbiased_ring_close_dist, f)
        
    





