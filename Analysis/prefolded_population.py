"""
TRAM-weighted probability of the pre-folded conformation for each lasso peptide.

Pre-folded state: Q >= 0.8 and ring-closure distance <= 6 A.
"""

import numpy as np
import pickle
import os

# User parameters
pwd = ''

lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin','caulonodin-V','caulosegnin-I','caulosegnin-II', 'chaxapeptin', 'citrocin',
                  'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I', 'xanthomonin-II']
cluster_number = [400, 100, 100, 100, 100, 200, 500, 100, 300, 200,
                  400, 400, 300, 200, 500, 200, 100, 200, 300, 400]
tram_lag = [150, 150, 150, 150, 100, 150, 150, 150, 150, 150,
            150, 150, 150, 100, 150, 150, 150, 150, 150, 100]

# State definition
Q_FOLDED = 0.8
RC_CUTOFF = 0.6   # nm (6 A)

# Bootstrap settings
n_bootstrap = 100
bootstrap_frac = 0.8

results = []

for i, lasso in enumerate(lasso_name):
    print(lasso)
    # Load TRAM object and compute weights
    tram_obj = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_ther_obj_cluster_' + str(cluster_number[i]) + '_lag_'+str(tram_lag[i])+'_msm_obj.pkl','rb'))
    stat_dis = tram_obj.stationary_distribution_full_state
    dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_dtrajs_cluster_' + str(cluster_number[i]) + '_lag_'+str(tram_lag[i])+'.pkl','rb'))
    txx_dtrajs = np.concatenate(dtrajs)
    unique_clusters = np.unique(txx_dtrajs)
    number_per_uni_clus = [len(np.where(txx_dtrajs==i)[0]) for i in unique_clusters]
    tram_weights =  np.array([stat_dis[txx_dtrajs[i]]/number_per_uni_clus[txx_dtrajs[i]] for i in range(len(txx_dtrajs))])

    # Load Q (fraction of native contacts)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_biased_q_list.pickle"), 'rb') as f:
        biased_q = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_unbiased_q_list.pickle"), 'rb') as f:
        unbiased_q = pickle.load(f)
    q_connected = np.concatenate(biased_q+unbiased_q, axis=0)

    # Load ring-closure distance
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_biased_ring_close_dist_ca_n.pickle"), 'rb') as f:
        biased_ring_close = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_unbiased_ring_close_dist_ca_n.pickle"), 'rb') as f:
        unbiased_ring_close = pickle.load(f)
    ring_close_connected = np.concatenate(biased_ring_close + unbiased_ring_close, axis=0)

    # Bootstrap the TRAM-weighted pre-folded percentage
    n_frames = q_connected.shape[0]
    sample_size = int(bootstrap_frac * n_frames)
    bootstrap_perc = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_frames, size=sample_size, replace=False)
        q_bs = q_connected[idx]
        rc_bs = ring_close_connected[idx]
        w_bs = tram_weights[idx]
        pct = np.sum(w_bs[(q_bs >= Q_FOLDED) & (rc_bs <= RC_CUTOFF)]) / np.sum(w_bs) * 100
        bootstrap_perc.append(pct)
    bootstrap_perc = np.array(bootstrap_perc)

    results.append((lasso, bootstrap_perc.mean(), bootstrap_perc.std()))
    print(f"[{lasso}] pre-folded population = {bootstrap_perc.mean():.4f} +/- {bootstrap_perc.std():.4f} %")
