"""
Calculate the folding free energy for each lasso peptide system

"""

import numpy as np
import pickle
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 
import tol_colors as tc
tol_cmap = tc.tol_cmap('rainbow_PuBr')

# User parameters
pwd = ''

lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin','caulonodin-V','caulosegnin-I','caulosegnin-II', 'chaxapeptin', 'citrocin', 
                  'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I', 'xanthomonin-II'] 
cluster_number = [400, 100, 100, 100, 100, 200, 500, 100, 300, 200, 
                  400, 400, 300, 200, 500, 200, 100, 200, 300, 400]
tram_lag = [150, 150, 150, 150, 100, 150, 150, 150, 150, 150,
                  150, 150, 150, 100, 150, 150, 150, 150, 150, 100]

# Thermodynamic constants
R = 0.001987  # kcal/mol·K
T = 300       # Kelvin

# Bootstrap settings
n_bootstrap = 100
bootstrap_frac = 0.8

results = []

for i, lasso in enumerate(lasso_name):
    print(lasso)
    # Load TRAM object and compute weights
    tram_obj = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_ther_obj_cluster_' + str(cluster_number[i]) + '_lag_'+str(tram_lag[i])+'_msm_obj.pkl','rb'))
    # print(tram_obj)
    stat_dis = tram_obj.stationary_distribution_full_state
    dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_dtrajs_cluster_' + str(cluster_number[i]) + '_lag_'+str(tram_lag[i])+'.pkl','rb'))
    # print(len(dtrajs), len(dtrajs[0]), len(dtrajs[-1]))
    txx_dtrajs = np.concatenate(dtrajs)
    # print(len(txx_dtrajs))
    unique_clusters = np.unique(txx_dtrajs)
    number_per_uni_clus = [len(np.where(txx_dtrajs==i)[0]) for i in unique_clusters]
    tram_weights =  np.array([stat_dis[txx_dtrajs[i]]/number_per_uni_clus[txx_dtrajs[i]] for i in range(len(txx_dtrajs))])
    # print(tram_weights.shape)

    # Load Q (fraction native contacts)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_biased_q_list.pickle"), 'rb') as f:
        biased_q = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_unbiased_q_list.pickle"), 'rb') as f:
        unbiased_q = pickle.load(f)
    q_connected = np.concatenate(biased_q+unbiased_q, axis=0)
    # print(q_connected.shape)

    # Load ring_closure distance
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_biased_ring_close_dist.pickle"), 'rb') as f:
        biased_ring_close = pickle.load(f)
    with open(os.path.join(pwd, lasso, 'features', f"{lasso}_unbiased_ring_close_dist.pickle"), 'rb') as f:
        unbiased_ring_close = pickle.load(f)
    ring_close_connected = np.concatenate(biased_ring_close + unbiased_ring_close, axis=0)


    # Bootstrap ΔG calculation
    delta_g_vals = []
    n_frames = len(q_connected)
    sample_size = int(bootstrap_frac * n_frames)
    delta_g_vals = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_frames, size=sample_size, replace=True)
        w_sample = tram_weights[idx]
        q_sample = q_connected[idx]
        rc_sample = ring_close_connected[idx]
        p_f = w_sample[(q_sample >= 0.8) & (rc_sample <= 0.7)].sum()
        p_u = w_sample[(q_sample <= 0.1) & (rc_sample > 0.7)].sum()
        delta_g_vals.append(-R * T * np.log(p_f / p_u))

    mean_dg = np.mean(delta_g_vals)
    stderr_dg = np.std(delta_g_vals)
    results.append((lasso, mean_dg, stderr_dg))         
