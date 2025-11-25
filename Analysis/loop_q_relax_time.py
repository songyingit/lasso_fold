"""
Calculate the loop Fraction of Native Contacts (Q) relaxation time to study lasso peptide loop stability

"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import pyemma
import pyemma.msm as msm
from tqdm import tqdm
import os
import pandas as pd

pwd = ''

lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin','caulonodin-V','caulosegnin-I', 'caulosegnin-II', 'chaxapeptin', 'citrocin', 
              'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I', 'xanthomonin-II']
cluster_number = [400, 100, 100, 100, 100, 200, 500, 100, 300, 200, 
                  400, 400, 300, 200, 500, 200, 100, 200, 300, 400]
tram_lag = [150, 150, 150, 150, 100, 150, 150, 150, 150, 150,
            150, 150, 150, 100, 150, 150, 150, 150, 150, 100]

x_time_scales = []
y_acf = []


# parameters
rel_tol   = 0.001       # relative tolerance: 1% of total signal change
n_consec  = 5          # require 5 successive small‐change steps
results   = []

# detect plateau
def detect_plateau(times, signal, rel_tol, n_consec):
    total_drop = signal[0] - signal[-1]
    tol = rel_tol * total_drop
    diffs = np.abs(np.diff(signal))
    # search for first window of n_consec consecutive diffs < tol
    for i in range(len(diffs) - n_consec + 1):
        if np.all(diffs[i : i + n_consec] < tol):
            # plateau time is times[i+1], since diffs[i] = |sig[i+1]-sig[i]|
            return times[i+1]
    return np.nan  # didn’t plateau within computed window

for i, lasso in tqdm(enumerate(lasso_name)):
      print(lasso)
      biased_q_list = pickle.load(open(pwd + lasso + '/features/' + lasso + '_biased_q_loop_list.pickle', 'rb'))
      unbiased_q_list = pickle.load(open(pwd + lasso + '/features/' + lasso + '_unbiased_q_loop_list.pickle', 'rb'))
      loop_NC =  biased_q_list + unbiased_q_list
      dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' +lasso+'_dtrajs_cluster_' + str(cluster_number[i]) + '_lag_' + str(tram_lag[i]) + '.pkl','rb'))
      tram_msm = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_ther_obj_cluster_' + str(cluster_number[i]) + '_lag_'+str(tram_lag[i])+'_msm_obj.pkl','rb'))
      P = tram_msm.transition_matrix
      M = msm.markov_model(P)


      n_states = cluster_number[i]
      obs      = np.zeros(n_states)
      counts   = np.zeros(n_states)

      for traj_states, q_vals in zip(dtrajs, loop_NC):
            for state, q in zip(traj_states, q_vals):
                  obs[state]  += q
                  counts[state] += 1

      mask = counts > 0
      obs[mask] /= counts[mask]

      
      p0 = np.zeros(n_states)
      for dtraj in dtrajs:
            p0[dtraj[0]] += 1
 
      p0 /= p0.sum()
  
      # relaxation
      times, exp_a = M.relaxation(p0, obs, maxtime=None)

      # convert to real time if you want (e.g. frames → ns, or τ‐units)
      real_times = times * tram_lag[i] / 10

      t_plateau = detect_plateau(real_times, exp_a, rel_tol, n_consec)

      results.append({
            'peptide':           lasso,
            't_plateau (ns)': t_plateau
      })


for res in results:
      lasso = res['peptide']
      relax_time = res['t_plateau (ns)']
      print(lasso, relax_time)
      csv_file = os.path.join("")
      df = pd.read_csv(csv_file)
      df.loc[df['Lasso Name'] == lasso, 'Loop Relaxation Time'] = relax_time
      df.to_csv(csv_file, index=False)     