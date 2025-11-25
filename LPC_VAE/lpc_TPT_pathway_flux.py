"""
Apply Transition path theory (TPT) to obtain folding pathways and calculate the corresponding flux. 

"""

import numpy as np
import mdtraj as md
from itertools import combinations
import matplotlib.pyplot as plt
import glob
import pickle
import pandas as pd
import pyemma
import os

pwd = ''
lasso_name = ['microcinJ25']
cluster_number = [400]
tram_lag = [150]

for i, lasso in enumerate(lasso_name):
    print(lasso)
    tram_msm = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_ther_obj_cluster_' + str(cluster_number[i]) + '_lag_' + str(tram_lag[i]) + '_msm_obj.pkl','rb'))
    print(tram_msm)
    dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_dtrajs_cluster_' + str(cluster_number[i]) + '_lag_' + str(tram_lag[i]) + '.pkl','rb'))
    print(len(dtrajs),dtrajs[0].shape, dtrajs[-1].shape)
    biased_NC_list = pickle.load(open(pwd + lasso + '/features/' + lasso + '_biased_q_list.pickle','rb'))
    print(len(biased_NC_list),biased_NC_list[0].shape,biased_NC_list[-1].shape)
    unbiased_NC_list = pickle.load(open(pwd + lasso + '/features/' + lasso + '_unbiased_q_list.pickle','rb'))
    print(len(unbiased_NC_list),unbiased_NC_list[0].shape,unbiased_NC_list[-1].shape)
    NC = biased_NC_list + unbiased_NC_list
    print(len(NC), NC[0].shape, NC[-1].shape)
    
    with open(pwd + lasso + '/MFPT/' + lasso + '_cluster_' + str(cluster_number[i]) + '_NC.pkl','rb') as f:
        cluster_NC = pickle.load(f)     

    cluster_indices = sorted(range(len(cluster_NC)), key=lambda i: cluster_NC[i], reverse=True)

    folded_state = [index for index, value in enumerate(cluster_NC) if value > 0.8]
    if len(folded_state) == 0:
        max_NC_state = [cluster_indices[0]]
        folded_state = max_NC_state
    unfolded_state = [index for index, value in enumerate(cluster_NC) if value < 0.1]
    print(len(folded_state), folded_state)
    print(len(unfolded_state), unfolded_state)

    os.makedirs(pwd + lasso + '/pathways/', exist_ok=True)

    TPT = pyemma.msm.tpt(tram_msm, unfolded_state, folded_state)
    print('TPT Done')
    pickle.dump(TPT, open(pwd + lasso + '/pathways/' + lasso + '_TPT_obj.pkl','wb'))

    pathways = TPT.pathways(fraction=1.0, maxiter=10000)

    flux = pathways[1]
    flux_res = [] 
    cumulative_flux_res = []
    for m in range(len(flux)): 
        flux_res.append(100*flux[m] / TPT.total_flux) 
        cumulative_flux_res.append(np.sum(flux_res))
    #print(cumulative_flux_res)
    len_flux = len(cumulative_flux_res)  ##3013
    print(len_flux)
    # len_flux = 10000
    fig, axs = plt.subplots(1,1, figsize = (10,7)) 
    axs.scatter(list(range(1, len_flux + 1)), cumulative_flux_res[:len_flux], s = 15, color = 'black') 
    axs.plot(list(range(1, len_flux + 1)), cumulative_flux_res[:len_flux], color = 'black') 
    axs.spines['bottom'].set_linewidth(2.0)
    axs.spines['left'].set_linewidth(2.0)
    axs.spines['top'].set_linewidth(2.0)
    axs.spines['right'].set_linewidth(2.0)

    plt.yticks([0,20,40,60,80,100], fontsize = 22) 
    plt.xticks([0,2000,4000,6000,8000,10000], fontsize = 18) 
    plt.xlabel('Number of pathways', fontsize = 24) 
    plt.ylabel('Relative Flux (percent)', fontsize = 24) 
    plt.savefig(pwd + lasso + '/pathways/' + lasso + '_Flux_plot.png', dpi = 300, transparent = True) 
    plt.close() 

    committor = TPT.committor
    
    np.save(pwd + lasso + '/pathways/' + lasso + '_TPT_pathways.npy',pathways[0])
    np.save(pwd + lasso + '/pathways/' + lasso + '_TPT_pathways_flux.npy',pathways[1])
    np.save(pwd + lasso + '/pathways/' + lasso + '_TPT_committor.npy',committor)

    
    
