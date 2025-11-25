"""
Train Variational AutoEncoder (VAE) to map pathways to 2D latent space.

"""

import numpy as np
import os
import glob
import pyemma
from tqdm import trange
import pickle

def microstates_distribution(reassign_trajs, num_clusters, idx1, idx2, x_initial, x_end,  y_initial, y_end, 
                             num_slices, output_dir="microstates_distribution_tica"):
    """
    Use the re-assigned conformations (belonging to every microstates) to get the microstate distribution;
    The simplest way is to discretize/grid the tICA space into bins and count the distribution of conformations
    in each bin; 
    The number of bins increase exponential o(N^3) with the dimensionality of tICA space; To simplify the input 
    of VAE,  we can visualize pathways on each pair of 2-D tICA space and concatenate the partial distributions;
    In this way, the complexity incease much slower o(N^2) with the dimensionarlity of tICA space.
    ---parameters---
    reassign_trajs: the ressigned tICA conformatiosn for every microstates
    num_clusters: the number of microstates (to calculate distribution)
    idx1: the dimensional index of the first embedding coordinate
    idx2: the dimensional index of the second embedding coordinate
    x_initial: the minimum value of the first embedding coordinate
    x_end: the maximum value of the first embedding coordinate
    y_initial: the minimum value of the second embedding coordinate
    y_end: the maximum value of the second embedding coordinate
    num_slices: the number of slices/bins to discretize along each coordinate
    output_dir: the output directory of the microstate distribution
    """

    if os.path.exists(output_dir):
        print("Attention! The ouput file already exists!")
    else:
        os.makedirs(output_dir)

    xdelta = (x_end - x_initial) / num_slices
    ydelta = (y_end - y_initial) / num_slices
    for i in range(num_clusters):
        dist = np.zeros((num_slices, num_slices))
        print("For No {} state, in total, there are {} frames;".format(i, len(reassign_trajs[i])))
        for j in range(0, len(reassign_trajs[i]), 1):
            x = (reassign_trajs[i][j, idx1]-x_initial) // xdelta
            y = (reassign_trajs[i][j, idx2]-y_initial) // ydelta
            if 0 < x < num_slices and 0 < y< num_slices:
                dist[num_slices-int(y)-1, int(x)] += 1
        dist = dist / (len(reassign_trajs[i]))
        np.save(output_dir + "/%03d_state_distribution.npy"%(i), dist)

        print("No {} state is completed for distribution calculation;".format(i))

pwd = ''
lasso_name = ['microcinJ25']
cluster_number = [400]
tram_lag = [150]
tic_dim = [10]

for m, lasso in enumerate(lasso_name):
    lasso = lasso_name[m]
    print(lasso)
    # biased_feature = pickle.load(open(pwd + lasso + '/features/' + lasso + '_biased_all_pairwise_features.pickle','rb'))
    # print(len(biased_feature))
    # print(biased_feature[0].shape)
    # unbiased_feature = pickle.load(open(pwd + lasso + '/features/' + lasso + '_unbiased_all_pairwise_features.pickle','rb'))
    # print(len(unbiased_feature))
    # print(unbiased_feature[0].shape)
    # all_feature = biased_feature + unbiased_feature
    # print(len(all_feature))
    # print(all_feature[0].shape)
    
    # tic = pyemma.coordinates.tica(all_feature,lag= lag_time[m], dim=tic_dim[m])
    # data_tic = tic.get_output()
    data_tic = pickle.load(open(pwd + lasso + '/tram_files/' + lasso + '_tram_tica_dim_' +str(tic_dim[m]) + '_cluster_' + str(cluster_number[m]) + '_lag_' + str(tram_lag[m]) + '.pkl','rb'))
    
    data_tic_connected = np.concatenate(data_tic)
    
    dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' +lasso+'_dtrajs_cluster_' + str(cluster_number[m]) + '_lag_' + str(tram_lag[m]) + '.pkl','rb'))
    print(len(dtrajs), dtrajs[0].shape, dtrajs[-1].shape)
    
    
    ## Set the reassign step, number of microstates, dimensionality of tICA
    step = 1
    reassign_trajs = {}
    for i in trange(len(dtrajs)):
        for j in range(0, len(dtrajs[i]), step):
            try:
                reassign_trajs[int(dtrajs[i][j])] = np.vstack((reassign_trajs[int(dtrajs[i][j])], data_tic[i][j]))
            except KeyError:
                reassign_trajs[int(dtrajs[i][j])] = data_tic[i][j]
    np.save("reassign_tica2micro.npy", reassign_trajs)    
    
    microstates_distribution(reassign_trajs=reassign_trajs, num_clusters= cluster_number[m], idx1=0, idx2=1, x_initial=np.floor(np.min(data_tic_connected[:,0])), x_end=np.ceil(np.max(data_tic_connected[:,0])), y_initial=np.floor(np.min(data_tic_connected[:,1])), y_end=np.ceil(np.max(data_tic_connected[:,1])), num_slices=50, output_dir= "microstates_distribution_tica01")
    
    microstates_distribution(reassign_trajs=reassign_trajs, num_clusters= cluster_number[m], idx1=0, idx2=2, x_initial=np.floor(np.min(data_tic_connected[:,0])), x_end=np.ceil(np.max(data_tic_connected[:,0])), y_initial=np.floor(np.min(data_tic_connected[:,2])), y_end=np.ceil(np.max(data_tic_connected[:,2])), num_slices=50, output_dir= "microstates_distribution_tica02")
    
    microstates_distribution(reassign_trajs=reassign_trajs, num_clusters= cluster_number[m], idx1=1, idx2=2, x_initial=np.floor(np.min(data_tic_connected[:,1])), x_end=np.ceil(np.max(data_tic_connected[:,1])), y_initial=np.floor(np.min(data_tic_connected[:,2])), y_end=np.ceil(np.max(data_tic_connected[:,2])), num_slices=50, output_dir= "microstates_distribution_tica12")
    
    # Load the pathways identified by Transition Path Theory, each pathway is a sequence of state indexes
    paths = np.load(pwd + lasso + '/pathways/' + lasso + "_TPT_pathways.npy", allow_pickle=True)
    print(paths.shape[0])

    ## Input number of pathways with largest flux to embed
    num_pathways = paths.shape[0]
    ## The directories of state distribution (in each pair of collective variables space)
    dirc1 = 'microstates_distribution_tica01'
    dirc2 = 'microstates_distribution_tica02'
    dirc3 = 'microstates_distribution_tica12'
    num_slices = 50
    for i in range(0, num_pathways):
        f = paths[i]
        dist = np.zeros((3*num_slices, num_slices))
        temp = 0
        for j in range(len(f)):
            mat1 = np.load("./"+dirc1+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
            mat2 = np.load("./"+dirc2+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
            mat3 = np.load("./"+dirc3+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
    
            dist[:num_slices] = dist[:num_slices] + mat1 
            dist[num_slices:2*num_slices] = dist[num_slices:2*num_slices] + mat2 
            dist[2*num_slices:3*num_slices] = dist[2*num_slices:3*num_slices] + mat3 
    
        print("No {} transition pathway is calculated as distribution;".format(i))
        os.makedirs('tpt_all_path_distribution', exist_ok=True)
        np.save("./tpt_all_path_distribution/No_%04d_path_distribution.npy"%i, dist)    
