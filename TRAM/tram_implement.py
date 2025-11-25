"""
Transition-based Reweighting Analysis Method (TRAM) implementation

"""

import numpy as np 
import pickle 
import pyemma
from tqdm import tqdm
import mdtraj as md
import glob
import os
from tqdm import tqdm
from natsort import natsorted
import re

def cal_biased_win_traj_no(lasso, biased_trajs):
    win_re = re.compile(rf"{lasso}_biased_traj_win_(\d+)_rep_(\d+).dcd")
    window_dict = {}
    for fn in biased_trajs:
        m = win_re.search(os.path.basename(fn))
        win = int(m.group(1))
        window_dict.setdefault(win, []).append(fn)
    window_counts = {win: len(trajs) for win, trajs in window_dict.items()}

    return window_counts

def load_ref_frames(pwd, lasso, win_traj_no):
    ref_frames = []
    for win in tqdm(win_traj_no):
        t = md.load(pwd + lasso + '/biased/ref_frames/' + lasso + '_biased_US_win_' + str(win) + '_ref_frame.ncrst', top=pwd + lasso + '/biased/ref_frames/' + lasso + '_biased_US_win_' + str(win) + '_ref_frame.prmtop')
        t_protein = t.atom_slice(t.topology.select('protein'))
        ref_frames.append(t_protein)

    return ref_frames

def cal_biased_energy(biased_trajs, biased_prmtop, biased_features, unbiased_trajs, unbiased_prmtop, unbiased_features, ref_frames, win_traj_no):
    biased_energy = []
    for traj in tqdm(range(len(biased_features))):
        traj_bias = np.zeros([len(biased_features[traj]),len(win_traj_no)+1])
        t = md.load(biased_trajs[traj], top=biased_prmtop)
        t_protein = t.atom_slice(t.topology.select('protein'))
        for w in range(len(win_traj_no)):
            rmsd = md.rmsd(t_protein, ref_frames[w])
            traj_bias[:,w] = np.array([20*(rmsd*10)**2])
        biased_energy.append(traj_bias)
    for traj in tqdm(range(len(unbiased_features))):
        traj_bias = np.zeros([len(unbiased_features[traj]),len(win_traj_no)+1])
        t = md.load(unbiased_trajs[traj], top=unbiased_prmtop)
        t_protein = t.atom_slice(t.topology.select('protein'))
        for w in range(len(win_traj_no)):
            rmsd = md.rmsd(t_protein, ref_frames[w])
            traj_bias[:,w] = np.array([20*(rmsd*10)**2])
        biased_energy.append(traj_bias)
    
    return biased_energy

def cal_ttrajs(win_traj_no, biased_features, unbiased_features): 
    ttrajs = []
    biased_traj_index = []
    for w in range(len(win_traj_no)):
        for r in range(win_traj_no[w]):
            biased_traj_index.append(w)
    for i in range(len((biased_features))):
        traj_length = len(biased_features[i])
        ttrajs.append(np.array([int(biased_traj_index[i])]*traj_length))
    for i in range(len(unbiased_features)):
        traj_length = len(unbiased_features[i])
        ttrajs.append(np.array([len(win_traj_no)]*traj_length))
    
    return ttrajs

def cal_dtrajs(biased_features,unbiased_features,cluster,lag_time,dim):
    biased_tic = pyemma.coordinates.tica(biased_features,lag=lag_time,dim=dim)
    data_biased_tic = biased_tic.get_output()
    transformed_biased_tic = biased_tic.transform(unbiased_features)

    unbiased_tic = pyemma.coordinates.tica(unbiased_features,lag=lag_time,dim=dim)
    data_unbiased_tic = unbiased_tic.get_output()
    transformed_unbiased_tic = unbiased_tic.transform(biased_features)
    
    data_tic = []
    for b,d in zip(data_biased_tic,transformed_unbiased_tic):
        temp = np.concatenate((b,d),axis=1)
        data_tic.append(temp)

    for b,d in zip(transformed_biased_tic,data_unbiased_tic):
        temp = np.concatenate((b,d),axis=1)
        data_tic.append(temp)

    dtrajs = pyemma.coordinates.cluster_kmeans(data_tic,k=cluster,max_iter=100, tolerance=1e-05, stride=2).dtrajs
    
    return dtrajs

def tram_implementation(ttrajs, dtrajs, bias, lag_time, win_traj_no):
    ther_obj = pyemma.thermo.tram(ttrajs, dtrajs, bias, lag=lag_time, unbiased_state = len(win_traj_no), maxerr=1e-02, init_maxerr=1e-02)
    
    return ther_obj

if __name__=='__main__':
    
    lasso_name =  ['']
    cluster_number = []
    tic_dims = []

    for lasso_idx, lasso in enumerate(lasso_name):
        print(lasso)
        pwd = ''

        biased_trajs = natsorted(glob.glob(pwd + lasso + '/biased/' + lasso + '_biased_traj_win_*_rep_*.dcd'))
        # biased_trajs = [traj for traj in biased_trajs if not traj.endswith('_wat.dcd')]
        print(len(biased_trajs))
        biased_prmtop = pwd + lasso + '/biased/' + lasso + '_nowat.prmtop'
        unbiased_trajs = natsorted(glob.glob(pwd + lasso + '/unbiased/' + lasso + '_unbiased_*_traj_*.dcd'))
        # unbiased_trajs = [traj for traj in unbiased_trajs if not traj.endswith('_wat.dcd')]
        print(len(unbiased_trajs))
        unbiased_prmtop = pwd + lasso + '/unbiased/' + lasso + '_nowat.prmtop'
        
        with open(pwd + lasso + '/features/' + lasso + '_biased_all_pairwise_features.pickle', 'rb') as f:
            biased_features = pickle.load(f)
        with open(pwd + lasso + '/features/' + lasso + '_unbiased_all_pairwise_features.pickle', 'rb') as f:
            unbiased_features = pickle.load(f)
        win_traj_no = cal_biased_win_traj_no(lasso, biased_trajs)
        print(win_traj_no)
        ref_frames = load_ref_frames(pwd, lasso, win_traj_no)
        # print(len(ref_frames))

        os.makedirs(pwd + lasso + '/tram_files/',exist_ok=True)

        # biased_energy = cal_biased_energy(biased_trajs, biased_prmtop, biased_features, unbiased_trajs, unbiased_prmtop, unbiased_features, ref_frames, win_traj_no)
        # print(len(biased_energy), biased_energy[0].shape, biased_energy[-1].shape)
        # pickle.dump(biased_energy,open(pwd + lasso + '/tram_files/'+lasso+'_bias_energy.pkl','wb'))
        biased_energy = pickle.load(open(pwd + lasso + '/tram_files/'+lasso+'_bias_energy.pkl','rb'))
        print(len(biased_energy), biased_energy[0].shape, biased_energy[-1].shape)
        
        # ttrajs = cal_ttrajs(win_traj_no, biased_features, unbiased_features)
        # print(len(ttrajs), ttrajs[0].shape, ttrajs[-1].shape)
        # pickle.dump(ttrajs, open(pwd + lasso + '/tram_files/'+lasso+'_ttrajs.pkl','wb')) 
        ttrajs = pickle.load(open(pwd + lasso + '/tram_files/'+lasso+'_ttrajs.pkl','rb'))
        print(len(ttrajs), ttrajs[0].shape, ttrajs[-1].shape)
        
        cluster = cluster_number[lasso_idx]
        lag = 150
        tic_dim = tic_dims[lasso_idx]
        
        dtrajs = cal_dtrajs(biased_features,unbiased_features,cluster,lag,tic_dim)
        print(len(dtrajs), dtrajs[0].shape, dtrajs[-1].shape)
        pickle.dump(dtrajs,open(pwd + lasso + '/tram_files/' +lasso+'_dtrajs_cluster_' + str(cluster) + '_lag_' + str(lag) + '.pkl','wb'))
        dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' +lasso+'_dtrajs_cluster_' + str(cluster) + '_lag_' + str(lag) + '.pkl','rb'))
        print(len(dtrajs), dtrajs[0].shape, dtrajs[-1].shape)

        print("start TRAM")
        ther_obj = tram_implementation(ttrajs, dtrajs, biased_energy, lag, win_traj_no)
        print("save TRAM")
        pickle.dump(ther_obj,open(pwd + lasso + '/tram_files/' +lasso+'_ther_obj_cluster_' + str(cluster) + '_lag_' + str(lag) + '_obj.pkl','wb'))
        print("save TRAM MSM")
        pickle.dump(ther_obj.msm,open(pwd + lasso + '/tram_files/' +lasso+'_ther_obj_cluster_' + str(cluster) + '_lag_' + str(lag) + '_msm_obj.pkl','wb'))






