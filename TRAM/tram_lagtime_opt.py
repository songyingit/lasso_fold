"""
Optimize the lag time for TRAM analysis

"""

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 
import pickle
import pyemma

pwd = ''

lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin','caulonodin-V','caulosegnin-I','caulosegnin-II', 'chaxapeptin', 'citrocin', 'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I','xanthomonin-II'] 
tic_lag_time =  [250, 300, 300, 300, 250, 300,250, 300, 300, 300, 300, 300, 350, 300, 300, 350, 300, 300, 250, 250] 
cluster_number = [400, 100, 100, 100, 100, 200,500, 100, 400, 200, 400, 400, 300, 400, 700, 200, 100, 200, 300, 400] 
tic_dims =  [4,   8,   8,  6,   10,    8,  8,  10,   8,   8,   10,  10, 10,  8,   6,   10,  6,   6,   10, 10] 

for i, lasso in enumerate(lasso_name):
      print('Processing ' + lasso)
      cluster = cluster_number[i]
      tic_dim = tic_dims[i]
      lag = 150
      fig,axs = plt.subplots(1,1,figsize=(10,7),constrained_layout=True)

      dtrajs = pickle.load(open(pwd + lasso + '/tram_files/' +lasso+'_dtrajs_cluster_' + str(cluster) + '_lag_' + str(lag) + '.pkl','rb'))
      print(len(dtrajs), dtrajs[0].shape, dtrajs[-1].shape)

      its = pyemma.msm.its(dtrajs=dtrajs, lags=[50, 100, 150], nits=5, errors='bayes')
      pyemma.plots.plot_implied_timescales(its,ax=axs, units ='ns',dt=0.1)
      axs.set_xlim(5, 15)
      axs.set_xticks([5, 10, 15])
      axs.tick_params(axis='x', labelsize=22)
      axs.tick_params(axis='y', labelsize=22)
      axs.set_xlabel('Lag Time (ns)', fontsize=30)
      axs.set_ylabel('Implied Timescale (ns)', fontsize=30)
      fig.savefig(f'{lasso}_tram_lagtime_plot.png')
      plt.close(fig)