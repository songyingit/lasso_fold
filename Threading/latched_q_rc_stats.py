"""
Pre-lasso / pre-tadpole statistics restricted to the pre-folded state definition,
i.e. frames with Q >= 0.8 AND ring-closure distance <= 6 A.

Reuses the LATCHED classifications written by latched_split.py (which are ordered
over the rc-filtered frames) and appends the results to latched_split_summary.csv.
"""

import os, pickle
import numpy as np
import pandas as pd

# User parameters
pwd = ''
OUT_DIR = 'latched_split'
LOG_DIR = os.path.join(OUT_DIR, 'latched_output')

Q_CUTOFF  = 0.8
RC_CUTOFF = 0.6   # nm (6 A)

lasso_name = [
    'acinetodin','astexin-1','benenodin-1','brevunsin','capistruin',
    'caulonodin-V','caulosegnin-I','caulosegnin-II','chaxapeptin','citrocin',
    'klebsidin','microcinJ25','rubrivinodin','sphaericin','sphingopyxin-I',
    'streptomonomicin','subterisin','ubonodin','xanthomonin-I','xanthomonin-II',
]
cluster_number = [400,100,100,100,100,200,500,100,300,200,
                  400,400,300,200,500,200,100,200,300,400]
tram_lag       = [150,150,150,150,100,150,150,150,150,150,
                  150,150,150,100,150,150,150,150,150,100]


def load_tram_weights(lasso, clust_n, t_lag):
    lasso_dir = os.path.join(pwd, lasso)
    tram_msm  = pickle.load(open(
        os.path.join(lasso_dir, 'tram_files',
                     f'{lasso}_ther_obj_cluster_{clust_n}_lag_{t_lag}_msm_obj.pkl'), 'rb'))
    dtrajs = pickle.load(open(
        os.path.join(lasso_dir, 'tram_files',
                     f'{lasso}_dtrajs_cluster_{clust_n}_lag_{t_lag}.pkl'), 'rb'))
    stat_dis = tram_msm.stationary_distribution_full_state
    txx      = np.concatenate(dtrajs)
    cnt      = np.array([np.sum(txx == k) for k in np.unique(txx)])
    return np.array([stat_dis[txx[j]] / cnt[txx[j]] for j in range(len(txx))])


def parse_latched(lasso):
    txt = os.path.join(LOG_DIR, f'{lasso}_latched.txt')
    is_pl = []
    with open(txt) as f:
        for line in f:
            if 'Pre-Lasso' in line:      is_pl.append(True)
            elif 'Pre-Tadpole' in line:  is_pl.append(False)
    return np.array(is_pl, dtype=bool)


rows_extra = []

for i, lasso in enumerate(lasso_name):
    print(f'Processing {lasso} ...')
    feat_dir = os.path.join(pwd, lasso, 'features')

    with open(os.path.join(feat_dir, f'{lasso}_biased_q_list.pickle'), 'rb') as f:
        bq = pickle.load(f)
    with open(os.path.join(feat_dir, f'{lasso}_unbiased_q_list.pickle'), 'rb') as f:
        uq = pickle.load(f)
    q_flat = np.concatenate(bq + uq)

    with open(os.path.join(feat_dir, f'{lasso}_biased_ring_close_dist_ca_n.pickle'), 'rb') as f:
        brc = pickle.load(f)
    with open(os.path.join(feat_dir, f'{lasso}_unbiased_ring_close_dist_ca_n.pickle'), 'rb') as f:
        urc = pickle.load(f)
    rc_flat = np.concatenate(brc + urc)

    tram_w = load_tram_weights(lasso, cluster_number[i], tram_lag[i])

    mask_rc   = rc_flat <= RC_CUTOFF
    mask_q_rc = (q_flat >= Q_CUTOFF) & (rc_flat <= RC_CUTOFF)

    is_prelasso = parse_latched(lasso)
    assert is_prelasso.shape[0] == mask_rc.sum(), \
        f'{lasso}: LATCHED lines={len(is_prelasso)} != n_rc={mask_rc.sum()}'

    # Map the Q+rc frames (a subset of the rc-filtered frames) onto the LATCHED array.
    rc_indices   = np.where(mask_rc)[0]
    q_rc_indices = np.where(mask_q_rc)[0]
    is_pl_q_rc   = is_prelasso[np.searchsorted(rc_indices, q_rc_indices)]

    n_q_rc     = len(q_rc_indices)
    n_q_rc_pl  = int(is_pl_q_rc.sum())
    pct_pl_raw = 100.0 * n_q_rc_pl / n_q_rc if n_q_rc > 0 else float('nan')

    w_q_rc        = tram_w[q_rc_indices]
    pct_pl_tram   = 100.0 * w_q_rc[is_pl_q_rc].sum() / w_q_rc.sum() if w_q_rc.sum() > 0 else float('nan')
    q_rc_pop_tram = 100.0 * w_q_rc.sum() / tram_w.sum()

    print(f'  Q>=0.8 & rc<=6A: {n_q_rc} frames  '
          f'prelasso={n_q_rc_pl} ({pct_pl_raw:.1f}%)  '
          f'TRAM-prelasso={pct_pl_tram:.1f}%  pop={q_rc_pop_tram:.2f}%')

    rows_extra.append({
        'lasso':                      lasso,
        'n_q08_rc6':                  n_q_rc,
        'n_q08_rc6_prelasso':         n_q_rc_pl,
        'n_q08_rc6_pretadpole':       n_q_rc - n_q_rc_pl,
        'pct_q08_rc6_prelasso_raw':   round(pct_pl_raw, 2),
        'pct_q08_rc6_pretadpole_raw': round(100 - pct_pl_raw, 2),
        'pct_q08_rc6_prelasso_tram':  round(pct_pl_tram, 2),
        'q08_rc6_pop_tram':           round(q_rc_pop_tram, 2),
    })

csv_out  = os.path.join(OUT_DIR, 'latched_split_summary.csv')
df_new   = pd.read_csv(csv_out).merge(pd.DataFrame(rows_extra), on='lasso', how='left')
df_new.to_csv(csv_out, index=False)
print(df_new[['lasso','n_q08_rc6','n_q08_rc6_prelasso','n_q08_rc6_pretadpole',
              'pct_q08_rc6_prelasso_raw','pct_q08_rc6_prelasso_tram']].to_string(index=False))
