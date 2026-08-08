"""
Compute the harmonic linear discriminant analysis (HLDA) collective variable of
da Hora et al. (J. Phys. Chem. B 2024, 128, 4063-4075, Figure 9) for microcin J25.

CV = weighted sum of the eight CA-CA distances from the ring residues (1-8) to Phe19,
with the distances in Angstroms.
"""

import pickle
import mdtraj as md
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import glob

HLDA_WEIGHTS = np.array([-0.3221, -0.0361, -0.1063, -0.1538,
                         -0.0913, -0.5176, -0.6589, -0.3867])

RING_RESIDUES = list(range(8))   # 0-indexed: Gly1 to Glu8
PHE19_RESIDUE = 18               # 0-indexed


def cal_hlda_cv(traj_files, prmtop_file):
    """Compute the HLDA CV for each trajectory. Returns a list of 1D arrays."""
    top = md.load_topology(prmtop_file)
    phe19_ca = top.select(f"resid {PHE19_RESIDUE} and name CA")
    if len(phe19_ca) == 0:
        raise ValueError(f"Could not find CA atom for resid {PHE19_RESIDUE} (Phe19)")

    atom_pairs = []
    for res_idx in RING_RESIDUES:
        ca = top.select(f"resid {res_idx} and name CA")
        if len(ca) == 0:
            raise ValueError(f"Could not find CA atom for resid {res_idx}")
        atom_pairs.append([ca[0], phe19_ca[0]])
    atom_pairs = np.array(atom_pairs)

    hlda_cv_list = []
    for traj_file in tqdm(traj_files):
        t = md.load(traj_file, top=prmtop_file)
        dists_ang = md.compute_distances(t, atom_pairs) * 10.0
        hlda_cv_list.append(dists_ang @ HLDA_WEIGHTS)
    return hlda_cv_list


if __name__ == '__main__':
    pwd = ''
    lasso = 'microcinJ25'

    biased_trajs = natsorted(glob.glob(f'{pwd}{lasso}/biased/{lasso}_biased_traj_win_*_rep_*.dcd'))
    biased_trajs = [t for t in biased_trajs if not t.endswith('_wat.dcd')]
    biased_hlda = cal_hlda_cv(biased_trajs, f'{pwd}{lasso}/biased/{lasso}_nowat.prmtop')
    with open(f'{lasso}_biased_hlda_cv_dahora2024.pickle', 'wb') as f:
        pickle.dump(biased_hlda, f)

    unbiased_trajs = natsorted(glob.glob(f'{pwd}{lasso}/unbiased/{lasso}_unbiased_*_traj_*.dcd'))
    unbiased_trajs = [t for t in unbiased_trajs if not t.endswith('_wat.dcd')]
    unbiased_hlda = cal_hlda_cv(unbiased_trajs, f'{pwd}{lasso}/unbiased/{lasso}_nowat.prmtop')
    with open(f'{lasso}_unbiased_hlda_cv_dahora2024.pickle', 'wb') as f:
        pickle.dump(unbiased_hlda, f)
