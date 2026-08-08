"""
Per-frame inputs for the capistruin confinement FEP.

For every frame of the converged TRAM dataset, collect:
    w    - unbiased Boltzmann weight (TRAM stationary distribution)
    Q    - fraction of native contacts (saved feature)
    rc   - ring-closure distance, nm (saved feature)
    U(R) - spherical flat-bottom wall energy the frame would feel at radius R

The wall has the same form and force constant as the confinement MD restraint, but is
measured about the peptide centre of mass so it depends only on the shape of the
conformation, not on where the peptide sat in the box:

    U_i(R) = k * sum_atoms max(0, |r_atom - COM|_i - R)^2 ,   k = 5 kcal/mol/A^2

Frame order is biased trajectories first, then unbiased, matching the feature lists
and the TRAM dtrajs. Output -> perframe.npz.
"""
import glob
import pickle
import numpy as np
import mdtraj as md
from natsort import natsorted

# User parameters
pwd = ''
LASSO   = 'capistruin'
CLUSTER = 100
LAG     = 100
OUT     = 'perframe.npz'

K_WALL = 5.0     # kcal/mol/A^2, same as the confinement simulations
RADII  = [3.0, 2.5, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]   # nm

DATA = f'{pwd}{LASSO}'
TOP  = f'{DATA}/unbiased/{LASSO}_nowat.prmtop'


def read(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# Unbiased Boltzmann weight of every frame, from the TRAM stationary distribution:
# each microstate's probability is spread evenly over the frames assigned to it.
msm    = read(f'{DATA}/tram_files/{LASSO}_ther_obj_cluster_{CLUSTER}_lag_{LAG}_msm_obj.pkl')
dtrajs = read(f'{DATA}/tram_files/{LASSO}_dtrajs_cluster_{CLUSTER}_lag_{LAG}.pkl')
pi     = np.asarray(msm.stationary_distribution_full_state)
state  = np.concatenate(dtrajs)
w = pi[state] / np.bincount(state, minlength=len(pi)).astype(float)[state]
w /= w.sum()

q_list  = read(f'{DATA}/features/{LASSO}_biased_q_list.pickle') \
        + read(f'{DATA}/features/{LASSO}_unbiased_q_list.pickle')
rc_list = read(f'{DATA}/features/{LASSO}_biased_ring_close_dist_ca_n.pickle') \
        + read(f'{DATA}/features/{LASSO}_unbiased_ring_close_dist_ca_n.pickle')
Q  = np.concatenate(q_list)
rc = np.concatenate(rc_list)

# The weights only line up if all three sources describe the same frames in the same order.
assert len(Q) == len(rc) == len(w), (len(Q), len(rc), len(w))
assert [len(a) for a in q_list] == [len(s) for s in dtrajs], 'Q vs dtrajs trajectory lengths differ'
print(f'{len(w)} frames, {len(pi)} microstates, weights sum to {w.sum():.6f}')
print(f'bulk %(Q>0.8) = {100 * w[Q > 0.8].sum():.4f}%')

# Wall energy per frame, streamed one trajectory at a time.
trajs  = natsorted(glob.glob(f'{DATA}/biased/{LASSO}_biased_traj_win_*_rep_*.dcd'))
trajs += natsorted(glob.glob(f'{DATA}/unbiased/{LASSO}_unbiased_*_traj_*.dcd'))
trajs  = [t for t in trajs if not t.endswith('_wat.dcd')]
assert len(trajs) == len(dtrajs), (len(trajs), len(dtrajs))

U = {R: np.empty(len(w)) for R in RADII}
n0 = 0
for k, dcd in enumerate(trajs):
    t = md.load(dcd, top=TOP)
    assert t.n_frames == len(dtrajs[k]), f'frame-count mismatch: {dcd}'
    xyz = t.xyz * 10.0                                                    # nm -> A
    d   = np.linalg.norm(xyz - xyz.mean(axis=1, keepdims=True), axis=2)   # |r - COM| per atom
    for R in RADII:
        over = np.maximum(0.0, d - 10.0 * R)
        U[R][n0:n0 + t.n_frames] = K_WALL * np.sum(over * over, axis=1)
    n0 += t.n_frames
    if (k + 1) % 100 == 0:
        print(f'  {k + 1}/{len(trajs)} trajectories')
assert n0 == len(w)

np.savez_compressed(OUT, w=w, Q=Q, rc=rc, radii=np.array(RADII),
                    **{f'U_{R}': U[R] for R in RADII})
print(f'wrote {OUT}')
