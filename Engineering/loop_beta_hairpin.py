"""
Loop-normalised beta-hairpin content of microcin J25 from full DSSP.

Primary metric: fraction of the 11 loop residues (0-based resid 8-18) assigned strict
DSSP 'E'. Normalising over the loop window rather than the whole 21-residue peptide is
what makes this number directly comparable to the engineered chimeras.
The E+B variant, which also counts isolated beta-bridges, is saved as a sensitivity check.
"""

import glob
import json
from multiprocessing import Pool

import mdtraj as md
import numpy as np
from natsort import natsorted

# User parameters
pwd = ''
LASSO = 'microcinJ25'
N_PROCESSES = 10

LOOP_START = 8     # 0-based, inclusive
LOOP_END = 18      # 0-based, inclusive

OUTPUT_E  = "beta_sheet_loop_E_res.npy"
OUTPUT_EB = "beta_sheet_loop_EB_res.npy"
METADATA  = "beta_sheet_loop_metadata.json"


def calculate_one(args):
    index, traj_file, topology = args
    traj = md.load(traj_file, top=topology)
    ss = md.compute_dssp(traj, simplified=False)[:, LOOP_START: LOOP_END + 1]
    sheet_e  = (ss == "E").mean(axis=1).astype(np.float32)
    sheet_eb = np.isin(ss, ["E", "B"]).mean(axis=1).astype(np.float32)
    return index, sheet_e, sheet_eb, traj.n_frames


if __name__ == "__main__":
    topology = f'{pwd}{LASSO}/unbiased/{LASSO}_nowat.prmtop'
    files  = natsorted(glob.glob(f'{pwd}{LASSO}/biased/{LASSO}_biased_traj_win_*_rep_*.dcd'))
    files += natsorted(glob.glob(f'{pwd}{LASSO}/unbiased/{LASSO}_unbiased_*_traj_*.dcd'))
    files  = [f for f in files if not f.endswith('_wat.dcd')]

    loop_len = LOOP_END - LOOP_START + 1
    print(f'loop: {LOOP_START}-{LOOP_END} inclusive ({loop_len} residues), {len(files)} trajectories')

    tasks = [(i, filename, topology) for i, filename in enumerate(files)]
    results = []
    with Pool(processes=min(N_PROCESSES, len(tasks))) as pool:
        for result in pool.imap_unordered(calculate_one, tasks):
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(tasks):
                print(f'processed {len(results)}/{len(tasks)}')

    results.sort(key=lambda item: item[0])
    e_values     = [item[1] for item in results]
    eb_values    = [item[2] for item in results]
    frame_counts = [item[3] for item in results]

    np.save(OUTPUT_E, np.array(e_values, dtype=object))
    np.save(OUTPUT_EB, np.array(eb_values, dtype=object))

    with open(METADATA, "w") as handle:
        json.dump({
            "metric_primary":             "loop beta-hairpin content, full DSSP strict E",
            "metric_sensitivity":         "loop beta-hairpin content, full DSSP E+B",
            "loop_start_0_based":         LOOP_START,
            "loop_end_0_based_inclusive": LOOP_END,
            "loop_length":                loop_len,
            "n_trajectories":             len(files),
            "n_frames_total":             int(sum(frame_counts)),
            "topology":                   topology,
        }, handle, indent=2)

    e_concat  = np.concatenate(e_values)
    eb_concat = np.concatenate(eb_values)
    print(f'strict E mean: {e_concat.mean():.6f}')
    print(f'E+B mean: {eb_concat.mean():.6f}')
