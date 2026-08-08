"""
Classify ring-closed conformations as pre-lasso (threaded) or pre-tadpole (unthreaded)
with the LATCHED geometric algorithm (da Hora et al., J. Phys. Chem. B 2024).

For each lasso peptide:
  1. Select frames with ring-closure distance <= 6 A and stream them into one DCD.
  2. Generate a per-peptide LATCHED VMD script and run it.
  3. Parse the output, split the trajectory into pre-lasso / pre-tadpole DCDs, and
     report raw and TRAM-weighted pre-lasso fractions.

Requires VMD on PATH.
"""

import os, glob, pickle, subprocess, shutil
import numpy as np
import mdtraj as md
from mdtraj.formats import DCDTrajectoryFile
import pandas as pd
from natsort import natsorted

# User parameters
pwd = ''
OUT_DIR = 'latched_split'
VMD_BIN = 'vmd'

TRAJ_DIR    = os.path.join(OUT_DIR, 'trajectories')
PL_DIR      = os.path.join(OUT_DIR, 'prelasso')
PT_DIR      = os.path.join(OUT_DIR, 'pretadpole')
TCL_DIR     = os.path.join(OUT_DIR, 'tcl')
OUT_LOG_DIR = os.path.join(OUT_DIR, 'latched_output')

RC_CUTOFF = 0.6    # nm (6 A)

lasso_name = [
    'acinetodin','astexin-1','benenodin-1','brevunsin','capistruin',
    'caulonodin-V','caulosegnin-I','caulosegnin-II','chaxapeptin','citrocin',
    'klebsidin','microcinJ25','rubrivinodin','sphaericin','sphingopyxin-I',
    'streptomonomicin','subterisin','ubonodin','xanthomonin-I','xanthomonin-II',
]
ring_size = [
    8, 9, 8, 9, 9, 9, 8, 9, 8, 8,
    8, 8, 9, 9, 9, 9, 8, 8, 7, 7,
]
# LATCHED tail start residue (VMD 1-indexed), taken as the first local maximum of the
# ring-plane CA projection in the native structure.
latched_turn = [
    11, 15, 15, 14, 12, 14, 12, 14, 12, 12,
    11, 14, 14, 13, 14, 13, 11, 12, 10, 10,
]
lasso_length = [
    18, 23, 24, 21, 19, 18, 19, 19, 15, 19,
    19, 21, 18, 18, 21, 21, 21, 28, 20, 20,
]
cluster_number = [
    400, 100, 100, 100, 100, 200, 500, 100, 300, 200,
    400, 400, 300, 200, 500, 200, 100, 200, 300, 400,
]
tram_lag = [
    150, 150, 150, 150, 100, 150, 150, 150, 150, 150,
    150, 150, 150, 100, 150, 150, 150, 150, 150, 100,
]


def _stream_write_dcd(dcd_out, prmtop, all_trajs, rc_list, mask):
    """Write only masked frames to dcd_out, one source trajectory at a time."""
    frame_offset = 0
    n_written = 0
    with DCDTrajectoryFile(dcd_out, 'w') as dcd_f:
        for traj_file, rc_arr in zip(all_trajs, rc_list):
            n_frames  = len(rc_arr)
            traj_mask = mask[frame_offset: frame_offset + n_frames]
            if traj_mask.any():
                t    = md.load(traj_file, top=prmtop)
                idxs = np.where(traj_mask)[0]
                xyz_A = t.xyz[idxs] * 10.0   # nm -> A (DCD format)
                if t.unitcell_lengths is not None:
                    cell_A = t.unitcell_lengths[idxs] * 10.0
                    angles = t.unitcell_angles[idxs]
                else:
                    cell_A = None
                    angles = None
                dcd_f.write(xyz_A, cell_A, angles)
                n_written += len(idxs)
            frame_offset += n_frames
    return n_written


def extract_rc_frames(lasso, clust_n, t_lag):
    """Select frames with ring-closure distance <= RC_CUTOFF and compute TRAM weights."""
    lasso_dir = os.path.join(pwd, lasso)
    feat_dir  = os.path.join(lasso_dir, 'features')
    prmtop    = os.path.join(lasso_dir, 'unbiased', f'{lasso}_nowat.prmtop')

    with open(os.path.join(feat_dir, f'{lasso}_biased_ring_close_dist_ca_n.pickle'), 'rb') as f:
        brc = pickle.load(f)
    with open(os.path.join(feat_dir, f'{lasso}_unbiased_ring_close_dist_ca_n.pickle'), 'rb') as f:
        urc = pickle.load(f)
    rc_list = brc + urc
    rc_flat = np.concatenate(rc_list)
    mask    = rc_flat <= RC_CUTOFF

    n_filtered = int(mask.sum())
    print(f'  rc filter: {n_filtered} / {len(mask)} frames ({100*n_filtered/len(mask):.2f}%)')
    if n_filtered == 0:
        print(f'  WARNING: no frames pass rc filter for {lasso}.')
        return None, prmtop, None, mask

    tram_msm = pickle.load(open(
        os.path.join(lasso_dir, 'tram_files',
                     f'{lasso}_ther_obj_cluster_{clust_n}_lag_{t_lag}_msm_obj.pkl'), 'rb'))
    dtrajs = pickle.load(open(
        os.path.join(lasso_dir, 'tram_files',
                     f'{lasso}_dtrajs_cluster_{clust_n}_lag_{t_lag}.pkl'), 'rb'))
    stat_dis = tram_msm.stationary_distribution_full_state
    txx      = np.concatenate(dtrajs)
    cnt      = np.array([np.sum(txx == k) for k in np.unique(txx)])
    tram_w   = np.array([stat_dis[txx[j]] / cnt[txx[j]] for j in range(len(txx))])

    biased_trajs   = natsorted(glob.glob(
        os.path.join(lasso_dir, 'biased', f'{lasso}_biased_traj_win_*_rep_*.dcd')))
    unbiased_trajs = natsorted(glob.glob(
        os.path.join(lasso_dir, 'unbiased', f'{lasso}_unbiased_*_traj_*.dcd')))
    all_trajs = biased_trajs + unbiased_trajs
    assert len(all_trajs) == len(rc_list)

    dcd_out = os.path.join(TRAJ_DIR, f'{lasso}_rc6.dcd')
    n_written = _stream_write_dcd(dcd_out, prmtop, all_trajs, rc_list, mask)
    shutil.copy(prmtop, os.path.join(TRAJ_DIR, f'{lasso}_nowat.prmtop'))
    print(f'  Saved {n_written} frames -> {dcd_out}')

    return dcd_out, prmtop, tram_w, mask


TCL_TEMPLATE = r"""
mol addfile {TOPOLOGY} type parm7
mol addfile {TRAJECTORY} type dcd waitfor all

set outfile2 [open "{OUTPUT}.txt" w]

proc triNorm {{comLoop com1 com2}} {{
    return [veccross [vecsub $com1 $comLoop] [vecsub $com2 $comLoop]]
}}
proc findIntersection {{pointOnLine pointOnPlane lineDirection planeNorm}} {{
    set denom [vecdot $lineDirection $planeNorm]
    if {{$denom == 0}} {{ return -1 }}
    set d [expr {{double([vecdot [vecsub $pointOnPlane $pointOnLine] $planeNorm]) / $denom}}]
    if {{$d > 1 || $d < 0}} {{ return -2 }}
    return [vecadd $pointOnLine [vecscale $d $lineDirection]]
}}
proc barycentricTransform {{comLoop comRes1 comRes2 intersectionPoint}} {{
    set w0 [vecsub $comRes2 $comLoop]
    set w1 [vecsub $comRes1 $comLoop]
    set w2 [vecsub $intersectionPoint $comLoop]
    set d00 [vecdot $w0 $w0]; set d01 [vecdot $w0 $w1]; set d02 [vecdot $w0 $w2]
    set d11 [vecdot $w1 $w1]; set d12 [vecdot $w1 $w2]
    set denom [expr {{($d00*$d11) - ($d01*$d01)}}]
    set u [expr {{(($d11*$d02) - ($d01*$d12)) / $denom}}]
    set v [expr {{(($d00*$d12) - ($d01*$d02)) / $denom}}]
    return [expr {{($u >= 0) && ($v >= 0) && (($u+$v) <= 1)}}]
}}

set loop_size  {RING_SIZE}
set tail_start {TAIL_START}
set tail_end   {TAIL_END}
set nf [molinfo top get numframes]

for {{set i 0}} {{$i < $nf}} {{incr i}} {{
    set resLoop [atomselect top "name CA and (resid 1 to $loop_size)"]
    $resLoop frame $i; $resLoop update
    set comLoop [measure center $resLoop weight mass]; $resLoop delete
    for {{set p 1}} {{$p <= $loop_size}} {{incr p}} {{
        set res($p) [atomselect top "resid $p and name CA"]
        $res($p) frame $i; $res($p) update
        set com($p) [measure center $res($p) weight mass]; $res($p) delete
    }}
    for {{set q $tail_start}} {{$q <= $tail_end}} {{incr q}} {{
        set res($q) [atomselect top "resid $q and name CA"]
        $res($q) frame $i; $res($q) update
        set com($q) [measure center $res($q) weight mass]; $res($q) delete
    }}
    for {{set r 1}} {{$r <= $loop_size}} {{incr r}} {{
        set rNext [expr {{$r < $loop_size ? $r+1 : 1}}]
        set norm($r) [triNorm $comLoop $com($r) $com($rNext)]
    }}
    for {{set s $tail_start}} {{$s < $tail_end}} {{incr s}} {{
        set tail($s) [vecsub $com([expr {{$s+1}}]) $com($s)]
    }}
    set counter 0
    for {{set u 1}} {{$u <= $loop_size}} {{incr u}} {{
        for {{set v $tail_start}} {{$v < $tail_end}} {{incr v}} {{
            set ip [findIntersection $com($v) $comLoop $tail($v) $norm($u)]
            if {{$ip != -1 && $ip != -2}} {{
                set uNext [expr {{$u < $loop_size ? $u+1 : 1}}]
                if {{[barycentricTransform $comLoop $com($u) $com($uNext) $ip] > 0}} {{
                    incr counter
                }}
                unset uNext
            }}
            unset ip
        }}
    }}
    if {{$counter > 0}} {{
        puts $outfile2 "Time $i ... Penetrating (Pre-Lasso)"
    }} else {{
        puts $outfile2 "Time $i ... Non-penetrating (Pre-Tadpole)"
    }}
    unset comLoop; unset com; unset norm; unset tail; unset counter
}}
close $outfile2
quit
"""


def generate_tcl(lasso, ring_sz, turn_start, tail_end_res, dcd_path, prmtop_path):
    tcl_text = TCL_TEMPLATE.format(
        TOPOLOGY   = prmtop_path,
        TRAJECTORY = dcd_path,
        OUTPUT     = os.path.join(OUT_LOG_DIR, f'{lasso}_latched'),
        RING_SIZE  = ring_sz,
        TAIL_START = turn_start,
        TAIL_END   = tail_end_res,
    )
    tcl_path = os.path.join(TCL_DIR, f'{lasso}_latched.tcl')
    with open(tcl_path, 'w') as f:
        f.write(tcl_text)
    return tcl_path


def run_vmd(tcl_path, lasso):
    log = os.path.join(OUT_LOG_DIR, f'{lasso}_vmd.log')
    result = subprocess.run([VMD_BIN, '-dispdev', 'text', '-e', tcl_path],
                            stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f'  WARNING: VMD non-zero exit for {lasso}, check {log}')


def parse_latched(lasso):
    txt_path = os.path.join(OUT_LOG_DIR, f'{lasso}_latched.txt')
    is_prelasso = []
    with open(txt_path) as f:
        for line in f:
            if 'Pre-Lasso' in line:      is_prelasso.append(True)
            elif 'Pre-Tadpole' in line:  is_prelasso.append(False)
    return np.array(is_prelasso, dtype=bool)


def split_trajectory(lasso, prmtop, all_trajs, rc_list, rc_mask, is_prelasso):
    """Second streaming pass: write pre-lasso and pre-tadpole frames to separate DCDs."""
    pl_out = os.path.join(PL_DIR, f'{lasso}_prelasso.dcd')
    pt_out = os.path.join(PT_DIR, f'{lasso}_pretadpole.dcd')

    n_pl = n_pt = 0
    latched_cursor = 0
    with DCDTrajectoryFile(pl_out, 'w') as pl_f, \
         DCDTrajectoryFile(pt_out, 'w') as pt_f:
        frame_offset = 0
        for traj_file, rc_arr in zip(all_trajs, rc_list):
            n_frames  = len(rc_arr)
            traj_mask = rc_mask[frame_offset: frame_offset + n_frames]
            if traj_mask.any():
                t    = md.load(traj_file, top=prmtop)
                idxs = np.where(traj_mask)[0]
                n_here = len(idxs)
                frame_labels = is_prelasso[latched_cursor: latched_cursor + n_here]
                xyz_A = t.xyz[idxs] * 10.0
                cl = t.unitcell_lengths[idxs] * 10.0 if t.unitcell_lengths is not None else None
                ang = t.unitcell_angles[idxs] if t.unitcell_angles is not None else None
                pl_sel = np.where(frame_labels)[0]
                if len(pl_sel):
                    pl_f.write(xyz_A[pl_sel],
                               cl[pl_sel] if cl is not None else None,
                               ang[pl_sel] if ang is not None else None)
                    n_pl += len(pl_sel)
                pt_sel = np.where(~frame_labels)[0]
                if len(pt_sel):
                    pt_f.write(xyz_A[pt_sel],
                               cl[pt_sel] if cl is not None else None,
                               ang[pt_sel] if ang is not None else None)
                    n_pt += len(pt_sel)
                latched_cursor += n_here
            frame_offset += n_frames

    for d in [PL_DIR, PT_DIR]:
        shutil.copy(prmtop, os.path.join(d, f'{lasso}_nowat.prmtop'))
    print(f'  Pre-lasso {n_pl} frames, pre-tadpole {n_pt} frames')
    return n_pl, n_pt


if __name__ == '__main__':
    for d in [TRAJ_DIR, PL_DIR, PT_DIR, TCL_DIR, OUT_LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    rows = []
    for i, lasso in enumerate(lasso_name):
        print(f'\n{lasso}  (ring=1-{ring_size[i]}, tail={latched_turn[i]}-{lasso_length[i]})')

        dcd_path, prmtop, tram_w, mask = extract_rc_frames(
            lasso, cluster_number[i], tram_lag[i])
        if dcd_path is None:
            continue
        n_rc = int(mask.sum())

        tcl = generate_tcl(lasso, ring_size[i], latched_turn[i], lasso_length[i],
                           dcd_path, prmtop)
        run_vmd(tcl, lasso)
        is_prelasso = parse_latched(lasso)

        n_pl = int(is_prelasso.sum())
        pct_raw = 100.0 * n_pl / n_rc
        print(f'  Raw:  Pre-Lasso={n_pl} ({pct_raw:.2f}%)  Pre-Tadpole={n_rc - n_pl} ({100-pct_raw:.2f}%)')

        w_rc = tram_w[np.where(mask)[0]]
        pct_tram = 100.0 * w_rc[is_prelasso].sum() / w_rc.sum()
        pct_rc_pop = 100.0 * w_rc.sum() / tram_w.sum()
        print(f'  TRAM: Pre-Lasso={pct_tram:.2f}%  rc-pop={pct_rc_pop:.2f}% of ensemble')

        lasso_dir      = os.path.join(pwd, lasso)
        biased_trajs   = natsorted(glob.glob(
            os.path.join(lasso_dir, 'biased', f'{lasso}_biased_traj_win_*_rep_*.dcd')))
        unbiased_trajs = natsorted(glob.glob(
            os.path.join(lasso_dir, 'unbiased', f'{lasso}_unbiased_*_traj_*.dcd')))
        with open(os.path.join(lasso_dir, 'features',
                               f'{lasso}_biased_ring_close_dist_ca_n.pickle'), 'rb') as f:
            brc2 = pickle.load(f)
        with open(os.path.join(lasso_dir, 'features',
                               f'{lasso}_unbiased_ring_close_dist_ca_n.pickle'), 'rb') as f:
            urc2 = pickle.load(f)

        n_pl, n_pt = split_trajectory(lasso, prmtop, biased_trajs + unbiased_trajs,
                                      brc2 + urc2, mask, is_prelasso)

        rows.append({
            'lasso':              lasso,
            'n_rc':               n_rc,
            'n_prelasso':         n_pl,
            'n_pretadpole':       n_pt,
            'pct_prelasso_raw':   round(pct_raw, 2),
            'pct_pretadpole_raw': round(100 - pct_raw, 2),
            'pct_prelasso_tram':  round(pct_tram, 2),
            'rc_pop_tram':        round(pct_rc_pop, 2),
        })

    df = pd.DataFrame(rows)
    csv_out = os.path.join(OUT_DIR, 'latched_split_summary.csv')
    df.to_csv(csv_out, index=False)
    print(df.to_string(index=False))
