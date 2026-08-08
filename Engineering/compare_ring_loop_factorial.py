"""
2x2 ring x loop factorial: folded-state stability of four systems.

  klebsidin (WT)  = klebsidin ring    + klebsidin loop      (negative control)
  chimera1        = klebsidin ring    + microcin J25 loop   (sufficiency / rescue)
  chimera2        = microcin J25 ring + klebsidin loop      (necessity / reciprocal)
  microcin J25    = microcin J25 ring + microcin J25 loop   (positive control)

Metrics per trajectory frame:
  * loop beta-hairpin content = fraction of loop-window residues with strict DSSP 'E'
    (the E+B variant, which also counts isolated beta-bridges, is saved for a
    sensitivity check but should not be mixed with the primary figure)
  * Q = Best-Hummer fraction of native contacts against each system's own equilibrated
    reference structure

Controls are the ten unbiased folded trajectories with the highest first-frame Q.
"""

import json
import os
import pickle
from itertools import combinations
from multiprocessing import Pool

import mdtraj as md
import numpy as np

# User parameters
pwd = ''            # unbiased/biased production data root (per-peptide subdirectories)
chimera_dir = ''    # chimera topologies, reference restarts and stripped trajectories

COMMON_FR = 5000    # frames used per trajectory (500 ns at 0.1 ns/frame)
DT_NS = 0.1

ORDER  = ["klebsidin", "chimera2", "chimera1", "mcj25"]
COLORS = {"klebsidin": "#1f77b4", "chimera2": "#ff7f0e", "chimera1": "#d62728", "mcj25": "#2ca02c"}
LABELS = {
    "klebsidin": "klebsidin",
    "chimera2":  "microcin J25 ring + klebsidin loop",
    "chimera1":  "klebsidin ring + microcin J25 loop",
    "mcj25":     "microcin J25",
}

AXIS_LABEL_FONTSIZE = 19
TICK_LABEL_FONTSIZE = 15
VIOLIN_XTICK_FONTSIZE = 11
LEGEND_FONTSIZE = 12

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
    "GLY": "G", "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L",
    "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def select_top_folded(lasso, n=10):
    """Return the n unbiased folded trajectories with the highest first-frame Q."""
    with open(os.path.join(pwd, lasso, 'features', f'{lasso}_unbiased_q_list.pickle'), 'rb') as fh:
        q_list = pickle.load(fh)

    unbiased = os.path.join(pwd, lasso, 'unbiased')
    candidates = []
    for traj in sorted(os.listdir(unbiased)):
        if not traj.startswith(f'{lasso}_unbiased_folded_traj_') or not traj.endswith('.dcd'):
            continue
        idx = int(traj[:-4].split('_')[-1])
        q = q_list[idx]
        if len(q) != COMMON_FR:
            continue
        candidates.append({'idx': idx, 'q0': float(q[0]), 'file': os.path.join(unbiased, traj)})
    return sorted(candidates, key=lambda item: item['q0'], reverse=True)[:n]


def make_systems():
    kleb = select_top_folded('klebsidin')
    mcc  = select_top_folded('microcinJ25')
    return {
        "klebsidin": {
            "top":     os.path.join(pwd, 'klebsidin', 'unbiased', 'klebsidin_nowat.prmtop'),
            "ref_rst": os.path.join(chimera_dir, 'klebsidin-nobond-HMR_EQ_center.rst7'),
            "ref_top": os.path.join(chimera_dir, 'klebsidin-nobond-HMR.prmtop'),
            "trajs":   [item['file'] for item in kleb],
            "loop":    (8, 16),
        },
        "chimera2": {
            "top":     os.path.join(chimera_dir, 'klebsidinloop_mccj25_nowat.prmtop'),
            "ref_rst": os.path.join(chimera_dir, 'klebsidinloop_mccj25-nobond-HMR_EQ_center.rst7'),
            "ref_top": os.path.join(chimera_dir, 'klebsidinloop_mccj25-nobond-HMR.prmtop'),
            "trajs":   sorted(_glob_trajs('chimera2_folded_traj_r')),
            "loop":    (8, 16),
        },
        "chimera1": {
            "top":     os.path.join(chimera_dir, 'MccJ25loop_klebsidin_nowat.prmtop'),
            "ref_rst": os.path.join(chimera_dir, 'MccJ25loop_klebsidin-nobond-HMR_EQ_center.rst7'),
            "ref_top": os.path.join(chimera_dir, 'MccJ25loop_klebsidin-nobond-HMR.prmtop'),
            "trajs":   sorted(_glob_trajs('chimera1_folded_traj_r')),
            "loop":    (8, 18),
        },
        "mcj25": {
            "top":     os.path.join(pwd, 'microcinJ25', 'unbiased', 'microcinJ25_nowat.prmtop'),
            "ref_rst": os.path.join(chimera_dir, 'microcinJ25-nobond-HMR_EQ_center.rst7'),
            "ref_top": os.path.join(chimera_dir, 'microcinJ25-nobond-HMR.prmtop'),
            "trajs":   [item['file'] for item in mcc],
            "loop":    (8, 18),
        },
    }


def _glob_trajs(prefix):
    traj_dir = os.path.join(chimera_dir, 'traj')
    files = [f for f in os.listdir(traj_dir) if f.startswith(prefix) and f.endswith('.dcd')]
    files.sort(key=lambda f: int(f[:-4].split('_r')[-1]))
    return [os.path.join(traj_dir, f) for f in files]


def native_contacts(ref):
    heavy = ref.topology.select_atom_indices("heavy")
    pairs = np.array(
        [(i, j) for i, j in combinations(heavy, 2)
         if abs(ref.topology.atom(i).residue.index - ref.topology.atom(j).residue.index) > 3]
    )
    d0 = md.compute_distances(ref[0], pairs)[0]
    keep = d0 < 0.45
    return pairs[keep], d0[keep]


def worker(args):
    order_index, filename, topology, native_pairs, native_distances, loop = args
    traj = md.load(filename, top=topology)
    distances = md.compute_distances(traj, native_pairs)
    q = np.mean(1.0 / (1 + np.exp(np.clip(50 * (distances - 1.8 * native_distances), -30, 30))), axis=1)
    ss = md.compute_dssp(traj, simplified=False)[:, loop[0]: loop[1] + 1]
    sheet = (ss == "E").mean(1)
    sheet_eb = np.isin(ss, ["E", "B"]).mean(1)
    return order_index, filename, q.astype(np.float32), sheet.astype(np.float32), sheet_eb.astype(np.float32)


def run_system(name, systems):
    system = systems[name]
    ref = md.load(system["ref_rst"], top=system["ref_top"])
    ref = ref.atom_slice(ref.topology.select("protein"))
    traj_top = md.load_prmtop(system["top"])
    assert ref.n_atoms == traj_top.n_atoms, f"{name}: ref {ref.n_atoms} != traj-top {traj_top.n_atoms} atoms"

    loop_res = "".join(
        THREE_TO_ONE.get(res.name, "?")
        for res in list(traj_top.residues)[system["loop"][0]: system["loop"][1] + 1]
    )
    native_pairs, native_distances = native_contacts(ref)
    print(f"[{name}] {len(system['trajs'])} trajs | {ref.n_residues} res | "
          f"{len(native_pairs)} native contacts | loop{system['loop']} = '{loop_res}'")

    tasks = [(idx, traj, system["top"], native_pairs, native_distances, system["loop"])
             for idx, traj in enumerate(system["trajs"])]
    with Pool(min(10, len(tasks))) as pool:
        results = pool.map(worker, tasks)

    results.sort(key=lambda item: item[0])
    out = {"q": [], "sheet": [], "sheet_eb": [], "files": [item[1] for item in results]}
    for _, _, q, sheet, sheet_eb in results:
        out["q"].append(q); out["sheet"].append(sheet); out["sheet_eb"].append(sheet_eb)

    np.savez(f"metrics_{name}.npz",
             q=np.array(out["q"], dtype=object), sheet=np.array(out["sheet"], dtype=object),
             sheet_eb=np.array(out["sheet_eb"], dtype=object), files=np.array(out["files"]))
    return out


def band(arrs):
    length = min(len(arr) for arr in arrs)
    matrix = np.array([arr[:length] for arr in arrs])
    return np.arange(length) * DT_NS, matrix.mean(0), matrix.std(0)


def win_mean(arrs, window=COMMON_FR):
    return np.array([arr[:window].mean() for arr in arrs])


def plot_outputs(data, summary):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Arial"
    import matplotlib.pyplot as plt

    def time_series(key, ylabel, fname):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ORDER:
            x, mean, sd = band(data[name][key])
            ax.plot(x, mean, color=COLORS[name], lw=2, label=LABELS[name])
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[name], alpha=0.15)
        ax.set_xlabel("Time (ns)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

    time_series("q", "Fraction of Native Contacts (Q)", "wholeQ_vs_time.png")
    time_series("sheet", "Loop beta-hairpin content", "loop_beta_sheet_vs_time.png")

    def draw_violin(ax, key, ylabel):
        points = [np.array(summary[name][key]) for name in ORDER]
        parts = ax.violinplot(points, showmeans=False, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLORS[ORDER[i]]); body.set_alpha(0.3)
        for i, name in enumerate(ORDER):
            ax.scatter([i + 1] * len(summary[name][key]), summary[name][key],
                       color=COLORS[name], s=42, zorder=3, edgecolor="k", lw=0.5)
        ax.set_xticks(list(range(1, len(ORDER) + 1)))
        ax.set_xticklabels([LABELS[name] for name in ORDER], rotation=18, ha="right",
                           fontsize=VIOLIN_XTICK_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_violin(axes[0], "sheet", "Loop beta-hairpin content (DSSP E)")
    draw_violin(axes[1], "q", "Fraction of Native Contacts (Q)")
    plt.tight_layout(); plt.savefig("comparison_violin.png", dpi=200); plt.close()


if __name__ == "__main__":
    systems = make_systems()
    data = {name: run_system(name, systems) for name in ORDER}

    print(f"\nSummary (per-traj mean +/- sd over first {COMMON_FR} frames = {COMMON_FR*DT_NS:.0f} ns)")
    print(f"{'system':40s} {'loop beta-E':>14s} {'loop E+B':>12s} {'whole Q':>12s}")
    summary = {}
    for name in ORDER:
        sheet = win_mean(data[name]["sheet"])
        sheet_eb = win_mean(data[name]["sheet_eb"])
        q = win_mean(data[name]["q"])
        summary[name] = {"sheet": sheet.tolist(), "sheet_eb": sheet_eb.tolist(), "q": q.tolist()}
        print(f"{LABELS[name]:40s} {sheet.mean():>6.3f}+/-{sheet.std():<5.3f} "
              f"{sheet_eb.mean():>5.3f}+/-{sheet_eb.std():<4.3f} {q.mean():>5.3f}+/-{q.std():<5.3f}")

    with open("summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_outputs(data, summary)
