"""
Soft-wall confinement FEP for capistruin folding.

Reweight the unbiased TRAM ensemble by the spherical wall U(R) to obtain the folding
free energy as a function of confinement radius, with no new sampling. Pre-folded and
unfolded are the two structural basins used in the folding study. For each basin we
also report the residual wall energy actually paid <U> (the enthalpic part) and the
reweighting effective sample size (the reliability).

    <A>_R       = sum_i w_i A_i e^{-U_i/kT} / sum_i w_i e^{-U_i/kT}
    G_wall(B,R) = -kT ln <e^{-U(R)/kT}>_B                       wall cost on basin B
    dG_fold(R)  = dG_fold(inf) + G_wall(folded) - G_wall(unfolded)
    <U>_B,R     = <U e^{-U/kT}>_B / <e^{-U/kT}>_B               residual wall energy

A negative ddG_fold means confinement stabilises folding. Because the wall is purely
geometric and the residual <U> it imposes is small, that stabilisation is entropic.
"""
import csv
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 17,
    'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 15, 'axes.linewidth': 1.3,
})

KT      = 0.596    # kcal/mol at 300 K
ESS_MIN = 1.0      # % of basin frames; below this the reweighting is starved

d     = np.load('perframe.npz')
w, Q, rc, radii = d['w'], d['Q'].astype(float), d['rc'], d['radii']
logw  = np.log(np.clip(w, 1e-300, None))

folded   = (Q >= 0.8) & (rc <= 0.6)
unfolded = (Q <= 0.1) & (rc >= 0.6)
print(f'frames {len(w)} | folded {folded.sum()} ({100*w[folded].sum():.4f}%) | '
      f'unfolded {unfolded.sum()} ({100*w[unfolded].sum():.2f}%)')


def wall_free_energy(mask, U):
    """Wall free energy, residual wall energy, and Kish ESS (% of basin) for a basin."""
    lw, u = logw[mask], U[mask]
    G   = -KT * (logsumexp(lw - u / KT) - logsumexp(lw))
    a   = np.exp((lw - u / KT) - (lw - u / KT).max())
    ess = a.sum() ** 2 / np.sum(a ** 2)
    return G, np.sum(a * u) / a.sum(), 100.0 * ess / mask.sum()


dG0 = -KT * np.log(w[folded].sum() / w[unfolded].sum())
print(f'dG_fold(bulk) = {dG0:.3f} kcal/mol\n')

rows = []
for R in radii:
    U = d[f'U_{R}']
    GF, UF, eF = wall_free_energy(folded, U)
    GU, UU, eU = wall_free_energy(unfolded, U)
    rows.append(dict(R=float(R), GF=GF, GU=GU, ddG=GF - GU, dG=dG0 + GF - GU,
                     UF=UF, UU=UU, eF=eF, eU=eU, ok=min(eF, eU) > ESS_MIN))

print(f'{"R":>5}{"dG_fold":>9}{"ddG":>8}{"Gwall_F":>9}{"Gwall_U":>9}{"<U>_U":>7}{"ESS_F%":>8}{"ESS_U%":>8}  trust')
for r in rows:
    print(f'{r["R"]:5.2f}{r["dG"]:9.3f}{r["ddG"]:8.3f}{r["GF"]:9.3f}{r["GU"]:9.3f}'
          f'{r["UU"]:7.3f}{r["eF"]:8.2f}{r["eU"]:8.2f}  {"yes" if r["ok"] else "no"}')

with open('fep_softwall_results.csv', 'w', newline='') as fh:
    cw = csv.writer(fh)
    cw.writerow(['R_nm', 'dG_fold', 'ddG_fold', 'Gwall_folded', 'Gwall_unfolded',
                 'Uresid_folded', 'Uresid_unfolded', 'ESS_folded_pct', 'ESS_unfolded_pct', 'trustworthy'])
    for r in rows:
        cw.writerow([r['R'], r['dG'], r['ddG'], r['GF'], r['GU'],
                     r['UF'], r['UU'], r['eF'], r['eU'], int(r['ok'])])

R   = np.array([r['R'] for r in rows])
ddG = np.array([r['ddG'] for r in rows])
GU  = np.array([r['GU'] for r in rows])
UU  = np.array([r['UU'] for r in rows])
eF  = np.array([r['eF'] for r in rows])
eU  = np.array([r['eU'] for r in rows])
ok  = np.array([r['ok'] for r in rows])

fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.axhline(0, ls='--', c='gray', lw=1)
ax.plot(R[ok], ddG[ok], '-o', c='#C44E58', ms=9, lw=2.2)
ax.set_xlabel('confinement radius R (nm)')
ax.set_ylabel(r'$\Delta\Delta G_\mathrm{fold}$ (kcal/mol)')
ax.invert_xaxis()
fig.tight_layout()
fig.savefig('fep_ddG.png', dpi=300)

fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.semilogy(R, eF, '-o', ms=9, lw=2.2, label='pre-folded')
ax.semilogy(R, eU, '-s', ms=9, lw=2.2, label='unfolded')
ax.axhline(ESS_MIN, ls=':', c='red', lw=1.6)
ax.set_xlabel('confinement radius R (nm)')
ax.set_ylabel('effective sample size (% of frames)')
ax.invert_xaxis()
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('fep_ess.png', dpi=300)

# The wall free energy of the unfolded basin is almost entirely entropic: the residual
# wall energy <U> it actually carries stays near zero.
fig, ax = plt.subplots(figsize=(6.8, 5.2))
ax.plot(R[ok], GU[ok], '-s', c='#3E7CB1', ms=9, lw=2.2, label=r'wall free energy $G_\mathrm{wall}$')
ax.plot(R[ok], UU[ok], '-o', c='#E1A140', ms=9, lw=2.2, label=r'residual wall energy $\langle U\rangle$')
ax.fill_between(R[ok], UU[ok], GU[ok], alpha=0.15, color='#3E7CB1', label=r'entropic part ($-T\Delta S$)')
ax.set_xlabel('confinement radius R (nm)')
ax.set_ylabel('kcal/mol  (unfolded basin)')
ax.invert_xaxis()
ax.legend(frameon=False, fontsize=14)
fig.tight_layout()
fig.savefig('fep_entropy.png', dpi=300)
print('\nwrote fep_softwall_results.csv, fep_ddG.png, fep_ess.png, fep_entropy.png')
