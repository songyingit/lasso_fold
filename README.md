# De novo Folding Mechanisms of Lasso Peptides

<p align="center">
  <img src="./Figures/TRAM.png" alt="Graphical Abstract"/>
</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.64898/2026.03.30.715466v1">Paper</a> | <a href="">Data Repository</a> | <a href="https://github.com/songyingit/lasso_fold/">Code Repository</a>
</p>

## Table of Contents
- [Abstract](#abstract)
- [MD Simulations](#md-simulations)
  - [Systems](#systems)
  - [Unbiased Simulations](#unbiased-simulations)
  - [Biased Simulations](#biased-simulations)
- [Markov State Model (MSM)](#markov-state-model-msm)
- [Transition-based Reweighting Analysis Method (TRAM)](#transition-based-reweighting-analysis-method-tram)
- [Thermodynamic Analysis](#thermodynamic-analysis)
- [Kinetic Analysis](#kinetic-analysis)
- [Threading and Ring-Closure Validation](#threading-and-ring-closure-validation)
- [Confinement Free Energy](#confinement-free-energy)
- [Peptide Engineering](#peptide-engineering)
- [Dependencies](#dependencies)
- [License](#license)

## Abstract

This study combined extensive unbiased and biased MD simulations with advanced statistical model (TRAM) and deep learning framework (LPC-VAE) to systematically investigate the ability to form native pre-folded conformation in solution and the folding mechanisms of 20 structurally characterized lasso peptides.

## MD Simulations

### Systems

We studied 20 structurally characterized lasso peptides lacking secondary post-translational modifications.

<p align="center">
  <img src="./Figures/Lassos.png" alt="Lasso Structures" width="600"/>
</p>

### Unbiased Simulations

Unbiased MD simulations were performed starting from the pre-folded structure and fully extended structure of all 20 lasso peptides. For each lasso peptide, we conducted ~200 µs unbiased MD simulations using OpenMM on the distributed computing platform Folding@home. 

### Biased Simulations

Biased MD simulations, Umbrella Sampling, were performed using the Fraction of Native Contacts (Q) as the reaction coordinate. All frames from the unbiased simulations were discretized into 50 evenly distributed bins where Q ranged from 0 (completely unfolded) to 1 (pre-folded), and each bin was an independent umbrella sampling window. For each umbrella sampling window, we applied a harmonic potential based on the root-mean-square deviation (RMSD) of heavy atoms relative to their reference structure.

  - **Sample Code:** [`umbrella_sampling.py`](https://github.com/songyingit/lasso_fold/tree/main/MD_data/umbrella_sampling.py)

## Markov State Model (MSM)

Markov State Model (MSM) was first employed to connect multiple short MD simulation trajectories to capture the global information of conformational dynamics:

1. **Featurization:** Pairwise residue-residue distances.

  - **Sample Code:** [`msm_feature.py`](https://github.com/songyingit/lasso_fold/tree/main/MSM/msm_feature.py)

2. **Dimensionality reduction:** Time-lagged independent component analysis (tICA) to identify slow timescale components.
3. **Clustering:** K-means clustering to discretize into microstates.
4. **Hyperparameter optimization:** tIC dimensions (2-10) and microstate numbers (100-700) optimized by maximizing VAMP-2 score via 10-fold cross-validation

  - **Sample Code:** [`msm_tica_cluster_hpopt.py`](https://github.com/songyingit/lasso_fold/tree/main/MSM/msm_tica_cluster_hpopt.py)

## Transition-based Reweighting Analysis Method (TRAM)

The kinetic asymmetry of folding for most of lasso peptides (unfolding proceeds much more rapidly than folding) poses fundamental challenge for the principle of detailed balance of MSM. To overcome these limitations, we implemented the Transition-based Reweighting Analysis Method (TRAM), a statistically optimal framework which leverages biased simulations to populate rarely visited states and estimating the thermodynamics and kinetics with reweighting estimators. 

The implementation of TRAM consisted of the following steps:

1. **Featurization:** Three quantities are computed for every frame of the biased and unbiased trajectories: pairwise Cα-Cα residue distances (the tICA input), the fraction of native contacts (Q), and the ring-closure distance, i.e. the distance between the N-terminus (backbone N of residue 1) and the side chain carboxylate carbon of the acceptor residue (Cδ of Glu or Cγ of Asp).

  - **Sample Code:** [`tram_feature.py`](https://github.com/songyingit/lasso_fold/tree/main/TRAM/tram_feature.py)

2. **Bias energy calculation:** Compute bias potential energy for all frames relative to all umbrella windows
3. **Thermodynamic state assignment (ttrajs):** Map frames to umbrella windows or unbiased ensemble. In total, there would be 'bias sampling windows' + 1 ensemble.  
4. **Conformational state discretization (dtrajs):** Apply tICA separately to biased/unbiased data, combine features, and cluster with k-means on TICA-transformed features to obtain dtraj.
5. **TRAM implementation:** Construct multi-ensemble Markov model (MEMM) using pyEMMA.

  - **Sample Code:** [`tram_implement.py`](https://github.com/songyingit/lasso_fold/tree/main/TRAM/tram_implement.py)

6. **Hyperparameter optimization:** Lag time for each system was optimized for TRAM analysis. 

  - **Sample Code:** [`tram_lagtime_opt.py`](https://github.com/songyingit/lasso_fold/tree/main/TRAM/tram_lagtime_opt.py)

## Thermodynamic Analysis

The pre-folded and unfolded states are defined jointly by Q and the ring-closure distance, because Q alone cannot separate an unthreaded proto-folded structure from a genuinely unfolded one. **Pre-folded state: Q ≥ 0.8 and ring-closure distance ≤ 6 Å; Unfolded state: Q ≤ 0.1 and ring-closure distance ≥ 6 Å.**

**Pre-folded population:** Sum of the TRAM stationary weights of all pre-folded frames. Uncertainty estimated via bootstrap resampling (100 iterations, 80% of frames per iteration).

  - **Sample Code:** [`prefolded_population.py`](https://github.com/songyingit/lasso_fold/tree/main/Analysis/prefolded_population.py)

**Folding free energy:** Calculated from TRAM stationary distributions as ∆G<sub>f</sub> = −RT ln(P<sub>pre-folded</sub> / P<sub>unfolded</sub>). Uncertainty estimated via bootstrap resampling (100 iterations, 80% of frames per iteration).

  - **Sample Code:** [`folding_free_energy.py`](https://github.com/songyingit/lasso_fold/tree/main/Analysis/folding_free_energy.py)

**Loop Q relaxation time:** Quantifies loop stability by monitoring loop native contacts evolution from MEMM dynamics.

  - **Sample Code:** [`loop_q_relax_time.py`](https://github.com/songyingit/lasso_fold/tree/main/Analysis/loop_q_relax_time.py)

**Entropy cost:** Calculated using [PARENT program](https://github.com/markusfleck/PARENT) with maximum information spanning tree (MIST) algorithm on 10,000 random frames drawn from the pre-folded and unfolded states as defined above.

  - **Sample Code:** [`entropy_frame_selection.py`](https://github.com/songyingit/lasso_fold/tree/main/Analysis/entropy_frame_selection.py), [`entropy_cost.py`](https://github.com/songyingit/lasso_fold/tree/main/Analysis/entropy_cost.py)

## Kinetic Analysis

**Pathway clustering:** Leveraged Variational AutoEncoder based Latent-space Path Clustering algorithm ([LPC-VAE](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00318)) identifing folding pathway channels and find the most representative pathway with maximum flux for each lasso peptide:

1. **Pathway identification and flux calculation:** Apply Transition path theory (TPT) to obtain folding pathways and calculate the corresponding flux. The source and sink sets are the microstates whose mean Q is below 0.1 (unfolded) and above 0.8 (pre-folded); if no microstate reaches a mean Q of 0.8, the single highest-Q microstate is used as the sink.
  - **Sample Code:** [`lpc_TPT_pathway_flux.py`](https://github.com/songyingit/lasso_fold/tree/main/LPC_VAE/lpc_TPT_pathway_flux.py)
2. **Pathway embedding:** Project each pathway onto three 2D tIC subspaces, discretize and then concatenate into 7500-dimensional 1-D vectors
  - **Sample Code:** [`lpc_ms_dist.py`](https://github.com/songyingit/lasso_fold/tree/main/LPC_VAE/lpc_ms_dist.py)
3. **VAE training:** Train Variational AutoEncoder (VAE) to map pathways to 2D latent space.
  - **Sample Code:** [`lpc_vae.py`](https://github.com/songyingit/lasso_fold/tree/main/LPC_VAE/lpc_vae.py)
4. **Pathway clustering:** K-means clustering and Silhouette analysis in the latent space to identify metastable pathway channels, and plot the flux-weighted pathways in the latent space for visualizations.
  - **Sample Code:** [`lpc_kmeans_cluster.py`](https://github.com/songyingit/lasso_fold/tree/main/LPC_VAE/lpc_kmeans_cluster.py)

## Threading and Ring-Closure Validation

The joint Q and ring-closure criterion is validated geometrically with [LATCHED](https://github.com/gabedahora/LATCHED), the threading detection algorithm of da Hora et al., which tests whether the tail penetrates the macrolactam ring and therefore separates pre-lasso conformations from unthreaded pre-tadpole ones.

1. **Pre-lasso vs pre-tadpole classification:** For each lasso peptide, all frames with a ring-closure distance ≤ 6 Å are passed to LATCHED and split into pre-lasso and pre-tadpole trajectories, with raw and TRAM-weighted fractions reported.
  - **Sample Code:** [`latched_split.py`](https://github.com/songyingit/lasso_fold/tree/main/Threading/latched_split.py)
2. **Pre-folded state purity:** Restricting the same classification to frames satisfying the full pre-folded definition (Q ≥ 0.8 and ring-closure ≤ 6 Å) quantifies how many of those conformations are pre-folded.
  - **Sample Code:** [`latched_q_rc_stats.py`](https://github.com/songyingit/lasso_fold/tree/main/Threading/latched_q_rc_stats.py)
3. **Other collective variable:** The harmonic linear discriminant analysis ([HLDA](https://pubs.acs.org/jpcbfk/article/128/17/4063/890133/One-Descriptor-to-Fold-Them-All-Harnessing)) collective variable of da Hora et al. is recomputed for microcin J25, together with the ring-closure distance, used as an alternative pair of coordinates; the basins of that landscape are then classified with LATCHED.
  - **Sample Code:** [`hlda_cv.py`](https://github.com/songyingit/lasso_fold/tree/main/Threading/hlda_cv.py), [`latched_basin_analysis.py`](https://github.com/songyingit/lasso_fold/tree/main/Threading/latched_basin_analysis.py)

## Confinement Free Energy

To quantify how the confinement provided by the cyclase pocket changes the folding thermodynamics, the TRAM ensemble of capistruin is reweighted with the same spherical flat-bottom wall used in the confinement MD simulations, U(R) = k Σ<sub>atoms</sub> max(0, |r − COM| − R)², k = 5 kcal·mol⁻¹·Å⁻². Because the wall only adds a position-dependent energy on top of the same force field, the confined ensemble is the unconfined ensemble reweighted by e<sup>−U/k<sub>B</sub>T</sup> (Zwanzig identity).

The wall free energy of a basin is G<sub>wall</sub>(B, R) = −k<sub>B</sub>T ln ⟨e<sup>−U(R)/k<sub>B</sub>T</sup>⟩<sub>B</sub>, and the relative stabilization of folding is ∆∆G<sub>f</sub>(R) = G<sub>wall</sub>(pre-folded, R) − G<sub>wall</sub>(unfolded, R). Reliability is judged by the reweighting effective sample size of both basins.

  - **Sample Code:** [`softwall_fep_perframe.py`](https://github.com/songyingit/lasso_fold/tree/main/Confinement/softwall_fep_perframe.py), [`softwall_fep.py`](https://github.com/songyingit/lasso_fold/tree/main/Confinement/softwall_fep.py)

## Peptide Engineering

A ring × loop factorial design tests whether the loop governs pre-folded stability for microcin J25 and klebsidin.They share the same ring size and ring-closure residues but differ sharply in loop β-hairpin content, so their loops were swapped reciprocally to give two chimeras, and all four systems were simulated with unbiased MD.

|  | klebsidin loop | microcin J25 loop |
|---|---|---|
| **klebsidin ring** | klebsidin (WT) | chimera1 |
| **microcin J25 ring** | chimera2 | microcin J25 (WT) |

1. **Chimera construction:** The loop is grafted between the two scaffolds in PyMOL by Cα-only superposition on the conserved ring and tail. Both chimeras are then solvated and minimized with the same protocol as the wild-type systems.
  - **Sample Code:** [`graft_chimera.py`](https://github.com/songyingit/lasso_fold/tree/main/Engineering/graft_chimera.py)
2. **Folded-state stability:** Loop β-hairpin content (fraction of loop-window residues assigned strict DSSP `E`) and the fraction of native contacts are compared across the four systems.
  - **Sample Code:** [`compare_ring_loop_factorial.py`](https://github.com/songyingit/lasso_fold/tree/main/Engineering/compare_ring_loop_factorial.py), [`loop_beta_hairpin.py`](https://github.com/songyingit/lasso_fold/tree/main/Engineering/loop_beta_hairpin.py)

## Dependencies

Analysis: Python 3, `pyemma`, `mdtraj`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `natsort`, `tol_colors`, `matplotlib`.

External tools: [OpenMM](https://openmm.org/) with ParmEd/AmberTools for the simulations, [PyMOL](https://pymol.org/) for the chimera grafts, [VMD](https://www.ks.uiuc.edu/Research/vmd/) for LATCHED, and [PARENT](https://github.com/markusfleck/PARENT) for the conformational entropy.

Each script carries a `pwd` placeholder at the top; set it to the root of the simulation data tree before running.

## License

**© 2026 Song Yin. All Rights Reserved.**

This code is made available for viewing purposes only. No permission is granted to use, copy, modify, or distribute this code for any purpose.
