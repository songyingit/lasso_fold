# How Lasso Peptides Fold

<p align="center">
  <img src="./Figures/TRAM.png" alt="Graphical Abstract"/>
</p>

<p align="center">
  <a href="">Paper</a> | <a href="">Data Repository</a> | <a href="">Code Repository</a>
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
- [License](#license)

## Abstract

This study combined extensive unbiased and biased MD simulations with advanced statistical model (TRAM) and deep learning framework (LPC-VAE) to systematically investigate the ability to form native pre-folded conformation in solution and the universal folding mechanisms of 20 structurally characterized lasso peptides.

## MD Simulations

### Systems

We studied 20 structurally characterized lasso peptides.

<p align="center">
  <img src="./Figures/Lassos.png" alt="Lasso Structures"/>
</p>

### Unbiased Simulations

Unbiased MD simulations were performed starting from the pre-folded structure and fully extended structure of all 20 lasso peptides. For each lasso peptide, we conducted ~200 µs unbiased MD simulations (200 trajectories from pre-folded and 200 trajectories from extended structures, each trajectory extending ~500 ns).

### Biased Simulations

Biased MD simulations (Umbrella Sampling) were performed using the Fraction of Native Contacts (Q) as the reaction coordinate. All frames from the unbiased simulations were discretized into 50 evenly distributed bins where Q ranged from 0 (completely unfolded) to 1 (pre-folded), and each bin was an independent umbrella sampling window. For each umbrella sampling window, we applied a harmonic potential based on the root-mean-square deviation (RMSD) of heavy atoms relative to their reference structure.

**Sample Code:** [`umbrella_sampling.py`]()

## Markov State Model (MSM)

Markov State Model (MSM) was first employed to connect multiple short MD simulation trajectories to capture the global information of conformational dynamics:

1. **Featurization:** Pairwise residue-residue distances
2. **Dimensionality reduction:** Time-lagged independent component analysis (tICA) to identify slow timescale components
3. **Clustering:** K-means clustering to discretize into microstates
4. **Hyperparameter optimization:** tIC dimensions (2-10) and microstate numbers (100-700) optimized by maximizing VAMP-2 score via 10-fold cross-validation

## Transition-based Reweighting Analysis Method (TRAM)

Integrates biased and unbiased simulations for enhanced sampling:

1. **Bias energy calculation:** Compute bias potential for all frames relative to all umbrella windows
2. **Thermodynamic state assignment:** Map frames to umbrella windows or unbiased ensemble
3. **Conformational state discretization:** Apply tICA separately to biased/unbiased data, combine features, and cluster with k-means
4. **TRAM implementation:** Construct multi-ensemble Markov model (MEMM) using pyEMMA

## Thermodynamic Analysis

**Folding free energy:** Calculated from TRAM stationary distributions. Pre-folded state: Q ≥ 0.8 and ring closure ≤ 7 Å; Unfolded state: Q ≤ 0.1 and ring closure ≥ 7 Å. Uncertainty estimated via bootstrap resampling (200 iterations).

**Loop Q relaxation time:** Quantifies folding stability by monitoring loop native contacts evolution from MEMM dynamics.

**Entropy cost:** Calculated using PARENT program with maximum information spanning tree (MIST) algorithm on 10,000 random frames per state.

## Kinetic Analysis

**Pathway clustering:** Latent-Space Path Clustering (LPC) identifies metastable folding pathways:

1. **Pathway identification:** TPT generates ~10,000 pathways (unfolded: Q ≤ 0.1; folded: Q ≥ 0.8)
2. **Pathway embedding:** Project each pathway onto three 2D tIC subspaces, discretize into 50×50 bins, concatenate into 7500-dimensional vectors
3. **VAE training:** Train variational autoencoder to map pathways to 2D latent space (100 epochs)
4. **Pathway clustering:** K-means clustering in latent space to identify metastable path channels, optimized by silhouette analysis

## License

**© 2025 Song Yin. All Rights Reserved.**

This code is made available for viewing purposes only. No permission is granted to use, copy, modify, or distribute this code for any purpose.