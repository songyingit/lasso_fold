# How Lasso Peptide Fold

![TRAM](./Figures/TRAM.png)

## Table of Contents
- [Abstract] (#Abstract)
- [Molecular dynamics (MD) Data](#MD_Data)
  - [Getting started with this repo](#getting-started-with-this-repo)
  - [Usage](#usage)
- [Repository Structure](#-repository-structure)
  - [`code`](#code)
  - [`example_notebook`](#examplenotebook)
  - [`data`](#data)
- [Reference](#reference)

## Abstract
Lasso peptides are ribosomally synthesized and post-translationally modified peptide (RiPP) natural products that adopt a unique [1]rotaxane conformation. However, the universal folding principles underlying this conformation remain poorly understood, limiting mechanistic insight and rational design. In this study, we integrate extensive molecular dynamics (MD) simulations with advanced deep learning approaches to elucidate de novo folding across 20 structurally characterized lasso peptides in solution. The kinetic asymmetry between lasso peptide folding and unfolding, which hinders sampling of pre-folded states in unbiased simulations, presents fundamental challenges for constructing Markov State Model (MSM) under detailed balance. To overcome this, we employed Transition-based reweighting analysis (TRAM), a statistically optimal framework that incorporates biased simulations to enhance sampling of rarely visited states and reweights them to estimates a robust mutiensemble Markov State Model (MEMM). For each lasso peptide, we resolved folding free energy landscape, the most representative folding pathway and kinetics, and distinct pathway channels clustered by a deep learning-based variational autoencoder (VAE). Our results reveal a universal uphill folding free energy profile, with folding probabilities generally below 1%. Loop stability and entropy cost emerged as the principal determinants of folding efficiency. Microcin J25 exhibited a distinguished higher folding probability due to enhanced loop stability and reduced entropic penalties. Cell-free biosynthesis (CFB) experiments confirmed a strong correlation between β-sheet propensity in the loop re- gion and lasso production. Together, these findings provide a comprehensive model of lasso peptide folding, highlight determinants of stability and kinetics, and establish guiding principles for the rational engineering of synthetic lasso peptides with enhanced formation efficiency.

## License

**© 2025 Song Yin. All Rights Reserved.**

This code is made available for viewing purposes only. No permission is granted to use, copy, modify, or distribute this code for any purpose.
