"""
Write the parameter files to calculate entropy cost using PARENT program for different lasso peptides.

Input ensembles are produced by Analysis/entropy_frame_selection.py:
  folded_NC08   -> Q >= 0.8 and ring-closure distance <= 6 A
  unfolded_NC01 -> Q <= 0.1 and ring-closure distance >= 6 A
"""

lasso_name = ['acinetodin','astexin-1','benenodin-1','brevunsin', 'capistruin','caulonodin-V','caulosegnin-I','caulosegnin-II', 'chaxapeptin', 'citrocin',
                  'klebsidin', 'microcinJ25','rubrivinodin', 'sphaericin','sphingopyxin-I','streptomonomicin','subterisin','ubonodin','xanthomonin-I', 'xanthomonin-II']

for i, lasso in enumerate(lasso_name):
      for fs in ['unfolded_NC01', 'folded_NC08']:
            with open(f'parameters_{fs}_{lasso}', 'w') as file:
                        file.write('''#!/bin/bash

# Specify the GROMACS .xtc and .top input files relative to the top directory
# TRJ="test_system/UBQ_UBM2.xtc"
# TOP="test_system/UBQ_UBM2.top"
TRJ="'''+lasso+'''/'''+lasso+'''_select_frame_''' + fs + '''_RC_CAN_06_for_entropy.xtc"
TOP="'''+lasso+'''/'''+lasso+'''_nowat.top"

#Optional: Specify the index file and the group names for which (leave blank if not wanted)
#NDX="test_system/UBQ_UBM2.ndx"
#GRP1="r_1"
#GRP2="r_2"

# Specify your output directory
OUTDIR=output_'''+fs+'''_'''+lasso+'''

# Use the name of the working directory as a base name for the output files ( you can change this to e. g. to name="my_project" if you prefer ) 
NAME=`pwd | awk 'BEGIN{FS="/"}{print $(NF)}'` 

# Specify the number of  bins for bonds, angles and dihedrals (torsions) you want PARENT.x to use for building the 1D and 2D histograms for entropy calculation.
BBINS1D=50
ABINS1D=50
DBINS1D=50
BBINS2D=50
ABINS2D=50
DBINS2D=50

#  Specify the name of the backbone atoms as in the .top file (can be left empty, but gives more accurate results)
BACKBONE_ATOMS="CA C N H1 O1"

''')