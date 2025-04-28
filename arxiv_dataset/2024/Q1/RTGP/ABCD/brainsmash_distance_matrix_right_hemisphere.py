# Script: Create distance matrix for right hemisphere between each vertex on the surface with Python package BrainSMASH (https://brainsmash.readthedocs.io/en/latest/index.html).
from brainsmash.workbench.geo import cortex
# Download surface template from Github at https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases
# Specifically, the file "R.sphere.32k_fs_LR.surf.gii" and set the path below.
surface = ""
# Calculate distance matrix.
cortex(surface=surface, outfile="RightDenseGeodesicDistmat.txt", euclid=False)
