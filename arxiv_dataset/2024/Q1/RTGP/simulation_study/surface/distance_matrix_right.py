from brainsmash.workbench.geo import cortex
path = ""
surface = path + "surf_cortex_right_2k.surf.gii"
cortex(surface=surface, outfile= path + "RightDenseGeodesicDistmat.txt", euclid=False)
