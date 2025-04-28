from brainsmash.workbench.geo import cortex
path = ""
surface = path + "surf_cortex_left_2k.surf.gii"
cortex(surface=surface, outfile= path + "LeftDenseGeodesicDistmat.txt", euclid=False)
