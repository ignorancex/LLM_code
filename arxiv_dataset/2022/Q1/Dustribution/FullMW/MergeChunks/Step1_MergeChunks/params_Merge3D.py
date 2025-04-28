import numpy as np

### Distance Chunks boundaries for weighting to be edited as we add more distance chunks
#!!!!!!!! To be edited by user!!!!!!!!!!!!
min_d_bounds_pred_Dchunk = [20, 700, 1300, 1900] #  ] #Dchunk1, 2,3,4..... #Boundary selected to pick the correct D chunk - can be any value within the given D chunk
dweight_lower_cutoff = [300, 800, 1300, 1900] # ] #Dchunk1, 2,3,4,..... #D weighting lower cutoff 
dweight_upper_cutoff = [800, 1300, 1900, 2400] #  ]   #Dchunk1, 2,3,4,..... #D weighting upper cutoff 
# chunk_boundaries_lower = [10, 600] #, 1200, 1800] #10, 600, 1200, 1800
# chunk_boundaries_upper = [1000, 1590] #, 2190, 2790] #1000, 1590, 2190

#These numbers are chose to produce an integer number of chunks that perfectly tessalate the sky as "squares" (in angular coordinates)
#In principle, other tesslatation schemes would be possible, but the other bits of code wouldn't be able to handle it properly
#Hence, if you change these numbers, make sure that you end up with the same conditions (0<= l < 360, integer number of chunks, etc)
l_sky_start = 0 #-180
l_sky_end = 361 #180
l_chunk_size = 18
b_lower = -90
b_upper = 90
b_step = 8

l_set = np.arange(l_sky_start, l_sky_end, l_chunk_size)
b_set = np.rad2deg(np.arcsin(np.linspace( np.sin(np.deg2rad(b_lower)), np.sin(np.deg2rad(b_upper)), (2*b_step)+1)))
