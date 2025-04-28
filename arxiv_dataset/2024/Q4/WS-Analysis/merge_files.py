import pandas as pd

# This small script shows how to merge different WS files, created by create_files.py

file_names = [ "WS2003-23_long.h5", "WS2023-24_long.h5" ] # two files to be merged 
outfile = "WS2003-24_long.h5"                             # the new merged output file name

# Do not edit from here on

dff0 = pd.read_hdf(file_names[0])
dff1 = pd.read_hdf(file_names[1])

dff = pd.concat([dff0, dff1])

dff.to_hdf(outfile,key="dff",mode="w") #writes data to hdf file

