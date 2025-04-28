#! /usr/bin/python
#

## this file is to calculate the line ratios resulting from different combinations of
## temperature and volume density
## We set it for HC3N (10-9) and (8-7)
import math
import os
import time
import numpy as np

# Grid boundaries
#
tmin = 8  # minimum kinetic temperature (K)
tmax = 16 # maximum kinetic temperature (K)
trange = np.linspace(tmin, tmax, num=9)
nmin = 1e4   # minimum H2 density (cm^-3)
nmax = 1e6   # maximum H2 density (cm^-3)
#
# Parameters to keep constant
#
tbg   = 2.73 # background radiation temperature
cdmol = 1e13 # low enough to be optically thin
dv    = 0.7  # line width (km/s)
bw    = 0.01 # "bandwidth": free spectral range around line

#
# Numerical parameters
#
ntemp = len(trange)-1  # number of temperature points
ndens = 30  # number of density points

# No user changes needed below this point.
#
def write_input(infile,tkin,nh2, flow, fupp):
    infile.write(mole+'.dat\n')
    infile.write('radex.out\n')
    infile.write(str(flow*(1-bw))+' '+str(fupp/(1-bw))+'\n')
    infile.write(str(tkin)+'\n')
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(nh2)+'\n')
    infile.write(str(tbg)+'\n')
    infile.write(str(cdmol)+'\n')
    infile.write(str(dv)+'\n')

def read_radex(outfile):
    line  = outfile.readline()
    words = line.split()
    while (words[1] != "T(kin)"):
        line  = outfile.readline()
        words = line.split()
    temp  = float(words[-1])
    line  = outfile.readline()
    words = line.split()
    dens  = float(words[-1])
    while (words[-1] != "FLUX"):
        line  = outfile.readline()
        words = line.split()
    line  = outfile.readline()
    line  = outfile.readline()
    words = line.split()
    ftmp  = float(words[4])
    while ((ftmp < flow*(1-bw)) or (ftmp > flow/(1-bw))):
        line  = outfile.readline()
        words = line.split()
        ftmp  = float(words[4])
    low   = float(words[-2])
    line  = outfile.readline()
    words = line.split()
    ftmp  = float(words[4])
    while ((ftmp < fupp*(1-bw)) or (ftmp > fupp/(1-bw))):
        line  = outfile.readline()
        words = line.split()
        ftmp  = float(words[4])
    upp   = float(words[-2])
    ratio = upp/low #low / upp
    return temp,dens,ratio

# Begin of main program

start = time.time()

mole = 'hc3n-h2'
act = [72.78382,90.97902,'HC3N_10-9_8-7.dat']


flow = act[0]
fupp = act[1]
gfil = act[2]
infile = open('radex.inp','w')
print ("Starting ",gfil)

for itemp in range(ntemp+1):
    for idens in range(ndens+1):
        temp = trange[itemp]
        # temp = tmin*((tmax/tmin)**(float(itemp)/ntemp))
        dens = nmin*((nmax/nmin)**(float(idens)/ndens))
        write_input(infile,temp,dens, flow, fupp)
        if (itemp == ntemp and idens == ndens):
            infile.write('0\n')
            infile.close()
        else:
            infile.write('1\n')

os.system('radex < radex.inp > /dev/null')
grid = open(gfil,'w')
fmt  = '%10.3e %10.3e %10.3e \n'

outfile  = open('radex.out')

rmin = 100
rmax = 0.1

for itemp in range(ntemp+1):
    for idens in range(ndens+1):
        # temp = tmin*((tmax/tmin)**(float(itemp)/ntemp))
        temp = trange[itemp]
        dens = nmin*((nmax/nmin)**(float(idens)/ndens))
        temp,dens,ratio = read_radex(outfile)

        if (ratio > 0.0):
            if (ratio < rmin):
                rmin = ratio
            if (ratio > rmax):
                rmax = ratio
        grid.write(fmt %(temp, math.log10(dens), ratio))

grid.close()
outfile.close()

    # print( "Min, max:", rmin, rmax)

stop = time.time()
dure = stop - start
print("Run time = ",dure, "seconds")
