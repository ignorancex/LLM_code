#! /usr/bin/python
#
import math
import os
import sys
#
# Run a series of Radex models to retrieve the column density
#
# Author: Floris van der Tak
# First version: 27oct05
# This version:  06oct08

maxiter = 100
debug   = False

### TO DO:
### select freq from file (type ? at frequency prompt)

radexpath = '/home/mvaldivi/Radex/data/'
extension = '.dat'

mole = 'hc3n-h2'
#make sure the file exists (else Radex bonks)
if (os.path.exists(radexpath+mole+extension)):
    print ("Using data file",radexpath+mole+extension)
else:
    print ("Cannot find file: ",radexpath+mole+extension)
    sys.exit()

freq = 90.979
tkin = 12.4
nh2 = 1e4
tbg = 2.73
obs = 1
dv = 0.7
bw = 0.01
tol = 0.01

def write_input(cdmol):
    file = open('radex.inp','w')
    file.write(mole+'.dat\n')
    file.write('radex.out\n')
    file.write(str(freq*(1-bw))+' '+str(freq/(1-bw))+'\n')
    file.write(str(tkin)+'\n')
    file.write('1\n')
    file.write('H2\n')
    file.write(str(nh2)+'\n')
    file.write(str(tbg)+'\n')
    file.write(str(cdmol)+'\n')
    file.write(str(dv)+'\n')
    file.write('0\n')
    file.close()

def read_radex():
    file  = open('radex.out')
    lines = file.readlines()
    file.close()
    if (lines[-2].split()[-1] != '(erg/cm2/s)'):
        print ("Error: Ambiguous line selection. Reduce bandwidth?")
        print ("See radex.out for details")
        sys.exit()
    return float(lines[-1].split()[-2])

# Begin of main program
oflx = obs*dv
eps  = 1.0e-20
iter = 0

# Starting values of column density and fit residual
cdmol = 1e13
ratio = 0

while (ratio > (1+tol)) or (ratio < (1-tol)) :
    iter += 1
    write_input(cdmol)
    os.system('radex < radex.inp > /dev/null')
    mflx  = read_radex()
    if (mflx < eps):
        print ("Error: Zero or negative line intensity")
        print ("See radex.out for details")
        sys.exit()
    if (debug):
        print ("mflx= ",mflx)
    ratio = oflx/mflx
    cdmol = cdmol * ratio
    if (iter > maxiter):
        print("Maximum number of iterations exceeded")
        ratio = 1

fmt = "The best-fit column density is %7.2e cm^-2"
print(fmt % cdmol)
print("See radex.out for details")
