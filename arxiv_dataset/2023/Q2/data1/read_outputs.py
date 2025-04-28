#! /usr/bin/env python

'''read SN code-comparison output files

To produce example plots based on files in https://github.com/sn-rad-trans/data1 simply run:

   python read_outputs.py /path/to/file1/output_file1.txt /path/to/file2/output_file2.txt etc.
   python read_outputs.py /path/to/file*/output_file*.txt

You can see the full list of options using:

   python read_outputs.py --help
'''

import os
import sys
import numpy as np
from pdb import set_trace as stop

### ensure Python2 (2.6 or 2.7) and Python3 compatibility
if sys.version_info.major == 2:
    input = raw_input # input() to mean raw_input() when running Python2

###############################################################################

def read_lbol_edep(file, checkformat=False):

    '''
lbol_edep_<model>_<code>.txt file format:

#<optional comment lines>
#NTIMES: int
#time[d] Lbol[erg/s] Edep[erg/s]
float float float
etc.
    '''

    print('INFO - reading file ' + file)

    with open(file, 'r') as f:

        ### perform a strict test of the file format
        if checkformat:
            okfmt = 0
            iline = 0
            ncols = 3
            while okfmt == 0:
                line = f.readline()
                iline += 1
                if line[0] != '#':
                    print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                    print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                    sys.exit(read_lbol_edep.__doc__)
                elif 'NTIMES' in line:
                    if line[:8] != '#NTIMES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NTIMES: int"')
                        sys.exit(read_lbol_edep.__doc__)
                    else:
                        try:
                            nt = int(line.split()[1])
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NTIMES: int"')
                            sys.exit(read_lbol_edep.__doc__)
                    line = f.readline()
                    iline += 1
                    if ' '.join(line.split()) != '#time[d] Lbol[erg/s] Edep[erg/s]':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#time[d] Lbol[erg/s] Edep[erg/s]"')
                        sys.exit(read_lbol_edep.__doc__)
                    else:
                        print('INFO - header conforms to standard. Will now check numerical content')
                        for i in range(nt):
                            line = f.readline()
                            iline += 1
                            if not line:
                                print('ERROR - format check FAILED. EOF reached (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' ===> check NTIMES={:d} value in header'.format(nt))
                                sys.exit(read_lbol_edep.__doc__)
                            elif len(line.rstrip()) == 0:
                                print('ERROR - format check FAILED. Empty line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                sys.exit(read_lbol_edep.__doc__)
                            elif len(line.split()) != ncols:
                                print('ERROR - format check FAILED. File contains {:d} instead of {:d} columns (line {:d})'.format(len(line.split()), ncols, iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting {:d} columns'.format(ncols))
                                sys.exit(read_lbol_edep.__doc__)
                            isfinite = [np.isfinite(float(line.split()[ii])) for ii in range(ncols)]
                            if False in isfinite:
                                print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "float float float"')
                                sys.exit(read_lbol_edep.__doc__)
                        # check there are no lines left
                        line = f.readline()
                        iline += 1
                        if line:
                            print('ERROR - lines remaining after NTIMES={:d} lines read (line {:d})'.format(nt, iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' ===> check NTIMES={:d} value in header'.format(nt))
                            if len(line.rstrip()) == 0:
                                print(' ===> check for empty lines!')
                            sys.exit(read_lbol_edep.__doc__)
                        else:
                            print('INFO - format check OK!')
                            okfmt = 1

        # no strict format check
        else:
                            
            ### read in header
            okhdr = 0
            while okhdr == 0:
                line = f.readline()
                if 'NTIMES' in line:
                    nt = int(line.split()[1])
                    okhdr = 1

    ### read numerical data
    t, lbol, edep = np.loadtxt(file, comments='#', unpack=True)
                        
    # check values
    if np.unique(t).size != t.size:
        u, c = np.unique(t, return_counts=True)
        print('ERROR - duplicate entries in time array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_lbol_edep.__doc__)

    # output
    out = {}
    out['time'] = t
    out['lbol'] = lbol
    out['edep'] = edep
    out['units'] = 'time: days\nlbol: erg/s\nedep: erg/s'

    return out
    
###############################################################################

def read_edep(file, checkformat=False):

    '''
edep_<model>_<code>.txt file format:

#<optional comment lines>
#NTIMES: int
#NVEL: int
#TIMES[d]: float float ... float
#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]
float float float ... float
etc.
    '''

    print('INFO - reading file ' + file)
    
    with open(file, 'r') as f:
        
        ### perform a strict test of the file format
        if checkformat:
            okfmt = 0
            iline = 0
            while okfmt == 0:
                line = f.readline()
                iline += 1
                if line[0] != '#':
                    print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                    print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                    sys.exit(read_edep.__doc__)
                elif 'NTIMES' in line:
                    if line[:8] != '#NTIMES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NTIMES: int"')
                        sys.exit(read_edep.__doc__)
                    else:
                        try:
                            nt = int(line.split()[1])
                            ncols = nt + 1
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NTIMES: int"')
                            sys.exit(read_edep.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:6] != '#NVEL:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NVEL: int"')
                        sys.exit(read_edep.__doc__)
                    else:
                        try:
                            nv = int(line.split()[1])
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NVEL: int"')
                            sys.exit(read_edep.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:10] != '#TIMES[d]:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#TIMES[d]: float float ... float"')
                        sys.exit(read_edep.__doc__)
                    elif len(line.split()) != ncols:
                        print('ERROR - format check FAILED. TIMES contains {:d} entries instead of {:d} (line {:d})'.format(len(line.split())-1, nt, iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting NTIMES={:d} entries'.format(nt))
                        sys.exit(read_edep.__doc__)
                    isfinite = [np.isfinite(float(line.split()[ii+1])) for ii in range(nt)]
                    if False in isfinite:
                        print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "float float float ... float"')
                        sys.exit(read_edep.__doc__)
                    else:
                        tarr = np.array([float(line.split()[ii+1]) for ii in range(nt)])
                    line = f.readline()
                    iline += 1
                    if ' '.join(line.split()) != '#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]"')
                        sys.exit(read_edep.__doc__)
                    else:
                        print('INFO - header conforms to standard. Will now check numerical content')
                        for i in range(nv):
                            line = f.readline()
                            iline += 1
                            if not line:
                                print('ERROR - format check FAILED. EOF reached (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' ===> check NVEL={:d} value in header'.format(nv))
                                sys.exit(read_edep.__doc__)
                            elif len(line.rstrip()) == 0:
                                print('ERROR - format check FAILED. Empty line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                sys.exit(read_edep.__doc__)
                            elif len(line.split()) != ncols:
                                print('ERROR - format check FAILED. File contains {:d} instead of {:d} columns (line {:d})'.format(len(line.split()), ncols, iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting NTIMES+1={:d} columns'.format(ncols))
                                sys.exit(read_edep.__doc__)
                            isfinite = [np.isfinite(float(line.split()[ii])) for ii in range(ncols)]
                            if False in isfinite:
                                print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "float float float ... float"')
                                sys.exit(read_edep.__doc__)
                        # check there are no lines left
                        line = f.readline()
                        iline += 1
                        if line:
                            print('ERROR - lines remaining after NVEL={:d} lines read (line {:d})'.format(nv, iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' ===> check NVEL={:d} value in header'.format(nv))
                            if len(line.rstrip()) == 0:
                                print(' ===> check for empty lines!')
                            sys.exit(read_edep.__doc__)
                        else:
                            print('INFO - format check OK!')
                            okfmt = 1

        # no strict format check
        else:
                            
            ### read in header
            okhdr = 0
            while okhdr == 0:
                line = f.readline()
                if 'NTIMES' in line:
                    nt = int(line.split()[1])
                elif 'NVEL' in line:
                    nv = int(line.split()[1])
                elif 'TIMES' in line:
                    split_line = line.split()
                    tarrstr = split_line[1:]
                    tarr = np.array([float(tt) for tt in tarrstr])
                    okhdr = 1

    ### read numerical data
    vals = np.loadtxt(file)
    velarr = vals[:,0]
    edeparr = np.zeros((nt,nv))
    for it in range(nt):
        edeparr[it,:] = vals[:,it+1]

    # check values
    if np.unique(tarr).size != tarr.size:
        u, c = np.unique(tarr, return_counts=True)
        print('ERROR - duplicate entries in time array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_edep.__doc__)
    elif np.unique(velarr).size != velarr.size:
        u, c = np.unique(velarr, return_counts=True)
        print('ERROR - duplicate entries in velocity array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_edep.__doc__)
    elif velarr.max() > 3e5:
        print('ERROR - maximum velocity exceeds speed of light (km/s)!')
        print(' max(vel)= {:.5e} km/s'.format(velarr.max()))
        print(' ===> velocity expected in km/s (not cm/s)')
        sys.exit(read_edep.__doc__)

    # output
    out = {}
    out['time'] = tarr
    out['vel'] = velarr
    out['edep'] = edeparr
    out['units'] = 'time: days\nvel: km/s\nedep: erg/s/cm^3'

    return out
    
###############################################################################

def read_phys(file, checkformat=False):

    '''
phys_<model>_<code>.txt file format:

#<optional comment lines>
#NTIMES: int
#TIMES[d]: float float ... float
#
#TIME: float
#NVEL: int
#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3] 
float float float float float
etc.
#TIME: float
#NVEL: int
#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]
float float float float float
etc.
    '''

    print('INFO - reading file ' + file)
    
    with open(file, 'r') as f:
        
        ### perform a strict test of the file format
        if checkformat:
            okhdr = 0
            iline = 0
            ncols = 5
            while okhdr == 0:
                line = f.readline()
                iline += 1
                if line[0] != '#':
                    print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                    print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                    sys.exit(read_phys.__doc__)
                elif 'NTIMES' in line:
                    if line[:8] != '#NTIMES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NTIMES: int"')
                        sys.exit(read_phys.__doc__)
                    else:
                        try:
                            nt = int(line.split()[1])
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NTIMES: int"')
                            sys.exit(read_phys.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:10] != '#TIMES[d]:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#TIMES[d]: float float ... float"')
                        sys.exit(read_phys.__doc__)
                    elif len(line.split()) != nt+1:
                        print('ERROR - format check FAILED. TIMES contains {:d} entries instead of {:d} (line {:d})'.format(len(line.split())-1, nt, iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting NTIMES={:d} entries'.format(nt))
                        sys.exit(read_phys.__doc__)
                    isfinite = [np.isfinite(float(line.split()[ii+1])) for ii in range(nt)]
                    if False in isfinite:
                        print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "float float float ... float"')
                        sys.exit(read_phys.__doc__)
                    else:
                        print('INFO - main header conforms to standard. Will now check intermediate headers and numerical content')
                        tarr = np.array([float(line.split()[ii+1]) for ii in range(nt)])
                        okhdr = 1
                        
            #
            # loop over times
            #
            okfmt = 0
            for it in range(nt):
                okhdr = 0
                while okhdr == 0:
                    line = f.readline()
                    iline += 1
                    if line[0] != '#':
                        print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        # NVEL is only defined from it=1 at this point
                        if it > 0:
                            print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                        sys.exit(read_phys.__doc__)
                    elif 'TIME' in line:
                        if line[:6] != '#TIME:':
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#TIME: float"')
                            sys.exit(read_phys.__doc__)
                        else:
                            try:
                                t = float(line.split()[1])
                            except ValueError:
                                print('ERROR - conflicting header line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "#TIME: float"')
                                sys.exit(read_phys.__doc__)
                            if t != tarr[it]:
                                print('ERROR - conflicting time entries (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "#TIME: {:s}" based on TIMES[d] entries'.format(str(tarr[it])))
                                sys.exit(read_phys.__doc__)
                        line = f.readline()
                        iline += 1
                        if line[:6] != '#NVEL:':
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NVEL: int"')
                            sys.exit(read_phys.__doc__)
                        else:
                            try:
                                nv = int(line.split()[1])
                            except ValueError:
                                print('ERROR - conflicting header line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "#NVEL: int"')
                                sys.exit(read_phys.__doc__)
                        line = f.readline()
                        iline += 1
                        if ' '.join(line.split()) != '#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]':
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]"')
                            sys.exit(read_phys.__doc__)
                        else:
                            # intermediate header OK
                            okhdr = 1

                #
                # loop over velocities
                #
                for i in range(nv):
                    line = f.readline()
                    iline += 1
                    if not line:
                        print('ERROR - format check FAILED. EOF reached (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                        sys.exit(read_phys.__doc__)
                    elif len(line.rstrip()) == 0:
                        print('ERROR - format check FAILED. Empty line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        sys.exit(read_phys.__doc__)
                    elif len(line.split()) != ncols:
                        print('ERROR - format check FAILED. File contains {:d} instead of {:d} columns (line {:d})'.format(len(line.split()), ncols, iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting NTIMES+1={:d} columns'.format(ncols))
                        print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                        sys.exit(read_phys.__doc__)
                    isfinite = [np.isfinite(float(line.split()[ii])) for ii in range(ncols)]
                    if False in isfinite:
                        print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "float float float float float"')
                        sys.exit(read_phys.__doc__)
                                    
                # increment okfmt
                okfmt += 1
                    
            # check on okfmt
            # check there are no lines left
            line = f.readline()
            iline += 1
            if line:
                print('ERROR - lines remaining after NVEL={:d} lines read (line {:d})'.format(nv, iline))
                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                if len(line.rstrip()) == 0:
                    print(' ===> check for empty lines!')
                sys.exit(read_phys.__doc__)
            elif okfmt == nt:
                print('INFO - format check OK!')
                okfmt = 1
            else:
                print('ERROR - format check FAILED (line {:d})'.format(iline))
                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                sys.exit(read_phys.__doc__)
                            
        # no strict format check
        else:
                            
            ### read in header
            okhdr = 0
            while okhdr == 0:
                line = f.readline()
                if 'NTIMES' in line:
                    nt = int(line.split()[1])
                elif 'TIMES' in line:
                    split_line = line.split()
                    tarrstr = split_line[1:]
                    tarr = np.array([float(tt) for tt in tarrstr])
                    okhdr = 1

    # check values
    if np.unique(tarr).size != tarr.size:
        u, c = np.unique(tarr, return_counts=True)
        print('ERROR - duplicate entries in time array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_phys.__doc__)                    
                    
    # re-open file to read in numerical values
    # not ideal, but would otherwise need to keep track of NVEL at each time
    with open(file, 'r') as f:
        
        ### output lists of arrays
        vel, temp, dens, ne, natom = [], [], [], [], []
                
        ### loop over times
        for it in range(nt):
            
            while 1:
                line = f.readline()
                if 'NVEL' in line:
                    split_line = line.split()
                    nvel = int(split_line[1])
                elif 'vel_mid' in line:
                    break

            # temporary arrays
            veltmp = np.zeros(nvel)
            temptmp = np.zeros(nvel)
            denstmp = np.zeros(nvel)
            netmp = np.zeros(nvel)
            natomtmp = np.zeros(nvel)
                
            for iv in range(nvel):
                line = f.readline()
                split_line = line.split()
                veltmp[iv] = float(split_line[0])
                temptmp[iv] = float(split_line[1])
                denstmp[iv] = float(split_line[2])
                netmp[iv] = float(split_line[3])
                natomtmp[iv] = float(split_line[4])

            # check values
            if np.unique(veltmp).size != veltmp.size:
                u, c = np.unique(veltmp, return_counts=True)
                print('ERROR - duplicate entries in velocity array:', u[c > 1])
                print(' ===> check number of significant digits')
                sys.exit(read_phys.__doc__)
            elif veltmp.max() > 3e5:
                print('ERROR - maximum velocity exceeds speed of light (km/s)!')
                print(' max(vel)= {:.5e} km/s'.format(veltmp.max()))
                print(' ===> velocity expected in km/s (not cm/s)')
                sys.exit(read_phys.__doc__)

            # append to lists
            vel.append(veltmp)
            temp.append(temptmp)
            dens.append(denstmp)
            ne.append(netmp)
            natom.append(natomtmp)
                
    # output
    out = {}
    out['time'] = np.array(tarr)
    out['vel'] = vel
    out['temp'] = temp
    out['dens'] = dens
    out['ne'] = ne
    out['natom'] = natom
    out['units'] = 'time: days\nvel: km/s\ntemp: K\ndens: g/cm^3\nne: /cm^3\nnatom: /cm^3'

    return out
    
###############################################################################

def read_ionfrac(file, checkformat=False):

    '''
ionfrac_<element>_<model>_<code>.txt file format:

#<optional comment lines>
#NTIMES: int
#NSTAGES: int
#TIMES[d]: float float ... float
#
#TIME: float
#NVEL: int
#vel_mid[km/s] <element>0 <element>1 ... <element><NSTAGES-1>
float float float ... float
etc.
#TIME: float
#NVEL: int
#vel_mid[km/s] <element>0 <element>1 ... <element><NSTAGES-1>
float float float ... float
etc.
    '''

    print('INFO - reading file ' + file)
    
    # element name
    tmpstr = os.path.basename(file)[8:]
    elem = tmpstr[:tmpstr.index('_')]
    if not elem.islower():
        print('ERROR - element name in filename and file headers should be lowercase')
        print(' change element "{:s}" to "{:s}"'.format(elem, elem.lower()))
        sys.exit(read_ionfrac.__doc__)

    with open(file, 'r') as f:
        
        ### perform a strict test of the file format
        if checkformat:
            okhdr = 0
            iline = 0
            while okhdr == 0:
                line = f.readline()
                iline += 1
                if line[0] != '#':
                    print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                    print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                    sys.exit(read_ionfrac.__doc__)
                elif 'NTIMES' in line:
                    if line[:8] != '#NTIMES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NTIMES: int"')
                        sys.exit(read_ionfrac.__doc__)
                    else:
                        try:
                            nt = int(line.split()[1])
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NTIMES: int"')
                            sys.exit(read_ionfrac.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:9] != '#NSTAGES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NSTAGES: int"')
                        sys.exit(read_ionfrac.__doc__)
                    else:
                        try:
                            nstages = int(line.split()[1])
                            ncols = nstages + 1
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NSTAGES: int"')
                            sys.exit(read_ionfrac.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:10] != '#TIMES[d]:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#TIMES[d]: float float ... float"')
                        sys.exit(read_ionfrac.__doc__)
                    elif len(line.split()) != nt+1:
                        print('ERROR - format check FAILED. TIMES contains {:d} entries instead of {:d} (line {:d})'.format(len(line.split())-1, nt, iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting NTIMES={:d} entries'.format(nt))
                        sys.exit(read_ionfrac.__doc__)
                    isfinite = [np.isfinite(float(line.split()[ii+1])) for ii in range(nt)]
                    if False in isfinite:
                        print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "float float float ... float"')
                        sys.exit(read_ionfrac.__doc__)
                    else:
                        print('INFO - main header conforms to standard. Will now check intermediate headers and numerical content')
                        tarr = np.array([float(line.split()[ii+1]) for ii in range(nt)])
                        okhdr = 1
                        
            #
            # loop over times
            #
            okfmt = 0
            for it in range(nt):
                okhdr = 0
                while okhdr == 0:
                    line = f.readline()
                    iline += 1
                    if line[0] != '#':
                        print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        # NVEL is only defined from it=1 at this point
                        if it > 0:
                            print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                        sys.exit(read_ionfrac.__doc__)
                    elif 'TIME' in line:
                        if line[:6] != '#TIME:':
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#TIME: float"')
                            sys.exit(read_ionfrac.__doc__)
                        else:
                            try:
                                t = float(line.split()[1])
                            except ValueError:
                                print('ERROR - conflicting header line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "#TIME: float"')
                                sys.exit(read_ionfrac.__doc__)
                            if t != tarr[it]:
                                print('ERROR - conflicting time entries (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "#TIME: {:s}" based on TIMES[d] entries'.format(str(tarr[it])))
                                sys.exit(read_ionfrac.__doc__)
                        line = f.readline()
                        iline += 1
                        if line[:6] != '#NVEL:':
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NVEL: int"')
                            sys.exit(read_ionfrac.__doc__)
                        else:
                            try:
                                nv = int(line.split()[1])
                            except ValueError:
                                print('ERROR - conflicting header line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "#NVEL: int"')
                                sys.exit(read_ionfrac.__doc__)
                        line = f.readline()
                        iline += 1
                        if ' '.join(line.split()) != '#vel_mid[km/s] ' + ' '.join([elem+str(ii) for ii in np.arange(nstages)]):
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#vel_mid[km/s] ' + ' '.join([elem+str(ii) for ii in np.arange(nstages)])+'"')
                            sys.exit(read_ionfrac.__doc__)
                        else:
                            # intermediate header OK
                            okhdr = 1

                #
                # loop over velocities
                #
                for i in range(nv):
                    line = f.readline()
                    iline += 1
                    if not line:
                        print('ERROR - format check FAILED. EOF reached (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                        sys.exit(read_ionfrac.__doc__)
                    elif len(line.rstrip()) == 0:
                        print('ERROR - format check FAILED. Empty line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        sys.exit(read_ionfrac.__doc__)
                    elif len(line.split()) != ncols:
                        print('ERROR - format check FAILED. File contains {:d} instead of {:d} columns (line {:d})'.format(len(line.split()), ncols, iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting NSTAGES+1={:d} columns'.format(ncols))
                        print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                        sys.exit(read_ionfrac.__doc__)
                    isfinite = [np.isfinite(float(line.split()[ii])) for ii in range(ncols)]
                    if False in isfinite:
                        print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "float float float ... float"')
                        sys.exit(read_ionfrac.__doc__)
                                    
                # increment okfmt
                okfmt += 1
                    
            # check on okfmt
            # check there are no lines left
            line = f.readline()
            iline += 1
            if line:
                print('ERROR - lines remaining after NVEL={:d} lines read (line {:d})'.format(nv, iline))
                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                print(' ===> also check NVEL={:d} value in intermediate header'.format(nv))
                if len(line.rstrip()) == 0:
                    print(' ===> check for empty lines!')
                sys.exit(read_ionfrac.__doc__)
            elif okfmt == nt:
                print('INFO - format check OK!')
                okfmt = 1
            else:
                print('ERROR - format check FAILED (line {:d})'.format(iline))
                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                sys.exit(read_ionfrac.__doc__)
                            
        # no strict format check
        else:
                            
            ### read in header
            okhdr = 0
            while okhdr == 0:
                line = f.readline()
                if 'NTIMES' in line:
                    nt = int(line.split()[1])
                elif 'NSTAGES' in line:
                    nstages = int(line.split()[1])
                elif 'TIMES' in line:
                    split_line = line.split()
                    tarrstr = split_line[1:]
                    tarr = np.array([float(tt) for tt in tarrstr])
                    okhdr = 1

    # check values
    if np.unique(tarr).size != tarr.size:
        u, c = np.unique(tarr, return_counts=True)
        print('ERROR - duplicate entries in time array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_ionfrac.__doc__)
                    
    # re-open file to read in numerical values
    # not ideal, but would otherwise need to keep track of NVEL at each time
    with open(file, 'r') as f:
        
        ### output lists of arrays
        vel, ionfrac = [], []
                
        ### loop over times
        for it in range(nt):
            
            while 1:
                line = f.readline()
                if 'NVEL' in line:
                    split_line = line.split()
                    nvel = int(split_line[1])
                elif 'vel_mid' in line:
                    break

            # temporary arrays
            veltmp = np.zeros(nvel)
            ionfractmp = np.zeros((nvel, nstages))
            for iv in range(nvel):
                line = f.readline()
                split_line = line.split()
                veltmp[iv] = float(split_line[0])
                ionfractmp[iv,:] = [float(x) for x in split_line[1:]]
                sumionfrac = np.sum(ionfractmp[iv,:])
                # need to check sumionfrac > 1e-20 to ensure element is actually present at this velocity
                if np.abs(sumionfrac - 1.0) > 2e-3 and sumionfrac > 1e-20:
                    print('ERROR - sum of ionization fractions should be equal to one!')
                    print(' TIME= {:s} d'.format(str(tarr[it])))
                    print(' vel_mid= {:s} km/s'.format(str(veltmp[iv])))
                    print(' Sum(ionfrac)= {:.5e}'.format(np.sum(ionfractmp[iv,:])))
                    sys.exit(read_ionfrac.__doc__)

            # check values
            if np.unique(veltmp).size != veltmp.size:
                u, c = np.unique(veltmp, return_counts=True)
                print('ERROR - duplicate entries in velocity array:', u[c > 1])
                print(' ===> check number of significant digits')
                sys.exit(read_ionfrac.__doc__)
            elif veltmp.max() > 3e5:
                print('ERROR - maximum velocity exceeds speed of light (km/s)!')
                print(' max(vel)= {:.5e} km/s'.format(veltmp.max()))
                print(' ===> velocity expected in km/s (not cm/s)')
                sys.exit(read_ionfrac.__doc__)
      
            # append to lists
            vel.append(veltmp)
            ionfrac.append(ionfractmp)

        # check values
        if np.sum(np.concatenate(ionfrac).ravel()) == 0.0:
            print('ERROR - ionization fraction = 0 everywhere for element: {:s}'.format(elem))
            print(' ===> only submit ionfrac_*.txt files for elements included in the calculation')
            sys.exit(read_ionfrac.__doc__)
            
    # output
    out = {}
    out['elem'] = elem
    out['nstages'] = nstages
    out['time'] = np.array(tarr)
    out['vel'] = vel
    out['ionfrac'] = ionfrac
    out['units'] = 'elem: N/A\nnstages: N/A\ntime: days\nvel: km/s\nionfrac: N/A'

    return out
    
###############################################################################

def read_spectra(file, checkformat=False):

    '''
spectra_<model>_<code>.txt file format:

#<optional comment lines>
#NTIMES: int
#NWAVE: int 
#TIMES[d]: float float ... float
#wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]
float float float ... float
etc.
    '''

    print('INFO - reading file ' + file)
    
    with open(file, 'r') as f:
        
        ### perform a strict test of the file format
        if checkformat:
            okfmt = 0
            iline = 0
            while okfmt == 0:
                line = f.readline()
                iline += 1
                if line[0] != '#':
                    print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                    print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                    sys.exit(read_spectra.__doc__)
                elif 'NTIMES' in line:
                    if line[:8] != '#NTIMES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NTIMES: int"')
                        sys.exit(read_spectra.__doc__)
                    else:
                        try:
                            nt = int(line.split()[1])
                            ncols = nt + 1
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NTIMES: int"')
                            sys.exit(read_spectra.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:7] != '#NWAVE:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NWAVE: int"')
                        sys.exit(read_spectra.__doc__)
                    else:
                        try:
                            nw = int(line.split()[1])
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NWAVE: int"')
                            sys.exit(read_spectra.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:10] != '#TIMES[d]:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#TIMES[d]: float float ... float"')
                        sys.exit(read_spectra.__doc__)
                    elif len(line.split()) != ncols:
                        print('ERROR - format check FAILED. TIMES contains {:d} entries instead of {:d} (line {:d})'.format(len(line.split())-1, nt, iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting NTIMES={:d} entries'.format(nt))
                        sys.exit(read_spectra.__doc__)
                    isfinite = [np.isfinite(float(line.split()[ii+1])) for ii in range(nt)]
                    if False in isfinite:
                        print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "float float float ... float"')
                        sys.exit(read_spectra.__doc__)
                    else:
                        tarr = np.array([float(line.split()[ii+1]) for ii in range(nt)])
                    line = f.readline()
                    iline += 1
                    if ' '.join(line.split()) != '#wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]"')
                        sys.exit(read_spectra.__doc__)
                    else:
                        print('INFO - header conforms to standard. Will now check numerical content')
                        for i in range(nw):
                            line = f.readline()
                            iline += 1
                            if not line:
                                print('ERROR - format check FAILED. EOF reached (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' ===> check NWAVE={:d} value in header'.format(nw))
                                sys.exit(read_spectra.__doc__)
                            elif len(line.rstrip()) == 0:
                                print('ERROR - format check FAILED. Empty line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                sys.exit(read_spectra.__doc__)
                            elif len(line.split()) != ncols:
                                print('ERROR - format check FAILED. File contains {:d} instead of {:d} columns (line {:d})'.format(len(line.split()), ncols, iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting NTIMES+1={:d} columns'.format(ncols))
                                sys.exit(read_spectra.__doc__)
                            isfinite = [np.isfinite(float(line.split()[ii])) for ii in range(ncols)]
                            if False in isfinite:
                                print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "float float float ... float"')
                                sys.exit(read_spectra.__doc__)
                        # check there are no lines left
                        line = f.readline()
                        iline += 1
                        if line:
                            print('ERROR - lines remaining after NWAVE={:d} lines read (line {:d})'.format(nw, iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' ===> check NWAVE={:d} value in header'.format(nw))
                            if len(line.rstrip()) == 0:
                                print(' ===> check for empty lines!')
                            sys.exit(read_spectra.__doc__)
                        else:
                            print('INFO - format check OK!')
                            okfmt = 1

        # no strict format check
        else:
                            
            ### read in header
            okhdr = 0
            while okhdr == 0:
                line = f.readline()
                if 'NTIMES' in line:
                    nt = int(line.split()[1])
                elif 'NWAVE' in line:
                    nw = int(line.split()[1])
                elif 'TIMES' in line:
                    split_line = line.split()
                    tarrstr = split_line[1:]
                    tarr = np.array([float(tt) for tt in tarrstr])
                    okhdr = 1

    ### read numerical data
    vals = np.loadtxt(file)
    warr = vals[:,0]
    farr = np.zeros((nt,nw))
    for it in range(nt):
        farr[it,:] = vals[:,it+1]

    # check values
    if np.unique(tarr).size != tarr.size:
        u, c = np.unique(tarr, return_counts=True)
        print('ERROR - duplicate entries in time array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_spectra.__doc__)
    elif np.unique(warr).size != warr.size:
        u, c = np.unique(warr, return_counts=True)
        print('ERROR - duplicate entries in wavelength array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_spectra.__doc__)
    elif warr.max() < 1e3:
        print('ERROR - maximum wavelength < 1000 Angstroms')
        print(' max(wave)= {:.5e} Angstroms'.format(warr.max()))
        print(' ===> wavelength expected in Angstroms (not, e.g., cm)')
        sys.exit(read_spectra.__doc__)
        
    # output
    out = {}
    out['time'] = np.array(tarr)
    out['wave'] = warr
    out['flux'] = farr
    out['units'] = 'time: days\nwave: angstroms (A)\nflux: erg/s/A'

    return out
    
###############################################################################

def read_mags(file, checkformat=False):

    '''
wsynphot_mags_<model>_<code>.txt file format:

#<optional header lines>
#NTIMES: int
#NBANDS: int
#time[d] <filtername_1> <filtername_2> ... <filtername_NBANDS>
float float float ... float
    '''

    print('INFO - reading file ' + file)
    
    with open(file, 'r') as f:
        
        ### perform a strict test of the file format
        if checkformat:
            okfmt = 0
            iline = 0
            while okfmt == 0:
                line = f.readline()
                iline += 1
                if line[0] != '#':
                    print('ERROR - header lines should start with "#" (line {:d})'.format(iline))
                    print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                    sys.exit(read_mags.__doc__)
                elif 'NTIMES' in line:
                    if line[:8] != '#NTIMES:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NTIMES: int"')
                        sys.exit(read_mags.__doc__)
                    else:
                        try:
                            nt = int(line.split()[1])
                            ncols = nt + 1
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NTIMES: int"')
                            sys.exit(read_mags.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:8] != '#NBANDS:':
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#NBANDS: int"')
                        sys.exit(read_mags.__doc__)
                    else:
                        try:
                            nbands = int(line.split()[1])
                            ncols = nbands + 1
                        except ValueError:
                            print('ERROR - conflicting header line (line {:d})'.format(iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' expecting "#NBANDS: int"')
                            sys.exit(read_mags.__doc__)
                    line = f.readline()
                    iline += 1
                    if line[:8] != '#time[d]' or len(line.split()) != ncols:
                        print('ERROR - conflicting header line (line {:d})'.format(iline))
                        print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                        print(' expecting "#time[d] followed by NBANDS={:d} filter names"'.format(nbands))
                        sys.exit(read_mags.__doc__)
                    else:
                        bands = line.split()[1:]
                        print('INFO - header conforms to standard. Will now check numerical content')
                        for i in range(nt):
                            line = f.readline()
                            iline += 1
                            if not line:
                                print('ERROR - format check FAILED. EOF reached (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' ===> check NTIMES={:d} value in header'.format(nt))
                                sys.exit(read_mags.__doc__)
                            elif len(line.rstrip()) == 0:
                                print('ERROR - format check FAILED. Empty line (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                sys.exit(read_mags.__doc__)
                            elif len(line.split()) != ncols:
                                print('ERROR - format check FAILED. File contains {:d} instead of {:d} columns (line {:d})'.format(len(line.split()), ncols, iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting NBANDS+1={:d} columns'.format(ncols))
                                sys.exit(read_mags.__doc__)
                            # NaN is acceptable as a null value in magnitude files
                            isfinite = [np.isfinite(float(line.split()[ii])) or np.isnan(float(line.split()[ii])) for ii in range(ncols)]
                            if False in isfinite:
                                print('ERROR - non-finite numerical values (line {:d})'.format(iline))
                                print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                                print(' expecting "float float float ... float"')
                                sys.exit(read_mags.__doc__)
                        # check there are no lines left
                        line = f.readline()
                        iline += 1
                        if line:
                            print('ERROR - lines remaining after NTIMES={:d} lines read (line {:d})'.format(nt, iline))
                            print(' line {:d}: "{:s}"'.format(iline, line.rstrip()))
                            print(' ===> check NTIMES={:d} value in header'.format(nt))
                            if len(line.rstrip()) == 0:
                                print(' ===> check for empty lines!')
                            sys.exit(read_mags.__doc__)
                        else:
                            print('INFO - format check OK!')
                            okfmt = 1

        # no strict format check
        else:
                            
            ### read in header
            okhdr = 0
            while okhdr == 0:
                line = f.readline()
                if 'NTIMES' in line:
                    nt = int(line.split()[1])
                elif 'NBANDS' in line:
                    nbands = int(line.split()[1])
                elif 'time[d]' in line:
                    split_line = line.split()
                    bands = split_line[1:]
                    okhdr = 1

    ### read numerical data
    vals = np.loadtxt(file)
    time = vals[:,0]
    mag = {b: None for b in bands}
    for ib, b in enumerate(bands):
        mag[b] = vals[:,ib+1]
                        
    # check values
    if np.unique(time).size != time.size:
        u, c = np.unique(time, return_counts=True)
        print('ERROR - duplicate entries in time array:', u[c > 1])
        print(' ===> check number of significant digits')
        sys.exit(read_mags.__doc__)
        
    # output
    out = {}
    out['time'] = time
    out['bands'] = bands
    out['mag'] = mag
    out['units'] = 'time: days\nbands: N/A\nmag: mag'
    
    return out
    
###############################################################################

if __name__ == '__main__':

    import argparse
    import glob
    import matplotlib.pyplot as plt

    # parse command-line options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('filename', default=None, type=str, nargs='*', help='e.g., /path/to/file/lbol_edep_*.txt')
    parser.add_argument('--time', default=20., type=float, help='pick a specific time for spectra, edep, phys, ionfrac')
    parser.add_argument('--log', action='store_true', help='log scale for Y-axis (overrides default)')
    parser.add_argument('--nolog', action='store_true', help='linear scale for Y-axis (overrides default)')
    parser.add_argument('--physvar', default='temp', type=str, help='variable to plot in phys*.txt file (temp, dens, ne, natom)')
    parser.add_argument('--xr', default=None, nargs=2, type=float, help="set X-range (two values), e.g. --xr 0. 15000.")
    parser.add_argument('--yr', default=None, nargs=2, type=float, help="set Y-range (two values), e.g. --yr 1e-15 1e-10")
    parser.add_argument('--checkformat', action='store_true', help='performs strict tests on files and reports conflicting formats')
    parser.add_argument('--noplot', action='store_true', help='skip plotting (useful, e.g., in combination with --checkformat')
    args = parser.parse_args()

    # list of files with errors
    files_with_errors = []

    # prepare for plotting
    if not args.noplot:
        plt.ion()
        fig = plt.figure()
        
    # loop over files
    for filename in args.filename:
        
        # determine file type
        basename = os.path.basename(filename)
        inverty = False # invert Y-axis (useful for magnitude plot)
        
        if basename[:9] == 'lbol_edep':
            try:
                lbol = read_lbol_edep(filename, checkformat=args.checkformat)
            except SystemExit as e:
                print(e)
                files_with_errors.append(filename)
            else:
                xx = lbol['time']
                yy = []
                label = []
                yy.append(lbol['lbol']) ; label.append('lbol')
                yy.append(lbol['edep']) ; label.append('edep')
                log = True
                xtit = 'Days since explosion'
                ytit = 'Log10(Lbol) [erg/s]'
                idx0 = 10
        elif basename[:4] == 'edep':
            try:
                edep = read_edep(filename, checkformat=args.checkformat)
            except SystemExit as e:
                print(e)
                files_with_errors.append(filename)
            else:
                idxt = np.argmin(abs(edep['time'] - args.time)) # pick closest time
                xx = edep['vel']
                yy = edep['edep'][idxt,:] ; label = str(edep['time'][idxt])+'d'
                log = True
                xtit = 'Velocity [km/s]'
                ytit = 'Edep [erg/s/cm^3]'
                idx0 = 5
        elif basename[:4] == 'phys':
            try:
                phys = read_phys(filename, checkformat=args.checkformat)
            except SystemExit as e:
                print(e)
                files_with_errors.append(filename)
            else:
                idxt = np.argmin(abs(phys['time'] - args.time)) # pick closest time
                xx = phys['vel'][idxt]
                if args.physvar == 'temp':
                    yy = phys['temp'][idxt]
                    log = False
                    ytit = 'Temperature [K]'
                elif args.physvar == 'dens':
                    yy = phys['dens'][idxt]                
                    log = True
                    ytit = 'Density [g/cm^3]'
                elif args.physvar == 'ne':
                    yy = phys['ne'][idxt]
                    log = True
                    ytit = 'Electron density [/cm^3]'
                elif args.physvar == 'natom':
                    yy = phys['natom'][idxt]
                    log = True
                    ytit = 'Atom density [/cm^3]'
                label = str(phys['time'][idxt])+'d'
                xtit = 'Velocity [km/s]'
                idx0 = 5
        elif basename[:7] == 'ionfrac':
            try:
                ionf = read_ionfrac(filename, checkformat=args.checkformat)
            except SystemExit as e:
                print(e)
                files_with_errors.append(filename)
            else:
                idxt = np.argmin(abs(ionf['time'] - args.time)) # pick closest time
                xx = ionf['vel'][idxt]
                yy = []
                label = []
                for i in range(ionf['nstages']):
                    yy.append(ionf['ionfrac'][idxt][:,i])
                    label.append(ionf['elem']+str(i))
                log = False
                xtit = 'Velocity [km/s]'
                ytit = 'Mass fraction'
                idx0 = 9 + len(ionf['elem'])
        elif basename[:7] == 'spectra':
            try:
                spec = read_spectra(filename, checkformat=args.checkformat)
            except SystemExit as e:
                print(e)
                files_with_errors.append(filename)
            else:
                idxt = np.argmin(abs(spec['time'] - args.time)) # pick closest time
                xx = spec['wave']
                yy = spec['flux'][idxt,:] ; label = str(spec['time'][idxt])+'d'
                log = False
                xtit = 'Wavelength [A]'
                ytit = 'Flux [erg/s/A]'
                idx0 = 8
        elif basename[:13] == 'wsynphot_mags':
            try:
                mags = read_mags(filename, checkformat=args.checkformat)
            except SystemExit as e:
                print(e)
                files_with_errors.append(filename)
            else:
                xx = mags['time']
                yy = [mags['mag'][b] for b in mags['bands']]
                label = mags['bands']
                log = False
                inverty = True
                xtit = 'Days since explosion'
                ytit = 'Absolute magnitude'
                idx0 = 14
        else:
            sys.exit('ERROR - unknown file type: ' + basename)

        # skip plotting if file contains errors
        if filename in files_with_errors:
            continue

        # plot
        if not args.noplot:
            
            # convert to lists
            if type(yy) is not list:
                yy = [yy]
                label = [label]
            
            # override default for log scale?
            if args.log:
                log = True
            elif args.nolog:
                log = False
            
            # get model and code name
            tmpstr = os.path.splitext(basename[idx0:])[0]
            model = tmpstr[:tmpstr.index('_')]
            code = tmpstr[tmpstr.index('_')+1:]
        
            # plot
            ax = fig.add_subplot(111)
            ax.set_xlabel(xtit)
            ax.set_ylabel(ytit)
            ax.set_title('Model: '+model+' ; Code: '+code)
            if args.xr is not None:
                ax.set_xlim(args.xr)
            if args.yr is not None:
                ax.set_ylim(args.yr)
            for iy, yvar in enumerate(yy):
                # exclude NULL values
                idx = np.where((yvar != 1e-99) & (~np.isnan(yvar)))[0]
                if not log:
                    ax.plot(xx[idx], yvar[idx], label=label[iy])
                else:
                    ax.semilogy(xx[idx], yvar[idx], label=label[iy])
            if inverty:
                plt.gca().invert_yaxis()
            ax.legend()
            plt.draw()
            zzz = input("===> Press <return> to quit ")
            plt.clf()

    # report files with errors
    if len(files_with_errors) > 0:
        print()
        print('#############################')
        print('# !!! Files with errors !!! #')
        print('#############################')
        print()
        for f in files_with_errors:
            print(f)
        print()
    elif args.checkformat:
        print()
        print('##########################################')
        print('# Congratulations! No files with errors! #')
        print('##########################################')
        print()
