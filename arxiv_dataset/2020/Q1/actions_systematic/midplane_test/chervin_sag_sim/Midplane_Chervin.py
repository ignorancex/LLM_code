
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from astropy.constants import G as G_astropy
import astropy.units as u

import sys
from joblib import Parallel, delayed

from matplotlib import rc
import matplotlib as mpl
import math
import os


# In[2]:


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
'#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

nbootstrap = 1000
np.random.seed(162)

rcut = 0.5
zcut = 1.0
nspoke = 50

Rsolar = 8.2
Rmin = 7.2
Rmax = 9.2
dR = 0.1

pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
mpl.rcParams.update(pgf_with_rc_fonts)


# In[3]:


class snapshot_header:
    def __init__(self, filename):
        if (not os.path.exists(filename)):
            print("file not found:", filename)
            sys.exit()
      
        self.filename = filename  
        f = open(filename,'rb')    
        blocksize = np.fromfile(f,dtype=np.int32,count=1)
        if blocksize[0] == 8:
            swap = 0
            format = 2
        elif blocksize[0] == 256:
            swap = 0
            format = 1  
        else:
            blocksize.byteswap(True)
            if blocksize[0] == 8:
                swap = 1
                format = 2
            elif blocksize[0] == 256:
                swap = 1
                format = 1
            else:
                print("incorrect file format encountered when reading header of", filename)
                sys.exit()
    
        self.format = format
        self.swap = swap
    
        if format==2:
            f.seek(16, os.SEEK_CUR)
    
        self.npart = np.fromfile(f,dtype=np.int32,count=6)
        self.massarr = np.fromfile(f,dtype=np.float64,count=6)
        self.time = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.redshift = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.sfr = (np.fromfile(f,dtype=np.int32,count=1))[0]
        self.feedback = (np.fromfile(f,dtype=np.int32,count=1))[0]
        self.nall = np.fromfile(f,dtype=np.int32,count=6)
        self.cooling = (np.fromfile(f,dtype=np.int32,count=1))[0]
        self.filenum = (np.fromfile(f,dtype=np.int32,count=1))[0]
        self.boxsize = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.omega_m = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.omega_l = (np.fromfile(f,dtype=np.float64,count=1))[0]
        self.hubble = (np.fromfile(f,dtype=np.float64,count=1))[0]
    
        if swap:
            self.npart.byteswap(True)
            self.massarr.byteswap(True)
            self.time = self.time.byteswap()
            self.redshift = self.redshift.byteswap()
            self.sfr = self.sfr.byteswap()
            self.feedback = self.feedback.byteswap()
            self.nall.byteswap(True)
            self.cooling = self.cooling.byteswap()
            self.filenum = self.filenum.byteswap()
            self.boxsize = self.boxsize.byteswap()
            self.omega_m = self.omega_m.byteswap()
            self.omega_l = self.omega_l.byteswap()
            self.hubble = self.hubble.byteswap()
     
        f.close()
 


#----- find offset and size of data block ----- 

def find_block(filename, format, swap, block, block_num, only_list_blocks=False):
    if (not os.path.exists(filename)):
        print("file not found:", filename)
        sys.exit()
            
    f = open(filename,'rb')
    f.seek(0, os.SEEK_END)
    filesize = f.tell()
    f.seek(0, os.SEEK_SET)
  
    found = False
    curblock_num = 1
    while ((not found) and (f.tell()<filesize)):
        if format==2:
            f.seek(4, os.SEEK_CUR)
            curblock = f.read(4)
            if (block == curblock):
                found = True
            f.seek(8, os.SEEK_CUR)  
        else:
            if curblock_num==block_num:
                found = True
        
        curblocksize = (np.fromfile(f,dtype=np.int32,count=1))[0]
        if swap:
            curblocksize = curblocksize.byteswap()
    
    # - print some debug info about found data blocks -
    #if format==2:
    #  print curblock, curblock_num, curblocksize
    #else:
    #  print curblock_num, curblocksize
    
        if only_list_blocks:
            print(curblock_num,curblock,f.tell(),curblocksize)
            found = False
    
        if found:
            blocksize = curblocksize
            offset = f.tell()
        else:
            f.seek(curblocksize, os.SEEK_CUR)
            blocksize_check = (np.fromfile(f,dtype=np.int32,count=1))[0]
            if swap: blocksize_check = blocksize_check.byteswap()
            if (curblocksize != blocksize_check):
                print("something wrong")
                sys.exit()
            curblock_num += 1
      
    f.close()
      
    if ((not found) and (not only_list_blocks)):
        print("Error: block not found")
        sys.exit()
    
    if (not only_list_blocks):
        return offset,blocksize
 
# ----- read data block -----
 
def read_block(filename, block, parttype=-1, physical_velocities=True, arepo=0, no_masses=False, verbose=False):
    if (verbose):
	    print("reading block", block)
  
    blockadd=0
    blocksub=0
  
    if arepo==0:
        if (verbose):	
	        print("Gadget format")
        blockadd=0
    if arepo==1:
        if (verbose):	
	        print("Arepo format")
        blockadd=1	
    if arepo==2:
        if (verbose):
	        print("Arepo extended format")
        blockadd=4	
    if no_masses==True:
        if (verbose):	
	        print("No mass block present")    
        blocksub=1
		 
    if parttype not in [-1,0,1,2,3,4,5]:
        print("wrong parttype given")
        sys.exit()
  
    if os.path.exists(filename):
        curfilename = filename
    elif os.path.exists(filename+".0"):
        curfilename = filename+".0"
    else:
        print("file not found:", filename)
        print("and:", curfilename)
        sys.exit()
  
    head = snapshot_header(curfilename)
    format = head.format
    swap = head.swap
    npart = head.npart
    massarr = head.massarr
    nall = head.nall
    filenum = head.filenum
    redshift = head.redshift
    time = head.time
    del head
  
  # - description of data blocks -
  # add or change blocks as needed for your Gadget version
    data_for_type = np.zeros(6,bool) # should be set to "True" below for the species for which data is stored in the data block
    dt = np.float32 # data type of the data in the block
    if block=="POS ":
        data_for_type[:] = True
        dt = np.dtype((np.float32,3))
        block_num = 2
    elif block=="VEL ":
        data_for_type[:] = True
        dt = np.dtype((np.float32,3))
        block_num = 3
    elif block=="ID  ":
        data_for_type[:] = True
        dt = np.uint32
        block_num = 4
    elif block=="MASS":
        data_for_type[np.where(massarr==0)] = True
        block_num = 5
        if parttype>=0 and massarr[parttype]>0:   
            if (verbose):	
	            print("filling masses according to massarr")   
            return np.ones(nall[parttype],dtype=dt)*massarr[parttype]
    elif block=="U   ":
        data_for_type[:] = True
        dt = np.dtype((np.float32))
        block_num = 6#-blocksub
    elif block=="RHO ":
        data_for_type[0] = True
        block_num = 7-blocksub
    elif block=="VOL ":
        data_for_type[0] = True
        block_num = 8-blocksub 
    elif block=="CMCE":
        data_for_type[0] = True
        dt = np.dtype((np.float32,3))
        block_num = 9-blocksub 
    elif block=="AREA":
        data_for_type[0] = True
        block_num = 10-blocksub
    elif block=="NFAC":
        data_for_type[0] = True
        dt = np.dtype(np.int32)	
        block_num = 11-blocksub
    elif block=="NE  ":
        data_for_type[0] = True
        block_num = 8+blockadd-blocksub
    elif block=="NH  ":
        data_for_type[0] = True
        block_num = 9+blockadd-blocksub
    elif block=="HSML":
        data_for_type[0] = True
        block_num = 10+blockadd-blocksub
    elif block=="SFR ":
        data_for_type[0] = True
        block_num = 11+blockadd-blocksub
    elif block=="AGE ":
        data_for_type[4] = True
        block_num = 12+blockadd-blocksub
    elif block=="Z   ":
        data_for_type[0] = True
        data_for_type[4] = True
        block_num = 13+blockadd-blocksub
    elif block=="BHMA":
        data_for_type[5] = True
        block_num = 14+blockadd-blocksub
    elif block=="BHMD":
        data_for_type[5] = True
        block_num = 15+blockadd-blocksub
    elif block=="COOR":
        data_for_type[0] = True
        block_num = -1 
    else:
        print("Sorry! Block type", block, "not known!")
        sys.exit()
  # - end of block description -

    if (block_num < 0 and format==1):
        print("Sorry! Block number of", block, "not known! Unable to read this block from format 1 file!")
        sys.exit() 
    
    actual_data_for_type = np.copy(data_for_type)  
    if parttype >= 0:
        actual_data_for_type[:] = False
        actual_data_for_type[parttype] = True
        if data_for_type[parttype]==False:
            print("Error: no data for specified particle type", parttype, "in the block", block)   
            sys.exit()
    elif block=="MASS":
        actual_data_for_type[:] = True  
    
    allpartnum = np.int64(0)
    species_offset = np.zeros(6,np.int64)
    for j in range(6):
        species_offset[j] = allpartnum
        if actual_data_for_type[j]:
            allpartnum += nall[j]
    filenum=1  
    for i in range(filenum): # main loop over files
        if filenum>1:
            curfilename = filename+"."+str(i)
      
        if i>0:
            head = snapshot_header(curfilename)
            npart = head.npart  
            del head
      
        curpartnum = np.int32(0)
        cur_species_offset = np.zeros(6,np.int64)
        for j in range(6):
            cur_species_offset[j] = curpartnum
            if data_for_type[j]:
                curpartnum += npart[j]
    
        if parttype>=0:
            actual_curpartnum = npart[parttype]      
            add_offset = cur_species_offset[parttype] 
        else:
            actual_curpartnum = curpartnum
            add_offset = np.int32(0)
      
        offset,blocksize = find_block(curfilename,format,swap,block,block_num)
    
        if i==0: # fix data type for ID if long IDs are used
            if block=="ID  ":
                if blocksize == np.dtype(dt).itemsize*curpartnum * 2:
                    dt = np.uint64 
        
        if np.dtype(dt).itemsize*curpartnum != blocksize:
            print("something wrong with blocksize! expected =",np.dtype(dt).itemsize*curpartnum,"actual =",blocksize)
            sys.exit()
    
        f = open(curfilename,'rb')
        f.seek(offset + add_offset*np.dtype(dt).itemsize, os.SEEK_CUR)  
        curdat = np.fromfile(f,dtype=dt,count=actual_curpartnum) # read data
        f.close()  
        if swap:
            curdat.byteswap(True)  
      
        if i==0:
            data = np.empty(allpartnum,dt)
    
        for j in range(6):
            if actual_data_for_type[j]:
                if block=="MASS" and massarr[j]>0: # add mass block for particles for which the mass is specified in the snapshot header
                    data[species_offset[j]:species_offset[j]+npart[j]] = massarr[j]
                else:
                    if parttype>=0:
                        data[species_offset[j]:species_offset[j]+npart[j]] = curdat
                    else:
                        data[species_offset[j]:species_offset[j]+npart[j]] = curdat[cur_species_offset[j]:cur_species_offset[j]+npart[j]]
                species_offset[j] += npart[j]

        del curdat

    if physical_velocities and block=="VEL " and redshift!=0:
        data *= math.sqrt(time)

    return data
  
# ----- list all data blocks in a format 2 snapshot file -----

def list_format2_blocks(filename):
    if (not os.path.exists(filename)):
        print("file not found:", filename)
        sys.exit()
  
    head = snapshot_header(filename)
    format = head.format
    swap = head.swap
    del head
  
    if (format != 2):
        print("not a format 2 snapshot file")
        sys.exit()
            
    print("#   BLOCK   OFFSET   SIZE")
    print("-------------------------")
  
    find_block(filename, format, swap, "XXXX", 0, only_list_blocks=True)
  
    print("-------------------------")


# In[4]:


def gen_pos():
    theta = np.linspace(0, 2.*np.pi, nspoke)

    posx = Rsolar * np.cos(theta)
    posy = Rsolar * np.sin(theta)
    posz = np.zeros(len(posx))
    pos = np.transpose([posx, posy, posz])
    return theta, pos


# In[5]:


t = [1,2,3,4,5]
print(t[2:])
print(t[3:])


# In[6]:


def get_init_keys(p, star_pos):
    pos_diff = np.subtract(star_pos, p)
    rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
    rbool = rmag < rcut
    zbool = np.abs(pos_diff[:,2]) < 2.0 * zcut
    keys = np.where(np.logical_and(rbool, zbool))[0]
    return keys


# In[7]:


def get_keys(p, part):
    pos_diff = np.subtract(part, p)
    rmag = np.linalg.norm(pos_diff[:,:2], axis=1)
    rbool = rmag < rcut
    zbool = np.abs(pos_diff[:,2]) < zcut
    keys = np.where(np.logical_and(rbool, zbool))[0]
    return keys


# In[8]:


def _midplane_med_(pos, init_pos, init_vel):
    mid_pos = pos.copy()
    for _ in range(10):
        keys = get_keys(mid_pos, init_pos)
        mid_pos[2] = np.median(init_pos[:,2][keys])
    mid_vel = np.median(init_vel[:,2][keys])
    return mid_pos[2], mid_vel


# In[9]:


def get_midplane_with_error(pos, star_pos, star_vel):
    midplane = _midplane_med_
    # get all particles within 2x zheight
    init_keys = get_init_keys(pos, star_pos)
    init_pos = star_pos[init_keys]
    init_vel = star_vel[init_keys]

    # calculate midplane using all particles
    midplane_central, midplane_vel = midplane(pos, init_pos, init_vel)
    
    # prepare to bootstrap
    
    keys_to_choose = list(range(len(init_pos)))
    rand_choice = np.random.choice(keys_to_choose, len(init_pos)*nbootstrap)
    rand_choice = np.reshape(rand_choice, (nbootstrap, len(init_pos)))
    init_pos_rand = init_pos[rand_choice]
    init_vel_rand = init_vel[rand_choice]
    med_rand = np.array([ midplane(pos, ipos, ivel) for ipos,ivel in zip(init_pos_rand,init_vel_rand) ])
    dist_pos = np.subtract(med_rand[:,0], midplane_central)
    dist_vel = np.subtract(med_rand[:,1], midplane_vel)
    up_pos = np.percentile(dist_pos, 50+68/2)
    low_pos = np.percentile(dist_pos, 50-68/2)
    up_vel = np.percentile(dist_vel, 50+68/2)
    low_vel = np.percentile(dist_vel, 50-68/2)
    l = midplane_central - up_pos
    h = midplane_central - low_pos
    l_v = midplane_vel - up_vel
    h_v = midplane_vel - low_vel
    return midplane_central, l, h, midplane_vel, l_v, h_v


# In[10]:


def fit(x, theta):
    return x[0]*np.cos(theta+x[1]) + x[2]

def chisq(x, theta, midplane_est):
    return np.sum(np.square(np.subtract(fit(x, theta), midplane_est)))


# In[11]:


from astropy.table import Table
def main(gal):
    
    snap = Table.read('/home/douglas/Chervin/PhaseSpace'+gal+'.fits',format = 'fits')
    
    star_pos = np.zeros((len(snap["X"]),3))
    star_pos[:,0] = snap["X"]
    star_pos[:,1] = snap["Y"]
    star_pos[:,2] = snap["Z"]
    star_vel = np.zeros((len(snap["U"]),3))
    star_vel[:,0] = snap["U"]
    star_vel[:,1] = snap["V"]
    star_vel[:,2] = snap["U"]
    
    
    Rsnap = np.sqrt(star_pos[:,0]**2 + star_pos[:,1]**2)
    Rchoose = Rsnap - Rsolar
    ichoose = np.where( (np.abs(Rchoose)<0.5) & (np.abs(star_pos[:,2])<10) & (star_pos[:,2] < 1) & (star_pos[:,2] > -1))
    star_pos = star_pos[ichoose]
    star_vel = star_vel[ichoose]
    
    
    theta, pos = gen_pos()

    result = Parallel(n_jobs=-1) (delayed(get_midplane_with_error)(p, star_pos, star_vel) for p in tqdm(pos))
    result = np.array(result)
    #for p in pos:
     #   result = get_midplane_with_error(p,star_pos,star_vel)
      #  result = np.array(result)

    midplane_est = result[:,0]

    res = minimize(chisq, np.array([0.1, 0, 0]), args=(theta, midplane_est), method='Nelder-Mead')
    A = res.x[0]
    B = res.x[1]
    C = res.x[2]
    print('A=', A, 'B=', B, 'C=', C)
    fit = A*np.cos(theta + B) + C

    out = np.concatenate((theta.reshape(nspoke, 1), result, fit.reshape(nspoke,1)), axis=1)
    np.save('/home/douglas/Chervin/out_'+gal+'.npy', out)


# In[12]:


glist = ['Disk_200', 'Disk_400', 'Disk_600', 'Disk_690']
for gal in glist:
    main(gal)

