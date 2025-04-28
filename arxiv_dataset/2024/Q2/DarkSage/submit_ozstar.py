# submit instances of Dark Sage in embarassingly parallel on OzSTAR

import os
import numpy as np

Ntasks = 212

minfileno = 300
maxfileno = 511
#file_base = 'lhalo-binary.'
file_base = 'MTNG-full-lhalo-binary-linear-nhalos.'
#file_base = 'MTNG-mini-sage-direct-write.'

param_dir = '/home/astevens/DarkSage/input/'
param_filename = 'MTNG.par'
param_file = param_dir + param_filename
#treedir = '/fred/oz245/MTNG/DM-Arepo/MTNG-L500-4320-A/output/treedata/lhalo-binary/lhalo-binary/'
#treedir = '/fred/oz004/msinha/simulations/MTNG/MTNG-mini/lhalo-binary-from-sage/uniform_in_nhalos/'
#treedir = '/fred/oz245/MTNG/DM-Arepo/MTNG-L62.5-540-A/output/treedata/binary-new/'
#treedir = '/fred/oz245/MTNG-full/lhalo-binary-from-sage/split_files/'
treedir = '/fred/oz004/msinha/simulations/MTNG/MTNG-full/lhalo-binary-from-sage/'
#treedir = '/fred/oz004/msinha/simulations/MTNG/MTNG-mini/new/'
batch_file = 'ds_MTNG.sh'

"""
# get list of files
#filenames_full = os.listdir(treedir)
#filenames = np.array(filenames_full)
#Ndel = 0
#for i, fname in enumerate(filenames_full):
#    if fname[:5]!='lhalo-':
#        filenames = np.delete(filenames, i-Ndel)
#        Ndel += 1

Nfiles = maxfileno+1-minfileno
filesizes = np.zeros(Nfiles)
for i in range(minfileno,maxfileno+1):
    fstat = os.stat(treedir+file_base+str(i))
    filesizes[i-minfileno] = fstat.st_size

filesizes_cum = np.cumsum(filesizes)
filesizes_frac = np.append(0, filesizes_cum/filesizes_cum[-1])

frac_to_interp = (np.arange(Ntasks) + 1.0) / float(Ntasks)
end_fileno = np.searchsorted(filesizes_frac, frac_to_interp) - 1
#ssert(len(np.unique(end_fileno)) == len(end_fileno))
start_fileno = np.append(0, end_fileno[:-1]+1)
print(start_fileno)
assert(len(np.unique(start_fileno)) == len(start_fileno))
#assert(end_fileno[-1] == maxfileno)
assert(np.all(end_fileno>=start_fileno))

start_fileno += minfileno
end_fileno += minfileno
"""

fileno_brackets = np.linspace(minfileno, maxfileno+1, Ntasks+1, dtype=np.int32)
start_fileno = fileno_brackets[:-1]
end_fileno = fileno_brackets[1:]-1

#print('start_fileno', start_fileno)
#print('end_fileno', end_fileno)

mem_per_halo = 0.03 # MB

# WARNING: HARD-CODING FILES THAT NEED REDOING HERE
#start_fileno = np.array([145, 155])
#end_fileno = start_fileno
#assert(len(start_fileno) == Ntasks)

for t in range(Ntasks):
    tt = t + minfileno
    tempname0 = param_dir + 'temp_'+str(tt)+'_0.par'
    temp_param_filename = 'temp_'+str(tt)+'_1.par'
    tempname1 = param_dir + temp_param_filename
    os.system('sed \'s/^FirstFile .*/FirstFile                   '+str(start_fileno[t])+'/\' '+param_file+' >'+tempname0)
    os.system('sed \'s/^LastFile .*/LastFile                  '+str(end_fileno[t])+'/\' '+tempname0+' >'+tempname1)
    
    sbatch_temp0 = 'ds_MTNG_temp_'+str(tt)+'_0.sh'
    sbatch_temp1 = 'ds_MTNG_temp_'+str(tt)+'_1.sh'
    sbatch_temp2 = 'ds_MTNG_temp_'+str(tt)+'_2.sh'
    sbatch_temp3 = 'ds_MTNG_temp_'+str(tt)+'_3.sh'

    os.system('sed \'s/'+param_filename+'/'+temp_param_filename+'/\' '+batch_file+' >'+sbatch_temp0)
    os.system('sed \'s/^#SBATCH --job-name.*/#SBATCH --job-name=DarkSage_'+str(tt)+'/\' '+sbatch_temp0+' >'+sbatch_temp1)
    os.system('sed \'s/^#SBATCH --output.*/#SBATCH --output=DS_full_dump_'+str(tt)+'.txt/\' '+sbatch_temp1+' >'+sbatch_temp2)

    # read the first file to figure out how much RAM needs to be requested
    f = open(treedir+file_base+str(start_fileno[t]), 'rb')
    Ntrees = np.fromfile(f, 'i4', 1)[0]
    totNHalos = np.fromfile(f, 'i4', 1)[0]
    TreeNHalos = np.fromfile(f, 'i4', Ntrees)
    f.close()
    mem_request = str(max(2080,int(np.max(TreeNHalos) * mem_per_halo)))
    os.system('sed \'s/^#SBATCH --mem-per-cpu.*/#SBATCH  --mem-per-cpu='+mem_request+'/\' '+sbatch_temp2+' >'+sbatch_temp3)

    os.system('sbatch '+sbatch_temp3)
    os.system('rm '+sbatch_temp0)
    os.system('rm '+sbatch_temp1)
    os.system('rm '+sbatch_temp2)
    os.system('rm '+tempname0)

