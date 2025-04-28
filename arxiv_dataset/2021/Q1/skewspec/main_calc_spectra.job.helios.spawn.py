from __future__ import print_function, division
import os


def main():

    ## OPTIONS
    do_submit = True

    tryid = 'RunA9'

    binfile = '/home/mschmittfull/CODE/skewspec/main_calc_spectra.py_%s' % tryid
    sim_seeds = range(400,406)
    #sim_seeds = [400]

    apply_RSD_lst = [0,1]
    Rsmooth_lst = [10.0, 20.0]

    # simulation boxsize
    boxsize = 1500.0

    # Ngrid to compute Perr (usually 512 or 1536)
    Ngrid = 512

    # 'catalog' or 'delta_2SPT'
    density_source = 'delta_2SPT'

    ## Options if density_source=='catalog'
    # DM subsample ratio. For L=1500: 0.04, 0.0015
    subsample_ratio_lst = [0.04]
    max_displacement = 100.0


    ## Options if density_source=='delta_2SPT'
    #b1, b2, bG2 = 1.0, 0.0, 0.0 # DM
    #b1, b2, bG2 = 2.0, -0.5, -0.4  # base model for halos
    b1, b2, bG2 = 2.0, -0.5, -0.8
    f_log_growth = 0.786295 # base
    #f_log_growth = 0.943554 # +20%


    # number of nodes to run on
    if Ngrid>1024:
        nodes = 32
    elif Ngrid>512:
        nodes = 12
    else:
        nodes = 1

    srun_cores = nodes * 14

    ## RUN SCRIPT
    send_mail = True
    for sim_seed in sim_seeds:
        for Rsmooth in Rsmooth_lst:
            for apply_RSD in apply_RSD_lst:
                for subsample_ratio in subsample_ratio_lst:

                    job_fname = 'main_calc_spectra.job.helios_%s_%d_%g_%d' % (
                        tryid, sim_seed, Rsmooth, apply_RSD)
                    if send_mail:
                        mail_string1 = '#SBATCH --mail-user=mschmittfull@gmail.com'
                        mail_string2 = '#SBATCH --mail-type=ALL'
                    else:
                        mail_string1 = ''
                        mail_string2 = ''

                    f = open(job_fname, "w")
                    f.write("""#!/bin/bash -l

#SBATCH -t 12:00:00
#SBATCH --nodes=%d
# #SBATCH --mem=40GB
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -V
%s
%s
#SBATCH --output=slurm-%%x.o%%j
#SBATCH --error=slurm-%%x.e%%j
#SBATCH -J %s_%d_%g_%d
# #SBATCH --dependency=afterany:7781387

set -x
export OMP_NUM_THREADS=2
# module load helios


tmp_hdf5_use_file_locking=$HDF5_USE_FILE_LOCKING
export HDF5_USE_FILE_LOCKING=FALSE
. /home/mschmittfull/anaconda2/etc/profile.d/conda.sh
conda activate nbodykit-0.3.7-env


# each helios noise has dual 14-core processors (so 28 cores per node?) and 128GB per node
mpiexec -n %d python %s --SimSeed %d --Ngrid %d --boxsize %g --ApplyRSD %d --Rsmooth %g --SubsampleRatio %g --MaxDisplacement %g --DensitySource %s --b1 %g --b2 %g --bG2 %g --fLogGrowth %g

conda deactivate
export HDF5_USE_FILE_LOCKING=$tmp_hdf5_use_file_locking

                """ % (nodes, mail_string1, mail_string2,
                       tryid, sim_seed, Rsmooth, apply_RSD,
                       srun_cores,
                       binfile, sim_seed, Ngrid, boxsize, apply_RSD, Rsmooth, 
                       subsample_ratio, max_displacement,
                       density_source, b1, b2, bG2, f_log_growth
                       ))

                    f.close()
                    print("Wrote %s" % job_fname)

                    if do_submit:
                        print("Submit %s" % job_fname)
                        os.system("sbatch %s" % job_fname)
                        #print("Sleep...")
                        #os.system("sleep 1")
                    # do not send more than 1 email
                    send_mail = False


if __name__ == '__main__':
    main()
