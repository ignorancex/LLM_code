from __future__ import print_function, division
import os


def main():

    ## OPTIONS
    do_submit = True

    tryid = 'RunB1e'

    binfile = '/home/mschmittfull/CODE/perr/main_calc_RSD_Perr.py_%s' % tryid
    #sim_seeds = range(400,406)
    #sim_seeds = [400,402,403,404,405]
    #sim_seeds = [400,402,403,404]
    sim_seeds = [400]

    SaveGrids4Plots = 1

    ## Select target
    # halos
    #halo_mass_strings = ['13.8_16.0', '12.8_16.0', '11.8_16.0', '10.8_16.0']
    #halo_mass_strings = ['10.8_11.8', '11.8_12.8', '12.8_13.8', '13.8_15.1']
    #halo_mass_strings = ['12.8_16.0', '13.8_16.0']
    #halo_mass_strings = ['10.8_11.8']
    #halo_mass_strings = ['11.8_12.8', '12.8_13.8','13.8_15.1']
    #halo_mass_strings = ['12.8_13.8']

    # hod galaxies
    halo_mass_strings = ['']

    # simulation boxsize
    if False:
        boxsize = 1500.
        shifted_fields_RPsi = 0.69
    else:
        boxsize = 500.
        shifted_fields_RPsi = 0.23

    ## Select shifting options
    #shifted_fields_Np_Nmesh = [(1536,1536)]  # used until 13 March 2020
    #shifted_fields_Np_Nmesh = [(256,256), (512,512), (1536,1536)]
    shifted_fields_Np_Nmesh = [(1536,1536)]

    # Ngrid to compute Perr (usually 512 or 1536)
    Ngrid = 512

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
    for halo_mass_string in halo_mass_strings:
        for sim_seed in sim_seeds:
            for shift_Np, shift_Nmesh in shifted_fields_Np_Nmesh:
                job_fname = 'main_calc_RSD_Perr.job.helios_%s_%s_%d_shift_%d_%d' % (
                    tryid, halo_mass_string, sim_seed, shift_Np, shift_Nmesh)
                if send_mail:
                    mail_string1 = '#SBATCH --mail-user=mschmittfull@gmail.com'
                    mail_string2 = '#SBATCH --mail-type=ALL'
                else:
                    mail_string1 = ''
                    mail_string2 = ''

                f = open(job_fname, "w")
                f.write("""#!/bin/bash -l

#SBATCH -t 48:00:00
#SBATCH --nodes=%d
# #SBATCH --mem=40GB
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -V
%s
%s
#SBATCH --output=slurm-%%x.o%%j
#SBATCH --error=slurm-%%x.e%%j
#SBATCH -J %s_%s_%d_shift_%d_%d
#SBATCH --dependency=afterany:7781387

set -x
export OMP_NUM_THREADS=2
# module load helios


# each helios noise has dual 14-core processors (so 28 cores per node?) and 128GB per node
./run.sh mpiexec -n %d python %s --SimSeed %d --HaloMassString '%s' --ShiftedFieldsNp %d --ShiftedFieldsNmesh %d --Ngrid %d --boxsize %g --ShiftedFieldsRPsi %g --SaveGrids4Plots %d


                """ % (nodes, mail_string1, mail_string2, 
                       tryid, halo_mass_string, sim_seed, shift_Np, shift_Nmesh,
                       srun_cores,
                       binfile, sim_seed, halo_mass_string,
                       shift_Np, shift_Nmesh, Ngrid, boxsize, shifted_fields_RPsi,
                       SaveGrids4Plots
                       ))

                f.close()
                print("Wrote %s" % job_fname)

                if do_submit:
                    print("Submit %s" % job_fname)
                    os.system("sbatch %s" % job_fname)
                    print("Sleep...")
                    os.system("sleep 3")
                # do not send more than 1 email
                send_mail = False


if __name__ == '__main__':
    main()
