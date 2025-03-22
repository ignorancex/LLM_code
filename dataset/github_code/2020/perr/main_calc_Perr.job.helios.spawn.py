from __future__ import print_function, division
import os


def main():

    ## OPTIONS
    do_submit = True

    tryid = 'tryQ1b'

    binfile = '/home/mschmittfull/CODE/perr/main_calc_Perr.py_%s' % tryid
    #sim_seeds = range(300,310)
    #sim_seeds = range(303,310)
    #sim_seeds = range(0,3)
    #sim_seeds = range(300,303)
    #sim_seeds = range(401,405)
    sim_seeds = [403]

    #halo_mass_strings = ['13.8_16.0', '12.8_16.0', '11.8_16.0', '10.8_16.0']
    #halo_mass_strings = ['10.8_11.8', '11.8_12.8', '12.8_13.8', '13.8_15.1']
    #halo_mass_strings = ['12.8_16.0', '13.8_16.0']
    halo_mass_strings = ['10.8_11.8']
    #halo_mass_strings = ['11.8_12.8', '12.8_13.8', '13.8_15.1']
    #halo_mass_strings = ['13.8_15.1']

    ## RUN SCRIPT
    send_mail = True
    for halo_mass_string in halo_mass_strings:
        for sim_seed in sim_seeds:
            job_fname = 'main_calc_Perr.job.helios_%s_%s_%d' % (
                tryid, halo_mass_string, sim_seed)
            if send_mail:
                mail_string1 = '#SBATCH --mail-user=mschmittfull@gmail.com'
                mail_string2 = '#SBATCH --mail-type=ALL'
            else:
                mail_string1 = ''
                mail_string2 = ''

            f = open(job_fname, "w")
            f.write("""#!/bin/bash -l

#SBATCH -t 08:00:00
#SBATCH --nodes=1
# #SBATCH --mem=40GB
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -V
%s
%s
#SBATCH --output=slurm-%%x.o%%j
#SBATCH --error=slurm-%%x.e%%j
#SBATCH -J %s_%s_%d
# #SBATCH --dependency=afterok:60705

set -x
export OMP_NUM_THREADS=1
# module load helios

./run.sh python %s "{'sim_seed': %d, 'halo_mass_string': '%s'}"

            """ % (mail_string1, mail_string2, tryid, halo_mass_string,
                   sim_seed, binfile, sim_seed, halo_mass_string))

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
