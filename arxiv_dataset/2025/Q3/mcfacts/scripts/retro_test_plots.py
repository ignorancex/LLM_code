#!/usr/bin/env python3

######## Imports ########
import matplotlib.pyplot as plt
import numpy as np
import mcfacts.vis.LISA as li
import mcfacts.vis.PhenomA as pa
import pandas as pd
import os
# Grab those txt files
from importlib import resources as impresources
from mcfacts.vis import data


######## Arg ########
#def arg():
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--fname-retros1",
#        default="/run000/output_bh_single_retro_0.dat",
#        type=str, help="output_retro file")
#    parser.add_argument("--fname-retros2",
#        default="/run000/output_bh_single_retro_999.dat",
#        type=str, help="output_retro file")
    #parser.add_argument("--fname-mergers",
    #    default="output_mergers_population.dat",
    #    type=str, help="output_mergers file")
    #parser.add_argument("--fname-lvk",
    #    default="output_mergers_lvk.dat",
    #    type=str, help="output_lvk file")
#    opts = parser.parse_args()
#    assert os.path.isfile(opts.fname_retros1)
#    assert os.path.isfile(opts.fname_retros2)
    #assert os.path.isfile(opts.fname_emris)
    #assert os.path.isfile(opts.fname_lvk)
#    return opts

######## Main ########
def main():
    # Saavik wrote this on the fly with lots of hardcoded bs to just get a quick look

    # need section for loading data
    #opts = arg()
    retros1 = np.loadtxt("../run000/output_bh_single_retro_0.dat", skiprows=2)
    retros2 = np.loadtxt("../run000/output_bh_single_retro_99.dat", skiprows=2)

    pros1 = np.loadtxt("../run000/output_bh_single_0.dat", skiprows=2)

    run_total = 100
    sma_accum = retros1[:,0]
    inc_accum = retros1[:,5]
    pro_sma_accum = pros1[:,0]
    pro_inc_accum = pros1[:,5]
    for iteration in range(run_total):
        iteration_zfilled_str = f"{iteration:>0{int(np.log10(run_total))+1}}"
        readname = f"../run{iteration_zfilled_str}/output_bh_single_retro_0.dat"
        temp = np.loadtxt(readname, skiprows=2)
        temp_sma = temp[:,0]
        temp_inc = temp[:,5]
        sma_accum = np.append(sma_accum,temp_sma)
        inc_accum = np.append(inc_accum,temp_inc)
        readname2 = f"../run{iteration_zfilled_str}/output_bh_single_0.dat"
        temp2 = np.loadtxt(readname2, skiprows=2)
        temp2_sma = temp2[:,0]
        temp2_inc = temp2[:,5]
        pro_sma_accum = np.append(pro_sma_accum,temp2_sma)
        pro_inc_accum = np.append(pro_inc_accum,temp2_inc)

    delta_sma = retros1[:,0] - retros2[:,0]
    delta_ecc = retros2[:,4] - retros1[:,4]
    delta_inc = retros1[:,5] - retros2[:,5]

    plt.figure()
    plt.scatter(retros1[:,0], delta_sma, color='teal')
    plt.ylabel(r'Delta Semi-Major Axis ($R_g$)')
    plt.xlabel(r'Initial Radius ($R_g$)')
    plt.xscale('log')
    #plt.legend(frameon=False)
    #plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./retro_delta_sma.png", format='png')
    plt.close()

    plt.figure()
    plt.scatter(retros1[:,4], delta_ecc, color='teal')
    plt.ylabel(r'Delta Eccentricity')
    plt.xlabel(r'Initial Eccentricity')
    #plt.xscale('log')
    #plt.legend(frameon=False)
    #plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./retro_delta_ecc.png", format='png')
    plt.close()

    plt.figure()
    plt.scatter(retros1[:,5], delta_inc, color='teal')
    plt.ylabel(r'Delta Inclination (rad)')
    plt.xlabel(r'Initial Inclination (rad)')
    #plt.xscale('log')
    #plt.legend(frameon=False)
    #plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./retro_delta_inc.png", format='png')
    plt.close()

    plt.figure()
    plt.scatter(retros1[:,5], delta_ecc, color='teal')
    plt.axvline(np.pi)
    plt.ylabel(r'Delta Eccentricity')
    plt.xlabel(r'Initial Inclination (rad)')
    #plt.xscale('log')
    #plt.legend(frameon=False)
    #plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./retro_delta_ecc_vs_inc.png", format='png')
    plt.close()

    plt.figure()
    plt.scatter(sma_accum, inc_accum, color='teal')
    plt.axhline(np.pi)
    plt.ylabel(r'Initial Inclination (rad)')
    plt.xlabel(r'Initial Semi-Major Axis ($R_g$)')
    plt.xscale('log')
    #plt.legend(frameon=False)
    #plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./retro_inc_vs_sma.png", format='png')
    plt.close()

    plt.figure()
    plt.scatter(pro_sma_accum, pro_inc_accum, color='teal')
    plt.axhline(0.0)
    plt.ylabel(r'Initial Inclination (rad)')
    plt.xlabel(r'Initial Semi-Major Axis ($R_g$)')
    plt.xscale('log')
    #plt.legend(frameon=False)
    #plt.ylim(10,1000)

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, color='gray', ls='dashed')
    plt.tight_layout()
    plt.savefig("./pro_inc_vs_sma.png", format='png')
    plt.close()


######## Execution ########
if __name__ == "__main__":
    main()