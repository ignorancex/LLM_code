#!/usr/local/bin/python
# Authors:
#          Mathieu Renzo <mrenzo@flatironinstitute.org>
#
# Keywords: files

# Copyright (C) 2021 Mathieu Renzo

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.

__author__ = ["Mathieu Renzo <mrenzo@flatironinstitute.org>"]

import sys
import os

# import glob
# import math
import numpy as np

sys.path.append("/mnt/home/mrenzo/codes/python_stuff/plotFunc/")
from utilsLib import checkFolder, gitPush
from scripts_aux import folder_name

sys.path.append("/mnt/home/mrenzo/Templates/binary_mesa/scripts/notebooks/")
from lib_simplified_profiles import mk_simplified_profile_from_pfile, get_M_boundary
from termcolor import colored


def setup_grid_same_core(mass, init_model=None):
    """
    sets up a grid of "engineered" models of given mass varying the slope
    of the core-envelope boundary. The core-envelope boundary region of
    the "engineered"" models is determined in the following way:
    - inner boundary = same as init_model
    - outer boundary = outer boundary of the init_model +/- a parameter dm

    The grid spans dm=-4..+4 Msun

    Parameters:
    ----------
    mass : `float`, mass of the models
    init_model: `path`, location of the TAMS profile to read in, will be guessed if None
    """
    ROOT = "/mnt/home/mrenzo/Templates/"
    TEMPLATE = ROOT + "binary_mesa/simplified_profile/evolve_to_C_depl/"
    WHERE_TO_RUN = (
        "/mnt/home/mrenzo/ceph/RUNS/accretors/mesa15140/simplified_profiles/same_core/grid"
        + str(mass)
        + "/"
    )
    # check if folders exist
    print("template:", TEMPLATE)
    print("destination:", WHERE_TO_RUN)
    go_on = input("should we go on? [Y/n]")
    if (go_on == "Y") or (go_on == "y"):
        content = checkFolder(WHERE_TO_RUN)
        if content:  # not empty
            print(str(WHERE_TO_RUN), "is not empty")
            print(content)
            go_on = input("Go on anyways? [Y/y]")
            if go_on != "Y" and go_on != "y":
                sys.exit()
    else:
        sys.exit()
    # push to repo
    gitPush("/mnt/home/mrenzo/Templates/binary_mesa/")
    # try to make grid folder
    os.system("mkdir -p " + WHERE_TO_RUN)
    RUNFILE = WHERE_TO_RUN + "grid.txt"
    with open(RUNFILE, "a") as F:
        # write disbatch file header
        header = (
            "#DISBATCH PREFIX export OMP_NUM_THREADS=5 ; export MESA_DIR=/mnt/home/mrenzo/codes/mesa/mesa_15140/mesa15140 ; export MESASDK_ROOT=/mnt/home/mrenzo/codes/mesa/mesa_15140/mesasdk ; source $MESASDK_ROOT/bin/mesasdk_init.sh ;"
            + "\n"
        )
        F.writelines(header)
        # create input file and model
        if init_model == None:
            init_model = (
                "/mnt/home/mrenzo/ceph/RUNS/accretors/mesa15140/simplified_profiles/TAMS_models/"
                + str(mass)
                + "_rot0_to_TAMS/LOGS/TAMS.data"
            )
        # get the CEB boundaries and central value from init_model
        delta_M_boundary, max_M_boundary, min_M_boundary = get_M_boundary(init_model)
        for dm in np.linspace(-4, 4, 21):
            # determine the outer mass coordinate of the CEB of the engineered model
            # from the outer boundary of init_model +/- a parameter dm to shift it
            m_out = max_M_boundary + dm
            if m_out >= min_M_boundary:
                DESTINATION = WHERE_TO_RUN + r"/" + f"{dm:.1f}" + r"/"
                if os.path.isdir(DESTINATION):
                    print(colored(DESTINATION + " exists!", "red"))
                    print(colored("refusing to overwrite", "red"))
                    pass
                else:
                    print(colored(DESTINATION, "green"))
                    os.system("mkdir -p " + DESTINATION)
                    outfile = DESTINATION + r"/" + f"{dm:.2f}" + "_TAMS_profile_input"
                    mk_simplified_profile_from_pfile(
                        init_model,
                        outfile=outfile,
                        plot=False,
                        m_in=min_M_boundary,
                        m_out=m_out,
                    )
                    chem = outfile + "_composition.txt"
                    entropy = outfile + "_entropy.txt"
                    backline = " && ./clean && ./mk && ./rn 2>&1 | tee out.txt"  # +"\n"
                    send_email = (
                        ' ; tail -n 50 out.txt |  mail -s "'
                        + f"{dm:.2f}"
                        + '" -r rusty@flatironinstitute.org -S from="rusty@flatironinstitute.org" mrenzo@flatironinstitute.org'
                    )
                    backline = backline + send_email + "\n"
                    os.chdir(DESTINATION)
                    # # copy stuff
                    copy = "cp -r " + TEMPLATE + "/* ./"
                    os.system(copy)
                    # add model to be run to the RUNFILE
                    F.writelines("cd " + DESTINATION + backline)
                    # modify the inlists
                    os.system(
                        "perl -pi.back -e 's/ENTROPYSTRING/"
                        + str(entropy.split("/")[-1])
                        + "/g;' inlist_simplified_profile"
                    )
                    os.system(
                        "perl -pi.back -e 's/CHEMSTRING/"
                        + str(chem.split("/")[-1])
                        + "/g;' inlist_simplified_profile"
                    )
                    os.system(
                        "perl -pi.back -e 's/MASS/"
                        + str(mass)
                        + "/g;' inlist_simplified_profile"
                    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        setup_grid_same_core(sys.argv[1])
    else:
        raise ValueError("Need a mass, can be 15, 17, 18, 20, 30, 34, 35Msun")
    # for m in [# 15,17,18,
    #           20,30,35]:
    #     setup_grid_same_core(m)
