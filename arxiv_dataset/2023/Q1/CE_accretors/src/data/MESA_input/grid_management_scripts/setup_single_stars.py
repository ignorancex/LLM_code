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
import glob
import math
import numpy as np

sys.path.append("/mnt/home/mrenzo/codes/python_stuff/plotFunc/")
sys.path.append("/mnt/home/mrenzo/Templates/binary_setup/notebooks/")
# sys.path.append("/home/math/Documents/Research/codes/plotFunc/")
# sys.path.append("/home/math/Documents/Research/Projects/binary_setup/notebooks/")
from utilsLib import checkFolder, gitPush
from scripts_aux import folder_name

# define the grid
primary_masses = [17]  # [15, 20, 30, 40, 50] # np.linspace(8, 100, 93)
metallicities = [0.0019]  # , 0.00019]
omegas = [0.0]  # , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# print(primary_masses)

# these will all become sys.argv if needed
ROOT = "/mnt/home/mrenzo/Templates/"
TEMPLATE = ROOT + "binary_mesa/template_single/"
WHERE_TO_RUN = (
    "/mnt/home/mrenzo/ceph/RUNS/accretors/mesa15140/single_stars/D_mix1d-2/radii/"
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
gitPush("/mnt/home/mrenzo/Templates/binary_mesa")


# set up the grid now
for Z in metallicities:
    WHERE_TO_RUN_Z = WHERE_TO_RUN + "Z_" + str(Z) + "/"
    os.system("mkdir -p " + WHERE_TO_RUN_Z)
    RUNFILE = WHERE_TO_RUN_Z + "/grid_single_rotating.txt"
    # save setup before setting up grid_index
    os.system(
        "tar -czf template.tar.xz "
        + TEMPLATE
        + " && mv template.tar.xz "
        + WHERE_TO_RUN_Z
    )
    # print(WHERE_TO_RUN_Z)
    with open(RUNFILE, "a") as F:
        # write disbatch file header
        header = (
            "#DISBATCH PREFIX export OMP_NUM_THREADS=5 ; export MESA_DIR=/mnt/home/mrenzo/codes/mesa/mesa_15140/mesa15140 ; export MESASDK_ROOT=/mnt/home/mrenzo/codes/mesa/mesa_15140/mesasdk ; source $MESASDK_ROOT/bin/mesasdk_init.sh ;"
            + "\n"
        )
        F.writelines(header)
        for m1 in primary_masses:
            # print(m1)
            for o in omegas:
                DESTINATION = WHERE_TO_RUN_Z + "/" + str(m1) + "_rot" + str(o) + "/"
                # now set up stuff
                backline = " && ./clean && ./mk && ./rn 2>&1 | tee out.txt"  # +"\n"
                send_email = (
                    ' ; tail -n 50 out.txt |  mail -s "'
                    + str(m1)
                    + "_"
                    + str(Z)
                    + "_"
                    + str(o)
                    + '" -r rusty@flatironinstitute.org -S from="rusty@flatironinstitute.org" mrenzo@flatironinstitute.org'
                )
                backline = backline + send_email + "\n"
                os.system("mkdir -p " + DESTINATION)
                os.chdir(DESTINATION)
                # # copy stuff
                copy = "cp -r " + TEMPLATE + "/* ./"
                os.system(copy)
                F.writelines("cd " + DESTINATION + backline)
                # modify the inlists
                os.system("perl -pi.back -e 's/MASS/" + str(m1) + "/g;' inlist_extra")
                os.system(
                    "perl -pi.back -e 's/METALLICITY/" + str(Z) + "/g;' inlist_extra"
                )
                os.system(
                    "perl -pi.back -e 's/OMEGA_DIV_OMEGA_CRIT/"
                    + str(o)
                    + "/g;' inlist_extra"
                )
                os.system(
                    "perl -pi.back -e 's/MESH_DELTA_COEFF/"
                    + str(0.5)
                    + "/g;' inlist_extra"
                )
                os.system(
                    "perl -pi.back -e 's/MESH_TIME_COEFF/"
                    + str(0.7)
                    + "/g;' inlist_extra"
                )
