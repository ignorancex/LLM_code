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
from scripts_aux import folder_name, checkFolder, gitPush


def get_period_from_masses_sep(a, M1, M2):
    """
    Kepler's 3rd law
    assumes M1, M2 in Msun and a in Rsun
    """
    from MESAreader import G_cgs, Msun, Rsun_cm

    omega = np.sqrt((G_cgs * (M1 + M2) * Msun) / (a * Rsun_cm) ** 3)
    P = 2 * math.pi / omega  # in sec
    Pdays = P / 86400  # in days
    return Pdays


if __name__ == "__main__":
    # define the grid
    primary_masses = [20]  # , 20, 38]
    mass_ratios = []  # 0.4, 0.5, 0.6, 0.7, 0.9]
    secondary_masses = [17]
    periods = [100]
    # separations = [300] # Rsun
    metallicities = [0.0019]  # , 0.00019]
    ST_efficiencies = [1]

    # these will all become sys.argv if needed
    ROOT = "../../MESA_input/"
    TEMPLATE = ROOT + "binaries/template_binary/"
    WHERE_TO_RUN = "/mnt/home/mrenzo/ceph/RUNS/accretors/mesa15140/binaries/no_rot/"

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

    RUNFILE = WHERE_TO_RUN + "/grid_of_accretors.txt"

    # push to repo
    gitPush("/mnt/home/mrenzo/Projects/CE_accretors_base/CE_accretors/")

    # set up the grid now
    for Z in metallicities:
        WHERE_TO_RUN_Z = WHERE_TO_RUN + "Z_" + str(Z) + "/"
        os.system("mkdir -p " + WHERE_TO_RUN_Z)
        RUNFILE = WHERE_TO_RUN_Z + "/add_25_17_100.txt"
        # save setup befpre setting up grid_index
        os.system(
            "tar -czf template.tar.xz "
            + TEMPLATE
            + " && mv template.tar.xz "
            + WHERE_TO_RUN_Z
        )
        with open(RUNFILE, "a") as F:
            # write disbatch file header
            header = (
                "#DISBATCH PREFIX export OMP_NUM_THREADS=10 ; export MESA_DIR=/mnt/home/mrenzo/codes/mesa/mesa_15140/mesa15140 ; export MESASDK_ROOT=/mnt/home/mrenzo/codes/mesa/mesa_15140/mesasdk ; source $MESASDK_ROOT/bin/mesasdk_init.sh ;"
                + "\n"
            )
            F.writelines(header)
            for m1 in primary_masses:
                # print(m1)
                if mass_ratios:
                    for q in mass_ratios:
                        m2 = round(q * m1, 1)
                        for p in periods:
                            for st_eff in ST_efficiencies:
                                # for a in separations:
                                # p = get_period_from_masses_sep(a, m1, m2)
                                # print(p)
                                DESTINATION = folder_name(
                                    m1, m2, p, Z=Z, root=WHERE_TO_RUN_Z
                                )
                                DESTINATION = DESTINATION + "_" + str(st_eff) + "/"
                                # print(DESTINATION)
                                # now set up stuff
                                backline = " && ./clean && ./mk && ./rn 2>&1 | tee out.txt"  # +"\n"
                                convert_history1 = (
                                    " ; python /mnt/home/mrenzo/codes/python_stuff/plotFunc/convert_to_npy.py "
                                    + DESTINATION
                                    + "/LOGS1/history.data"
                                )
                                convert_history2 = (
                                    " ; python /mnt/home/mrenzo/codes/python_stuff/plotFunc/convert_to_npy.py "
                                    + DESTINATION
                                    + "/LOGS2/history.data"
                                )
                                convert_bin_history = (
                                    " ; python /mnt/home/mrenzo/codes/python_stuff/plotFunc/convert_to_npy.py "
                                    + DESTINATION
                                    + "/binary_history.data"
                                )
                                ### leave root empty in line belowfor relative folder name
                                send_email = (
                                    ' ; tail -n 50 out.txt |  mail -s "'
                                    + folder_name(m1, m2, p, Z=Z, root="")
                                    + '" -r rusty@flatironinstitute.org -S from="rusty@flatironinstitute.org" mrenzo@flatironinstitute.org'
                                )
                                backline = (
                                    backline
                                    + convert_history1
                                    + convert_history2
                                    + convert_bin_history
                                    + send_email
                                    + "\n"
                                )
                                # backline = backline+send_email+"\n"
                                os.system("mkdir -p " + DESTINATION)
                                os.chdir(DESTINATION)
                                # # copy stuff
                                copy = "cp -r " + TEMPLATE + "/* ./"
                                os.system(copy)
                                F.writelines("cd " + DESTINATION + backline)
                                # modify the inlists
                                os.system(
                                    "perl -pi.back -e 's/MASS1/"
                                    + str(m1)
                                    + "/g;' inlist1"
                                )
                                os.system(
                                    "perl -pi.back -e 's/MASS2/"
                                    + str(m2)
                                    + "/g;' inlist2"
                                )
                                os.system(
                                    "perl -pi.back -e 's/MASS1/"
                                    + str(m1)
                                    + "/g;' inlist_binary"
                                )
                                os.system(
                                    "perl -pi.back -e 's/MASS2/"
                                    + str(m2)
                                    + "/g;' inlist_binary"
                                )
                                os.system(
                                    "perl -pi.back -e 's/PERIOD/"
                                    + str(p)
                                    + "/g;' inlist_binary"
                                )
                                os.system(
                                    "perl -pi.back -e 's/METALLICITY/"
                                    + str(Z)
                                    + "/g;' inlist_both"
                                )
                                os.system(
                                    "perl -pi.back -e 's/METALLICITY/"
                                    + str(Z)
                                    + "/g;' inlist_extra"
                                )
                                # ST efficiency
                                os.system(
                                    "perl -pi.back -e 's/ST_EFFICIENCY/"
                                    + str(st_eff)
                                    + "/g;' inlist_both"
                                )
                                # resolution
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
                else:
                    m2 = secondary_masses[primary_masses.index(m1)]
                    for p in periods:
                        for st_eff in ST_efficiencies:
                            # for a in separations:
                            # p = get_period_from_masses_sep(a, m1, m2)
                            # print(p)
                            DESTINATION = folder_name(
                                m1, m2, p, Z=Z, root=WHERE_TO_RUN_Z
                            )
                            DESTINATION = DESTINATION + "_" + str(st_eff) + "/"
                            # print(DESTINATION)
                            # now set up stuff
                            backline = " && ./clean && ./mk && ./rn 2>&1 | tee out.txt"  # +"\n"
                            convert_history1 = (
                                " ; python /mnt/home/mrenzo/codes/python_stuff/plotFunc/convert_to_npy.py "
                                + DESTINATION
                                + "/LOGS1/history.data"
                            )
                            convert_history2 = (
                                " ; python /mnt/home/mrenzo/codes/python_stuff/plotFunc/convert_to_npy.py "
                                + DESTINATION
                                + "/LOGS2/history.data"
                            )
                            convert_bin_history = (
                                " ; python /mnt/home/mrenzo/codes/python_stuff/plotFunc/convert_to_npy.py "
                                + DESTINATION
                                + "/binary_history.data"
                            )
                            ### leave root empty in line belowfor relative folder name
                            send_email = (
                                ' ; tail -n 50 out.txt |  mail -s "'
                                + folder_name(m1, m2, p, Z=Z, root="")
                                + '" -r rusty@flatironinstitute.org -S from="rusty@flatironinstitute.org" mrenzo@flatironinstitute.org'
                            )
                            backline = (
                                backline
                                + convert_history1
                                + convert_history2
                                + convert_bin_history
                                + send_email
                                + "\n"
                            )
                            # backline = backline+send_email+"\n"
                            os.system("mkdir -p " + DESTINATION)
                            os.chdir(DESTINATION)
                            # # copy stuff
                            copy = "cp -r " + TEMPLATE + "/* ./"
                            os.system(copy)
                            F.writelines("cd " + DESTINATION + backline)
                            # modify the inlists
                            os.system(
                                "perl -pi.back -e 's/MASS1/" + str(m1) + "/g;' inlist1"
                            )
                            os.system(
                                "perl -pi.back -e 's/MASS2/" + str(m2) + "/g;' inlist2"
                            )
                            os.system(
                                "perl -pi.back -e 's/MASS1/"
                                + str(m1)
                                + "/g;' inlist_binary"
                            )
                            os.system(
                                "perl -pi.back -e 's/MASS2/"
                                + str(m2)
                                + "/g;' inlist_binary"
                            )
                            os.system(
                                "perl -pi.back -e 's/PERIOD/"
                                + str(p)
                                + "/g;' inlist_binary"
                            )
                            os.system(
                                "perl -pi.back -e 's/METALLICITY/"
                                + str(Z)
                                + "/g;' inlist_both"
                            )
                            os.system(
                                "perl -pi.back -e 's/METALLICITY/"
                                + str(Z)
                                + "/g;' inlist_extra"
                            )
                            # ST efficiency
                            os.system(
                                "perl -pi.back -e 's/ST_EFFICIENCY/"
                                + str(st_eff)
                                + "/g;' inlist_both"
                            )
                            # resolution
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
