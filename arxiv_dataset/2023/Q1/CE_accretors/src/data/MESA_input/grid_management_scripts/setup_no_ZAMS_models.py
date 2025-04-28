#!/usr/local/bin/python

import sys
import os
import glob
import math
import numpy as np

sys.path.append("/mnt/home/mrenzo/codes/python_stuff/plotFunc/")
from utilsLib import *

# setup the grid
MASSES1 = [10, 15, 20, 25]
PERIODS = [5, 10, 12, 15, 20, 100]

# these will all become sys.argv if needed
TEMPLATE = "/mnt/home/mrenzo/Templates/binary_setup/template_binary/"
ROOT = "/mnt/home/mrenzo/ceph/RUNS/mesa_testing_grid/"
DESTINATION_ROOT = "/mnt/home/mrenzo/ceph/RUNS/mesa_testing_grid/"


for Per in PERIODS:
    DESTINATION = DESTINATION_ROOT + "/Period" + str(Per) + "/"
    RUNFILE = ROOT + "/runFile.txt"  # to use with disBatch.py

    print("template:", TEMPLATE)
    print("root:", ROOT)
    print("destination:", DESTINATION)
    go_on = input("should we go on? [Y/n]")
    if (go_on == "Y") or (go_on == "y"):
        content = checkFolder(ROOT)
        if not content:
            go_on_root = "Y"
        else:
            print(str(ROOT), "is not empty")
            print(content)
            go_on_root = input("Go on anyways? [Y/y]")
            if go_on_root != "Y" and go_on_root != "y":
                sys.exit()
            content = checkFolder(DESTINATION)
        if not content:
            go_on_dest = "Y"
        else:
            print(str(DESTINATION), "is not empty")
            print(content)
            go_on_dest = input("Go on anyways? [Y/y]")
            if go_on_dest != "Y" and go_on_dest != "y":
                sys.exit()
        if ((go_on_root == "Y") or (go_on_root == "y")) and (
            (go_on_dest == "Y") or (go_on_dest == "y")
        ):
            # now set up stuff
            description = "binary P" + str(Per) + ", q=0.8, small grid, mesa testing"
            # save template in destination
            os.system(
                "tar -czf template.tar.xz "
                + TEMPLATE
                + " && mv template.tar.xz "
                + DESTINATION
            )
            # now write the RUNFILE
            with open(RUNFILE, "a") as F:
                headerline = "export OMP_NUM_THREADS=7 && export MESA_DIR=/mnt/home/mrenzo/codes/mesa/mesa_testing/mesa_testing && export MESASDK_ROOT=/mnt/home/mrenzo/codes/mesa/mesa_12778/mesasdk && source $MESASDK_ROOT/bin/mesasdk_init.sh"
                backline = (
                    " && ./clean && ./mk && ./rn 2>&1 | tee output" + "\n"
                )  # missing: move to destination
                for M1 in MASSES1:
                    M2 = round(0.8 * M1, 1)
                    folder_name = str(M1) + "_" + str(M2) + "_P" + str(Per) + "/"
                    print("----------")
                    # print(M1, M2)
                    print(folder_name)
                    print("----------")
                    os.system("mkdir -p " + ROOT + "/" + folder_name)
                    os.chdir(ROOT + "/" + folder_name)
                    # # copy stuff
                    copy = "cp -r " + TEMPLATE + "/* ./"
                    os.system(copy)
                    F.writelines(
                        headerline + " && cd " + ROOT + "/" + folder_name + backline
                    )
                    # modify the inlists
                    os.system("perl -pi.back -e 's/MASS1/" + str(M1) + "/g;' inlist1")
                    os.system("perl -pi.back -e 's/MASS2/" + str(M2) + "/g;' inlist2")
                    os.system(
                        "perl -pi.back -e 's/MASS1/" + str(M1) + "/g;' inlist_binary"
                    )
                    os.system(
                        "perl -pi.back -e 's/MASS2/" + str(M2) + "/g;' inlist_binary"
                    )
                    os.system(
                        "perl -pi.back -e 's/PERIOD/" + str(Per) + "/g;' inlist_binary"
                    )
                    os.chdir(ROOT)
