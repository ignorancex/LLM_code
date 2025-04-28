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


__author__ = ["Mathieu Renzo <mrenzo@flatironinstitute.org"]

import sys
import os
import glob


def folder_name(m1, m2, P, Z, grid_index=0, root="/tmp/"):
    """
    given the initial parameters for a binary run returns a string
    that can be used as name for the MESA work directory and is
    compliant with POSYDON's expectations

    mandatory input:
    --------------
    m1 = mass of the primary in Msun
    m2 = mass of the secondary in Msun
    P = initial orbital period in days
    Z = initial metallicity
    optional input:
    --------------
    grid_index = an integer number that can be used to label grids
    output:
    a string
    """
    folder_name = f"/m1_{m1:.4f}_m2_{m2:.4f}_initial_z_{Z:.3}_initial_period_in_days_{P:.4e}_grid_index_{grid_index:d}"
    return root + folder_name


def checkFolder(folder):
    """checks if folder exists, if not, it creates it, and returns its content"""
    found = glob.glob(folder)
    if found:
        print("Found folder:", folder)
        content = glob.glob(folder + "/*")
        return content
    if not found:
        os.system("mkdir -p " + str(folder))
        return glob.glob(folder + "/*")  ## will be empty


def gitPush(repo, description="[skip ci] started a run with no description"):
    push = input("should we push to the git repo first? [Y/n]")
    if (push == "Y") or (push == "y"):
        if not description:
            description = input("commit message:")
        pwd = os.getcwd()  # where am I?
        os.chdir(repo)
        os.system(
            "git add . && git commit -am 'about to start a run: "
            + description
            + " ' && git push"
        )
        os.chdir(pwd)  # go back to previous folder
