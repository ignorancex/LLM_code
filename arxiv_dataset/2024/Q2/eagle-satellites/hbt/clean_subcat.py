#!/cosma7/data/durham/dc-sifo1/miniconda3/bin/python
"""Used to produce a cleaned, smaller catalog for Nicole Mejia"""
from astropy.io import fits
from astropy.io.misc import hdf5
import h5py
import numpy as np
import os
import pandas as pd
from time import time

from HBTReader import HBTReader

from hbtpy import hbt_tools
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos


def main():
    print("Running...")
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)

    to = time()
    reader = HBTReader(sim.path)
    print("Loaded reader in {0:.1f} seconds".format(time() - to))
    to = time()
    subs = Subhalos(
        reader.LoadSubhalos(-1),
        sim,
        -1,
        as_dataframe=True,
        load_distances=False,
        load_velocities=False,
    )
    print(
        "Loaded {1} subhalos in {0:.2f} minutes".format(
            (time() - to) / 60, subs.catalog.size
        )
    )
    print()
    print(type(subs.catalog))
    print(np.sort(subs.catalog.columns))

    cat = subs.catalog
    print("cat =", cat.shape)
    # remember masses are in 1e10 Msun/h
    mask = cat["Mbound"] > 0.1
    print("mask =", mask.sum(), cat[mask].size)

    exclude = ["TrackId"]
    cols = cat.columns[~np.isin(cat.columns, exclude)]
    print("cols =", np.sort(cols))

    output = os.path.split(reader.GetFileName(-1))[1].split(".")[0]
    path = "subcat/{0}".format(sim.name)
    if not os.path.isdir(path):
        os.makedirs(path)
    output = os.path.join(path, "{0}_clean.hdf5".format(output))
    cat[cols].to_hdf(output, key="Subhalos", mode="w")
    output = output.replace(".hdf5", ".txt")
    cat[cols].to_csv(output)
    # output = output.replace('.txt', '.fits')
    print("Saved to", output)

    return


main()
