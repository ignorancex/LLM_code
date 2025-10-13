#!/usr/bin/env python3
"""
Compute and save radial density, temperature, and entropy profiles
for a series of simulation snapshots, given Velociraptor and SWIFT outputs.
"""
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import unyt
from velociraptor import load as load_catalogue, VelociraptorCatalogue
from swiftsimio import load as load_snapshot_data, SWIFTDataset
from scipy.stats import binned_statistic
from warnings import filterwarnings, warn
from tqdm import tqdm

# Suppress specific numpy/runtime warnings
filterwarnings("ignore", category=RuntimeWarning, message="Mixing ufunc arguments")

# Physical constants and defaults
MEAN_MOLECULAR_WEIGHT = 0.5954  # primordial ionized gas
MEAN_WEIGHT_PER_ELECTRON = 1.14  # atomic weight per free electron
TEMP_HOT = 1e5 * unyt.K  # temperature threshold for "hot" gas
DEFAULT_NUM_SNAPSHOTS = 200


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute radial profiles from simulation snapshots"
    )
    parser.add_argument(
        "--snapshots-dir", type=Path, default=Path(".."),
        help="Root directory containing 'snapshots' and 'stf' subfolders"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Directory to save output .npy files"
    )
    parser.add_argument(
        "--num-snapshots", type=int, default=DEFAULT_NUM_SNAPSHOTS,
        help="Total number of snapshots to attempt loading"
    )
    parser.add_argument(
        "--rmin", type=float, default=0.005,
        help="Minimum radius for bins (in R500 units)"
    )
    parser.add_argument(
        "--rmax", type=float, default=2.5,
        help="Maximum radius for bins (in R500 units)"
    )
    parser.add_argument(
        "--nbins", type=int, default=30,
        help="Number of radial bins"
    )
    return parser.parse_args()


def write_array(output_dir: Path, name: str, array: np.ndarray) -> None:
    filepath = output_dir / f"{name}.npy"
    np.save(filepath, array)
    logging.info(f"Saved array '{name}' to {filepath}")


def radial_bin_edges(rmin: float, rmax: float, nbins: int) -> np.ndarray:
    """Compute log-spaced radial bin edges in R500 units."""
    return np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)


def radial_bin_centers(rmin: float, rmax: float, nbins: int) -> np.ndarray:
    """Compute radial bin centers from edges."""
    edges_log = np.log10(radial_bin_edges(rmin, rmax, nbins))
    centers = 10 ** ((edges_log[:-1] + edges_log[1:]) / 2)
    return centers


def radial_bin_volumes(rmin: float, rmax: float, nbins: int) -> np.ndarray:
    """Compute shell volumes for each radial bin (in R500 units^3)."""
    edges = radial_bin_edges(rmin, rmax, nbins)
    return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def load_snapshot(snapshot_dir: Path, index: int):
    """
    Load a single snapshot and its Velociraptor catalogue.
    """
    snapfile = snapshot_dir / 'snapshots' / f"snap_{index:04d}.hdf5"
    catfile = snapshot_dir / 'stf' / f"snap_{index:04d}" / f"snap_{index:04d}.properties"
    ds: SWIFTDataset = load_snapshot_data(snapfile)
    cat: VelociraptorCatalogue = load_catalogue(catfile)
    return ds, cat


def compute_scaled_radius(ds: SWIFTDataset, cat: VelociraptorCatalogue) -> np.ndarray:
    """Compute each gas cell's radius scaled to R500."""
    r500 = cat.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
    center = np.array([
        cat.positions.xcmbp[0],
        cat.positions.ycmbp[0],
        cat.positions.zcmbp[0]
    ]) / cat.units.a
    coords = ds.gas.coordinates.to('Mpc') - center
    radii = np.linalg.norm(coords, axis=1)  # Mpc
    return (radii / r500).to('')


def density_profile(
        ds: SWIFTDataset,
        cat: VelociraptorCatalogue,
        rmin: float,
        rmax: float,
        nbins: int,
) -> unyt.unyt_array:
    """Compute hot-gas density profile [Msun / (Mpc^3)]."""
    scaled_r = compute_scaled_radius(ds, cat)
    masses = ds.gas.masses.to('Msun')
    temps = ds.gas.temperatures.to('K')
    # select hot gas
    mask = temps > TEMP_HOT
    mass_sum, _ = binned_statistic(
        scaled_r[mask], masses[mask], bins=radial_bin_edges(rmin, rmax, nbins), statistic='sum'
    )
    volumes = radial_bin_volumes(rmin, rmax, nbins)
    density = mass_sum / volumes  # Msun / Mpc^3
    return unyt.unyt_array(density, 'Msun/Mpc**3')


def temperature_profile(
        ds: SWIFTDataset,
        cat: VelociraptorCatalogue,
        rmin: float,
        rmax: float,
        nbins: int,
) -> unyt.unyt_array:
    """Compute mass-weighted hot-gas temperature profile [K]."""
    scaled_r = compute_scaled_radius(ds, cat)
    masses = ds.gas.masses.to('Msun')
    temps = ds.gas.temperatures.to('K')
    mask = temps > TEMP_HOT
    # mass-weighted sum of temperature
    mt_sum, _ = binned_statistic(
        scaled_r[mask], masses[mask] * temps[mask], bins=radial_bin_edges(rmin, rmax, nbins), statistic='sum'
    )
    mass_sum, _ = binned_statistic(
        scaled_r[mask], masses[mask], bins=radial_bin_edges(rmin, rmax, nbins), statistic='sum'
    )
    temp_prof = mt_sum / mass_sum
    return unyt.unyt_array(temp_prof, 'K')


def entropy_profile(
        density: unyt.unyt_array,
        temperature: unyt.unyt_array,
) -> unyt.unyt_array:
    """Compute entropy profile K = k_B T / n_e^{2/3} [keV cm^2]."""
    # convert density to electron number density
    ne = (density / (MEAN_WEIGHT_PER_ELECTRON * unyt.mp)).to('1/cm**3')
    # temperature in keV
    t_keV = (temperature * unyt.kb).to('keV')
    k = t_keV / (ne ** (2.0 / 3.0))
    return k.to('keV*cm**2')


def main():
    logger = setup_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snapshots: List[tuple] = []
    logger.info("Loading snapshots...")
    for i in tqdm(range(args.num_snapshots), desc="Snapshots"):
        try:
            ds, cat = load_snapshot(args.snapshots_dir, i)
            snapshots.append((ds, cat))
        except FileNotFoundError:
            break
    n_loaded = len(snapshots)
    if n_loaded < args.num_snapshots:
        warn(f"Only loaded {n_loaded}/{args.num_snapshots} snapshots.")

    # bin definitions
    centers = radial_bin_centers(args.rmin, args.rmax, args.nbins)
    write_array(args.output_dir, 'radial_bin_centers', centers)

    # compute and save profiles
    density_all = []
    temp_all = []
    ent_all = []
    for ds, cat in tqdm(snapshots, desc="Computing profiles"):
        dens = density_profile(ds, cat, args.rmin, args.rmax, args.nbins)
        temp = temperature_profile(ds, cat, args.rmin, args.rmax, args.nbins)
        ent = entropy_profile(dens, temp)
        density_all.append(dens.value)
        temp_all.append(temp.value)
        ent_all.append(ent.value)

    write_array(args.output_dir, 'density_profiles', np.array(density_all))
    write_array(args.output_dir, 'temperature_profiles', np.array(temp_all))
    write_array(args.output_dir, 'entropy_profiles', np.array(ent_all))


if __name__ == '__main__':
    main()
