#!/usr/bin/env python3
"""
Mask gas particles by temperature and radial shell, then save particle IDs.

Loads a SWIFT snapshot and Velociraptor catalogue, computes R500 and halo center,
then applies a series of ParticleMask definitions to select gas particles
in hot/cold phases and radial shells, printing summary stats and saving IDs.
"""
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import unyt
from sciris import printgreen
from swiftsimio import load as load_snapshot, SWIFTDataset
from velociraptor import load as load_catalogue, VelociraptorCatalogue

# ---------------------------------------------------------------------------- #
# Default configuration
# ---------------------------------------------------------------------------- #
DEFAULT_SNAP_DIR = Path("..") / "snapshots"
DEFAULT_CAT_DIR = Path("..") / "stf"
DEFAULT_CENTER_FILE = Path("final_center_0199.npy")
DEFAULT_OUTPUT_DIR = Path(".")
TEMP_HOT_THRESHOLD = 1e5 * unyt.K
RADIAL_SCALES = [0.15, 1.0, 6.0]  # in units of r500


@dataclass
class Snapshot:
    """Holds paths and data for a single SWIFT snapshot."""

    number: int
    snap_file: Path
    cat_file: Path
    data: SWIFTDataset
    catalogue: VelociraptorCatalogue


@dataclass
class ParticleMask:
    """Defines a temperature and radial shell mask for gas particles."""

    name: str
    phase: str  # 'hot' or 'cold'
    r_min: float  # inner radius in units of R500
    r_max: float  # outer radius in units of R500


# ---------------------------------------------------------------------------- #
# Logging setup
# ---------------------------------------------------------------------------- #
def setup_logger(verbose: bool) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Core functionality
# ---------------------------------------------------------------------------- #


def load_snapshot_data(number: int, snap_dir: Path, cat_dir: Path) -> Snapshot:
    """Load SWIFT snapshot HDF5 and Velociraptor catalogue for given snapshot number."""
    snap_file = snap_dir / f"snap_{number:04d}.hdf5"
    cat_file = cat_dir / f"snap_{number:04d}" / f"snap_{number:04d}.properties"
    data = load_snapshot(snap_file)
    catalogue = load_catalogue(cat_file)
    return Snapshot(number, snap_file, cat_file, data, catalogue)


def compute_r500(catalogue: VelociraptorCatalogue) -> unyt.unyt_quantity:
    """Compute R500 (comoving) in Mpc."""
    return (
        catalogue.spherical_overdensities.r_500_rhocrit[0].to("Mpc") / catalogue.units.a
    )


def compute_halo_center(catalogue: VelociraptorCatalogue) -> unyt.unyt_array:
    """Compute 3D halo centre (comoving) in Mpc."""
    a = catalogue.units.a
    pos = catalogue.positions
    coords = np.array([pos.xcmbp[0], pos.ycmbp[0], pos.zcmbp[0]]) / a
    return unyt.unyt_array(coords * unyt.Mpc, "Mpc")


def load_center(center_file: Path) -> unyt.unyt_array:
    """Load precomputed halo centre from .npy file."""
    arr = np.load(center_file)
    return unyt.unyt_array(arr, "Mpc")


def compute_scaled_radial_distances(
    coordinates: np.ndarray, center: unyt.unyt_array, r500: unyt.unyt_quantity
) -> np.ndarray:
    """Return array of radial distances in units of R500."""
    disp = coordinates - center
    radii = np.linalg.norm(disp, axis=1) * center.units
    radii_mpc = radii.to("Mpc")
    return (radii_mpc / r500).d


def apply_mask(
    snap: SWIFTDataset, scaled_r: np.ndarray, mask_def: ParticleMask
) -> np.ndarray:
    """Return indices of gas particles matching phase and radial shell."""
    temp = snap.gas.temperatures
    if mask_def.phase == "hot":
        phase_mask = temp > TEMP_HOT_THRESHOLD
    else:
        phase_mask = temp <= TEMP_HOT_THRESHOLD

    radial_mask = (scaled_r >= mask_def.r_min) & (scaled_r < mask_def.r_max)
    return np.where(phase_mask & radial_mask)[0]


def save_particle_ids(
    ids: np.ndarray, mask_def: ParticleMask, snapshot_num: int, outdir: Path
) -> None:
    """Save selected particle IDs to .npy file with descriptive name."""
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"gas_ids_{mask_def.name}_{mask_def.phase}_{snapshot_num:04d}.npy"
    path = outdir / fname
    np.save(path, ids)
    logging.info("Saved %d IDs to %s", ids.size, path)


# ---------------------------------------------------------------------------- #
# Main workflow
# ---------------------------------------------------------------------------- #


def main(
    snapshot_num: int,
    snap_dir: Path,
    cat_dir: Path,
    center_file: Path,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Execute masking pipeline for a single snapshot."""
    setup_logger(verbose)
    printgreen(f"Processing snapshot {snapshot_num:04d}...")

    # Load data and compute geometry
    snap_obj = load_snapshot_data(snapshot_num, snap_dir, cat_dir)
    r500 = compute_r500(snap_obj.catalogue)
    center = load_center(center_file)

    logging.info("R500 = %.3f Mpc; centre = %s", r500, center)

    # Precompute scaled radii
    coords = snap_obj.data.gas.coordinates
    scaled_r = compute_scaled_radial_distances(coords, center, r500)

    # Define masks: (name, phase, r_min, r_max)
    masks: List[ParticleMask] = []
    for phase in ("hot", "cold"):
        for i in range(len(RADIAL_SCALES) - 1):
            masks.append(
                ParticleMask(
                    name=f"shell{i}",
                    phase=phase,
                    r_min=RADIAL_SCALES[i],
                    r_max=RADIAL_SCALES[i + 1],
                )
            )

    # Apply each mask, print stats, and save
    for mdef in masks:
        idx = apply_mask(snap_obj.data, scaled_r, mdef)
        med_r = np.median(scaled_r[idx]) if idx.size > 0 else np.nan
        med_t = (
            np.median(snap_obj.data.gas.temperatures[idx]) if idx.size > 0 else np.nan
        )

        print(f"\nMask: {mdef.name}, phase={mdef.phase}")
        print(
            f"Count = {idx.size:d}, Median r/R500 = {med_r:.3f}, Median T = {med_t:.3E}"
        )

        save_particle_ids(idx, mdef, snapshot_num, output_dir)


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask gas particles by temperature and radius and save IDs."
    )
    parser.add_argument(
        "--snapshot", "-s", type=int, required=True, help="Snapshot number (e.g. 199)"
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=DEFAULT_SNAP_DIR,
        help="Directory for snapshot HDF5 files",
    )
    parser.add_argument(
        "--cat-dir",
        type=Path,
        default=DEFAULT_CAT_DIR,
        help="Directory for Velociraptor catalogues",
    )
    parser.add_argument(
        "--center-file",
        type=Path,
        default=DEFAULT_CENTER_FILE,
        help=".npy file with halo centre coordinates",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for particle ID files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    main(
        snapshot_num=args.snapshot,
        snap_dir=args.snap_dir,
        cat_dir=args.cat_dir,
        center_file=args.center_file,
        output_dir=args.out_dir,
        verbose=args.verbose,
    )
