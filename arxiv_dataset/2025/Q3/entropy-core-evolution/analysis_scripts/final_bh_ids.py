#!/usr/bin/env python3
"""
Extract the most massive central black hole properties from a single SWIFT snapshot.

This script loads a Velociraptor catalogue and SWIFT snapshot,
computes R500 and the 3D center, identifies the most massive black hole
within R500, and writes out its ID and mass.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import unyt
from sciris import printgreen
from swiftsimio import load as load_snapshot, SWIFTDataset
from velociraptor import load as load_catalogue, VelociraptorCatalogue

# ---------------------------------------------------------------------------- #
# Defaults
# ---------------------------------------------------------------------------- #
DEFAULT_SNAP_DIR = Path("..") / "snapshots"
DEFAULT_CAT_DIR = Path("..") / "stf"
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_FORMAT = ".npy"


# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #
def setup_logging(level=logging.INFO):
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Core Functions
# ---------------------------------------------------------------------------- #


def compute_r500(catalogue: VelociraptorCatalogue) -> unyt.unyt_quantity:
    """Return R500 radius (enclosing 500×critical density) in comoving Mpc."""
    return (
        catalogue.spherical_overdensities.r_500_rhocrit[0].to("Mpc") / catalogue.units.a
    )


def compute_center(catalogue: VelociraptorCatalogue) -> unyt.unyt_array:
    """Return comoving center coordinates of the main halo in Mpc."""
    a = catalogue.units.a
    pos = catalogue.positions
    coords = unyt.unyt_array([pos.xcmbp[0], pos.ycmbp[0], pos.zcmbp[0]] / a, "Mpc")
    return coords


def find_most_massive_bh(
    snap: SWIFTDataset, center: unyt.unyt_array, r500: unyt.unyt_quantity
) -> tuple[int, unyt.unyt_quantity, unyt.unyt_quantity]:
    """
    Identify the particle ID, mass, and radius of the most massive BH within R500.
    Returns (particle_id, mass, radius).
    """
    # Compute distances
    coords = snap.black_holes.coordinates
    disp = coords - center
    dist = np.linalg.norm(disp, axis=1) * center.units
    dist = dist.to("Mpc")

    # Select BHs inside R500
    inside = dist <= r500
    if not inside.any():
        logging.warning("No black holes found within R500 = %.3f Mpc", r500)
        return (0, unyt.unyt_quantity(0, "Msun"), unyt.unyt_quantity(0, "Mpc"))

    # Ensure physical masses
    masses = snap.black_holes.subgrid_masses.to_physical()
    # Find index of max mass
    idx = np.argmax(masses[inside])
    # Map back to original array indices
    bh_indices = np.where(inside)[0]
    sel = bh_indices[idx]

    return (int(snap.black_holes.particle_ids[sel]), masses[sel].to("Msun"), dist[sel])


def save_property(name: str, snapshot_num: int, value, outdir: Path) -> None:
    """Save a single value (ID or mass) to a .npy file with naming convention."""
    fname = f"{name}_{snapshot_num:04d}{DEFAULT_FORMAT}"
    path = outdir / fname
    np.save(path, value)
    logging.info("Saved %s → %s", name, path)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #


def main(
    snapshot_num: int, snap_dir: Path, cat_dir: Path, output_dir: Path, verbose: bool
):
    """Load data, extract BH properties for a given snapshot, and save results."""
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    printgreen(f"Processing snapshot {snapshot_num:04d}...")

    # Load catalogue and snapshot
    cat_path = (
        cat_dir / f"snap_{snapshot_num:04d}" / f"snap_{snapshot_num:04d}.properties"
    )
    snap_path = snap_dir / f"snap_{snapshot_num:04d}.hdf5"
    catalogue = load_catalogue(cat_path)
    snapshot = load_snapshot(snap_path)

    # Compute R500 and center
    r500 = compute_r500(catalogue)
    center = compute_center(catalogue)
    logging.info("R500 = %.3f Mpc; Center = %s", r500, center)

    # Identify most massive BH
    bh_id, bh_mass, bh_radius = find_most_massive_bh(snapshot, center, r500)
    logging.info(
        "Selected BH ID=%d, Mass=%.3E Msun, Radius=%.3f Mpc", bh_id, bh_mass, bh_radius
    )

    # Output to console
    print(f"Snapshot: {snapshot_num:d}")
    print(f"Most massive BH ID: {bh_id}")
    print(f"Mass: {bh_mass:.3E}")
    print(f"Radius: {bh_radius:.3f}")

    # Save ID and mass
    output_dir.mkdir(parents=True, exist_ok=True)
    save_property("final_central_bh_id", snapshot_num, bh_id, output_dir)
    save_property("final_central_bh_mass", snapshot_num, bh_mass, output_dir)


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract final central BH ID and mass from a SWIFT snapshot."
    )
    parser.add_argument(
        "--snapshot", "-s", type=int, required=True, help="Snapshot number (e.g. 199)"
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=DEFAULT_SNAP_DIR,
        help="Directory containing HDF5 snapshot files",
    )
    parser.add_argument(
        "--cat-dir",
        type=Path,
        default=DEFAULT_CAT_DIR,
        help="Directory containing Velociraptor catalogue folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output .npy files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug-level logging"
    )
    args = parser.parse_args()
    main(
        snapshot_num=args.snapshot,
        snap_dir=args.snap_dir,
        cat_dir=args.cat_dir,
        output_dir=args.out_dir,
        verbose=args.verbose,
    )
