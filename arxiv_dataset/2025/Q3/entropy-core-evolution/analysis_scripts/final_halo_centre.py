#!/usr/bin/env python3
"""
Locate the final completed Velociraptor catalogue for a given snapshot,
load its halo center and mass, and save these properties as .npy files.

This script:
  1. Steps backward until a catalogue file exists.
  2. Steps forward until the catalogue is 'finished' (scale factor a≈1).
  3. Extracts the most massive halo's center and M200mean mass.
  4. Writes `final_center_{num:04d}.npy` and `final_m200_{num:04d}.npy`.
"""
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import unyt
from velociraptor import load as load_catalogue

# ---------------------------------------------------------------------------- #
# Defaults
# ---------------------------------------------------------------------------- #
DEFAULT_CAT_DIR = Path("..") / "stf"
DEFAULT_START_SNAP = 199
MAX_BACK_ITERATIONS = 500
MAX_FORWARD_ITERATIONS = 500
DEFAULT_OUTPUT_DIR = Path(".")
TOLERANCE_A = 1e-6  # tolerance for scale factor ≈1


@dataclass
class CatalogueRef:
    """Holds reference to a Velociraptor catalogue file and its snapshot number."""

    path: Path
    number: int


# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #
def setup_logger(verbose: bool) -> None:
    """Configure logger format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Core functions
# ---------------------------------------------------------------------------- #
def catalogue_exists(cat_ref: CatalogueRef) -> bool:
    """Return True if the catalogue file exists."""
    return cat_ref.path.is_file()


def catalogue_finished(cat_ref: CatalogueRef) -> bool:
    """Return True if catalogue scale factor a ≈ 1.0 (finished)."""
    ds = load_catalogue(cat_ref.path)
    a = ds.units.a
    return abs(a - 1.0) < TOLERANCE_A


def step_back(cat_ref: CatalogueRef, max_iters: int) -> CatalogueRef:
    """Decrement snapshot number until existing catalogue is found or limit reached."""
    for _ in range(max_iters):
        if catalogue_exists(cat_ref):
            logging.info("Found existing catalogue: %s", cat_ref.path)
            return cat_ref
        cat_ref.number -= 1
        if cat_ref.number < 1:
            break
        cat_ref.path = cat_ref.path.with_name(
            cat_ref.path.name.replace(
                f"{cat_ref.number+1:04d}", f"{cat_ref.number:04d}"
            )
        )
        logging.debug("Stepping back to %04d", cat_ref.number)
    raise FileNotFoundError(
        f"No existing catalogue found within {max_iters} steps backward."
    )


def step_forward(cat_ref: CatalogueRef, max_iters: int) -> CatalogueRef:
    """Increment snapshot number until catalogue a≈1 or limit reached."""
    for _ in range(max_iters):
        if catalogue_finished(cat_ref):
            logging.info("Found finished catalogue: %s", cat_ref.path)
            return cat_ref
        cat_ref.number += 1
        cat_ref.path = cat_ref.path.with_name(
            cat_ref.path.name.replace(
                f"{cat_ref.number-1:04d}", f"{cat_ref.number:04d}"
            )
        )
        logging.debug("Stepping forward to %04d", cat_ref.number)
    raise RuntimeError(f"No finished catalogue found within {max_iters} steps forward.")


def extract_center_and_mass(
    cat_ref: CatalogueRef,
) -> tuple[unyt.unyt_array, unyt.unyt_quantity]:
    """Load Velociraptor catalogue and return center coords and M200mean mass."""
    ds = load_catalogue(cat_ref.path)
    # Center in comoving Mpc
    pos = ds.positions
    a = ds.units.a
    coords = unyt.unyt_array([pos.xcmbp[0], pos.ycmbp[0], pos.zcmbp[0]] / a, "Mpc")
    # M200mean mass
    m200 = ds.masses.mass_200mean[0].to("Msun")
    return coords, m200


def save_results(
    center: unyt.unyt_array,
    mass: unyt.unyt_quantity,
    cat_ref: CatalogueRef,
    outdir: Path,
) -> None:
    """Save center and mass to .npy files with snapshot number suffix."""
    outdir.mkdir(parents=True, exist_ok=True)
    num = cat_ref.number
    np.save(outdir / f"final_center_{num:04d}.npy", center.value)
    np.save(outdir / f"final_m200_{num:04d}.npy", mass.value)
    logging.info("Saved final_center_%04d.npy and final_m200_%04d.npy", num, num)


# ---------------------------------------------------------------------------- #
# Main workflow
# ---------------------------------------------------------------------------- #
def main(
    start_snap: int,
    cat_dir: Path,
    back_iters: int,
    forward_iters: int,
    outdir: Path,
    verbose: bool,
) -> None:
    """Orchestrate finding the final catalogue and extracting its properties."""
    setup_logger(verbose)
    # Initialize reference
    cat_path = cat_dir / f"snap_{start_snap:04d}" / f"snap_{start_snap:04d}.properties"
    ref = CatalogueRef(path=cat_path, number=start_snap)

    # Step back to existing
    ref = step_back(ref, back_iters)
    # Step forward to finished
    ref = step_forward(ref, forward_iters)

    # Extract data
    center, mass = extract_center_and_mass(ref)
    print(f"Snapshot: {ref.number:d}")
    print(f"Center: {center}")
    print(f"M200mean: {mass:.3E}")

    # Save outputs
    save_results(center, mass, ref, outdir)


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and extract final Velociraptor catalogue center and mass."
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=DEFAULT_START_SNAP,
        help="Starting snapshot number",
    )
    parser.add_argument(
        "--cat-dir",
        type=Path,
        default=DEFAULT_CAT_DIR,
        help="Base directory for catalogues",
    )
    parser.add_argument(
        "--back-steps",
        type=int,
        default=MAX_BACK_ITERATIONS,
        help="Max steps backward to find existing catalogue",
    )
    parser.add_argument(
        "--forward-steps",
        type=int,
        default=MAX_FORWARD_ITERATIONS,
        help="Max steps forward to find finished catalogue",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    main(
        start_snap=args.start,
        cat_dir=args.cat_dir,
        back_iters=args.back_steps,
        forward_iters=args.forward_steps,
        outdir=args.out_dir,
        verbose=args.verbose,
    )
