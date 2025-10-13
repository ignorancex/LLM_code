#!/usr/bin/env python3
"""
Compute the time evolution of gas-phase properties for predefined particle masks.

For each ParticleMask (defined by radial shell, temperature phase, and MPI origin),
this script:
  1. Loads particle ID slices.
  2. Iterates through SWIFT snapshots to compute median entropy, density contrast,
     temperature, radius, and redshift for the selected particles.
  3. Appends normalization entries at z=0 (e.g., K500, T500).
  4. Writes out each history array as .npy files named:
     gas_history_<metric>_R<radial>_T<temp>_O<origin>.npy

Usage:
    python particle_history.py [--snap-dir SNAP_DIR]
                                   [--cat-dir CAT_DIR]
                                   [--num-snaps N]
                                   [--out-dir OUT_DIR]
                                   [--verbose]
"""
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import unyt
from swiftsimio import load as load_snapshot
from velociraptor import load as load_catalogue
from warnings import filterwarnings
from tqdm import tqdm

# Suppress runtime warnings from unyt & NumPy mixtures
filterwarnings("ignore", category=RuntimeWarning, message="Mixing ufunc arguments")

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #
TEMP_HOT = 1e5 * unyt.K
MU = 0.5954  # mean molecular weight
MU_E = 1.14  # atomic weight per free electron
DEFAULT_NUM = 200
DEFAULT_SNAP = Path("..") / "snapshots"
DEFAULT_CAT = Path("..") / "stf"
DEFAULT_OUT = Path(".")


# ---------------------------------------------------------------------------- #
# Data Classes
# ---------------------------------------------------------------------------- #
@dataclass
class SnapshotRef:
    """Reference to one SWIFT snapshot and its catalogue."""

    index: int
    snap_file: Path
    cat_file: Path


@dataclass
class ParticleMask:
    """Defines a particle selection mask by shell, phase, and origin."""

    radial: str  # e.g. 'core', 'shell', 'field'
    temperature: str  # 'hot' or 'cold'
    origin: str  # 'nr' or 'ref'
    file: Path  # path to .npy ID file


# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #
def setup_logger(verbose: bool) -> None:
    """Configure root logger format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #


def astropy_to_unyt(q):
    """Convert an astropy Quantity to unyt."""
    return unyt.unyt_quantity(q.value, q.unit.to_string(format="ogip"))


def write_history(metric: str, mask: ParticleMask, array: np.ndarray, outdir: Path):
    """Save a history array to .npy with mask-coded filename."""
    outdir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"gas_history_{metric}_R{mask.radial}_T{mask.temperature}_O{mask.origin}.npy"
    )
    path = outdir / fname
    np.save(path, array)
    logging.info("Saved %s (%d entries) to %s", metric, array.size, path)


def find_snapshots(num: int, snap_dir: Path, cat_dir: Path) -> List[SnapshotRef]:
    """Build list of SnapshotRef for which files exist."""
    refs = []
    for i in range(num):
        snap_file = snap_dir / f"snap_{i:04d}.hdf5"
        cat_file = cat_dir / f"snap_{i:04d}" / f"snap_{i:04d}.properties"
        if snap_file.is_file() and cat_file.is_file():
            refs.append(SnapshotRef(i, snap_file, cat_file))
    return refs


# ---------------------------------------------------------------------------- #
# Core Computation
# ---------------------------------------------------------------------------- #


def compute_mask_history(snap_refs: List[SnapshotRef], mask: ParticleMask) -> dict:
    """
    For a given ParticleMask, compute time series of metrics:
      redshift, entropy, density contrast, temperature, radius
    Returns dict of numpy arrays (length = max_index+2)
    """
    max_len = max(r.index for r in snap_refs) + 2
    # initialize arrays
    arrays = {
        "redshift": np.zeros(max_len),
        "entropy": np.zeros(max_len),
        "density": np.zeros(max_len),
        "temperature": np.zeros(max_len),
        "radius": np.zeros(max_len),
    }

    # load slice IDs
    ids = np.load(mask.file)

    last_meta = None
    # iterate snapshots
    for ref in tqdm(
        snap_refs, desc=f"Mask {mask.radial}-{mask.temperature}-{mask.origin}"
    ):
        ds_cat = load_catalogue(ref.cat_file)
        ds_snap = load_snapshot(ref.snap_file)

        z = ds_snap.metadata.redshift
        arrays["redshift"][ref.index] = z

        # compute R500 and center
        r500 = (
            ds_cat.spherical_overdensities.r_500_rhocrit[0].to("Mpc") / ds_cat.units.a
        )
        pos = ds_cat.positions
        center = unyt.unyt_array(
            [pos.xcmbp[0], pos.ycmbp[0], pos.zcmbp[0]] / ds_cat.units.a, "Mpc"
        )

        # radial distances scaled
        disp = ds_snap.gas.coordinates - center
        rd = np.linalg.norm(disp, axis=1) * center.units
        r_scaled = rd.to("Mpc") / r500

        # convert to physical
        ds_snap.gas.densities.convert_to_physical()
        ds_snap.gas.temperatures.convert_to_physical()
        if not hasattr(ds_snap.gas, "electron_number_densities"):
            ds_snap.gas.electron_number_densities = (
                ds_snap.gas.densities.in_cgs() / unyt.mh / MU_E
            )
        else:
            ds_snap.gas.electron_number_densities.convert_to_physical()

        ent = (
            ds_snap.gas.temperatures
            * unyt.kb
            / ds_snap.gas.electron_number_densities.in_cgs() ** (2 / 3)
        ).to("keV*cm**2")

        # background density
        rho_bg = ds_snap.metadata.cosmology.Om0 * astropy_to_unyt(
            ds_snap.metadata.cosmology.critical_density(z)
        ).to("Msun/Mpc**3")

        # slice mask
        sel = np.isin(ds_snap.gas.particle_ids, ids, assume_unique=True)

        arrays["entropy"][ref.index] = np.median(ent[sel].to("keV*cm**2"))
        arrays["density"][ref.index] = (
            np.median(ds_snap.gas.densities[sel].to("Msun/Mpc**3")) / rho_bg
        )
        arrays["temperature"][ref.index] = np.median(
            ds_snap.gas.temperatures[sel].to("K")
        )
        arrays["radius"][ref.index] = np.median(r_scaled[sel])

        last_meta = (ds_cat, ds_snap, r500)

    # normalization at z=0
    ds_cat, ds_snap, r500 = last_meta
    arrays["redshift"][-1] = 1.0

    m500 = ds_cat.spherical_overdensities.mass_500_rhocrit[0].to("Msun")
    kBT = (unyt.G * MU * m500 * unyt.mp / r500 / 2).to("keV")

    fb = ds_snap.metadata.cosmology.Ob0 / ds_snap.metadata.cosmology.Om0
    rho0 = astropy_to_unyt(ds_snap.metadata.cosmology.critical_density0).to(
        "Msun/Mpc**3"
    )
    ne0 = 500 * fb * rho0 / (MU_E * unyt.mp)
    ent0 = kBT / (ne0 ** (2 / 3))

    arrays["entropy"][-1] = ent0.to("keV*cm**2")
    arrays["density"][-1] = 1.0
    arrays["temperature"][-1] = (kBT / unyt.kb).to("K")
    arrays["radius"][-1] = 1.0

    return arrays


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #


def main(
    snap_dir: Path,
    cat_dir: Path,
    num_snaps: int,
    masks: List[ParticleMask],
    out_dir: Path,
    verbose: bool,
) -> None:
    """Coordinate history computations for all ParticleMasks."""
    setup_logger(verbose)
    logging.info("Building snapshot list...")
    snap_refs = find_snapshots(num_snaps, snap_dir, cat_dir)
    logging.info("Found %d snapshots to process.", len(snap_refs))

    for mask in masks:
        logging.info(
            "Processing mask: %s-%s (%s)", mask.radial, mask.temperature, mask.origin
        )
        histories = compute_mask_history(snap_refs, mask)
        for metric, arr in histories.items():
            write_history(metric, mask, arr, out_dir)


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute gas history for pre-defined particle masks."
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=DEFAULT_SNAP,
        help="Directory containing snapshots",
    )
    parser.add_argument(
        "--cat-dir",
        type=Path,
        default=DEFAULT_CAT,
        help="Directory containing Velociraptor catalogues",
    )
    parser.add_argument(
        "--num-snaps",
        type=int,
        default=DEFAULT_NUM,
        help="Number of snapshots to attempt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for history arrays",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # detect origins based on script path
    parent = Path(__file__).parent.name
    if "adiabatic" in parent:
        this_origin = "nr"
        other_origin = "ref"
        other_dir = Path(parent).with_name(parent.replace("adiabatic", "ref"))
    elif "ref" in parent:
        this_origin = "ref"
        other_origin = "nr"
        other_dir = Path(parent).with_name(parent.replace("ref", "adiabatic"))
    else:
        this_origin = other_origin = "nr"
        other_dir = Path(".")

    # define masks
    base_file = Path(".")
    masks = []
    for origin, odir in [(this_origin, base_file), (other_origin, other_dir)]:
        for shell in ("core", "shell", "field"):
            masks.append(
                ParticleMask(
                    shell,
                    "hot",
                    origin,
                    odir / f"gas_particle_ids_{shell}_hot_0199.npy",
                )
            )

    main(
        snap_dir=args.snap_dir,
        cat_dir=args.cat_dir,
        num_snaps=args.num_snaps,
        masks=masks,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )
