#!/usr/bin/env python3
"""
Compute 2D hydrogen density–temperature histograms (phase space)
for gas particles selected by predefined masks across multiple snapshots.

This script:
  - Loads SWIFT snapshots
  - Converts densities & temperatures to physical units
  - Computes hydrogen number density n_H = rho/(mu*m_H)
  - For each ParticleMask, generates histogram(n_H, T) per snapshot
  - Saves per-mask 3D histogram arrays as .npy files

Usage:
    python gas_phase_space.py [--snap-dir SNAP_DIR]
                              [--cat-dir CAT_DIR]
                              [--num-snaps N]
                              [--dens-bounds a b]
                              [--temp-bounds c d]
                              [--bins B]
                              [--mask-list MASK_JSON]
                              [--out-dir OUT_DIR]
                              [--verbose]
"""
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import unyt
from swiftsimio import load as load_snapshot

# velociraptor catalogue loading not needed here
from warnings import filterwarnings
from tqdm import tqdm

# suppress noisy runtime warnings
filterwarnings("ignore", category=RuntimeWarning, message="Mixing ufunc arguments")

# physical constants
MU = 0.5954  # mean molecular weight
MH = unyt.mh  # proton mass


@dataclass
class SnapshotRef:
    """Reference to a SWIFT snapshot."""

    index: int
    path: Path


@dataclass
class ParticleMask:
    """Defines a selection mask for gas particles."""

    radial: str  # e.g. 'core','shell','field','all'
    temperature: str  # 'hot','cold','all'
    origin: str  # e.g. 'nr','ref'
    file: Optional[Path]  # .npy file of particle IDs, or None for all


# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #
def setup_logger(verbose: bool) -> None:
    """Initialize root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Data Loading
# ---------------------------------------------------------------------------- #
def find_snapshots(snap_dir: Path, num_snaps: int) -> List[SnapshotRef]:
    """Return list of existing SWIFT snapshot references up to num_snaps."""
    refs = []
    for i in range(num_snaps):
        path = snap_dir / f"snap_{i:04d}.hdf5"
        if path.is_file():
            refs.append(SnapshotRef(i, path))
    return refs


# ---------------------------------------------------------------------------- #
# Processing
# ---------------------------------------------------------------------------- #
def compute_histogram(
    snap_ref: SnapshotRef,
    ids: Union[np.ndarray, slice],
    density_bins: np.ndarray,
    temp_bins: np.ndarray,
) -> np.ndarray:
    """Compute 2D histogram (n_H vs T) for one snapshot and ID selection."""
    ds = load_snapshot(snap_ref.path)

    # convert to physical units
    ds.gas.densities.convert_to_physical()
    ds.gas.temperatures.convert_to_physical()

    # compute hydrogen number density (in cgs)
    rho_cgs = ds.gas.densities.in_cgs()  # [g/cm^3]
    nH = (rho_cgs / (MU * MH)).in_cgs().value  # [1/cm^3]

    # select particles
    mask = (
        ids
        if isinstance(ids, slice)
        else np.isin(ds.gas.particle_ids, ids, assume_unique=True)
    )

    # temperatures in cgs (K)
    T_cgs = ds.gas.temperatures.in_cgs().value

    # 2D histogram
    hist, _, _ = np.histogram2d(nH[mask], T_cgs[mask], bins=[density_bins, temp_bins])
    return hist


# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
def save_histogram(hist_array: np.ndarray, mask: ParticleMask, out_dir: Path) -> None:
    """Save 3D histogram array to .npy with mask-coded filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"gas_phase_space_R{mask.radial}_T{mask.temperature}_O{mask.origin}.npy"
    path = out_dir / fname
    np.save(path, hist_array)
    logging.info("Saved histogram %s (shape %s)", path, hist_array.shape)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #


def main(
    snap_dir: Path,
    num_snaps: int,
    density_bounds: List[float],
    temp_bounds: List[float],
    bins: int,
    masks: List[ParticleMask],
    out_dir: Path,
    verbose: bool,
) -> None:
    """Run histogram computation for all masks and snapshots."""
    setup_logger(verbose)
    logging.info("Finding snapshots in %s (up to %d)", snap_dir, num_snaps)
    snap_refs = find_snapshots(snap_dir, num_snaps)
    logging.info("%d snapshots found", len(snap_refs))

    # define bin edges
    density_bins = np.logspace(*np.log10(density_bounds), bins + 1)
    temp_bins = np.logspace(*np.log10(temp_bounds), bins + 1)

    # process each mask
    for mask in masks:
        logging.info(
            "Processing mask: R%s T%s O%s", mask.radial, mask.temperature, mask.origin
        )
        # load IDs or select all
        ids = slice(None) if mask.file is None else np.load(mask.file)
        # allocate 3D array: (snapshots, dens_bins, temp_bins)
        hist_all = np.zeros((len(snap_refs), bins, bins))
        # loop snapshots
        for j, ref in enumerate(
            tqdm(snap_refs, desc=f"Hist R{mask.radial}_T{mask.temperature}")
        ):
            hist_all[j] = compute_histogram(ref, ids, density_bins, temp_bins)
        save_histogram(hist_all, mask, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute gas phase-space histograms across snapshots."
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=Path("..") / "snapshots",
        help="Directory of snapshot HDF5 files",
    )
    parser.add_argument(
        "--num-snaps", type=int, default=200, help="Number of snapshots to examine"
    )
    parser.add_argument(
        "--dens-bounds",
        type=float,
        nargs=2,
        default=[1e-8, 1e3],
        help="Density bounds [min max] in H/cm^3",
    )
    parser.add_argument(
        "--temp-bounds",
        type=float,
        nargs=2,
        default=[1e2, 1e10],
        help="Temperature bounds [min max] in K",
    )
    parser.add_argument("--bins", type=int, default=256, help="Number of bins per axis")
    parser.add_argument(
        "--mask",
        dest="masks",
        action="append",
        help="Mask spec in format radial,phase,origin,file (file optional)",
        metavar="RAD,PHASE,ORIG,FILE",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # parse masks
    masks: List[ParticleMask] = []
    if args.masks:
        for m in args.masks:
            parts = m.split(",")
            radial, phase, orig = parts[:3]
            file = Path(parts[3]) if len(parts) == 4 and parts[3] else None
            masks.append(ParticleMask(radial, phase, orig, file))
    else:
        # default masks auto-detected by directory name
        parent = Path(__file__).parent.name
        if "adiabatic" in parent:
            this, other = "nr", "ref"
            other_dir = Path(parent).with_name(parent.replace("adiabatic", "ref"))
        elif "ref" in parent:
            this, other = "ref", "nr"
            other_dir = Path(parent).with_name(parent.replace("ref", "adiabatic"))
        else:
            this = other = "nr"
            other_dir = Path(".")
        # define hot-only masks
        for origin, odir in [(this, Path(".")), (other, other_dir)]:
            masks.append(ParticleMask("all", "all", origin, None))
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
        num_snaps=args.num_snaps,
        density_bounds=args.dens_bounds,
        temp_bounds=args.temp_bounds,
        bins=args.bins,
        masks=masks,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )
