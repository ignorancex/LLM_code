#!/usr/bin/env python3
"""
Compute and save base visualization metrics for SWIFT snapshots.
Optionally skip non-adiabatic metrics with `--adiabatic`.
"""
import argparse
import logging
from pathlib import Path
from warnings import filterwarnings, warn

import attr
import numpy as np
import unyt
from astropy.cosmology import z_at_value
from sciris import printgreen
from swiftsimio import load as load_snapshot_file, SWIFTDataset
from tqdm import tqdm
from velociraptor import load as load_catalogue, VelociraptorCatalogue

# Suppress known runtime warnings
filterwarnings("ignore", category=RuntimeWarning, message="Mixing ufunc arguments")

# ---------------------------------------------------------------------------- #
# Default settings
# ---------------------------------------------------------------------------- #
DEFAULT_SNAP_DIR = Path("..") / "snapshots"
DEFAULT_CATALOGUE_DIR = Path("..") / "stf"
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_BH_ID_FILE = Path("final_central_bh_id_0199.npy")
NUM_SNAPSHOTS = 200
HOT_TEMPERATURE = 1e5 * unyt.K
MEAN_MU = 0.5954  # Mean molecular weight
MEAN_PER_FREE_E = 1.14  # Atomic weight per free electron


@attr.s(auto_attribs=True)
class Snapshot:
    """Container for SWIFT dataset and Velociraptor catalogue."""

    data: SWIFTDataset
    catalogue: VelociraptorCatalogue


# ---------------------------------------------------------------------------- #
# Logging setup
# ---------------------------------------------------------------------------- #
def setup_logger(level=logging.INFO):
    """Configure root logger with timestamped output."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def compute_radial_distance(
    coordinates: np.ndarray, center: unyt.unyt_array
) -> unyt.unyt_array:
    """Return array of distances from center for each point."""
    displacement = coordinates - center
    radii = np.linalg.norm(displacement, axis=1)
    return unyt.unyt_array(radii, center.units)


def save_array(name: str, array: np.ndarray, outdir: Path):
    """Save NumPy array to disk as .npy and log the action."""
    path = outdir / f"{name}.npy"
    np.save(path, array)
    logging.info(f"Saved '{name}' to {path}")


# ---------------------------------------------------------------------------- #
# Data Loading
# ---------------------------------------------------------------------------- #
def load_snapshot(index: int, snap_dir: Path, cat_dir: Path) -> Snapshot:
    """Load SWIFT snapshot and matching Velociraptor catalogue by index."""
    snap_file = snap_dir / f"snap_{index:04d}.hdf5"
    cat_file = cat_dir / f"snap_{index:04d}" / f"snap_{index:04d}.properties"
    data = load_snapshot_file(snap_file)
    catalogue = load_catalogue(cat_file)
    return Snapshot(data=data, catalogue=catalogue)


# ---------------------------------------------------------------------------- #
# Metric Functions
# ---------------------------------------------------------------------------- #
def compute_redshift(snap: Snapshot) -> float:
    """Return cosmological redshift of snapshot."""
    return snap.data.metadata.redshift


def compute_r500(snap: Snapshot) -> unyt.unyt_quantity:
    """Return R500 (radius enclosing 500×critical density) in Mpc."""
    return snap.catalogue.spherical_overdensities.r_500_rhocrit[0].to("Mpc")


def compute_m500(snap: Snapshot) -> unyt.unyt_quantity:
    """Return mass enclosed within R500 in solar masses."""
    return snap.catalogue.spherical_overdensities.mass_500_rhocrit[0].to("Msun")


def compute_center(snap: Snapshot) -> unyt.unyt_array:
    """Return 3D center position in comoving Mpc."""
    a = snap.catalogue.units.a
    pos = snap.catalogue.positions
    coords = unyt.unyt_array([pos.xcmbp[0], pos.ycmbp[0], pos.zcmbp[0]] / a, "Mpc")
    return coords


def compute_t500(snap: Snapshot) -> unyt.unyt_quantity:
    """Return characteristic temperature T500 in Kelvin."""
    r = compute_r500(snap)
    m = compute_m500(snap)
    kBT = (unyt.G * MEAN_MU * m * unyt.mp / r / 2).to("keV")
    return (kBT / unyt.kb).to("K")


def compute_k500(snap: Snapshot) -> unyt.unyt_quantity:
    """Return entropy K500 in keV·cm²."""
    z = compute_redshift(snap)
    T = (unyt.G * MEAN_MU * compute_m500(snap) * unyt.mp / compute_r500(snap) / 2).to(
        "keV"
    )
    cosmo = snap.data.metadata.cosmology
    fb = cosmo.Ob0 / cosmo.Om0
    rho_c = cosmo.critical_density(z).to("Msun/Mpc**3").value
    ne = 500 * fb * rho_c / (MEAN_PER_FREE_E * unyt.mp.value)
    K = T / (ne ** (2 / 3))
    return unyt.unyt_quantity(K, "keV*cm**2")


def compute_hot_gas_mass(snap: Snapshot) -> unyt.unyt_quantity:
    """Return mass of gas with T > HOT_TEMPERATURE within R500."""
    cen = compute_center(snap)
    rd = compute_radial_distance(snap.data.gas.coordinates, cen).to("Mpc")
    rp = compute_r500(snap) / snap.catalogue.units.a
    mask = (rd / rp < 1) & (snap.data.gas.temperatures > HOT_TEMPERATURE)
    return snap.data.gas.masses.to_physical()[mask].sum().to("Msun")


def compute_cold_gas_mass(snap: Snapshot) -> unyt.unyt_quantity:
    """Return mass of gas with T < HOT_TEMPERATURE within R500."""
    cen = compute_center(snap)
    rd = compute_radial_distance(snap.data.gas.coordinates, cen).to("Mpc")
    rp = compute_r500(snap) / snap.catalogue.units.a
    mask = (rd / rp < 1) & (snap.data.gas.temperatures < HOT_TEMPERATURE)
    return snap.data.gas.masses.to_physical()[mask].sum().to("Msun")


def compute_star_mass(snap: Snapshot) -> unyt.unyt_quantity:
    """Return stellar mass within R500."""
    return snap.catalogue.spherical_overdensities.mass_star_500_rhocrit[0].to("Msun")


def compute_gas_mass(snap: Snapshot) -> unyt.unyt_quantity:
    """Return total gas mass within R500."""
    return snap.catalogue.spherical_overdensities.mass_gas_500_rhocrit[0].to("Msun")


def compute_specific_sfr(snap: Snapshot) -> unyt.unyt_quantity:
    """Compute specific star formation rate over last 50 Myr inside 50 kpc."""
    cosmo = snap.data.metadata.cosmology
    z = compute_redshift(snap)
    dt = 50 * unyt.Myr
    t_now = cosmo.age(z).to("Gyr").value
    z0 = z_at_value(cosmo.age, (t_now - dt.to("Gyr").value) * unyt.Gyr)
    a0 = 1 / (1 + z0)
    cen = compute_center(snap)
    rd = (
        compute_radial_distance(snap.data.stars.coordinates, cen).to("Mpc")
        * snap.catalogue.units.a
    )
    mask_all = rd < 50 * unyt.kpc
    mask_new = mask_all & (snap.data.stars.birth_scale_factors >= a0)
    m_new = snap.data.stars.masses[mask_new].sum().to("Msun")
    m_tot = snap.data.stars.masses[mask_all].sum().to("Msun")
    return (m_new / dt / m_tot).to("1/Gyr")


def compute_bh_mass(snap: Snapshot) -> unyt.unyt_quantity:
    """Return mass of the most massive BH within R500, or zero."""
    if snap.data.metadata.n_black_holes == 0:
        warn("No black holes in snapshot.")
        return unyt.unyt_quantity(0, "Msun")
    cen = compute_center(snap)
    rd = compute_radial_distance(snap.data.black_holes.coordinates, cen).to("Mpc")
    mask = rd < (compute_r500(snap) / snap.catalogue.units.a)
    if not mask.any():
        warn("No BH found within R500.")
        return unyt.unyt_quantity(0, "Msun")
    masses = snap.data.black_holes.subgrid_masses.to_physical()[mask]
    return masses.max().to("Msun")


def compute_bh_mass_by_id(snap: Snapshot, id_file: Path) -> unyt.unyt_quantity:
    """Return BH mass matching ID from `id_file`, or zero."""
    bh_id = int(np.load(id_file))
    ids = snap.data.black_holes.particle_ids
    idx = np.where(ids == bh_id)[0]
    if idx.size == 0:
        warn(f"BH ID {bh_id} not found.")
        return unyt.unyt_quantity(0, "Msun")
    return snap.data.black_holes.subgrid_masses.to_physical()[idx[0]].to("Msun")


def compute_bh_edd_fraction(snap: Snapshot) -> float:
    """Return maximum BH Eddington fraction within R500, or zero."""
    if snap.data.metadata.n_black_holes == 0:
        return 0.0
    cen = compute_center(snap)
    rd = compute_radial_distance(snap.data.black_holes.coordinates, cen).to("Mpc")
    mask = rd < (compute_r500(snap) / snap.catalogue.units.a)
    if not mask.any():
        return 0.0
    edd = snap.data.black_holes.eddington_fractions.to_physical()[mask]
    return float(edd.max())


def compute_bh_edd_fraction_by_id(snap: Snapshot, id_file: Path) -> float:
    """Return BH Eddington fraction by ID, or zero if not found."""
    bh_id = int(np.load(id_file))
    ids = snap.data.black_holes.particle_ids
    idx = np.where(ids == bh_id)[0]
    if idx.size == 0:
        return 0.0
    return float(snap.data.black_holes.eddington_fractions.to_physical()[idx[0]])


# ---------------------------------------------------------------------------- #
# Main execution
# ---------------------------------------------------------------------------- #
def main(use_adiabatic: bool):
    """Load snapshots, compute metrics, and save result arrays."""
    setup_logger()
    printgreen("Loading snapshots...")

    snapshots = []
    for i in tqdm(range(NUM_SNAPSHOTS), desc="Snapshots"):
        try:
            snapshots.append(load_snapshot(i, DEFAULT_SNAP_DIR, DEFAULT_CATALOGUE_DIR))
        except Exception:
            continue
    if len(snapshots) < NUM_SNAPSHOTS:
        warn(f"Loaded {len(snapshots)}/{NUM_SNAPSHOTS} snapshots")

    # Define metrics to compute
    metrics = {
        "redshift": lambda s: compute_redshift(s),
        "r500": lambda s: compute_r500(s).value,
        "m500": lambda s: compute_m500(s).value,
        "t500": lambda s: compute_t500(s).value,
        "k500": lambda s: compute_k500(s).value,
        "hot_gas_mass": lambda s: compute_hot_gas_mass(s).value,
        "cold_gas_mass": lambda s: compute_cold_gas_mass(s).value,
        "star_mass": lambda s: compute_star_mass(s).value,
        "gas_mass": lambda s: compute_gas_mass(s).value,
    }

    if not use_adiabatic:
        metrics.update(
            {
                "specific_sfr": lambda s: compute_specific_sfr(s).value,
                "bh_mass": lambda s: compute_bh_mass(s).value,
                "bh_mass_by_id": lambda s: compute_bh_mass_by_id(
                    s, DEFAULT_BH_ID_FILE
                ).value,
                "bh_edd_fraction": lambda s: compute_bh_edd_fraction(s),
                "bh_edd_fraction_by_id": lambda s: compute_bh_edd_fraction_by_id(
                    s, DEFAULT_BH_ID_FILE
                ),
            }
        )

    # Compute and save each metric
    for name, func in metrics.items():
        logging.info(f"Computing {name}...")
        values = np.array([func(s) for s in tqdm(snapshots, desc=name)])
        save_array(name, values, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualization data from SWIFT snapshots."
    )
    parser.add_argument(
        "--adiabatic", action="store_true", help="Only compute adiabatic metrics"
    )
    args = parser.parse_args()
    main(use_adiabatic=args.adiabatic)
