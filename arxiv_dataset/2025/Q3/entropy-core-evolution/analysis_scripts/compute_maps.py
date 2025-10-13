#!/usr/bin/env python3
"""
projection_visualizations.py

Generate gas projections (density, temperature, entropy, etc.)
for selected simulation snapshots using SWIFTSimIO and Velociraptor outputs.
"""
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import unyt
from velociraptor import load as load_catalogue
from swiftsimio import load as load_snapshot, mask
from swiftsimio.visualisation.projection import project_pixel_grid
from warnings import filterwarnings
from dataclasses import dataclass

# Suppress specific runtime warnings
filterwarnings("ignore", category=RuntimeWarning, message="Mixing ufunc arguments")

# Physical constants
MEAN_WEIGHT_PER_ELECTRON = 1.14  # atomic weight per free electron
TEMP_LOW = 1e2 * unyt.K
TEMP_HIGH = 1e5 * unyt.K
DEFAULT_SNAP_NUMS = [1, 16, 49, 65, 100, 112, 140, 160, 199]


@dataclass
class ProjectionConfig:
    name: str
    size: unyt.unyt_quantity
    depth: unyt.unyt_quantity
    backend: str


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create base gas projections for selected snapshots"
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path(".."),
        help="Root directory containing 'snapshots' and 'stf' subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./projections"),
        help="Directory to save projection arrays",
    )
    parser.add_argument(
        "--resolution", type=int, default=2048, help="Pixel resolution for projections"
    )
    parser.add_argument(
        "--snap-nums",
        type=int,
        nargs="+",
        default=DEFAULT_SNAP_NUMS,
        help="List of snapshot indices to process",
    )
    return parser.parse_args()


def write_array(
    output_dir: Path, base: str, proj_name: str, snap_num: int, data: np.ndarray
) -> None:
    filename = output_dir / f"{base}_{proj_name}_{snap_num:04d}.npy"
    np.save(filename, data)
    logging.info(f"Saved: {filename}")


def compute_center(catalogue) -> unyt.unyt_array:
    """Compute comoving center position from Velociraptor catalogue."""
    pos = (
        np.array(
            [
                catalogue.positions.xcmbp[0],
                catalogue.positions.ycmbp[0],
                catalogue.positions.zcmbp[0],
            ]
        )
        / catalogue.units.a
    )
    return unyt.unyt_array(pos, "Mpc")


def constrain_region(
    slice_obj,
    center: unyt.unyt_array,
    size: unyt.unyt_quantity,
    depth: unyt.unyt_quantity,
):
    box = slice_obj.metadata.boxsize
    x0, y0, z0 = center
    slice_obj.constrain_spatial(
        restrict=[
            [x0 - size, x0 + size],
            [y0 - size, y0 + size],
            [z0 - depth, z0 + depth],
        ]
    )


def project_field(ds, field: str, projection_kwargs: dict) -> np.ndarray:
    return project_pixel_grid(ds.gas, project=field, **projection_kwargs)


def weighted_projection(ds, weight: np.ndarray, projection_kwargs: dict) -> np.ndarray:
    ds.gas.__dict__["__temp_weight"] = weight
    img_num = project_pixel_grid(ds.gas, project="__temp_weight", **projection_kwargs)
    img_den = project_pixel_grid(ds.gas, project="masses", **projection_kwargs)
    del ds.gas.__dict__["__temp_weight"]
    return img_num / img_den


def process_snapshot(
    snap_num: int, args, projections: List[ProjectionConfig], logger: logging.Logger
):
    snap_path = args.snapshots_dir / "snapshots" / f"snap_{snap_num:04d}.hdf5"
    cat_path = (
        args.snapshots_dir
        / "stf"
        / f"snap_{snap_num:04d}/snap_{snap_num:04d}.properties"
    )
    try:
        ds = load_snapshot(snap_path)
        cat = load_catalogue(cat_path)
    except FileNotFoundError:
        logger.warning(f"Snapshot {snap_num} files not found, skipping.")
        return

    center = compute_center(cat)
    logger.info(f"Snapshot {snap_num:04d}: center={center}")

    for proj in projections:
        slice_obj = mask(filename=str(snap_path), spatial_only=True)
        constrain_region(slice_obj, center, proj.size, proj.depth)
        ds_masked = load_snapshot(snap_path, mask=slice_obj)

        projection_kwargs = dict(
            resolution=args.resolution,
            backend=proj.backend,
            region=slice_obj.region,
            boxsize=ds_masked.metadata.boxsize,
            parallel=True,
        )

        # Basic gas particle density
        dens = project_field(ds_masked, project=None, **projection_kwargs)
        write_array(args.output_dir, "gas_density", proj.name, snap_num, dens)

        # Temperature masks
        temps = ds_masked.gas.temperatures.to("K")
        masses = ds_masked.gas.masses.to("Msun")
        low_mask = (temps < TEMP_LOW) | (temps > TEMP_HIGH)
        high_mask = temps < TEMP_HIGH

        # Define projection tasks: (output_basename, weight_array)
        weight_tasks: List[Tuple[str, np.ndarray]] = [
            ("mw_temp", masses * temps),
            ("mw_temp_low", masses * temps * low_mask),
            ("mw_temp_high", masses * temps * high_mask),
            ("em_temp", masses * temps * ds_masked.gas.densities),
            ("em_temp_low", masses * temps * ds_masked.gas.densities * low_mask),
            ("em_temp_high", masses * temps * ds_masked.gas.densities * high_mask),
        ]

        for name, weight in weight_tasks:
            img = weighted_projection(ds_masked, weight.value, projection_kwargs)
            write_array(args.output_dir, f"gas_{name}", proj.name, snap_num, img)

        # Entropy: k_BT / n_e^(2/3)
        ne = (
            ds_masked.gas.densities.to("g/cm**3") / unyt.mp / MEAN_WEIGHT_PER_ELECTRON
        ).to("1/cm**3")
        kT = (temps * unyt.kb).to("keV")
        ent = (kT / ne ** (2 / 3)).to("keV*cm**2")
        em = masses * ent.value
        img_ent = weighted_projection(ds_masked, em.value, projection_kwargs)
        write_array(args.output_dir, "gas_mw_entropy", proj.name, snap_num, img_ent)

        # Metallicity fraction high-temp
        if hasattr(ds_masked.gas, "metal_mass_fractions"):
            metal = ds_masked.gas.metal_mass_fractions * masses * high_mask
            img_mf = weighted_projection(ds_masked, metal.value, projection_kwargs)
            write_array(
                args.output_dir, "gas_metal_frac_high", proj.name, snap_num, img_mf
            )


def main():
    logger = setup_logger()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    projections = [
        ProjectionConfig("far", 10.0 * unyt.Mpc, 10.0 * unyt.Mpc, "subsampled_extreme"),
        ProjectionConfig("mid", 5.0 * unyt.Mpc, 5.0 * unyt.Mpc, "subsampled_extreme"),
        ProjectionConfig("close", 2.0 * unyt.Mpc, 2.0 * unyt.Mpc, "subsampled_extreme"),
        ProjectionConfig("inner", 0.2 * unyt.Mpc, 0.2 * unyt.Mpc, "subsampled_extreme"),
    ]

    for snap_num in args.snap_nums:
        process_snapshot(snap_num, args, projections, logger)


if __name__ == "__main__":
    main()
