import os
import sys
import unyt
import h5py
import numpy as np
import argparse
import pandas as pd
from mpi4py import MPI
from swiftsimio.visualisation.projection_backends import backends_parallel
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths

scatter = backends_parallel["subsampled"]

# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
        )
    )
)

comm = MPI.COMM_WORLD
num_processes = comm.size
rank = comm.rank

from read import MacsisDataset
from register import Macsis

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--particle-type',
    type=str,
    default='gas',
    required=True,
    choices=['gas', 'dm']
)
parser.add_argument(
    '-r',
    '--redshift-index',
    type=int,
    default=22,
    required=False,
    choices=list(range(23))
)
args = parser.parse_args()

ksz_const = - unyt.thompson_cross_section / 1.16 / unyt.speed_of_light / unyt.proton_mass
tsz_const = unyt.thompson_cross_section * unyt.boltzmann_constant / 1.16 / \
            unyt.speed_of_light ** 2 / unyt.proton_mass / unyt.electron_mass


def rotate(coord: np.ndarray, angular_momentum_hot_gas: np.ndarray, tilt: str = 'x'):
    if tilt == 'x':
        rotation_matrix = rotation_matrix_from_vector(np.array([1, 0, 0], dtype=np.float))
    elif tilt == 'y':
        rotation_matrix = rotation_matrix_from_vector(np.array([0, 1, 0], dtype=np.float))
    elif tilt == 'z':
        rotation_matrix = rotation_matrix_from_vector(np.array([0, 0, 1], dtype=np.float))
    elif tilt == 'faceon':
        rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas)
    elif tilt == 'edgeon':
        rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas, axis='y')

    new_coord = np.matmul(rotation_matrix, coord.T).T
    return new_coord


def rksz_map(halo, resolution: int = 1024, alignment: str = 'edgeon'):
    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot('PartType0/Coordinates')
    densities = data.read_snapshot('PartType0/Density')
    masses = data.read_snapshot('PartType0/Mass')
    velocities = data.read_snapshot('PartType0/Velocity')
    temperatures = data.read_snapshot('PartType0/Temperature')
    smoothing_lengths = data.read_snapshot('PartType0/SmoothingLength')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1] / data.a
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / data.a
    m500_crit = data.read_catalogue_subfindtab('FOF/Group_M_Crit500')[1]

    # Select ionised hot gas
    temperature_cut = np.where(temperatures > 1.e5)[0]

    coordinates = coordinates[temperature_cut]
    densities = densities[temperature_cut]
    masses = masses[temperature_cut]
    velocities = velocities[temperature_cut]
    temperatures = temperatures[temperature_cut]
    smoothing_lengths = smoothing_lengths[temperature_cut]

    # Rescale coordinates to CoP
    coordinates[:, 0] -= centre_of_potential[0]
    coordinates[:, 1] -= centre_of_potential[1]
    coordinates[:, 2] -= centre_of_potential[2]

    # Compute mean velocity inside R500
    radial_dist = np.sqrt(
        coordinates[:, 0] ** 2 +
        coordinates[:, 1] ** 2 +
        coordinates[:, 2] ** 2
    )
    r500_mask = np.where(radial_dist < r500_crit)[0]

    mean_velocity_r500 = np.sum(velocities[r500_mask] * masses[r500_mask, None], axis=0) / np.sum(masses[r500_mask])
    angular_momentum_r500 = np.sum(
        np.cross(coordinates[r500_mask], velocities[r500_mask] * masses[r500_mask, None]), axis=0
    ) / np.sum(masses[r500_mask])

    velocities_rest_frame = velocities.copy()
    velocities_rest_frame[:, 0] -= mean_velocity_r500[0]
    velocities_rest_frame[:, 1] -= mean_velocity_r500[1]
    velocities_rest_frame[:, 2] -= mean_velocity_r500[2]

    # Rotate coordinates and velocities
    coordinates_edgeon = rotate(coordinates, angular_momentum_r500, tilt=alignment)
    velocities_rest_frame_edgeon = rotate(velocities_rest_frame, angular_momentum_r500, tilt=alignment)

    # Rotate angular momentum vector for cross check
    angular_momentum_r500_rotated = rotate(
        angular_momentum_r500 / np.linalg.norm(angular_momentum_r500), angular_momentum_r500, tilt=alignment
    ) * r500_crit / 2

    compton_y = unyt.unyt_array(
        masses * velocities_rest_frame_edgeon[:, 2], 1.e10 * unyt.Solar_Mass * 1.e3 * unyt.km / unyt.s
    ) * ksz_const / unyt.unyt_quantity(1., unyt.Mpc) ** 2
    compton_y = compton_y.value

    # Restrict map to 2*R500
    spatial_filter = np.where(
        (np.abs(coordinates_edgeon[:, 0]) < r500_crit / 2) &
        (np.abs(coordinates_edgeon[:, 1]) < r500_crit / 2) &
        (np.abs(coordinates_edgeon[:, 2]) < r500_crit / 2)
    )[0]

    coordinates_edgeon = coordinates_edgeon[spatial_filter]
    velocities_rest_frame_edgeon = velocities_rest_frame_edgeon[spatial_filter]
    smoothing_lengths = smoothing_lengths[spatial_filter]
    compton_y = compton_y[spatial_filter]

    # Make map using swiftsimio
    x = (coordinates_edgeon[:, 0] - coordinates_edgeon[:, 0].min()) / (
            coordinates_edgeon[:, 0].max() - coordinates_edgeon[:, 0].min())
    y = (coordinates_edgeon[:, 1] - coordinates_edgeon[:, 1].min()) / (
            coordinates_edgeon[:, 1].max() - coordinates_edgeon[:, 1].min())
    h = smoothing_lengths / (coordinates_edgeon[:, 0].max() - coordinates_edgeon[:, 0].min())

    # Gather and handle coordinates to be processed
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.asarray(compton_y, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    smoothed_map = scatter(x=x, y=y, m=m, h=h, res=resolution).T

    # Parse info about smoothing lengths
    smoothing_lengths_info = {
        'smoothing_lengths_mean': np.nanmean(smoothing_lengths),
        'smoothing_lengths_std': np.nanstd(smoothing_lengths),
        'smoothing_lengths_median': np.nanmedian(smoothing_lengths),
        'smoothing_lengths_max': np.nanmax(smoothing_lengths),
        'smoothing_lengths_min': np.nanmin(smoothing_lengths),
    }

    return smoothed_map, smoothing_lengths_info


def dm_rotation_map(halo, resolution: int = 1024, alignment: str = 'edgeon'):
    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot('PartType1/Coordinates')
    velocities = data.read_snapshot('PartType1/Velocity')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1] / data.a
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / data.a
    m500_crit = data.read_catalogue_subfindtab('FOF/Group_M_Crit500')[1]

    # Generate smoothing lengths for dark matter
    smoothing_lengths = generate_smoothing_lengths(
        coordinates * unyt.Mpc,
        data.read_header('BoxSize') * unyt.Mpc,
        kernel_gamma=1.897367,  # Taken from Dehnen & Aly 2012
        neighbours=57,
        speedup_fac=1,
        dimension=3,
    ).value

    # Rescale coordinates to CoP
    coordinates[:, 0] -= centre_of_potential[0]
    coordinates[:, 1] -= centre_of_potential[1]
    coordinates[:, 2] -= centre_of_potential[2]

    # Compute mean velocity inside R500
    radial_dist = np.sqrt(
        coordinates[:, 0] ** 2 +
        coordinates[:, 1] ** 2 +
        coordinates[:, 2] ** 2
    )
    r500_mask = np.where(radial_dist < r500_crit)[0]

    mean_velocity_r500 = np.sum(velocities[r500_mask], axis=0) / len(velocities[r500_mask])
    angular_momentum_r500 = np.sum(
        np.cross(coordinates[r500_mask], velocities[r500_mask]), axis=0
    ) / len(velocities[r500_mask])

    velocities_rest_frame = velocities.copy()
    velocities_rest_frame[:, 0] -= mean_velocity_r500[0]
    velocities_rest_frame[:, 1] -= mean_velocity_r500[1]
    velocities_rest_frame[:, 2] -= mean_velocity_r500[2]

    # Rotate coordinates and velocities
    coordinates_edgeon = rotate(coordinates, angular_momentum_r500, tilt=alignment)
    velocities_rest_frame_edgeon = rotate(velocities_rest_frame, angular_momentum_r500, tilt=alignment)

    # Rotate angular momentum vector for cross check
    angular_momentum_r500_rotated = rotate(
        angular_momentum_r500 / np.linalg.norm(angular_momentum_r500), angular_momentum_r500, tilt=alignment
    ) * r500_crit / 2

    compton_y = - velocities_rest_frame_edgeon[:, 2]

    # Restrict map to 2*R500
    spatial_filter = np.where(
        (np.abs(coordinates_edgeon[:, 0]) < r500_crit) &
        (np.abs(coordinates_edgeon[:, 1]) < r500_crit) &
        (np.abs(coordinates_edgeon[:, 2]) < r500_crit)
    )[0]

    coordinates_edgeon = coordinates_edgeon[spatial_filter]
    velocities_rest_frame_edgeon = velocities_rest_frame_edgeon[spatial_filter]
    smoothing_lengths = smoothing_lengths[spatial_filter]
    compton_y = compton_y[spatial_filter]

    # Make map using swiftsimio
    x = (coordinates_edgeon[:, 0] - coordinates_edgeon[:, 0].min()) / (
            coordinates_edgeon[:, 0].max() - coordinates_edgeon[:, 0].min())
    y = (coordinates_edgeon[:, 1] - coordinates_edgeon[:, 1].min()) / (
            coordinates_edgeon[:, 1].max() - coordinates_edgeon[:, 1].min())
    h = smoothing_lengths / (coordinates_edgeon[:, 0].max() - coordinates_edgeon[:, 0].min())

    # Gather and handle coordinates to be processed
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.asarray(compton_y, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    smoothed_map = scatter(x=x, y=y, m=m, h=h, res=resolution).T

    # Parse info about smoothing lengths
    smoothing_lengths_info = {
        'smoothing_lengths_mean': np.nanmean(smoothing_lengths),
        'smoothing_lengths_std': np.nanstd(smoothing_lengths),
        'smoothing_lengths_median': np.nanmedian(smoothing_lengths),
        'smoothing_lengths_max': np.nanmax(smoothing_lengths),
        'smoothing_lengths_min': np.nanmin(smoothing_lengths),
    }

    return smoothed_map / 1.e10, smoothing_lengths_info


def dump_to_hdf5_parallel(particle_type: str = 'gas', resolution: int = 1024):
    # Switch the type of map between gas and DM
    if particle_type == 'gas':
        generate_map = rksz_map
    elif particle_type == 'dm':
        generate_map = dm_rotation_map

    macsis = Macsis()
    with h5py.File(
            f'{macsis.output_dir}/rksz_{particle_type}_{args.redshift_index:03d}.hdf5', 'w',
            driver='mpio', comm=comm
    ) as f:

        # Retrieve all zoom handles in parallel (slow otherwise)
        data_handles = np.empty(0, dtype=np.object)
        for zoom_id in range(macsis.num_zooms):
            if zoom_id % num_processes == rank:
                print(f"Collecting metadata for process ({zoom_id:03d}/{macsis.num_zooms - 1})...")
                data_handles = np.append(
                    data_handles,
                    macsis.get_zoom(zoom_id).get_redshift(args.redshift_index)
                )

        zoom_handles = comm.allgather(data_handles)
        zoom_handles = np.concatenate(zoom_handles).ravel()
        zoom_handles = zoom_handles[~pd.isnull(zoom_handles)]

        if rank == 0:
            print(
                f"Redshift index (from argvals): {args.redshift_index:d}\n"
                f"Handles working on redshift {zoom_handles[0].z:.3f}\n"
                f"Analysing {len(data_handles):d} data-handles. Names printed below.\n",
                [data_handle.run_name for data_handle in data_handles]
            )

        # Editing the structure of the file MUST be done collectively
        if rank == 0:
            print("Preparing structure of the file (collective operations)...")
        for zoom_id, data_handle in enumerate(zoom_handles):
            if rank == 0:
                print(f"Structuring ({zoom_id:03d}/{macsis.num_zooms - 1}): {data_handle.run_name}")
            if data_handle.run_name not in f.keys():
                halo_group = f.create_group(f"{data_handle.run_name}")
            for projection in ['x', 'y', 'z', 'faceon', 'edgeon']:
                if projection not in halo_group.keys():
                    halo_group.create_dataset(f"map_{projection}", (resolution, resolution), dtype=np.float64)

        # Data assignment can be done through independent operations
        for zoom_id, data_handle in enumerate(zoom_handles):
            if zoom_id % num_processes == rank:
                print((
                    f"Rank {rank:03d} processing halo ({zoom_id:03d}/{macsis.num_zooms - 1}) | "
                    f"MACSIS name: {data_handle.run_name}"
                ))
                for projection in ['x', 'y', 'z', 'faceon', 'edgeon']:

                    # Check that the arrays is not all zeros
                    if not np.any(f[f"{data_handle.run_name}/map_{projection}"][:]):
                        smoothed_map, smoothing_length_info = generate_map(
                            data_handle,
                            resolution=resolution,
                            alignment=projection
                        )
                        f[f"{data_handle.run_name}/map_{projection}"][:] = smoothed_map

                # Write attributes to hdf5 group
                for info_name in smoothing_length_info:
                    f[f"{data_handle.run_name}"].attrs.create(
                        info_name,
                        smoothing_length_info[smoothing_length_info]
                    )


if __name__ == "__main__":
    dump_to_hdf5_parallel(args.particle_type, resolution=1024)
