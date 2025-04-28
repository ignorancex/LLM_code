import os
import sys
import unyt
import h5py
import numpy as np
import argparse
import pandas as pd
from mpi4py import MPI
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector

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


def mean_velocity(halo, particle_type: str = 'gas'):
    # Switch the type of angular momentum analysis
    if particle_type == 'gas':
        pt_number = '0'
    elif particle_type == 'dm':
        pt_number = '1'
    elif particle_type == 'stars':
        pt_number = '4'

    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot(f'PartType{pt_number}/Coordinates')
    if particle_type == 'dm':
        dm_particle_mass = data.read_header('MassTable')[1]
        masses = np.ones(len(coordinates), dtype=np.float) * dm_particle_mass
    else:
        masses = data.read_snapshot(f'PartType{pt_number}/Mass')
    velocities = data.read_snapshot(f'PartType{pt_number}/Velocity')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1] / data.a
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / data.a

    # Select ionised hot gas
    if particle_type == 'gas':
        temperatures = data.read_snapshot(f'PartType{pt_number}/Temperature')
        temperature_cut = np.where(temperatures > 1.e5)[0]
        coordinates = coordinates[temperature_cut]
        masses = masses[temperature_cut]
        velocities = velocities[temperature_cut]

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

    return mean_velocity_r500


def velocity_dispersion(halo, particle_type: str = 'gas'):
    # Switch the type of angular momentum analysis
    if particle_type == 'gas':
        pt_number = '0'
    elif particle_type == 'dm':
        pt_number = '1'
    elif particle_type == 'stars':
        pt_number = '4'

    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot(f'PartType{pt_number}/Coordinates')
    if particle_type == 'dm':
        dm_particle_mass = data.read_header('MassTable')[1]
        masses = np.ones(len(coordinates), dtype=np.float) * dm_particle_mass
    else:
        masses = data.read_snapshot(f'PartType{pt_number}/Mass')
    velocities = data.read_snapshot(f'PartType{pt_number}/Velocity')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1] / data.a
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / data.a

    # Select ionised hot gas
    if particle_type == 'gas':
        temperatures = data.read_snapshot(f'PartType{pt_number}/Temperature')
        temperature_cut = np.where(temperatures > 1.e5)[0]
        coordinates = coordinates[temperature_cut]
        masses = masses[temperature_cut]
        velocities = velocities[temperature_cut]

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

    velocities_rest_frame = velocities.copy()
    velocities_rest_frame[:, 0] -= mean_velocity_r500[0]
    velocities_rest_frame[:, 1] -= mean_velocity_r500[1]
    velocities_rest_frame[:, 2] -= mean_velocity_r500[2]

    velocity_dispersion_r500 = np.sqrt(
        np.sum(
            np.power(velocities_rest_frame * masses[:, None], 2.),
            axis=0
        ) / np.sum(masses[r500_mask])
    )

    return velocity_dispersion_r500


def angular_momentum(halo, particle_type: str = 'gas'):
    # Switch the type of angular momentum analysis
    if particle_type == 'gas':
        pt_number = '0'
    elif particle_type == 'dm':
        pt_number = '1'
    elif particle_type == 'stars':
        pt_number = '4'

    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot(f'PartType{pt_number}/Coordinates')
    if particle_type == 'dm':
        dm_particle_mass = data.read_header('MassTable')[1]
        masses = np.ones(len(coordinates), dtype=np.float) * dm_particle_mass
    else:
        masses = data.read_snapshot(f'PartType{pt_number}/Mass')
    velocities = data.read_snapshot(f'PartType{pt_number}/Velocity')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1] / data.a
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / data.a

    # Select ionised hot gas
    if particle_type == 'gas':
        temperatures = data.read_snapshot(f'PartType{pt_number}/Temperature')
        temperature_cut = np.where(temperatures > 1.e5)[0]
        coordinates = coordinates[temperature_cut]
        masses = masses[temperature_cut]
        velocities = velocities[temperature_cut]

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

    return angular_momentum_r500


def component_mass_r500(halo, particle_type: str = 'gas'):
    # Switch the type of angular momentum analysis
    if particle_type == 'gas':
        pt_number = '0'
    elif particle_type == 'dm':
        pt_number = '1'
    elif particle_type == 'stars':
        pt_number = '4'

    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot(f'PartType{pt_number}/Coordinates')
    if particle_type == 'dm':
        dm_particle_mass = data.read_header('MassTable')[1]
        masses = np.ones(len(coordinates), dtype=np.float) * dm_particle_mass
    else:
        masses = data.read_snapshot(f'PartType{pt_number}/Mass')
    velocities = data.read_snapshot(f'PartType{pt_number}/Velocity')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1] / data.a
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / data.a

    # Select ionised hot gas
    if particle_type == 'gas':
        temperatures = data.read_snapshot(f'PartType{pt_number}/Temperature')
        temperature_cut = np.where(temperatures > 1.e5)[0]
        coordinates = coordinates[temperature_cut]
        masses = masses[temperature_cut]
        velocities = velocities[temperature_cut]

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

    total_mass_500 = np.sum(masses[r500_mask])

    return total_mass_500


def dump_to_hdf5_parallel():
    macsis = Macsis()
    with h5py.File(
            f'{macsis.output_dir}/properties_{args.redshift_index:03d}.hdf5', 'w',
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
        sort_keys = np.argsort(np.array([int(z.run_name[-4:]) for z in zoom_handles]))
        zoom_handles = zoom_handles[sort_keys]

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

        halo_number = f.create_dataset("halo_number", (macsis.num_zooms,), dtype=np.uint16)
        m_500crit = f.create_dataset("m_500crit", (macsis.num_zooms,), dtype=np.float)
        r_500crit = f.create_dataset("r_500crit", (macsis.num_zooms,), dtype=np.float)

        hot_gas_mass_500crit = f.create_dataset("hot_gas_mass_500crit", (macsis.num_zooms,), dtype=np.float)
        dark_matter_mass_500crit = f.create_dataset("dark_matter_mass_500crit", (macsis.num_zooms,), dtype=np.float)
        star_mass_500crit = f.create_dataset("star_mass_500crit", (macsis.num_zooms,), dtype=np.float)

        angular_momentum_hotgas_r500 = f.create_dataset(
            "angular_momentum_hotgas_r500", (macsis.num_zooms, 3), dtype=np.float
        )
        angular_momentum_dark_matter_r500 = f.create_dataset(
            "angular_momentum_dark_matter_r500", (macsis.num_zooms, 3), dtype=np.float
        )
        angular_momentum_stars_r500 = f.create_dataset(
            "angular_momentum_stars_r500", (macsis.num_zooms, 3), dtype=np.float
        )

        mean_velocity_hotgas_r500 = f.create_dataset(
            "mean_velocity_hotgas_r500", (macsis.num_zooms, 3), dtype=np.float
        )
        mean_velocity_dark_matter_r500 = f.create_dataset(
            "mean_velocity_dark_matter_r500", (macsis.num_zooms, 3), dtype=np.float
        )
        mean_velocity_stars_r500 = f.create_dataset(
            "mean_velocity_stars_r500", (macsis.num_zooms, 3), dtype=np.float
        )

        velocity_dispersion_hotgas_r500 = f.create_dataset(
            "velocity_dispersion_hotgas_r500", (macsis.num_zooms, 3), dtype=np.float
        )
        velocity_dispersion_dark_matter_r500 = f.create_dataset(
            "velocity_dispersion_dark_matter_r500", (macsis.num_zooms, 3), dtype=np.float
        )
        velocity_dispersion_stars_r500 = f.create_dataset(
            "velocity_dispersion_stars_r500", (macsis.num_zooms, 3), dtype=np.float
        )

        # Data assignment can be done through independent operations
        for zoom_id, data_handle in enumerate(zoom_handles):
            if zoom_id % num_processes == rank:
                print((
                    f"Rank {rank:03d} processing halo ({zoom_id:03d}/{macsis.num_zooms - 1}) | "
                    f"MACSIS name: {data_handle.run_name}"
                ))

                halo_number[zoom_id] = int(data_handle.run_name[-4:])
                m_500crit[zoom_id] = MacsisDataset(data_handle).read_catalogue_subfindtab('FOF/Group_M_Crit500')[1]
                r_500crit[zoom_id] = MacsisDataset(data_handle).read_catalogue_subfindtab('FOF/Group_R_Crit500')[1] / MacsisDataset(data_handle).a

                hot_gas_mass_500crit[zoom_id] = component_mass_r500(data_handle, particle_type='gas')
                dark_matter_mass_500crit[zoom_id] = component_mass_r500(data_handle, particle_type='dm')
                star_mass_500crit[zoom_id] = component_mass_r500(data_handle, particle_type='stars')

                angular_momentum_hotgas_r500[zoom_id] = angular_momentum(data_handle, particle_type='gas')
                angular_momentum_dark_matter_r500[zoom_id] = angular_momentum(data_handle, particle_type='dm')
                angular_momentum_stars_r500[zoom_id] = angular_momentum(data_handle, particle_type='stars')

                mean_velocity_hotgas_r500[zoom_id] = mean_velocity(data_handle, particle_type='gas')
                mean_velocity_dark_matter_r500[zoom_id] = mean_velocity(data_handle, particle_type='dm')
                mean_velocity_stars_r500[zoom_id] = mean_velocity(data_handle, particle_type='stars')

                velocity_dispersion_hotgas_r500[zoom_id] = velocity_dispersion(data_handle, particle_type='gas')
                velocity_dispersion_dark_matter_r500[zoom_id] = velocity_dispersion(data_handle, particle_type='dm')
                velocity_dispersion_stars_r500[zoom_id] = velocity_dispersion(data_handle, particle_type='stars')


if __name__ == "__main__":
    dump_to_hdf5_parallel()
