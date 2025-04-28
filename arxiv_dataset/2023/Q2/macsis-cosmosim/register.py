import os
import numpy as np

Tcut_halogas = 1.e5  # Hot gas temperature threshold in K


class Macsis:
    name: str = 'MACSIS'
    cosma_repository: str = '/cosma5/data/dp004/dc-hens1/macsis/macsis_gas'
    output_dir: str = '/cosma6/data/dp004/dc-alta2/macsis_analysis'

    def __init__(self) -> None:

        # Sort halos by index number
        halos_list = os.listdir(self.cosma_repository)
        halos_list = [i for i in halos_list if i.startswith('halo')]
        halos_list.sort(key=lambda x: int(x[-4:]))

        # Load halos data directories
        self.halo_paths = [os.path.join(self.cosma_repository, i) for i in halos_list]
        self.num_zooms = len(self.halo_paths)

    def get_zoom(self, index: int):
        assert len(self.halo_paths) > 0
        try:
            directory_select = self.halo_paths[index]
        except IndexError as err:
            print((
                f"Trying to access zoom object with output index {index:d}, "
                f"but the maximum index available is {len(self.halo_paths) - 1:d}."
            ))
            raise err

        return Zoom(directory_select)


class Redshift(object):
    __slots__ = (
        'run_name',
        'scale_factor',
        'a',
        'redshift',
        'z',
        'snapshot_path',
        'catalogue_grouptab_path',
        'catalogue_subfindtab_path',
        'catalogue_particles_path',
    )

    run_name: str
    scale_factor: float
    a: float
    redshift: float
    z: float
    snapshot_path: str
    catalogue_grouptab_path: str
    catalogue_subfindtab_path: str
    catalogue_particles_path: str

    def __init__(self, info_dict: dict):
        for key in info_dict:
            setattr(self, key, info_dict[key])

        setattr(self, 'a', self.scale_factor)
        setattr(self, 'z', self.redshift)

    def __str__(self):
        return (
            f"Run name:                 {self.run_name}\n"
            f"Scale factor (a):         {self.scale_factor}\n"
            f"Redshift (z):             {self.redshift}\n"
            f"Snapshot file:            {self.snapshot_path}\n"
            f"Catalog group-tab file:   {self.catalogue_grouptab_path}\n"
            f"Catalog subfind-tab file: {self.catalogue_subfindtab_path}\n"
            f"Catalog particles file:   {self.catalogue_particles_path}"
        )


class Zoom(object):

    def __init__(self, run_directory: str) -> None:
        self.run_name = os.path.basename(run_directory)
        self.run_directory = run_directory
        self.scale_factors, self.redshifts = self.read_output_list()

        # Retrieve complete data paths to files
        self.snapshot_paths = []
        self.catalogue_grouptab_paths = []
        self.catalogue_subfindtab_paths = []
        self.catalogue_particles_paths = []

        for dir_output in os.listdir(os.path.join(run_directory, 'data')):
            path_output = os.path.join(run_directory, 'data', dir_output)

            # Retrieve snapshots file paths
            if dir_output.startswith('snapshot') and not dir_output.endswith('_023'):

                files = os.listdir(path_output)
                if len(files) == 1 and isinstance(files, list):
                    files = os.path.join(path_output, files[0])
                elif len(files) > 1:
                    files = tuple([os.path.join(path_output, file) for file in files])

                self.snapshot_paths.append(files)

            # Retrieve group_tab and subfind_tab files
            elif dir_output.startswith('groups'):

                files = os.listdir(path_output)
                for file in files:
                    if file.startswith('eagle_subfind_tab'):
                        self.catalogue_subfindtab_paths.append(os.path.join(path_output, file))
                    elif file.startswith('group_tab'):
                        self.catalogue_grouptab_paths.append(os.path.join(path_output, file))

            # Retrieve subfind particle data
            elif dir_output.startswith('particledata'):

                files = os.listdir(path_output)
                if len(files) == 1 and isinstance(files, list):
                    files = os.path.join(path_output, files[0])
                else:
                    files = tuple([os.path.join(path_output, file) for file in files])

                self.catalogue_particles_paths.append(files)

        assert len(self.scale_factors) == len(self.snapshot_paths), (
            f"[Halo {self.run_name}] {len(self.scale_factors)} != {len(self.snapshot_paths)}"
        )
        assert len(self.scale_factors) == len(self.catalogue_subfindtab_paths), (
            f"[Halo {self.run_name}] {len(self.scale_factors)} != {len(self.catalogue_subfindtab_paths)}"
        )
        assert len(self.scale_factors) == len(self.catalogue_grouptab_paths), (
            f"[Halo {self.run_name}] {len(self.scale_factors)} != {len(self.catalogue_grouptab_paths)}"
        )
        assert len(self.scale_factors) == len(self.catalogue_particles_paths), (
            f"[Halo {self.run_name}] {len(self.scale_factors)} != {len(self.catalogue_particles_paths)}"
        )

        # Sort redshift outputs by index
        sort_arg = dict(key=lambda x: int(x.split('.')[0][-3:]))
        self.snapshot_paths.sort(**sort_arg)
        self.catalogue_subfindtab_paths.sort(**sort_arg)
        self.catalogue_grouptab_paths.sort(**sort_arg)
        self.catalogue_particles_paths.sort(**sort_arg)

    def read_output_list(self):
        output_list_file = os.path.join(self.run_directory, 'output_list')
        scale_factors = np.genfromtxt(output_list_file)
        redshifts = 1 / scale_factors - 1
        return scale_factors, redshifts

    def get_redshift(self, index: int = -1):
        """
        To get z = 0 data promptly, specify index = -1. This
        selects the last output in the index list, which is the
        last redshift produced at runtime.

        :param index: int
            The integer index describing the output sequence.
        :return: Redshift instance
            The Redshift object contains fast-access absolute
            paths to the key files to read data from.
        """

        try:
            redshift_select = self.redshifts[index]
        except IndexError as err:
            print((
                f"Trying to access redshift with output index {index:d}, "
                f"but the maximum index available is {len(self.redshifts) - 1:d}."
            ))
            raise err

        redshift_info = dict()
        redshift_info['run_name'] = self.run_name
        redshift_info['scale_factor'] = self.scale_factors[index]
        redshift_info['redshift'] = redshift_select
        redshift_info['snapshot_path'] = self.snapshot_paths[index]
        redshift_info['catalogue_subfindtab_path'] = self.catalogue_subfindtab_paths[index]
        redshift_info['catalogue_grouptab_path'] = self.catalogue_grouptab_paths[index]
        redshift_info['catalogue_particles_path'] = self.catalogue_particles_paths[index]

        check_index = np.where(self.scale_factors == redshift_info['scale_factor'])[0][0]
        assert f"{check_index:03d}" in os.path.basename(redshift_info['snapshot_path'])
        assert f"{check_index:03d}" in os.path.basename(redshift_info['catalogue_grouptab_path'])
        assert f"{check_index:03d}" in os.path.basename(redshift_info['catalogue_subfindtab_path'])
        assert f"{check_index:03d}" in os.path.basename(redshift_info['catalogue_particles_path'])

        return Redshift(redshift_info)


if __name__ == "__main__":
    macsis = Macsis()
    for i in range(10):
        halo = macsis.get_zoom(i).get_redshift(-1)
        print(halo)
