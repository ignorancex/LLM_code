from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from glob import glob
import numpy as np
import os
import six

# debugging
from icecream import ic
import sys

from HBTReader import HBTReader


class BaseSimulation:
    """
    This class is imported by external objects such as Subhalo or
    Track. For now, it only provides the capability to load a
    Simulation object

    """

    def __init__(self, sim):
        # initialize simulation and reader. This cannot be modified
        # within the same instance
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        self.reader = HBTReader(self.sim.path)

    def load_subhalos(self, isnap=None, z=None, selection=None, **kwargs):
        """Wrapper to initialize a Subhalos object from the simulation"""
        if isnap is None:
            if z is None:
                raise ValueError("must provide either isnap or z")
            isnap = self.snapshot_index_from_redshift(z)
        subs = self.reader.LoadSubhalos(isnap, selection=selection)
        subs = Subhalos(subs, self, isnap, **kwargs)
        return subs


class Simulation(object):
    """Simulation object

    **Note**: all snapshot-related attributes include only those actually
    found in the directory tree, except for ``self.snapshot_list``, which
    includes all entries exactly as listed in ``snapshotlist.txt``
    """

    def __init__(self, label, root):
        self.label = label
        self._root = root
        self._cosmology = None
        self._family = None
        self._formatted_name = None
        self._reader = None
        self._redshifts = None
        self._scale_factor_dict = None
        self._scale_factors = None
        self._snapshots = None
        self._snapshot_files = None
        self._snapshot_list = None
        self._snapshot_mask = None
        self._snapshot_indices = None
        self._t_lookback = None
        self._virial_snapshots = None
        self.initialize_tree()
        self.history_file = os.path.join(self.data_path, "history", "history.h5")
        self.has_history = os.path.isfile(self.history_file)
        print(f"Loaded {self.formatted_name} from {self.path}")

    ### attributes ###

    @property
    def cosmology(self):
        if self._cosmology is None:
            cosmo = {
                "apostle": FlatLambdaCDM(H0=70.4, Om0=0.272, Ob0=0.0455),
                "eagle": FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0=0.04825),
            }
            self._cosmology = cosmo[self.family]
        return self._cosmology

    @property
    def data_path(self):
        return os.path.join("data", self.name.replace("/", "_"))

    @property
    def family(self):
        if self._family is None:
            self._family = self.name.split("/")[0]
        return self._family

    @property
    def formatted_name(self):
        if self._formatted_name is None:
            _name = self.name.split("/")
            if self.name.startswith("apostle"):
                _name[0] = _name[0].capitalize()
            elif self.name.startswith("eagle"):
                _name[0] = _name[0].upper()
            self._formatted_name = "/".join([_name[0], _name[1].upper()])
        return self._formatted_name

    @property
    def infall_file(self):
        return os.path.join("data", self.name.replace("/", "_"), "infall.txt")

    @property
    def lookback_dict(self):
        return {
            i: self.cosmology.lookback_time(zi) for i, zi in self.redshift_dict.items()
        }

    @property
    def mass_columns(self):
        return [
            "LastMaxMass",
            "Mbound",
            "Mstar",
            "Mdm",
            "Mgas",
            "Mass",
            "M200",
            "M200Mean",
            "MVir",
        ]

    @property
    def masstypes(self):
        return np.array(["gas", "dm", "disk", "bulge", "stars", "bh"])

    @property
    def masstype_pandas_columns(self):
        return np.array(
            ["MboundType{0}".format(self._masstype_index(i)) for i in self.masstypes]
        )

    @property
    def masstype_labels(self):
        return np.array(["gas", "DM", "disk", "bulge", "stars", "BH"])

    @property
    def _mapping(self):
        return {
            "LR": "apostle/V1_LR",
            "MR": "apostle/V1_MR",
            "HR": "apostle/V1_HR",
            "L25": "eagle/L0025N0376",
            "L100": "eagle/L0100N1504",
        }

    @property
    def name(self):
        return self._mapping[self.label]

    @property
    def path(self):
        return os.path.join(self.root, self.name, "subcat")

    @property
    def plot_path(self):
        return os.path.join("plots", self.name.replace("/", "_"))

    @property
    def reader(self):
        if self._reader is None:
            self._reader = HBTReader(self.path)
        return self._reader

    @property
    def redshift_dict(self):
        return {i: 1 / a - 1 for i, a in self.scale_factor_dict.items()}

    @property
    def redshifts(self):
        if self._redshifts is None:
            self._redshifts = 1 / self.scale_factors - 1
        return self._redshifts

    @property
    def root(self):
        return self._root

    @property
    def scale_factor_dict(self):
        if self._scale_factor_dict is None:
            self._scale_factor_dict = self.reader.GetScaleFactorDict()
        return self._scale_factor_dict

    @property
    def scale_factors(self):
        if self._scale_factors is None:
            self._scale_factors = np.array(
                [a for i, a in self.scale_factor_dict.items()]
            )
        return self._scale_factors

    @property
    def snapshot_files(self):
        if self._snapshot_files is None:
            self._snapshot_files = sorted(glob(os.path.join(self.path, "SubSnap*")))
        return self._snapshot_files

    @property
    def snapshot_list(self):
        if self._snapshot_list is None:
            # self._snapshot_list = np.loadtxt(
            #     os.path.join(self.path, 'snapshotlist.txt'), dtype=str)
            self._snapshot_list = np.array(
                [
                    file.split("/")[-1].replace(".hdf5", "").split("_")[1]
                    for file in self.snapshot_files
                ],
                dtype=int,
            )
        return self._snapshot_list

    @property
    def snapshot_mask(self):
        if self._snapshot_mask is None:
            self._snapshot_mask = np.array(
                [
                    (self.get_snapshot_file_from_index(i) in self.snapshot_files)
                    for i in range(self.snapshot_list.size)
                ]
            )
        return self._snapshot_mask

    @property
    def snapshots(self):
        if self._snapshots is None:
            snaps = [
                i.split("/")[-1].split("_")[1].split(".")[0]
                for i in self.snapshot_files
            ]
            self._snapshots = np.array(snaps, dtype=np.uint16)
        return self._snapshots

    @property
    def snapshot_indices(self):
        if self._snapshot_indices is None:
            self._snapshot_indices = np.array(
                [
                    i
                    for i in range(self.snapshot_list.size)
                    if self.get_snapshot_file_from_index(i) in self.snapshot_files
                ]
            )
        return self._snapshot_indices

    @property
    def t_lookback(self):
        if self._t_lookback is None:
            self._t_lookback = self.cosmology.lookback_time(self.redshifts)
        return self._t_lookback

    @property
    def virial_path(self):
        return os.path.join(self.path, "HaloSize")

    @property
    def virial_snapshots(self):
        if self._virial_snapshots is None:
            lsdir = glob(os.path.join(self.virial_path, "HaloSize_*.hdf5"))
            self._virial_snapshots = np.sort(
                [
                    int(i.split("/")[-1].split(".")[0].split("_")[1])
                    for i in glob(os.path.join(self.virial_path, "HaloSize_*.hdf5"))
                ]
            )
        return self._virial_snapshots

    ### methods ###

    def get_snapshot_file_from_index(self, idx):
        return os.path.join(self.path, f"SubSnap_{idx:03d}.hdf5")

    def mass_to_sim_h(self, m, h, mtype="total", log=False):
        """Correction to mass measurement to account for different h

        the result of this is to be *added* to the reported mass
        """
        hcor = h / self.cosmology.h
        if mtype == "stars":
            hcor = hcor**2
        if log:
            return m + np.log10(hcor)
        return m * hcor

    def masstype_pandas_column(self, mtype):
        """Name of the column containing requested mass type

        Parameters
        ----------
        mtype : str
            mass type. See ``self.masstypes``

        Returns
        -------
        column : str
            column name
        """
        return self.masstype_pandas_columns[self._masstype_index(mtype)]

    # THIS CANNOT WORK DUE TO CIRCULAR IMPORT
    # def load_subhalos(self, isnap=None, z=None, selection=None, **kwargs):
    #     """Wrapper to initialize a Subhalos object from the simulation"""
    #     reader = HBTReader(self.path)
    #     if isnap is None:
    #         if z is None:
    #             raise ValueError('must provide either isnap or z')
    #         isnap = self.snapshot_index_from_redshift(z)
    #     subs = reader.LoadSubhalos(isnap, selection=selection)
    #     subs = Subhalos(subs, self, isnap, **kwargs)
    #     return subs

    def redshift(self, isnap: int):
        """Redshift given snapshot index

        Parameters
        ----------
        isnap : int
            snapshot number (not index!)
        """
        if isnap < 0:
            isnap = self.snapshots[-1] + 1 + isnap
        if isnap in self.scale_factor_dict:
            return 1 / self.scale_factor_dict.get(isnap) - 1
        else:
            # try:
            #     return self.redshifts[self.snapshots == isnap][0]
            # except IndexError:
            msg = f"snapshot {isnap} not found in {self.name}"
            raise IndexError(msg)

    def snapshot_index(self, snap):
        """Given a snapshot number, return the index to which it
        corresponds

        Parameters
        ----------
        snap : int
            snapshot number

        Returns
        -------
        isnap : int
            snapshot index
        """
        return self.snapshot_indices[self.snapshots == snap][0]

    def snapshot_index_from_redshift(self, z, return_zsnap=False):
        """Return snapshot index closest to a given redshift

        Parameters
        ----------
        redshift : float

        Returns
        -------
        isnap : int
            snapshot index
        zsnap : float
            Only returned if ``return_zsnap==True``. Redshift of the
            selected snapshot
        """
        j = np.argmin(np.abs(self.redshifts - z))
        isnap = self.snapshot_indices[j]
        ic(j, self.redshifts[j], isnap, self.redshift(isnap))
        if return_zsnap:
            zsnap = self.redshift(isnap)
            return isnap, zsnap
        return isnap

    def initialize_tree(self):
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)
        return

    def masslabel(self, latex=True, suffix=None, unit="Msun", **kwargs):
        """Return label of a given mass type

        Paramters
        ---------
        mtype : str, optional
            name of the mass type. Must be one of ``self.masstypes``, or
            either 'total' or 'host'
        index : int, optional
            index of the mass type. An index of -1 corresponds to total
            mass. Otherwise see ``self.masstypes``
        latex : bool, optional
            whether the label should be in latex or plain text format
        unit : str, optional (NOT IMPLEMENTED)
            which unit to include. Can be 'Msun' or a latex-formatted
            string
        suffix : str, optional
            a suffix to add to the subscript, separated by a comma if
            ``latex=True`` and by an underscore otherwise

        Returns
        -------
        label : str
            mass label in latex format
        """
        subscript = self.masstype(**kwargs)
        if latex:
            if suffix:
                subscript = "{0},{1}".format(subscript, suffix)
            label = rf"M_\mathrm{{{subscript}}}"
        else:
            if suffix:
                subscript = "{0}_{1}".format(subscript, suffix)
            label = f"M{subscript.lower()}"
        return label

    def masstype(self, mtype=None, index=None):
        assert (
            mtype is not None or index is not None
        ), "must provide either ``mtype`` or ``index``"
        if mtype is not None:
            if mtype.lower() in ("total", "mbound"):
                return "total"
            if mtype.lower() == "host":
                return "host"
            return self.masstype_labels[self._masstype_index(mtype)]
        if index == -1:
            return "total"
        return self.masstype_labels[index]

    ### private methods ###

    def _masstype_index(self, mtype):
        rng = np.arange(self.masstypes.size, dtype=int)
        return rng[self.masstypes == mtype.lower()][0]
