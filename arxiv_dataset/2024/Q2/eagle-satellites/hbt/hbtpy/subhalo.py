from __future__ import absolute_import, division, print_function, unicode_literals
from cmath import log

from astropy.io import fits
from astropy.table import Table
import h5py
from itertools import combinations, count
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import append_fields
import os
import pandas as pd

# import cudf as pd
from scipy.stats import binned_statistic as binstat
import six
from time import time
import warnings

# debugging
from icecream import ic
import sys

from HBTReader import HBTReader

from .simulation import BaseSimulation


class BaseDataSet:
    def __init__(self, catalog, as_dataframe=True):
        self._catalog = catalog
        self._as_dataframe = as_dataframe
        self._none_value = 999

    ### private properties ###

    @property
    def _range(self):
        # return np.arange(self.catalog.size, dtype=int)
        return np.arange(self.catalog[self.colnames[0]].size, dtype=int)

    ### public properties ###

    @property
    def as_dataframe(self):
        return self._as_dataframe

    @as_dataframe.setter
    def as_dataframe(self, as_df):
        assert isinstance(as_df, bool), "as_dataframe must be boolean"
        self._as_dataframe = as_df

    @property
    def catalog(self):
        if self.as_dataframe and not isinstance(self._catalog, pd.DataFrame):
            self._catalog = self.DataFrame(self._catalog)
        elif not self.as_dataframe and isinstance(self._catalog, pd.DataFrame):
            ic(self._catalog.dtype.names)
            self._catalog = self._catalog.to_records()
        return self._catalog

    # @catalog.setter
    # def catalog(self, cat):
    # self._catalog = cat

    @property
    def colnames(self):
        if self.as_dataframe:
            colnames = self.catalog.columns
        else:
            colnames = self.catalog.dtype.names
        return np.array(colnames, dtype=str)

    @property
    def columns(self):
        return self.colnames

    @property
    def size(self):
        return self.catalog[self.colnames[0]].size

    @property
    def shape(self):
        return (len(self.colnames), self.size)

    ### methods ###

    def DataFrame(self, recarray):
        """Return a ``pandas.DataFrame`` object"""
        df = {}
        for dtype in recarray.dtype.descr:
            name = dtype[0]
            if len(recarray[name].shape) == 1:
                df[name] = recarray[name]
            else:
                for i in range(recarray[name].shape[1]):
                    df["{0}{1}".format(name, i)] = recarray[name][:, i]
        return pd.DataFrame(df)

    def groupby(self, *args, **kwargs):
        """Wrapper for ``pandas.DataFrame.groupby``"""
        return self.catalog.groupby(*args, **kwargs)

    def merge(
        self, right, left_cols=None, right_cols=None, in_place=True, *args, **kwargs
    ):
        """Wrapper for ``pandas.DataFrame.merge``.
        Returns a ``Subhalos`` object
        ``right`` can be a ``Subhalos`` object or a ``DataFrame``
        """
        if isinstance(right, Subhalos):
            right = right.catalog
        if left_cols is not None:
            left = self.__getitem__(left_cols)
        else:
            left = self.catalog
        if right_cols is not None:
            right = right[right_cols]
        new = left.merge(right, *args, **kwargs)
        if in_place:
            self._catalog = new
        else:
            return Subhalos(
                new, self.sim, self.isnap, load_any=False, exclude_non_FoF=False
            )

    def sort(self, column, inplace=False, **kwargs):
        """Sort catalog by column or columns

        ``kwargs`` are passed to ``pd.DataFrame.sort_values"""
        if not self.as_dataframe:
            self._catalog = self.DataFrame(self._catalog)
        c = self._catalog.sort_values(column, **kwargs)
        if inplace:
            self._catalog = c
        if not self.as_dataframe:
            self._catalog = self._catalg.to_records()
        if not inplace:
            return c


class BaseSubhalo(BaseDataSet):
    """BaseSubhalo class"""

    def __init__(self, catalog, sim, as_dataframe=True, pvref="MostBound"):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``Subhalos.data``
        sim : ``Simulation``
        """
        assert pvref in ("Average", "MostBound")
        super(BaseSubhalo, self).__init__(catalog, as_dataframe=as_dataframe)
        BaseSimulation.__init__(self, sim)
        self.cosmology = self.sim.cosmology
        self.as_dataframe = as_dataframe
        self.pvref = pvref
        self._update_mass_columns()
        # self._update_position_columns()

    def __getitem__(self, col):
        cols = self.colnames
        ic(col, type(col))
        if isinstance(col, np.ndarray):
            ic(col.dtype)
        if (
            (isinstance(col, str) and col in cols)
            or (isinstance(col, np.ndarray) and (col.dtype == bool))
            or (isinstance(col, pd.Series) and (col.dtype == bool))
        ):
            return self.catalog[col]
        # conveninence names
        colmap = {
            "Mgas": "MboundType0",
            "Mdm": "MboundType1",
            "Mstar": "MboundType4",
            "Ngas": "NboundType0",
            "Ndm": "NboundType1",
            "Nstar": "NboundType4",
        }
        if not isinstance(col, str):
            X = pd.DataFrame()
            for c in col:
                X[c] = self.__getitem__(c)
            return X
        if "/" in col:
            col = col.split("/")
            if len(col) != 2:
                raise ValueError(
                    "Can only calculate ratio of two columns," f" received {col}"
                )
            return self.__getitem__(col[0]) / self.__getitem__(col[1])
        if "-" in col:
            col = col.split("-")
            if len(col) != 2:
                raise ValueError(
                    "can only calculate difference of two columns," f" received {col}"
                )
            return self.__getitem__(col[0]) - self.__getitem__(col[1])
        if col in cols:
            return self.catalog[col]
        if col in colmap:
            return self.catalog[colmap[col]]
        if col in ("Mtot", "Mtotal"):
            return self._get_total_mass()
        raise KeyError(f"column {col} not found")

    def _get_total_mass(self):
        x = self.catalog["M200Mean"]
        mask = self.satellite_mask
        x.loc[mask] = self.catalog["Mbound"][mask]
        return x

    ### hidden properties ###

    @property
    def _valid_axes(self):
        return ("", "0", "1", "2", "01", "02", "12")

    ### properties ###

    @property
    def centrals(self):
        return self.catalog[self.central_mask]

    @property
    def central_idx(self):
        return self._range[self._central_mask]

    @property
    def central_mask(self):
        return self.catalog["Rank"] == 0

    @property
    def galaxies(self):
        return self.catalog[self.galaxy_mask]

    @property
    def galaxy_idx(self):
        return self._range[self._galaxy_mask]

    @property
    def galaxy_mask(self):
        return self.catalog["Mstar"] > 0

    @property
    def n(self):
        return self.catalog["Rank"].size

    @property
    def ncen(self):
        return self.central_mask.sum()

    @property
    def nsat(self):
        return self.satellite_mask.sum()

    @property
    def orphan(self):
        """boolean mask, True if object is orphan"""
        return self.catalog["Nbound"] == 1

    @property
    def satellites(self):
        return self.catalog.loc[self.satellite_mask]

    @property
    def satellite_idx(self):
        return self._range[self.satellite_mask]

    @property
    def satellite_mask(self):
        return self.catalog["Rank"] > 0

    @property
    def shape(self):
        return self.catalog.shape

    @property
    def Mbound(self):
        return self.catalog["Mbound"]

    @property
    def MboundType(self):
        if self.as_dataframe:
            cols = ["MboundType{0}".format(i) for i in range(6)]
            return np.array(self.catalog[cols])
        return self.catalog["MboundType"]

    @property
    def mvir(self):
        return self.catalog["MVir"]

    @property
    def Nbound(self):
        return self.catalog["Nbound"]

    @property
    def NboundType(self):
        if self.as_dataframe:
            cols = ["NboundType{0}".format(i) for i in range(6)]
            return np.array(self.catalog[cols])
        return self.catalog["NboundType"]

    ### hidden methods ###

    def _get_ax1d_label(self, ax):
        try:
            ax = int(ax)
        except ValueError as err:
            msg = "ax must be an int, 0 <= ax <= 2"
            raise ValueError(err)
        assert ax in [0, 1, 2]
        return "xyz"[[0, 1, 2].index(ax)]

    def _update_mass_columns(self, history_only=False):
        cols = (
            self.catalog.columns
            if isinstance(self.catalog, pd.DataFrame)
            else self.catalog.dtype.names
        )
        for col in cols:
            if np.any(
                [
                    i in col and "SnapshotIndex" not in col
                    for i in (
                        "Mbound",
                        "MboundType",
                        "Mgas",
                        "Mdm",
                        "Mstar",
                        "LastMaxMass",
                    )
                ]
            ):
                # it's easier to add other exceptions here
                if (
                    "Depth" in col
                    or "time" in col
                    or "isnap" in col
                    or col.split(":")[-1] == "z"
                ):
                    continue
                # in this case only divide by h - units are already Msun
                if history_only and "history" in col:
                    self.catalog[col] = self.catalog[col] / self.sim.cosmology.h
                    continue
                if self.catalog[col].max() < 1e6:
                    self.catalog[col] = 1e10 * self.catalog[col] / self.sim.cosmology.h

    def _update_position_columns(self, history_only=False):
        for col in self.colnames:
            # distances will already be using these updated positions
            if np.any(
                [
                    i in col and "SnapshotIndex" not in col
                    for i in ("Position", "R200", "RVir", "RHalf", "Rmax")
                ]
            ):  # it's easier to add other exceptions here
                if (
                    "Depth" in col
                    or "time" in col
                    or "isnap" in col
                    or col.split(":")[-1] == "z"
                ):
                    continue
                if self.catalog[col].max() < 70:
                    self.catalog[col] = self.catalog[col] / self.sim.cosmology.h

    ### methods ###

    def mass(self, mtype=None, index=None):
        assert (
            mtype is not None or index is not None
        ), "must provide either ``mtype`` or ``index``"
        if mtype is not None:
            if mtype.lower() in ("total", "mbound"):
                return self.Mbound
            return self.MboundType[:, self.sim._masstype_index(mtype)]
        if index == -1:
            return np.array(self.Mbound)
        return self.MboundType[:, index]

    def nbound(self, mtype=None, index=None):
        assert (
            mtype is not None or index is not None
        ), "must provide either ``mtype`` or ``index``"
        if mtype is not None:
            if mtype.lower() in ("total", "nbound"):
                return self.Nbound
            return self.NboundType[:, self.sim._masstype_index(mtype)]
        if index == -1:
            return np.array(self.Nbound)
        return self.NboundType[:, index]

    ## easy access to positions, distances and velocities ##

    def column(self, value, ax):
        # consider moving `ref` to a class attribute
        value = value.lower()
        acceptable = ("distance", "position", "velocity")
        assert value in acceptable, "`value` must be one of {0}".format(acceptable)
        if value == "distance":
            return dcol(ax)
        if value == "position":
            return pcol(ax)
        if value == "velocity":
            return vcol(ax)

    def dcol(self, ax="", frame="Physical"):
        """Distance column name

        Parameters
        ----------
        ax : {'0', '1', '2', '01', '02', '12'}, optional
            distance axis or plane. If not given, will return
            3-dimenional distance column

        Returns
        -------
        dcol : str
            distance column name
        """
        assert ax in ("", "0", "1", "2", "01", "02", "12")
        return "{2}{0}Distance{1}".format(self.pvref, ax, frame.capitalize())

    def dcols(self, ndim=1, frame="Physical"):
        """Distance column names

        Parameters
        ----------
        ndim : int {1,2, 3}, optional
            number of dimensions over which distances are desired
        """
        assert ndim in (1, 2, 3)
        if ndim == 3:
            return self.dcol(frame=frame)
        return [
            self.dcol("".join(plane), frame=frame)
            for plane in combinations("012", ndim)
        ]

    def dlabel(self, ax=""):
        ax = str(ax)
        assert ax in self._valid_axes, "ax must be one of {0}".format(self._valid_axes)
        if ax != 0 and not ax:
            return "r_\mathrm{3D}"
        return "r_{{{0}}}".format("".join([self._get_ax1d_label(i) for i in ax]))

    def distance(self, ax=""):
        return self.catalog[self.dcol(ax=ax)]

    def pcol(self, ax, frame="Comoving"):
        assert int(ax) in [0, 1, 2]
        return "{2}{0}Position{1}".format(self.pvref, ax, frame)

    def pcols(self, frame="Comoving"):
        """All position column names"""
        return [self.pcol(i, frame) for i in range(3)]

    def plabel(self, ax):
        return self._get_ax1d_label(ax)

    def position(self, ax):
        return self.catalog[self.pcol(ax)]

    def scol(self, ax=""):
        assert not ax or int(ax) in [0, 1, 2]
        return "Physical{0}HostVelocityDispersion{1}".format(self.pvref, ax)

    def scols1d(self):
        return [self.scol(i) for i in range(3)]

    def slabel(self, ax=""):
        assert not ax or int(ax) in [0, 1, 2]
        if ax != 0 and not ax:
            return r"\sigma_\mathrm{3D}"
        return r"\sigma_{0}".format(self._get_ax1d_label(ax))

    def sigma(self, ax=""):
        """Velocity dispersion"""
        return self.catalog[self.scol(ax)]

    def vcol(self, ax="", peculiar=True):
        assert not ax or int(ax) in [0, 1, 2]
        if peculiar:
            return "Physical{0}PeculiarVelocity{1}".format(self.pvref, ax)
        return "Physical{0}Velocity{1}".format(self.pvref, ax)

    def vcols1d(self):
        return [self.vcol(i) for i in range(3)]

    def vlabel(self, ax):
        assert not ax or int(ax) in [0, 1, 2]
        if ax != 0 and not ax:
            return r"v_\mathrm{3D}"
        return r"v_{0}".format(self._get_ax1d_label(ax))

    def velocity(self, ax=""):
        return self.catalog[self.vcol(ax=ax)]


class HostHalos(BaseDataSet, BaseSimulation):
    """Class containing virial quantities of host halos"""

    def __init__(self, sim, isnap, as_dataframe=True, force_isnap=False):
        """

        if `force_isnap` is True, a ValueError will be raised if
        virial quantities do not exist for snapshot `isnap`. Otherwise,
        the nearest snapshot will be used (raises UserWarning).

        """
        BaseSimulation.__init__(self, sim)
        # alias
        self.available_snapshots = self.sim.virial_snapshots
        self.force_isnap = force_isnap
        self.isnap = self._set_isnap(isnap)
        self._filename = None
        catalog = h5py.File(self.filename, "r")["HostHalos"]
        super(HostHalos, self).__init__(catalog, as_dataframe=as_dataframe)

    @property
    def filename(self):
        if self._filename is None:
            self._filename = os.path.join(
                self.sim.path, "HaloSize", f"HaloSize_{self.isnap:03d}.hdf5"
            )
        return self._filename

    ### private methods

    def _set_isnap(self, isnap):
        if isnap < 0:
            isnap = self.sim.snapshots[isnap]
        if isnap not in self.available_snapshots:
            if self.force_isnap:
                msg = "Halo sizes not present for snapshot {0}".format(isnap)
                raise ValueError(msg)
            j = np.argmin(abs(isnap - self.available_snapshots))
            isnap_used = self.available_snapshots[j]
            warn = (
                f"Snapshot {isnap} does not have halo information. Using"
                f" snapshot {isnap_used} instead."
            )
            warnings.warn(warn)
            isnap = isnap_used
        return isnap


class Subhalos(BaseSubhalo):
    """Class to manage a sample of subhalos at a given snapshot

    An object can be indexed by column name or by the ratio or
    difference of two columns:
    ```
    sub = Subhalos(*args, **kwargs)
    mass_ratio = sub['Mbound/Mstar']
    time_diff = sub['history:first_infall:time-history:last_infall:time']
    ```
    This is useful for automatic pipelines as it is not
    necessary to index the table twice when one wants one
    of these operations
    """

    def __init__(
        self,
        catalog,
        sim,
        isnap=None,
        load_any=True,
        logMmin=9,
        logMstar_min=9,
        logM200Mean_min=12,
        exclude_non_FoF=True,
        as_dataframe=True,
        load_hosts=True,
        load_distances=True,
        load_velocities=True,
        load_history=True,
        verbose_when_loading=True,
    ):
        """
        Parameters
        ----------
        catalog : output of ``reader.LoadSubhalos``
            subhalo sample, defined at a specific snapshot
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)
        isnap : ``int``
            snapshot index at which the subhalos were retrieved.
            Unfortunately this has to be given by hand for now.

        Optional parameters
        -------------------
        logMmin : float
            minimum subhalo mass
        logMstar_min : float
            minimum stellar mass
        logM200Mean_min : float
            minimum host halo mass. Note that this requires
            ``load_hosts=True``
        exclude_non_FoF : bool
            whether to exclude all objects not part of any FoF halo
            (i.e., those with HostHaloId=-1). Some attributes or
            methods may not work if set to False.
        as_dataframe : bool
            whether to return a ``pd.DataFrame`` or a ``np.recarray``
        load_hosts, load_distances, load_velocities : bool
            whether to load host information. If the first one is
            ``True``, then whether to load cluster-centric distances
            and peculiar velocities
        load_history : bool
            whether to load the histories of each subhalo, as stored
            in ``{self.sim.data_path}/history/history.h5``. These
            will be loaded as additional columns in ``self.catalog``
        """
        bool_ = (bool, np.bool_)
        assert isinstance(as_dataframe, bool_)
        assert isinstance(load_hosts, bool_)
        assert isinstance(load_distances, bool_)
        assert isinstance(load_velocities, bool_)
        super(Subhalos, self).__init__(catalog, sim, as_dataframe=as_dataframe)
        if not load_any:
            load_hosts = False
            load_distances = False
            load_velocities = False
            load_history = False
            verbose_when_loading = False
        self.verbose_when_loading = verbose_when_loading
        self.exclude_non_FoF = exclude_non_FoF
        if "HostHaloId" in self.colnames:
            self.non_FoF = self.catalog["HostHaloId"] == -1
        else:
            self.non_FoF = np.zeros(self.size, dtype=bool)
        # correct if they are given in linear space
        if logMmin is not None and logMmin > 100:
            logMmin = np.log10(logMmin)
        self.logMmin = logMmin
        if logM200Mean_min is not None and logM200Mean_min > 100:
            logM200Mean_min = np.log10(logM200Mean_min)
        self.logM200Mean_min = logM200Mean_min
        if logMstar_min is not None and logMstar_min > 100:
            logMstar_min = np.log10(logMstar_min)
        self.logMstar_min = logMstar_min
        if self.exclude_non_FoF:
            if self.verbose_when_loading:
                print(f"Excluding {self.non_FoF.sum()} non-FoF subhalos")
            self._catalog = self.catalog[~self.non_FoF]
        # apply mass cuts
        if "Mbound" in self.colnames and self.logMmin is not None:
            self._catalog = self.catalog[self.mass("total") >= 10**self.logMmin]
        if "MboundType4" in self.colnames and self.logMstar_min is not None:
            self._catalog = self.catalog[self.mass("stars") >= 10**self.logMstar_min]
        if "M200Mean" in self.colnames and self.logM200Mean_min is not None:
            M200mask = self.catalog["M200Mean"] > 10**self.logM200Mean_min
            self._catalog = self.catalog[M200mask]
        if "Nbound" in self.colnames:
            if self.as_dataframe:
                self.catalog["IsDark"] = self.nbound("stars") == 0
            else:
                self._catalog = append_fields(
                    self.catalog, "IsDark", (self.nbound("stars") == 0)
                )
        self.isnap = isnap
        if self.isnap is not None:
            self.redshift = self.sim.redshift(self.isnap)
        self._has_host_properties = False
        self._has_distances = False
        self._has_velocities = []
        self.load_hosts = load_hosts
        if self.load_hosts:
            self.host_properties()
        else:
            load_distances = False
            load_velocities = False
        self.load_distances = load_distances
        self.load_velocities = load_velocities
        if self.load_distances:
            # self.distance2host('Physical')
            self.distance2host("Comoving")
        if self.load_velocities:
            self.host_velocities()
        self.load_history = load_history
        if self.load_history:
            self.read_history()

    # @classmethod
    # def from_sample(cls, mask, load_hosts=False, ):
    #     return cls(self.catalog[mask], self.sim,)

    ### attributes ###

    ### hidden methods ###

    ### methods ###

    def distance2massive(self, logM200Mean_min=13, frame="Comoving", verbose=False):
        """Calculate the distance of each subhalo to the nearest
        massive halo

        This method adds two columns to ``self``:
            "ComovingMostBoundDistanceToMassive" the distance in Mpc
            "NearestMassiveTrackId" the TrackId of the central subhalo
                of the nearest massive halo

        Parameters
        ----------
        logM200Mean_min : float
            log-minimum M200Mean mass of what's considered a
            "massive" halo

        Notes
        -----
        - This method can only be run in a snapshot which has halo
            properties computed
        """
        if "M200Mean" not in self.colnames:
            raise ValueError(
                "can only run distance2massive in a" " snapshot with halo information"
            )
        cat = self.catalog
        pcol = f"{frame}MostBoundPosition"
        dcol = f"{frame}MostBoundDistance"
        col = f"{dcol}ToMassive"
        tcol = "NearestMassiveCentralTrackId"  # pending
        self._catalog[col] = -np.ones(self.size)
        self._catalog[tcol] = -np.ones(self.size, dtype=np.int64)
        in_massive = cat["M200Mean"] > 10**logM200Mean_min
        self._catalog[col][in_massive] = cat[dcol][in_massive]
        cen_massive = in_massive & (cat["Rank"] == 0)
        self._catalog[tcol][cen_massive] = cat["TrackId"][cen_massive]
        dist = np.sum(
            np.array(
                [
                    (
                        cat[f"{pcol}{i}"][~in_massive].values
                        - cat[f"{pcol}{i}"][cen_massive].values[:, None]
                    )
                    ** 2
                    for i in "012"
                ]
            ),
            axis=0,
        )
        closest = np.argmin(dist, axis=0)
        mindist = np.min(dist, axis=0)
        self._catalog[col][~in_massive] = mindist
        return

    def distance2host(self, frame="Comoving", verbose=False):
        """Calculate the distance of all subhalos to the center of
        their host

        Parameters
        ----------

        Returns
        -------
        dist2host : float array
            distances to host in Mpc, in 3d or along the given
            projection
        """
        # if self._has_distances:
        # print('Distances already calculated')
        # return
        if self.verbose_when_loading:
            print("Calculating distances...")
        input_fmt = self.as_dataframe
        self.as_dataframe = True
        # alias
        sub = self.catalog
        columns = list(np.append(self.pcols(), ["HostHaloId", "Rank"]))
        to = time()
        hosts = sub[columns].join(
            sub[columns][sub["Rank"] == 0].set_index("HostHaloId"),
            on="HostHaloId",
            rsuffix="_h",
        )
        if self.verbose_when_loading:
            print("hosts:", np.sort(hosts.columns))
        if self.verbose_when_loading:
            print("Joined hosts in {0:.2f} min".format((time() - to) / 60))
        # 1d
        if self.verbose_when_loading:
            ti = time()
            print("1d:")
            print(self.pcols())
        for dcol, pcol in zip(self.dcols(1, frame), self.pcols(frame)):
            self.catalog[dcol] = ((hosts[pcol] - hosts[pcol + "_h"]) ** 2) ** 0.5
            if self.verbose_when_loading:
                print(dcol, pcol)
                print(
                    "percentiles:",
                    np.percentile(self.satellites[dcol], [0, 1, 25, 50, 99, 100]),
                )
        if self.verbose_when_loading:
            # print(hosts[['HostHaloId','HostHaloId_h','Rank','Rank_h']][j])
            # print(hosts[['ComovingMostBoundPosition0'[j])
            print("1d distances in {0:.2f} s".format(time() - ti))
        # 2d
        if self.verbose_when_loading:
            ti = time()
        for dcol in self.dcols(2, frame):
            dcols = [self.dcol(dcol[-i], frame) for i in (2, 1)]
            self.catalog[dcol] = np.sum(self.catalog[dcols] ** 2, axis=1) ** 0.5
            if self.verbose_when_loading:
                print(dcol)
                print(
                    "percentiles:",
                    np.percentile(self.satellites[dcol], [0, 1, 25, 50, 99, 100]),
                )
        if self.verbose_when_loading:
            print("2d distances in {0:.2f} s".format(time() - ti))
        # 3d
        if self.verbose_when_loading:
            ti = time()
        self.catalog[self.dcol(frame=frame)] = (
            np.sum(self.catalog[self.dcols(frame=frame)] ** 2, axis=1) ** 0.5
        )
        if self.verbose_when_loading:
            print("3d distances in {0:.2f} s".format(time() - ti))
            print(
                "percentiles:",
                np.percentile(
                    self.satellites[self.dcol(frame=frame)], [0, 1, 25, 50, 99, 100]
                ),
            )
        if verbose or self.verbose_when_loading:
            print("Calculated distances in {0:.2f} s".format(time() - to))
        self.as_dataframe = input_fmt
        self._has_distances = True
        return

    def host(self, trackid, return_value="index"):
        """Host halo of a given trackid

        Parameters
        ----------
        trackid : int
            track ID
        return_value : {'index', 'mask', 'trackid', 'table'}, optional
            whether to return the index, a boolean mask, or the full
            table containing properties of siblings only

        Returns
        -------
        The returned value will depend on ``return_value``:
            -if ``return_value='index'``:
                int corresponding to the index of the host in
                ``self.track``
            -if ``return_value='trackid'``:
                int corresponding to the host's track ID
            -if ``return_value='mask'``:
                boolean mask to be applied to ``self.track``
            -if ``return_value='table'``:
                np.struct_array corresponding to the full entry for the
                host in ``self.table``
        """
        assert trackid // 1 == trackid, "trackid must be an int"
        _valid_return = ("index", "mask", "table", "track", "trackid")
        assert return_value in _valid_return, "return_value must be one of {0}".format(
            _valid_return
        )
        sib = self.siblings(trackid, return_value="mask")
        if sib is None:
            return None
        else:
            host_mask = sib & (self.catalog["Rank"] == 0)
        if return_value == "mask":
            return host_mask
        if return_value == "track":
            return self.reader.GetTrack(np.array(self.catalog["TrackId"][host_mask])[0])
        if return_value == "trackid":
            return np.array(self.catalog["TrackId"][host_mask])[0]
        if return_value == "index":
            return self._range[host_mask][0]
        return np.array(self.catalog[host_mask])

    def _shmr_binning(self, x, bins, log=False):
        if not np.iterable(bins):
            bins = (
                np.logspace(np.log10(x.min()), np.log10(x.max()), bins)
                if log
                else np.linspace(x.min(), x.max(), bins)
            )
        if log:
            logbins = np.log10(bins)
            xo = 10 ** ((logbins[1:] + logbins[:-1]) / 2)
        else:
            xo = (bins[:-1] + bins[1:]) / 2
        return bins, xo

    def _shmr_xy(self, relation, mask=None):
        if relation == "hsmr":
            x = self.MboundType[:, 4]
            y = self.Mbound
        elif relation == "shmr":
            x = self.Mbound
            y = self.MboundType[:, 4]
        if mask is None:
            return x, y
        return x[mask], y[mask]

    def _shmr_wrapper(self, relation, plot_kwargs={}, **kwargs):
        if kwargs["sample"] == "centrals":
            mask = self.central_mask
        elif kwargs["sample"] == "satellites":
            mask = self.satellite_mask
        elif kwargs["sample "] == "all":
            mask = np.ones(self.central_mask.size, dtype=bool)
        ic(kwargs["sample"])
        ic(mask.sum())
        if kwargs["min_hostmass"] is not None:
            hostmass = kwargs["hostmass"]  # for clearer error message
            assert hostmass in ("M200Mean", "MVir", "Mbound")
            mask = mask & (self.catalog[hostmass] >= kwargs["min_hostmass"])
            ic(mask.sum())
        x, y = self._shmr_xy(relation, mask=mask)
        bins, xo = self._shmr_binning(x, kwargs["bins"])
        mr = np.histogram(x, bins, weights=y)[0] / np.histogram(x, bins)[0]
        ic(mr)
        if kwargs["ax"] is not None:
            kwargs["ax"].plot(xo, mr, **plot_kwargs)
        return xo, mr

    def hsmr(
        self,
        bins=10,
        sample="all",
        hostmass="M200Mean",
        min_hostmass=None,
        ax=None,
        **kwargs,
    ):
        """Halo-to-stellar mass relation (HSMR)

        Parameters
        ----------
        bins : int or array-like
            ``bins`` argument of ``np.histogram``
        sample : {'all','centrals','satellites'}
            over which subhaloes to calculate the HSMR

        Optional parameters
        -------------------
        min_hostmass : float
            minimum host mass to consider
        hostmass : one of {'M200Mean','MVir','Mbound'}
            host mass definition
        ax : ``matplotlib.axes.Axes`` instance
            axis over which to plot the HSMR
        kwargs : dict
            arguments passed to ``plt.plot``

        Returns
        -------
        xo, hsmr : ndarray
            central values of stellar mass and mean subhalo mass
        """
        xo, hsmr = self._shmr_wrapper(
            "hsmr",
            bins=bins,
            sample=sample,
            hostmass=hostmass,
            min_hostmass=min_hostmass,
            ax=ax,
            plot_kwargs=kwargs,
        )
        return xo, hsmr

    def shmr(
        self,
        bins=10,
        sample="all",
        hostmass="M200Mean",
        min_hostmass=None,
        ax=None,
        **kwargs,
    ):
        """Stellar-to-halo mass relation (SHMR)

        Parameters
        ----------
        bins : int or array-like
            ``bins`` argument of ``np.histogram``
        sample : {'all','centrals','satellites'}
            over which subhaloes to calculate the HSMR

        Optional parameters
        -------------------
        min_hostmass : float
            minimum host mass to consider
        hostmass : one of {'M200Mean','MVir','Mbound'}
            host mass definition
        ax : ``matplotlib.axes.Axes`` instance
            axis over which to plot the SHMR
        kwargs : dict
            arguments passed to ``plt.plot``

        Returns
        -------
        xo, hsmr : ndarray
            central values of subhalo mass and mean stellar mass
        """
        xo, shmr = self._shmr_wrapper(
            "shmr",
            bins=bins,
            sample=smple,
            hostmass=hostmass,
            min_hostmass=min_hostmass,
            ax=ax,
            plot_kwargs=kwargs,
        )
        return xo, shmr

    def _mass_weighted_stat(self, values, mass, label):
        """Apply mass weighting

        Parameters
        ----------
        values : ``np.array``
            quantities to be mass-weighted
        mass : ``pandas.Series``
            mass with which to weight
        label : str
            name of the returned Series

        Returns
        -------
        weighted_stat : ``pd.Series``
            mass-weighted statistic
        """
        wstat = {label: np.sum(values * mass) / np.sum(mass)}
        return pd.Series(wstat, index=[label])

    def _mass_weighted_average(self, x, cols, mcol, label="vcl"):
        """Mass-weighted average

        Calculate the mass-weighted average of the quantity(ies)
        defined in column(s) `cols`. Meant to be used with
        ``pandas.groupby.apply``

        Parameters
        ----------
        x : ``pd.groupby
        cols : str or list of str
            column(s) to be averaged
        mcol : str
            mass column used for the weighting. See
            ``Simulation.masstypes`` and
            ``Simulation.masstype_pandas_columns``


        Returns
        -------
        avg : ``pandas.Series``
            average quantities
        """
        cols = [cols] if isinstance(cols, six.string_types) else cols
        if x[cols[0]].count() == 1:
            return pd.Series({label: 0}, index=[label])
        vmean = np.mean([x[i] for i in cols], axis=0)
        vxyz = np.sum([(x[i] - vi) ** 2 for i, vi in zip(cols, vmean)], axis=0)
        return self._mass_weighted_stat(vxyz, np.array(x[mcol]), label) ** 0.5

    def _mass_weighted_std(self, x, cols, mcol, label="sigma_cl"):
        cols = [cols] if isinstance(cols, six.string_types) else cols
        if x[cols[0]].count() == 1:
            return pd.Series({label: 0}, index=[label])
        # this is correct only if they are peculiar velocities
        sigma_xyz = np.sum([x[i] ** 2 for i in cols], axis=0)
        return self._mass_weighted_stat(sigma_xyz, np.array(x[mcol]), label) ** 0.5

    def host_velocities(self, mass_weighting=None):
        # note that this screws things up if self.pvref changes
        if mass_weighting in self._has_velocities:
            print("velocities already loaded")
            return
        if not self._has_host_properties:
            self.host_properties()
        if self.verbose_when_loading:
            print("Calculating velocities...")
        to = time()
        adf = self.as_dataframe
        self.as_dataframe = True
        # alias
        cx = self.catalog
        axes = "xyz"
        vcols = ["Physical{1}Velocity{0}".format(i, self.pvref) for i in range(3)]
        if mass_weighting is None:
            mweight = 1
        else:
            mcol = self.sim.masstype_pandas_column(mass_weighting)
            mweight = cx[mcol]
        keys = list(cx)
        new_keys = []
        # mean velocities
        grcols = ["HostHaloId", "Nsat"]
        skip = len(grcols)
        mvcols = []
        for i, vcol in enumerate(vcols):
            mvcols.append("mv{0}".format(i))
            cx[mvcols[-1]] = cx[vcol] * mweight
        grcols = np.append(grcols, mvcols)
        group = cx[grcols].groupby("HostHaloId")
        if mass_weighting is None:
            wmean = group[mvcols].mean()
        else:
            wmean = group[mvcols].sum()
        if mass_weighting is None:
            # msum = group['Nsat']
            msum = 1
        else:
            msum = group[mcol].sum()
        ## host mean velocities
        hosts = pd.DataFrame({"HostHaloId": np.array(group.size().index)})
        vhcol = "Physical{0}HostMeanVelocity".format(self.pvref)
        vhcols = [vhcol + str(i) for i in range(3)]
        for i, mvcol in enumerate(mvcols):
            hosts[vhcol + str(i)] = wmean[mvcol] / msum
        hosts[vhcol] = np.sum(wmean[mvcols] ** 2, axis=1) ** 0.5
        ## velocity dispersions
        if self.verbose_when_loading:
            print("velocity dispersions...")
            ti = time()
        hostkeys = np.append(["HostHaloId", vhcol], vhcols)
        cx = cx.join(
            hosts[hostkeys].set_index("HostHaloId"), on="HostHaloId", rsuffix="_h"
        )
        for i, vcol in enumerate(vcols):
            vdiff = (
                cx[vcol] - cx["Physical{0}HostMeanVelocity{1}".format(self.pvref, i)]
            )
            cx[mvcols[i]] = mweight * vdiff**2
        # group again because I want the new mvcols grouped as well
        group = cx.groupby("HostHaloId")
        wvar = group[mvcols].sum()
        scol = "Physical{0}HostVelocityDispersion".format(self.pvref)
        scols = []
        for i, mvcol in enumerate(mvcols):
            scols.append(scol + str(i))
            hosts[scols[-1]] = (wvar[mvcol] / msum) ** 0.5
        hosts[scol] = np.sum(wvar / msum, axis=1) ** 0.5
        new_keys = np.append(["HostHaloId", scol], scols)
        cx = cx.join(hosts[new_keys].set_index("HostHaloId"), on="HostHaloId")
        # -1 or not?
        for col in np.append(scol, scols):
            cx[col] = cx[col] / (cx["Nsat"] - 1) ** 0.5
            if self.verbose_when_loading:
                print("dispersions in {0:.1f} seconds".format(time() - ti))
        # peculiar velocities
        if self.verbose_when_loading:
            ti = time()
        vpcol = "Physical{0}PeculiarVelocity".format(self.pvref)
        vpcols = [vpcol + str(i) for i in range(3)]
        for col in (vhcol, vhcols, scol, scols):
            new_keys = np.append(new_keys, col)
        for i in range(3):
            cx[vpcol + str(i)] = (
                cx["Physical{0}Velocity{1}".format(self.pvref, i)]
                - cx["Physical{0}HostMeanVelocity{1}".format(self.pvref, i)]
            )
        # 3d
        cx["Physical{0}Velocity".format(self.pvref)] = np.sum(
            cx[self.vcols1d()] ** 2, axis=1
        ) ** 0.5 * np.sign(np.sum(cx[self.vcols1d()], axis=1))
        cx[vpcol] = (
            cx["Physical{0}Velocity".format(self.pvref)]
            - cx["Physical{0}HostMeanVelocity".format(self.pvref)]
        )
        if self.verbose_when_loading:
            print(f"Peculiar velocities in {time()-ti:.2f} seconds")
        cx.drop(columns=mvcols)
        self._catalog = cx
        self._has_velocities.append(mass_weighting)
        self.as_dataframe = adf
        if self.verbose_when_loading:
            print(f"Calculated velocities in {time()-to:.1f} seconds")
            print()
        return

    def index(self, trackid):
        """Index of a given trackid in the subhalo catalog

        Parameters
        ----------
        trackid : int or array of int
            track ID(s)
        """
        assert trackid // 1 == trackid, "trackid must be an int or an array of int"
        if hasattr(trackid, "__iter__"):
            return self._range[self.catalog["TrackId"] == trackid]
        return self._range[self.catalog["TrackId"] == trackid][0]

    def is_central(self, trackid):
        """Whether a given subhalo is a central subhalo

        Parameters
        ----------
        trackid : int or array of int
            track ID of the subhalo in question

        Returns
        -------
        is_central : bool or array of bool
        """
        assert trackid // 1 == trackid, "trackid must be an int or an array of int"
        # in case they're floats of ints
        trackid = np.array(trackid, dtype=int)
        cent = self.data["Rank"][self.data["TrackId"] == trackId] == 0
        if hasattr(trackid, "__iter__"):
            return cent
        return cent[0]

    def host_properties(self, force_isnap=False, verbose=False):
        """Load halo masses and sizes into the subhalo data

        See `HostHalos` for details

        Creates two new attributes:
            -`Nsat`: number of satellite subhalos
            -`Ndark`: number of dark subhalos. Note that this includes
                the central subhalo
        """
        if self._has_host_properties:
            print("Hosts already loaded")
            return
        if verbose:
            print("Loading hosts...")
            to = time()
        adf = self.as_dataframe
        self.as_dataframe = True
        ic(self.isnap)
        hosts = HostHalos(self.sim, self.isnap, force_isnap=force_isnap)
        ti = time()
        # number of star particles, to identify dark subhalos
        cols = ["HostHaloId"]
        if "IsDark" in self.colnames:
            cols.append("IsDark")
        grouped = self.catalog[cols].groupby("HostHaloId")
        nmdict = {"Nsat": grouped.size() - 1}
        if "IsDark" in self.colnames:
            nmdict["Ndark"] = grouped["IsDark"].sum()
        nm = pd.DataFrame(nmdict)
        self._catalog = self.catalog.join(nm, on="HostHaloId", rsuffix="_h")
        self._catalog = self.catalog.join(hosts.catalog, on="HostHaloId", rsuffix="_h")
        # update host masses
        for col in list(self.catalog):
            if "M200" in col or col == "MVir":
                if self.catalog[col].max() < 1e6:
                    self.catalog[col] = 1e10 * self.catalog[col] / self.sim.cosmology.h
        if verbose:
            print("Joined hosts in {0:.2f} s".format(time() - to))
        del hosts
        # R200Mean
        if "M200Mean" in self.catalog.columns:
            rho_m = self.cosmology.critical_density(self.redshift) * self.cosmology.Om0
            rho_m = rho_m.to("Msun/Mpc^3").value
            self.catalog["R200Mean"] = (
                3 * self.catalog["M200Mean"] / (4 * np.pi * 200 * rho_m)
            ) ** (1 / 3)
        if self.logM200Mean_min is not None:
            mask = self.catalog["M200Mean"] > 10**self.logM200Mean_min
            self._catalog = self.catalog[mask]
        self._has_host_properties = True
        self.as_dataframe = adf
        if verbose:
            print("Loaded in {0:.2f} s".format(time() - to))
        return

    def read_history(self):
        file = os.path.join(self.sim.data_path, "history", "history.h5")
        if not os.path.isfile(file):
            wrn = f"cannot load history: history file {file} does not exist"
            warnings.warn(wrn)
            return
        is_df = isinstance(self.catalog, pd.DataFrame)
        if not is_df:
            self.catalog = pd.DataFrame.from_records(self.catalog)
        history = pd.DataFrame()
        with h5py.File(file, "r") as hdf:
            for grp in hdf.keys():
                group = hdf.get(grp)
                for col in group.keys():
                    dfcol = col if grp == "trackids" else f"{grp}:{col}"
                    # it's easier to correct for h here than in
                    # _update_mass_columns
                    # norm = self.sim.cosmology.h if \
                    #     (dfcol.split(':')[-1] in
                    #      ('Mbound','Mgas','Mstar','Mdm', 'M200Mean','MVir')) \
                    #     else 1
                    history[f"history:{dfcol}"] = np.array(group.get(col))  #  / norm
        self._catalog = self.catalog.merge(
            history.reset_index(),
            how="left",
            left_on="TrackId",
            right_on="history:TrackId",
        )
        # self._update_mass_columns(history_only=True)
        return

    def siblings(self, trackid, return_value="index"):
        """All subhalos hosted by the same halo at present

        Parameters
        ----------
        trackid : int
            track ID
        return_value : {'index', 'mask', 'table'}, optional
            whether to return the indices, a boolean mask, or the full
            table containing properties of siblings only

        Returns
        -------
        see ``return_value``
        """
        assert trackid // 1 == trackid, "trackid must be an int"
        _valid_return = ("index", "mask", "table", "track", "trackid")
        assert return_value in _valid_return, "return_value must be one of {0}".format(
            _valid_return
        )
        try:
            idx = self.index(trackid)
        except IndexError as e:
            print("IndexError:", e)
            return None
        hostids = np.array(self.catalog["HostHaloId"])
        sibling_mask = hostids == hostids[idx]
        if sibling_mask.sum() == 0:
            return None
        if return_value == "mask":
            return sibling_mask
        if return_value == "track":
            return self.catalog[sibling_mask]
        if return_value == "trackid":
            return np.array(self.catalog["TrackId"])[sibling_mask]
        if return_value == "index":
            return self._range[sibling_mask]


class _Track(BaseSubhalo):
    def __init__(self, track, sim, as_dataframe=False):
        """
        Discontinued. Please use ``track.Track`` instead.

        Parameters
        ----------
        track : output of ``reader.GetTrack``
            track
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)

        Examples
        --------
        To load a subhalo and its track from Aspotle/V1_LR, do
        >>> track = Subhalo(trackid, 'LR')
        or
        >>> track = Subhalo(trackid, Simulation('LR'))
        """
        super(Track, self).__init__(track, sim, as_dataframe=as_dataframe)
        # print(as_dataframe)
        # print(track.dtype)
        # print(track.dtype[0])
        # print(track.dtype[0][:4])
        # print()
        # if isinstance(track, np.ndarray) and as_dataframe:
        # track = pd.DataFrame.from_records(track, columns=track.dtype.names)
        self._track = track
        # print(track.dtype)
        # print(type(track))
        # print(type(self._track))
        # print(np.sort(self._track.columns))
        self._trackid = self._track["TrackId"]
        self.hostid = np.array(self._track["HostHaloId"])
        self.current_hostid = self.hostid[-1]
        self._first_satellite_snapshot = self._none_value
        self._first_satellite_snapshot_index = self._none_value
        self._last_central_snapshot = self._none_value
        self._last_central_snapshot_index = self._none_value
        self._zcentral = None
        self._zinfall = None

    def __str__(self):
        return f"Track ID {self.trackid} in {self.sim.name}"

    ### attributes ###

    @property
    def first_satellite_snapshot(self):
        """
        If the track has never been a satellite, this will remain None
        """
        if self.first_satellite_snapshot_index == self._none_value:
            return self._none_value
        if self._first_satellite_snapshot == self._none_value:
            self._first_satellite_snapshot = self.track["Snapshot"][
                self.first_satellite_snapshot_index
            ]
        return self._first_satellite_snapshot

    @property
    def first_satellite_snapshot_index(self):
        """
        If the track has never been a satellite, this will remain None
        """
        if self._first_satellite_snapshot_index == self._none_value:
            sat = self.track["Rank"] > 0
            if sat.sum() > 0:
                self._first_satellite_snapshot_index = self._range[sat][0]
        return self._first_satellite_snapshot_index

    @property
    def icent(self):
        """alias for ``self.last_central_snapshot_index``"""
        return self.last_central_snapshot_index

    @property
    def last_central_snapshot(self):
        """
        If the track has never been a central, this will remain None
        """
        if self.last_central_snapshot_index == self._none_value:
            return self._none_value
        if self._last_central_snapshot == self._none_value:
            self._last_central_snapshot = self.track["Snapshot"][
                self.last_central_snapshot_index
            ]
        return self._last_central_snapshot

    @property
    def last_central_snapshot_index(self):
        """
        If the track has never been a central, this will remain None
        """
        if self._last_central_snapshot_index == self._none_value:
            cent = self.track["Rank"] == 0
            if cent.sum() > 0:
                self._last_central_snapshot_index = self._range[cent][-1]
        return self._last_central_snapshot_index

    @property
    def scale(self):
        return self.track["ScaleFactor"]

    @property
    def track(self):
        if self.as_dataframe and isinstance(self._track, np.ndarray):
            self._track = self.DataFrame(self._track)
        elif not self.as_dataframe and isinstance(self._track, pd.DataFrame):
            self._track = self._track.to_records()
        return self._track

    @property
    def trackid(self):
        if hasattr(self._trackid, "__iter__"):
            self._trackid = self._trackid[0]
        return self._trackid

    @property
    def z(self):
        return 1 / self.scale - 1

    @property
    def zcentral(self):
        if self._zcentral is None:
            if self.is_central():
                self._zcentral = self.z[-1]
            else:
                self._zcentral = self.z[self.icent]
        return self._zcentral

    ### methods ###

    def host(self, isnap=-1, return_value="trackid"):
        """Host halo (i.e., central subhalo) information at a given
        snapshot

        Parameters
        ----------
        isnap : int, optional
            snapshot number at which the host is to be identified

        Returns
        -------
        host : int
            track ID of the host halo

        Usage
        -----
        The host halo may be loaded as a Track with:
        >>> track = Track(trackid, sim)
        >>> host_track = track.host(isnap)
        >>> host = Track(host_track, sim)
        """
        _valid_return = ("index", "mask", "table", "track", "trackid")
        assert return_value in _valid_return, "return_value must be one of {0}".format(
            _valid_return
        )
        hostid = self.hostid[isnap]
        # Rank is necessary for the definition of the Subhalos object
        snap = Subhalos(
            self.reader.LoadSubhalos(
                isnap, ["TrackId", "HostHaloId", "Rank", "Mbound"]
            ),
            self.sim,
            isnap,
            load_hosts=False,
        )
        # hostid = snap.host(self.trackid, return_value='trackid')
        return snap.host(self.trackid, return_value=return_value)
        return self.reader.GetTrack(hostid)

        sib = self.siblings(trackid, return_value="mask")
        host_mask = sib & (self.catalog["Rank"] == 0)
        if return_value == "mask":
            return host_mask
        if return_value == "track":
            return self.reader.GetTrack(self.catalog["TrackId"][host_mask][0])
        if return_value == "trackid":
            return self.catalog["TrackId"][host_mask][0]
        if return_value == "index":
            return self._range[host_mask][0]
        return self.catalog[host_mask]

    def is_central(self, isnap):
        """Whether the subhalo is a central at a given moment

        Parameters
        ----------
        isnap : int or array of int
            snapshot index or indices

        Returns
        -------
        is_central : bool or array of bool
            whether the subhalo is a central at the specified time(s)
        """
        return self.data["Rank"][isnap] == 0

    def lookback_time(self, z=None, scale=None, isnap=None, include_units=False):
        """
        if z is not None:
            t = self.sim.cosmology.lookback_time(z)
        elif scale is not None:
            t = self.sim.cosmology.lookback_time(1/scale - 1)
        elif isnap is not None:
            z = 1/self.track['ScaleFactor'][isnap] - 1
            return self.sim.cosmology.lookback_time(z)
        return self.sim.cosmology.lookback_time(self.z)
        """
        if z is None:
            if scale is not None:
                z = 1 / scale - 1
            elif isnap is not None:
                z = 1 / self.track["ScaleFactor"][isnap] - 1
            else:
                z = self.z
        t = self.sim.cosmology.lookback_time(z)
        if not include_units:
            t = t.value
        return t

    def mergers(self, output="index"):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return

    def infall(self, return_value="index", min_snap_range_brute=3):
        """Last redshift at which the subhalo was not in its present
        host

        **Should be able to read the infall file if it exists**

        Parameters
        ----------
        return_value : {'index', 'tlookback', 'redshift'}, optional
            output value. Options are:
        """
        valid_outputs = ("index", "redshift", "tlookback")
        assert (
            return_value in valid_outputs
        ), "return_value must be one of {0}. Got {1} instead".format(
            valid_outputs, return_value
        )
        hostid = self.host(isnap=-1, return_value="trackid")
        iinf = 0
        # first jump by halves until we've narrowed it down to
        # very few snapshots
        # imin = self.sim.snapshots.min()
        imin = self.track["SnapshotIndexOfBirth"]
        imax = self.sim.snapshots.max()
        do_smart = True
        if do_smart:
            while imax - imin > min_snap_range_brute:
                isnap = (imin + imax) // 2
                # note that this will fail if isnap == 281
                subs = self.reader.LoadSubhalos(isnap, ["TrackId", "HostHaloId"])
                if len(subs) == 0:
                    imin = isnap
                    continue
                snapcat = Subhalos(subs, self.sim, isnap, load_hosts=False, logMmin=0)
                sib = snapcat.siblings(hostid, "trackid")
                if sib is None or self.trackid not in sib:
                    imin = isnap
                else:
                    imax = isnap
        # once we've reached the minimum range allowed above,
        # we just do backwards brute-force
        for isnap in range(imax, imin - 1, -1):
            subs = self.reader.LoadSubhalos(isnap, ["TrackId", "HostHaloId"])
            if len(subs) == 0:
                iinf_backward = 0
                break
            snapcat = Subhalos(subs, self.sim, isnap, load_hosts=False, logMmin=0)
            sib = snapcat.siblings(hostid, "trackid")
            # this means we reached the beginning of the track
            if sib is None:
                iinf_backward = 0
                break
            elif self.trackid not in sib:
                iinf_backward = isnap - self.sim.snapshots.max()
                break
        else:
            iinf_backward = 0
        iinf = self._range[iinf_backward]
        if return_value == "tlookback":
            return self.lookback_time(isnap=iinf)
        if return_value == "redshift":
            return self.z[iinf]
        return iinf
