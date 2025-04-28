from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import h5py
from icecream import ic
import numpy as np
import six
#debugging
from icecream import ic
import sys

from HBTReader import HBTReader

from .simulation import BaseSimulation, Simulation
from .subhalo import BaseSubhalo, Subhalos


class HaloTracks:
    """Class containing tracks of all subhalos in a given halo

    """
    def __init__(self, hosthaloid, subhalos, sim=None, isnap=-1):
        if sim is None and not hasattr(subhalos, 'sim'):
            err = 'object `subhalos` does not contain a `sim` attribute;' \
                ' `sim` argument must be provided'
            raise ValueError(err)
        self._hosthaloid = hosthaloid
        self._subhalos = subhalos
        self._isnap = isnap
        #super().__init__(subhalos, subhalos.sim)
        self._central = self._get_track_central()
        self._satellites_mask = \
            (self.satellites['HostHaloId'] == self.hosthaloid)
        self._satellites = self.satellites.loc[self._satellites_mask]

    @property
    def central(self):
        """Central subhalo track"""
        return self._central

    @property
    def hosthaloid(self):
        return self._hosthaloid

    @property
    def satellites(self):
        """Satellite subhalo tracks"""
        return self._satellites

    @property
    def satellites_mask(self):
        return self._satellites_mask

    def _get_track_central(self):
        _central = (self.catalog['HostHaloId'] == self.hosthaloid) \
            & (self.catalog['Rank'] == 0)
        return Track(self.catalog['TrackId'].loc[_central], self.sim)

    def _get_track_satellites(self):
        _satellites = (self.catalog['HostHaloId'] == self.hosthaloid) \
            & (self.catalog['Rank'] > 0)
        return TrackArray(
            self.catalog['TrackId'].loc[_satellites], self.sim)


class TrackArray(BaseSubhalo):

    def __init__(self, trackid, sim):
        if not np.iterable(trackid):
            trackid = [trackid]
        self.trackids = np.array(trackid, dtype=np.int32)
        # load simulation
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        super(TrackArray, self).__init__(self.trackid, self.sim)

    def history(self, cols=None, when='first_infall'):
        """Return data from events in the history of the track

        Parameters
        ----------
        cols : str or list of str, optional
            what data to return
        when : one of (None,'first_infall','last_infall','cent','sat')
            event from which to retrieve data

        Returns
        -------
        history : pd.DataFrame
            requested elements
        """
        if not self.sim.has_history:
            err = f'file {self.sim.history_file} not found'
            raise ValueError(err)
        assert when in (None, 'first_infall', 'last_infall', 'cent', 'sat')
        with h5py.File(self.sim.history_file, 'r') as hdf:
            trackids = hdf.get('trackids')
            jsort = np.argsort(self.trackid)
            j = np.isin(np.array(trackids.get('TrackId')), self.trackid)
            if j.sum() != self.trackids.size:
                err = f'not all TrackIds found in {self.sim.history_file}'
                raise ValueError(err)
            grp = hdf.get(when)
            if isinstance(cols, str):
                cols = [cols]
            elif cols is None:
                cols = grp.keys()
            return [np.array(grp.get(col))[j][0] for col in cols]


class Track(BaseSubhalo):
    """Track object containing history of individual subhalo"""

    def __init__(self, trackid, sim):
        """
        Parameters
        ----------
        trackid : int or output of ``reader.GetTrack``
            ID of the track to be loaded or the track itself
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)

        To load a track from Aspotle/V1_LR, do
        >>> track = Track(trackid, 'LR')
        or
        >>> track = Track(trackid, Simulation('LR'))

        """
        # load simulation
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        # load reader and track
        self.reader = HBTReader(self.sim.path)
        if isinstance(trackid, (int,np.integer)):
            self.trackid = trackid
            self.track = self._load_track()
        elif isinstance(trackid, np.ndarray):
            self.track = trackid
            self.trackid = self.track['TrackId']
            if hasattr(self.trackid, '__iter__'):
                self.trackid = self.trackid[0]
        else:
            msg = f'argument trackid {trackid} of wrong type ({type(trackid)})'
            raise ValueError(msg)
        super(Track, self).__init__(self.track, self.sim)
        # other attributes
        #self.current_host = self.hosts[-1]
        self.hostid = np.array(self.track['HostHaloId'])
        self.scale = self.track['ScaleFactor']
        self.z = 1/self.scale - 1
        self.t_lookback = self.sim.cosmology.lookback_time(self.z)
        self._birth_snapshot = self._none_value
        self._first_satellite_snapshot = self._none_value
        self._first_satellite_snapshot_index = self._none_value
        self._infall_snapshot = None
        self._infall_snapshot_index = None
        self._last_central_snapshot = None
        self._last_central_snapshot_index = None
        #self._Mbound = None
        #self._MboundType = None
        self._zcentral = None
        self._zinfall = None

    def __repr__(self):
        return f'Track({self.trackid}, {self.sim.label})'

    def __str__(self):
        return f'Track ID {self.trackid} in {self.sim.name}'

    @property
    def birth_snapshot(self):
        if self._birth_snapshot == self._none_value:
            self._birth_snapshot \
                = self.reader.GetSub(self.trackid)['SnapshotIndexOfBirth']
        return self._birth_snapshot

    @property
    def first_satellite_snapshot(self):
        """
        If the track has never been a satellite, this will remain None
        """
        if self.first_satellite_snapshot_index == self._none_value:
            return self._none_value
        if self._first_satellite_snapshot == self._none_value:
            self._first_satellite_snapshot = \
                self.track['Snapshot'][self.first_satellite_snapshot_index]
        return self._first_satellite_snapshot

    @property
    def first_satellite_snapshot_index(self):
        """
        If the track has never been a satellite, this will remain None
        """
        if self._first_satellite_snapshot_index == self._none_value:
            sat = (self.track['Rank'] > 0)
            if sat.sum() > 0:
                self._first_satellite_snapshot_index = self._range[sat][0]
        return self._first_satellite_snapshot_index

    @property
    def icent(self):
        """alias for ``self.last_central_snapshot_index``"""
        return self.last_central_snapshot_index

    @property
    def isat(self):
        """alias for ``self.first_satellite_snapshot_index``"""
        return self.first_satellite_snapshot_index

    @property
    def infall_snapshot(self):
        if self._infall_snapshot == self._none_value:
            self._infall_snapshot = \
                self.track['Snapshot'][self.infall_snapshot_index]
        return self._infall_snapshot

    @property
    def infall_snapshot_index(self):
        if self._infall_snapshot_index is None:
            self._infall_snapshot_index \
                = self._range[self.host != self.current_host][-1]
        return self._infall_snapshot_index

    @property
    def last_central_snapshot(self):
        if self._last_central_snapshot is None:
            #key = 'SnapshotIndexOfLastIsolation'
            #self._last_central_snapshot = \
                #self.reader.GetSub(self.trackid)[key]
            self._last_central_snapshot = \
                self.track['Snapshot'][self.track['Rank'] == 0][-1]
        return self._last_central_snapshot

    @property
    def last_central_snapshot_index(self):
        if self._last_central_snapshot_index is None:
            #self._last_central_snapshot_index = \
                #self._range[self.track['Snapshot'] == \
                            #self.last_central_snapshot][0]
            self._last_central_snapshot_index = \
                self._range[self.track['Rank'] == 0][-1]
        return self._last_central_snapshot_index

    @property
    def length(self):
        return self.track['Mbound'].size

    @property
    def Mbound(self):
        return 1e10 * self.track['Mbound']

    @property
    def MboundType(self):
        #if self.as_dataframe:
            #cols = ['MboundType{0}'.format(i) for i in range(6)]
            #return 1e10 * np.array(self.track[cols])
        return 1e10 * self.track['MboundType']

    @property
    def zcentral(self):
        """NOTE: this is returning an array instead of a float"""
        if self._zcentral is None:
            if self.track['Rank'][-1] == 0:
                self._zcentral = self.z[-1]
            else:
                self._zcentral = \
                    1/self.track['ScaleFactor'][self._last_central_snapshot_index] - 1
        return self._zcentral

    @property
    def zinfall(self):
        if self._zinfall is None:
            self._zinfall = self.z[self.host != self.current_host][-1]
        return self._zinfall

    def _load_track(self):
        """Load track, accounting for missing snapshots"""
        track = self.reader.GetTrack(self.trackid)
        return track

    ### methods ###

    def central(self, isnap=-1, return_value='trackid'):
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
        >>> host_track = track.central(isnap)
        >>> host = Track(host_track, sim)
        """
        _valid_return = ('index','mask','table','track','trackid')
        assert return_value in _valid_return, \
            'return_value must be one of {0}'.format(_valid_return)
        hostid = self.hostid[isnap]
        # Rank is necessary for the definition of the Subhalos object
        snap = Subhalos(
            self.reader.LoadSubhalos(
                isnap, ['TrackId','HostHaloId','Rank','Mbound']),
            self.sim, isnap, load_hosts=False, verbose_when_loading=False)
        #hostid = snap.host(self.trackid, return_value='trackid')
        return snap.host(self.trackid, return_value=return_value)

    def host(self, isnap=-1):
        if self['Rank'][isnap] == 0:
            ...
        return


    def host_history(self, isnap=-1, host_track=None,
                     host_history_file=None):
        """Get history for the host at a given snapshot
        
        Parameters
        ----------
        isnap : int, optional
            snapshot defining the host halo of ``self``. Ignored if
            ``host_track`` is defined.
        host_track : int or ``Track`` object, optional
            host TrackId or ``Track``.
        host_history_file : str, optional
            name of file containing host halo histories. Must be full
            (absolute or relative) path
        """
        raise NotImplementedError
        if host_track is None or isinstance(host_track, (int,np.integer)):
            with h5py.File(host_history_file, 'r') as hdf:
                trackids = hdf.get('TrackId')
                # check whether ``self`` is central or find its central
                if host_track is None:
                    if self.track['Rank'][isnap] == 0:
                        j_history = (trackids == self.trackid)
                    else:
                        ...
                ic(trackids)
        return

    def history(self, cols=None, when='first_infall', raise_error=True):
        """Return data from events in the history of the track

        Parameters
        ----------
        cols : str or list of str, optional
            what data to return
        when : one of (None,'first_infall','last_infall','cent','sat')
            event from which to retrieve data
        raise_error : bool
            whether to raise an error if the TrackId cannot be found in
            the history file. If ``False``, will return ``None``

        Returns
        -------
        history : list
            list with requested elements
        """
        if not self.sim.has_history:
            err = f'file {self.sim.history_file} not found'
            raise ValueError(err)
        assert when in (None, 'first_infall', 'last_infall', 'cent', 'sat')
        with h5py.File(self.sim.history_file, 'r') as hdf:
            trackids = hdf.get('trackids')
            j = (np.array(trackids.get('TrackId')) == self.trackid)
            if j.sum() == 1:
                grp = hdf.get(when)
                if cols is None:
                    cols = grp.keys()
                if isinstance(cols, str):
                    data = grp[cols][j][0]
                else:
                    data = [grp[col][j][0] for col in cols]
                return data
            elif j.sum() > 1:
                err = f'ambiguous TrackId {self.trackid} match in' \
                      f'{self.sim.history_file}.'
            else:
                err = f'TrackId {self.trackid} not found in' \
                      f'{self.sim.history_file}.'
            if raise_error:
                raise IndexError(err)
            else:
                return None

    def infall(self, return_value='index', min_snap_range_brute=3):
        """Last redshift at which the subhalo was not in its present
        host

        **Should be able to read the infall file if it exists**

        Parameters
        ----------
        return_value : {'index', 'tlookback', 'redshift'}, optional
            output value
        """
        valid_outputs = ('index', 'redshift', 'tlookback')
        assert return_value in valid_outputs, \
            'return_value must be one of {0}. Got {1} instead'.format(
                valid_outputs, return_value)
        hostid = self.host(isnap=-1, return_value='trackid')
        iinf = 0
        # first jump by halves until we've narrowed it down to
        # very few snapshots
        imin = self.sim.snapshots.min()
        imax = self.sim.snapshots.max()
        while imax - imin > min_snap_range_brute:
            isnap = (imin+imax) // 2
            subs = self.reader.LoadSubhalos(
                isnap, ['TrackId','HostHaloId'])
            if len(subs) == 0:
                imin = isnap
                continue
            snapcat = Subhalos(
                subs, self.sim, isnap, load_hosts=False, logMmin=0,
                verbose_when_loading=False)
            sib = snapcat.siblings(hostid, 'trackid')
            if sib is None or self.trackid not in sib:
                imin = isnap
            else:
                imax = isnap
        # once we've reached the minimum range allowed above,
        # we just do backwards brute-force
        iinf_backward = 0
        for isnap in range(imax, imin-1, -1):
            subs = self.reader.LoadSubhalos(
                isnap, ['TrackId','HostHaloId'])
            if len(subs) == 0:
                #iinf_backward = 0
                #break
                continue
            snapcat = Subhalos(
                subs, self.sim, isnap, load_hosts=False, logMmin=0,
                verbose_when_loading=False)
            sib = snapcat.siblings(hostid, 'trackid')
            # this means we reached the beginning of the track
            if sib is None:
                iinf_backward = 0
                #break
            elif self.trackid not in sib:
                iinf_backward = isnap - self.sim.snapshots.max()
                #break
            if isnap < 250:
                break
        else:
            iinf_backward = 0
        iinf =  self._range[iinf_backward]
        if return_value == 'tlookback':
            return self.lookback_time(isnap=iinf)
        if return_value == 'redshift':
            return self.z[iinf]
        return iinf

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
        return (self.data['Rank'][isnap] == 0)

    def lookback_time(self, z=None, scale=None, snapshot=None):
        """This should probably be in Simulation"""
        if z is not None:
            return self.sim.cosmology.lookback_time(z)
        if scale is not None:
            return self.sim.cosmology.lookback_time(1/scale - 1)
        if snapshot is not None:
            z = 1/self.track['ScaleFactor'][snapshot] - 1
            return self.sim.cosmology.lookback_time(z)
        return self.sim.cosmology.lookback_time(self.z)

    def mass(self, mtype=None, index=None):
        assert mtype is not None or index is not None, \
            'must provide either ``mtype`` or ``index``'
        if mtype is not None:
            if mtype.lower() in ('total', 'mbound'):
                return self.Mbound
            return self.MboundType[:,self.sim._masstype_index(mtype)]
        if index == -1:
            return self.Mbound
        return self.MboundType[:,index]

    def _mergers(self, output='index'):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return
