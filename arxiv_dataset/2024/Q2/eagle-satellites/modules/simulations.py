"""
Classes and methods to operate simulations

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.table import Table
from astropy.io import ascii, fits
from itertools import count
import numpy as np
import os
import subprocess


from . import query


class Catalog(Table):
    """Simulated catalog object"""

    def __init__(self, filename, file_format='ascii'):
        assert file_format.lower() in ('ascii', 'fits'), \
            '`file_format` must be one of {"ascii","fits"}'
        self.filename = filename
        self.file_format = file_format.lower()
        self._data = self.read(self.filename, self.file_format)
        self._ngal = self._data['GalaxyID'].size
        # to inherit attributes from astropy Tables (e.g., colnames)
        # must go after defining `self.filename` and `self.file_format`
        # as that is required to define `self.data`
        super().__init__(self._data)

    @property
    def box(self):
       return self.path.split('/')[1]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value=None):
        #if value is None:
            #self._data = self.read(self.filename, self.file_format)
        #else:
        self._data = value
        #super().__init__(self._data)
        print('value =', value['GalaxyID'].size)
        print('_data =', self._data['GalaxyID'].size)

    @property
    def database(self):
        if self.physics == 'DMONLY':
            return '{0}..{1}'.format(self.physics, self.box)
        else:
            return self.physics + self.box

    @property
    def ngal(self):
        return self._ngal

    @ngal.setter
    def ngal(self, value):
        self._ngal = value

    @property
    def path(self):
        return os.path.split(self.filename)[0]

    @property
    def physics(self):
        return [i for i in self.path.split('/')
                if i in ('DMONLY', 'Ref', 'Recal', 'AGNd5T9')][0]

    @property
    def snapshot(self):
        return [int(i.replace('snapshot', ''))
                for i in self.path.split('/') if i.startswith('snapshot')][0]

    def _requirement_infall(self, groupid):
        return 'GroupID != {0}'.format(groupid)

    def _requirement_central(self, dummy=None):
        """
        `dummy` is there just in case I want to be agnostic as to
        whether I'm calling this function or `self._requirement_infall`,
        which takes one positional argument
        """
        return 'SubGroupNumber = 0'

    def cut(self, bounds, in_place=False):
        """Cut (i.e., mask) the sample based on lower and upper bounds
        of parameter(s)

        Parameters
        ----------
        bounds : dict
            For each (key, value) item, `key` corresponds to a
            parameter found in the catalog and `value` corresponds to
            the lower and upper bound of that parameter
        in_place : bool
            whether to modify the array in-place or create a new array
            with the mask applied
        """
        mask = np.ones(self.ngal, dtype=bool)
        for key, values in bounds.items():
            assert key in self.data.colnames, \
                'column {0} not in table'
            mask = mask & (self.data[key] >= values[0]) \
                   & (self.data[key] <= values[1])
        if in_place:
            masked = self._data[mask]
            print('mask =', masked['GalaxyID'].size)
            self.data = masked
            #self.data = 'blabla'
            self.ngal = mask.sum()
        else:
            return self.data[mask]

    def previous_snapshot(self):
        """Return the state of each galaxy in the catalog in the
        previous snapshot.

        More specifically, return the GalaxyID, GroupID and
        SubGroupNumber (0 for centrals, >0 for subhalos)
        """
        last_GalaxyID = self.data['GalaxyID'] + 1
        last_snapshot = self.snapshot - 1
        last_filename = os.path.join(
            self.path, 'snapshot{0}'.format(last_snapshot),
            self.filename.split('/')[-1])
        last_data = set(self.read(last_filename))
        match = np.array(
            [i for i, gal in enumerate(last_GalaxyID)
             if gal in last_data['GalaxyID']])
        

    def query_history(self, moment):
        """
        Write an SQL query to retrieve the properties of present-day
        satellites at infall

        Parameters
        ----------
        moment : str {'infall', 'central'}
            Whether to retrieve information at the time of infall or the
            last snapshot where the galaxy was a central
        """
        baseurl = "http://galaxy-catalogue.dur.ac.uk:8080/" \
                  "Eagle?action=doQuery&SQL=select"
        # should probably be a separate function from here on
        if moment == 'infall':
            requirement = _requirement_infall
        elif moment == 'central':
            requirement = _requirement_central
        constraints = []
        for i, galid, topleafid, groupid in zip(
                count(), self.data['GalaxyID'], self.data['TopLeafID'],
                self.data['GroupID']):
            for j in range(topleafid-galid):
                constraints.append(
                    '(GalaxyID = {0} and SnapNum = {1} and {2})'.format(
                        galid+j, self.snapshot-j, self.requirement(groupid)))
            constraints = ' or '.join(constraints)
        # here should probably ask for all the same crap I ask for
        # in query_database.sh for the "parent" galaxies. That will
        # mean copying all the aliases and stuff
        #sql = 'GalaxyID, GroupID, SnapNum, 

    def read(self, filename, file_format='ascii'):
        if file_format == 'ascii':
            data = ascii.read(filename)
        elif file_format == 'fits':
            data = fits.getdata(filename)
        return data

    def trace_history(self, moment):
        """
        Search the local database for ascendants of galaxies in the
        catalog

        Parameters
        ----------
        moment : str {'infall', 'central'}
            Whether to retrieve information at the time of infall or the
            last snapshot where the galaxy was a central
        """
        
        #for s in range(self.snapshot, -1, -1):
            



class LocalDatabase:
    """Class and methods handling the local database"""

    def __init__(self):
        return


    
