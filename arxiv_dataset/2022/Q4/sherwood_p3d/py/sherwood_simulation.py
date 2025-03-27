import numpy as np


def redshift_from_snapshot(snap):
    """Return redshift for a given simulation snapshot"""

    # right now we only use 4 snapshots
    switcher={8:3.2, 9:2.8, 10:2.4, 11:2.0}

    return switcher.get(snap,"Unknown snapshot "+str(snap))


def snapshot_from_redshift(z):
    """Return snapshot corresponding to a given redshift"""

    # right now we only use 4 snapshots
    switcher={3.2:8, 2.8:9, 2.4:10, 2.0:11}

    return switcher.get(z,"Unknown snapshot "+str(z))


class SherwoodSimulation(object):
    """Object describing one of the Sherwood simulations."""

    def __init__(self,L_hMpc=80,n_part=1024):
        """Specify simulation (box size and number of particles) """

        # store information about the simulation
        self.L_hMpc=L_hMpc
        self.n_part=n_part


class Grid(object):
    """Define regular grid in a simulated box. Inputs:
      - simulation: SherwoodSimulation object with basic info about sim.
      - snapshot_num: snapshot number (8, 9, 10 or 11)
      - n_xy: number of cells per in transverse directions (x,y)
      - n_z: number of cells along the line of sight (if None, use n_xy)
      - axis: box axis along which we set line of sight (1,2,3)"""

    def __init__(self,simulation,snapshot_num=9,n_xy=1024,n_z=2048,axis=0):

        self.sim = simulation
        self.snapshot_num = snapshot_num
        self.n_xy = n_xy
        self.n_z = n_z
        self.axis = axis


    def get_z(self):
        """Return redshift of snapshot"""

        # get redshift from hard-coded relation to snapshot number
        z = redshift_from_snapshot(self.snapshot_num)

        return z


    def get_nametag(self):
        """String identifying base grid, useful to create unique filenames"""

        # add information about simulation box
        nametag='{}_{}'.format(self.sim.L_hMpc,self.sim.n_part)
        # add snapshot
        nametag+='_{}'.format(self.snapshot_num)
        # add axis and number of cells
        nametag+='_{}_{}_{}'.format(self.axis,self.n_xy,self.n_z)

        return nametag



class HaloGrid(Grid):
    """Define regular grid of halo density from simulation. Inputs:
      - simulation: SherwoodSimulation object with basic info about sim.
      - snapshot_num: snapshot number (8, 9, 10 or 11)
      - logMh_min: minimum (log10) halo mass (in solar masses over h)
      - logMh_max: minimum (log10) halo mass (in solar masses over h)
      - n_xy: number of cells in the transverse directions (x, y)
      - n_z: number of cells in the line-of-sight directions (z)
      - axis: box axis to use for redshift direction (1,2,3)
      - add_rsd: set to true to add redshift-space distortions. """

    def __init__(self,simulation,snapshot_num=9,logMh_min=None,logMh_max=None,
                n_xy=1024,n_z=2048,axis=0,add_rsd=True):

        super().__init__(simulation,snapshot_num,n_xy=n_xy,n_z=n_z,axis=axis)

        self.logMh_min = logMh_min
        self.logMh_max = logMh_max
        self.add_rsd = add_rsd


    def get_nametag(self):
        """String identifying halo grid, useful to create unique filenames"""

        # get base nametag
        nametag=super().get_nametag()
        # add mass bin (might be None) 
        if self.logMh_min is None: str_logMh_min='None'
        else: str_logMh_min='{:.2f}'.format(self.logMh_min)
        if self.logMh_max is None: str_logMh_max='None'
        else: str_logMh_max='{:.2f}'.format(self.logMh_max)
        nametag+='_{}_{}'.format(str_logMh_min,str_logMh_max)
        # add RSD info
        if self.add_rsd: 
            nametag+='_rsd'

        return nametag


