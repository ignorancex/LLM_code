"""Class for pixel grids in one sensitivity scan."""

from functools import partial
from tqdm import tqdm

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def grid_edges(pixel_size_func, a_min, a_max, a_mid):
    """Pixelization grid edges consistent with a pixel size function."""
    grid_edges = [a_mid]
    a = a_mid
    a += pixel_size_func(a)
    while a < a_max:
        grid_edges.append(a)
        a += pixel_size_func(a)
    grid_edges.append(a_max)
    
    a = a_mid
    a -= pixel_size_func(a)
    while a > a_min:
        grid_edges.insert(0, a)
        a -= pixel_size_func(a)
    grid_edges.insert(0, a_min)
    
    return np.array(grid_edges)


class PixelConfig:
    """
    Pixel grid configurations for a given telescope, handling the spatial and frequency dimensions.

    Args:
        telescope (Telescope)
        n_nu (int): The number of frequency bins.
        n_ra_shift (int): The number of right ascension shifts.
        n_dec_shift (int): The number of declination shifts.
        wdir (str): Working directory.
        comment (str): Comment to add to the name.
    """
    def __init__(self, telescope, n_nu, n_ra_shift, n_dec_shift, wdir, comment=None):
        
        self.telescope = telescope
        self.n_nu = n_nu
        self.n_ra_shift = n_ra_shift
        self.n_dec_shift = n_dec_shift
        self.wdir = wdir

        self.name = f'{self.telescope.name}-nnu{self.n_nu}-nra{self.n_ra_shift}-ndec{self.n_dec_shift}'
        if comment is not None:
            self.name += f'-{comment}'
        
        self.save_dir = f'{self.wdir}/{self.name}'
        self.nu_s = np.geomspace(telescope.nu_min, telescope.nu_max, n_nu)

    def build(self, i_nu, i_ra_shift, i_dec_shift):

        self.postfix = f'inu{i_nu}-ira{i_ra_shift}-idec{i_dec_shift}'

        self.nu = self.nu_s[i_nu]

        ra_mid_base = (self.telescope.survey_ra_max + self.telescope.survey_ra_min) / 2
        self.ra_mids = ra_mid_base + np.linspace(0, self.telescope.ra_pixel_size(self.nu, ra_mid_base), self.n_ra_shift, endpoint=False)
        dec_mid_base = self.telescope.dec
        self.dec_mids = dec_mid_base + np.linspace(0, self.telescope.dec_pixel_size(self.nu, dec_mid_base), self.n_dec_shift, endpoint=False)

        self.ra_mid = self.ra_mids[i_ra_shift]
        self.dec_mid = self.dec_mids[i_dec_shift]
        self.ra_edges = grid_edges(
            partial(self.telescope.ra_pixel_size, self.nu),
            self.telescope.survey_ra_min,
            self.telescope.survey_ra_max,
            self.ra_mid
        )
        self.dec_edges = grid_edges(
            partial(self.telescope.dec_pixel_size, self.nu),
            self.telescope.survey_dec_min(self.nu),
            self.telescope.survey_dec_max(self.nu),
            self.dec_mid
        )
        self.ra_s = (self.ra_edges[1:] + self.ra_edges[:-1]) / 2
        self.dec_s = (self.dec_edges[1:] + self.dec_edges[:-1]) / 2
        self.n_ra = len(self.ra_s)
        self.n_dec = len(self.dec_s)
        self.ra_grid, self.dec_grid = np.meshgrid(self.ra_s, self.dec_s)
        self.radec_shape = (len(self.dec_s), len(self.ra_s))

        coord_grid = SkyCoord(ra=self.ra_grid*u.rad, dec=self.dec_grid*u.rad, frame='icrs')
        self.l_grid = np.array(coord_grid.galactic.l.rad)
        self.b_grid = np.array(coord_grid.galactic.b.rad)

    @property
    def radec_flat(self):
        return np.stack([self.ra_grid.ravel(), self.dec_grid.ravel()], axis=-1)
    
    @property
    def lb_flat(self):
        return np.stack([self.l_grid.ravel(), self.b_grid.ravel()], axis=-1)
    
    @property
    def pixel_area_map(self):
        """Pixel area map [sr]."""
        return np.outer(
            (self.dec_edges[1:] - self.dec_edges[:-1]),
            (self.ra_edges[1:] - self.ra_edges[:-1])
        ) * np.cos(self.dec_grid)
    
    @property
    def exposure_map(self):
        """Total exposure map [s]."""
        return self.telescope.t_obs(self.nu, self.dec_grid)
    
    def iter_over_func(self, func, **kwargs):
        pbar = tqdm(total=self.n_nu * self.n_ra_shift * self.n_dec_shift)
        for i_nu in range(self.n_nu):
            for i_ra in range(self.n_ra_shift):
                for i_dec in range(self.n_dec_shift):
                    
                    self.build(i_nu, i_ra, i_dec)
                    func(self, **kwargs)
                    pbar.update()