"""Sky Processing."""
import astropy.coordinates as apc
import astropy.units as apu
import astropy.time as apt
import healpy as hp
import numpy as np
import matplotlib
import tqdm
from foreground_map_visual import plot_healpy

font = {'weight': 'normal', 'size': 12}

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rc('font', **font)


def rotate_map(map_in, location_zenith, obstime, orientation):
    """
    Rotate a given healpix map.
    The function rotates the heal pix map onto the altaz coordinaes of an
    antenna at the specified location and time.
    """
    nside = hp.npix2nside(map_in.shape[0])
    theta_list, phi_list = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    phi_list = np.subtract(phi_list, orientation)
    theta_list = np.subtract(np.pi / 2, theta_list)
    phi_list = np.mod(np.subtract(np.pi / 2, phi_list), 2 * np.pi)

    ant_coords = apc.SkyCoord(az=phi_list * apu.rad, alt=theta_list * apu.rad,
                              unit=apu.rad, obstime=obstime,
                              location=location_zenith, frame='altaz')
    b_gal_full = np.radians(90 - ant_coords.galactic.b.value)
    l_gal_full = np.radians(ant_coords.galactic.l.value)

    map_out = hp.get_interp_val(map_in, b_gal_full, l_gal_full)
    return map_out


def map_integ(map_in, start_time, lon, lat, h, orientation, time_array):
    """
    Rotation of a map repeatedly for a specified array of observing times.
    After rotation the function integrates over those times.
    """
    location_zenith = apc.EarthLocation.from_geodetic(lat=lat, lon=lon, height=h)

    if time_array is None:
        obstime = start_time
        map_av = rotate_map(map_in, location_zenith, obstime, orientation)

    else:

        map_time = np.zeros(len(map_in))

        for i in tqdm.tqdm(np.arange(len(time_array))):
            obstime = start_time + time_array[i] * apu.min
            print(obstime)
            map_time = np.add(map_time, rotate_map(map_in, location_zenith, obstime, orientation))

        map_av = map_time / len(time_array)
    return map_av


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    haslam_408 = np.load("./../data/baseMaps/base_map_230.npy")
    # haslam_408 = hp.ud_grade(haslam_408, 128)
    haslam_408_rot = rotate_map(map_in=haslam_408,
                                location_zenith=apc.EarthLocation.from_geodetic(lon=21.4476236, lat=-30.71131, height=25),
                                obstime=apt.Time('2019-10-01 00:00:00', scale='utc'),
                                orientation=0)

    plot_healpy(data=haslam_408_rot, max_val=1000, show=False, cmap='inferno',
                save=True, filename="gsm_00-00-00.pdf", cbar=False)
    
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    fig.colorbar(image, ax=ax, orientation='horizontal', aspect=60, extend='max',
                 pad=0.09, label=r'$T_{\rm b}\ (\rm K)$')
    plt.savefig('gsm_00-00-00.pdf', bbox_inches='tight')
