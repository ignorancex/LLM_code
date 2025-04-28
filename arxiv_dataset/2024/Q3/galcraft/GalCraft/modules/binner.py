from . import utils
import numpy as np
from astropy.table import Table
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def spatial_binner(d_t, cube_params, other_params, age_grid, metal_grid, alpha_grid,
                   filepath, logger, input_arg, galaxy_dist, nparticles):
    '''
    Preparing the stellar particle catalog, N_particle spatial distribution,
    x and y axis edges from the loaded particle catalog and data cube configurations.
    :param d_t: stellar particle catalog
    :param cube_params: cube configurations
    :param other_params: other configurations
    :param age_grid: age grid of SSP templates
    :param metal_grid: metallicity grid of SSP templates
    :param alpha_grid: alpha grid of SSP templates
    :param filepath: outputs file path of this run
    :param logger: log file
    :param input_arg: setup input cube name path
    :param galaxy_dist: galaxy distance to the observer
    :param nparticles: number of particles in the catalog
    :return: list of stellar particle catalog, N_particle spatial distribution,
    x and y axis edges to be used for each data cube run.
    '''

    spatial_resolution = cube_params['spatial_resolution']
    spatial_resolution_deg = np.array(spatial_resolution) / 3600.
    spatial_resolution_x = spatial_resolution_deg[0]
    spatial_resolution_y = spatial_resolution_deg[1]

    x_coord = cube_params['x_coord']
    y_coord = cube_params['y_coord']


    if cube_params['instrument'].upper() != "DEFAULT":
        logger.info('Load the cube list from %s' % (input_arg + '_list'))
        cube_centers = np.genfromtxt(input_arg + '_list', dtype=float, delimiter=',')
        if len(cube_centers.shape) == 1:
            cube_centers = np.array([cube_centers])
        logger.info('Load the list of center positions of data cubes.')
        nbin_x = cube_params['spatial_nbin'][0]
        nbin_y = cube_params['spatial_nbin'][1]
    else:
        logger.info(
            'The instrument is in %s, will generate a huge cube using all the particles.' % cube_params['instrument'].upper())
        if cube_params['spatial_percentile'] == True:
            x_edges = np.arange(np.percentile(d_t[x_coord], 0.05),
                                np.percentile(d_t[x_coord], 99.95) + spatial_resolution_x, spatial_resolution_x)
            y_edges = np.arange(np.percentile(d_t[y_coord], 0.05),
                                np.percentile(d_t[y_coord], 99.95) + spatial_resolution_y, spatial_resolution_y)
        else:
            x_edges = np.arange(np.nanmin(d_t[x_coord]), np.nanmax(d_t[x_coord]) + spatial_resolution_x,
                                spatial_resolution_x)
            y_edges = np.arange(np.nanmin(d_t[y_coord]), np.nanmax(d_t[y_coord]) + spatial_resolution_y,
                                spatial_resolution_y)
        nbin_x = x_edges.shape[0] - 1
        nbin_y = y_edges.shape[0] - 1
        cube_centers = np.array([[(x_edges[-1] + x_edges[0]) / 2, (y_edges[-1] + y_edges[0]) / 2]])



    logger.info('================Cube info=================')
    logger.info('x_coordinate:          %s' % x_coord)
    logger.info('y_coordinate:          %s' % y_coord)
    logger.info('instrument:            %s' % cube_params['instrument'])
    logger.info('Spatial resolution:    %s' % spatial_resolution)
    logger.info('n_bins in x and y:     %s' % [nbin_x, nbin_y])
    logger.info('Number of cubes:       %s' % len(cube_centers))
    logger.info('         %s          %s' % (x_coord, y_coord))
    for i in range(len(cube_centers)):
        logger.info('      %.3f      %.3f' % (cube_centers[i][0], cube_centers[i][1]))
    logger.info('use losvd?:            %s' % cube_params['use_losvd'])
    logger.info('use extinc?:           %s' % cube_params['use_extinc'])
    logger.info('extinc factor:         %s' % cube_params['extinc_factor'])
    logger.info('use dist? :            %s' % cube_params['use_dist'])
    logger.info('add noise?:            %s' % cube_params['add_noise'])
    logger.info('bootstrapping?:        %s' % cube_params['bootstrap_table'])
    logger.info('==========================================')


    if 'cube_m_h' in d_t.keys() and 'cube_alpha_fe' in d_t.keys() and 'cube_age' in d_t.keys():
        logger.info("Not create the 'cube_x', 'mass', 'dist' columns, already exist.")
        d_t['cube_m_h'] = np.clip(d_t['cube_m_h'], metal_grid[0], metal_grid[-1])
        d_t['cube_alpha_fe'] = np.clip(d_t['cube_alpha_fe'], alpha_grid[0], alpha_grid[-1])
        d_t['cube_age'] = np.clip(d_t['cube_age'], age_grid[0], age_grid[-1])
        d_t['cube_logage'] = np.log10(d_t['cube_age']) + 9
    else:
        logger.info('Make the initial mass to be 1 M_sun.')
        d_t['mass'] = np.ones(d_t['m_ini'].shape)  # Make the mass equal to 1
        # np.clip the values in ([M/H], [alpha/Fe], age)
        logger.info('Transfer from [Fe/H] to [M/H], clipping [M/H], age and [alpha/Fe].')
        d_t['m_h'] = d_t['fe_h'] + np.log10(0.684 * (10 ** d_t['alpha_fe']) + 0.306)
        d_t['cube_m_h'] = np.clip(d_t['m_h'], metal_grid[0], metal_grid[-1])
        d_t['cube_alpha_fe'] = np.clip(d_t['alpha_fe'], alpha_grid[0], alpha_grid[-1])
        d_t['cube_age'] = np.clip(d_t['age'], age_grid[0], age_grid[-1])
        d_t['cube_logage'] = np.log10(d_t['cube_age']) + 9
        
    # Subdivide mass of the particle based on its alpha value
    # This fraction is not needed, but will keep in case one day will add it back
    d_t['fraction'] = 2.5 * d_t['cube_alpha_fe']


    # Add dist to the table
    if cube_params['use_dist']==True:
        d_t['dist'] = d_t['r']
    else:
        d_t['dist'] = np.ones(d_t['r'].shape) * galaxy_dist

    # Randomly add some numbers to bin_x and bin_y
    d_t['bin_x'] = np.zeros(len(d_t['mass']))
    d_t['bin_y'] = np.zeros(len(d_t['mass']))

    x_edges_plt = np.arange(np.percentile(d_t[x_coord], 0.05),
                            np.percentile(d_t[x_coord], 99.95) + spatial_resolution_deg[0], spatial_resolution_deg[0])
    y_edges_plt = np.arange(np.percentile(d_t[y_coord], 0.05),
                            np.percentile(d_t[y_coord], 99.95) + spatial_resolution_deg[1], spatial_resolution_deg[1])

    if other_params['plot_statistics'] == True:
        logger.info('Plotting the statistical distribution using several parameters using the original particles...')
        plt.figure(figsize=[12, 10])
        plt.subplot(321)
        im, ax, num_counts, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['vr'],
                                                                 statistic='count', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                                 xlabel=r"$" + x_coord + "$" + ' (degrees)',
                                                                 ylabel=r"$" + y_coord + "$" + ' (degrees)',
                                                                 cmap=plt.cm.hot, cblabel='Count', color_Lognorm=True)
        plt.subplot(322)
        im, ax, stats_age, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['cube_age'],
                                                                statistic='median', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                                xlabel=r"$" + x_coord + "$" + ' (degrees)',
                                                                ylabel=r"$" + y_coord + "$" + ' (degrees)',
                                                                cmap=plt.cm.Spectral_r, cblabel='Age (Gyr)', color_Lognorm=False)
        plt.subplot(323)
        im, ax, stats_feh, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['cube_m_h'],
                                                                statistic='median', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                                xlabel=r"$" + x_coord + "$" + ' (degrees)',
                                                                ylabel=r"$" + y_coord + "$" + ' (degrees)',
                                                                cmap=plt.cm.Spectral_r, cblabel='[M/H]', color_Lognorm=False)
        plt.subplot(324)
        im, ax, stats_alpha, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['cube_alpha_fe'],
                                                                  statistic='median', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                                  xlabel=r"$" + x_coord + "$" + ' (degrees)',
                                                                  ylabel=r"$" + y_coord + "$" + ' (degrees)',
                                                                  cmap=plt.cm.Spectral_r, cblabel='[Alpha/Fe]', color_Lognorm=False)
        plt.subplot(325)
        im, ax, stats_vlos, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['vr'],
                                                                 statistic='median', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                                 xlabel=r"$" + x_coord + "$" + ' (degrees)',
                                                                 ylabel=r"$" + y_coord + "$" + ' (degrees)',
                                                                 cmap=plt.cm.Spectral, cblabel='V_los (km/s)', color_Lognorm=False)
        plt.subplot(326)
        im, ax, stats_vsig, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['vr'],
                                                                 statistic='std', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                                 xlabel=r"$" + x_coord + "$" + ' (degrees)',
                                                                 ylabel=r"$" + y_coord + "$" + ' (degrees)',
                                                                 cmap=plt.cm.Spectral_r, cblabel='V_std (km/s)', color_Lognorm=False)
        plt.tight_layout()
        plt.savefig(filepath + 'particles_distrib.png', dpi=150)
        logger.info('The plot has been finished and saved into the outputs folder.')



    d_t = Table(d_t)
    if cube_params['invert_b'] == True:
        d_t['b'] = -d_t['b']
        logger.info('Invert b coordinates due to the wrong inclination used.')
    if cube_params['bootstrap_table'] == True:
        d_t = d_t[np.random.randint(len(d_t), size=len(d_t))]
        logger.info('Randomly Sample the table for a bootstrapping process.')
    if other_params['mode'] == 'debug':
        d_t = d_t[np.random.randint(len(d_t), size=nparticles)]
        logger.info("Running mode is 'debug', randomly selecting %s particles." % nparticles)
        if other_params['write_table'] == True:
            logger.info('Writing the model into astropy.table')
            d_t.write(filepath + 'particle_table.fits', format='fits', overwrite=True)



    d_t_l = []
    statistic_count_l = []
    x_edges_l = []
    y_edges_l = []

    for i in range(len(cube_centers)):
        cube_center_i = cube_centers[i]
        cube_center_i_x = cube_center_i[0]
        cube_center_i_y = cube_center_i[1]
        start_x = cube_center_i_x - float(nbin_x) / 2 * spatial_resolution_x
        start_y = cube_center_i_y - float(nbin_y) / 2 * spatial_resolution_y
        end_x = cube_center_i_x + float(nbin_x) / 2 * spatial_resolution_x
        end_y = cube_center_i_y + float(nbin_y) / 2 * spatial_resolution_y
        x_edges = np.arange(start_x, end_x + spatial_resolution_x, spatial_resolution_x)
        y_edges = np.arange(start_y, end_y + spatial_resolution_y, spatial_resolution_y)

        statistic_count, x_edges, y_edges, bin_index = binned_statistic_2d(x=d_t[x_coord], y=d_t[y_coord],
                                                                           values=d_t['mass'], statistic='count',
                                                                           bins=[x_edges, y_edges],
                                                                           expand_binnumbers=True)
        d_t['bin_x'] = bin_index[0, :]
        d_t['bin_y'] = bin_index[1, :]

        d_t_i = d_t[(bin_index[0, :] != 0) & (bin_index[0, :] != x_edges.shape[0]) &
                    (bin_index[1, :] != 0) & (bin_index[1, :] != y_edges.shape[0])]

        if other_params['mode'] != 'debug':
            if cube_params['use_extinc'] == False:
                d_t_i.keep_columns(['ra', 'dec', 'l', 'b', 'vr', 'cube_m_h', 'cube_alpha_fe', 'cube_age', 'cube_logage',
                                    'mass', 'bin_x', 'bin_y', 'fraction', 'r', 'dist'])
            else:
                d_t_i.keep_columns(['ra', 'dec', 'l', 'b', 'vr', 'cube_m_h', 'cube_alpha_fe', 'cube_age', 'cube_logage',
                                    'mass', 'bin_x', 'bin_y', 'fraction', 'r', 'dist', 'exbv'])


        if other_params['write_table'] == True:
            logger.info('Write the E-Galaxia model into astropy.Table of particles %s in datacube No.%s' % (len(d_t_i), i + 1))
            d_t_i.write(filepath + 'particle_table_' + str(i) + '.fits', format='fits', overwrite=True)


        # save the statistic_count into npy array
        np.save(filepath + 'statistic_count' + str(i) + '.npy', np.flip(statistic_count.T, axis=1))
        logger.info('Write the number of stars array into file using particles %s in datacube No.%s' % (len(d_t_i), i + 1))

        d_t_l.append(d_t_i)
        statistic_count_l.append(statistic_count)
        x_edges_l.append(x_edges)
        y_edges_l.append(y_edges)


    logger.info('Plotting the distribution of the datacubes on the Galaxy.')
    plt.figure(figsize=[6, 2.5])
    ax = plt.subplot(111)
    im, ax, num_counts, xbins, ybins = utils.plot_binned_grids_color(x=d_t[x_coord], y=d_t[y_coord], values=d_t['vr'],
                                                         statistic='count', x_edges=x_edges_plt, y_edges=y_edges_plt,
                                                         xlabel=r"$"+x_coord+"$" + ' (degrees)',
                                                         ylabel=r"$"+y_coord+"$" + ' (degrees)',
                                                         cblabel='Number Count', color_Lognorm=True, alpha=0.7,
                                                         plot_cb=False)
    plt.gca().set_aspect('equal')
    ax.set_xlim(x_edges_plt[-1], x_edges_plt[0])
    cbar = plt.colorbar(im, ax=ax, aspect=15, pad=0.01)
    cbar.set_label('Number Count', fontsize=8)
    for i in range(len(x_edges_l)):
        ax.add_patch(Rectangle((x_edges_l[i][0], y_edges_l[i][0]), x_edges_l[i][-1] - x_edges_l[i][0],
                               y_edges_l[i][-1] - y_edges_l[i][0],
                               alpha=1, facecolor='none', edgecolor='slateblue', linewidth=1.6))
    plt.tight_layout(pad=0.03)
    plt.savefig(filepath + 'datacube_distrib.png', dpi=150)
    logger.info('The figure was successfully saved.')

    np.savetxt(filepath + 'cube_list', cube_centers, delimiter=',')


    return [d_t_l, statistic_count_l, x_edges_l, y_edges_l]




def spatial_binner_continue(cube_params, other_params, filepath, logger, input_arg):
    '''
    Same as function spatial_binner but in "continue" mode
    '''

    spatial_resolution = cube_params['spatial_resolution']
    spatial_resolution_deg = np.array(spatial_resolution) / 3600.
    spatial_resolution_x = spatial_resolution_deg[0]
    spatial_resolution_y = spatial_resolution_deg[1]

    x_coord = cube_params['x_coord']
    y_coord = cube_params['y_coord']


    assert cube_params['instrument'].upper() != "DEFAULT", "There has to be an instrument assigned for running the 'continue' mode."

    logger.info('Load the cube list from %s' % (filepath + 'cube_list'))
    cube_centers = np.genfromtxt(input_arg + '_list', dtype=float, delimiter=',')
    if len(cube_centers.shape) == 1:
        cube_centers = np.array([cube_centers])
    logger.info('Load the list of center positions of data cubes.')
    nbin_x = cube_params['spatial_nbin'][0]
    nbin_y = cube_params['spatial_nbin'][1]



    logger.info('================Cube info=================')
    logger.info('x_coordinate:          %s' % x_coord)
    logger.info('y_coordinate:          %s' % y_coord)
    logger.info('instrument:            %s' % cube_params['instrument'])
    logger.info('Spatial resolution:    %s' % spatial_resolution)
    logger.info('n_bins in x and y:     %s' % [nbin_x, nbin_y])
    logger.info('Number of cubes:       %s' % len(cube_centers))
    logger.info('         %s          %s' % (x_coord, y_coord))
    for i in range(len(cube_centers)):
        logger.info('      %.3f      %.3f' % (cube_centers[i][0], cube_centers[i][1]))
    logger.info('use losvd?:            %s' % cube_params['use_losvd'])
    logger.info('use dist? :            %s' % cube_params['use_dist'])
    logger.info('add noise?:            %s' % cube_params['add_noise'])
    logger.info('bootstrapping?:        %s' % cube_params['bootstrap_table'])
    logger.info('==========================================')


    d_t_l = []
    statistic_count_l = []
    x_edges_l = []
    y_edges_l = []

    for i in range(len(cube_centers)):
        cube_center_i = cube_centers[i]
        cube_center_i_x = cube_center_i[0]
        cube_center_i_y = cube_center_i[1]
        start_x = cube_center_i_x - float(nbin_x) / 2 * spatial_resolution_x
        start_y = cube_center_i_y - float(nbin_y) / 2 * spatial_resolution_y
        end_x = cube_center_i_x + float(nbin_x) / 2 * spatial_resolution_x
        end_y = cube_center_i_y + float(nbin_y) / 2 * spatial_resolution_y
        x_edges = np.arange(start_x, end_x + spatial_resolution_x, spatial_resolution_x)
        y_edges = np.arange(start_y, end_y + spatial_resolution_y, spatial_resolution_y)


        d_t_i = Table.read(filepath + 'particle_table_' + str(i) + '.fits')
        logger.info(
            'Read the E-Galaxia model table of particles %s in datacube No.%s' % (len(d_t_i), i + 1))
        statistic_count = np.flip(np.load(filepath + 'statistic_count' + str(i) + '.npy'), axis=1).T
        logger.info(
            'Read the number of stars array into file using particles %s in datacube No.%s' % (len(d_t_i), i + 1))

        d_t_l.append(d_t_i)
        statistic_count_l.append(statistic_count)
        x_edges_l.append(x_edges)
        y_edges_l.append(y_edges)


    return [d_t_l, statistic_count_l, x_edges_l, y_edges_l]



