import pytest
from pathlib import Path

import numpy as np

from plotter import combine_axes
from histogram import HistogramData
from histogram import HistogramData1D
from histogram import HistogramData2D

from utils.postprocess import collision_stamp
from utils.gen_utils import mathematica_to_python


plot_1d_params = {
    'axes.labelsize': 20,
    'ylim': (0, 1.3),
    'xlim': (1e-4, 1e0),
    'xlabel': r'$\theta_1$',
    'ylabel': r'$\frac{\text{d}\Sigma}{\text{d}\log_{10}\theta_1}$',
    'x_scale': 'log'
}
old_plot_1d_params = {
    'axes.labelsize': 20,
    'ylim': (0, 1.3),
    'xlim': (1e-4, 1e0),
    'xlabel': r'$\theta_1$',
    'ylabel': r'$\frac{\text{d}\Sigma}{\text{d}\log_{10}\theta_1}$',
    'x_scale': 'log'
}
density_plot_params = {
    'axes.labelsize': 20,
    'ylim': (0, 1.0),
    'xlim': (1e-3, 1e0),
    'xlabel': r'$\theta_1$',
    'ylabel': r'$\theta_2/\theta_1$',
    'x_scale': 'log'
}
old_density_plot_params = {
    'axes.labelsize': 20,
    'ylim': (0, 1.0),
    'xlim': (1e-3, 1e0),
    'xlabel': r'$\theta_L$',
    'ylabel': r'$\theta_S/\theta_L$',
    'x_scale': 'log'
}
bullseye_plot_params = {
    'axes.labelsize': 20,
    'ylim': (0, 1.0),
}

@pytest.mark.parametrize("file_path, plot_kwargs, normalized", [
    ("tests/data/oneangle_100_10bins_nu1-00.py",
     {}, True),
    ("tests/data/oneangle_100_20bins_nu1-00.py",
     {}, True),
    ("tests/data/oneangle_1k_50bins_nu1-00.py",
     {}, True),
    ("tests/data/oneangle_1k_100bins_nu1-00.py",
     {}, True),
    ("tests/data/oneangle_10k_50bins_nu1-00.py",
     {}, True),
    ("tests/data/oneangle_10k_100bins_nu1-00.py",
     plot_1d_params, True),
    ("tests/data/twoangle_integrated_100_10bins_nus_1-00_1-00.py",
     {}, True),
    ("tests/data/twoangle_integrated_100_20bins_nus_1-00_1-00.py",
     {}, True),
    ("tests/data/twoangle_integrated_1k_50bins_nus_1-00_1-00.py",
     {}, True),
    ("tests/data/twoangle_integrated_1k_100bins_nus_1-00_1-00.py",
     {}, True),
    ("tests/data/twoangle_integrated_10k_50bins_nus_1-00_1-00.py",
     {}, True),
    ("tests/data/twoangle_integrated_10k_100bins_nus_1-00_1-00.py",
     plot_1d_params, True),
    ("../output/new_encs/2particle_od_100k_200bins_nu1-00.py",
     plot_1d_params, True),
])
def test_1d_histogram(file_path, plot_kwargs, normalized):
    # Histogram validation test
    hist1d = HistogramData1D(file_path,
                                         validate=True)

    # Normalization test
    if normalized:
        hist1d.integrate_histogram(scheme='log10',
                                   outflow_weight=1)
        assert np.isclose(hist1d.integral, 1.0, rtol=5e-2)

    # plotting test
    if plot_kwargs:
        hist1d.make_plot(**plot_kwargs)

        # check that the plot object exists
        assert hist1d.plot is not None

        # putting stamp on histogram
        hist1d.metadata.update({'jet_alg': 'akt',
                                'jet_rad': 0.5,
                                'pt_min': 500, 'pt_max': 550})
        collision_stamp(hist1d.plot.axes[0], hist1d.metadata)

        # saving
        hist1d.plot.savefig(Path(file_path).stem+'.pdf',
                            'tests/figures')


@pytest.mark.parametrize("file_path", [
    ("tests/data/oneangle_10k_100bins_nu1-00.py"),
    ("tests/data/twoangle_integrated_10k_100bins_nus_1-00_1-00.py"),
    ("../output/new_encs/2particle_od_100k_200bins_nu1-00.py"),
])
def test_ankita_comparison(file_path):
    plot_kwargs = plot_1d_params

    # Histogram validation test
    hist1d = HistogramData1D(file_path,
                                         validate=True)

    # Plotting my own data
    hist1d.make_plot(**plot_kwargs, label='Sam')

    # Plotting Ankita's data
    if hasattr(hist1d.metadata['weight'], '__iter__'):
        xs, ys = mathematica_to_python(
                        filename='tests/data/ankita_e3c')
    else:
        xs, ys = mathematica_to_python(
                        filename='tests/data/ankita_eec')
    ys = np.array(ys)
    ys = ys / 100000.
    hist1d.plot.axes[0].plot(xs, ys, label='Ankita',
                             color='lightcoral')

    # Putting stamp on histogram
    hist1d.metadata.update({'jet_alg': 'akt',
                            'jet_rad': 0.5,
                            'pt_min': 500, 'pt_max': 550})
    collision_stamp(hist1d.plot.axes[0], hist1d.metadata)

    # Legend and saving
    hist1d.plot.axes[0].legend()
    hist1d.plot.savefig("compare_"+Path(file_path).stem+'.pdf',
                        'tests/figures')


@pytest.mark.parametrize("file_path_1d, file_path_3d, plot_kwargs", [
    ("tests/data/twoangle_integrated_10k_100bins_nus_1-00_1-00.py",
     "tests/data/twoangle_10k_100bins_nus_1-00_1-00.py",
     plot_1d_params)
])
def test_1d_vs_3d(file_path_1d, file_path_3d, plot_kwargs):
    # Histogram validation test
    hist1d= HistogramData1D(file_path_1d, validate=True)

    # Plotting 1d histogram
    hist1d.make_plot(**plot_kwargs, color='cornflowerblue',
                     linewidth=4, label='1d')

    try:
        hist3d = HistogramData2D(file_path_3d,
             variable_order=['theta1', 'theta2_over_theta1', 'phi'])
        t1_key = 'theta1'
        t2_t1_key = 'theta2_over_theta1'
    except:
        hist3d = HistogramData(file_path_3d, variable_order=['thetaL',
                                               'thetaS_over_thetaL',
                                               'phi'])
        t1_key = 'thetaL'
        t2_t1_key = 'thetaS_over_thetaL'

    hist2d_integrated  = hist3d.integrate_over_variable('phi')
    hist1d_integrated  = HistogramData1D(
                            hist_data=hist2d_integrated.\
                                integrate_over_variable(t2_t1_key))

    # Plotting
    hist1d_integrated.make_plot(**plot_kwargs,
                                color='lightcoral',
                                label='Integrated 3d')

    # Combining plots
    combined_plotter = combine_axes([hist1d.plot.axes[0],
                                     hist1d_integrated.plot.axes[0]])

    ax = combined_plotter.axes[0]
    collision_stamp(ax, hist1d.metadata)
    ax.legend()
    combined_plotter.fig.tight_layout()
    combined_plotter.savefig('integration_test.pdf', 'tests/figures')


@pytest.mark.parametrize("file_path, plot_kwargs", [
    ("tests/data/test_hist2d.py", density_plot_params),
    ("tests/data/twoangle_100_1phi_20bins_nus_1-00_1-00.py",
     density_plot_params),
    ("../output/new_encs/3particle_od_100k_150bins_nus_1-00_1-00.py",
     density_plot_params),
])
def test_2d_histogram(file_path, plot_kwargs):
    hist2d = HistogramData(file_path, validate=False,
             variable_order=['theta1', 'theta2_over_theta1', 'phi'])

    if len(hist2d.hist.shape) == 3:
        hist2d = HistogramData2D(
            hist_data=hist2d.integrate_over_variable('phi'))
    else:
        # Remove irrelevant 'phi' information
        hist2d.edges.pop('phi')
        hist2d.centers.pop('phi')
        hist2d.variable_order = ['theta1', 'theta2_over_theta1']
        hist2d = HistogramData2D(hist_data=hist2d)

    hist2d.validate()

    # Remove outflow bins for theta1 and validate
    finite_theta_edges = hist2d.edges['theta1'][1:-1]
    finite_theta_centers = hist2d.centers['theta1'][1:-1]
    finite_hist2d = [np.array(hist1d[1:-1])
                     for hist1d in hist2d.hist.T]
    hist2d.hist = np.array(finite_hist2d).T
    hist2d.edges['theta1'] = finite_theta_edges
    hist2d.centers['theta1'] = finite_theta_centers
    hist2d.validate()

    # Testing plots:
    if plot_kwargs:
        # Test log-normalized plot
        hist2d.make_plot('density', log_norm=True, **plot_kwargs)
        hist2d.density.savefig(Path(file_path).stem\
                               +'_lognorm_density.pdf',
                               'tests/figures')

        # Test linear-normalized plot
        hist2d.make_plot('density', log_norm=False, **plot_kwargs)
        hist2d.density.savefig(Path(file_path).stem\
                               +'_linnorm_density.pdf',
                               'tests/figures')


@pytest.mark.parametrize("file_path, plot_kwargs", [
    ("tests/data/twoangle_100_20bins_nus_1-00_1-00.py",
     density_plot_params),
    ("tests/data/twoangle_1k_50bins_nus_1-00_1-00.py",
     density_plot_params),
    ("tests/data/twoangle_1k_100bins_nus_1-00_1-00.py",
     density_plot_params),
    ("tests/data/twoangle_10k_50bins_nus_1-00_1-00.py",
     density_plot_params),
    ("tests/data/twoangle_10k_100bins_nus_1-00_1-00.py",
     density_plot_params),
    ("tests/data/old_100_20bins.py", old_density_plot_params),
    ("tests/data/old_1k_50bins.py", old_density_plot_params),
    ("tests/data/old_1k_100bins.py", old_density_plot_params),
    ("../output/new_encs/3particle_od_100k_150bins_nus_1-00_1-00.py",
     density_plot_params),
])
def test_3d_histogram(file_path, plot_kwargs):
    try:
        hist3d = HistogramData(file_path, variable_order=['theta1',
                                               'theta2_over_theta1',
                                               'phi'])
        integration_scheme_3d = {'theta1': 'log10',
                                 'theta2_over_theta1': 'linear',
                                 'phi': 'linear'}
        t1_key = 'theta1'
    except:
        hist3d = HistogramData(file_path, variable_order=['thetaL',
                                               'thetaS_over_thetaL',
                                               'phi'])
        integration_scheme_3d = {'thetaL': 'log10',
                                 'thetaS_over_thetaL': 'linear',
                                 'phi': 'linear'}
        t1_key = 'thetaL'

    # Check that the object is created
    assert hist3d.hist is not None

    # Normalization test
    try:
        hist3d.integrate_histogram(scheme=integration_scheme_3d,
                                   outflow_weight=1)
    except Exception as err:
        print()
        print(hist3d.hist.shape)
        print([hist3d.edges[k].shape
               for k in hist3d.variable_order])
        print([hist3d.centers[k].shape
               for k in hist3d.variable_order])
        raise err
    assert np.isclose(hist3d.integral, 1.0, rtol=6e-2)

    # Testing integration over phi:
    hist2d = HistogramData2D(
                hist_data=hist3d.integrate_over_variable('phi'))

    # Testing new normalization
    integration_scheme_2d = integration_scheme_3d.copy()
    integration_scheme_2d.pop('phi')
    hist2d.integrate_histogram(scheme=integration_scheme_2d,
                               outflow_weight=1)
    assert np.isclose(hist2d.integral, hist3d.integral, rtol=1e-3)

    # Testing plots:
    if plot_kwargs:
        # Remove outflow bins for theta1 and validate
        finite_theta_edges = hist2d.edges[t1_key][1:-1]
        finite_theta_centers = hist2d.centers[t1_key][1:-1]
        finite_hist2d = [np.array(hist1d[1:-1])
                         for hist1d in hist2d.hist.T]
        hist2d.hist = np.array(finite_hist2d).T
        hist2d.edges[t1_key] = finite_theta_edges
        hist2d.centers[t1_key] = finite_theta_centers
        hist2d.validate()

        # Test log-normalized plot
        hist2d.make_plot('density', log_norm=True, **plot_kwargs,
                         vmin=1e-4)
        hist2d.density.savefig(Path(file_path).stem\
                               +'_lognorm_density.pdf',
                               'tests/figures')

        # Test linear-normalized plot
        hist2d.make_plot('density', log_norm=False, **plot_kwargs)
        hist2d.density.savefig(Path(file_path).stem\
                               +'_linnorm_density.pdf',
                               'tests/figures')


@pytest.mark.parametrize("file_path, plot_kwargs", [
    ("tests/data/test_hist3d.py", bullseye_plot_params),
    ("tests/data/twoangle_100_20bins_nus_1-00_1-00.py", {}),
    ("tests/data/twoangle_1k_50bins_nus_1-00_1-00.py", {}),
    ("tests/data/twoangle_1k_100bins_nus_1-00_1-00.py", {}),
    ("tests/data/twoangle_10k_50bins_nus_1-00_1-00.py", {}),
    ("tests/data/twoangle_10k_100bins_nus_1-00_1-00.py",
     bullseye_plot_params),
    ("tests/data/old_100_20bins.py", {}),
    ("tests/data/old_1k_50bins.py", bullseye_plot_params),
    ("tests/data/old_1k_100bins.py", bullseye_plot_params),
    ("../output/new_encs/3particle_od_100k_150bins_nus_1-00_1-00.py",
     bullseye_plot_params),
])
def test_dimensional_reduction(file_path, plot_kwargs):
    try:
        hist3d = HistogramData(file_path, variable_order=['theta1',
                                               'theta2_over_theta1',
                                               'phi'])
        t1_key = 'theta1'
        t2_t1_key = 'theta2_over_theta1'
        hist3d.edges[t1_key]
    except:
        hist3d = HistogramData(file_path, variable_order=['thetaL',
                                                'thetaS_over_thetaL',
                                                'phi'])
        t1_key = 'thetaL'
        t2_t1_key = 'thetaS_over_thetaL'
        hist3d.edges[t1_key]


    # Fix a theta1 value and find the associated bin
    theta1_val = 0.2
    theta1_ind = np.digitize([theta1_val],
                             hist3d.edges[t1_key])[0] - 1
    t1_bin = (hist3d.edges[t1_key][theta1_ind],
              hist3d.edges[t1_key][theta1_ind+1])

    # Get sub-histogram and normalize
    bullseye2d = hist3d.get_sub_histogram(t1_key, theta1_val)
    bullseye2d = HistogramData2D(hist_data=bullseye2d)

    # Normalize correctly
    bin2_centers = bullseye2d.centers[t2_t1_key]
    theta2_centers = bin2_centers * theta1_val
    bullseye2d.hist = np.array([
        bullseye2d.hist[i2] / (theta2_centers[i2] * theta1_val**2)
        for i2 in range(len(theta2_centers))])

    # Update variable names for plotting
    bullseye2d.edges[t2_t1_key] = bin2_centers*theta1_val
    bullseye2d.centers[t2_t1_key] = theta2_centers

    # Plot bullseye and save
    if plot_kwargs:
        plot_kwargs.update({'ylim': (0, theta1_val)})
        bullseye2d.make_plot('bullseye', log_norm=True,
                             radial_coordinate=t2_t1_key,
                             **plot_kwargs)

        # Set labels and limits
        ax = bullseye2d.bullseye.axes[0]
        ax.set_ylim(0, t1_bin[1])
        theta1_bounds = fr"${t1_bin[0]} < \theta_1 < {t1_bin[-1]}$"
        ax.text(0.7, 1.05, theta1_bounds, transform=ax.transAxes)

        # Boundaries for old definition in terms of thetaL, thetaS
        if "old" in file_path:
            phi_bound = np.linspace(-np.pi/2, np.pi/2, 1000)
            r_bound = np.minimum(2*np.cos(phi_bound),
                                 1 / (2 * np.cos(phi_bound)))
            ax.fill(phi_bound, theta1_val*r_bound, color='lightblue',
                    alpha=0.05)
            ax.plot(phi_bound, theta1_val*r_bound, color='blue')


        # Save
        bullseye2d.bullseye.savefig(Path(file_path).stem\
                                    +'_bullseye.pdf',
                                    'tests/figures')


# ====================================
# Run with pytest
# ====================================
if __name__ == "__main__":
    pytest.main()
