
import numpy as np
import pandas as pd
from collections import defaultdict
import arviz











def import_orbit_data(orbit_row):
    import math

    orbit_array = []
    for row in orbit_row:
        found_vals = False
        if not isinstance(row, float):
            for ii in range(0, len(row)):
                val = row[ii]
                if val == " " and not found_vals and ii > 1:
                    found_vals = True
                    orbit_array.append(np.array([float(row[1:ii]), float(row[ii:len(row)-1])]))

    orbit_array = np.array(orbit_array)


    #orbit_array = remove_nans(orbit_array)
    
    return orbit_array





def remove_nans(array):
    array = array[~np.isnan(array)]
    
    return array






def make_rv_astrometry_plots(sim_data_file, trace_file, nplanets, noRoman=False):
    import matplotlib.pyplot as plt

    

    
    #pull in simulated data
    simulated_data = pd.read_csv(sim_data_file)
    
    #simulated RV observations
    x_rv = simulated_data['times_rv_observed'].values
    y_rv = simulated_data['rv_observed'].values
    y_rv_err = simulated_data['rv_err_observed'].values
   
    #simulated RV full orbits
    x_rv_orbit = simulated_data['times_rv_orbit'].values
    y_rv_orbit = simulated_data['rv_orbit'].values
    
    #simulated astrometric observations
    x_astrometry = simulated_data['times_astrometry_observed'].values
    y_ra = simulated_data['ra_observed'].values
    y_dec = simulated_data['dec_observed'].values
    y_ra_err = simulated_data['ra_err_observed'].values
    y_dec_err = simulated_data['dec_err_observed'].values
    
    #simulated astrometric full orbits
    x_astrometry_orbit = simulated_data['times_astrometry_orbit'].values
    y_ra_orbit = simulated_data['ra_orbit'].values
    y_dec_orbit = simulated_data['dec_orbit'].values
     
    
    #remove any nans from simulated data (created because of mismatching lengths)
    x_rv = remove_nans(x_rv)
    y_rv = remove_nans(y_rv)
    y_rv_err = remove_nans(y_rv_err)
    x_rv_orbit = remove_nans(x_rv_orbit)
    y_rv_orbit = import_orbit_data(y_rv_orbit)
    x_astrometry = remove_nans(x_astrometry)
    y_ra = remove_nans(y_ra)
    y_dec = remove_nans(y_dec)
    y_ra_err = remove_nans(y_ra_err)
    y_dec_err = remove_nans(y_dec_err)
    x_astrometry_orbit = remove_nans(x_astrometry_orbit)
    y_ra_orbit = import_orbit_data(y_ra_orbit)
    y_dec_orbit = import_orbit_data(y_dec_orbit)
    
    
    
    
    # make a fine grid that spans the observation window for plotting purposes
    t_astrometry = np.linspace(x_astrometry.min() - 5, x_astrometry.max() + 5, 1000)
    t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)

    # for predicted orbits
    t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)
    
    
         
    
    
    # add NaN for plotting purpose on true model
    x_astrometry_orbit = np.insert(x_astrometry_orbit, 10000, float('NaN'))
    y_ra_orbit = np.insert(y_ra_orbit, 10000, float('NaN'), axis=0)
    y_dec_orbit = np.insert(y_dec_orbit, 10000, float('NaN'),axis=0)
            
    
    


    trace = arviz.from_netcdf(trace_file)
    
    
    
    
    ################
    #plot astrometry
    
    figsAs = []
    figsRV = []
    for n in range(0, nplanets):

        figAs, ax = plt.subplots(2, 2, figsize = [18,9], sharey='row', gridspec_kw={'height_ratios': [3, 1]})
        ax0, ax1, ax2, ax3 = ax[0][0], ax[1][0], ax[0][1], ax[1][1]

        # Get the posterior median orbital parameters
        p = np.median(trace.posterior["P"].values[:, :, n])
        t0 = np.median(trace.posterior["tperi"].values[:, :, n])

        # Compute the median of posterior estimate the other planet. Then we can remove
        # this from the data to plot just the planet we care about.
        other = np.median(
            trace.posterior["dec"].values[:, :, :, (n + 1) % 2], axis=(0, 1)
        )






        # plot the DEC data
        ax0.errorbar(x_astrometry, y_dec - other, yerr=y_dec_err, fmt=".", 
                     color = "#00257c", label = "data", alpha = 0.7)

        # plot the MCMC DEC model
        pred = np.percentile(
            trace.posterior["dec_fine"].values[:, :, :, n],
            [16, 50, 84],
            axis=(0, 1),
        )
        
        med_dec = []
        for an_x in x_astrometry:
            med_dec.append(np.interp(an_x, t_fine, pred[1]))
        


        ax0.plot(t_fine, pred[1], color="r", label="posterior")
        art = ax0.fill_between(
            t_fine, pred[0], pred[2], color="r", alpha=0.3
        )
        art.set_edgecolor("none")

        # plot the true DEC model
        ax0.plot(x_astrometry_orbit, y_dec_orbit.T[n], color = 'k', label = "truth")


        
        if not noRoman:
            ax0.arrow(0.5, 0., 0, 1, width = 0.001, transform=ax0.transAxes,
                      head_width= 0., head_length = 0., color = 'k', ls = '--')
            ax0.text(0.49, 0.9, "Gaia", fontsize = 18, ha='right', transform=ax0.transAxes)
            ax0.text(0.51, 0.9, "Roman", fontsize = 18, ha='left', transform=ax0.transAxes)
            ax0.arrow(0.49, 0.87, -.15, 0, width = 0.001, transform=ax0.transAxes,
                      head_width= 0.03, head_length = .03, color ='k')
            ax0.arrow(0.51, 0.87, .15, 0, width = 0.001, transform=ax0.transAxes,
                      head_width= 0.03, head_length = .03, color ='k')
            



        ax0.legend(fontsize=9, loc=1)
        ax0.set_xlabel("time [days]", fontsize=18)
        ax0.set_ylabel(r"$\Delta \delta$ ['']", fontsize = 18)
        
        
        ax1.set_ylabel(r"[O-C]", fontsize = 18)
        ax1.set_xlabel("phase [days]", fontsize=18)
        
        
        #plot residuals                
        ax1.axhline(0.0, color="k")
        ax1.errorbar(
            x_astrometry, y_dec - other - med_dec,
            yerr=y_dec_err, color = '#00257c', fmt="."
        )



        # Get the posterior median orbital parameters
        p = np.median(trace.posterior["P"].values[:, :, n])
        t0 = np.median(trace.posterior["tperi"].values[:, :, n])

        # Compute the median of posterior estimate the other planet. Then we can remove
        # this from the data to plot just the planet we care about.
        other = np.median(
            trace.posterior["ra"].values[:, :, :, (n + 1) % 2], axis=(0, 1)
        )
        





        # Plot the RA data
        ax2.errorbar(x_astrometry, y_ra - other, yerr=y_ra_err, fmt=".", 
                     color = "#00257c", label = "data", alpha = 0.7)


        pred = np.percentile(
            trace.posterior["ra_fine"].values[:, :, :, n],
            [16, 50, 84],
            axis=(0, 1), 
        )
        
        
        med_ra = []
        for an_x in x_astrometry:
            med_ra.append(np.interp(an_x, t_fine, pred[1]))


        # plot the MCMC RA model
        ax2.plot(t_fine, pred[1], color="r", label="posterior")
        art = ax2.fill_between(
            t_fine, pred[0], pred[2], color="r", alpha=0.3
        )
        art.set_edgecolor("none")
        
        # plot the true RA model
        ax2.plot(x_astrometry_orbit, y_ra_orbit.T[n], color = 'k', label = "truth")



    
        if not noRoman:
            ax2.arrow(0.5, 0., 0, 1, width = 0.001, transform=ax2.transAxes,
                      head_width= 0., head_length = 0., color = 'k', ls = '--')
            ax2.text(0.49, 0.9, "Gaia", fontsize = 18, ha='right', transform=ax2.transAxes)
            ax2.text(0.51, 0.9, "Roman", fontsize = 18, ha='left', transform=ax2.transAxes)
            ax2.arrow(0.49, 0.87, -.15, 0, width = 0.001, transform=ax2.transAxes,
                      head_width= 0.03, head_length = .03, color ='k')
            ax2.arrow(0.51, 0.87, .15, 0, width = 0.001, transform=ax2.transAxes,
                      head_width= 0.03, head_length = .03, color ='k')


        
        ax2.legend(fontsize=9, loc=1)
        ax2.set_xlabel("time [days]", fontsize=18)
        ax2.set_ylabel(r"$\Delta \alpha \cos \delta$ ['']", fontsize = 18)
        
        ax3.set_ylabel(r"[O-C]", fontsize = 18)
        ax3.set_xlabel("phase [days]", fontsize=18)

        
        #plot residuals
        ax3.axhline(0.0, color="k")
        ax3.errorbar(
            x_astrometry, y_ra - other - med_ra,
            yerr=y_ra_err, color = '#00257c', fmt="."
        )
        
        figAs.tight_layout()
        figAs.subplots_adjust(wspace=.2)
        
        
        
        
        
            
        
        
    ######
    #make rv plots
    
    
        figRV, ax = plt.subplots(2, 1, figsize = [13, 6], gridspec_kw={'height_ratios': [3, 1]})
        ax0, ax1 = ax[0], ax[1]
        
        # Get the posterior median orbital parameters
        p = np.median(trace.posterior["P"].values[:, :, n])
        t0 = np.median(trace.posterior["tperi"].values[:, :, n])

        # Compute the median of posterior estimate the other planet. Then we can remove
        # this from the data to plot just the planet we care about.
        other = np.median(
            trace.posterior["vrad"].values[:, :, :, (n + 1) % 2], axis=(0, 1)
        )







        # Plot the data
        ax0.errorbar(x_rv, y_rv - other, yerr=y_rv_err, fmt=".", 
                     color = "#00257c", label = "data", alpha = 0.7)


        pred = np.percentile(
            trace.posterior["vrad_pred"].values[:, :, :, n],
            [16, 50, 84],
            axis=(0, 1),
        )
        
        med_rv = []
        for an_x in x_rv:
            med_rv.append(np.interp(an_x, t_rv, pred[1]))



        
        # plot the MCMC model
        ax0.plot(t_rv, pred[1], color="r", label="posterior")
        art = ax0.fill_between(
            t_rv, pred[0], pred[2], color="r", alpha=0.3
        )
        art.set_edgecolor("none")
        
        # plot the true RA model
        ax0.plot(x_rv_orbit, y_rv_orbit.T[n], color = 'k', label = "truth")








        ax0.legend(fontsize=9, loc=1)
        ax0.set_xlabel("phase [days]", fontsize=18)
        ax0.set_ylabel("radial velocity [m/s]", fontsize=18)
        
        ax1.set_ylabel(r"[O-C]", fontsize = 18)
        ax1.set_xlabel("phase [days]", fontsize=18)



        
        
        
        #plot residuals
        ax1.axhline(0.0, color="k")
        ax1.errorbar(
            x_rv, y_rv - other - med_rv,
            yerr=y_rv_err, color = '#00257c', fmt="."
        )
        
        
        figRV.tight_layout()
        figRV.subplots_adjust(hspace=0.5)
        figRV.suptitle("planet {0}".format(n+1), fontsize=33)
        figRV.tight_layout()
        
        
        figsAs.append(figAs)
        figsRV.append(figRV)
        

        
    return figsAs, figsRV




def load_posterior_params(filename, nplanets):
    trace = arviz.from_netcdf(filename)


    parameters = ['P', 'ecc', 'tperi', 'omega', 'Omega', 'incl', 'm_planet']


    params_earth = defaultdict(list)
    params_jup = defaultdict(list)
    params_earth_err = defaultdict(list)
    params_jup_err = defaultdict(list)

    table_params_all = []
    posterior_meds_all = []
    posterior_errs_all = []
    for ii in range(0, nplanets):
        table_params = defaultdict(list)
        posterior_meds = defaultdict(list)
        posterior_errs = defaultdict(list)
        for param in parameters:


            planet_med = np.median(trace.posterior[param].values[:, :, ii])

            planet_quantile = [np.quantile(trace.posterior[param].values[:, :, ii], 0.16),
                                np.quantile(trace.posterior[param].values[:, :, ii], 0.84)]



            planet_err = [
                planet_med - planet_quantile[0],
                planet_quantile[1] - planet_med 
            ]



            print(param + "_" + str(ii+1) + ": " + str(planet_med) + " +/- " + str(planet_err))

            posterior_meds[param] = np.round(planet_med, 3)
            posterior_errs[param] = np.round(planet_err, 3)
            table_params[param].append(str(np.round(planet_med, 3)) + " (+" + 
                                             str(np.round(planet_err[1], 3)) + " -" + 
                                             str(np.round(planet_err[0], 3)) + ")")






        table_params_all.append(table_params)
        posterior_meds_all.append(posterior_meds)
        posterior_errs_all.append(posterior_errs)
        print("")
        print("")



    return table_params_all, posterior_meds_all, posterior_errs_all







def load_input_params_and_keys(filename):
    planet_param_keys = ['period', 'ecc', r'T$_\mathrm{per}$', r'$\omega$', r'$\Omega$', 'inc', 'mass']
    planet_params_df = pd.read_csv(filename)
    
    planet_params = []
    for row in planet_params_df.iterrows():
        params = []
        params.append(row[1]['period'])
        params.append(row[1]['ecc'])
        params.append(row[1]['Tper'])
        params.append(row[1]['omega'])
        params.append(row[1]['Omega'])
        params.append(row[1]['inclination'])
        params.append(row[1]['mass']*332946.07)
        


        planet_params.append(params)
    
    return planet_param_keys, planet_params






def make_comparison_figures(input_params, posterior_orbit_params, keys_orbit_params, 
                            posterior_meds, posterior_errs, nplanets):

    import matplotlib
    import matplotlib.pyplot as plt

    comparisons = []
    for ii in range(0, nplanets):
        fig, axs = plt.subplots(1, 2, figsize = [13,6], gridspec_kw={'width_ratios': [1, 1]})
        plt.subplots_adjust(hspace=1)
        
        #### add comparison plots
        
        
        input_mass = input_params[ii][-1]
        input_period = input_params[ii][0]
        
        xs = posterior_meds[ii]['m_planet']
        xs_err = np.array([posterior_errs[ii]['m_planet']]).T

        ys = posterior_meds[ii]['P']
        ys_err = np.array([posterior_errs[ii]['P']]).T



        color = '#1c245f'

        axs[0].errorbar(xs, ys, xerr=xs_err, yerr = ys_err, marker = 'o', 
                        color = color, markersize=13, label = r'med and 1$\sigma$ errs',
                       linewidth = 3, alpha = 0.7)

        axs[0].axvline(x=input_mass, ymin=0, ymax=1, color = 'k', ls = '--', label = 'simulated')
        axs[0].axhline(y=input_period, xmin=0, xmax=1, color = 'k', ls = '--')


        axs[0].set_xlabel(r'mass [M$_{earth}$]', fontsize = 27)
        axs[0].set_ylabel(r'period [days]', fontsize = 27)


        fig.suptitle('input and posterior parameters', fontsize = 27)

        axs[0].legend(fontsize = 13)
        
        
        
        #### add comparison tables


        sim_params = input_params[ii]
        post_params = posterior_orbit_params[ii]
        data = np.column_stack((np.array(keys_orbit_params), 
                                np.round(np.array(sim_params),3),
                                np.array(list(post_params.values()))))

        labels = ['param', 'truth', r'MCMC med +/- 1$\sigma$']

        df = pd.DataFrame(data, columns=labels)

        table = axs[1].table(cellText=df.values, colLabels=df.columns, 
                         colColours =["#D7E5F0"]*3, loc='center', 
                         bbox=[0,0,1,1], colWidths = [0.2, 0.2, 0.6])
        #for (row, col), cell in table.get_celld().items():
        #    if (row == 0):
        #        cell.set_text_props(fontproperties=FontProperties(weight='bold'))


        table.auto_set_column_width(col=labels)

        #ax.set_title('test_sim_file.csv'[:-4], fontsize=18, loc='left', fontweight ="bold")
        #t[(np.argmin(jwst_table.values), 0)].set_facecolor("#53b568")
        #t[(np.argmin(jwst_table.values), 1)].set_facecolor("#53b568")
        axs[1].set_axis_off()


        table.auto_set_font_size(False)
        table.set_fontsize(13)

        
        
        



        comparisons.append(fig)
        
        
        
        
    return comparisons












def make_corner_plots(posterior_data_filename, input_planet_params_filename):
    import corner
    import astropy.units as u
    import matplotlib 

    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18) 
    
    trace = arviz.from_netcdf(posterior_data_filename)
    planet_param_keys, planet_params = load_input_params_and_keys(input_planet_params_filename)
    
    truths = []
    for ii in range(0, len(planet_params[0])):
        truths.append(planet_params[0][ii])
        truths.append(planet_params[1][ii])

    

    
    
    
    corner_fig = corner.corner(
        trace, var_names=["P", "ecc", "tperi", "omega", "Omega", "incl", "m_planet"], 
        labels=[r"P$_1$", r"P$_2$", r"ecc$_1$", r"ecc$_2$", r"T$_{peri, 1}$", r"T$_{peri, 2}$", 
        r"$\omega_1$", r"$\omega_2$", r"$\Omega_1$", r"$\Omega_2$", r"incl$_1$", r"incl$_2$", r"$m_{p, 1}$", r"$m_{p, 2}$"],
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 18}, 
        truths = truths, truth_color = "#00257c", label_kwargs={"fontsize": 18})



    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18) 
    
    return corner_fig






def trace_stats(posterior_data_filename):
    trace = arviz.from_netcdf(posterior_data_filename)

    df = arviz.summary(
        trace,
        var_names=[
            "P",
            "ecc",
            "m_planet",
            "Omega",
            "omega",
            "incl"    ],
    )
    
    return df




def make_diagnostic_plots(simulated_data_filename, posterior_data_filename, input_planet_params_filename,
                          nplanets, diagnostic_figname, corner_figname, trace_stats_figname, trace_filename, 
                          noRoman=False):


    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages


    table_params_all, posterior_meds_all, posterior_errs_all = load_posterior_params(posterior_data_filename, nplanets)
    planet_param_keys, planet_params = load_input_params_and_keys(input_planet_params_filename)

    figs = make_rv_astrometry_plots(simulated_data_filename, posterior_data_filename, nplanets, noRoman=noRoman)


    comparison_figs = make_comparison_figures(planet_params, table_params_all, planet_param_keys, 
                                              posterior_meds_all, posterior_errs_all, nplanets)
    
    
    

    



    for ii in range(0, nplanets):
        pp = PdfPages(diagnostic_figname + str(ii+1) + '.pdf')

        pp.savefig(figs[1][ii])
        pp.savefig(figs[0][ii])
        pp.savefig(comparison_figs[ii])
        pp.close()
        
    
    
    corner_fig = make_corner_plots(posterior_data_filename, input_planet_params_filename)
    pp = PdfPages(corner_figname + '.pdf')
    pp.savefig(corner_fig)
    pp.close()
    
    
    
    
    
    
    
    matplotlib.rc('text', usetex=False)
    post_stats_df = trace_stats(posterior_data_filename)
    fig_post_stats, ax = plt.subplots(figsize = [13, 6])
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=post_stats_df.values,colLabels=post_stats_df.columns,
                         rowLabels=post_stats_df.axes[0],loc='center')
    
    
    
    pp = PdfPages(trace_stats_figname+'.pdf')
    pp.savefig(fig_post_stats, bbox_inches='tight')
    pp.close()
    


    
    
    
    
    
    
    trace = arviz.from_netcdf(posterior_data_filename)
    import arviz.labels as azl
    axes= arviz.plot_trace(trace, figsize=[13,18], var_names=[
        "P",
        "ecc",
        "m_planet",
        "Omega",
        "omega",
        "incl"])


    fig = axes.ravel()[0].figure
    fig.savefig(trace_filename + '.pdf')
    matplotlib.rc('text', usetex=True)
    
    return None


def rhat_passes_test(posterior_data_filename, max_rhat = 1.05):
    trace = arviz.from_netcdf(posterior_data_filename)

    df = arviz.rhat(
        trace,
        var_names=[
            "P",
            "m_planet"    ],
    )

    small_rhat = True
    if np.array(df['P'])[0] > max_rhat:
        small_rhat = False
    if np.array(df['P'])[1] > max_rhat:
        small_rhat = False
    if np.array(df['m_planet'])[0] > max_rhat:
        small_rhat = False
    if np.array(df['m_planet'])[1] > max_rhat:
        small_rhat = False
    
    return small_rhat 
        



