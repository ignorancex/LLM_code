import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import numpy as np
import numpy.ma as ma
from scipy import stats
import matplotlib.ticker as mticker
from plotting_tools import *



# color mapping
fsed_colors = pl.cm.viridis
logg_colors = pl.cm.plasma

fsed_num = [1, 2, 3, 4, 8, 10]
fsed_ticks = ["Cloudy", '2', '3', '4', '8', 'No \n Clouds']
fsed_bounds = [0.5, 1.5, 2.5, 3.5, 6, 9, 11]

logg_num = [3.5, 4, 4.5, 5, 5.5]
logg_ticks = ['Less dense', '4', '4.5', '5', 'More dense']
logg_bounds = [3.25, 3.75, 4.25, 4.75, 5.25, 5.75]

norm_f = mpl.colors.BoundaryNorm(fsed_bounds, fsed_colors.N, extend='neither')
norm_g = mpl.colors.BoundaryNorm(logg_bounds, logg_colors.N, extend='max')



def long_plot(parameter_df, convolve_data_dict, x_min = 1.16, x_max = 1.185, 
                  x_increment = 0.005, const_spacing = 1.5, section_spacing = 2, norm_scaling = 7e10,
                  title = "Potassium Doublets", color_by_logg = True):
    # TODO: add doc string for new variables
    """
    Plots normalized and convolved spectral data with annotations and color-coding.

    Parameters:
    parameter_df (pandas.DataFrame): DataFrame containing the sources and their parameters.
                                   It should have columns 'name', 'logg', and 'clouds'.
    convolve_data_dict (dict): Dictionary containing convolved spectral data arrays. 
                               Keys are source names, and values are numpy arrays with shape (3, length),
                               where the first row is wavelength data and the second row is flux data.
    x_min (float, optional): Minimum x-axis value for the plot. Default is 1.16.
    x_max (float, optional): Maximum x-axis value for the plot. Default is 1.185.
    x_increment (float, optional): Increment for x-axis ticks. Default is 0.005.
    const_spacing (float, optional): Constant spacing added to each spectrum for separation in the plot. Default is 1.5.
    norm_scaling (float, optional): Scaling factor for normalizing the flux data. Default is 7e10.
    title (str, optional): Title of the plot. Default is "Potassium Doublets".
    color_by_logg (bool, optional): If True, spectra are color-coded by logg. If False, spectra are color-coded by clouds. Default is True.

    This function performs the following tasks:
    1. Sorts the sources based on 'logg' and 'clouds' and determines the plotting order.
    2. Sets up constants for vertical spacing of the spectra.
    3. Initializes a matplotlib figure and axis for plotting.
    4. Iterates over the sorted sources to:
        a. Normalize the convolved flux data.
        b. Plot each spectrum with appropriate color-coding and vertical spacing.
        c. Annotate the plot with fitted parameters.
    5. Configures plot aesthetics including x and y limits, ticks, grid lines, and axis labels.
    6. Adds potassium doublet vertical lines and annotations.
    7. Optionally adds horizontal lines and labels to indicate different clouds or logg values.
    8. Adds a color bar indicating the parameter used for color-coding.
    """
    if color_by_logg:
        index = parameter_df.sort_values(
            by=['clouds', 'logg']).index.to_numpy()  # order of spectra plotted
    else:
        index = parameter_df.sort_values(by=['logg', 'clouds']).index.to_numpy()

    num_of_spectra = len(parameter_df)

    label_loc_l = []  # where left label located on y axis
    label_loc_r = []  # where right label located on y axis
    # max and min point of spectra used for spacing purposes
    top_of_spectra = np.zeros(num_of_spectra) # max point in the spectra
    bottom_of_spectra = np.zeros(num_of_spectra) # max point in the spectra



    if color_by_logg:
        const = [(i*const_spacing) + section_spacing*(i//5)
                 for i in range(num_of_spectra)]  # constants to be added to flux
    else:
        const = [(i*const_spacing) + section_spacing*(i//6) 
                 for i in range(num_of_spectra)] # constants to be added to flux

    fig, ax = plt.subplots(figsize=(6, 16))

    # plotting each spectra
    for n, i in enumerate(index):
        # setting constants, labels, color needed
        if color_by_logg:
            color = logg_colors(norm_g(parameter_df.logg[i]))
        else:
            color = fsed_colors(norm_f(parameter_df.clouds[i]))
        name = parameter_df.name[i]
        c = const[n]

        # normalizing
        norm = (convolve_data_dict[name][1, :]) / norm_scaling

        # plotting
        ax.plot(convolve_data_dict[name][0, :], norm + c, alpha=1, color=color)

        # getting y max and y min of plotted values
        region = ma.masked_inside(convolve_data_dict[name][0, :], x_min, x_max).mask
        plotted_flux = (norm + c)[region]
        top_of_spectra[n] = max(plotted_flux)
        bottom_of_spectra[n] = min(plotted_flux)

        # adding label for avg(A) and avg(FWHM)
        l_index = np.where(np.isclose(convolve_data_dict[name][0, :], x_min))[
            0][0]  # x index in spectra of left side
        r_index = np.where(np.isclose(convolve_data_dict[name][0, :], x_max))[
            0][0]  # x index in spectra of right side
        label_loc_r.append(norm[r_index] + c)   # y axis location for labels
        label_loc_l.append(norm[l_index] + c)
        A_label = f'{-(parameter_df.A1[i] + parameter_df.A2[i]) / 2e11 :.2f} '  # string of A
        ax.annotate(A_label,
                    xy=(x_max, label_loc_l[n]), xycoords='data', color=color)
        FWHM_label = f' {(parameter_df.FWHM1[i] + parameter_df.FWHM2[i]) * 1e3 :.2f}'  # string for FWHM
        ax.annotate(FWHM_label, horizontalalignment='right',
                    xy=(x_min, label_loc_r[n]), xycoords='data', color=color)

        

    # setting limits
    y_max = max(top_of_spectra) + 2*const_spacing
    y_min =  min(bottom_of_spectra) - const_spacing / 2
    # limits in x and y
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ## Configuring plot aesthetics
    # setting ticks
    ax.set_yticks([n for n in range(int(y_min), int(y_max) + 1)])  # y-ticks
    # creating blank label in y
    ax.set_yticklabels(['' for n in range(int(y_min), int(y_max) + 1)])
    # setting x-ticks with x-increment set at top
    ax.set_xticks(ticks=[round(x_min + n*x_increment, 3)
                for n in range(0, int((x_max-x_min)/x_increment)+1)])
    # creating grid
    ax.tick_params(which='major', length=5, width=0,
                direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=2, width=0,
                direction='in', top=True, right=True)
    # grid lines
    ax.minorticks_on()
    ax.grid(visible=True, which='major', alpha=.5)
    ax.grid(visible=True, which='minor', alpha=.1)


    # adding axis titles/labels
    ax.set_xlabel('Wavelength (μm)', fontsize=13)
    ax.set_ylabel("Normalized Flux + Constant\n\n", fontsize=13)
    ax.set_title(title, fontsize=13)

    # setting parameter labels
    ax.annotate(r"$avg(A)$  " "\n Max Depth  \n", xy=(x_min, y_max),
                ha='right',
                va='center',
                xycoords='data', color='k')
    
    ax.annotate(r"  $avg(\sigma)$" + "\n  FWHM \n", xy=(x_max, y_max),
                ha='left',
                va='center',
                xycoords='data', color='k')



    # horizontal lines
    # location is defined by the top of spectra within each group
    hline_y = []
    if color_by_logg:
        for i in range(1, 7):
            hline_y.append(top_of_spectra[(i*5)-1] + const_spacing / 4)

        pretty_fsed = [' Cloudy \n $f_{sed} = 1$',
                    r' $f_{sed} = 2$', r' $f_{sed} = 3$',
                    r' $f_{sed} = 4$', r' $f_{sed} = 8$',
                    'No Clouds']

        for i, y in enumerate(hline_y):
            ax.annotate(pretty_fsed[i], xy=(x_min + 0.0001, y ),
                        xycoords='data', color='k', fontsize=9, va = 'center')
    else:
        for i in range(1, 6):
            hline_y.append(top_of_spectra[(i*6)-1] + const_spacing / 4)

        pretty_logg = [' least dense\n'+r' $\log(g) = 3.5$',
                    r' $\log(g) = 4.0$', r' $\log(g) = 4.5$',
                    r' $\log(g) = 5.0$',
                    ' most dense \n' + r' $\log(g) = 5.5$']
        
        for i, y in enumerate(hline_y):
            ax.annotate(pretty_logg[i], xy=(x_min + 0.0001, y - .1),
                        xycoords='data', color='k', fontsize=9, va = 'center')

    ax.hlines(hline_y, xmin=x_min + (x_max-x_min)/4, xmax=x_max,  color='k')
    
     # Adding potassium doublet vertical lines and annotations
    potassium_lines = [1.16935, 1.1775, 1.2435, 1.2525]
    ax.vlines(potassium_lines, ymin=y_min, ymax= max(hline_y) +  (const_spacing / 10) ,
            linestyle='dotted', color='k', linewidth=1.2, alpha=.8)
    for k in potassium_lines:
        ax.annotate("K I", xy=(k, max(hline_y) +  (const_spacing / 10)),
                    ha='center', va='bottom',
                    xycoords='data', color='k', fontsize=12)
        

     # color bar
    if color_by_logg:
        # axcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_g, cmap=logg_colors),
        #                     ticks=logg_num, shrink=1, format=mticker.FixedFormatter(logg_ticks),
        #                     aspect=50,  pad=.14)
        axcb =logg_colorbar(fig, ax= ax, orientation='vertical', shrink=1, aspect=50, pad=.14,)
        axcb.set_label(' ', fontsize=12)  # empty label
        ax.annotate(r'$\log(g)$', xy=(.9, .5), xycoords='figure fraction',
                    rotation=270, fontsize=13)  # actual color bar label

    else:
        # axcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_f, cmap=fsed_colors),
        #                     ticks=fsed_num, shrink=1, format=mticker.FixedFormatter(fsed_ticks),
        #                     aspect=50, pad=.14)
        axcb = fsed_colorbar(fig, ax=ax, orientation='vertical', shrink=1, aspect=50, pad=.14,)
        axcb.set_label(' ', fontsize=12)  # empty label
        ax.annotate(r'$f_{sed}$', xy=(.9, .5), xycoords='figure fraction',
                    rotation=270, fontsize=13)  # actual color bar label
