import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
import corner
import matplotlib.lines as mlines
from astropy.cosmology import Planck15 as cosmo

class Maps():
    """Methods to plot our results"""
    def __init__(self):
        a=1

    def mass_luminosity(self, COMAP21_params,ls=None, label=None):
        
        fig, ax = plt.subplots(1,1, figsize=(11,9))
        if ls is None:
            ls = ['solid']*len(COMAP21_params)
        if label is None:
            label = ['']*len(COMAP21_params)
        Mvir = 10**np.arange(10,13.5,0.01)
        for i in range(len(COMAP21_params)):
            lco_p =  COMAP21_params[i]['C'] / ((Mvir/(cosmo.h*COMAP21_params[i]['M']))**COMAP21_params[i]['A'] +
                                            (Mvir/(cosmo.h*COMAP21_params[i]['M']))**COMAP21_params[i]['B'])
            
            
            label =('A ='+str(np.around(COMAP21_params[i]['A'], 3))+
                    ' B ='+str(np.around(COMAP21_params[i]['B'], 3))+
                    ' C ='+str(np.around(np.log10(COMAP21_params[i]['C']), 3))+
                    ' M ='+str(np.around(np.log10(COMAP21_params[i]['M']), 3))+
                    ' sco ='+str(np.around(np.log10(COMAP21_params[i]['sigma_co']),3)))

            #lco_p= 10**np.random.normal(np.log10(lco_p), COMAP21_params['sigma_co'])

            ax.plot(Mvir, lco_p, ls=ls[i], label=label )
        ax.set_xlabel(r'$M_{halo} [M_{\odot}/h]$')
        ax.set_ylabel(r'$L_{CO} [L_{\odot}]$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(axis='both', which='both')
        ax.legend(framealpha=0, fontsize=20)

        return fig, ax

    def plot_co_map_slices(self, fig, ax, co_temp, z, vmin=None, vmax=None, title=''):
        """Plot 2D slices for the cO LIM map"""
        cmap = plt.get_cmap('jet')
        im = ax.imshow(co_temp[:,:,z], extent=[0, co_temp.shape[0], 0, co_temp.shape[1]], 
                       origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.17, pad=0.01)
        #cb.ax.tick_params(labelsize=5, width=5, length=2)
        cb.set_label(r'$T_{co} \ [\mu K]$')
        ax.set_title(title, fontsize=10)
        
        
    def plot_lya_map_slices(self, fig, ax, lya_map, z, vmin=None, vmax=None, title='', cb_label=r'$\delta_F$'):
        """Plot 2D slices for the Lya tomography map"""
        cmap = plt.get_cmap('jet').reversed()
        im = ax.imshow(lya_map[:,:,z], extent=[0, lya_map.shape[0], 0, lya_map.shape[1]], 
                       origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.17, pad=0.01)
        #cb.ax.tick_params(labelsize=5, width=5, length=2)
        cb.set_label(cb_label)
        ax.set_title(title, fontsize=20)
        
        
    def plot_density(self, fig, ax, dens, z, vmin=None, vmax=None, title='', cb_label=r'$\delta_m$'):
        """ Plot DM density or halo density map """
        cmap = plt.get_cmap('jet')
        dens_temp = dens
        im = ax.imshow(dens_temp[:,:,z], extent=[0, dens_temp.shape[0], 0, dens_temp.shape[1]], 
                       origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.17, pad=0.01)
        #cb.ax.tick_params(labelsize=5, width=5, length=2)
        cb.set_label(cb_label)
        ax.set_title(title, fontsize=20)


class Inference:
    """A class for inference realted plots"""
    def __init__(self, Tbco_range = (2.1,3.1), pshotco_range = (200,2000)):
        self.Tbco_range = Tbco_range
        self.pshotco_range = pshotco_range

    def Tbco_pshot(self, samples, colors, labels, Tb_co_true=None, pshot_co_true=None, fig=None):
        """Plot a corner plot for CO <T_CO>b_CO and P_shot_co paramters 
        Paramters:
            infs[i].samples : a list of infs[i].samples . Each element of the list is
            an array of (m,n) dimensions.
            colors : list of colors
            labels : list of labels
            Tb_co_true: Optional, the true <T_CO>b_CO to be drawn 
            on the corner pot
            pshot_co_true : Optional, the true pshot_co to be drawn 
            on the corner pot
            fig : matplotlib figure or subfigure instance
        Returns :
            fig, ax
        """
        rng =  [(1.2,2.15), (250,1100)]
        if fig is None:
            fig = plt.figure(figsize=(10,10))

        legend_handles=[]
        alpha = [0.5, 1, 1, 0.9, 0.9]
        for i in range(len(samples)):
            legend_handles.append(mlines.Line2D([], [], color=colors[i], label=labels[i]))
            fig = corner.corner(data = samples[i][:,0:2], labels=[r'$\langle T_{CO} \rangle b_{CO}$',
                                                r'$P_{shot, CO}$'], fig=fig, color=colors[i], range=rng,
                                                bins=[220,300], plot_datapoints=False, show_titles=False,
                                                truths=[Tb_co_true,pshot_co_true], truth_color='C0',
                                                quantiles=None, levels=[0.01,0.68,0.95], hist_kwargs={'density':True}, 
                                                contour_kwargs={'alpha':alpha[i]}, no_fill_contours=True,
                                                fill_contours=True, plot_density=False, 
                                                contourf_kwargs={'linewidths':6})
        
        plt.legend(handles=legend_handles, bbox_to_anchor=(0.25, 1.0, 1.0, .0), loc=4, framealpha=0)
        ax = np.array(fig.axes).reshape((2,2))
        #if pshot_co_true is not None:
        #    ax[1,1].axvline(pshot_co_true, ls='--', color='k')
        #    ax[1,0].axhline(pshot_co_true,ls='--', color='k')
        #ax[1,0].set_xlim(self.Tbco_range[0], self.Tbco_range[1])
        #ax[1,0].set_ylim(self.pshotco_range[0], self.pshotco_range[1])
        #ax[1,0].set_ylim(self.pshotco_range[0], self.pshotco_range[1])
        #ax[0,0].set_xlim(self.Tbco_range[0], self.Tbco_range[1])
        #ax[1,1].set_xlim(self.pshotco_range[0], self.pshotco_range[1])
        ax[1,0].grid(which='both', axis='both')

        return fig, ax
    
    def latex_table_tbco_pshot(self, samples, labels):
        """
        Write a LaTex table for <T_co> b_co and P_shot,CO
        """
        latex_file = open('stan/tbco_pshot.txt','w')

        latex_table = [r'\begin{table}', '\n',r'\caption{Additional commands for mathematical symbols. These can only be used in maths mode.}',
                       '\n', r'\label{tab:mathssymbols}','\n', r'\begin{tabular*}{\columnwidth}{l@{\hspace*{20pt}}l@{\hspace*{10pt}}l}', '\n', 
                       r'\hline','\n', r'Survey & $\langle T_{CO} \rangle b_{CO}$ & $P_{shot, CO}$\\[4pt]','\n', r'\hline \\[1pt]', '\n']
        for i in range(len(samples)):
            latex_table.append(f'{labels[i]} & {np.median(samples[i][:,0]):.2f} \\pm  {np.std(samples[i][:,0]):.2f} & '+
                               f'{np.median(samples[i][:,1]):.0f}  \\pm {np.std(samples[i][:,1]):.0f} \\\\[2pt]'+'\n')

        latex_table.append(r'\hline')
        latex_table.append('\n') 
        latex_table.append(r'\end{tabular*}')
        latex_table.append('\n')
        latex_table.append(r'\end{table}')
        print(type(latex_table))
        latex_file.writelines(latex_table)




    def all_params_corner(self, fig, infs, colors, labels, truths=[None, None, None, None, None], 
                        tick_label_size=None, axis_label_size= None, labelpad=None):
        """Plot a corner plot for all infered paramters 
        Paramters:
            colors : list of colors
            labels : list of labels
            pshot_co_true : Optional, the true pshot_co to be drawn 
            on the corner pot
            fig : matplotlib figure or subfigure instance
        Returns :
            fig, ax
        """
        
        #subfigs = fig.subfigures(1,2, width_ratios=[1,2], wspace=0.0)
        subfig_corner = fig
        legend_handles=[]
        for i in range(len(infs)):
            if infs[i].samples.shape[1] == 3:
                #subfig_corner = subfigs[0].subfigures(2,1, height_ratios=[1,1.5])[1]
                ax_labels = [r'$\langle T_{CO} \rangle b_{CO}$', r'$P_{shot, CO}$', 
                        r'$b_{Ly\alpha}$']
                legend_loc = 1
                rng = [(1.3,2.0), (250,1100),(-0.225, -0.210)]
                bins=[200,120,1000]
                bbox_to_anchor=(0, 4, 1.0, .0)
                ncol = 2
                fontsize_legend=12
                if tick_label_size is None:
                    tick_label_size=12
                    axis_label_size= 17
            elif infs[i].samples.shape[1] == 5:
                #subfig_corner = subfigs[0].subfigures(2,1, height_ratios=[1,1])[1]
                ax_labels = [r'$\langle T_{CO} \rangle b_{CO}$', r'$P_{shot, CO}$', 
                        r'$b_{gal}$', r'$P_{shot, Gal}$',r'$P_{shot, \times}$']
                legend_loc = 8
                bbox_to_anchor=(-0.6, 4.5, 1.0, .0)
                ncol=1
                rng = [(1.3,2.0), (250,1100),(2.1,3.7), (0, 3300),(-50, 700)]
                bins= [120,120,450,450,120]
                fontsize_legend=12
                if tick_label_size is None:
                    tick_label_size=12
                    axis_label_size = 17
                

            subfig_corner = corner.corner(infs[i].samples, labels=ax_labels, fig=subfig_corner, color=colors[i], range=rng,
                    bins=bins, plot_datapoints=False, show_titles=False, quantiles=None, hist_kwargs={'density':True},
                    no_fill_contours=True, fill_contours=True, plot_density=False,  contour_kwargs={'alpha':0.8}, truths=truths, 
                    levels=[0.01, 0.68, 0.95], truth_color='k', contourf_kwargs={'linewidths':1})
            if labels is not None:
                legend_handles.append(mlines.Line2D([], [], color=colors[i], label=labels[i]))
        if labels is not None:
            plt.legend(handles=legend_handles, bbox_to_anchor=bbox_to_anchor, 
                    framealpha=0, fontsize=15, ncol=ncol)
        ax = np.array(subfig_corner.axes)

        print(len(ax))
        for i in range(len(ax)):
            ax[i].grid(which='both', axis='both')
            ax[i].tick_params(labelsize=tick_label_size, pad=-0.1)
            if labelpad is not None:
                ax[i].xaxis.labelpad = labelpad
                ax[i].yaxis.labelpad = labelpad
            ax[i].xaxis.label.set_fontsize(axis_label_size)
            ax[i].yaxis.label.set_fontsize(axis_label_size)

    def model_vs_signal(self, fig, stats, infs, labels, colors):
        """Comapre the simulated signal with the linear model
        """
        ratio_height = 0.1
        power_height = 0.15
        width = 0.82
        floor = 0.04
        gal_legend=False
        
        ax_auto = fig.add_axes([ratio_height, floor+3*ratio_height+2*power_height, width, power_height])
        ax_auto_ratio = fig.add_axes([ratio_height, floor+2*ratio_height+2*power_height, width, ratio_height])
        ax_cross = fig.add_axes([ratio_height, floor+2*ratio_height+power_height ,width, power_height])
        ax_cross_ratio = fig.add_axes([ratio_height, floor+ratio_height+power_height, width, ratio_height])
        ax_co = fig.add_axes([ratio_height, floor+ratio_height, width, power_height])
        ax_co_ratio = fig.add_axes([ratio_height, floor, width, ratio_height])
        
        axis_label_size_x = 24
        axis_label_size_y = 27
        tick_label_size = 22
        

        for i in range(len(stats)):
            
            if i==2:
                hatch='/'
                color_shade = 'none'
                edgecolor = colors[i] 
            else:
                hatch=None
                color_shade =  colors[i]
                edgecolor = 'none'
            
            if stats[i].lya_pk is not None:
                k = stats[i].lya_pk['k']
                ind = np.where((k > infs[i].kmin)*(k<infs[i].kmax))
                ax_auto.plot(k, stats[i].lya_pk['power'], color='C0')
                print(infs[i].lya_pk_model.shape)
                ax_auto.plot(k[ind], infs[i].lya_pk_model, label=labels[i], ls='--', color=colors[i])
                ax_auto.set_ylabel(r'$P_{Lya}$', fontsize = axis_label_size_y)
                ax_auto_ratio.plot(k[ind], infs[i].lya_pk_model/stats[i].lya_pk['power'][ind], label=labels[i], ls='--', color=colors[i])
                ax_auto.fill_between(k[ind], infs[i].lya_pk_bounds[0], infs[i].lya_pk_bounds[1], color=colors[i], alpha=0.4, )
                ax_auto_ratio.fill_between(k[ind], infs[i].lya_pk_bounds[0]/stats[i].lya_pk['power'][ind], 
                        infs[i].lya_pk_bounds[1]/stats[i].lya_pk['power'][ind], color=colors[i], alpha=0.4, )

                ax_cross.plot(k, -stats[i].lim_lya_pk['power'], color='C0')
                ax_cross.plot(k[ind], -infs[i].lim_lya_pk_model, label=labels, ls='--', color=colors[i])
                ax_cross.set_ylabel(r'$|P_{CO \times Lya}|$', fontsize = axis_label_size_y)
                ax_cross_ratio.plot(k[ind], infs[i].lim_lya_pk_model/stats[i].lim_lya_pk['power'][ind], label=labels[i], ls='--', color=colors[i])
                ax_cross.fill_between(k[ind], infs[i].lim_lya_pk_bounds[0], infs[i].lim_lya_pk_bounds[1], color=colors[i], alpha=0.4, )
                ax_cross_ratio.fill_between(k[ind], infs[i].lim_lya_pk_bounds[0]/stats[i].lim_lya_pk['power'][ind], 
                        infs[i].lim_lya_pk_bounds[1]/stats[i].lim_lya_pk['power'][ind], color=colors[i], alpha=0.4)

            if stats[i].gal_pk is not None:
                gal_legend = True
                k = stats[i].gal_pk['k']
                ind = np.where((k > infs[i].kmin)*(k<infs[i].kmax))
                ax_auto.plot(k, stats[i].gal_pk['power'], color='C0')
                ax_auto.plot(k[ind], infs[i].gal_pk_model, label=labels[i], ls='--', color=colors[i])
                ax_auto.set_ylabel(r'$P_{gal}$', fontsize = axis_label_size_y)
                ax_auto_ratio.plot(k[ind], infs[i].gal_pk_model/stats[i].gal_pk['power'][ind], label=labels[i], ls='--', color=colors[i])
                ax_auto.fill_between(k[ind], infs[i].gal_pk_bounds[0], infs[i].gal_pk_bounds[1], color=colors[i], alpha=0.4)
                ax_auto_ratio.fill_between(k[ind], infs[i].gal_pk_bounds[0]/stats[i].gal_pk['power'][ind], 
                        infs[i].gal_pk_bounds[1]/stats[i].gal_pk['power'][ind], color=colors[i], alpha=0.4)

                ax_cross.plot(k, stats[i].lim_gal_pk['power'], color='C0')
                ax_cross.plot(k[ind], infs[i].lim_gal_pk_model, label=labels, ls='--', color=colors[i])
                ax_cross.set_ylabel(r'$P_{CO \times gal}$', fontsize = axis_label_size_y)
                ax_cross_ratio.plot(k[ind], infs[i].lim_gal_pk_model/stats[i].lim_gal_pk['power'][ind], label=labels[i], ls='--', color=colors[i])
                ax_cross.fill_between(k[ind], infs[i].lim_gal_pk_bounds[0], infs[i].lim_gal_pk_bounds[1], color=colors[i], alpha=0.4)
                ax_cross_ratio.fill_between(k[ind], infs[i].lim_gal_pk_bounds[0]/stats[i].lim_gal_pk['power'][ind], 
                        infs[i].lim_gal_pk_bounds[1]/stats[i].lim_gal_pk['power'][ind], color=colors[i], alpha=0.4)
            
            if i%2 == 0:
                hatch='x'
            else:
                hatch=None
            print(stats)
            ax_co.plot(k, stats[i].lim_pk['power'], color='C0')
            ax_co.plot(k[ind], infs[i].lim_pk_model, label=labels, ls='--', color=colors[i])
            ax_co.set_ylabel(r'$P_{CO}$', fontsize = axis_label_size_y)
            ax_co_ratio.plot(k[ind], infs[i].lim_pk_model/stats[i].lim_pk['power'][ind], label=labels[i], ls='--', color=colors[i])
            ax_co_ratio.fill_between(k[ind], infs[i].lim_pk_bounds[0]/stats[i].lim_pk['power'][ind], 
                        infs[i].lim_pk_bounds[1]/stats[i].lim_pk['power'][ind], color=color_shade, alpha=0.4, hatch = hatch,  edgecolor=colors[i])
            ax_co.set_ylim(5e2,3e4)
            ax_co_ratio.set_ylim(0.75,1.5)
        
        if gal_legend:
            ax_auto.legend( bbox_to_anchor=(0, 1.0, 1.1, 0.3), ncol=2, fontsize=20, framealpha=0)   
        else:
            ax_auto.legend( bbox_to_anchor=(0, 1.0, 1.1, 0.3), ncol=2, fontsize=20, framealpha=0)

        for ax_power, ax_ratio in zip([ax_auto, ax_cross, ax_co],[ax_auto_ratio, ax_cross_ratio, ax_co_ratio]):
            ax_power.set_xscale('log')
            ax_power.set_yscale('log')
            ax_ratio.set_xscale('log')
            ax_ratio.set_xlabel(r'k [h/cMpc]', fontsize=axis_label_size_x)
            ax_ratio.set_ylabel(r'$\frac{\hat{P}}{P}$', fontsize=28)
            ax_power.tick_params(labelsize=tick_label_size, labelbottom=False)
            ax_ratio.tick_params(labelsize=tick_label_size)
            ax_power.grid(axis='both', which='both')
            ax_ratio.grid(axis='both', which='both')
            xlim = ax_power.get_xlim()
            ax_ratio.set_xlim(xlim)
        ax_auto_ratio.set_ylim((0.8,1.2))
        ax_auto_ratio.set_yticks(np.arange(0.9,1.19,0.1))
        ax_cross_ratio.set_yticks(np.arange(0.75,1.49,0.25))
        ax_cross_ratio.set_ylim((0.5,1.5))
        ax_co_ratio.set_yticks(np.arange(0.75,1.49,0.25))



    def plot_single_co(self, sts, labels, colors, title=''):
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        for i, st in enumerate(sts):
            st.get_co_sn()
            ax[1].plot(st.lim_pk['k'][:], st.co_sn, label='CO '+labels[i], ls='solid', color=colors[i])

            ax[0].plot(st.lim_pk['k'][:], np.abs(st.lim_pk['power'][:]), label=r'$P_{CO}, $'+labels[i], ls='solid', alpha=0.7, color=colors[i])
            ax[0].plot(st.lim_pk['k'][:], np.abs(st.sigma_co_pk[:]), label=r'$\sigma_{P_{CO}},$'+labels[i], ls='--', alpha=0.7, color=colors[i])
        
        for i in range(2):
            ax[i].set_xscale('log')
            ax[i].grid(axis='both', which='both')
            ax[i].set_xlabel('$k \ h(cMpc)^{-1}$')
            ax[i].set_xlim(2e-2,6e-1)
            ax[i].legend(framealpha=0, loc='upper left', fontsize=20)
        ax[0].set_yscale('log')
        ax[0].set_ylim(1e2,1e5)
        ax[1].set_ylabel('S/N')
        ax[1].set_ylim((0,10))
        ax[1].set_yticks((np.arange(0,10,2)))
        
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        return fig, ax


class SN():
    """A class to plot S/N related info"""
    def __init__(self) -> None:
        pass

    def plot_CO_covariance(self, cov, mean, vmin=0, vmax=2):
        cov /= mean
        cov = (cov.T / mean).T
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(cov, extent=[0, cov.shape[0], 0, cov.shape[1]], vmin=vmin, vmax=vmax,
                       origin='lower', cmap=cmap, interpolation=None)
        cb = fig.colorbar(im , ax=ax, orientation='horizontal', fraction=0.17, pad=0.01)
        #cb.ax.tick_params(labelsize=5, width=5, length=2)
        cb.set_label(r'$r(k)$')
        ax.grid()

        return cov
    
    def details(self, fig, ax, sts, labels):
        signal_plotted = False
        for st, label in zip(sts, labels):
            if st.lya_pk is not None:
                if not signal_plotted:
                    ax.plot(st.lya_pk['k'], st.lya_pk['power'], label='signal')
                    signal_plotted = True
                ax.plot(st.lya_pk['k'], st.sigma_lya_pk, ls='--', label=f'Noise {label}')
                ax.set_ylabel(r'$P_{Ly\alpha}(k)$'+' or '+r'$P_{n, Ly\alpha}(k)$')
            if st.gal_pk is not None:
                ax.plot(st.gal_pk['k'], st.gal_pk['power'], label=f'Signal {label}')
                ax.plot(st.gal_pk['k'], st.gal_noise_pk, ls='--', label=f'Noise {label}')
            if st.lim_pk is not None:
                ax.plot(st.lim_pk['k'], np.abs(st.lim_pk['power']), label=f'Signal  LIM ')
                ax.plot(st.lim_pk['k'], np.abs(st.sigma_lim_pk), ls='--', label=f'Noise LIM ')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(framealpha=0)
        ax.set_xlabel(r'k[h/Mpc]')
        ax.grid(which='both', axis='both')



    def plot_mcmc(slef, fig, ax, stats_lya=None, stats_gal=None, vol_ratio=3, labels=[''], 
                ls='dashed', plot_co =False, lw=[4.5,6]):
        """Plot S/N curves for MCMCs of a mock survey """
        for i in range(len(stats_gal)):
                st = stats_gal[i]
                k = st.lim_pk['k']
                sn_co = st.get_co_sn()
                rk = st.lim_gal_pk['power'] / np.sqrt(st.lim_pk['power'] * st.gal_pk['power'])

                ax[0].plot(k, np.nanmedian(rk, axis=0), ls=ls, lw=lw[j], label=label)
                ax[0].fill_between(k, np.nanquantile(rk, .16, axis=0), 
                                np.nanquantile(rk, .84, axis=0), alpha=0.3, ls=ls)    
                if plot_co and i==0:
                    ax[j+1].plot(k, np.nanmedian(sn_co, axis=0), ls=ls, label='COMAP', color='k')
                ax[j+1].plot(k, np.nanmedian(sn, axis=0), alpha=0.5, label=labels[i], 
                        ls=ls, lw=lw[j])
        
    def compare_lya_gal(self, title='COMAP CO Model', vol_ratio=[1,5], savefile=None):
        """Comparison plot between S/N curves of  CO X Lya and CO X galaxy surveys"""
        fig, ax = plt.subplots(1,3, figsize=(21,6))
        
        self.plot_mcmc(fig, ax, stat_func=self.get_lya_stat, surveys=['LATIS', 'PFS','eBOSS'], 
                vol_ratio=vol_ratio, ls='solid', plot_co=True, labels=[r'$d_{\perp} = 2.5 \ cMpc/h$',
                                        r'$d_{\perp} = 4 \ cMpc/h$',r'$d_{\perp} = 13 \ cMpc/h$'])
        self.plot_mcmc(fig, ax, stat_func=self.get_gal_stat, surveys=['Rz7e-4','Rz02'],
                vol_ratio=vol_ratio, labels=[r'$Rz = 7 \times 10^{-4}$',r'$ Rz = 2 \times 10^{-2}$'])
        
        for i in range(3):
            ax[i].set_xscale('log')
            ax[i].grid(axis='both', which='both')
            ax[i].set_xlabel('$k \ h(cMpc)^{-1}$')
        ax[0].set_ylabel(r'$r(k) = \frac{P_{\times}(k)}{\sqrt{(P_{A} (k) \times P_{B} (k)}}$')
        ax[1].set_ylabel('S/N')
        ax[0].legend(framealpha=0)
        ax[1].legend(framealpha=0)
        ax[1].set_ylim((0,20))
        ax[2].set_ylim((0,20))
        ax[0].set_title('Perfect \ Surveys', fontsize=20)
        ax[1].set_title(r'$Mock \ surveys, \ v_{PFS} = V_{COMAP-Y5}/5$', fontsize=20)
        ax[2].set_title(r'$Mock \ surveys. \ V_{COMAP-Y5}$', fontsize=20)
        
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        
        if savefile is not None:
            fig.savefig(savefile)

    def compare_tot_sn(self, plot=True, savefile=None):
        """Comparison between total S/N of  CO X Lya and CO X galaxy surveys
        """
        lya_surveys=['LATIS','PFS','eBOSS']
        lya_labels = [r'$d_{\perp} = 2.5 \ cMpc/h$', r'$d_{\perp} = 4 \ cMpc/h$', r'$d_{\perp} = 13 \ cMpc/h$']
        gal_surveys=['Rz7e-4','Rz02']
        gal_labels = [r'$R_{z} = 7 \times 10^{-4}$', r'$R_{z} = 2 \times 10^{-2}$']
        sn_lya, sn_gal = {}, {}
        err_lya, err_gal = np.zeros((2,len(lya_surveys))), np.zeros((2,len(gal_surveys)))
        
        for i,s in enumerate(lya_surveys):
            _, _, sn, sn_lim = self.get_lya_stat(survey_type=s, vol_ratio=5)
            sn_lya[s] = np.linalg.norm(np.nanmedian(sn, axis=0))
            err_lya[:,i] = [sn_lya[s] - np.linalg.norm(np.nanquantile(sn, 0.16, axis=0)) , 
                np.linalg.norm(np.nanquantile(sn, 0.84, axis=0)) - sn_lya[s]]

            #err_lya[s] = np.sum(np.nanmedian(sn, axis=0))
        for i,s in enumerate(gal_surveys):
            _, _, sn, _ = self.get_gal_stat(survey_type=s, vol_ratio=5)
            sn_gal[s] = np.linalg.norm(np.nanmedian(sn, axis=0))
            err_gal[:,i] = [sn_gal[s] - np.linalg.norm(np.nanquantile(sn, 0.16, axis=0)) , 
                np.linalg.norm(np.nanquantile(sn, 0.84, axis=0)) - sn_gal[s]]

        sn_lim_median = np.linalg.norm(np.nanmedian(sn_lim, axis=0))
        err_lim = [sn_lim_median - np.linalg.norm(np.nanquantile(sn_lim, 0.16, axis=0)) , 
                np.linalg.norm(np.nanquantile(sn_lim, 0.84, axis=0)) - sn_lim_median]
        err_lim = np.array(err_lim)
        err_lim.reshape((2,1))
        
        if plot:
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            ax.bar(['COMAP'], sn_lim_median)
            ax.bar(lya_labels, sn_lya.values(), yerr=err_lya)
            ax.bar(gal_labels, sn_gal.values(), yerr=err_gal)
            ax.set_ylabel('S/N')
            ax.grid(True, axis='y')
            ax.set_title('COMAP-Y5')
            plt.xticks(rotation=-30)
            plt.yticks(np.arange(0,90,5))
            fig.tight_layout()
            if savefile is not None:
                fig.savefig(savefile)
        else :
            return sn_lim, sn_lya, sn_gal

    def plot_single_mock(self, sts, labels, colors, fig=None, ax=None, title='', plot_uncer=True, plot_rk=True, alpha=0.90, legend=True, lw=3):
        """Plot S/N curves for single CO model (CO X Galaxies)
        Paramters :
            sts: A list of stats.Stats instances
            labels : a list of corresponding labels
            colors : a list of corresponding colors
         """
        if fig is None:
            if plot_uncer:
                fig, ax = plt.subplots(1,3, figsize=(21,6))
                s=1
            else:
                fig, ax = plt.subplots(1,2, figsize=(18,6))
                s=0
        else:
            s = 0 
         
        label_lya = r'$CO \times Lya$'
        label_gal = r'$CO \times Gal$'
        for i, st in enumerate(sts):
            if st.gal_pk is not None:
                rk = np.abs(st.lim_gal_pk['power'][:]/ np.sqrt(st.gal_pk['power'][:]*
                                                                    st.lim_pk['power'][:]))
                if plot_uncer:
                    ax[0].plot(st.gal_pk['k'][:], np.abs(st.lim_gal_pk['power'][:]), label=r'$P_{Gal \times CO} $'+labels[i], ls='solid', alpha=0.7, color=colors[i])
                    ax[0].plot(st.gal_pk['k'][:], np.abs(st.sigma_co_gal_pk[:]), alpha=0.7, ls='--', color=colors[i])
                if label_gal is not None:
                    ax[s].plot(st.gal_pk['k'][:], np.abs(rk), alpha=0.7, color=colors[i], label=label_gal, ls='--')
                #label_gal = None

                ax[s+1].plot(st.gal_pk['k'][:], st.co_gal_sn, color=colors[i], ls='--', label=labels[i], alpha=alpha, lw=lw)
        
            if st.lya_pk is not None:
                rk = np.abs(st.lim_lya_pk['power'][:]/ np.sqrt(st.lya_pk['power'][:]*
                                                                    st.lim_pk['power'][:]))
                if plot_uncer:
                    ax[0].plot(st.lya_pk['k'][:], np.abs(st.lim_lya_pk['power'][:]), label=r'$P_{Lya \times CO} $'+labels[i], ls='solid', alpha=0.7, color=colors[i])
                    ax[0].plot(st.lya_pk['k'][:], np.abs(st.sigma_co_lya_pk[:]), alpha=0.7, ls='--', color=colors[i])
                if label_lya is not None:
                    ax[s].plot(st.lya_pk['k'][:], np.abs(rk), alpha=0.7, color=colors[i], label=label_lya)
                    label_lya = None
                ax[s+1].plot(st.lya_pk['k'][:], st.co_lya_sn, color=colors[i], label=labels[i], alpha=alpha, lw=lw)
        st.get_co_sn()
        ax[s+1].plot(st.lim_pk['k'][:], st.co_sn, ls='solid', color='k', label='COMAP-Y5', alpha=alpha, lw=lw)
        if plot_uncer:
            ax[0].plot(st.lim_pk['k'][:], np.abs(st.lim_pk['power'][:]), label=r'$P_{CO} $', ls='solid', alpha=0.7, color='k')
            ax[0].plot(st.lim_pk['k'][:], np.abs(st.sigma_co_pk[:]), ls='--', alpha=0.7, color='k')

            
        for i in range(s+2):
            ax[i].set_xscale('log')
            ax[i].grid(axis='both', which='both')
            ax[i].set_xlabel('$k \ (h/cMpc)$')
            ax[i].set_xlim(2e-2,1)
        if plot_uncer:
            ax[0].set_xscale('log')
            ax[0].grid(axis='both', which='both')
            ax[0].set_xlabel('$k \ (h/cMpc)$')
            ax[0].set_xlim(2e-2,1)
            ax[0].legend(framealpha=0, loc='lower left', fontsize=20)
            ax[0].set_yscale('log')
            ax[0].set_ylim(1e-2,1e5)
        ax[s].set_ylim((0,1))
        ax[s+1].set_ylabel('S/N')
        #ax[s+1].set_ylim((0,16))
        print(s+1)
        #ax[s+1].set_yticks((np.arange(0,20,2)))
        ax[s].set_ylabel(r'$r(k) = \frac{P_{\times}(k)}{\sqrt{(P_{A} (k) \times P_{B} (k)}}$')
        if legend:
            ax[s].legend()
            ax[s+1].legend( framealpha=0.7, loc=(1.05, 0.0), fontsize=16, facecolor=None,
                      frameon=True)   
        
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        return fig, ax
    
    def compare_tot_sn_single_mock(self, sts_lya, sts_gal, fig=None, ax=None, savefile=None, 
                                alpha=0.95, lya_labels=None, auto_label='COMAP-Y5', gal_labels=None, text_offset=40):
        """Comparison between total S/N of  CO X Lya and CO X galaxy surveys
        The version used for a single mock
        """
        """
        if len(sts_lya) == 4:
            lya_labels = [r'$\ d_{\perp} = \\ 2.5 \ cMpc/h$', r'$\ d_{\perp} = \\ 4 \ cMpc/h$', r'$\ d_{\perp} =\\ 10 \ cMpc/h$',r'$d_{\perp} =\\ 13 \ cMpc/h$']
        else :
            lya_labels = [r'$\ d_{\perp} = \\ 2.5 \ cMpc/h$', r'$d_{\perp} = \\ 4 \ cMpc/h$', r'$d_{\perp} =\\ 13 \ cMpc/h$']
        
        gal_labels =[  r'$R_{z} = \\ 7 \times 10^{-4}$', r'$R_{z} = \\ 2 \times 10^{-2}$']
        """
        sn_lya, sn_gal = {}, {}
        for i,s in enumerate(lya_labels):
            sn_lya[s]= np.linalg.norm(sts_lya[i].lim_lya_sn)
        for i,s in enumerate(gal_labels):
            sn_gal[s] = np.linalg.norm(sts_gal[i].lim_gal_sn)
        sn_lim = np.linalg.norm(sts_lya[0].lim_sn)
        if fig is None:
            fig, ax = plt.subplots(1,1, figsize=(12,6))
        labels = [auto_label]+gal_labels+lya_labels

        ax.barh([0], sn_lim, alpha=alpha, color='C0')
        ax.barh(1+np.arange(len(gal_labels)), sn_gal.values(), alpha=alpha, color='C2')
        ax.barh(1+len(gal_labels)+np.arange(len(lya_labels)), sn_lya.values(), alpha=alpha, color='C1')
        ypos = np.arange(len(labels))
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        
        ax.text(text_offset, 0-0.25, f'{sn_lim:.0f}', fontsize=25 , alpha=alpha)
        for i in range(len(gal_labels)):
            ax.text(text_offset, 1+i - 0.25, f'{list(sn_gal.values())[i]:.0f}', fontsize=25 , alpha=alpha)
        for i in range(len(lya_labels)):
            ax.text(text_offset, 1+len(gal_labels)+i -0.25, f'{list(sn_lya.values())[i]:.0f}', fontsize=25, alpha=alpha)
        
        ax.set_xlabel('total \ S/N')
        ax.set_ylabel('Survey')
        ax.grid(True, axis='y')
        ax.yaxis.tick_right()
        ax.set_xticks(np.arange(0,55,5))
        if savefile is not None:
            fig.savefig(savefile)
    
        return sn_lim, sn_lya, sn_gal

    def plot_simple_SN(self, fig, ax, sts, labels, colors, auto_label='COMAP-Y5', title='', legend=True):

        
        for i, st in enumerate(sts):
            print(i)
            st =sts[i]
            if st.gal_pk is not None:
                ax.plot(st.gal_pk['k'][:], st.lim_gal_sn, color=colors[i], ls='--', label=labels[i])

            if st.lya_pk is not None:
                ax.plot(st.lya_pk['k'][:], st.lim_lya_sn, color=colors[i], label=labels[i])
        # Using the first stat paased for plotting auto LIM signal, be careful when stats have different
        # volumes (e.g. for Exclaim mocks)
        ax.plot(sts[0].lim_pk['k'][:], sts[0].lim_sn, ls='solid', color='k', label=auto_label)

        ax.set_xscale('log')
        ax.grid(axis='both', which='both')
        ax.set_xlabel('$k \ (h/cMpc)$')
        ax.set_xlim(2e-2,1)
        ax.set_ylabel('S/N')
        ax.set_ylim((-0.1,14))
        ax.set_yticks((np.arange(0,16, 2)))

        if legend:
            ax.legend( framealpha=0.7, loc=(-0.7, 0.0), fontsize=20, facecolor=None,
                      frameon=True)   
        return fig, ax

    def plot_rk(self, sts, fig, ax, color, labels, lss=None):

        for i, st in enumerate(sts):
            if st.gal_pk is not None:
                rk = (st.lim_gal_pk['power'][:]/ np.sqrt(st.gal_pk['power'][:]*
                                            st.lim_pk['power'][:])).squeeze()
                ax.plot(st.gal_pk['k'][:], rk, alpha=0.9, color=color[i], label=labels[i], ls='--')
                #ax.errorbar(x=st.gal_pk['k'][:], y=-1*rk, yerr=np.sqrt(3)*rk/np.sqrt(st.lim_pk['modes'][:]), fmt='-', alpha=0.9, label=labels[i], color=color[i])
            if st.lya_pk is not None:
                rk = st.lim_lya_pk['power'][:]/ np.sqrt(st.lya_pk['power'][:]*
                                            st.lim_pk['power'][:])
                if lss is not None:
                    ls = lss[i]
                else:
                    ls=None
                ax.plot(st.lya_pk['k'][:], -1*rk, alpha=0.9, label=labels[i], color=color[i], ls=ls)
                #ax.errorbar(x=st.lya_pk['k'][:], y=-1*rk, yerr=np.sqrt(3)*rk/np.sqrt(st.lim_pk['modes'][:]), fmt='-', 
                #                            elinewidth=3,capsize=5, alpha=0.5, label=labels[i], color=color[i])
        ax.set_ylim(0,1)
        ax.set_xscale('log')
        ax.grid(axis='both', which='both')
        ax.set_ylabel(r'$| r(k) |$')
        ax.set_xlabel('$k \ (h/cMpc)$')
        ax.legend(loc='lower left', framealpha=0)

            

    def plot_pkmu(self,st, plot_Nmodes=False, title='', savefig=None):
        """Plot 2D P(k, mu) for all avaiable statistics.
        Parameters:
        ----------------------------
            st: An instance of `lim_lytomo.stats.Stats()`
        """

        powers = []
        sigma_pks = []
        labels = []
        if st.co_pkmu is not None:
            ind = np.where(np.abs(st.co_pkmu['power']) > 0)
            pow = np.zeros_like(np.abs(st.co_pkmu['power'][:]))
            sig = np.zeros_like(np.abs(st.co_pkmu['power'][:]))
            pow[ind] = np.abs(st.co_pkmu['power'][ind])
            sig[ind] = np.abs(st.sigma_co_pkmu[ind])
            powers.append(pow)
            sigma_pks.append(sig)
            labels.append(r'$P_{CO}$')
        if st.gal_pkmu is not None:
            ind = np.where(np.abs(st.gal_pkmu['power']) > 0)
            pow = np.zeros_like(np.abs(st.gal_pkmu['power'][:]))
            sig = np.zeros_like(np.abs(st.gal_pkmu['power'][:]))
            pow[ind] = np.abs(st.gal_pkmu['power'][ind])
            sig[ind] = np.abs(st.sigma_gal_pkmu[ind])
            powers.append(pow)
            sigma_pks.append(sig)
            labels.append(r'$P_{Gal}$')
            # CO X Gal
            ind = np.where(np.abs(st.co_gal_pkmu['power']) > 0)
            pow = np.zeros_like(np.abs(st.co_gal_pkmu['power'][:]))
            sig = np.zeros_like(np.abs(st.co_gal_pkmu['power'][:]))
            pow[ind] = np.abs(st.co_gal_pkmu['power'][ind])
            sig[ind] = np.abs(st.sigma_co_gal_pkmu[ind])
            powers.append(pow)
            sigma_pks.append(sig)
            labels.append(r'$P_{CO X Gal}$')

        if st.lya_pkmu is not None:
            ind = np.where(np.abs(st.lya_pkmu['power']) > 0)
            pow = np.zeros_like(np.abs(st.lya_pkmu['power'][:]))
            sig = np.zeros_like(np.abs(st.lya_pkmu['power'][:]))
            pow[ind] = np.abs(st.lya_pkmu['power'][ind])
            sig[ind] = np.abs(st.sigma_lya_pkmu[ind])
            powers.append(pow)
            sigma_pks.append(sig)
            labels.append(r'$P_{Lya}$')
            pow = np.zeros_like(np.abs(st.co_lya_pkmu['power'][:]))
            sig = np.zeros_like(np.abs(st.sigma_CO_lya_pkmu['power'][:]))
            pow[ind] = np.abs(st.co_lya_pkmu['power'][ind])
            sig[ind] = np.abs(st.sigma_co_lya_pkmu[ind])
            powers.append(pow)
            sigma_pks.append(sig)
            labels.append(r'$P_{cO X Lya}$')


        
        num_axs = len(labels)
        if plot_Nmodes:
            num_axs+=1
        fig, ax = plt.subplots(2,num_axs, figsize=(6*num_axs,12))

        for i in range(len(powers)):
            cmap = plt.get_cmap('jet')
            im = ax[0,i].imshow(powers[i][:,np.min(ind[1])::], origin='lower', cmap=cmap, interpolation='bilinear',
                        extent=[0, 1, st.kmin, st.kmax], norm=matplotlib.colors.LogNorm())
            cb = fig.colorbar(im , ax=ax[0,i], orientation='horizontal', fraction=0.1, pad=0.2)
            cb.set_label(labels[i])
            ax[0,i].set_ylabel(r'$k$')
            ax[0,i].set_xlabel(r'$\mu$')
            ax[0,i].grid(which='both', axis='both')

            im = ax[1,i].imshow(sigma_pks[i][:,np.min(ind[1])::], origin='lower', cmap=cmap, interpolation='bilinear',
                        extent=[0, 1, st.kmin, st.kmax], norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e12))
            cb = fig.colorbar(im , ax=ax[1,i], orientation='horizontal', fraction=0.1, pad=0.2)
            cb.set_label(r'$\sigma$'+labels[i])
            ax[1,i].set_ylabel(r'$k$')
            ax[1,i].set_xlabel(r'$\mu$')
            ax[1,i].grid(which='both', axis='both')

            if i== len(powers)-1:
                Nmodes = np.zeros_like(powers[i])
                Nmodes[ind] = st.co_pkmu['modes'][ind]
                im = ax[0,i+1].imshow(Nmodes[:,np.min(ind[1])::], origin='lower', cmap=cmap, interpolation='bilinear',
                            extent=[0, 1, st.kmin, st.kmax])
                cb = fig.colorbar(im , ax=ax[0,i+1], orientation='horizontal', fraction=0.1, pad=0.2)
                cb.set_label(r'$N_{modes}$')
                ax[0,i+1].set_ylabel(r'$k$')
                ax[0,i+1].set_xlabel(r'$\mu$')
                ax[0,i+1].grid(which='both', axis='both')
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()

        if savefig is not None:
            fig.savefig(savefig)


