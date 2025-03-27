# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'

from matplotlib import gridspec
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

font = {'weight' : 'normal'}

matplotlib.rc('font', **font)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# -

try:
    from jdutil import jd_to_date, mjd_to_jd, jd_to_mjd, date_to_jd
except:
    from libs.jdutil import jd_to_date, mjd_to_jd, jd_to_mjd, date_to_jd


def add_year_ticks(ax):
    ax1 = ax.twiny()
    
    min_year, min_month, min_day = jd_to_date(mjd_to_jd(ax.get_xlim()[0]))
    max_year, max_month, max_day = jd_to_date(mjd_to_jd(ax.get_xlim()[1]))
    
    mjds = []
    labels = []
    if max_year - min_year == 1:
        
        for y in range(min_year, max_year + 1):
            for m in range(1, max_month + 1):
                mjds.append(jd_to_mjd(date_to_jd(y, m, 1)))
                labels.append("%s" % "%s/%s" % (y, m))        
    else:
        for y in range(min_year, max_year + 1):
            mjds.append(jd_to_mjd(date_to_jd(y, 1, 1)))
            labels.append("%s" % y)
    labels[-1] = ''
 
    ax1.set_xlim(ax.get_xlim())
    ax1.set_xticks(mjds)
    ax1.set_xticklabels(["                               %s" % l for l in labels])
    ax1.tick_params(axis="x")


class Labeloffset():
    def __init__(self,  ax, label_first=None, label="", axis="y"):
        self.axis = {"y": ax.yaxis, "x": ax.xaxis}[axis]
        self.label = label
        self.label_first = label_first
        
        ax.callbacks.connect(axis + 'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        text = ""
        _off = fmt.get_offset().replace('\\times', '')
        if self.label_first is None:
            text = self.label.replace("(", "[%s$\,$" % _off)
        else:
            text = (self.label_first + "\n" + self.label).replace("\n(", "\n(%s$\,$" % _off)
        self.axis.set_label_text(text, size=12.5, weight='normal')







def plot_lcs(  *DS, 
               ylims=None, 
               extra_data=None,
               figsize=(10, 10), 
               highlights=None, 
               color='#1f77b4',
               extra_color='black',
               filename=None,
               timerange=None, 
               grid=False, 
               inline_labels=False,
               TS=9,
               show_ul=True,
               inset_id = 0,
               inset_creator = None,
               extra_plot = None
             ):
    subs = len(DS)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(subs, 1, height_ratios=[1] * subs)
    ax, lines, yticks = [None] * subs, [None] * subs, [None] * subs
    
    for i in range(subs):
        data = DS[i]
        title = DS[i].meta['title']
        units = DS[i].meta['f_units']
        ax[i] = plt.subplot(gs[i]) if i == 0 else plt.subplot(gs[i], sharex = ax[0])
        
        if extra_data and extra_data[i] is not None:
            edata = extra_data[i]
            edata = edata[~np.isin(edata['tc'], data['tc'])]
            
            ul_used = False
            ul_data_t, ul_data_f, ul_data_df = None, None, None
            nul_data_t, nul_data_f, nul_data_df = None, None, None 
            if 'UL95' in edata.columns:
                ul_used = True
                m = edata['TS'] < TS
                if edata['UL95'].dtype != np.float64:
                    m = edata['UL95'] != 'False'
                ul_data_t = edata['tc'][m]
                ul_data_f = edata['UL95'][m].astype(float)
                ul_data_df = edata['df'][m]

                nul_data_t = edata['tc'][~m]
                nul_data_f = edata['f'][~m]
                nul_data_df = edata['df'][~m]
            else:
                nul_data_t = edata['tc']
                nul_data_f = edata['f']
                nul_data_df = edata['df']
            
            if ul_used:
                if show_ul == True:
                    lines[i] = ax[i].errorbar(ul_data_t, ul_data_f, marker='v', markersize=2, linestyle = 'None', ecolor='grey', color=extra_color)
                lines[i] = ax[i].errorbar(nul_data_t, nul_data_f, yerr=nul_data_df, marker='o', markersize=2, linestyle = 'None', ecolor='grey', color=extra_color)
            else:
                lines[i] = ax[i].errorbar(nul_data_t, nul_data_f, yerr=nul_data_df, marker='o', markersize=2, linestyle = 'None', ecolor='grey', color=extra_color)


        ul_used = False
        ul_data_t, ul_data_f, ul_data_df = None, None, None
        nul_data_t, nul_data_f, nul_data_df = None, None, None 
        if 'UL95' in data.columns:
            print("%s using UL" % DS[i].meta['source'])
            ul_used = True
            m = data['TS'] < TS
            if data['UL95'].dtype != np.float64:
                m = data['UL95'] != 'False'
            ul_data_t = data['tc'][m]
            ul_data_f = data['UL95'][m].astype(float)
            ul_data_df = data['df'][m]

            nul_data_t = data['tc'][~m]
            nul_data_f = data['f'][~m]
            nul_data_df = data['df'][~m]
        else:
            nul_data_t = data['tc']
            nul_data_f = data['f']
            nul_data_df = data['df']
                
        if ul_used:
            if show_ul == True:
                lines[i] = ax[i].errorbar(ul_data_t, ul_data_f, marker='v', markersize=2, linestyle = 'None', ecolor='grey', color=color)
            lines[i] = ax[i].errorbar(nul_data_t, nul_data_f, yerr=nul_data_df, marker='o', markersize=2, linestyle = 'None', ecolor='grey', color=color, label=title)
        else:
            lines[i] = ax[i].errorbar(nul_data_t, nul_data_f, yerr=nul_data_df, marker='o', markersize=2, linestyle = 'None', ecolor='grey', color=color, label=title)

        if timerange is not None:
            _d = 0.005 * (timerange[1] - timerange[0])
            ax[i].set_xlim([timerange[0] - _d, timerange[1] + _d])
        ax[i].tick_params(axis="y", labelsize=13)
        ax[i].tick_params(axis="x", labelsize=13)    
        ax[i].grid(grid)
        
        if inset_creator is not None and inset_id == i:
            axins = inset_axes(ax[i], width=2, height=0.9)
            inset_creator(axins)
        
        if inline_labels:
            ax[i].legend()
        
        if highlights is not None and highlights[i] is not None:
            if isinstance(highlights[i][0], list):
                for h in highlights[i]:
                    ax[i].axvspan(h[0], h[1], color='lightgray', alpha=0.25)
            else:  
                ax[i].axvspan(highlights[i][0], highlights[i][1], color='lightgray', alpha=0.25)
        if ylims is not None and ylims[i] is not None:
            ax[i].set_ylim(ylims[i])
                        
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3,3))
        ax[i].yaxis.set_major_formatter(formatter)
        if not inline_labels:
            lo = Labeloffset(ax[i], label_first=title, label="(%s)" % units, axis="y")
        else:
            lo = Labeloffset(ax[i], label_first="Flux", label="(%s)" % units, axis="y")
        
        if i != subs - 1:
            ax[i].tick_params(axis="x", labelsize=0)
            plt.setp(ax[i].get_xticklabels(), visible=False)

    if extra_plot is not None:
        extra_plot(ax)
        
    add_year_ticks(ax[0])
    ax[subs - 1].set_xlabel("$t$ (MJD)", size=13, weight='normal')
    plt.subplots_adjust(hspace=.0)
    if filename is not None:
        #fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from dataset import radio_3c273, gev_3c273, radio_3c273_orig, coincidence_mask_3c273, gev_3c273_7d
    from dataset import radio_mrk421, gev_mrk421, radio_mrk421_orig, coincidence_mask_mrk421
    from dataset import radio_mrk501, gev_mrk501, radio_mrk501_orig, coincidence_mask_mrk501
    
    def test(ax):
        #x = np.linspace(1, 2)
        #y = x ** 2
        #ax.plot(x, y)
        pass
    
    plot_lcs(gev_3c273, radio_3c273, figsize=(15,8), timerange=gev_3c273.meta['timerange'], inset_creator=test, inset_id=1, show_ul=False)
    
    plot_lcs(gev_mrk421, radio_mrk421, figsize=(15,8), timerange=gev_mrk421.meta['timerange'], inset_creator=test, inset_id=1, show_ul=False)
    
    plot_lcs(gev_mrk501, radio_mrk501, figsize=(15,8), timerange=gev_mrk501.meta['timerange'], inset_creator=test, inset_id=1, show_ul=False)















