import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib.colors as colors
from matplotlib import ticker
import math

font = {"weight": "normal", "size": 12, "family": "STIXGeneral"}
axislabelfontsize = "x-large"
plt.rc("font", **font)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["text.usetex"] = True

def newlegend(col=1, fsize=12, space=None):
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    plt.gca().legend(*zip(*unique), fontsize=fsize, ncol=int(col), columnspacing=space)

def get_cmap(n, name='jet'):
    return plt.cm.get_cmap(name, n)

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

class Visual:
    def __init__(self, nu, nLST, ant, save=False, prefix="", index=["", ""]):
        """Initialize to generate the plots.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
            save (bool, optional): Option to save the plot. Defaults to False.
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.save = save
        self.prefix = prefix
        self.index = index
        if self.index == ["", ""]: self.ind = ""
        else: self.ind = "_iFg21-%d-%d"%(self.index[0], self.index[1])
        if self.save:
            if not os.path.exists('FigsOutput'): os.mkdir('FigsOutput')
    
    def getSaveName(self):
        return "_ant-%s_tbins-%d"%(''.join(a[0] for a in self.ant), self.nLST)
        
    def plotModset(self, set, opt, n_curves=100):
        """Plots the modeling set.

        Args:
            set (array): Modeling set
            opt (string): '21' or 'FG'
            n_curves (int, optional): Number of curves to plot. Defaults to 100.
        """
        plt.figure(figsize=(6, 4))
        if opt == '21':
            plt.plot(self.nu, set[:, :n_curves], lw=0.2)
        if opt == 'FG':
            cmap = get_cmap(self.nLST)
            for i in range(self.nLST):
                plt.plot(self.nu, set[i*len(self.nu):(i+1)*len(self.nu), :n_curves],
                         c=cmap(i), label=r'$t_{%d}$'%(i+1), alpha=0.2, lw=0.8)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            newlegend()
        plt.xlim(50, 200)
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        if self.save: plt.savefig('FigsOutput/modeling%sSet.pdf'%opt, bbox_inches='tight')
        else: plt.show()

    def plotMockObs(self, y21, yFg, noise):
        """Plots the mock observation.

        Args:
            y21 (array): Input 21cm component
            yFg (array): Input beam-weighted foreground component
            noise (array): Noise realization
        """
        plt.figure(figsize=(14, 3.5))
        plt.subplot(131)
        plt.plot(self.nu, y21, label=r'$y_{21}$', c='k')
        plt.legend()
        plt.xlim(50, 200)
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        
        plt.subplot(132)
        cmap = get_cmap(self.nLST)
        lsty = ['-', '--', ':']
        for j in range(len(self.ant)):
            for i in range(self.nLST):
                plt.plot(self.nu, yFg[(i+self.nLST*j)*len(self.nu)
                                      :(i+self.nLST*j+1)*len(self.nu)],
                         c=cmap(i), ls=lsty[j],
                         label=r'$(y_{\rm FG})_{t_{%d}}^{\rm %s}$'%(i+1, self.ant[j]))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.xlim(50, 200)
        newlegend(fsize=4/len(self.ant)+6, col=len(self.ant), space=1)
        
        plt.subplot(133)
        for j in range(len(self.ant)):
            for i in range(self.nLST):
                plt.plot(self.nu, noise[(i+self.nLST*j)*len(self.nu)
                                        :(i+self.nLST*j+1)*len(self.nu)],
                        c=cmap(i), ls=lsty[j],
                        label=r'$(n)_{t_{%d}}^{\rm %s}$'%(i+1, self.ant[j]))                
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        newlegend(fsize=4/len(self.ant)+6, col=len(self.ant))
        plt.xlim(50, 200)
        if self.save: plt.savefig('FigsOutput/%smockObs%s%s.pdf'
                                  %(self.prefix, self.getSaveName(), self.ind),
                                  bbox_inches='tight')
        else: plt.show()

    def plotBasis(self, basis, opt, n_curves=5):
        """Plots the basis functions.

        Args:
            basis (array): Basis functions (or modes)
            opt (string): '21' or 'FG'
            n_curves (int, optional): Number of modes to plot. Defaults to 5.
        """
        plt.figure(figsize=(6, 4))
        cmap = get_cmap(n_curves)
        if opt == '21':
            for i in range(n_curves):
                # plt.plot(self.nu, basis.T[i], c=cmap(i),
                #          label=r'$F^{21}_{%d}$'%(i+1))
                plt.plot(self.nu, basis[:, i], c=cmap(i),
                         label=r'$F^{21}_{%d}$'%(i+1))
            saveName = 'basis%s.pdf'%opt
        if opt == 'FG':
            lsty = ['-', '--', '-.', ':']*3
            for i in range(n_curves):
                for j in range(self.nLST):
                    # plt.plot(self.nu, basis.T[i, j*len(self.nu):(j+1)*len(self.nu)],
                    #          c=cmap(i), ls=lsty[j], label=r'$F^{\rm FG}_{%d\ (t_{%d})}$'%(i+1, j+1))
                    plt.plot(self.nu, basis[j*len(self.nu):(j+1)*len(self.nu), i],
                             c=cmap(i), ls=lsty[j], label=r'$F^{\rm FG}_{%d\ (t_{%d})}$'%(i+1, j+1))
            saveName = 'basis%s%s.pdf'%(opt, self.getSaveName())
        plt.legend(fontsize=8.5, ncol=n_curves)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xlim(50, 200)
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        if self.save: plt.savefig('FigsOutput/%s'%saveName, bbox_inches='tight') 
        else: plt.show()

    def plotInfoGrid(self, file, modesFg, modes21, quantity='DIC',
                     minModesFg=None, minModes21=None):
        """Plots the information criteria estimated on a grid.

        Args:
            file (string): Filename that contains the gridded information
            modesFg (int): Number of foreground modes 
            modes21 (int): Number of 21cm modes
            quantity (string, optional): IC ('DIC' or 'BIC'). Defaults to 'DIC'.
            minModesFg, minModes21 (int, optional): To highlight the minimized IC pixel.
        """
        modesFg = np.linspace(1, modesFg, num=modesFg)
        modes21 = np.linspace(1, modes21, num=modes21)

        fp = np.loadtxt(file)
        dicVals = fp[:, 2]
        X, Y = np.meshgrid(modesFg, modes21)
        DIC_array = dicVals.reshape(len(modesFg), len(modes21))

        plt.figure(figsize=(5.7, 5))
        plt.pcolormesh(X, Y, DIC_array.T, cmap="inferno_r",
                       shading='auto', rasterized=True,
                       norm=colors.LogNorm(vmin=min(dicVals), vmax=min(dicVals)*1.1))
        
        if minModesFg and minModes21 is not None:
            fig = plt.gcf(); ax = fig.gca()
            minima = plt.Rectangle((minModesFg-0.5, minModes21-0.5), 1, 1,
                                   color='darkcyan', fill=False)
            ax.add_patch(minima)
        
        plt.xlabel(r'Foreground modes $(n_{b}^{\rm FG})$')
        plt.ylabel(r'21cm modes $(n_{b}^{21})$')
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = plt.colorbar(aspect=30, format=fmt, extend='max', pad=0.02)
        if quantity == 'DIC':
            cbar.set_label(r"$\delta^T C^{-1} \delta + 2 (n_b^{\rm FG} + n_b^{21})$")
        if quantity == 'BIC':
            cbar.set_label(r"$\delta^T C^{-1} \delta + (n_b^{\rm FG} + n_b^{21}) \log\, n_c$")
        if self.save: plt.savefig('FigsOutput/%s%sGrid%s%s.pdf'
                                  %(self.prefix, quantity, self.getSaveName(), self.ind),
                                  bbox_inches='tight')
        else: plt.show()

    def plotExtSignal(self, y21, recons21, sigma21, ylim=[-0.5, 0.15]):
        """Plots the extracted 21cm signal with the input signal.

        Args:
            y21 (array): Input 21cm signal
            recons21 (array): Reconstructed signal
            sigma21 (array): 1sigma interval
        """
        plt.figure(figsize=(6, 4))
        plt.plot(self.nu, recons21[0:len(self.nu)], c='darkcyan',
                 label=r'$y_{21}^{\rm ext}$')
        plt.fill_between(self.nu, recons21[0:len(self.nu)] - (1*sigma21),
                        recons21[0:len(self.nu)] + (1*sigma21),
                        color="darkcyan", alpha=0.4)
        plt.plot(self.nu, y21, c='k', label=r'$y_{21}^{\rm inp}$')
        plt.legend()
        plt.xlim(50, 200)
        plt.ylim(ylim)
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        if self.save: plt.savefig('FigsOutput/%sextract21%s%s.pdf'
                                  %(self.prefix, self.getSaveName(), self.ind),
                                  bbox_inches='tight')
        else: plt.show()

    def plotBiasCDF(self, antNames, fnames, xlim=None, ylim=[0, 1], ncol=2, bins=5000, save=False):
        """Plots the CDF of the signal bias for an ensemble of mockobs.

        Args:
            antNames (list): Antennas   
            fnames (list): Files
            xlim (list, optional): X-axis limits. Defaults to None.
            bins (list, optional): Number of bins. Defaults to 5000.
            save (bool, optional): For saving the figure. Defaults to False.
        """
        plt.figure(figsize=(6.5, 4.3))
        color = get_cmap(len(fnames))
        lim = [0.68, 0.95]
        text = self.getCDF(paths=fnames, ind=3, lim=lim, labels=antNames, color=color, bins=bins)
        self.getTable(paths=fnames, labels=antNames, text=text, color=color)
        self.plotChiCDF()
        plt.xlabel(r"${\rm Signal\ bias\ statistic\ }(\varepsilon)$")
        plt.ylabel(r"${\rm CDF}$")
        plt.xlim(xlim); plt.ylim(ylim)
        plt.legend(loc='upper center', fontsize=8, ncol=ncol)
        if save: plt.savefig('FigsOutput/biasCDF_tbins-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()
        
    def plotRmsCDF(self, antNames, fnames, xlim=None, ylim=[0, 1], ncol=2, bins=5000, save=False):
        """Plots the CDF of the rms uncertainty for an ensemble of mockobs.

        Args:
            antNames (list): Antennas
            fnames (list): Files
            xlim (list, optional): X-axis limits. Defaults to None.
            bins (int, optional): Number of bins. Defaults to 5000.
            save (bool, optional): For saving the figure. Defaults to False.
        """
        plt.figure(figsize=(6.5, 4.3))
        color = get_cmap(len(fnames))
        lim = [0.68, 0.95]
        text = self.getCDF(paths=fnames, ind=5, lim=lim, labels=antNames, color=color, bins=bins)
        self.getTable(paths=fnames, labels=antNames, text=text, color=color)
        plt.xscale("log")
        plt.ylim(ylim); plt.xlim(xlim)
        plt.legend(loc='upper center', fontsize=8, ncol=ncol)
        plt.xlabel(r"${\rm RMS\ uncertainity\ }({\rm mK})$")
        plt.ylabel(r"${\rm CDF}$")
        if save: plt.savefig('FigsOutput/rmsCDF_tbins-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()
    
    def plotNormD(self, antNames, fnames, xlim=None, ylim=None, ncol=2, bins=20, save=False):
        """Plots the normalized Deviance for an ensemble of mockobs.

        Args:
            antNames (list): Antennas
            fnames (list): Files
            xlim (list, optional): X-axis limits. Defaults to None.
            bins (int, optional): Number of bins. Defaults to 20.
            save (bool, optional): For saving the figure. Defaults to False.
            bins (int, optional): Number of bins. Defaults to 20.
        """
        plt.figure(figsize=(6.5, 4.3))
        color = get_cmap(len(fnames))
        labels = antNames
        paths = fnames
        for i in range(len(paths)):
            f = np.loadtxt("%s"%paths[i])
            weights = np.ones_like(f[:,4])/float(len(f[:,4]))
            plt.hist(f[:,4], weights=weights, bins=bins, histtype='step', label=labels[i],
                     color=color(i))
        plt.xlim(xlim); plt.ylim(ylim)
        plt.legend(loc='upper center', fontsize=8, ncol=ncol)
        plt.xlabel(r"$\chi^2$")
        plt.ylabel(r"${\rm PDF}$")
        plt.tight_layout()
        if save: plt.savefig('FigsOutput/normDevPDF_tbins-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()

    def plotNormDnew(self, antNames, fnames, xlim=None, ylim=None, ncol=2, bins=20, save=False):
        """Plots the normalized Deviance for an ensemble of mockobs.

        Args:
            antNames (list): Antennas
            fnames (list): Files
            xlim (list, optional): X-axis limits. Defaults to None.
            bins (int, optional): Number of bins. Defaults to 20.
            save (bool, optional): For saving the figure. Defaults to False.
            bins (int, optional): Number of bins. Defaults to 20.
        """
        plt.figure(figsize=(6.5, 4.3))
        color = get_cmap(len(fnames))
        labels = antNames
        paths = fnames
        for i in range(len(paths)):
            f = np.loadtxt("%s"%paths[i])
            plt.hist(f[:, 4], density=True, bins=bins, histtype="step", label=labels[i],
                     color=color(i))
            # weights = np.ones_like(f[:,4])/float(len(f[:,4]))
            # plt.hist(f[:,4], weights=weights, bins=bins, histtype='step', label=labels[i],
            #          color=color(i))
        plt.xlim(xlim); plt.ylim(ylim)
        plt.legend(loc='upper center', fontsize=8, ncol=ncol)
        plt.xlabel(r"$\chi^2$")
        plt.ylabel(r"${\rm PDF}$")
        plt.tight_layout()
        if save: plt.savefig('FigsOutput/normDevPDF_tbins-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()    

    def plotModesDist(self, file, modesFg, modes21, contour=False, vmax=100,
                      nLevels=10, zoom=[None, None], save=False):
        """Plots the distribution of modes that minimize IC for an ensemble of inputs.

        Args:
            file (string): File to read in the statistical information.
            modesFg (int): Total number of FG modes over which IC was calculated.
            modes21 (int): Total number of 21 modes over which IC was calculated.
            contour (bool, optional): To show the contours. Defaults to False.
            vmax (int, optional): Max value in the distribution. Defaults to 100.
            nLevels (int, optional): Number of levels to show. Only for contours. Defaults to 10.
            zoom (list, optional): To zoom in the distribution. Defaults to [None, None].
            save (bool, optional): For saving the figure. Defaults to False.
        """
        data = np.loadtxt(file)
        mFg = np.array(data[:, 0], dtype=np.int32)
        m21 = np.array(data[:, 1], dtype=np.int32)
        info = np.zeros((modesFg, modes21))
        for lfg, l21 in zip(mFg, m21):
            info[lfg-1, l21-1] += 1
        
        xFg = np.linspace(1, modesFg, num=modesFg)
        y21 = np.linspace(1, modes21, num=modes21)
        XFg, Y21 = np.meshgrid(xFg, y21)
        
        plt.figure(figsize=(6, 5))
        if contour:
            plt.contourf(XFg-0.5, Y21-0.5, info.T, cmap='inferno', vmax=vmax,
                         levels=np.linspace(0, vmax, nLevels), extend='max')
            cbar = plt.colorbar(aspect=30, pad=0.02)
        else:
            plt.pcolormesh(XFg, Y21, info.T, cmap='inferno', vmax=vmax, shading='auto')
            cbar = plt.colorbar(aspect=30, extend='max', pad=0.02)
        cbar.set_label(r"Number of samples $(n_{\rm samples})$")
        plt.xlim(zoom[0]); plt.ylim(zoom[1])
        plt.xlabel(r'Foreground modes $(n_{b}^{\rm FG})$')
        plt.ylabel(r'21cm modes $(n_{b}^{21})$')
        if save: plt.savefig('FigsOutput/ModesDist%s.pdf'%self.getSaveName(),
                             bbox_inches='tight')
        else: plt.show()
        
    @staticmethod
    def getCDF(paths, ind, lim, labels, color, bins):
        text = np.zeros(shape=(len(paths), len(lim)))
        for i in range(len(paths)):
            f = np.loadtxt("%s"%paths[i])
            values, base = np.histogram(f[:, ind], bins=bins)
            cumulative = np.cumsum(values)/np.sum(values)
            for j in range(len(lim)):
                total_ind = list(np.where(cumulative-lim[j]>=pow(10,-31)))
                index = total_ind[0][0]
                text[i][j] = "%.2f"%base[:-1][index]
            plt.plot(base[:-1], cumulative, label = labels[i], color = color(i))     
        plt.axhline(0.68, c = "lightgray", ls = ":")
        plt.axhline(0.95, c = "lightgray", ls = "-.")   
        return text

    @staticmethod
    def getTable(paths, labels, text, color):
        cell_t = []
        for i in range(len(paths)):
            cell_t.append("%s" %labels[i])
            cell_t.append("%.2f"%text[:,0][i])
            cell_t.append("%.2f"%text[:,1][i])
        ct = np.reshape(cell_t, (len(paths), 3))
        cellC = []
        for i in range(len(paths)):
            cellC.append(lighten_color(color(i), 0.25))
        cellcol = np.empty(shape=(len(paths), 3)).tolist()
        for i in range(len(paths)):
            for j in range(3):
                cellcol[i][j] = cellC[i]
        c_lab = (r"${\rm Observation}$", r"$68\%$", r"$95\%$")
        the_table = plt.table(cellText=ct, colLabels=c_lab, cellLoc="center",
                              colWidths=[0.32, 0.1, 0.1], cellColours=cellcol, loc = 4)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)

    @staticmethod
    def plotChiCDF():
        pm = []; pm2 = []
        ii = np.linspace(0, 100, 5000)
        for jj in ii:
            pm.append(np.sqrt(2/np.pi)*np.exp(-pow(jj,2.)/2))
            pm2.append(math.fsum(pm))
        pm2 = pm2/sum(pm)
        plt.plot(ii, pm2, c="k")
