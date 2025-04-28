import astropy.io.ascii as io 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from astropy import constants as const
from astropy import units as u
import os
import blueshift_util
import matplotlib.patheffects as pe


def make_figure(alpha = "1"):
    print ("Making figure 10...", end="")

    data_dir  = "{}/steppar_specs_alpha{}/".format(blueshift_util.g_DataDir, alpha)

    plt.figure()
    iplot = 0

    par_strings = ["diskmax", "omega", "alpha", "rv", "mdotw", "vinf"]

    if alpha == "0.5":
        root = "run47"
    else:
        root = "run128"

    fnames_all = [
        [".spec","_diskmax_1e18.spec","_diskmax_3e17.spec","_diskmax_1e17.spec","_diskmax_3e16.spec"],
        ["_omega0p1.spec",".spec","_omega0p3.spec","_omega0p4.spec","_omega0p5.spec"],
        ["_alpha0.5.spec","_alpha0.75.spec","_alpha1.spec","_alpha1.25.spec","_alpha1.5.spec"],
        [".spec","_rv_3e18.spec","_rv_1e18.spec","_rv_3e17.spec","_rv_1e17.spec"],
        ["_mdotw0.1.spec","_mdotw0.5.spec","_mdotw1.0.spec",".spec","_mdotw10.spec"],
        [".spec","_vinf1.5.spec","_vinf2.0.spec","_vinf2.5.spec","_vinf3.0.spec"]]


    titles = {
        "diskmax": "Obscuring Disc Size", 
        "omega": "Covering Factor",
        "alpha": "Acceleration Exponent",
        "rv": "Acceleration Length",
        "mdotw": "Mass-loss Rate",
        "vinf": "Terminal Velocity"
    }

    symbols = {
        "diskmax": r"R_{{\rm mid}}", 
        "omega": r"\Omega/4\pi",
        "alpha": r"\alpha",
        "rv": r"R_v",
        "mdotw": r"\dot{M}_w~(M_\odot~{\rm yr}^{-1})",
        "vinf": "f_\infty"
    }

    values = {
        "diskmax": [r"R_{\rm max}", r"10^{{18}}~{\rm cm}",r"3\times 10^{{17}}~{\rm cm}",r"10^{{17}}~{\rm cm}",r"3\times 10^{{16}}~{\rm cm}"], 
        "omega": [0.1,0.2,0.3,0.4,0.5],
        "alpha": [0.5,0.75,1,1.25,1.5],
        "rv": [r"10^{{19}}~{\rm cm}",r"3\times 10^{{18}}~{\rm cm}",r"10^{{18}}~{\rm cm}",r"3\times 10^{{17}}~{\rm cm}",r"10^{{17}}~{\rm cm}"],
        "mdotw": [0.1,0.5,1,5,10],
        "vinf": [1,1.5,2,2.5,3]
    }

    blueshift_util.set_cmap_cycler("viridis_r", 5)

    plt.figure(figsize=(10,7))
    for itb, tb in enumerate(par_strings):


        # norm, mappable = blueshift_util.get_norm_mappable(cmap=cmap_names[tb])

        #fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and tb in f]
        fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and tb in f]
        fnames = fnames_all[itb]
        #print (fnames)
        plt.subplot(2,3,iplot+1)
        for j,fsuffix in enumerate(fnames):

            #try:
            f = root + fsuffix
            if f == "run128_vinf1.5.spec":
                f = "run128_vinf1.5_noup.spec"

            s = io.read("{}/{}".format(data_dir, f))

            wavelength = s["Lambda"]
            velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs 
            velocity = velocity.to(u.km/u.s)

            fmax = 0

            #for i in range(len(angles)):
            angle = blueshift_util.fiducial_angle
            colname = "A{:02d}P0.50".format(int(angle))

            fcont = blueshift_util.fit_continuum(wavelength, s[colname])
            try:
                flux = savgol_filter(s[colname]/fcont, 15, 3)
            except: 
                flux = s[colname]/fcont
                print (f)

            #plot_label = "${}={}$".format(symbols[tb], values[tb][j])
            plot_label = "${}$".format(values[tb][j])
            if f == "{}.spec".format(root):
                path_effects =[pe.Stroke(linewidth=6, foreground='#7f7f7f'), pe.Normal()]
            else:
                path_effects = None
            plt.plot(velocity/1e3, flux, lw=3, label = plot_label, path_effects=path_effects)
            plt.xlim(-8,8)
            plt.ylim(0.8, 5)

            # if iplot == 12:
            if iplot == 0 or iplot == 3:
                plt.ylabel(r"Normalised $F_\lambda$", fontsize=20)
            elif iplot == 4:
                plt.xlabel("Velocity~($10^3$~km~s$^{-1}$)", fontsize=20)
                plt.gca().set_yticklabels([])
                # plt.gca().set_yticklabels([])
            else:
                plt.gca().set_yticklabels([])

            if iplot < 3:
                plt.gca().set_xticklabels([])
            # 	plt.gca().set_xticklabels([])
                # plt.gca().set_yticklabels([])

            if tb == "omega":
                plt.title(r"${}$ for fixed $\theta_{{\rm min}}$".format(symbols[tb]), fontsize=16)
            else:
                plt.title(r"{}, ${}$".format(titles[tb],symbols[tb]), fontsize=16)

            #plt.grid(ls=":")
            plt.gca().axvline(0, ls="--", alpha=0.5, color="k")
        plt.legend(frameon=True, labelspacing=0.25, borderpad=0.25, loc="upper right")
        
        iplot+=1
        
            
    plt.tight_layout(pad=0.05, w_pad=0.1)
    plt.subplots_adjust(wspace=0.05)
    blueshift_util.save_paper_figure("fig10.pdf")
    print ("Done.")

if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    make_figure()