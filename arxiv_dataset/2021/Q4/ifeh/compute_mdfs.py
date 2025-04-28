import os
from scipy.stats import gaussian_kde
from ifeh_io import *
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

rootdir = "."

# Set environment for MDF computation:
#  ("inner_bulge" or "outer_bulge" or "transition" or "disk" or "sgr" or "halo" or "smc" or "lmc")
env = "inner_bulge"

# Compare model predictions with the Mullen et al. V-band photometric metallicities?
#   Note: data for doing this is only available for the inner and outer bulge.
compare_with_vfeh = True
inputfile_rrab_v = "o4rrab_bulge_v_gpr_dff_param.dat"

split_oo = False

# -----------------------------------------------------------------
# Select models to evaluate:

models = dict()

models.update({"RRc": ("model_mc1_poly1.4_RRc.sav", [0, 1, 2, 5], 1, "RRc", None)})
models.update({"RRab": ("model_mc1_poly1.3_RRab.sav", [0, 2, 5], 1, "RRab", None)})

# -----------------------------------------------------------------
# Select the plots to create:

plot_compare_mdf = "compare_mdf"
# plot_compare_mdf = None
plot_mdf_old_vs_new = "mdf_old_vs_"
# plot_mdf_old_vs_new = None
plot_old_vs_new = "smolec_vs_"
# plot_old_vs_new = None
plot_pred_vs_period = "ifeh_vs_period_"
# plot_pred_vs_period = None
plot_pred_vs_phi21 = "ifeh_vs_phi21"
# plot_pred_vs_phi21 = None
plot_pred_vs_phi31 = "ifeh_vs_phi31"
# plot_pred_vs_phi31 = None

plot_pred_vs_a1 = "ifeh_vs_a1"
# plot_pred_vs_a1 = None
plot_pred_vs_a2 = "ifeh_vs_a2"
# plot_pred_vs_a2 = None
plot_pred_vs_a3 = "ifeh_vs_a3"
# plot_pred_vs_a3 = None

# ----------------------------------------------------------------------------------------------------------------------


environments = ("inner_bulge", "outer_bulge", "transition", "disk", "sgr", "halo", "smc", "lmc")
assert env in environments, "env must be one of these: " + str(environments)

usecols_rrab = ['id', 'Nep', 'period', 'totamp', 'A1', 'A2', 'A3', 'phi21', 'phi31', 'phcov', 'costN', 'snr', 'FeH_S05',
                'FeH_S16J', 'FeH_S16N', 'meanmag', 'VmI']
usecols_rrc = ['id', 'Nep', 'period', 'totamp', 'A1', 'A2', 'A3', 'phi21', 'phi31', 'phcov', 'costN', 'snr',
               'meanmag', 'VmI']

if split_oo:
    usecols_rrab.append('oo')

if env == "inner_bulge":
    usecols_rrab.append('L')
    usecols_rrab.append('B')
    usecols_rrc.append('L')
    usecols_rrc.append('B')
    inputfile_rrab = "o4rrab_bulge+disk_gpr_param.dat"
    inputfile_rrc = "o4rrc_bulge+disk_gpr_param.dat"
    outputdir = "."
    subset_rrab = "sqrt(L**2+B**2)<10 and" \
                  "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and" \
                  "(" \
                  "(VmI != VmI and B<-5 and L<10 and L>0 and meanmag<17) or" \
                  "(VmI == VmI and (meanmag>1.1*(VmI)+13) and (meanmag<1.1*(VmI)+16) and (VmI>0.3)) or" \
                  "(VmI != VmI and not (B<-5 and L<10 and L>0))" \
                  ")"

    subset_rrc = "sqrt(L**2+B**2)<10 and" \
                 "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002) and" \
                 "(" \
                 "(VmI != VmI and B<-5 and L<10 and L>0 and meanmag<17) or" \
                 "(VmI == VmI and (meanmag>1.1*(VmI)+13.1) and (meanmag<1.1*(VmI)+16.1) and (VmI>0.2)) or" \
                 "(VmI != VmI and not (B<-5 and L<10 and L>0))" \
                 ")"
    plotlabel = "Inner bulge ($r < 10^\circ$)"

elif env == "outer_bulge":
    usecols_rrab.append('L')
    usecols_rrab.append('B')
    usecols_rrc.append('L')
    usecols_rrc.append('B')
    inputfile_rrab = "o4rrab_bulge+disk_gpr_param.dat"
    inputfile_rrc = "o4rrc_bulge+disk_gpr_param.dat"
    outputdir = "."
    subset_rrab = "sqrt(L**2+B**2)>10 and sqrt(L**2+B**2)<20 and" \
                  "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and" \
                  "(" \
                  "(VmI != VmI and B<-5 and L<10 and L>0 and meanmag<17) or" \
                  "(VmI == VmI and (meanmag>1.1*(VmI)+13) and (meanmag<1.1*(VmI)+16) and (VmI>0.3)) or" \
                  "(VmI != VmI and not (B<-5 and L<10 and L>0))" \
                  ")"

    subset_rrc = "sqrt(L**2+B**2)>10 and sqrt(L**2+B**2)<20 and" \
                 "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002) and" \
                 "(" \
                 "(VmI != VmI and B<-5 and L<10 and L>0 and meanmag<17) or" \
                 "(VmI == VmI and (meanmag>1.1*(VmI)+13.1) and (meanmag<1.1*(VmI)+16.1) and (VmI>0.2)) or" \
                 "(VmI != VmI and not (B<-5 and L<10 and L>0))" \
                 ")"
    plotlabel = "Outer bulge ($10^\circ < r < 20^\circ$)"

elif env == "transition":
    usecols_rrab.append('L')
    usecols_rrab.append('B')
    usecols_rrc.append('L')
    usecols_rrc.append('B')
    inputfile_rrab = "o4rrab_bulge+disk_gpr_param.dat"
    inputfile_rrc = "o4rrc_bulge+disk_gpr_param.dat"
    outputdir = "."
    subset_rrab = "sqrt(L**2+B**2)>20 and sqrt(L**2+B**2)<30 and" \
                  "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1)"

    subset_rrc = "sqrt(L**2+B**2)>20 and sqrt(L**2+B**2)<30 and" \
                 "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002)"

    plotlabel = "Transition region ($20^\circ < r < 30^\circ$)"   # 20 deg < r < 30 deg

elif env == "disk":
    usecols_rrab.append('L')
    usecols_rrab.append('B')
    usecols_rrc.append('L')
    usecols_rrc.append('B')
    inputfile_rrab = "o4rrab_bulge+disk_gpr_param.dat"
    inputfile_rrc = "o4rrc_bulge+disk_gpr_param.dat"
    outputdir = "."
    subset_rrab = "sqrt(L**2+B**2)>30 and" \
                  "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1)"

    subset_rrc = "sqrt(L**2+B**2)>30 and" \
                 "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002)"

    plotlabel = "Disk region ($30^\circ < r$)"   # 30 deg < r

elif env == "sgr":
    usecols_rrab.append('L')
    usecols_rrab.append('B')
    usecols_rrc.append('L')
    usecols_rrc.append('B')
    inputfile_rrab = "o4rrab_bulge+disk_gpr_param.dat"
    inputfile_rrc = "o4rrc_bulge+disk_gpr_param.dat"
    outputdir = "."
    subset_rrab = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and" \
                  "(B<-5 and L<10 and L>0) and" \
                  "(" \
                  "(VmI != VmI and meanmag>17) or" \
                  "(VmI == VmI and meanmag>1.1*(VmI)+16 and VmI>0.3)" \
                  ")"

    subset_rrc = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002) and" \
                 "(B<-5 and L<10 and L>0) and" \
                 "(" \
                 "(VmI != VmI and meanmag>17) or" \
                 "(VmI == VmI and meanmag>1.1*(VmI)+16.1 and VmI>0.2)" \
                 ")"
    plotlabel = "Sgr dSph"

elif env == "halo":
    inputfile_rrab = "o4rrab_lmc+smc_gpr_param.dat"
    inputfile_rrc = "o4rrc_lmc+smc_gpr_param.dat"
    outputdir = "."
    subset_rrab = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and meanmag<17.5"
    subset_rrc = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and meanmag<17.5"
    plotlabel = "Halo"

elif env == "smc":
    inputfile_rrab = "o4rrab_smc_gpr_param.dat"
    inputfile_rrc = "o4rrc_smc_gpr_param.dat"
    outputdir = "."
    subset_rrab = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and meanmag>18.5"
    subset_rrc = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and meanmag>18.6"
    plotlabel = "SMC"

elif env == "lmc":
    inputfile_rrab = "o4rrab_lmc_gpr_param.dat"
    inputfile_rrc = "o4rrc_lmc_gpr_param.dat"
    outputdir = "."
    subset_rrab = "(phcov>0.9 and snr>300 and Nep>100 and costN<0.002 and totamp<1.1) and meanmag>18"
    subset_rrc = "(phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.1) and meanmag>18.1"
    plotlabel = "LMC"

# ----------------------------------------------------------------------------------------------------------------------

kdes = list()
kdes_oo1 = list()
kdes_oo2 = list()
preds = list()
model_names = list()

feh_s05_kde = None
feh_s16j_kde = None
feh_s16n_kde = None

feh_pred_oo1_kde = None
feh_pred_oo2_kde = None

t = np.linspace(-3, 0, 1000)

for model_name, model_pars in models.items():

    model_file = model_pars[0]
    feature_indx = model_pars[1]
    n_poly = model_pars[2]
    rrtype = model_pars[3]
    pca_model = model_pars[4]

    print("RRab subset, model: ".format(model_name))

    X, feh_s05, _, df, feature_names, df_orig = \
        load_dataset(os.path.join(rootdir, inputfile_rrab),
                     trim_quantiles=None, n_poly=n_poly, plothist=False,
                     usecols=usecols_rrab,
                     input_feature_names=['period', 'A1', 'A2', 'A3', 'phi21', 'phi31'],
                     output_feature_indices=feature_indx, pca_model=pca_model,
                     y_col='FeH_S05', yerr_col=None, subset_expr=subset_rrab)

    feh_s16j = df['FeH_S16J'].to_numpy()    # Skowron et al. (2016) metallicity estimates using JK96
    feh_s16n = df['FeH_S16N'].to_numpy()    # Skowron et al. (2016) metallicity estimates using N13

    if rrtype == "RRc":
        X, _, _, df, feature_names, df_orig = \
            load_dataset(os.path.join(rootdir, inputfile_rrc),
                         trim_quantiles=None, n_poly=n_poly, plothist=False,
                         usecols=usecols_rrc,
                         input_feature_names=['period', 'A1', 'A2', 'A3', 'phi21', 'phi31'],
                         output_feature_indices=feature_indx, pca_model=pca_model,
                         y_col=None, yerr_col=None, subset_expr=subset_rrc)

    print('Input features:')
    print(feature_names)
    n_features = len(feature_names)

    loaded_model = joblib.load(model_file)

    ndata = X.shape[0]

    model_name = model_name + " (" + str(ndata) + ")"

    feh_pred = loaded_model.predict(X)

    if rrtype == "RRab" and plot_old_vs_new is not None:
        # Plot pred vs Smolec(pyrime) prediction:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('[Fe/H] (pyrime)')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(feh_s05, feh_pred, 'k.')
        xx = np.linspace(np.min([np.min(feh_s05), np.min(feh_pred)]),
                         np.max([np.max(feh_s05), np.max(feh_pred)]), 100)
        plt.plot(xx, xx, 'r-')
        plt.xlim(-2.5, 0)
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_old_vs_new+model_name+'.png'))
        plt.close(fig)

    if split_oo and rrtype == "RRab":
        oo1_mask = (df['oo'] == 1)
        oo2_mask = (df['oo'] == 2)

    if feh_s05_kde is None:
        kde = gaussian_kde(feh_s05)
        feh_s05_kde = kde(t)

    if feh_s16j_kde is None:
        kde = gaussian_kde(feh_s16j)
        feh_s16j_kde = kde(t)

    if feh_s16n_kde is None:
        kde = gaussian_kde(feh_s16n)
        feh_s16n_kde = kde(t)

    kde = gaussian_kde(feh_pred)
    feh_pred_kde = kde(t)

    if split_oo and rrtype == "RRab":
        kde = gaussian_kde(feh_pred[oo1_mask])
        feh_pred_oo1_kde = kde(t)
        kdes_oo1.append(feh_pred_oo1_kde)

        kde = gaussian_kde(feh_pred[oo2_mask])
        feh_pred_oo2_kde = kde(t)
        kdes_oo2.append(feh_pred_oo2_kde)
    else:
        kdes_oo1.append(None)
        kdes_oo2.append(None)

    # if we use an RRab model, determine the position of the KDE's mode:
    if "RRab" in model_name:
        maxind = np.argmax(feh_pred_kde)
        mode = t[maxind]

    kdes.append(feh_pred_kde)
    preds.append(feh_pred)
    model_names.append(model_name)

    if plot_mdf_old_vs_new is not None:
        fig = plt.figure(figsize=(5, 4))
        fig.subplots_adjust(bottom=0.13, top=0.85, hspace=0, left=0.1, right=0.85, wspace=0)
        plt.hist(feh_s05, facecolor='black', alpha=0.4, bins='sqrt', density=True, label='[Fe/H] (pyrime)')
        plt.hist(feh_pred, facecolor='red', alpha=0.4, bins='sqrt', density=True, label='[Fe/H] ('+model_name+')')
        plt.plot(t, feh_s05_kde, 'k--', label='KDE (S05)')
        plt.plot(t, feh_s16j_kde, 'k:', label='KDE (S16J)')
        plt.plot(t, feh_s16n_kde, 'k-.', label='KDE (S16N)')
        plt.plot(t, feh_pred_kde, 'r-', label='KDE ('+model_name+')')
        plt.xlabel("[Fe/H]")
        plt.xlim(-2.6, 0.1)
        plt.ylabel('norm. density')
        plt.tick_params(direction='in')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
        plt.savefig(os.path.join(outputdir, env + "_" + plot_mdf_old_vs_new+model_name+'.png'))
        plt.close(fig)

    # Plot pred vs period:
    if plot_pred_vs_period is not None:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('period')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(df['period'], feh_pred, 'k,')
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_pred_vs_period+model_name+'.png'))
        plt.close(fig)

    # Plot pred vs phi21:
    if plot_pred_vs_phi21 is not None:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('$\phi_{21}$')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(df['phi21'], feh_pred, 'k,')
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_pred_vs_phi21+model_name+'.png'))
        plt.close(fig)

    # Plot pred vs phi31:
    if plot_pred_vs_phi31 is not None:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('$\phi_{31}$')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(df['phi31'], feh_pred, 'k,')
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_pred_vs_phi31+model_name+'.png'))
        plt.close(fig)

    # Plot pred vs A1:
    if plot_pred_vs_a1 is not None:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('$A_1$')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(df['A1'], feh_pred, 'k,')
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_pred_vs_a1+model_name+'.png'))
        plt.close(fig)

    # Plot pred vs A2:
    if plot_pred_vs_a2 is not None:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('$A_2$')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(df['A2'], feh_pred, 'k,')
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_pred_vs_a2+model_name+'.png'))
        plt.close(fig)

    # Plot pred vs A3:
    if plot_pred_vs_a3 is not None:
        fig = plt.figure(figsize=(5, 4))
        plt.xlabel('$A_3$')
        plt.ylabel('[Fe/H] ('+model_name+')')
        plt.plot(df['A3'], feh_pred, 'k,')
        plt.ylim(-2.5, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, env + "_" + plot_pred_vs_a3+model_name+'.png'))
        plt.close(fig)

# Plot comparing the MDFs from different models for a given stellar population:
if plot_compare_mdf is not None:

    if env in ["inner_bulge", "outer_bulge"] and compare_with_vfeh:

        Xv, _, _, dfv, feature_names_v, df_orig_v = \
            load_dataset(os.path.join(rootdir, inputfile_rrab_v),
                         usecols=['period', 'phi31', 'phcov', 'snr', 'Nep', 'costN', 'totamp'],
                         input_feature_names=['period', 'phi31'],
                         subset_expr='phcov>0.9 and snr>100 and Nep>100 and costN<0.002 and totamp<1.6')

        nvdata = len(dfv)
        print("{} objects with V data.".format(nvdata))
        # The Mullen et al (2021) V-band [Fe/H] formula:
        feh_v_m21 = -1.22 - 7.60 * (dfv['period'] - 0.58) + 1.42 * (dfv['phi31'] - 5.25)
        kde_v_m21 = gaussian_kde(feh_v_m21)
        feh_v_m21_kde = kde_v_m21(t)

        feh_v_jk96 = -5.038 - 5.394 * dfv['period'] + 1.345 * dfv['phi31']
        kde_v_jk96 = gaussian_kde(feh_v_jk96)
        feh_v_jk96_kde = kde_v_jk96(t)

    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(bottom=0.13, top=0.85, hspace=0, left=0.1, right=0.95, wspace=0)
    for i, model_name in enumerate(model_names):
        p = plt.plot(t, kdes[i], '-', label=model_name)
        if split_oo and "RRab" in model_name:
            plt.plot(t, kdes_oo1[i], '--', color=p[0].get_color())
            plt.plot(t, kdes_oo2[i], ':', color=p[0].get_color())
    plt.plot(t, feh_s05_kde, 'k--', label='S05')
    plt.plot(t, feh_s16j_kde, 'k:', label='S16J')
    plt.plot(t, feh_s16n_kde, 'k-.', label='S16N')

    if env in ["inner_bulge", "outer_bulge"] and compare_with_vfeh:
        plt.plot(t, feh_v_m21_kde, '--', c="red", label='M21 ({})'.format(nvdata))
        plt.plot(t, feh_v_jk96_kde, '--', c="purple", label='JK96 ({})'.format(nvdata))

    plt.xlabel("[Fe/H]")
    plt.xlim(-3.0, 0)
    plt.ylabel('norm. density')
    plt.tick_params(direction='in')
    plt.grid(which='both', axis='x', alpha=0.7, ls=':')
    ax = plt.gca()
    plt.text(0.05, 0.9, plotlabel, transform=ax.transAxes)
    plt.text(0.9, 0.9, "{:.2f}".format(mode), transform=ax.transAxes)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    plt.savefig(os.path.join(outputdir, env + "_" + plot_compare_mdf + ".pdf"), format="pdf")
    plt.tight_layout()
    plt.close(fig)
