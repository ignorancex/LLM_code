import sys
import os
import astropy.units as u

# Path to COMPAS
compasRootDir = '/n/home04/aboesky/berger/COMPAS'

# Import COMPAS specific scripts
sys.path.append(compasRootDir + '/utils/CosmicIntegration/')

# Scripts I will need to write a pickle
sys.path.append('/n/home04/aboesky/berger/Exploring_Parameter_Space/Data_Analysis/Scripts')

from pickle_writers import pickle_rates_and_boostraps

def pickle_results(condensed_paths, alpha_vals, beta_vals, dco_type, weight_column=None,
                merges_hubble_time=True, pessimistic_CEE=True, no_RLOF_after_CEE=True,
                max_redshift=10.0, max_redshift_detection=1.0, redshift_step=0.001, z_first_SF = 10,
                use_sampled_mass_ranges=True, m1_min=5 * u.Msun, m1_max=150 * u.Msun, m2_min=0.1 * u.Msun, fbin=1.0,
                aSF = 0.01, bSF = 2.77, cSF = 2.90, dSF = 4.70,
                mu0=0.035, muz=-0.23, sigma0=0.39,sigmaz=0., alpha=0.0, 
                min_logZ=-12.0, max_logZ=1.0, step_logZ=0.01,
                sensitivity="O1", snr_threshold=8, 
                Mc_max=300.0, Mc_step=0.1, eta_max=0.25, eta_step=0.01,
                snr_max=1000.0, snr_step=0.1, lw=2, 
                lognormal=False, Zprescription='MZ_GSMF', SFRprescription='Madau et al. (2017)',        # ADAM'S FIDUCIAL MSSFR PARAMETERS
                GSMFprescription='Panter et al. (2004) Single', ZMprescription='Ma et al. (2016)',      # ADAM'S FIDUCIAL MSSFR PARAMETERS
                logNormalPrescription=None, n_iters=5, num_free_cpus=2):


    # Iterate through all alpha and beta values and write the results of each pair to a file
    for alpha_val in alpha_vals:
        for beta_val in beta_vals:
            
            # Make a folder to put the data into
            dco_type_folder = '/n/holystore01/LABS/berger_lab/Users/aboesky/Pickled_Rates/MSSFR_111/' + dco_type

            if not os.path.exists(dco_type_folder):
                os.mkdir(dco_type_folder)

            # Put that data into it
            pickle_rates_and_boostraps(condensed_paths[alpha_val][beta_val], dco_type_folder + '/alpha_CE_' + alpha_val + '_beta_' + beta_val + '.pkl', 
            dco_type=dco_type, merger_output_filename=None, weight_column=weight_column,
            merges_hubble_time=merges_hubble_time, pessimistic_CEE=pessimistic_CEE,
            no_RLOF_after_CEE=no_RLOF_after_CEE, max_redshift=max_redshift,
            max_redshift_detection=max_redshift_detection,
            redshift_step=redshift_step, z_first_SF=z_first_SF,
            m1_min=m1_min, m1_max=m1_max, m2_min=m2_min,
            fbin=fbin, aSF=aSF, bSF=bSF, cSF=cSF, dSF=dSF,
            mu0=mu0, muz=muz, sigma0=sigma0, alpha=alpha,
            min_logZ=min_logZ, max_logZ=max_logZ, step_logZ=step_logZ,
            sensitivity=sensitivity, snr_threshold=snr_threshold,
            Mc_max=Mc_max, Mc_step=Mc_step,
            eta_max=eta_max, eta_step=eta_step,
            snr_max=snr_max, snr_step=snr_step, 
            lognormal=lognormal, Zprescription=Zprescription, SFRprescription=SFRprescription,              # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS
            GSMFprescription=GSMFprescription, ZMprescription=ZMprescription,                               # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS
            logNormalPrescription=logNormalPrescription, n_iters=n_iters, num_free_cpus=num_free_cpus)      # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS

def main():

    # Declare grid values
    alpha_vals = ['0.1', '0.5', '2.0', '10.0'] # All the alpha values
    beta_vals = ['0.25', '0.5', '0.75'] # All the beta values

    condensed_paths = {} # 2D dictionary holding the paths to the output h5 files for which the first dimension is the alpha CE value and the second is the beta value

    # Iterate through the alpha-beta grid and fill in the corresponding paths to their outputs
    output_files_path = '/n/holystore01/LABS/berger_lab/Users/aboesky/output_alpha_CE_beta/'
    for alpha in alpha_vals:
        condensed_paths[alpha] = {}
        for beta in beta_vals:
            condensed_paths[alpha][beta] = output_files_path + 'output_alpha_CE_' + alpha + '_beta_' + beta + '/complete.h5'
    
    


    # DEFINE PARAMETERS
    # For what DCO would you like the rate?  options: all, BBH, BHNS BNS
    weight_column   = None
                            
    merges_hubble_time     = True
    pessimistic_CEE        = True
    no_RLOF_after_CEE      = True

    # Options for the redshift evolution 
    max_redshift           = 10.0
    max_redshift_detection = 2.0
    redshift_step          = 0.001
    z_first_SF             = 10

    # Metallicity of the Universe
    min_logZ               = -12.0 
    max_logZ               = 1.0 
    step_logZ              = 0.01

    #and detector sensitivity
    sensitivity            = "O1" 
    snr_threshold          = 8 

    Mc_max                 = 300.0 * (1 + max_redshift_detection)
    Mc_step                = 0.1 
    eta_max                = 0.25 
    eta_step               = 0.01
    snr_max                = 1000.0 
    snr_step               = 0.1

    # Parameters to calculate the representing SF mass (make sure these match YOUR simulation!)
    m1_min          = 5 * u.Msun 
    m1_max          = 150 * u.Msun
    m2_min          = m1_min * 0.01 # m2_min = min(q) * min(m1)
    fbin            = 1.0

    # Van Son 2022 prescriptions:
    mu0             = 0.025
    muz             = -0.048
    sigma0          = 1.125
    sigmaz          = 0.048
    alpha           = -1.77

    aSF             = 0.02
    bSF             = 1.48 
    cSF             = 4.45 
    dSF             = 5.9

    # MY FIDUCIAL MSSFR PRESCRIPTIONS
    lognormal=False                                     # !!! IF THIS PARAMETER IS FALSE, IT WILL USE THE NON-LOGNORMAL MSSFR PRESCRIPTIONS !!!
    Zprescription='MZ_GSMF'
    SFRprescription='Madau et al. (2017)'
    GSMFprescription='Panter et al. (2004) Single'
    ZMprescription='Ma et al. (2016)'
    logNormalPrescription=None


    # Write to Pickle!!!!
    pickle_results(condensed_paths, alpha_vals, beta_vals, dco_type='BNS', weight_column=weight_column,
                merges_hubble_time=merges_hubble_time, pessimistic_CEE=pessimistic_CEE,
                no_RLOF_after_CEE=no_RLOF_after_CEE, max_redshift=max_redshift,
                max_redshift_detection=max_redshift_detection,
                redshift_step=redshift_step, z_first_SF=z_first_SF,
                m1_min=m1_min, m1_max=m1_max, m2_min=m2_min,
                fbin=fbin, aSF=aSF, bSF=bSF, cSF=cSF, dSF=dSF,
                mu0=mu0, muz=muz, sigma0=sigma0, alpha=alpha,
                min_logZ=min_logZ, max_logZ=max_logZ, step_logZ=step_logZ,
                sensitivity=sensitivity, snr_threshold=snr_threshold,
                Mc_max=Mc_max, Mc_step=Mc_step,
                eta_max=eta_max, eta_step=eta_step,
                snr_max=snr_max, snr_step=snr_step, 
                lognormal=lognormal, Zprescription=Zprescription, SFRprescription=SFRprescription,      # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS
                GSMFprescription=GSMFprescription, ZMprescription=ZMprescription,                       # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS
                logNormalPrescription=logNormalPrescription, n_iters=100, num_free_cpus=2)

if __name__ == '__main__':
    main()