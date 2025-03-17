import sys
import os

# Append paths to sys
compasRootDir   = '/Users/adamboesky/Research/PRISE/COMPAS'
sys.path.append(compasRootDir + '/utils/CosmicIntegration/')
clusterCompasRootDir   = '/n/home04/aboesky/pgk/COMPAS'
sys.path.append(clusterCompasRootDir + '/utils/CosmicIntegration/')


import pickle
import numpy as np
import astropy.units as u
import FastCosmicIntegration as FCI
import h5py as h5
import formation_channels
import ClassCOMPAS

from multiprocessing import Pool
from KDEpy import FFTKDE  # Fastest 1D algorithm




# def get_bootstraps(COMPAS, n_iters, calculation_kwargs, Mc_bins, num_free_cpus=2):
def get_bootstraps(output_file_path, COMPAS, n_iters, calculation_kwargs, Mc_bins, num_free_cpus=2):

    # Get the seeds for the DCO type
    dco_seeds = COMPAS.seedsDCO

    # Get the number of binaries of the DCO type in the data
    num_binaries = len(dco_seeds)

    # Declare the object to get results from
    results_obj = []
    # cumulative_detection_rates  = []
    # total_formation_rates       = []
    # total_merger_rates          = []
    # mass_kde                    = []
    rates_and_bootstraps = h5.File(output_file_path, 'a')
    bootstraps = rates_and_bootstraps['Bootstraps']
    
    print('getting bootstraps')
    # Declare a pool for multiprocessing
    with Pool(processes=2) as pool: # os.cpu_count() - num_free_cpus) as pool:

        # Iterate through different bootstrap iterations
        for _ in range(n_iters):
            print('Iterating over the declared!')

            # Make a mask that randomly draws from the data for the DCOs that you have
            mask = np.random.randint(low=0, high=num_binaries, size=num_binaries)

            # Assign the task and append the result to the results object
            result = pool.apply_async(FCI.find_sampled_detection_rates, (COMPAS.path, mask, Mc_bins), calculation_kwargs)
            results_obj.append(result)
        
        # Get the results
        for index, result in enumerate(results_obj):
            vals = result.get()
            bootstraps['cumulative_detection_rates'][index] = vals[0]
            bootstraps['total_formation_rates'][index] = vals[1]
            bootstraps['total_merger_rates'][index] = vals[2]
            bootstraps['mass_kde'][index] = vals[3]
            # vals = result.get()
            # cumulative_detection_rates.append(vals[0])
            # total_formation_rates.append(vals[1])
            # total_merger_rates.append(vals[2])
            # mass_kde.append(vals[3])
            
            del vals
            del result

    # # Put all of the data into one dictionary to store
    # dict = {}
    # dict['cumulative_detection_rates']  = np.array(cumulative_detection_rates)
    # dict['total_formation_rates']       = np.array(total_formation_rates)
    # dict['total_merger_rates']          = np.array(total_merger_rates)
    # dict['mass_kde']                    = np.array(mass_kde)

    # return dict

def pickle_weights_for_all_redshfits(path, output_path, dco_type=None, merger_output_filename=None, weight_column=None,
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
    logNormalPrescription=None, n_iters=100, num_free_cpus=2):                              # ADAM'S FIDUCIAL MSSFR PARAMETERS
    
    # Make sure the dco_type is given
    assert dco_type != None, 'dco_type must be given'

    bootstrap_kwargs = {'dco_type': dco_type, 'merger_output_filename': merger_output_filename, 'weight_column': weight_column, 'merges_hubble_time': merges_hubble_time, 'pessimistic_CEE': pessimistic_CEE, 
                        'no_RLOF_after_CEE': no_RLOF_after_CEE, 'max_redshift': max_redshift, 'max_redshift_detection': max_redshift_detection, 'redshift_step': redshift_step, 'z_first_SF': z_first_SF,
                        'use_sampled_mass_ranges': use_sampled_mass_ranges, 'm1_min': m1_min, 'm1_max': m1_max, 'm2_min': m2_min, 'fbin': fbin,
                        'aSF': aSF, 'bSF': bSF, 'cSF': cSF, 'dSF': dSF,
                        'mu0': mu0, 'muz': muz, 'sigma0': sigma0, 'sigmaz': sigmaz, 'alpha': alpha, 
                        'min_logZ': min_logZ, 'max_logZ': max_logZ, 'step_logZ': step_logZ,
                        'sensitivity': sensitivity, 'snr_threshold': snr_threshold, 
                        'Mc_max': Mc_max, 'Mc_step': Mc_step, 'eta_max': eta_max, 'eta_step': eta_step,
                        'snr_max': snr_max, 'snr_step': snr_step,
                        'lognormal': lognormal, 'Zprescription': Zprescription, 'SFRprescription': SFRprescription,
                        'GSMFprescription': GSMFprescription, 'ZMprescription': ZMprescription,
                        'logNormalPrescription': logNormalPrescription}

    # Declare an h5 file to put everything in
    final_filename = 'all_rates_weights_at_redshifts.h5'
    rates_and_bootstraps = h5.File(output_path + '/' + final_filename, 'w')

    print('Finding run rates')
    _, formation_rates, merger_rates, redshifts, COMPAS = \
        FCI.find_detection_rate(path, dco_type=dco_type, weight_column=weight_column,
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
        logNormalPrescription=logNormalPrescription)                                            # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS

    actual = rates_and_bootstraps.create_group('actual')             # The group for the rates calculated with the actual runs

    # sum things up across binaries
    actual.create_dataset('redshifts', data=redshifts)
    actual.create_dataset('total_formation_rates', data=formation_rates)
    actual.create_dataset('total_merger_rates', data=merger_rates)

    # empty trash
    del formation_rates
    del merger_rates

    # Delete trash object
    del COMPAS

    rates_and_bootstraps.close()


def pickle_rates_and_boostraps_for_desired_redshfits(path, output_path, dco_type=None, merger_output_filename=None, weight_column=None,
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
    logNormalPrescription=None, n_iters=100, num_free_cpus=2):                              # ADAM'S FIDUCIAL MSSFR PARAMETERS
    
    # Make sure the dco_type is given
    assert dco_type != None, 'dco_type must be given'

    bootstrap_kwargs = {'dco_type': dco_type, 'merger_output_filename': merger_output_filename, 'weight_column': weight_column, 'merges_hubble_time': merges_hubble_time, 'pessimistic_CEE': pessimistic_CEE, 
                        'no_RLOF_after_CEE': no_RLOF_after_CEE, 'max_redshift': max_redshift, 'max_redshift_detection': max_redshift_detection, 'redshift_step': redshift_step, 'z_first_SF': z_first_SF,
                        'use_sampled_mass_ranges': use_sampled_mass_ranges, 'm1_min': m1_min, 'm1_max': m1_max, 'm2_min': m2_min, 'fbin': fbin,
                        'aSF': aSF, 'bSF': bSF, 'cSF': cSF, 'dSF': dSF,
                        'mu0': mu0, 'muz': muz, 'sigma0': sigma0, 'sigmaz': sigmaz, 'alpha': alpha, 
                        'min_logZ': min_logZ, 'max_logZ': max_logZ, 'step_logZ': step_logZ,
                        'sensitivity': sensitivity, 'snr_threshold': snr_threshold, 
                        'Mc_max': Mc_max, 'Mc_step': Mc_step, 'eta_max': eta_max, 'eta_step': eta_step,
                        'snr_max': snr_max, 'snr_step': snr_step,
                        'lognormal': lognormal, 'Zprescription': Zprescription, 'SFRprescription': SFRprescription,
                        'GSMFprescription': GSMFprescription, 'ZMprescription': ZMprescription,
                        'logNormalPrescription': logNormalPrescription}

    # Declare an h5 file to put everything in
    final_filename = 'rates_weights_at_redshifts.h5'
    rates_and_bootstraps = h5.File(output_path + '/' + final_filename, 'w')
    desired_redshifts = np.array([0.2, 2.0, 6.0])

    print('Finding run rates')
    formation_rates, merger_rates, redshifts, COMPAS = \
        FCI.find_rate_weights_at_redshifts(path, desired_redshifts=desired_redshifts, dco_type=dco_type, weight_column=weight_column,
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
        logNormalPrescription=logNormalPrescription)                                            # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS

    for desired_index, desired_z in enumerate(desired_redshifts):
        actual = rates_and_bootstraps.create_group('redshift_'+str(desired_z))             # The group for the rates calculated with the actual runs

        # sum things up across binaries
        actual.create_dataset('redshifts', data=redshifts)
        actual.create_dataset('total_formation_rates', data=formation_rates[desired_index])
        actual.create_dataset('total_merger_rates', data=merger_rates[desired_index])

    # empty trash
    del formation_rates
    del merger_rates

    # Delete trash object
    del COMPAS

    rates_and_bootstraps.close()


def pickle_rates_and_boostraps_by_metallicity(path, output_path, dco_type=None, merger_output_filename=None, weight_column=None,
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
    logNormalPrescription=None, n_iters=100, num_free_cpus=2):                              # ADAM'S FIDUCIAL MSSFR PARAMETERS
    
    # Make sure the dco_type is given
    assert dco_type != None, 'dco_type must be given'

    bootstrap_kwargs = {'dco_type': dco_type, 'merger_output_filename': merger_output_filename, 'weight_column': weight_column, 'merges_hubble_time': merges_hubble_time, 'pessimistic_CEE': pessimistic_CEE, 
                        'no_RLOF_after_CEE': no_RLOF_after_CEE, 'max_redshift': max_redshift, 'max_redshift_detection': max_redshift_detection, 'redshift_step': redshift_step, 'z_first_SF': z_first_SF,
                        'use_sampled_mass_ranges': use_sampled_mass_ranges, 'm1_min': m1_min, 'm1_max': m1_max, 'm2_min': m2_min, 'fbin': fbin,
                        'aSF': aSF, 'bSF': bSF, 'cSF': cSF, 'dSF': dSF,
                        'mu0': mu0, 'muz': muz, 'sigma0': sigma0, 'sigmaz': sigmaz, 'alpha': alpha, 
                        'min_logZ': min_logZ, 'max_logZ': max_logZ, 'step_logZ': step_logZ,
                        'sensitivity': sensitivity, 'snr_threshold': snr_threshold, 
                        'Mc_max': Mc_max, 'Mc_step': Mc_step, 'eta_max': eta_max, 'eta_step': eta_step,
                        'snr_max': snr_max, 'snr_step': snr_step,
                        'lognormal': lognormal, 'Zprescription': Zprescription, 'SFRprescription': SFRprescription,
                        'GSMFprescription': GSMFprescription, 'ZMprescription': ZMprescription,
                        'logNormalPrescription': logNormalPrescription}

    # Declare an h5 file to put everything in
    final_filename = 'rates_and_bootstraps_by_Z.h5'
    rates_and_bootstraps = h5.File(output_path + '/' + final_filename, 'w')
    Z_solar = 0.0142
    Z_bins = [[-1, Z_solar/10], [Z_solar/10, Z_solar/5], [Z_solar/5, Z_solar/2], [Z_solar/2, Z_solar], [Z_solar, 100]]

    for bin in Z_bins:
        print('Finding run rates')
        detection_rates, formation_rates, merger_rates, redshifts, COMPAS = \
            FCI.find_detection_rate_by_metallicity(path, bin_edges=bin, dco_type=dco_type, weight_column=weight_column,
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
            logNormalPrescription=logNormalPrescription)                                            # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS

        actual = rates_and_bootstraps.create_group('Z_in_'+str(bin[0])+'_'+str(bin[1]))             # The group for the rates calculated with the actual runs

        # sum things up across binaries
        if COMPAS:
            actual.create_dataset('redshifts', data=redshifts)
            actual.create_dataset('total_formation_rates', data=np.sum(formation_rates, axis=0))
            actual.create_dataset('total_merger_rates', data=np.sum(merger_rates, axis=0))
            total_detection_rates = np.sum(detection_rates, axis=0)

            # and across redshifts
            actual.create_dataset('cumulative_detection_rates', data=np.cumsum(total_detection_rates))
            detection_rates_by_binary = np.sum(detection_rates, axis=1)

            # MASS KDE STUFF:
            # Mak masks to filer for the channel only
            all_seeds, Zs = COMPAS.get_COMPAS_variables("BSE_System_Parameters", ["SEED", "Metallicity@ZAMS(1)"])
            dco_seeds = COMPAS.get_COMPAS_variables("BSE_Double_Compact_Objects",["SEED"])
            dco_mask = np.isin(all_seeds, dco_seeds)

            # A mask for all the binaries in the metallicity
            Z_mask = np.logical_and(Zs[dco_mask][COMPAS.DCOmask] >= bin[0], Zs[dco_mask][COMPAS.DCOmask] < bin[1])

            # Declare bins for the chirp mass KDE
            chirp_masses = (COMPAS.mass1[Z_mask]*COMPAS.mass2[Z_mask])**(3./5.) / (COMPAS.mass1[Z_mask] + COMPAS.mass2[Z_mask])**(1./5.)
            actual.create_dataset('chirp_masses', data=chirp_masses) # Scale the KDE

            # empty trash
            del detection_rates
            del formation_rates
            del merger_rates
            del chirp_masses
            del detection_rates_by_binary

            # Delete trash object
            del COMPAS

    rates_and_bootstraps.close()


def pickle_rates_and_boostraps_by_formation_channel(path, output_path, dco_type=None, merger_output_filename=None, weight_column=None,
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
    logNormalPrescription=None, n_iters=100, num_free_cpus=2):                              # ADAM'S FIDUCIAL MSSFR PARAMETERS
    
    # Make sure the dco_type is given
    assert dco_type != None, 'dco_type must be given'

    bootstrap_kwargs = {'dco_type': dco_type, 'merger_output_filename': merger_output_filename, 'weight_column': weight_column, 'merges_hubble_time': merges_hubble_time, 'pessimistic_CEE': pessimistic_CEE, 
                        'no_RLOF_after_CEE': no_RLOF_after_CEE, 'max_redshift': max_redshift, 'max_redshift_detection': max_redshift_detection, 'redshift_step': redshift_step, 'z_first_SF': z_first_SF,
                        'use_sampled_mass_ranges': use_sampled_mass_ranges, 'm1_min': m1_min, 'm1_max': m1_max, 'm2_min': m2_min, 'fbin': fbin,
                        'aSF': aSF, 'bSF': bSF, 'cSF': cSF, 'dSF': dSF,
                        'mu0': mu0, 'muz': muz, 'sigma0': sigma0, 'sigmaz': sigmaz, 'alpha': alpha, 
                        'min_logZ': min_logZ, 'max_logZ': max_logZ, 'step_logZ': step_logZ,
                        'sensitivity': sensitivity, 'snr_threshold': snr_threshold, 
                        'Mc_max': Mc_max, 'Mc_step': Mc_step, 'eta_max': eta_max, 'eta_step': eta_step,
                        'snr_max': snr_max, 'snr_step': snr_step,
                        'lognormal': lognormal, 'Zprescription': Zprescription, 'SFRprescription': SFRprescription,
                        'GSMFprescription': GSMFprescription, 'ZMprescription': ZMprescription,
                        'logNormalPrescription': logNormalPrescription}

    # Declare an h5 file to put everything in
    final_filename = 'rates_and_bootstraps_by_formation_channel.h5'
    rates_and_bootstraps = h5.File(output_path + '/' + final_filename, 'w')

    # The channel types and colors
    channel_types = {1: '(I) Classic', 2: '(II) Only stable', 3: '(III) Single core CE at first mass transfer', 4: '(IV) Double Core CE at first mass transfer', 0: '(V) Other'}

    # Identify the formation channels of each DCO
    print("Identifying formation channels for all DCOs")
    ch_COMPAS = ClassCOMPAS.COMPASData(path=path)
    ch_COMPAS.Mlower = 5
    ch_COMPAS.Mupper = 150
    ch_COMPAS.binaryFraction = 1.0
    ch_COMPAS.setGridAndMassEvolved()
    ch_COMPAS.Amin = 0.01
    ch_COMPAS.Mupper = 1000
    ch_COMPAS.setCOMPASDCOmask(types=dco_type, withinHubbleTime=True)
    ch_COMPAS.setCOMPASData()
    all_channels = formation_channels.identify_formation_channels(ch_COMPAS.seedsDCO, h5.File(path))

    # Get the channel numbers from the keys of channel_types
    keys = channel_types.copy().keys()
    
    # # Delete the channel_type of any channel that has a count of zero
    # for ch_num in keys:
    #     if np.count_nonzero(np.array(channel)==ch_num) == 0:
    #         del channel_types[ch_num]

    
    for ch_index, channel in channel_types.items():
        print('Channel: ', channel)

        # Make sure that there are any DCOs in the channel
        if np.count_nonzero(np.array(all_channels)==ch_index) != 0:
            print('Not zero!')

            # Calculate rates
            print('Finding run rates')
            detection_rates, formation_rates, merger_rates, redshifts, COMPAS = \
                FCI.find_detection_rate_by_channel(
                path, ch_index, dco_type=dco_type, weight_column=weight_column,
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
                logNormalPrescription=logNormalPrescription)                                            # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS

            actual = rates_and_bootstraps.create_group(channel)             # The group for the rates calculated with the actual runs

            # sum things up across binaries
            if COMPAS:
                actual.create_dataset('redshifts', data=redshifts)
                actual.create_dataset('total_formation_rates', data=np.sum(formation_rates, axis=0))
                actual.create_dataset('total_merger_rates', data=np.sum(merger_rates, axis=0))
                total_detection_rates = np.sum(detection_rates, axis=0)

                # and across redshifts
                actual.create_dataset('cumulative_detection_rates', data=np.cumsum(total_detection_rates))
                detection_rates_by_binary = np.sum(detection_rates, axis=1)

                # MASS KDE STUFF:
                # Mak masks to filer for the channel only
                all_seeds = COMPAS.get_COMPAS_variables("BSE_System_Parameters", ["SEED"])
                dco_seeds = COMPAS.get_COMPAS_variables("BSE_Double_Compact_Objects",["SEED"])
                dco_mask = np.isin(all_seeds, dco_seeds)

                # A mask for all the binaries in the metallicity
                channel_mask = (all_channels==ch_index)

                # Declare bins for the chirp mass KDE
                chirp_masses = (COMPAS.mass1[channel_mask]*COMPAS.mass2[channel_mask])**(3./5.) / (COMPAS.mass1[channel_mask] + COMPAS.mass2[channel_mask])**(1./5.)
                actual.create_dataset('chirp_masses', data=chirp_masses) # Scale the KDE

                # empty trash
                del detection_rates
                del formation_rates
                del merger_rates
                del chirp_masses
                del detection_rates_by_binary

                # Delete trash object
                del COMPAS

    rates_and_bootstraps.close()


def pickle_rates_and_boostraps(path, output_path, dco_type=None, merger_output_filename=None, weight_column=None,
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
    logNormalPrescription=None, n_iters=100, num_free_cpus=2):                              # ADAM'S FIDUCIAL MSSFR PARAMETERS
    
    # Make sure the dco_type is given
    assert dco_type != None, 'dco_type must be given'

    bootstrap_kwargs = {'dco_type': dco_type, 'merger_output_filename': merger_output_filename, 'weight_column': weight_column, 'merges_hubble_time': merges_hubble_time, 'pessimistic_CEE': pessimistic_CEE, 
                        'no_RLOF_after_CEE': no_RLOF_after_CEE, 'max_redshift': max_redshift, 'max_redshift_detection': max_redshift_detection, 'redshift_step': redshift_step, 'z_first_SF': z_first_SF,
                        'use_sampled_mass_ranges': use_sampled_mass_ranges, 'm1_min': m1_min, 'm1_max': m1_max, 'm2_min': m2_min, 'fbin': fbin,
                        'aSF': aSF, 'bSF': bSF, 'cSF': cSF, 'dSF': dSF,
                        'mu0': mu0, 'muz': muz, 'sigma0': sigma0, 'sigmaz': sigmaz, 'alpha': alpha, 
                        'min_logZ': min_logZ, 'max_logZ': max_logZ, 'step_logZ': step_logZ,
                        'sensitivity': sensitivity, 'snr_threshold': snr_threshold, 
                        'Mc_max': Mc_max, 'Mc_step': Mc_step, 'eta_max': eta_max, 'eta_step': eta_step,
                        'snr_max': snr_max, 'snr_step': snr_step,
                        'lognormal': lognormal, 'Zprescription': Zprescription, 'SFRprescription': SFRprescription,
                        'GSMFprescription': GSMFprescription, 'ZMprescription': ZMprescription,
                        'logNormalPrescription': logNormalPrescription}

    print('Finding run rates')
    detection_rates, formation_rates, merger_rates, redshifts, COMPAS = \
        FCI.find_detection_rate(
        path, dco_type=dco_type, weight_column=weight_column,
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
        logNormalPrescription=logNormalPrescription)                                            # ADAM'S NON-LOGNORMAL MSSFR PRESCRIPTIONS

    chirp_masses = (COMPAS.mass1*COMPAS.mass2)**(3./5.) / (COMPAS.mass1 + COMPAS.mass2)**(1./5.)

    # Declare an h5 file to put everything in
    rates_and_bootstraps = h5.File(output_path + '/rates_and_bootstraps.h5', 'w')
    actual = rates_and_bootstraps.create_group('Actual')      # The group for the rates calculated with the actual runs

    # sum things up across binaries
    actual.create_dataset('redshifts', data=redshifts)
    actual.create_dataset('total_formation_rates', data=np.sum(formation_rates, axis=0))
    actual.create_dataset('total_merger_rates', data=np.sum(merger_rates, axis=0))
    total_detection_rates = np.sum(detection_rates, axis=0)

    # and across redshifts
    actual.create_dataset('cumulative_detection_rates', data=np.cumsum(total_detection_rates))
    detection_rates_by_binary = np.sum(detection_rates, axis=1)

    # MASS KDE STUFF:
    # Declare bins for the chirp mass KDE
    Mc_bins = np.arange(0, max(chirp_masses)*1.3, max(chirp_masses)*1.2/100) # We need to pass in the bins for the chirp mass distribution so we will declare them here
    hist, _ = np.histogram(chirp_masses, weights = detection_rates_by_binary, bins=Mc_bins) # Get the Hist
    axis = np.arange(Mc_bins[0],Mc_bins[-1],0.1) # The x-axis for the chirp masses
    mass_kde = FFTKDE(bw=0.2).fit(chirp_masses, weights=detection_rates_by_binary).evaluate(axis) # Get the KDE
    actual.create_dataset('mass_kde_scaled', data=mass_kde*sum(hist)*sum(hist)*np.diff(Mc_bins)[0]) # Scale the KDE




    # # Dict to put everything in
    # rates_and_bootstraps = {}

    # # sum things up across binaries
    # rates_and_bootstraps['redshifts'] = redshifts
    # rates_and_bootstraps['total_formation_rates'] = np.sum(formation_rates, axis=0)
    # rates_and_bootstraps['total_merger_rates'] = np.sum(merger_rates, axis=0)
    # total_detection_rates = np.sum(detection_rates, axis=0)

    # # and across redshifts
    # rates_and_bootstraps['cumulative_detection_rates'] = np.cumsum(total_detection_rates)
    # detection_rates_by_binary = np.sum(detection_rates, axis=1)

    # # MASS KDE STUFF:
    # # Declare bins for the chirp mass KDE
    # Mc_bins = np.arange(0, max(chirp_masses)*1.3, max(chirp_masses)*1.2/100) # We need to pass in the bins for the chirp mass distribution so we will declare them here
    # hist, _ = np.histogram(chirp_masses, weights = detection_rates_by_binary, bins=Mc_bins) # Get the Hist
    # axis = np.arange(Mc_bins[0],Mc_bins[-1],0.1) # The x-axis for the chirp masses
    # mass_kde = FFTKDE(bw=0.2).fit(chirp_masses, weights=detection_rates_by_binary).evaluate(axis) # Get the KDE
    # rates_and_bootstraps['mass_kde_scaled'] = mass_kde*sum(hist)*sum(hist)*np.diff(Mc_bins)[0] # Scale the KDE

    # Create the group and dataset for the bootstraps
    rates_and_bootstraps.create_group('Bootstraps') # The group for the boostrapped rates
    bootstraps = rates_and_bootstraps['Bootstraps']
    bootstraps.create_dataset('cumulative_detection_rates', (100, len(total_detection_rates.flatten())))
    bootstraps.create_dataset('total_formation_rates', (100, len(np.sum(formation_rates, axis=0))))
    bootstraps.create_dataset('total_merger_rates', (100, len(np.sum(merger_rates, axis=0))))
    bootstraps.create_dataset('mass_kde', (100, len(axis)))
    rates_and_bootstraps.close()

    # empty trash
    del detection_rates
    del formation_rates
    del merger_rates
    del chirp_masses
    del detection_rates_by_binary
    del hist
    del axis

    # # Calculate confidence intervals
    # rates_and_bootstraps['bootstraps'] = get_bootstraps(COMPAS, n_iters, bootstrap_kwargs, Mc_bins, num_free_cpus=num_free_cpus)
    get_bootstraps(output_path + '/rates_and_bootstraps.h5', COMPAS, n_iters, bootstrap_kwargs, Mc_bins, num_free_cpus=num_free_cpus)

    # Delete trash object
    del COMPAS

    # # Dump everything in a pickle file
    # with open(output_path + '/rates_and_bootstraps.pkl', 'wb') as f:
    #     pickle.dump(rates_and_bootstraps, f)
    
    # del rates_and_bootstraps

