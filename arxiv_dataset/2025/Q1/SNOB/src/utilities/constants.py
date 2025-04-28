import numpy as np

# constants for the axisymmetric and spiral arm models
h_spiral_arm = 2.4    # kpc, scale length of the disk. The value Higdon and Lingenfelter used
h_axisymmetric = 2.4  # kpc, scale length of the disk. The value Higdon and Lingenfelter used
h_lyc = 3.5           # kpc, scale length of the disk for the Lyc emission. The value McKee and Williams used (1997)
r_s = 8.178            # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
# This rho_max and rho_min seems to be taken from Valee
rho_min_spiral_arm = np.array([2.9, 2.9, 2.9, 2.9])           # kpc, minimum distance from galactic center to the beginning of the spiral arms. Values taken from Valleé (see Higdon and lingenfelter)
rho_max_spiral_arm = np.array([35, 35, 35, 35])            # kpc, maximum distance from galactic center to the end of the spiral arms. Values taken from Valleé (see Higdon and lingenfelter)
rho_min_axisymmetric = 3           # kpc, minimum distance from galactic center to bright H 2 regions. Evaluates to 3.12 kpc
rho_max_axisymmetric = 11          # kpc, maximum distance from galactic center to bright H 2 regions. Evaluates to 10.4 kpc
sigma_height_distr = 0.15          # kpc, scale height of the disk
sigma_arm = 0.5                    # kpc, dispersion of the spiral arms
total_galactic_n_luminosity = 1.6e40    #total galactic N 2 luminosity in erg/s
gum_nii_luminosity = 1.2e36 # erg/s, luminosity of the Gum Nebula in N II 205 micron line. Number from Higdon and Lingenfelter
cygnus_nii_luminosity = 2.4e37 # erg/s, luminosity of the Cygnus Loop in N II 205 micron line. Number from Higdon and Lingenfelter
measured_nii_30_deg = 0.00011711056373558678 # erg/s/cm²/sr measured N II 205 micron line intensity at 30 degrees longitude. Retrieved from the FIRAS data with the function firas_data.find_firas_intensity_at_central_long(30)
measured_nii_25_deg = 0.00012774937641419457  # erg/s/cm²/sr measured N II 205 micron line intensity at 30 degrees longitude. Retrieved from the FIRAS data with the function firas_data.find_firas_intensity_at_central_long(30)
kpc = 3.08567758e21    # 1 kpc in cm
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
a_d_axisymmetric = 2*np.pi*h_axisymmetric**2 * ((1+rho_min_axisymmetric/h_axisymmetric)*np.exp(-rho_min_axisymmetric/h_axisymmetric) - (1+rho_max_axisymmetric/h_axisymmetric)*np.exp(-rho_max_axisymmetric/h_axisymmetric)) 
# starting angles, pitch-angles and fractional contributions for the spiral arms, respectively Norma-Cygnus(NC), Perseus(P), Sagittarius-Carina(SA), Scutum-Crux(SC)
# arm angles higdon lingenfelter: 70◦ (Norma–Cygnus arm), 160◦ (Perseus arm), 250◦ (Sagittarius–Carina arm), and 340◦ 
arm_angles = np.radians([68, 165, 240, 333])
pitch_angles = np.radians([13.5, 15.1, 13.8, 16.2])
fractional_contribution = [0.19, 0.35, 0.16, 0.29]
spiral_arm_names = ['Norma-Cygnus', 'Perseus', 'Sagittarius-Carina', 'Scutum-Crux']
number_of_end_points = 45 # number of points to use for the circular projection at the end points of the spiral arms

# parameters for the local arm
pitch_local = np.radians(2.77) # degrees to radians
theta_start_local = np.radians(55.1) # degrees to radians
theta_max_local = np.radians(107) # degrees to radians
rho_min_local = 8.21 # kpc
rho_max_local = rho_min_local * np.exp(np.tan(pitch_local) * (theta_max_local - theta_start_local))
# add the local arm to the spiral arm parameters. If the local arm is not to be included, change settings.py
rho_min_spiral_arm = np.concatenate((rho_min_spiral_arm, [rho_min_local]))
rho_max_spiral_arm = np.concatenate((rho_max_spiral_arm, [rho_max_local]))
arm_angles = np.concatenate((arm_angles, [theta_start_local]))
pitch_angles = np.concatenate((pitch_angles, [pitch_local]))
fractional_contribution = np.concatenate((fractional_contribution, [0.01]))
# add the devoid region in Sagittarius to the spiral arm parameters. If the devoid region is not to be included, change settings.py
rho_min_sagittarius = 5.1 # kpc
rho_max_sagittarius = 7 # kpc
sigma_devoid = 0.25 # Value to make the width of the spiral arm smaller for the devoid region. Enters the function generate_transverse_spacing_densities in spiral_arm_model.py


# Stuff for SN, Ass and Galaxy classes:
seconds_in_myr = 3.156e13
km_in_kpc = 3.2408e-17
year_in_seconds = 31556926 # about 31.5 M
f_binary = 0.7 # fraction of massive stars born in binaries
v_kick = 8 # km/s, fixed distribution for the velocity kick the less massive binary star recieves when the more massive companion dies

# Parameters for the modified Kroupa IMF:
alpha = np.array([0.3, 1.3, 2.3, 2.7])
m_lim_imf_powerlaw = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
# paramanters for the power law describing lifetime as function of mass. Schulreich et al. (2018)
tau_0 = 1.6e8 * 1.65 # fits better with the data for he higher masses, though the slope is still too shallow
beta = -0.932

# Folder locations
FOLDER_GALAXY_DATA = 'galaxy_data'
FOLDER_OBSERVATIONAL_DATA = 'data/observational'
FOLDER_GALAXY_TESTS = 'data/plots/tests_galaxy_class'
FOLDER_MODELS_GALAXY = 'data/plots/models_galaxy'
FOLDER_OBSERVATIONAL_PLOTS = 'data/plots/observational_plots'
FOLDER_CHI_SQUARED = 'data/chi_square_tests'

