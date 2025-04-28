from pylab import *
from scipy.integrate import quad
# The photGalIMF code provides at its current stage only the Ks-band, the IRAC [3.6]-band, and the V-band from the PARSEC isochrones
isochrones_selection = "PARSEC"
band_selection = "V"  # can be "V", "Ks", or "IRAC36"
# Downloaded metallicity grid for PARSEC stellar luminosity
Z_list_value = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.03, 0.04]
# Z_list_value = [0.0001, 0.004, 0.02, 0.04]
# Z_list_value = [0.02]

def polynomial(x,a,b,c,d,e,f,g,h,i,j,k):
    y = a*x**10 + b*x**9 + c*x**8 + d*x**7 + e*x**6 + f*x**5 + g*x**4 + h*x**3 + i*x**2 + j*x + k
    return y

logAge_AGB_list = []
MH_AGB_list = []
logMini_AGB_list = []
loglum_AGB_list = []
logAge_MS_list = []
MH_MS_list = []
logMini_MS_list = []
loglum_MS_list = []
for Z__ in Z_list_value:
    # Load AGB data
    data_AGB = np.loadtxt('stellar_isochrones/{}/{}_band/{}_{}_band_Average_Interpolation_AGB_Z_{}.txt'.format(isochrones_selection, band_selection, isochrones_selection, band_selection, Z__))
    logAge_AGB_list.append(np.round(data_AGB[:, 0], 5))
    MH_AGB_list.append(np.round(data_AGB[:,1], 5))
    logMini_AGB_list.append(data_AGB[:,2])
    loglum_AGB_list.append(data_AGB[:,3])
    # Load MS data for time < 10**8.5yr and all metallicities
    data_MS = np.loadtxt('stellar_isochrones/{}/{}_band/{}_{}_band_Interpolation_MS_Z_{}.txt'.format(isochrones_selection, band_selection, isochrones_selection, band_selection, Z__))
    logAge_MS_list.append(np.round(data_MS[:, 0], 5))
    MH_MS_list.append(np.round(data_MS[:, 1], 5))
    logMini_MS_list.append(data_MS[:, 2])
    loglum_MS_list.append(data_MS[:, 3])
#Load MS data for time >= 10**8.5yr and all metallicities
data_fit = np.loadtxt('stellar_isochrones/{}/{}_band/{}_{}_band_Average_Fits_MS.txt'.format(isochrones_selection, band_selection, isochrones_selection, band_selection))
logAge_fit = np.round(data_fit[:, 0], 5)
Z_fit = np.round(data_fit[:, 1], 5)
logMass_interpolation_threshold = data_fit[:, 2]

# Define time range
logAge_list_value = np.round(sorted(set(logAge_fit)), 5)


def igimf_luminous_function(mass, igimf_of_this_epoch, logAge_value, stellar_Z_select, index_MS_fit, index_AGB, index_MS):
    xi__ = igimf_of_this_epoch.custom_imf(mass, 0)
    log_mass = np.log10(mass)
    log_lum = -999

    Z_list_index = np.argmin(np.abs(np.array(Z_list_value) - stellar_Z_select))

    # MS EARLY Phase
    logMini_MS = logMini_MS_list[Z_list_index]
    loglum_MS = loglum_MS_list[Z_list_index]
    # AGB Phase
    logMini_AGB = logMini_AGB_list[Z_list_index]
    loglum_AGB = loglum_AGB_list[Z_list_index]

    # Interpolation-Averaging Method of AGB (Pulsar) phase
    if (log_mass >= logMass_interpolation_threshold[index_MS_fit]):  # From MS Phase
        log_mass_selected = logMini_AGB[index_AGB]
        log_lum_selected = loglum_AGB[index_AGB]
        N = len(log_mass_selected)
        # Calculate the luminosity in the given Band
        if ((log_mass <= max(log_mass_selected)) & (log_mass >= min(log_mass_selected))):
            for i in range(0, N - 1):
                log_mass_now = log_mass_selected[i]
                log_mass_next = log_mass_selected[i + 1]
                log_lum_now = log_lum_selected[i]
                log_lum_next = log_lum_selected[i + 1]
                if (log_mass_next >= log_mass):
                    log_lum = log_lum_now + (log_lum_next - log_lum_now) / (log_mass_next - log_mass_now) * (log_mass - log_mass_now)
                    break
        lum__ = 10 ** log_lum
    # Fitting Method for MS Phase
    else:
        if logAge_value < 8.5:
            # Interpolation of MS phase
            log_mass_selected = logMini_MS[index_MS]
            log_lum_selected = loglum_MS[index_MS]
            N = len(log_mass_selected)
            # Calculate the luminosity in the given Band
            if ((log_mass <= max(log_mass_selected)) & (log_mass >= min(log_mass_selected))):
                for i in range(0, N - 1):
                    log_mass_now = log_mass_selected[i]
                    log_mass_next = log_mass_selected[i + 1]
                    log_lum_now = log_lum_selected[i]
                    log_lum_next = log_lum_selected[i + 1]
                    if (log_mass_next >= log_mass):
                        log_lum = log_lum_now + (log_lum_next - log_lum_now) / (log_mass_next - log_mass_now) * (log_mass - log_mass_now)
                        break
            lum__ = 10 ** log_lum
        else:
            # Calculate the luminosity in the given Band
            a_coeff = data_fit[index_MS_fit, 3]
            b_coeff = data_fit[index_MS_fit, 4]
            c_coeff = data_fit[index_MS_fit, 5]
            d_coeff = data_fit[index_MS_fit, 6]
            e_coeff = data_fit[index_MS_fit, 7]
            f_coeff = data_fit[index_MS_fit, 8]
            g_coeff = data_fit[index_MS_fit, 9]
            h_coeff = data_fit[index_MS_fit, 10]
            i_coeff = data_fit[index_MS_fit, 11]
            j_coeff = data_fit[index_MS_fit, 12]
            k_coeff = data_fit[index_MS_fit, 13]
            log_lum = polynomial(log_mass, a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, f_coeff, g_coeff, h_coeff, i_coeff, j_coeff, k_coeff)
            lum__ = 10 ** log_lum
    return xi__ * lum__

# Function to calculate the stellar luminosity of a SSP with an interpolated metallicity.
def SSP_luminosity(igimf_of_this_epoch, stellar_Z, stellar_age, lower_boundary, upper_boundary, bin_number):
    if len(Z_list_value)==1 or stellar_Z < Z_list_value[0] or stellar_Z == Z_list_value[0]:
        lum__ = SSP_luminosity_metal(igimf_of_this_epoch, Z_list_value[0], stellar_age, lower_boundary, upper_boundary, bin_number)
    elif stellar_Z > Z_list_value[-1] or stellar_Z == Z_list_value[-1]:
        lum__ = SSP_luminosity_metal(igimf_of_this_epoch, Z_list_value[-1], stellar_age, lower_boundary, upper_boundary, bin_number)
    else:
        low, high = 0, len(Z_list_value) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_value = Z_list_value[mid]
            if mid_value == stellar_Z:
                lum__ = SSP_luminosity_metal(igimf_of_this_epoch, Z_list_value[mid], stellar_age, lower_boundary, upper_boundary, bin_number)
                return lum__
            elif mid_value < stellar_Z:
                low = mid + 1
            else:
                high = mid - 1
        lum_up = SSP_luminosity_metal(igimf_of_this_epoch, Z_list_value[high], stellar_age, lower_boundary, upper_boundary, bin_number)
        lum_down = SSP_luminosity_metal(igimf_of_this_epoch, Z_list_value[low], stellar_age, lower_boundary, upper_boundary, bin_number)
        lum__ = np.interp(stellar_Z, [Z_list_value[low], Z_list_value[high]], [lum_down, lum_up])
    return lum__

# Function to calculate the stellar luminosity of a SSP with a given metallicity.
def SSP_luminosity_metal(igimf_of_this_epoch, stellar_Z_select, stellar_age, lower_boundary, upper_boundary, bin_number):
    # index search
    # # Main-sequence Phase #Definition of Age and Metallicity
    # # Search for the closest metallicity in the Padova sample
    # Z_list_index = np.argmin(np.abs(np.array(Z_list_value) - stellar_Z))
    # stellar_Z_select = Z_list_value[Z_list_index]
    # Search for the closest age in the Padova sample
    logAge_value = np.round(np.log10(stellar_age), 5)
    logAge_list_index = np.argmin(np.abs(np.array(np.round(logAge_list_value, 5)) - np.round(logAge_value, 5)))
    logAge_value_extrapolated = logAge_list_value[logAge_list_index]
    index_MS_fit = np.where((logAge_fit == np.round(logAge_value_extrapolated, 5)) & (Z_fit == np.round(stellar_Z_select, 5)))
    index_MS_fit = index_MS_fit[0]

    Z_list_index = np.argmin(np.abs(np.array(Z_list_value) - stellar_Z_select))
    # AGB Phase
    logAge_AGB = logAge_AGB_list[Z_list_index]
    # MS EARLY Phase
    logAge_MS = logAge_MS_list[Z_list_index]

    # Select stars for a given age and metallicity
    index_AGB = np.where((np.round(logAge_AGB, 5) == np.round(logAge_value_extrapolated, 5))) # Metallicity already selected
    index_AGB = index_AGB[0]

    index_MS = np.where((np.round(logAge_MS, 5) == np.round(logAge_value_extrapolated, 5)))  # Metallicity already selected
    index_MS = index_MS[0]

    # Integrate luminosity for a SSP with a given IMF and stellar mass range
    lum_SSP = quad(igimf_luminous_function, lower_boundary, upper_boundary, args=(igimf_of_this_epoch, logAge_value, stellar_Z_select, index_MS_fit, index_AGB, index_MS), limit=bin_number)[0]
    return lum_SSP


#Function to calculate the stellar luminosity of a given stellar mass.
def stellar_luminosity_function(mass, stellar_Z, stellar_age):
    log_mass = np.log10(mass)  # mass from Yan's code

    #Main-sequence Phase #Definition of Age and Metallicity
    #Search for the closest metallicity in the Padova sample
    Z_list_index = np.argmin(np.abs(np.array(Z_list_value) - stellar_Z))
    stellar_Z_select = Z_list_value[Z_list_index]

    # Search for the closest age in the Padova sample
    logAge_value = np.round(np.log10(stellar_age), 5)
    logAge_list_index = np.argmin(np.abs(np.array(np.round(logAge_list_value, 5)) - np.round(logAge_value, 5)))
    logAge_value_extrapolated = logAge_list_value[logAge_list_index]
    index_MS_fit = np.where((logAge_fit == np.round(logAge_value_extrapolated, 5)) & (Z_fit == np.round(stellar_Z_select, 5)))
    index_MS_fit = index_MS_fit[0]

    # AGB Phase
    logAge_AGB = logAge_AGB_list[Z_list_index]
    logMini_AGB = logMini_AGB_list[Z_list_index]
    loglum_AGB = loglum_AGB_list[Z_list_index]

    # Select stars for a given age and metallicity
    index_AGB = np.where((np.round(logAge_AGB, 5) == np.round(logAge_value_extrapolated, 5))) # Metallicity already selected
    index_AGB = index_AGB[0]

    log_lum = -999
    #Interpolation-Averaging Method of AGB (Pulsar) phase
    if log_mass >= logMass_interpolation_threshold[index_MS_fit]: #From MS Phase
        log_mass_selected = logMini_AGB[index_AGB]
        log_lum_selected = loglum_AGB[index_AGB]
        N = len(log_mass_selected)
        # Calculate the luminosity in a given band
        if (log_mass <= max(log_mass_selected)) & (log_mass >= min(log_mass_selected)):
            for i in range(0, N - 1):
                log_mass_now = log_mass_selected[i]
                log_mass_next = log_mass_selected[i + 1]
                log_lum_now = log_lum_selected[i]
                log_lum_next = log_lum_selected[i + 1]
                if log_mass_next >= log_mass:
                    log_lum = log_lum_now + (log_lum_next - log_lum_now) / (log_mass_next - log_mass_now) * (log_mass - log_mass_now)
                    break
        lum__ = 10 ** log_lum
    #Fitting Method for MS Phase
    else:
        if logAge_value < 8.5:
            #MS EARLY Phase
            logAge_MS = logAge_MS_list[Z_list_index]
            logMini_MS = logMini_MS_list[Z_list_index]
            loglum_MS = loglum_MS_list[Z_list_index]

            # Select stars for a given age and metallicity
            index_MS = np.where((np.round(logAge_MS, 5) == np.round(logAge_value_extrapolated, 5))) # Metallicity already selected
            index_MS = index_MS[0]
            #Interpolation of MS phase
            log_mass_selected = logMini_MS[index_MS]
            log_lum_selected = loglum_MS[index_MS]
            N = len(log_mass_selected)
            # Calculate the luminosity in a given Band
            if ((log_mass <= max(log_mass_selected)) & (log_mass >= min(log_mass_selected))):
                for i in range(0, N - 1):
                    log_mass_now = log_mass_selected[i]
                    log_mass_next = log_mass_selected[i + 1]
                    log_lum_now = log_lum_selected[i]
                    log_lum_next = log_lum_selected[i + 1]

                    if (log_mass_next >= log_mass):
                        log_lum = log_lum_now + (log_lum_next - log_lum_now) / (log_mass_next - log_mass_now) * (log_mass - log_mass_now)
                        break
            lum__ = 10 ** log_lum
        else:
            # Calculate the luminosity in a given band
            a_coeff = data_fit[index_MS_fit,3]
            b_coeff = data_fit[index_MS_fit,4]
            c_coeff = data_fit[index_MS_fit,5]
            d_coeff = data_fit[index_MS_fit,6]
            e_coeff = data_fit[index_MS_fit,7]
            f_coeff = data_fit[index_MS_fit,8]
            g_coeff = data_fit[index_MS_fit,9]
            h_coeff = data_fit[index_MS_fit,10]
            i_coeff = data_fit[index_MS_fit,11]
            j_coeff = data_fit[index_MS_fit,12]
            k_coeff = data_fit[index_MS_fit,13]

            log_lum = polynomial(log_mass,a_coeff,b_coeff,c_coeff,d_coeff,e_coeff,f_coeff,g_coeff,h_coeff,i_coeff,j_coeff,k_coeff)
            lum__ = 10 ** log_lum
    return lum__


if __name__ == '__main__':
    stellar_metallicity = 0.02
    print("This example plot the adopted stellar mass--luminosity relation given by PARSEC for stars with Z = {} and at ages of 10^7, 10^8, 10^9, and 10^10 yr.".format(stellar_metallicity))

    import matplotlib.pyplot as plt

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(0, figsize=(10, 7))
    fig.add_subplot(1, 1, 1)

    start_time = time.time()

    age = 1e7
    while age < 13e9:
        mass_list = []
        lumn_list = []
        mass = 0.1
        while mass < 30:
            mass_list += [mass]
            lumn_list += [stellar_luminosity_function(mass, stellar_metallicity, age)]
            (mass) = (mass * 1.001)
        plt.plot(mass_list, lumn_list, color="k")
        (age) = (age * 10)

    computation_time_seconds = round((time.time() - start_time), 2)
    minutes, seconds = divmod(computation_time_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    days = int(days)
    hours = int(hours)
    minutes = int(minutes)
    seconds = round(seconds, 4)
    print("- Simulation complete. Computation time: {} d {} h {} m {} s -".format(days, hours, minutes, seconds))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'Stellar mass [$M_\odot$]')
    plt.ylabel(r'Stellar luminosity [$L_\odot$]')
    plt.tight_layout()
    plt.show()
