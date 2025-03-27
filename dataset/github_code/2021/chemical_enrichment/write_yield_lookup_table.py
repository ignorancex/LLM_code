# This script creates a yield table with CHEMPY (Rybizki et al. 2017) in numpy format and writes it in xdr file format.
# Yield tables are 4 dimensional depending on: stellar age, stellar (total) metallicity, source (AGB, SNia, SNii) and elemental species.
# 
# 
# table format:
#
# the general table format is as follows: A Header stating the structure of the table, the table body consisiting of N_z blocks 
# of Ns * Ne lines for every stellar metallicity and columns for the stellar age of the SSP (although in xdr format we do not really have new lines...)
#
# Header:     nuber of steps in stellar metallicity Z (N_z), number of steps in stellar age t (N_t), number of elemental species Ne, number of sources Ns
# (no xdr    nSpecies, nSources
# format)    Z_min, Zmax, dZ
#             t_min=0, t_max=t_universe, dt (in log space to sample early stellar evolutionary phases better than late phases - see what Jan provides)
#            list of strings denoting the elements, e.g. 'Ag', 'Al', 'Ba', 'C', 'Fe', 'He', 'Mg', 'N', 'Na', 'Ne', 'O', 'S', 'Si', 'Ti', 'Zn' (should have same length as Ne)
#            write every element in a new line, makes it easier to read in in c.
#
# Table Body:     -------------> stellar age of SSP: N_t
# (xdr format)   |SSP of Z_min:       Mass Loss, source 1
#                |                    ...
#                |                    Mass Loss, source Ns
#                |                    Number of events, source 1
#                |                    ...
#                |                    Number of events, soure Ns
#                |                    Element 1, source 1
#                |                    Element 1, source 2
#                |                    ...
#                |                    Element 1, source Ns
#                |                    ...
#                |                    Element 2, source 1
#                |                    ...
#                |
#                |                    Element Ne, source Ns
#                |SSP of Z_min+dZ:    Mass loss, source 1
#                |                    ...
#                |                    Mass Loss, source Ns
#                |                    Number of events
#                |                    Element 1, source 1
#                |                    ...
#                |                    Element Ne, source Ns
#                |                    ...
#                |SSP of Z_max:       Mass Loss, source 1
#                |                    ....
#                |                    Element 1, source 1
#                |                    ...
#                |                    Element Ne, source Ns
#       Z of SSP |
#
#
# in fact, mass loss and unprocessed mass is the same.
# caveat: SNIA yields are not metallicity dependent and unprocessed mass loss is 0, thus we only store the yields once at the lowest metallicity.
# caveat: direct BH collapse returns only intial element fractions and happens before SNII so we combine the mass loss with SNII mass loss.

import numpy as np
import xdrlib
import numpy as np
import multiprocessing as mp
import os

####### SETTING THE CHEMPY PARAMETER #######################

from Chempy.parameter import ModelParameters
a = ModelParameters()

# Load solar abundances
from Chempy.solar_abundance import solar_abundances
basic_solar = solar_abundances()
getattr(basic_solar, 'Asplund09')()

# Load the yields
from Chempy.yields import SN2_feedback, AGB_feedback, SN1a_feedback
basic_sn2 = SN2_feedback()
getattr(basic_sn2, 'chieffi04_net')()
basic_1a = SN1a_feedback()
getattr(basic_1a, "Seitenzahl")()
basic_agb = AGB_feedback()
getattr(basic_agb, "Karakas16_net")()

# Print all supported elements
elements_to_trace = list(np.unique(basic_agb.elements+basic_sn2.elements+basic_1a.elements))
print(elements_to_trace)

# Producing the SSP birth elemental fractions (here we use solar)
solar_fractions = []
elements = np.hstack(basic_solar.all_elements)
for item in elements_to_trace:
    solar_fractions.append(float(basic_solar.fractions[np.where(elements==item)]))

# Initialise the SSP class with time-steps
time_steps = np.logspace(-2.44,1.139879,100) #np.linspace(0.,13.8,1024)
a.log_time = True

# yieldset
a.yield_table_name_sn2 = 'chieffi04_net'
a.yield_table_name_agb = 'Karakas16_net'
a.yield_table_name_1a = 'Seitenzahl'

# imf parameters
a.only_net_yields_in_process_tables = True
a.imf_type_name = 'Chabrier_1'
a.chabrier_para1 = 0.69
a.chabrier_para2 = 0.079
a.high_mass_slope = -2.3
a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.high_mass_slope)
a.mmin = 0.1
a.mmax = 100
# 100,000,000 mass steps are smooth enough for 1000 time steps
a.mass_steps = 1000000 #100000000 #2000 # 200000
a.sn2mmin = 8.
a.sn2mmax = 40.
a.bhmmin = float(a.sn2mmax) ## maximum of hypernova
a.bhmmax = float(a.mmax) ## maximum of the IMF

# sn1a delay parameters for maoz
a.N_0 = np.power(10,-2.9)
a.sn1a_time_delay = np.power(10,-1.39794) #40 Myr
a.sn1a_exponent = 1.12
a.dummy = 0.0
a.sn1a_parameter = [a.N_0,a.sn1a_time_delay,a.sn1a_exponent,a.dummy]
######################## END OF SETTING CHEMPY PARAMETER ########################

######################## SETTING THE YIELDTABLE PARAMETERS ######################
list_of_metallicities = np.logspace(-5,-1.3,50)

from Chempy.wrapper import SSP_wrap

def create_one_SSP_table_old(parameters, source='SNII'):
    differential_table = True
    metallicity = parameters
    print(metallicity,a.yield_table_name_sn2)
    basic_ssp = SSP_wrap(a)
    basic_ssp.calculate_feedback(metallicity,list(elements_to_trace),list(solar_fractions),np.copy(time_steps),1)

    x = basic_ssp.agb_table
    y = basic_ssp.sn1a_table
    z = basic_ssp.sn2_table
    s = basic_ssp.bh_table
    d = basic_ssp.table

    u = np.zeros_like(x)
    names = list(u.dtype.names)

    for j,jtem in enumerate(names):
        if source == 'SNII':
            u[jtem] = z[jtem]
        if source == 'SNIA':
            u[jtem] = y[jtem]
        if source == 'AGB':
            u[jtem] = x[jtem]
        if source == 'BH':
            u[jtem] = s[jtem]
        if source == 'ALL':
            u[jtem] = x[jtem] + y[jtem] + z[jtem] + s[jtem]
    if differential_table:
        for el in elements_to_trace:
            d[el] = u[el]
    else:
        for el in elements_to_trace:
            d[el] = np.cumsum(u[el])
        for name in ['mass_of_ms_stars_dying', 'mass_in_remnants', 'sn2', 'sn1a', 'pn', 'bh', 'hydrogen_mass_accreted_onto_white_dwarfs', 'unprocessed_ejecta']:
            d[name] = np.cumsum(d[name])

    return(d)

def create_one_SSP_table(parameters, source='SNII'):
    metallicity = parameters
    print(metallicity,a.yield_table_name_sn2)
    basic_ssp = SSP_wrap(a)
    basic_ssp.calculate_feedback(metallicity,list(elements_to_trace),list(solar_fractions),np.copy(time_steps),1)

    x = basic_ssp.agb_table
    y = basic_ssp.sn1a_table
    z = basic_ssp.sn2_table
    s = basic_ssp.bh_table
    d = basic_ssp.table

    if source == 'SNII':
        return z
    if source == 'SNIA':
        return y
    if source == 'AGB':
        return x
    if source == 'BH':
        return s
    if source == 'ALL':
        u = np.zeros_like(x)
        names = list(u.dtype.names)
        for j,jtem in enumerate(names):
            u[jtem] = x[jtem] + y[jtem] + z[jtem] + s[jtem]
        return u

def my_wrap_AGB_table(parameters):
    return create_one_SSP_table(parameters, source='AGB')

def my_wrap_SNIA_table(parameters):
    return create_one_SSP_table(parameters, source='SNIA')

def my_wrap_SNII_table(parameters):
    return create_one_SSP_table(parameters, source='SNII')

def my_wrap_BH_table(parameters):
    return create_one_SSP_table(parameters, source='BH')

########## END OF SETTING YIELD TABLE PARAMETERS ################

# Call the SSP table creation routine
print('This python script reads a numpy file created with chempy to transform it to xdr file format to be read in by Gasoline!')

print("There are %d CPUs on this machine" % mp.cpu_count())
number_processes = max(1,20)# mp.cpu_count() - 1)
print("Using %d of them.", number_processes)

file = 'chempy_table_agb'
if not os.path.isfile(file+'.npy'):
    ############ CREATING THE ACTUAL TABLES ####################
    list_of_SSP_tables = []
    list_of_SSP_tables.append(list_of_metallicities)
    list_of_SSP_tables.append(time_steps)
    pool = mp.Pool(number_processes)
    results = pool.map(my_wrap_AGB_table, list_of_metallicities)
    pool.close()
    pool.join()
    list_of_SSP_tables.append(results)
    np.save(file, list_of_SSP_tables)

    list_of_SSP_tables = []
    list_of_SSP_tables.append(list_of_metallicities)
    list_of_SSP_tables.append(time_steps)
    pool = mp.Pool(number_processes)
    results = pool.map(my_wrap_SNIA_table, list_of_metallicities)
    pool.close()
    pool.join()
    list_of_SSP_tables.append(results)
    np.save('chempy_table_snia', list_of_SSP_tables)

    list_of_SSP_tables = []
    list_of_SSP_tables.append(list_of_metallicities)
    list_of_SSP_tables.append(time_steps)
    pool = mp.Pool(number_processes)
    results = pool.map(my_wrap_SNII_table, list_of_metallicities)
    pool.close()
    pool.join()
    list_of_SSP_tables.append(results)
    np.save('chempy_table_snii', list_of_SSP_tables)

    list_of_SSP_tables = []
    list_of_SSP_tables.append(list_of_metallicities)
    list_of_SSP_tables.append(time_steps)
    pool = mp.Pool(number_processes)
    results = pool.map(my_wrap_BH_table, list_of_metallicities)
    pool.close()
    pool.join()
    list_of_SSP_tables.append(results)
    np.save('chempy_table_bh', list_of_SSP_tables)


############### DOING THE EXPORT TO XDR FORMAT ######################
# import yield table created with chempy
print('Reading chempy yield table...')
yield_table_agb = np.load('chempy_table_agb.npy')
yield_table_snIa = np.load('chempy_table_snia.npy')
yield_table_snII = np.load('chempy_table_snii.npy')
yield_table_bh = np.load('chempy_table_bh.npy')

yield_sn2 = 'chieffi+04'
yield_snia = 'seitenzahl+13'
yield_agb = 'karakas+16'

# get parameters from yield table
N_Z = len(yield_table_snII[0])
N_t = len(yield_table_snII[1])
N_e = len(elements_to_trace) #len(yield_table_snII[2][1][0])-9 #change length of elements once final yield table is there

Zmin = yield_table_snII[0][0]
Zmax = yield_table_snII[0][len(yield_table_snII[0])-1]
dZ = np.log10(yield_table_snII[0][1])-np.log10(yield_table_snII[0][0])

tmin = yield_table_snII[1][0]*1e9
tmax = yield_table_snII[1][len(yield_table_snII[1])-1]*1e9
dt = np.log10(yield_table_snII[1][1])-np.log10(yield_table_snII[1][0])

# open new xdr file
xdr_table = open("yieldtable_xdr_high_Ia_norm","wb")

# initialize packing
p = xdrlib.Packer()

# wirte Header
print('Writing Header, stating information about metallicity steps, timesteps and number of elements and sources.')

xdr_table.write(b'###################################################################################\n')
#xdr_table.write(b'\n')
xdr_table.write(b'###   Yield table created with CHEMPY (https://github.com/jan-rybizki/Chempy)   ###\n')
#xdr_table.write(b'\n')
xdr_table.write(b'###               using the write_yield_lookup_table.py script.                 ###\n')
#xdr_table.write(b'\n')
xdr_table.write(b'###                Tobias Buck (tbuck@aip.de) in February 2021.                 ###\n')
#xdr_table.write(b'\n')
xdr_table.write(b'###################################################################################\n')

# here should go which yield sets we use.
# We should actually add all chempy paramters, like e.g. IMF and SNIA delay time parameters.
# In this way the yieldtable is self-consistent.

yieldsets = b'SNII, SNIA and AGB yieldsets used: %s\t %s\t %s\n'%(a.yield_table_name_sn2.encode('latin-1'),a.yield_table_name_1a.encode('latin-1'),a.yield_table_name_agb.encode('latin-1')) #(yield_sn2,yield_snia,yield_agb)
xdr_table.write(yieldsets)

imf = b'IMF parameters used: type: %s (%.3f,%.3f,%.3f)\t IMF min/max mass: %.2f/%.2f\t SN min/max mass: %.1f/%.1f\n'%(a.imf_type_name.encode('latin-1'),a.chabrier_para1,a.chabrier_para2,a.high_mass_slope,a.mmin,a.mmax,a.sn2mmin,a.sn2mmax)
xdr_table.write(imf)

snia = b'SNIA parameters used: type: Maoz+2012, normalization: %.3f\t delay time (Myr): %.2f\t exponent: %.2f\n'%(a.N_0,a.sn1a_time_delay,a.sn1a_exponent)
xdr_table.write(snia)

firstline = b'%i %i %i 3\n'%(N_Z, N_t, N_e)
xdr_table.write(firstline)

secondline = b'%E %E %E\n'%(Zmin, Zmax, dZ)
xdr_table.write(secondline)

thirdline = b'%E %E %E\n'%(tmin, tmax, dt)
xdr_table.write(thirdline)

#SNIA do not eject unprocessed material calculate the massloss from the newly synthesized fraction

for elem in elements_to_trace: #yield_table_snII[2][0].dtype.names[4:]:
    element = b'%s\t'%(elem.encode('latin-1'))
    xdr_table.write(element)
    yield_table_snIa[2][0]['unprocessed_ejecta'] += yield_table_snIa[2][0][elem]
xdr_table.write(b'\n')

print('Packing data to xdr format.')

for i in range(N_Z):
    #step through all metallicity bins
    #combine massloss from direct BH collapse with SNII mass loss
    yield_table_snII[2][i]["unprocessed_ejecta"] += yield_table_bh[2][i]["unprocessed_ejecta"]

    if i == 0:
        # at the lowest Z bin include the SNIA yields
        #p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['mass_of_ms_stars_dying']) - np.cumsum(yield_table_snII[2][i]['mass_in_remnants']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['unprocessed_ejecta']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_snIa[2][i]['unprocessed_ejecta']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_agb[2][i]['unprocessed_ejecta']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['number_of_events']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_snIa[2][i]['number_of_events']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_agb[2][i]['number_of_events']),p.pack_double)
        for elem in elements_to_trace: #yield_table_snII[2][0].dtype.names[4:]:
            # step through all elements
            # and finally step also through all sources once we have them...
            p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i][elem]),p.pack_double)
            p.pack_farray(N_t,np.cumsum(yield_table_snIa[2][i][elem]),p.pack_double)
            p.pack_farray(N_t,np.cumsum(yield_table_agb[2][i][elem]),p.pack_double)
    else:
        # now only SNII and AGB
        #p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['mass_of_ms_stars_dying']) - np.cumsum(yield_table_snII[2][i]['mass_in_remnants']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['unprocessed_ejecta']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_agb[2][i]['unprocessed_ejecta']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['number_of_events']),p.pack_double)
        #p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i]['sn1a']),p.pack_double)
        p.pack_farray(N_t,np.cumsum(yield_table_agb[2][i]['number_of_events']),p.pack_double)
        for elem in elements_to_trace: #yield_table_snII[2][0].dtype.names[4:]:
            # step through all elements
            # and finally step also through all sources once we have them...
            p.pack_farray(N_t,np.cumsum(yield_table_snII[2][i][elem]),p.pack_double)
            #p.pack_farray(N_t,np.cumsum(yield_table_snIa[2][i][elem]),p.pack_double)
            p.pack_farray(N_t,np.cumsum(yield_table_agb[2][i][elem]),p.pack_double)

print('Writing xdr part.')
#xdr_table.write(p.get_buffer())

xdr_table.write(p.get_buffer())

xdr_table.close()





