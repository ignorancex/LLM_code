import camb
import pickle
from camb import model, initialpower

#Set up a new set of parameters for CAMB
print('reading in params')
pars = camb.read_ini('../data/input/universe_Planck15/camb/params_planck15_highl.ini')

#calculate results for these parameters
print('computing results')
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
print('Getiting cmb power dict')
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
print('getting PP PT PE')
cl = results.get_lens_potential_cls( CMB_unit='muK')  # PP PT PE
print('getting TT EE BB TE')
c_lensed = results.get_lensed_scalar_cls( CMB_unit='muK') #TT EE BB TE
print('getting TGradT')
c_lens_response = results.get_lensed_gradient_cls( CMB_unit='muK') #T GradT and others

print('saving results')

oup_fname = '../data/input/universe_Planck15/camb/CAMB_outputs.pkl'
print(oup_fname)
f = open(oup_fname, 'wb') 
pickle.dump((powers,cl,c_lensed,c_lens_response) , f)
f.close()
