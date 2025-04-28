import numpy as np 

sel_pop = {'detached' : lambda x : x['CLASS'] == 'Detached', 'ckbo' : lambda x : (x['CLASS'] == 'Classical') & (x['i_free'] < 5), \
			'hkbo' : lambda x : (x['CLASS'] == 'Classical') & (x['i_free'] > 5), 'classical': lambda x: x['CLASS'] == 'Classical',\
			'scattering' : lambda x : x['CLASS'] == 'Scattering', 'resonant' : lambda x : x['CLASS'] == 'Resonant',\
			'plutino' : lambda x : x['Class'] == 'Resonant 3:2', 'twotino' : lambda x : x['Class'] == 'Resonant 2:1',  \
			'trojan' : lambda x : x['Class'] == 'Resonant 1:1', 'fivetwo' : lambda x : x['Class'] == 'Resonant 5:2',\
			'all' : lambda x : np.ones(len(x), dtype='bool'), 'noncold'  : lambda x : np.logical_not((x['CLASS'] == 'Classical') & (x['i_free'] < 5)),\
            'nonscat' : lambda x : x['CLASS'] != 'Scattering', 'nondet' : lambda x : x['CLASS'] != 'Detached',\
			'nonscatdet' : lambda x : (x['CLASS'] != 'Detached') & (x['CLASS'] != 'Scattering'), 'innerbelt' : lambda x : (x['CLASS'] == 'Resonant') & (x['a'] < 39) & (x['a'] > 31), 'mainbelt' : lambda x : (x['CLASS'] == 'Resonant') & (x['a'] < 47) & (x['a'] > 40), 'outer' : lambda x : (x['CLASS'] == 'Resonant') & (x['a'] > 49), 'distant' : lambda x : (x['CLASS'] == 'Resonant') & (x['a'] > 49) & (x['Class'] != 'Resonant 5:2'), 'scatdet' : lambda x : np.logical_or(x['CLASS'] == 'Detached', x['CLASS'] == 'Scattering'), 'scatdethot' : lambda x : np.logical_or(np.logical_and(x['CLASS'] == 'Classical', x['i_free'] > 5), np.logical_or(x['CLASS'] == 'Detached', x['CLASS'] == 'Scattering'))}


sel_inc_ifree = {'2ifree' : lambda x : (x['i_free'] < 2.5), '5ifree' : lambda x : (x['i_free'] > 2.5) & (x['i_free'] < 5), '10ifree' : lambda x : (x['i_free'] > 5) & (x['i_free'] < 10), '15ifree' : lambda x : (x['i_free'] > 10) & (x['i_free'] < 15), '25ifree' : lambda x : (x['i_free'] > 15) & (x['i_free'] < 25), '35ifree' : lambda x : (x['i_free'] > 25) & (x['i_free'] < 35), '45ifree' : lambda x : (x['i_free'] > 35) & (x['i_free'] < 45), '90ifree' : lambda x : (x['i_free'] > 45) }

def join_functions(func_a, func_b):
	def create_join(x):
		return func_a(x) & func_b(x)
	return lambda x : create_join(x)

cl_dict = {}
for i in sel_inc_ifree:
    cl_dict['classical' + i] = join_functions(sel_pop['classical'], sel_inc_ifree[i])

sel_inc = {'10' : lambda x : (x['i'] > 0) & (x['i'] < 10), 	'20' : lambda x : (x['i'] > 10) & (x['i'] < 20), '30' : lambda x : (x['i'] > 20) & (x['i'] < 30),'40' : lambda x : (x['i'] > 30) & (x['i'] < 40), '90' : lambda x : (x['i'] > 40) }

pl_dict = {}
for i in sel_inc:
    pl_dict['plutino' + i] = join_functions(sel_pop['plutino'], sel_inc[i])

sc_dict = {}
for i in sel_inc:
    sc_dict['scatdet' + i] = join_functions(sel_pop['scatdet'], sel_inc[i])
    


sel = sel_pop | cl_dict | sc_dict | pl_dict

def create_function(func_a):
	def func(x):
		return func_a(x)
	return lambda x : x[func_a(x)]

selections = {i : create_function(sel[i]) for i in sel}