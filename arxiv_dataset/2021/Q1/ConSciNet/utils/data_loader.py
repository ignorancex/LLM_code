import pickle
import numpy as np

def add_dict(dict1, dict2):
    """
    Combines two data dicts
    """

    if 'length' in dict1.keys():
      out_dict = {}
      out_dict['x'] = np.vstack([dict1['x'],dict2['x']])
      out_dict['aux_vars'] = np.vstack([dict1['aux_vars'],dict2['aux_vars']])
      out_dict['qp_dot'] = np.vstack([dict1['qp_dot'],dict2['qp_dot']])

      out_dict['q'] = dict1['q'] + dict2['q']
      out_dict['p'] = dict1['p'] + dict2['p']
      out_dict['qdot'] = dict1['qdot'] + dict2['qdot']
      out_dict['pdot'] = dict1['pdot'] + dict2['pdot']
      out_dict['t_eval'] = dict1['t_eval'] + dict2['t_eval']
      out_dict['length'] = dict1['length'] + dict2['length']

    if 'k' in dict1.keys():
      out_dict = {}
      out_dict['x'] = np.vstack([dict1['x'],dict2['x']])
      out_dict['aux_vars'] = np.vstack([dict1['aux_vars'],dict2['aux_vars']])
      out_dict['qp_dot'] = np.vstack([dict1['qp_dot'],dict2['qp_dot']])

      out_dict['q'] = dict1['q'] + dict2['q']
      out_dict['p'] = dict1['p'] + dict2['p']
      out_dict['qdot'] = dict1['qdot'] + dict2['qdot']
      out_dict['pdot'] = dict1['pdot'] + dict2['pdot']
      out_dict['t_eval'] = dict1['t_eval'] + dict2['t_eval']
      out_dict['k'] = dict1['k'] + dict2['k']
      out_dict['m'] = dict1['m'] + dict2['m']

    return out_dict

def load_data(system, n_to_load = 5):
    """
    Loads data for a given system
    Parameters:
    system: either 'spring' or 'pendulum' 
    n_to_load: number of 10k entry data files to load
    """

    with open(f'data/{system}_data_10k_trjs1.pickle', 'rb') as handle:
      data = pickle.load(handle)

    if n_to_load > 1:
      data_l = []
      idx = 1
      for i in range(n_to_load-1):
        idx += 1
        with open(f'data/{system}_data_10k_trjs{idx}.pickle', 'rb') as handle:
          data2 = pickle.load(handle)

        data = add_dict(data,data2)

    return data
