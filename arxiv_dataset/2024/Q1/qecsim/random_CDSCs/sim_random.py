import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
import matplotlib.pyplot as plt
import qecsim
from qecsim import app
from qecsim.models.generic import PhaseFlipErrorModel,DepolarizingErrorModel,BiasedDepolarizingErrorModel,BiasedYXErrorModel
from qecsim.models.planar import PlanarCode,PlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder

import app_defp
import _rotatedplanarmpsdecoder_defp
import importlib as imp
imp.reload(app_defp)
imp.reload(_rotatedplanarmpsdecoder_defp)

import os,time,sys
import multiprocessing as mp
from functools import partial

import pickle

def parallel_step(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability,realization_index):
    # np.random.seed(1234 * (run_id + 1))*(realization_index+1)
    random_seed=1234*(realization_index+1)
    result_one_realiz=app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs,None,random_seed)
    # def run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs=None,max_failures=None,random_seed=None):
    return result_one_realiz

def TNDresult(code,decoder,error_model,max_runs,perm_rates,err_prob,code_name,layout,num_realiz):  
    result=[]
    p=mp.Pool()
    func=partial(parallel_step,code,error_model,decoder,max_runs,perm_rates,code_name,layout,err_prob)
    result.append(p.map(func,range(num_realiz)))
    p.close()
    p.join()
    
    return result


if __name__=='__main__':

    code_size  = int(sys.argv[1])
    chi_val    = int(sys.argv[2])
    bias       = float(sys.argv[3])
    max_runs   = int(sys.argv[4])
    num_realiz = int(sys.argv[5])
    run_id     = int(sys.argv[6])
    err_prob = float(sys.argv[7])
    code_name= str(sys.argv[8])
        
    def square(a):
        return a**2
    
    vsquare=np.vectorize(square)
    bdry_name = 'rotated'

    if code_name == 'random_XZ_YZ':
        perm_rates=[1/4,1/4,1/2,0,0,0]
    elif code_name == 'random_XZ':
        perm_rates=[1/2,1/2,0,0,0,0]
    elif code_name=='randomqtrqtr':
	    perm_rates=[1/2,1/4,1/4,0,0,0]
    elif code_name=='randomhalfqtr':
	    perm_rates=[1/4,1/2,1/4,0,0,0]
    elif code_name=='random_YZ':
        perm_rates=[1/2,0,1/2,0,0,0]
    elif code_name=='randomhalfhalf':
        perm_rates=[0,1/2,1/2,0,0,0]
    elif code_name=='randomqtrqtrep':
        perm_rates=[1/2+0.05,1/4-0.05,1/4,0,0,0]
    elif code_name=='randomhalfqtrep':
        perm_rates=[1/4-0.05,1/2+0.05,1/4,0,0,0]        

    from itertools import cycle
    
    code = RotatedPlanarCode(*(code_size,code_size))
    decoder = _rotatedplanarmpsdecoder_defp.RotatedPlanarMPSDecoder_defp(chi=chi_val)
    layout='rotated'
    bias_str='Z'
            
    error_model = BiasedDepolarizingErrorModel(bias,bias_str)
    # print run parameters
    print('code:',code.label )
    print('Error model:',error_model.label)
    print('number of realizations:',num_realiz)
    print('Decoder:',decoder.label)
    print('Error probability:',err_prob)
    print('Maximum runs:',max_runs)
    
    results = TNDresult(code,decoder,error_model,max_runs,perm_rates,err_prob,code_name,layout,num_realiz)
    
    output = {}
    output['code'] = code_name
    output['error_probability'] = err_prob
    output['bias'] = bias
    output['maxruns'] = max_runs
    output['layout'] = layout
    output['chi'] = chi_val
    output['nrod'] = num_realiz
    output['bias_str'] = bias_str
    output['L'] =code_size 

    # output['success_list']  =             [results[j]['success_list'] for j in range(len(results))]
    # output['coset_ps_list'] =             [results[j]['coset_ps_list'] for j in range(len(results))]
    # output['logical_commutations_list'] = [results[j]['logical_commutations_list'] for j in range(len(results))]

    # print(np.shape(results))
    
    output['success_list']  =             [[results[k][j]['success_list'] for j in range(len(results[k]))] for k in range(len(results))]
    output['coset_ps_list'] =             [[results[k][j]['coset_ps_list'] for j in range(len(results[k]))] for k in range(len(results))]
    output['logical_commutations_list'] = [[results[k][j]['logical_commutations_list'] for j in range(len(results[k]))] for k in range(len(results))]
    
    outputpath  = 'data/' + code_name + '_L'+str(code_size) + '_bias' + str(bias)+'_chi' + str(chi_val)+'_rate'+str(err_prob)+'_n' + str(run_id)+'.pickle'
    fout = open(outputpath, 'wb')
    pickle.dump(output, fout, pickle.HIGHEST_PROTOCOL)
    fout.close()

