import collections
import itertools
import numpy as np
from qecsim import paulitools as pt
# import matplotlib.pyplot as plt
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

def parallel_step_code(code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability,run_id,realization_index):
    np.random.seed(1234 * (realization_index + 1)*(run_id+1))
    result_one_realiz=app_defp.run_defp(code,error_model,decoder,error_probability,perm_rates,code_name,layout,max_runs)
    return result_one_realiz

def TNDresult_random(code,decoder,error_model,max_runs,perm_rates,error_probability,code_name,layout,run_id,num_realiz):  
    result=[]
    p=mp.Pool()
    func=partial(parallel_step_code,code,error_model,decoder,max_runs,perm_rates,code_name,layout,error_probability,run_id)
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
    err_prob   = float(sys.argv[7])
    code_name  = str(sys.argv[8])

    def square(a):
        return a**2
    
    vsquare=np.vectorize(square)
    bdry_name='rotated'
    
    error_probability= err_prob

    perm_rates=[0,0,0,0,0,0]

    from itertools import cycle
    
    code = RotatedPlanarCode(*(code_size,code_size))
    decoder = _rotatedplanarmpsdecoder_defp.RotatedPlanarMPSDecoder_defp(chi=chi_val)
    layout='rotated'
    if code_name=='rotXY':
        bias_str='Y'
    else:
        bias_str='Z'

    error_model = BiasedDepolarizingErrorModel(bias,bias_str)
    # print run parameters
    print('code:',code.label )
    print('Error model:',error_model.label)
    print('number of realizations:',num_realiz)
    print('Decoder:',decoder.label)
    print('Error probabilities:',error_probability)
    print('Maximum runs:',max_runs)
    
    results = TNDresult_random(code,decoder,error_model,max_runs,perm_rates,error_probability,code_name,layout,run_id,num_realiz)
    
    output = {}
    output['code'] = code_name
    output['error_probabilities'] = err_prob
    output['bias'] = bias
    output['maxruns'] = max_runs
    output['layout'] = layout
    output['chi'] = chi_val
    output['nrod'] = num_realiz
    output['bias_str'] = bias_str
    output['L'] =code_size 
    output["success_list"]  =             [[results[k][j]["success_list"] for j in range(len(results[k]))] for k in range(len(results))]
    output["coset_ps_list"] =             [[results[k][j]["coset_ps_list"] for j in range(len(results[k]))] for k in range(len(results))]
    output["logical_commutations_list"] = [[results[k][j]["logical_commutations_list"] for j in range(len(results[k]))] for k in range(len(results))]
    
    outputpath  = 'data/' + code_name + '_L'+str(code_size) + '_bias' + str(bias)+'_chi' + str(chi_val)+'_rate'+str(err_prob)+'_n' + str(run_id)+'.pickle'
    fout = open(outputpath, 'wb')
    pickle.dump(output, fout, pickle.HIGHEST_PROTOCOL)
    fout.close()

