from xtb.ase.calculator import XTB
from ase import Atoms
from ase.geometry import distance
from ase.visualize import view
from ase.io import read as read
from ase.constraints import Hookean
from ase import units
from ase.calculators.qchem import QChem
from ase.build.molecule import molecule

import matplotlib.pyplot as plt
import numpy as np
import os as os

from Zorro import ZoRRo


#%% ZoRRo input parameters

molName = "CH4"
hlName = "GFNFF"
add_beta = True

pore = read("pore.xyz")
mol = molecule(molName)
linear = False
# T = np.arange(6) * 100 + 100
# beta = 1/(units.kB*T)
beta = np.array([30])

#B3LYP oder omega-B97M-V + cc-pVTZ oder cc-pVDZ
# B97-D   ----------------------

# calc_hl = QChem(label='calc/methan',
#             method='LDA',
#             basis='SV',
#             nt = 4)

calc_hl = XTB(method = "GFNFF")
calc_ll = XTB(method = "GFNFF")

# for non linear molecule
coordinate_list = {"full":   [["theta","phi","xi","r","chi","z"],[],[]],
                   "ridge":  [["theta","phi","xi","r","chi"],["z"],[0]]}
ranges = {"full":  np.array([[-np.pi/2,np.pi/2], [0,2*np.pi],[0,2*np.pi/3],[0,3],[np.pi/6,np.pi/2],[0,3]]),
          "ridge": np.array([[-np.pi/2,np.pi/2], [0,2*np.pi],[0,2*np.pi/3],[0,1],[np.pi/6,np.pi/2]])}

# for linear molecule with mirror symmetry perpendicular to high symmetry axis
# coordinate_list = {"full":   [["theta","phi","r","chi","z"],["xi"],[0]],
#                    "ridge":  [["theta","phi","r","chi"],["xi","z"],[0,0]]}
# ranges = {"full":  np.array([[-np.pi/2,np.pi/2], [0,np.pi],[0,3],[np.pi/6,np.pi/2],[0,3]]),
#           "ridge": np.array([[-np.pi/2,np.pi/2], [0,np.pi],[0,1],[np.pi/6,np.pi/2]])}



pos_information = [np.zeros(3),[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]]
relax_hl = {"full": True, "ridge": False}

approx_method = "LJ"
f_grid_nums = 100
approx_level = "ll"

rhs_deg = 7
rhs_maxnum = 16
rhs_tol = 1e-2
rhs_convergence = True
rhs_method = "mc"

Integrator_List = [["l1",5],["mc_is",2**np.arange(10)]]

method_dict = {"proposal_factor": 5,
                "proposal_threshold": 0.00,
                "nested_mc": True,
                "rhs_analytic": False}


#%% ======================= Calculation of Partition Sums =======================
#%% make directory and save temperatures

try:
    os.mkdir(molName)
    print("Directory " , molName ,  " Created ") 
except FileExistsError:
    print("Directory " , molName ,  " already exists")
    
try:
    os.mkdir(molName + "/" + hlName)
    print("Directory " , molName + "/" + hlName ,  " Created ") 
except FileExistsError:
    print("Directory " , molName + "/" + hlName ,  " already exists")
    
target_dir = molName + "/" + hlName + "/"


if os.path.exists(target_dir + "beta.npy") and add_beta:
    all_beta = np.load(target_dir + "beta.npy")
    start_index = len(all_beta)
    all_beta = np.concatenate((all_beta,beta),axis=0)
    np.save(target_dir + "beta",all_beta)
else:
    start_index = 0
    np.save(target_dir + "beta",beta)


#%% beta sweep 

Zorro = {"full": None, "ridge": None}
Int_data = {"full": None, "ridge": None}

for i_beta in range(len(beta)):
    print("============= calculation for beta = ",beta[i_beta],"=============")
    
    for key in ["full","ridge"]:
        print("------- calculating " + key + " partition sum! -------")
        
        
#%% Initialize

        Zorro[key] = ZoRRo(ranges = ranges[key],
                           interpolation_ranges = np.array([[-1,1]]*len(coordinate_list[key][0])),
                           coordinate_list = coordinate_list[key], 
                           ll = calc_ll, hl = calc_hl,
                           beta = beta[i_beta],
                           pore = pore, mol = mol, 
                           relax_hl = relax_hl[key],
                           pos_information = pos_information,
                           linear = linear)
        
        if relax_hl[key]: # take relaxed pore for ridge calculation
            pore = Zorro[key].mol_hl.Pore[0:Zorro[key].mol_hl.atom_num_pore]


#%% Approx PES

        Zorro[key].Approx_PES(f_grid_nums,
                              method = approx_method,
                              level = approx_level)
        
        Q_calc, Q_int = Zorro[key].Predictor_ll.check_interpol(quantity = "E")
        Q_calc_w, Q_int_w = Zorro[key].Predictor_ll.check_interpol_w(Zorro[key].mol_ll, Zorro[key].Predictor_ll, 50, Zorro[key].ranges, Zorro[key].interpolation_ranges, Zorro[key].beta)


#%% get RHS
        
        if "l1" in [IL[0] for IL in Integrator_List]:
            RHS_list_full = Zorro[key].get_RHS_for_Integrator(numeric = True,
                                                              deg = rhs_deg,
                                                              maxnum = rhs_maxnum,
                                                              tol = rhs_tol,
                                                              convergence = rhs_convergence,
                                                              method = rhs_method)


#%% get integrator

        for method, deg in Integrator_List:
            Zorro[key].get_Integrator(method = method, 
                                      deg = deg, 
                                      sequence = True, 
                                      method_dict=method_dict)


#%% Integrate

        I = []
        F = []
        NP = []
        
        for i in range(len(Zorro[key].Integrator_List)):
            tmp_I, tmp_F, tmp_NP = Zorro[key].get_Integral(integrator=i,level='hl',eval_index='conv',tol=1e-1,
                                                      approx_quantity="E",use_nesting=True,convergence=None)
            NP.append( tmp_NP )
            I.append( tmp_I )
            F.append( tmp_F )
            
            
        Int_data[key] = {"NP": NP, "I": I, "F": F, "beta": Zorro[key].beta, "Q_type": Zorro[key].Integrator_List}
        np.save(target_dir + key + "_dict_" + str(i_beta+start_index), Int_data[key])
        

#%% ============================= Reaction rate calculation ============================================

# Ridge_dict = np.load(target_dir + "ridge_dict_0.npy",allow_pickle = True).item()
# Full_dict = np.load(target_dir + "full_dict_0.npy",allow_pickle = True).item()

# R,Z = Ridge_dict["I"][0][-1],Full_dict["I"][0][-1]

# k = R/Z * 1/np.sqrt(2*np.pi*beta) * units.s