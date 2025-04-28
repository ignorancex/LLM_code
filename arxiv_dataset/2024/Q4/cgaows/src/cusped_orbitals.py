import numpy as np
import basis_set_info
import integrand_at_rtp
from scipy.integrate import dblquad, fixed_quad
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.linalg import eig
import sys
import time
from multiprocessing import Pool, cpu_count
import os
import pickle
from numpy import linalg

np.set_printoptions(precision=12, linewidth=np.inf, threshold=np.inf)

# get nuclear positions
def get_mol_xyz(dict_key):
  nuc_lines = dict_key.split('\n')
  nuc_lines = [line.strip() for line in nuc_lines if line.strip()]
  all_nuc_xyz = np.array([list(map(float, line.split())) for line in nuc_lines])
  return all_nuc_xyz

# run HF with pyscf
def pyscf_result(z_array, all_nuc_xyz, basis_set):
  """
  Input:
    z_array = options['Z']
    all_nuc_xyz = options['nuclei'] in [N,3] array
    basis_set = options['basis_type']

  Output:
    s = AOxAO overlap matrix   
    pyscf_1e_energy = AOxAO 1 electron energy
    mocoeff = occupied mo coeffs ---> C matrix
  """
  from pyscf import gto, scf, lo
  nuc_dict = {
  1: 'H',
  2: 'He',
  3: 'Li',
  4: 'Be',
  5: 'B',
  6: 'C',
  7: 'N',
  8: 'O',
  9: 'F',
  10: 'Ne',
  11: 'Na',
  12: 'Mg',
  13: 'Al',
  14: 'Si',
  15: 'P',
  16: 'S',
  17: 'Cl',
  18: 'Ar',
  }

  geom = []
  for i, val in enumerate(z_array.reshape(-1)):
      atom_key = nuc_dict[int(val)]
      nuc_xyz = tuple(all_nuc_xyz[i])
      geom.append([atom_key, nuc_xyz])
 
  #print()
  #print("From Pyscf with basis set: ", basis_set)
  mol = gto.Mole()
  mol.build(
  unit = 'Bohr',
  atom = geom,
  basis = basis_set,
  )
  #print(mol.ao_labels())
  
  my_hf = scf.RHF(mol)
  e_mol = my_hf.kernel()
  
  # Overlap, kinetic, nuclear attraction
  s = mol.intor('int1e_ovlp')
  t = mol.intor('int1e_kin')
  v = mol.intor('int1e_nuc')

  occ_orb_coeff = my_hf.mo_coeff[:,my_hf.mo_occ > 0.]

  pm_loc_orb = lo.PM(mol, mo_coeff=occ_orb_coeff) #, pop_method='mulliken') # Pipek-Mezey
  pm_loc_orb.pop_method = 'mulliken'
  pm_loc_orb.init_guess = None

  fb_loc_orb = lo.Boys(mol, mo_coeff=occ_orb_coeff) #, init_guess=None)	 # Foster Boys
  er_loc_orb = lo.ER(mol, occ_orb_coeff)	 # Edmiston Ruedenberg
  chol_loc_orb = lo.cholesky_mos(occ_orb_coeff)	 # Cholesky

  PM_loc = np.asarray(pm_loc_orb.kernel(occ_orb_coeff), order='C')
  PM_loc[np.abs(PM_loc) < 0.2] = 0.0
  #print("returning FB occ_orb_coeff")
  return np.asarray(fb_loc_orb.kernel(occ_orb_coeff), order='C')

def set_basis_orb_info(basis, basis_orb_type, basis_centers, Z_array):
  """ Determines the basis specific parmeters for each AO:

  returns:

       rc_matrix --- nuc x AO matrix of cusp radii for each AO (nuc x AOs)
  orth_orb_array --- array with non-zero values indicate that orb will be orthogonalized against it's core, value = (core index + 1)

  """
  num_AOs = int(len(basis_orb_type))
  rc_matrixv2 = np.zeros((len(Z_array), num_AOs))
  orth_orb_array = np.zeros(num_AOs, dtype=int)

  add_count = count = 0 

  nonH_nuc_ind = np.where(Z_array > 1.0)[0] # nuc ind of Z>1

  for i, Z in enumerate(Z_array.reshape(-1)):   # loop through nuclei - and each orb for columns
    on_center_ind = np.where(basis_centers == i)[0]

    if basis == str('sto-3g') or basis == str('STO-3G'):
      if Z == 1.0:                # hydrogen
        add_count = count + 1

        orth_orb_array[count+0] = count+1      # 1s (3G)
      elif Z in {3., 4., 5., 6., 7., 8., 9., 10.}:              # carbon
        add_count = count + 5

        orth_orb_array[count+1] = count+1      # orthogonalize the 2s orb against 1s core
      else:
        raise RuntimeError("STO-3G: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")

    elif basis == str('6-31G') or basis == str('6-31g'):
      if Z == 1.0:                # hydrogen    1s(3G), 2s_orth(1G)
        add_count = count + 2
        
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind :
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            print("bf type : ", bf_ind_type,flush=True)
            if bf_ind_type in {0,1,}:  # s-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
       
        orth_orb_array[count+1] = count+1      # orthogonalize 1s (1G) against 1s (3G)
      elif Z > 1.:              # carbon
        add_count = count + 9

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind :
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            if bf_ind_type in {0,1,}:  # s-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.2
            elif bf_ind_type in {2,3,4}: # p-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0
       
        orth_orb_array[count+1:count+3] = count+1      # orthogonalize the s orbs against 1s core
      else:
        raise RuntimeError("6-31G: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")
    elif basis == str('6-31G*') or basis == str('6-31g*'):

      if Z == 1.0:                # hydrogen
        add_count = count + 2
        
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind:
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            if bf_ind_type in {0,1,}:  # s-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")

        orth_orb_array[count+1] = count+1      # orthogonalize 1s (1G) against 1s (3G)
      elif Z in np.arange(2., 11.):              # carbon
        add_count = count + 14

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind:
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            if bf_ind_type in {0,1,}:  # s-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.2
            elif bf_ind_type in {2,3,4}: # p-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0

        orth_orb_array[count+1:count+3] = count+1      # orthogonalize the s orbs against 1s core
      elif Z in np.arange(11., 19.): # third row 
        add_count = count + 18

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        rc_matrixv2[i, :] = 0.05
        for ind in nonH_nuc_ind:
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            if bf_ind_type in {0,1,}:  # s-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.05
            elif bf_ind_type in {2,3,4}: # p-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0

        orth_orb_array[count+1:count+4] = count+1      # orthogonalize the s orbs against 1s core
      else:
        raise RuntimeError("6-31G*: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")
    else:		# no defined basis, all radii default to zero
      break
    count = add_count
  
  return rc_matrixv2, orth_orb_array

def get_proj_mat(basis_centers, orth_orb_array, all_nuc_xyz, z_array, basis_orb_type, basis_exp, basis_coeff):
  """ Calculate the AO x AO matrix identifying overlap between core AO (rows) and AO of the Gram-Schmidt orthogonalized (column) orbs

  params:
    basis_centers - norb array indicating the nucleus (by index) each ao is centered on
   orth_orb_array - norb array with non-0 on of ao indices being orthogonalized, value = index+1 of orb to orthogonalize against 
      all_nuc_xyz - xyz coords 
          z_array - num_nuv array of atom charges 
   basis_orb_type - norb array of basis type (ie. 1s = 0, 2s = 1, 2px = 2, 2py = 3, ...)
        basis_exp - basis set specific exponents of the primitive for each nuc
      basis_coeff - basis set specific coefficients of the primitive for each nuc

  return:
      proj_ao_mat - ao x ao array, non-0 only for overlap between Ns and it's 1s core
                    this results in most of the matrix being 0s, could make more efficient

  """

  proj_ao_mat = np.zeros((len(basis_centers), len(basis_centers)))

  for nuc_ind, Z in enumerate(z_array):
    nuc_xyz = np.reshape(all_nuc_xyz[nuc_ind], [1,3])

    for ao_ind, ao_type in enumerate(basis_orb_type):
      basis_center_ind = basis_centers[ao_ind]                   # index of nucleus this AO is centered on
      basis_center_xyz = np.reshape(all_nuc_xyz[basis_center_ind], [1, 3])  # coords of nucleus this AO is centered on
      core_ind = orth_orb_array[ao_ind]
      alpha = np.reshape(basis_exp[ao_ind], [-1,1])
      d = np.reshape(basis_coeff[ao_ind], [-1,1])

      if core_ind > 0:
        core_ind = core_ind - 1
        alpha_core = np.reshape(basis_exp[core_ind], [-1,1])
        d_core = np.reshape(basis_coeff[core_ind], [-1,1])

        if proj_ao_mat[ao_ind, core_ind] == 0:
          proj = get_proj_2_1(basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d)
          proj_ao_mat[ao_ind, core_ind] = proj

  return proj_ao_mat

def get_proj_2_1(basis_center, integration_center, alpha_1, d_1, alpha_2, d_2):
    """ This function was written by Sonja Bumann

      orthogonalize AO2 against AO1: 	 proj = < AO2 | AO1 > 
       				        	                      _____________

       				        	                      < AO1 | AO1 > 
    """
    proj_2_1 = 0.0
    norm_1 = 0.0

    # because trailing zeros to match man nG length, find max n between basis func 1 and 2
    basis_1_len = np.trim_zeros(alpha_1.reshape(-1)).size
    basis_2_len = np.trim_zeros(alpha_2.reshape(-1)).size
    nG = max([basis_1_len, basis_2_len]) 
    for i in range(nG):
      for j in range(nG):
          A01_g_overlap = ( 2.0 * alpha_1[i] / np.pi )**0.75 * ( 2.0 * alpha_1[j] / np.pi )**0.75 * d_1[i] * d_1[j] * ( np.pi / (alpha_1[i] + alpha_1[j]) )**1.5     # <1s|1s>
          A01_A02_g_overlap = ( 2.0 * alpha_1[i] / np.pi )**0.75 * ( 2.0 * alpha_2[j] / np.pi )**0.75 * d_1[i] * d_2[j] * ( np.pi / (alpha_1[i] + alpha_2[j]) )**1.5  # <1s|2s>
          proj_2_1 += A01_A02_g_overlap
          norm_1 += A01_g_overlap

    proj_2_1 = proj_2_1 / norm_1
    return proj_2_1

def orth_transform(orth, proj, nnuc, basis_centers, num_bf):
  """ change of basis matrix for orthogonalized orbitals in cusping scheme

  In simple_vmc.py, initializing the calculation will require change of basis in C (orbital coeffs)
    and in the evaluation of the orbitals during sampling (X) 
    AKA:
              X' C' = (X B) (B_inv C)  <- cusp ready basis

  params:
             orth - norb array with non-0 on of ao indices being orthogonalized
             proj - ao x ao array, non-0 only for overlap between Ns and it's 1s core
             nnuc - number of nuclei 
    basis_centers - norb array indicating the nucleus (by index) each ao is centered on 
           num_bf - number of AOs 
  return 
                B - change of basis matrix to orth orbs 
  """
  B = np.identity(num_bf)    # AOs x new AOs

  for i in range(0, nnuc):    # loop through nuclei
    on_nuc_ind = np.where(basis_centers == i)[0]  # indicies of current nuc centered orb
    orth_nuc_ind = np.where(orth[on_nuc_ind] != 0)[0]  # indices of orbs to orthogonalize against core orbital (in all ao space)
    
    for j in [x + on_nuc_ind[0] for x in orth_nuc_ind]:    # loop through indices orthogonalized AOs centered on given nucleus
      core_ind = int(orth[j] - 1)	# in full AO list
      B[core_ind, j] = - proj[j, core_ind]

  return B 

def norm_each_MO(C_mat):
  """ return normalized MO matrix """
  # normalize each max coeff in MO to 1 
  max_MO_vals = np.max(np.abs(C_mat), axis=0)

  norm_C_mat = C_mat / max_MO_vals
  return norm_C_mat

def get_a0_matrix(options):

  # molecular info
  z_array = options['Z'].flatten()
  
  num_nuc = len(z_array) 
  num_ao = options['nbf']

  final_a0_matrix = np.zeros((num_nuc, num_ao)) 
  
  # find unique nuclei (ie, 1 and 6), calc max orb for each in dict 
  unique_nuc = []
  max_orb_info = {}
  run_pairs = []

  for ind, Z in enumerate(z_array):
     if Z not in unique_nuc:    # if new atom - calculate max value corresponding to all it's basis functions, store in dict
        unique_nuc.append(Z)
        # ind of orbitals for 1st appearance of this nuclei
        max_ind = np.where(options["basis_centers"].astype(int) == ind)[0]
        #print(Z, max_ind)
        ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, max_orb_info[int(Z)] = gaussian_info_and_eval(ind, max_ind, options, get_bf='max')

  nuc_list = np.arange(num_nuc)     # all nuc
  orb_list = np.arange(num_ao)      # all AOs

  # determine cusp elements to opt
  for i in nuc_list:
    for j in orb_list:   
      ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, bf_vals = gaussian_info_and_eval(i, j, options, get_bf='nuc')
      
      for val in bf_vals:
        if np.isnan(val) == False:

          Z = int(z_array[basis_center_ind[0]]) 
          ao_max = max_orb_info[Z][ao_type[0]]

          ao_on_nuc = np.abs(val/ao_max)

          if (ao_on_nuc > 10e-16) and (r_cusp > 0.0):
            final_a0_matrix[i][j] = 1.0

  return final_a0_matrix

def gaussian_info_and_eval(nuc_ind, ao_ind, options, get_bf=None): #, add_to_out=None):
  """ orbital info and evaluation (if get_bf != None, should be string indicating where to eval the bf at or array with value to eval at) 
      returns shifted oth_orb_array"""
  ij = (nuc_ind, ao_ind)

  nuc_ind = np.atleast_1d(nuc_ind)
  ao_ind = np.atleast_1d(ao_ind)

  all_nuc_xyz = get_mol_xyz(options['nuclei'])

  nuc_xyz = (all_nuc_xyz[nuc_ind]).reshape(-1,3)

  ao_type = (options['basis_orb_type'][ao_ind]).astype(int).reshape(-1)

  basis_center_ind = (options["basis_centers"][ao_ind]).astype(int).reshape(-1)

  on_center = (basis_center_ind == nuc_ind)
  
  basis_center_xyz = (all_nuc_xyz[basis_center_ind]).reshape(-1,3)

  alpha = (options["basis_exp"][ao_ind]).reshape(-1, options['ng'])

  d = (options["basis_coeff"][ao_ind]).reshape(-1, options['ng'])

  r_cusp = options['cusp_radii_mat'][nuc_ind,ao_ind]

  orth_orb_shifted = np.atleast_1d(options['orth_orb_array'][ao_ind]).astype(int)
  orth_orb_bool = np.atleast_1d(orth_orb_shifted != 0)

  if True in orth_orb_bool:
    core_true = np.where(orth_orb_bool == True)[0]
    core_ind = orth_orb_shifted[core_true] - 1

    alpha_core = (options["basis_exp"][core_ind]).reshape(-1, options['ng'])

    d_core = (options["basis_coeff"][core_ind]).reshape(-1, options['ng'])
    
    proj = (options["proj_mat"][ao_ind[core_true],core_ind]).reshape(-1)
  else:
    core_ind = alpha_core = d_core = proj = 0.0

  if get_bf != None:
    xyz_eval = np.zeros((1,3))
    if type(get_bf) == type(np.array([])):
      get_max = False
      xyz_eval = get_bf 
    elif get_bf == 'nuc':    # evaluate orbs over nuclei input
      get_max = False
      xyz_eval = nuc_xyz
    elif get_bf == 'max':    # find max of orbs
      get_max = True 
      xyz_eval = basis_center_xyz
    else:
      raise RuntimeError('get_bf input in gaussian_r_val is not valid')

    # returns bf value over indicated nuclei
    bf_vals = gaussian_r_val(xyz_eval, ao_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj, get_max)
    return ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, bf_vals
  else:
    return ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj

def r_from_xyz(xyz, origin=np.zeros((1,3))):
  """ input xyz must be [m,3] or [3,] 
  return matrix of radial distances [m,N], N=# of nuclei, m = # xyz
	if all_diff is true output the r between each xyz and origin input, must be same length """
  xyz = np.reshape(xyz, [-1,3])         # [m,3]
  origin = np.reshape(origin, [-1,3])[:,:,np.newaxis]    # [N,3,1]

  diff_mat = np.transpose(np.sqrt(np.sum((origin - xyz.T)**2,axis=1)))   # (sum([N,3,1] - [3,m])_3 = [N,m]).T = [m,N]
  return diff_mat   # [m,N]

def STOnG_s_eval(r, alpha, d):
  """ evaluate s-orb at given r, 
  r: [m,]/[m,1]/[1,m] and alpha/d: [n,]/[n,1]/[1,n] 
  """
  N = np.reshape(d * (2 * alpha / np.pi )**(3/4),[-1,1])                # [n,1]
  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  orb_val = np.sum(N * np.exp(-exp_val), axis=0).reshape(-1)

  return orb_val  # [m, ]

def STOnG_orth_s_eval(r, alpha_1, d_1, alpha_2, d_2, proj_2_1):
  """   input: _2 = orth_orb, _1 = core 
        return: 2 - proj_1(2) * 1 """
  orth_2_1 = STOnG_s_eval(r, alpha_2, d_2) - proj_2_1 * STOnG_s_eval(r, alpha_1, d_1)
  return orth_2_1  # [m, ]

def STOnG_p_eval(xyz, xyz_o, alpha, d, ax, r_for_plot=None):
  """ eval from xyz to xyz_o (basis_center)
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
        delta_BA = [k_B0 - k_BA] for (k=x,y,z) 
            r: [m,]/[m,1]/[1,m], xyz_o: [m or 1,3]; xyz: [3,]/[3,m]/[m,3]; alpha/d: [n,]/[n,1]/[1,n] 
  """
  if r_for_plot is not None:
    # replace current xyz
    r_vec = np.zeros(3)
    r_vec[ax] = r_for_plot
    xyz = xyz_o + r_vec

  N = np.reshape(2.0 * np.sqrt(alpha) * d * (2.0 * alpha / np.pi )**(3/4),[-1,1])              # [n,1]
  r = r_from_xyz(xyz, xyz_o)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r) if r.shape[1] > 1 else r
  xyz_BA = np.reshape(xyz-xyz_o, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 
  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  delta = xyz_BA[:,ax].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * delta                # [m, ] * [m, ]
  return ret_val  

def STOnG_d_eval(xyz, xyz_o, alpha, d, ax, r_for_plot=None):
  """ eval from xyz to xyz_o (basis_center)
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
        delta_BA = [k_B0 - k_BA] for (k=x,y,z) 
            r: [m,]/[m,1]/[1,m], xyz_o: [m or 1,3]; xyz: [3,]/[3,m]/[m,3]; alpha/d: [n,]/[n,1]/[1,n] 
  """
  if r_for_plot is not None:
    # replace current xyz
    r_vec = np.zeros(3)
    r_vec[0] = r_for_plot
    xyz = xyz_o + r_vec

  N = np.reshape((2048.0 * alpha**7.0 / np.pi**3.0 )**(1/4) * d,[-1,1])              # [n,1]
  r = r_from_xyz(xyz, xyz_o)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r) if r.shape[1] > 1 else r
  xyz_BA = np.reshape(xyz-xyz_o, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 
  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  x = xyz_BA[:,0].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  y = xyz_BA[:,1].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  z = xyz_BA[:,2].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  if ax == 5:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * x * y                # [m, ] * [m, ]
  elif ax == 6:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * y * z                # [m, ] * [m, ]
  elif ax == 7:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * 144.0**-0.25 * (3.0 * z * z - x * x - y * y - z * z)                # [m, ] * [m, ]
  elif ax == 8:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * x * z                # [m, ] * [m, ]
  else:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * (x * x - y * y) * 0.5             # [m, ] * [m, ]
  return ret_val 

def gaussian_r_val(eval_xyz, orb_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj_2_1, get_max=False): # evaluate radial portion of gaussian orbital at some r from xyz_eval to basis_center_xyz 
  """ eval gaussian orb at r: from  eval_xyz (relative to basis_center_xyz) [m,3], all else either length 1 or m
    input shifted orth_orb array """

  r_mat = r_from_xyz(eval_xyz, basis_center_xyz)      # xyz_Bmol - xyz_Amol --> r_BA, [m,m] 
  r = np.diagonal(r_mat) if r_mat.shape[1] > 1 else r_mat
  ret_val = np.zeros_like(r)
  theta = phi = 0.0
  total_orth_orb = 0 # index for orth elements

  # need to adapt incase only one r with same params for everything else, should be able to pipe in array of r

  for ind, r_val in enumerate(r):
    nuc_xyz_now = nuc_xyz[ind] if len(nuc_xyz) > 1 else nuc_xyz

    # s-orbital
    if orb_type[ind] < 2:

      # orthogonalized valence s orbital
      if orth_orb_shifted[ind] != 0:
        orth_against = orth_orb_shifted[ind] - 1

        ret_val[ind] = STOnG_orth_s_eval(r_val, alpha_core[total_orth_orb], d_core[total_orth_orb], alpha[ind], d[ind], proj_2_1[total_orth_orb])[0]
        total_orth_orb += 1

      # non-orthogonalized s-orb
      else: 
        ret_val[ind] = STOnG_s_eval(r_val, alpha[ind], d[ind])[0]

    # p-orbital
    elif orb_type[ind] in {2, 3, 4}:
      if get_max == True:   # opt to find max val
          func = lambda r_max: - np.abs(STOnG_p_eval(basis_center_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind]-2, r_max))
          opt_val = minimize(func, x0=0.1, method='BFGS')
          ret_val[ind] = opt_val.fun
      
      elif on_center[ind] == False: # code taken from plot_orb.py plot_thru_nuc()
          ret_val[ind] = STOnG_p_eval(eval_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind]-2)

      # centered p-orb
      else:
        ret_val[ind] = np.nan

    # d-orbital
    elif orb_type[ind] in {5, 6, 7, 8, 9}:
      if get_max == True:   # opt to find max val
          func = lambda r_max: - np.abs(STOnG_d_eval(basis_center_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind], r_max))
          opt_val = minimize(func, x0=0.1, method='BFGS')
          ret_val[ind] = opt_val.fun
      
      elif on_center[ind] == False: # code taken from plot_orb.py plot_thru_nuc()
          ret_val[ind] = STOnG_d_eval(eval_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind])

      # centered d-orb
      else:
        ret_val[ind] = np.nan
    else:
      raise RuntimeError("orbital category not classified in gaussian_r_val() --- orb_type = ", orb_type[ind])

  return ret_val 

def cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, z, int_type):
  """ one electron energy element only within the cusp of nth-slater func
      returns H_nm = <Pn|H|Pm> and S_nm = <Pn|Pm>; P_n = nth order slater  """
 
  zeta = z

  xi_xyz = (nuc_xyz - basis_center_xyz).reshape(-1)

  # s - orbital
  if orb_type < 2:

    # orthogonalized s orbital
    if orth_orb_shifted > 0: # apply to 3g and 1g valence orbs
      alpha_core = alpha_core.reshape(-1,1)
      d_core = d_core.reshape(-1,1)
      proj_2_1 = proj_2_1[0]
      
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, orthogonalized=True, H=False)

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, orthogonalized=True)

    else:
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, H=False)

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type)

  # p-orbital
  elif orb_type == 2 or orb_type == 3 or orb_type == 4:
    # s cusp on p orb tail
    if on_center == False:
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, H=False)

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type)
    
    else:
      zeta = z/2
      raise RuntimeError("calculating one_elec_energy() of atom centered p orbital to cusp, no p-cusps on p orb built in the code, a0/ao_ind/nuc_ind/orb_type/on_center: ", a0, ao_ind, nuc_ind, orb_type, on_center)
  
  # d-orbitals
  elif orb_type == 5 or orb_type == 6 or orb_type == 7 or orb_type == 8 or orb_type == 9:
    if on_center == False:
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, H=False)
      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type)
    else:
      zeta = z/2
      raise RuntimeError("calculating one_elec_energy() of atom centered p orbital to cusp, no p-cusps on p orb built in the code, a0/ao_ind/nuc_ind/orb_type/on_center: ", a0, ao_ind, nuc_ind, orb_type, on_center)
  else:
    raise RuntimeError("orbital category not classified in one_elec_energy() --- orb_type = ", orb_type)

  return H_rc, S_rc
 
def cusp_coeff_vec(ij, all_nuc_xyz, nuc_xyz, orb_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, core_ind, alpha_core, d_core, proj_2_1, Z, a0, order_n_list):
  """ input: (the output of gaussian_info_and_eval for a single {nuc/orb and a0!=0} with (Z, a0, orb_n_list) appended)
  this func: build matrices out of order_n_list = list of i, i=polynomial of slater, diag to get {q_i} per a0 """
  alpha = alpha.reshape(-1,1)
  d = d.reshape(-1,1)
  nuc_ind = ij[0] 
  ao_ind  = ij[1] 
  # build eigenvalue prob in P_i basis, diag to get q_i coeffs
  H_mat = np.zeros((len(order_n_list), len(order_n_list)))
  S_mat = np.zeros((len(order_n_list), len(order_n_list)))

  # < P_i | H | P_j > and < P_i | P_j > - loop through elements for now

  #######################################################  
  # Get main block of elements: < bQr^n | H | bQr^m >
  for count_n, n in enumerate(order_n_list):
    for count_m, m in enumerate(order_n_list):
      H_mat[count_n,count_m], S_mat[count_n,count_m] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 0)

  # Get first row and column corresponding to elements < (1-b)X | H | bQr^m >
  H_row_0 = np.zeros((len(order_n_list)+1 , 1))
  S_row_0 = np.zeros((len(order_n_list)+1, 1))

  # Except first entry is < (1-b)X | H | (1-b)X >
  H_row_0[0][0], S_row_0[0][0] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 2)
    
  for count_m, m in enumerate(order_n_list):
    H_row_0[count_m+1,0], S_row_0[count_m+1,0] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 1)

  H_mat = np.concatenate((H_row_0[1:][:], H_mat), axis=1)
  H_mat = np.concatenate((H_row_0.reshape(1,-1), H_mat), axis=0)
  S_mat = np.concatenate((S_row_0[1:][:], S_mat), axis=1)
  S_mat = np.concatenate((S_row_0.reshape(1,-1), S_mat), axis=0)

  # NEXT solve eigenvalue prob
  evals, evec_mat = eig(H_mat, S_mat)

  idx = evals.argsort()	# indices that sort array in acsending order
  evec_best = evec_mat[:,idx[0]]		# resort corrresponding eigenvectors
    
  # test for linear dependence
  evals_S, evec_S = np.linalg.eig(S_mat)
  #print("Evals of slater_poly_plus_ratio mat", evals_S, flush=True) 
 
  # normalize st vec sums to 1
  coeff_vec = evec_best / evec_best[0]
  coeff_vec = coeff_vec[1:]

  return nuc_ind, ao_ind, coeff_vec


def get_cusp_coeff_matrix(options): # or input options which ontains cusp_a0
  """ calculate all 7 primitives needed to defind the slater expanded cusping function for each nuc/AO pair 

  Note: order_n_list = list of bth order terms to add to slater cusp

  return [ n, num_orb, num_nuc, ] of Q_i coeffs for each cusp (ao over nuc)
  """
  order_n_list = options['order_n_list']
  # reassign if on savio or work computer
  num_cores = cpu_count()
  a0_ind_to_opt = np.argwhere(np.abs(options["cusp_a0"]) > 0.)
  cusp_coeff_mat = np.zeros((options["cusp_a0"].shape[0], options["cusp_a0"].shape[1], len(order_n_list) ))  # [ num_qi_coeffs, num_orb, num_nuc, ]
  input_info = [gaussian_info_and_eval(*ij, options) + (options['Z'][(ij[0])], options["cusp_a0"][(ij[0]),(ij[1])], order_n_list) for ij in a0_ind_to_opt]  # each element of list is info coresponding to a single orb/nuc pair
 
  with Pool(num_cores) as p:
    all_coeff_vecs = p.starmap(cusp_coeff_vec, input_info) # returns orb_ind, nuc_ind, normed_coeff_vec

  for val in all_coeff_vecs:
    cusp_coeff_mat[val[0], val[1], :] = val[-1] # assign coeff vector to corresponding a0 nuc/orb/indice

  return cusp_coeff_mat


def get_cusped_orb_info(options, input_path, save_pkl=False):
  """ Calculates basis and molecule specific cusp parameters
       options - python dictionary containing input information
    input_path - path to save molecule specific cusp parameters 
      save_pkl - if True save option to pkl file ONLY (for testing),
                 else save to individual *.txt files
  """

  print("Path to save cusp parameters:", input_path)

  # update dictionary with specific basis set information 
  options = basis_set_info.make_full_input(options) 

  # get cusp radii matrix and array of orbitals to orthogonalize based on basis set info
  options['cusp_radii_mat'], options['orth_orb_array'] = set_basis_orb_info(options['basis_type'], options['basis_orb_type'], options['basis_centers'], options['Z']) 

  # get nuclear positions
  all_nuc_xyz = get_mol_xyz(options['nuclei'])     
  
  # get projection matrix for orthogonalized orbitals
  options['proj_mat'] = get_proj_mat(options['basis_centers'].astype(int), options['orth_orb_array'], all_nuc_xyz, options['Z'], options['basis_orb_type'], options['basis_exp'], options['basis_coeff']) 
  
  # get mocoeff matrix from pyscf
  options["mocoeff"] = pyscf_result(options["Z"], all_nuc_xyz, options["basis_type"]) 
  
  # transform the mocoeff matrix to account for orthogonalized orbitals on the c++ side
  B_mat = orth_transform(options["orth_orb_array"], options['proj_mat'], len(options['Z']), options["basis_centers"].astype(int), options["nbf"])
  B_inv = np.linalg.inv(B_mat)
  options["mocoeff"] = B_inv @ options["mocoeff"]
  options["mocoeff"] = norm_each_MO(options["mocoeff"])
 
  print("Use this mocoeff matrix for your VMC code:\n", options["mocoeff"], flush=True)
  
  # get a0 matrix
  options["cusp_a0"] = get_a0_matrix(options)
  
  # set n_vec
  options["order_n_list"] = np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

  # get cusp coeff matrix
  options["cusp_coeff_matrix"] = get_cusp_coeff_matrix(options)

  # Save the dictionary to a pickle file for testing purposes
  if save_pkl == True:
    file_path = input_path+'/test_cusp1.pkl' 
    with open(file_path, 'wb') as file_name:
      pickle.dump(options, file_name)
    return 0
    
  # save all arrays to text files 

  # number of orbitals
  np.savetxt(input_path+'no.txt', [options["nbf"]], fmt='%d')

  # number of electrons
  np.savetxt(input_path+'ne.txt', [options["num_elec"]/2], fmt='%d')

  # charges of nuclei
  np.savetxt(input_path+'Z.txt', options["Z"])

  # number of nuclei
  np.savetxt(input_path+'nn.txt', [len(options["Z"])], fmt='%d')

  # basis function centers
  np.savetxt(input_path+'bf_cen.txt', options["basis_centers"], fmt='%d')

  # maximum number of gaussian primitives per basis function
  np.savetxt(input_path+'ng.txt', [options["ng"]], fmt='%d')

  # basis function types
  np.savetxt(input_path+'bf_type.txt', options["basis_orb_type"], fmt='%d')

  # basis function exponents
  np.savetxt(input_path+'bf_exp.txt', options["basis_exp"].transpose())

  # basis function coefficients
  np.savetxt(input_path+'bf_coeff.txt', options["basis_coeff"].transpose())

  # cusp radii mat
  np.savetxt(input_path+'cusp_radii_mat.txt', options["cusp_radii_mat"].transpose())
 
  # cusp a0 mat
  np.savetxt(input_path+'cusp_a0_mat.txt', options["cusp_a0"].transpose())

  # orth orb array
  np.savetxt(input_path+'orth_orb.txt', options["orth_orb_array"], fmt='%d')

  # projection matrix
  np.savetxt(input_path+'proj_mat.txt', options["proj_mat"].transpose())

  # cusp coefficient tensor
  np.savetxt(input_path+'cusp_coeff_mat.txt', options["cusp_coeff_matrix"].reshape((options["cusp_coeff_matrix"].shape[0]*options["cusp_coeff_matrix"].shape[1]), options["cusp_coeff_matrix"].shape[2]))
  
  # vector of M powers
  np.savetxt(input_path+'n_vec.txt', options["order_n_list"].transpose())

  print("Done!", flush=True)
 
  return 0


