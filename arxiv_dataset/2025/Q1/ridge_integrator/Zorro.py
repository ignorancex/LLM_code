from xtb.ase.calculator import XTB
from ase import Atoms
from ase.optimize import QuasiNewton
from ase.geometry import distance
from ase.visualize import view
from ase.io import read as read
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from scipy import optimize 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from scipy import special
import bisect as bisect
import math as math
import numpy.matlib
from ase.optimize import BFGS
from ase.constraints import Hookean
from ase import units
import itertools
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from ase.calculators.qchem import QChem

# from MyClasses.Zorro.Mol import*
# from MyClasses.Zorro.QuadratureRules import*
# from MyClasses.Zorro.Interpol import*
from Mol import *
from QuadratureRules import *
from Interpol import *

from ase.build.molecule import molecule

#%% 

class ZoRRo:
    
    def __init__(self,ranges = None,interpolation_ranges = None,coordinate_list = [["theta","phi","xi","r","chi","z"],[],[]], 
                 hl = None, ll = None, beta = 30, pore = None, mol = None, relax_hl = False,
                 pos_information = [np.zeros(3),[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]], linear = False):  
        
        atom_num_mol = len(mol.get_atomic_numbers())
        atom_num_pore = len(pore.get_atomic_numbers())
        poreMol = pore + mol
        
        masses = poreMol.get_masses()
        mol_masses = poreMol.get_masses()[atom_num_pore:atom_num_pore + atom_num_mol]
        pore_masses = poreMol.get_masses()[0:atom_num_pore]

        self.poreMol_hl = poreMol.copy()
        self.poreMol_hl.set_calculator(hl)
        self.beta = beta
        self.mol_hl = Mol(beta, self.poreMol_hl, atom_num_mol, atom_num_pore, coordinate_list[0], coordinate_list[1], coordinate_list[2],
                          relax = relax_hl, pos_information = pos_information, linear = linear)
        
        self.poreMol_ll = self.mol_hl.Pore.copy()
        self.poreMol_ll.set_calculator(ll)
        self.mol_ll = Mol(beta, self.poreMol_ll, atom_num_mol, atom_num_pore, coordinate_list[0], coordinate_list[1], coordinate_list[2],
                          relax = False, pos_information = pos_information, linear = linear)
        self.mol_ll.mol_reference = self.mol_hl.mol_reference
        
        # ##### TEMPORARY #######
        center = np.array([0,0,0])
        c1 = Hookean(a1=45, a2=center, rt=3.0, k=2.0)
        self.poreMol_hl.set_constraint([c1])
        
        center = np.array([0,0,0])
        c1 = Hookean(a1=45, a2=center, rt=3.0, k=2.0)
        self.poreMol_ll.set_constraint([c1])
        # ###### TEMPORARY OVER#########
        
        self.ranges = ranges
        self.interpolation_ranges = interpolation_ranges
        self.integration_ranges = np.repeat(np.array([[0,1]]),len(coordinate_list[0]),0)
        self.dim = len(coordinate_list[0])
        
        coord_dict = dict()
        if "x" in coordinate_list[0] or "y" in coordinate_list[0]:
            coord_dict["theta"], coord_dict["phi"], coord_dict["xi"], coord_dict["x"], coord_dict["y"], coord_dict["z"] = 0,1,2,3,4,5
            self.cart = True
        else:
            coord_dict["theta"], coord_dict["phi"], coord_dict["xi"], coord_dict["r"], coord_dict["chi"], coord_dict["z"] = 0,1,2,3,4,5
            self.cart = False
        self.grid_coordinates = coordinate_list[0]
        self.fixed_coordinates = coordinate_list[1]
        self.fixed_coordinates_values = coordinate_list[2]
        
        self.Predictor_hl = Int_and_Fit(self.mol_hl,self.ranges,self.interpolation_ranges,self.integration_ranges,self.dim,self.grid_coordinates,self.fixed_coordinates,self.fixed_coordinates_values)
        self.Predictor_ll = None
        
        self.Integrator = []
        self.Integrator_List = []
        self.RHS_for_Integrator = {'numeric': None, 'analytic': None}
        self.numeric_rhs = None
        
        self.full_ranges = {"theta":[0,np.pi], 
                            "phi":[0,2*np.pi], 
                            "xi":[0,2*np.pi], 
                            "r": self.ranges[np.where(np.array(coordinate_list[0]) == "r")[0][0]] if "r" in coordinate_list[0] else [0,1], 
                            "chi":[0,2*np.pi], 
                            "x": self.ranges[np.where(np.array(coordinate_list[0]) == "x")[0][0]] if "x" in coordinate_list[0] else [0,1], 
                            "y": self.ranges[np.where(np.array(coordinate_list[0]) == "y")[0][0]] if "y" in coordinate_list[0] else [0,1], 
                            "z": self.ranges[np.where(np.array(coordinate_list[0]) == "z")[0][0]] if "z" in coordinate_list[0] else [0,1]}
        self.mult_factor = np.prod(np.array([np.diff(self.full_ranges[self.grid_coordinates[i]]) for i in range(len(self.grid_coordinates))])) \
            * np.linalg.det(np.array([pos_information[1]])) #* self.mol_hl.get_max_jac_basis()

        pass 
    
    
    def Approx_PES(self,f_grid_nums = None, A_grid_nums = None, ceta = None, method = None, level = "ll"):
        
        mol = self.mol_ll if level == "ll" else self.mol_hl
        self.Predictor_ll = Int_and_Fit(mol,self.ranges,self.interpolation_ranges,self.integration_ranges,self.dim,self.grid_coordinates,self.fixed_coordinates,self.fixed_coordinates_values)

        if method == "linfit":
            self.rbf_gridpoints, self.rbf_rescaled_gridpoints, _ = self.Predictor_ll.get_grid_nD(ranges,interpolation_ranges,gridnum = A_grid_nums)
            self.gridpoints, self.rescaled_gridpoints, _ = self.Predictor_ll.get_grid_nD(self.ranges, self.interpolation_ranges, f_grid_nums)
            w = self.Predictor_ll.mol.get_w(self.gridpoints)
            self.Predictor_ll.LinFitGaussians(ceta, self.rescaled_gridpoints, self.rbf_rescaled_gridpoints, w, chopping = True)
            self.Kernel = self.Predictor_ll.Return_Integral_Kernel()
            
        elif method == "linfit_iter":
            self.rbf_gridpoints, self.rbf_rescaled_gridpoints, _ = self.Predictor_ll.get_grid_nD(ranges,interpolation_ranges,gridnum = A_grid_nums)
            self.gridpoints, self.rescaled_gridpoints, _ = self.Predictor_ll.get_grid_nD(self.ranges, self.interpolation_ranges, f_grid_nums)
            w = self.Predictor_ll.mol.get_w(self.gridpoints)
            self.Predictor_ll.LinFitGaussians_iter(ceta, self.rescaled_gridpoints, self.rbf_rescaled_gridpoints, w)
            self.Kernel = self.Predictor_ll.Return_Integral_Kernel()
            
        elif method == "LJ":
            LJ_params = self.Predictor_ll.get_LJ_params(np.array([0.02,2.3]),PointNum = f_grid_nums)
            self.Kernel = self.Predictor_ll.Predict
            
        elif method == "PoorMansIS":
            self.Kernel = self.Predictor_ll.PoorMansPredict
            self.Predictor_ll.method = "PoorMansIS"
            
        return self.Predictor_ll
    
    
    def get_RHS_for_Integrator(self, numeric = True, deg = 6, maxnum = 7, tol = 1e-2, convergence = True, method = "t"):
        
        if numeric and method == "t":
            T = TensorProduct_Quadrature(Dimensions = self.dim)
            T.get_Tensor_Quadrature_Sequence(typ='l', NumberofPoints=maxnum)
            poly = ndPolynomial(Dimensions=self.dim,maxdeg=deg)
            rhs = []
            
            if convergence:
                
                for i in range(maxnum-1,-1,-1):
                    abscissas = T.Abscissas[i]
                    print('integrating rhs numerically @ #points',len(abscissas))
                    eval_poly = poly.evaluate_MonomialBasis(abscissas)
                    
                    w = self.Predictor_ll.Predict(abscissas)
                    F = w * eval_poly
                    rhs.append(F @ T.Weights[i])
                    
                    if len(rhs) > 1 and np.mean(np.abs((rhs[-1] - rhs[-2]) / rhs[-1])) < tol:
                        print('tolerance reached @',i,'/',maxnum)
                        break
                    
            else:
                
                abscissas = T.Abscissas[0]
                eval_poly = poly.evaluate_MonomialBasis(abscissas)
                print('integrating rhs numerically @ #points',len(abscissas))
                w = self.Predictor_ll.Predict(abscissas)
                print(w)
                F = w * eval_poly
                rhs = [F @ T.Weights[0]]
                
            self.RHS_for_Integrator['numeric'] = rhs[-1]
            
            
        elif numeric and method == "mc":
            poly = ndPolynomial(Dimensions=self.dim,maxdeg=deg)
            rhs = []
            
            if convergence:
                
                for i in range(maxnum):
                    abscissas = np.random.rand(2**i,self.dim)
                    print('integrating rhs numerically @ #points',len(abscissas))
                    eval_poly = poly.evaluate_MonomialBasis(abscissas)
                    w = self.Predictor_ll.Predict(abscissas)
            
                    F = np.sum(w * eval_poly, axis = 1) / (2**i)
                    rhs.append(F)
                    
                    if len(rhs) > 1 and np.mean(np.abs((rhs[-1] - rhs[-2]) / rhs[-1])) < tol:
                        print('tolerance reached @',i,'/',maxnum)
                        break
                    
            else:
                
                abscissas = np.random.rand(2**maxnum,self.dim)
                eval_poly = poly.evaluate_MonomialBasis(abscissas)
                print('integrating rhs numerically @ #points',len(abscissas))
                w = self.Predictor_ll.Predict(abscissas)
                #print(w)
                F = np.sum(w * eval_poly, axis = 1) / (2**maxnum)
                rhs = [F]
                
            self.RHS_for_Integrator['numeric'] = rhs[-1]
            
            
        elif numeric == False:
            print('integrating rhs analytically with Kernel')
            poly = ndPolynomial(Dimensions=self.dim,maxdeg=deg)
            poly.Kernel = self.Predictor_ll.Return_Integral_Kernel()
            poly.get_MonomIntegrals(analytic=True)
            rhs = [poly.MonomIntegrals]
            self.RHS_for_Integrator['analytic'] = rhs[-1]
            
        print('RHS integration completed')
        return rhs
        
    
    def get_Proposal_for_Integrator(self, NumberofPoints = 1, Sampling_Procedure = 'r', threshold = 0.0):
        
        print('obtaining ProposalAbscissas / #points',NumberofPoints)
        if Sampling_Procedure == 'r':
            p = np.zeros([NumberofPoints,self.dim])
            wp = np.zeros(NumberofPoints)
            i = 0
            count = 0
            w_max = np.array(self.Predictor_ll.Predict(np.random.rand(500,self.dim))).max() 
            while i < NumberofPoints:
                count += 1
                p_proposal = np.random.rand(self.dim)
                tmp = self.Predictor_ll.Predict(p_proposal) / w_max
                if np.random.rand() < tmp and tmp > threshold:
                    p[i] = p_proposal
                    wp[i] = tmp
                    i += 1
                             
        print('done with',count,'point evaluations for',NumberofPoints,'proposal abscissas')             
        return p, wp
    
    
    def check_Proposal(self, p, threshold = 0.0):
        
        wp = self.Predictor_ll.Predict(p)
        L = wp > threshold
        return L
    
        
    def get_Integrator(self, method = "l1", deg = 5, sequence = True, method_dict = dict()):
        
        print('generating',method,'integrator')
        print("sequence =",sequence)
        
        if method.lower() == "mc":
            nested = method_dict["nested_mc"] if "nested_mc" in method_dict else False
            print("nested =", nested)
            
            Q = MonteCarlo_Quadrature(Dimensions=self.dim)
            if sequence:
                Q.get_MC_Sequence(NumberofPoints=deg,nested=nested)
            else:
                Q.get_MC_Abscissas_and_Weights(NumberofPoints=deg)
                
        
        
        elif method.lower() == "mc_is":
            nested = method_dict["nested_mc"] if "nested_mc" in method_dict else False
            proposal_threshold = method_dict["proposal_threshold"] if "proposal_threshold" in method_dict else 0.0
            print("nested =", nested)
            print("proposal_threshold =", proposal_threshold)
            
            Q = MonteCarlo_Quadrature(Dimensions=self.dim)
            Q.Kernel = self.Predictor_ll.Predict
            if sequence:
                Q.get_MC_Sequence(NumberofPoints=deg,nested=nested,threshold=proposal_threshold)
            else:
                Q.get_MC_Abscissas_and_Weights(NumberofPoints=deg,threshold=proposal_threshold)
            
         
            
        elif method.lower() == "t":
            typ = method_dict["typ"] if "typ" in method_dict else 'l'
            print("typ =", typ)
            print("Number of 1D Points:", deg)
            
            Q = TensorProduct_Quadrature(Dimensions=self.dim)
            if sequence:
                Q.get_Tensor_Quadrature_Sequence(typ = typ,NumberofPoints=deg)
            else:
                Q.set_UnivariateBasis(types = [typ], NumberofPoints=[deg])
                Q.get_Tensor_Quadrature()
        
        
        
        elif method.lower() == "l1":
            proposal_factor = method_dict["proposal_factor"] if "proposal_factor" in method_dict else 5
            proposal_threshold = method_dict["proposal_threshold"] if "proposal_threshold" in method_dict else 0.0
            print("deg =", deg)
            print("proposal_factor =", proposal_factor)
            print("proposal_threshold =", proposal_threshold)
            
            Q = L1_Quadrature(Dimensions=self.dim)
            Q.Kernel = self.Predictor_ll.Predict
            Q.set_PolynomialBasis(MaximumDegree=deg)
            n_deg = get_Number_of_Polynomials_below_Degree(self.dim,deg)
            n = int( n_deg * proposal_factor )
            
            Q.AbscissasProposal, _ = self.get_Proposal_for_Integrator(NumberofPoints=n,threshold=proposal_threshold)
            if "rhs_analytic" in method_dict and method_dict["rhs_analytic"]:
                Q.RHS = self.RHS_for_Integrator["analytic"][0:n_deg]
            else:
                Q.RHS = self.RHS_for_Integrator["numeric"][0:n_deg]
            
            if sequence:
                Q.get_nested_L1_Quadrature_Sequence()
            else:
                Q.get_L1_Quadrature()
        
        
        
        elif method.lower() == "l1d":
            proposal_factor = method_dict["proposal_factor"] if "proposal_factor" in method_dict else 5
            proposal_threshold = method_dict["proposal_threshold"] if "proposal_threshold" in method_dict else 0.0
            reduction_method = method_dict["reduction_method"] if "reduction_method" in method_dict else 'greedy'
            quality_limits = method_dict["quality_limits"] if "quality_limits" in method_dict else [0.2,0.2,0.05]
            print("deg =", deg)
            print("proposal_factor =", proposal_factor)
            print("proposal_threshold =", proposal_threshold)
            print("reduction_method =", reduction_method)
            print("quality_limits =", quality_limits)
            
            if "rq_first" in method_dict and method_dict["rq_first"]:
                deg_rq = method_dict["deg_rq"] if "deg_rq" in method_dict else deg+1
                rq_ftol = method_dict["rq_ftol"] if "rq_ftol" in method_dict else 1e-10
                rq_tol = method_dict["rq_tol"] if "rq_tol" in method_dict else 1e-8
                rq_add_points = method_dict["rq_add_points"] if "rq_add_points" in method_dict else 0
                print("deg_rq =", deg_rq)
                print("rq_ftol =", rq_ftol)
                print("rq_tol =", rq_tol)
                print("rq_add_points =", rq_add_points)
                
                RQ = Reduced_Quadrature(Dimensions = self.dim)
                RQ.Kernel = self.Predictor_ll.Predict
                RQ.set_PolynomialBasis(MaximumDegree=deg_rq)
                n_deg_rq = get_Number_of_Polynomials_below_Degree(self.dim,deg_rq)
                n = int( n_deg_rq * proposal_factor )
                
                RQ.AbscissasProposal, _ = self.get_Proposal_for_Integrator(NumberofPoints=n,threshold=proposal_threshold)
                if "rhs_analytic" in method_dict and method_dict["rhs_analytic"]:
                    RQ.RHS = self.RHS_for_Integrator["analytic"][0:n_deg_rq]
                else:
                    RQ.RHS = self.RHS_for_Integrator["numeric"][0:n_deg_rq]
                
                RQ.get_L1_Abscissas_and_Weights()
                RQ.get_Cluster_Sequence()
                RQ.get_Objective_and_Derivative(scalar=False,NumberofBasisFunctions=n_deg_rq)
                RQ.get_Reduced_Quadrature(NumberofBasisFunctions=n_deg_rq,ftol=rq_ftol,tol=rq_tol)
            
            Q = L1D_Quadrature(Dimensions=self.dim)
            Q.Kernel = self.Predictor_ll.Predict
            Q.set_PolynomialBasis(MaximumDegree=deg)
            n_deg = get_Number_of_Polynomials_below_Degree(self.dim,deg)
            if "rq_first" in method_dict and method_dict["rq_first"]:
                n = len(RQ.Abscissas[-1])
                L = self.check_Proposal(RQ.Abscissas[-1],threshold=proposal_threshold)
                p, _ = self.get_Proposal_for_Integrator(NumberofPoints=rq_add_points + np.sum(~L),threshold=proposal_threshold)
                Q.AbscissasProposal = np.concatenate((RQ.Abscissas[-1][L],p),axis=0)
            else:
                n = int( n_deg * proposal_factor )
                Q.AbscissasProposal, _ = self.get_Proposal_for_Integrator(NumberofPoints=n,threshold=proposal_threshold)
                
                
            if "rhs_analytic" in method_dict and method_dict["rhs_analytic"]:
                Q.RHS = self.RHS_for_Integrator["analytic"][0:n_deg]
            else:
                Q.RHS = self.RHS_for_Integrator["numeric"][0:n_deg]
                
            if sequence:
                Q.get_nested_L1D_Quadrature_Sequence(quality_limits=quality_limits)
            else:
                Q.get_L1D_Quadrature_reduced(quality_limits=quality_limits,reduction_method=reduction_method)



        elif method.lower() == "rq":
            proposal_factor = method_dict["proposal_factor"] if "proposal_factor" in method_dict else 5
            proposal_threshold = method_dict["proposal_threshold"] if "proposal_threshold" in method_dict else 0.0
            ftol = method_dict["ftol"] if "ftol" in method_dict else 1e-10
            tol = method_dict["tol"] if "tol" in method_dict else 1e-8
            print("deg =", deg)
            print("proposal_factor =", proposal_factor)
            print("proposal_threshold =", proposal_threshold)
            print("ftol =", ftol)
            print("tol =", tol)
            
            Q = Reduced_Quadrature(Dimensions = self.dim)
            Q.Kernel = self.Predictor_ll.Predict
            Q.set_PolynomialBasis(MaximumDegree=deg)
            n_deg = get_Number_of_Polynomials_below_Degree(self.dim,deg)
            n = int( n_deg * proposal_factor )
            
            Q.AbscissasProposal, _ = self.get_Proposal_for_Integrator(NumberofPoints=n,threshold=proposal_threshold)
            if "rhs_analytic" in method_dict and method_dict["rhs_analytic"]:
                Q.RHS = self.RHS_for_Integrator["analytic"][0:n_deg]
            else:
                Q.RHS = self.RHS_for_Integrator["numeric"][0:n_deg]
            
            Q.get_L1_Abscissas_and_Weights()
            Q.get_Cluster_Sequence()
            if sequence:
                Q.get_Reduced_Quadrature_Sequence(ftol=ftol,tol=tol)
            else:
                Q.get_Objective_and_Derivative(scalar=False,NumberofBasisFunctions=n_deg)
                Q.get_Reduced_Quadrature(NumberofBasisFunctions=n_deg,ftol=ftol,tol=tol)
            


        self.Integrator.append( Q )
        self.Integrator_List.append( method )
        
        print('done')
        return
    
    
    def get_Integrand(self, points, approx_quantity = "w", difference = False, grad = False, level = 'll', cancel_outlier_values = False):
        
        if not difference:
            
            f = self.Predictor_ll.Predict(points) if level == "ll" else self.Predictor_hl.Predict(points)
            
        else:
            
            if approx_quantity == "w":
                
                if grad:
                    energies_hl = self.Predictor_hl.Predict(points)
                    w_hl = self.Predictor_hl.Predict(points)
                    w_ll = self.Predictor_ll.Predict(points)
                    f = w_hl / w_ll
                    
                    dw_hl = self.Predictor_hl.Predict_Grad(points)
                    dw_ll = self.Predictor_ll.Predict_Grad(points)
                    
                    df = dw_hl / w_ll - f * dw_ll / w_ll
                                        
                else:
                    w_hl = self.Predictor_hl.Predict(points)
                    w_ll = self.Predictor_ll.Predict(points)
                    f = w_hl / w_ll
                
            else:
                
                if grad:
                    energies_hl = self.Predictor_hl.Predict(points,quantity = "E")
                    energies_ll = self.Predictor_ll.Predict(points,quantity = "E")
                    grad_ll = self.Predictor_ll.Predict_Grad(points,quantity = "E")
                    grad_hl = self.Predictor_hl.Predict_Grad(points,quantity = "E")
                    f = np.exp(-self.beta * (energies_hl - energies_ll))
                    df = f[:,None] * (grad_ll - grad_hl) * self.beta

                else:
                    energies_hl = self.Predictor_hl.Predict(points,quantity = "E")
                    energies_ll = self.Predictor_ll.Predict(points,quantity = "E")
                    f = np.exp(-self.beta * (energies_hl - energies_ll))
                    
        if cancel_outlier_values:
            f[(f>np.median(f)*1e2) + (f>1e5)] = 1
            if grad:
                df[f>np.median(df)*1e3] = np.median(df)
        F = [f] if not grad else [f,df]
        return F
        
    
    def get_Integral(self, integrator = 0, level = "ll", eval_index = 0, tol = 1e-1, 
                     approx_quantity = "w", use_nesting = True, convergence = None):
        I = []
        F_List = []
        NP = []
        cancel_outlier_values = True if self.Integrator_List[integrator].lower() in ["rq","l1d"] else False
        difference = False if self.Integrator_List[integrator].lower() in ["mc","mc_is","t"] else True
        grad = True if self.Integrator_List[integrator].lower() == "l1d" else False
        use_nesting = use_nesting if self.Integrator_List[integrator].lower() in ["mc","mc_is","l1","l1d"] else False
        print("Integration via",self.Integrator_List[integrator],"integrator")
        print("mode:",eval_index)
        if eval_index is str: print("NestStructure usage:",use_nesting)
        
        
        if type(eval_index) is not str:
            abscissas = self.Integrator[integrator].Abscissas[eval_index]
            print('integrating',self.Integrator_List[integrator],'integrator / #points', len(abscissas))
            F = self.get_Integrand(abscissas, approx_quantity=approx_quantity, difference=difference, 
                                   grad=grad, level=level, cancel_outlier_values=cancel_outlier_values)
            F_List.append( F )
            NP.append( len(abscissas) )
            
            I.append( self.Integrator[integrator].Integrate(*F,ListIndex = eval_index) * self.mult_factor  )
            
        else:
            for i in range(len(self.Integrator[integrator].Weights)-1,-1,-1):
                full_abscissas = self.Integrator[integrator].Abscissas[i]
                if use_nesting and len(F_List) > 0 and self.Integrator[integrator].NestStructure is not None:
                    L_nest = self.Integrator[integrator].NestStructure[i]
                    new_abscissas = full_abscissas[~L_nest]
                else:
                    new_abscissas = full_abscissas
                    
                    
                print("total points:",len(full_abscissas),' / evaluating',len(new_abscissas),"new points")
                F_new = self.get_Integrand(new_abscissas, approx_quantity=approx_quantity, difference=difference, 
                                           grad=grad, level=level, cancel_outlier_values=cancel_outlier_values)
                if use_nesting and len(F_List) > 0 and self.Integrator[integrator].NestStructure is not None:
                    F_full = [np.zeros(len(full_abscissas))]
                    F_full[0][L_nest] = F_List[-1][0]
                    F_full[0][~L_nest] = F_new[0]
                    if grad:
                        F_full.append( np.zeros([len(full_abscissas),self.dim]) )
                        F_full[1][L_nest] = F_List[-1][1]
                        F_full[1][~L_nest] = F_new[1]
                else:
                    F_full = F_new
                
                F_List.append( F_full )
                NP.append( len(full_abscissas) )

                I.append( self.Integrator[integrator].Integrate(*F_full,ListIndex = i) * self.mult_factor )
                if convergence == "rel" and len(I) > 1:
                    conv_quant = np.abs( (I[-1] - I[-2]) / I[-1] )
                elif convergence == "var" and len(I) > 1:
                    conv_quant = np.std(F_List[-1][0]) / np.sqrt(len(F_List[-1][0]))
                else:
                    conv_quant = tol + 1
                
                if conv_quant < tol:
                    print('Integral converged @',i,'/',len(self.Integrator[integrator].Weights)-1)
                    break
        
        return np.array(I), F_List, np.array(NP)
        
    
    def clear_Integrator(self,index = None):
        if index is None:
            self.Integrator = []
            self.Integrator_List = []
            self.Integrator_analytic = []
        else:
            del self.Integrator[index]
            del self.Integrator_List[index]
            del self.Integrator_analytic[index]
        pass
            
        























