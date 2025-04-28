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
from ase.constraints import FixAtoms
from ase import units
import itertools
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import scipy.optimize
from itertools import product
from scipy.optimize import approx_fprime


def rescale_points(points, ranges, rescaled_ranges):
    rescaled_points = (points - ranges[:,0][None,:]) / (np.diff(ranges, axis = 1).T) * np.diff(rescaled_ranges, axis = 1).T + rescaled_ranges[:,0]
    return rescaled_points

def get_fprime(f,x,delta,*args):
    x = x if len(x.shape) > 1 else x.reshape(1,len(x))
    grad = np.zeros_like(x)
    for i in range(x.shape[1]):
        xp = x
        xp[:,i] = xp[:,i] + delta
        xm = x
        xm[:,i] = xm[:,i] - delta
        grad[:,i] = ( f(xp,*args) - f(xm,*args) ) / (2*delta)
    return grad

class Int_and_Fit:
    
    def __init__(self,mol,ranges,interpolation_ranges,integration_ranges,dim,coordinates,fixed_coordinates,fixed_coordinate_values):  
        self.ranges = ranges
        self.interpolation_ranges = interpolation_ranges
        self.integration_ranges = integration_ranges
        self.dim = len(coordinates)
        self.mol = mol
        self.method = None
        
        coord_dict = dict()
        if "x" in coordinates or "y" in coordinates:
            coord_dict["theta"], coord_dict["phi"], coord_dict["xi"], coord_dict["x"], coord_dict["y"], coord_dict["z"] = 0,1,2,3,4,5
            self.cart = True
        else:
            coord_dict["theta"], coord_dict["phi"], coord_dict["xi"], coord_dict["r"], coord_dict["chi"], coord_dict["z"] = 0,1,2,3,4,5
            self.cart = False
        self.coordinate_numbers = np.array([coord_dict[i] for i in coordinates])
        self.fixed_coordinate_numbers = np.array([coord_dict[i] for i in fixed_coordinates])
        self.fixed_coordinate_values = fixed_coordinate_values
        
        return 
    
    
    def get_6D_points(self,points):
        points_6D = np.zeros([int(points.size/len(self.coordinate_numbers)),6])
        points = points.reshape(1,points.size) if len(points.shape) == 1 else points
        if points.shape[1] != 6: 
            points_6D[:,self.coordinate_numbers] = points
            points_6D[:,self.fixed_coordinate_numbers] = np.ones([int(points.size/self.dim),len(self.fixed_coordinate_numbers)]) * self.fixed_coordinate_values
        else:
            points_6D = points
        return points_6D


    def rbf_interpol(self,gridpoints,energies,epsilon = 0.5, smooth = -0.001):
        rbfs = Rbf(gridpoints[:,0],gridpoints[:,1],gridpoints[:,2],gridpoints[:,3],gridpoints[:,4],energies,function = "gaussian",epsilon = epsilon)
        return rbfs
    
    
    def rescale_points(self,points, ranges, rescaled_ranges):
        rescaled_points = (points - ranges[:,0][None,:]) / (np.diff(ranges, axis = 1).T) * np.diff(rescaled_ranges, axis = 1).T + rescaled_ranges[:,0]
        return rescaled_points
    
    
    def rescale_points_back(self,rescaled_points, ranges, rescaled_ranges):
        points = (rescaled_points - rescaled_ranges[:,0]) / np.diff(rescaled_ranges, axis = 1).T  * (np.diff(ranges, axis = 1).T) + ranges[:,0].reshape(1,len(ranges))
        return points
    
    
    def rescaled_points_to_unit_interval(self,points, ranges):
        rescaled_points = (points - ranges[:,0]) / np.diff(ranges, axis = 1).T
        return rescaled_points
    
    
    def rescale_points_back_from_unit_interval(self,rescaled_points, ranges):
        points = rescaled_points * np.diff(ranges, axis = 1).T  + ranges[:,0]
        return points
    
    
    def get_grid(self,ranges,rescaled_ranges,gridnum):
        theta_range, phi_range, xi_range, r_range, chi_range = ranges[0], ranges[1], ranges[2], ranges[3], ranges[4]
        xis = np.linspace(xi_range[0], xi_range[1], gridnum[2])
        thetas = np.linspace(theta_range[0], theta_range[1], gridnum[0])
        phis = np.linspace(phi_range[0], phi_range[1], gridnum[1])
        rs = np.linspace(r_range[0], r_range[1], gridnum[3])
        chis = np.linspace(chi_range[0], chi_range[1], gridnum[4])
        
        gridpoints = np.array(list(itertools.product(thetas,phis,xis,rs,chis)))
        rescaled_gridpoints = (gridpoints - ranges[:,0].reshape(1,len(ranges))) / np.diff(ranges, axis = 1).T * np.diff(rescaled_ranges, axis = 1).T + rescaled_ranges[:,0]
        scaling_factor = np.prod(np.diff(ranges,axis = 1))
        symmetry_factor = 8 * np.pi**4 / np.prod(np.diff(ranges,axis = 1))
        return gridpoints, rescaled_gridpoints, symmetry_factor * scaling_factor
    
    
    def get_grid_nD(self, ranges, rescaled_ranges, gridnum):
        gridpoints_1D = [np.linspace(self.ranges[i,0],self.ranges[i,1],gridnum[i]) for i in range(len(ranges))]
        gridpoints = np.array(list(itertools.product(*gridpoints_1D)))
        rescaled_gridpoints = (gridpoints - ranges[:,0].reshape(1,len(ranges))) / np.diff(ranges, axis = 1).T * np.diff(rescaled_ranges, axis = 1).T + rescaled_ranges[:,0]
        scaling_factor = np.prod(np.diff(ranges,axis = 1))
        symmetry_factor = 8 * np.pi**4 / np.prod(np.diff(ranges,axis = 1))
        return gridpoints, rescaled_gridpoints, symmetry_factor * scaling_factor
    
    
    def Find_abscissas(self,ranges, rescaled_gridpoints,rbfs_w,abscissas_num = 10):
        abscissas = np.zeros([abscissas_num,self.dim])
        c = 0
        while c < abscissas_num:
            absc_suggestion = np.random.rand(self.dim)
            r = np.random.rand()
            if r < rbfs_w(absc_suggestion[0],absc_suggestion[1],absc_suggestion[2],absc_suggestion[3],absc_suggestion[4]):
                abscissas[c,:] = absc_suggestion
                c += 1
        return abscissas, self.rescale_points_back(abscissas, ranges)
    
    
    def Find_abscissas_threshold_distance(self,ranges,rbfs_w,abscissas_num = 10, thres = 0.1):
        abscissas = np.zeros([abscissas_num,self.dim])
        c = 0
        while c < abscissas_num:
            absc_suggestion = np.random.rand(self.dim)
            r = np.random.rand()
            if r < rbfs_w(absc_suggestion[0],absc_suggestion[1],absc_suggestion[2],absc_suggestion[3],absc_suggestion[4]) and c > 0:
                if np.any(np.linalg.norm(abscissas[0:c,:] - absc_suggestion[None,:], axis = 1) > thres) and c > 0:
                    abscissas[c,:] = absc_suggestion
                    c += 1
            elif r < rbfs_w(absc_suggestion[0],absc_suggestion[1],absc_suggestion[2],absc_suggestion[3],absc_suggestion[4]) and c ==0:
                abscissas[c,:] = absc_suggestion
                c += 1
        return abscissas, self.rescale_points_back(abscissas, ranges)
        
    
    def check_interpol(self,quantity = "w",check_num=50):
        rand_points = np.random.rand(check_num,self.dim)
        points_in_ranges = self.rescale_points(rand_points,self.integration_ranges,self.ranges)
        
        Q_int = np.zeros(len(rand_points))
        Q_calc = np.zeros(len(rand_points))
       
        for i in range(len(rand_points)):
            mol_pos = self.mol.transrot_mol(self.mol.mol_reference, *self.get_6D_points(points_in_ranges[i])[0])
            Q_calc[i] = self.mol.energy(mol_pos) - self.mol.E_ref 
            Q_int[i] = self.Predict(rand_points[i],quantity)
                
        return Q_calc, Q_int
    
    
    def get_Jac(self,gridpoint):

        if not self.cart and self.mol.atom_num_mol > 1: 
            theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
            r_arg = np.where(self.coordinate_numbers == 3)[0][0]
            Jac = np.abs(np.sin(gridpoint[theta_arg])) * gridpoint[r_arg]
        if self.cart and self.mol.atom_num_mol > 1:
            theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
            Jac = np.abs(np.sin(gridpoint[theta_arg]))
        if not self.cart and self.mol.atom_num_mol == 1:
            r_arg = np.where(self.coordinate_numbers == 3)[0][0]
            Jac = gridpoint[r_arg]
        if self.cart and self.mol.atom_num_mol == 1:
            Jac = 1
        
        return Jac
    
    
    def check_interpol_w(self,mol,predictor, check_num, ranges, rescaled_ranges, beta):
        rand_points = np.random.rand(check_num,self.dim) 
        points = self.rescale_points(rand_points, self.integration_ranges, self.ranges)
        rbf_points = self.rescale_points_back_from_unit_interval(rand_points, rescaled_ranges)
        Q_int = np.zeros(len(points))
        Q_calc = np.zeros(len(points))
        
        if type(predictor) == "scipy.interpolate.rbf.Rbf":
            for i in range(len(points)):
                mol_pos = mol.transrot_mol(mol.mol_reference, *self.get_6D_points(points[i])[0])
                Q_calc[i] = np.exp(-beta * (mol.energy(mol_pos) - mol.E_ref)) * self.get_Jac(points[i])
                Q_int[i] = predictor(*rbf_points[i])
                
        else:
            for i in range(len(points)):
                mol_pos = mol.transrot_mol(mol.mol_reference, *self.get_6D_points(points[i])[0])
                Q_calc[i] = np.exp(-beta * (mol.energy(mol_pos) - mol.E_ref)) * self.get_Jac(points[i])
                Q_int[i] = self.Predict(rand_points[i])
                
        return Q_calc, Q_int
    
    
    def get_LJ_params(self,start_vals,PointNum = 100):
	
        self.method = "LJ"    
    
        grid_points = self.rescale_points(np.random.rand(PointNum,self.dim), self.integration_ranges, self.ranges)
        Es = self.mol.get_energies(grid_points)
        mol_positions = np.array([self.mol.transrot_mol(self.mol.mol_reference, *self.get_6D_points(grid_points[i])[0]) for i in range(len(grid_points))])
        self.pore_position = self.mol.getPore_positions()

        def LJ(LJ_params):
            distance_vecs = mol_positions[:,:,None,:] - self.pore_position[None,None,:,:]
            distances = np.sqrt(np.sum(distance_vecs**2,axis = 3))	
            LJ_energies = np.einsum("ilk->i",LJ_params[0] * ((LJ_params[1]/distances)**12 - (LJ_params[1]/distances)**6))
            return np.sum(np.exp(-self.mol.beta*Es)*(Es - LJ_energies)**2)

        LJ_optimized = scipy.optimize.minimize(LJ,start_vals,method = "BFGS")
        self.LJ_params = LJ_optimized.x
        
        return self.LJ_params
    
    
    def PoorMansPredict(self,grid,alpha = 100):
        
        self.method = "PoorMansIS"
        
        dist_vec_ref = self.mol.mol_reference[:,None,:] - self.mol.pore_position[None,:,:]
        dist_ref_min = np.min(np.sqrt(np.sum(dist_vec_ref**2,axis = 2)))

        grid = grid.reshape(int(grid.size/self.dim),self.dim)
        grid = self.rescale_points(grid, self.integration_ranges, self.ranges)
        grid_6D = self.get_6D_points(grid)
            
        mol_positions = np.array([self.mol.transrot_mol(self.mol.mol_reference, *self.get_6D_points(grid[i])[0]) for i in range(len(grid))])
        distance_vecs = mol_positions[:,:,None,:] - self.mol.pore_position[None,None,:,:]
        distances = np.sqrt(np.sum(distance_vecs**2,axis = 3))
        
        min_dist_list = np.array([np.min(distances[i,:,:]) for i in range(len(distances))])
        min_dist_list[min_dist_list > dist_ref_min] = dist_ref_min
        w = 1/(1- alpha * (min_dist_list - dist_ref_min)**3)
        w[w>1] = 1
        w[w<0] = 0.01
        
        return w

    
    def LinFitGaussians(self, ceta, f_grid_points, rbf_grid_points, f, chopping = False):
        
        self.cetas = np.ones(len(rbf_grid_points)) * ceta
        self.mus = rbf_grid_points
        
        fnums = len(f_grid_points)
        rbf_nums = len(rbf_grid_points)
        
        problem_size = fnums * rbf_nums * 8 / 1e9
        
        if problem_size > 1 and chopping:
            print("chopping problem into "+str(int(problem_size / 1))+" pieces")
            As_fit = np.zeros(rbf_nums)
            
            T = np.zeros([rbf_nums,rbf_nums])
            
            j_vec = np.linspace(0,rbf_nums,int(problem_size / 1),dtype = int)
            i_vec = np.linspace(0,fnums,int(problem_size / 1),dtype = int)
            range_list_j = [[j_vec[i],j_vec[i+1]] for i in range(len(j_vec)-1)]
            range_list_i = [[i_vec[i],i_vec[i+1]] for i in range(len(j_vec)-1)]
            range_combinations = list(product(range_list_j,range_list_j))
            
            for rc in range_combinations:
                subM_j = np.zeros([fnums,int(np.diff(rc[0]))])
                subM_k = np.zeros([fnums,int(np.diff(rc[1]))])
                for i in range(int(fnums)):
                    subM_j[i,:] = np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points[rc[0][0]:rc[0][1]])**2, axis = 1))
                    subM_k[i,:] = np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points[rc[1][0]:rc[1][1]])**2, axis = 1))
                T[rc[0][0]:rc[0][1],rc[1][0]:rc[1][1]] = np.matmul(subM_j.T,subM_k)
            print("done with chopping")
            #del subM_j
            #del subM_k
            T_inv = np.linalg.inv(T)
            
            for r in range_list_i:
                subX = np.zeros([int(np.diff(r)),rbf_nums])
                for i in range(int(np.diff(r))):
                    subX[i,:] = np.exp(-ceta * np.sum((f_grid_points[r[0]:r[1]][i][None,:] - rbf_grid_points)**2, axis = 1))
                sub_pinv = np.matmul(T_inv,subX.T)
                As_fit = As_fit + np.matmul(sub_pinv,f[r[0]:r[1]])
        
        else:        
            M = np.zeros([fnums,rbf_nums])
            
            for i in range(int(fnums)):
                M[i,:] = np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points)**2, axis = 1))
                
            P = np.linalg.pinv(M)
            As_fit = np.matmul(P,f)
        
        self.As = As_fit

        return As_fit
    
    
    def LinFitGaussians_iter(self, ceta, f_grid_points, rbf_grid_points, f, method = "CG", tol = 0.01):
            
        self.method = "rbf"
        
        As_start = np.ones(len(rbf_grid_points))
        non_zero_args_f, vec_list_f, _ = self.Find_non_zero_args(f_grid_points,rbf_grid_points,ceta)
        non_zero_args_rbf, vec_list_rbf, _ = self.Find_non_zero_args(rbf_grid_points,f_grid_points,ceta)
        print("done finding non zero args")
            
        def fitfun(vals):
            y = np.zeros(f.size)
            for i in range(int(f.size)):
                #y[i] = np.dot(np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points[non_zero_args_rbf[i]])**2, axis = 1)) , vals[non_zero_args_rbf[i]])
                y[i] = np.dot(vec_list_rbf[i] , vals[non_zero_args_rbf[i]])
            print(np.sum((y - f)**2))
            return np.sum((y - f)**2)
      
        def yPred(vals):
            y = np.zeros(f.size)
            for i in range(int(f.size)):
                #y[i] = np.dot(np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points[non_zero_args_rbf[i]])**2, axis = 1)) , vals[non_zero_args_rbf[i]])
                y[i] = np.dot(vec_list_rbf[i] , vals[non_zero_args_rbf[i]])
            return y
      
        def Jacobian(vals):
            
            y = yPred(vals)
            J_As = np.zeros(np.shape(vals))
            
            for m in range(len(vals)):
                vec = vec_list_f[m]
                J_As[m] = 2*np.dot((y-f)[non_zero_args_f[m]],vec)
            return J_As
      
        opt_object = scipy.optimize.minimize(fitfun, As_start, method = method, jac = Jacobian, options={"maxiter":2000,"gtol":1},tol = tol)
      
        self.cetas = np.ones(len(rbf_grid_points)) * ceta
        self.mus = rbf_grid_points
        self.As = opt_object.x
        
        return self.As
    
    
    def LinFitGaussians_grad_iter(self, ceta, f_grid_points, rbf_grid_points, f, df, alpha = 0.1, method = "CG", tol = 0.01):
            
        self.method = "rbf"
        
        As_start = np.ones(len(rbf_grid_points))
        non_zero_args_jac, exponent_list_jac, exponent_der_list_jac = self.Find_non_zero_args(f_grid_points,rbf_grid_points,ceta)
        non_zero_args_fun, exponent_list_fun, exponent_der_list_fun = self.Find_non_zero_args(rbf_grid_points,f_grid_points,ceta)
        print("done finding non zero args")
            
        def fitfun(vals):
            y = np.zeros(f.size)
            dy = np.zeros(df.shape)
            for i in range(int(f.size)):
                y[i] = np.dot(exponent_list_fun[i] , vals[non_zero_args_fun[i]])
                dy[i,:] = np.einsum("l,lk->k", vals[non_zero_args_fun[i]] , exponent_der_list_fun[i])
            error = np.sum((y - f)**2) + alpha * np.sum((df-dy)**2)
            print(error)
            return error
      
        def yPred(vals):
            y = np.zeros(f.size)
            dy = np.zeros(df.shape)
            for i in range(int(f.size)):
                y[i] = np.dot(exponent_list_fun[i] , vals[non_zero_args_fun[i]]) 
                dy[i,:] = np.einsum("l,lk->k", vals[non_zero_args_fun[i]] , exponent_der_list_fun[i])
            return y, dy
      
        def Jacobian(vals):
            
            y, dy = yPred(vals)
            J_As = np.zeros(np.shape(vals))
            
            for m in range(len(vals)):
                J_As[m] = 2*np.dot((y-f)[non_zero_args_jac[m]],exponent_list_jac[m]) + 2*alpha* np.sum((dy - df)[non_zero_args_jac[m],:] * exponent_der_list_jac[m])
            return J_As
      
        opt_object = scipy.optimize.minimize(fitfun, As_start, method = method, jac = Jacobian, options={"maxiter":2000,"gtol":1},tol = tol)
      
        self.cetas = np.ones(len(rbf_grid_points)) * ceta
        self.mus = rbf_grid_points
        self.As = opt_object.x
        
        return self.As
    
    
    def Find_non_zero_args(self,rest_grid,fixed_grid_points,ceta):
        non_zero_arg_list = list()
        exponent_list = list()
        exponent_der_list = list()
        for i in range(len(fixed_grid_points)):
            non_zero_args = np.where(np.linalg.norm(rest_grid - fixed_grid_points[i], axis = 1)**2 * ceta < 11)[0]
            non_zero_arg_list.append(non_zero_args)
            exponent_list.append(np.exp(-ceta * np.sum((fixed_grid_points[i][None,:] - rest_grid[non_zero_args])**2, axis = 1)))
            exponent_der_list.append(exponent_list[-1][:,None] * (-2*ceta * (fixed_grid_points[i][None,:] - rest_grid[non_zero_args])))
        return non_zero_arg_list, exponent_list, exponent_der_list


    def LinFitGaussiansWithGrad(self, ceta, f_grid_points, rbf_grid_points, f, df, grad_weight = 0.1):
        
        self.cetas = np.ones(len(rbf_grid_points)) * ceta
        self.mus = rbf_grid_points
        
        fnums = len(f_grid_points)
        rbf_nums = len(rbf_grid_points)

        M = np.zeros([fnums*(self.dim+1),rbf_nums])
        
        for i in range(int(fnums)):
            M[i,:] = np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points)**2, axis = 1))
            M[fnums + self.dim*i:fnums + self.dim*(i+1),:] = grad_weight * -2*ceta * np.exp(-ceta * np.sum((f_grid_points[i][None,:] - rbf_grid_points)**2, axis = 1))[None,:] * (f_grid_points[i][None,:] - rbf_grid_points).T
            
        b = np.concatenate((f,df.reshape(df.size) * grad_weight))
        P = np.linalg.pinv(M)
        As_fit = np.matmul(P,b)
        
        self.As = As_fit
        
        return As_fit
    
    
    def SparseInterpol(self, ceta, rbf_gridpoints,f,threshold):
        
        self.cetas = np.ones(len(rbf_gridpoints)) * ceta
        self.mus = rbf_gridpoints
        
        n = len(rbf_gridpoints)
        
        M_rows = np.zeros(int(n**2))
        M_columns = np.zeros(int(n**2))
        data = np.zeros(int(n**2))
        
        c, c_old = 0, 0
        
        for i in range(int(n)):
            row = np.exp(-ceta * np.sum((rbf_gridpoints[i][None,:] - rbf_gridpoints)**2, axis = 1))
            non_zero_indices = np.where(row > threshold)[0]
            c += len(non_zero_indices)
            M_columns[c_old:c] = non_zero_indices
            M_rows[c_old:c] = np.ones(len(non_zero_indices))*i
            data[c_old:c] = row[row > threshold]
            c_old = np.copy(c)

        
        data = data[0:c]
        M_rows = M_rows[0:c]
        M_columns = M_columns[0:c]
        
        M_sparse = csr_matrix((data,(M_columns,M_rows)),(n,n))
        A = spsolve(M_sparse,f)
        
        self.As = A
        
        return A
        
        
    def CurveFitGaussians(self, f_grid, f, As, cetas, mus, method = "BFGS", tol = 0.01):
        
        start_vals = np.hstack((As,cetas,mus.reshape(mus.size)))
        n = As.size
        
        def fitfun(vals):
            y = np.zeros(f.size)
            for i in range(int(f.size)):
                y[i] = np.dot(np.exp(-vals[n:2*n] * np.sum((f_grid[i][None,:] - vals[2*n:None].reshape(n,self.dim))**2, axis = 1)) , vals[0:n])
            print(np.sum((y - f)**2))
            return np.sum((y - f)**2)
        
        def CalcPred(vals):
            y = np.zeros(f.size)
            for i in range(int(f.size)):
                y[i] = np.dot(np.exp(-vals[n:2*n] * np.sum((f_grid[i][None,:] - vals[2*n:None].reshape(n,self.dim))**2, axis = 1)) , vals[0:n])
            return y
        
        def Jacobian(vals):
            y = CalcPred(vals)
            J_As, J_cetas, J_mus = np.zeros(np.shape(As)), np.zeros(np.shape(cetas)), np.zeros(mus.size)
            for i in range(n):
                J_As[i] = np.dot(2 * (y-f) , (np.exp(-vals[n:2*n][i] * np.sum((f_grid - vals[2*n:None].reshape(n,self.dim)[i][None,:])**2, axis = 1))))
                J_cetas[i] = - np.dot(2 * (y-f) , vals[0:n][i] * (np.exp(-vals[n:2*n][i] * np.sum((f_grid - vals[2*n:None].reshape(n,self.dim)[i][None,:])**2, axis = 1))) * np.sum((f_grid - vals[2*n:None].reshape(n,self.dim)[i][None,:])**2, axis = 1))
                J_mus[self.dim*i:self.dim*(i+1)] = np.sum(4 * (y-f)[:,None] * vals[0:n][i] * (np.exp(-vals[n:2*n][i] * np.sum((f_grid - vals[2*n:None].reshape(n,self.dim)[i][None,:])**2, axis = 1)))[:,None] * (f_grid - vals[2*n:None].reshape(n,self.dim)[i][None,:]) * cetas[i] , axis = 0)
                #print("Jac")
            return np.hstack((J_As, J_cetas, J_mus))
        
        #check if Jacobian works
        a = int(np.random.rand() * len(As))
        check_vals = np.copy(start_vals)
        check_vals2 = np.copy(start_vals)
        check_vals2[a] = check_vals[a] + 1e-4
        print("a = ", a, (fitfun(check_vals2) - fitfun(check_vals)) / 1e-4 , Jacobian(check_vals)[a])
        #end of check
        
        vals_opt = scipy.optimize.minimize(fitfun,start_vals,method = method, jac = Jacobian, options={"maxiter":25})
        As_opt = vals_opt.x[0:n]
        cetas_opt = vals_opt.x[n:2*n]
        mus_opt = vals_opt.x[2*n:None].reshape(n,self.dim)
        
        self.cetas = cetas_opt
        self.mus = mus_opt
        self.As = As_opt
        
        return As_opt, cetas_opt, mus_opt
    
    
    def Predict(self, grid, quantity = "w"):
        
        if self.method == "rbf":
            grid = grid.reshape(int(grid.size/self.dim),self.dim)
            grid = self.rescale_points(grid, self.integration_ranges, self.interpolation_ranges)
            if quantity == "w":
                output = np.array([np.sum( self.As * np.exp(-self.cetas * np.sum((grid[i][None,:] - self.mus)**2, axis = 1))) for i in range(len(grid))])
            else:
                output = "not possible to provide E"
                print(output)
                
        elif self.method == "LJ":
            #grid = grid.reshape(int(grid.size/self.dim),self.dim) 
            grid = self.rescale_points(grid, self.integration_ranges, self.ranges)
            grid_6D = self.get_6D_points(grid)
            
            mol_positions = np.array([self.mol.transrot_mol(self.mol.mol_reference, *self.get_6D_points(grid[i])[0]) for i in range(len(grid))])
            distance_vecs = mol_positions[:,:,None,:] - self.pore_position[None,None,:,:]
            distances = np.sqrt(np.sum(distance_vecs**2,axis = 3))
            LJ_energies = np.einsum("ilk->i",self.LJ_params[0] * ((self.LJ_params[1]/distances)**12 - (self.LJ_params[1]/distances)**6))
            if quantity == "w":
                output = self.mol.get_w_from_E(LJ_energies, grid) #vll ausbessern
            else:
                output = LJ_energies
        
        elif self.method == "PoorMansIS":
            output = self.PoorMansPredict(grid)
            
        elif self.method is None:
            grid = self.rescale_points(grid, self.integration_ranges, self.ranges)
            output = self.mol.get_w(grid) if quantity == "w" else self.mol.get_energies(grid)
            
        return output
            
        
    def Predict_Grad(self, grid, quantity = "w"):
        
        if self.method == "rbf":
            grid = grid.reshape(int(grid.size/self.dim),self.dim)
            grid = self.rescale_points(grid, self.integration_ranges, self.interpolation_ranges)
            if quantity == "w":
                output = -np.array([np.sum((self.As * np.exp(-self.cetas * np.sum((grid[i][None,:] - self.mus)**2, axis = 1)))[:,None] * 2 * self.cetas[:,None] * (grid[i][None,:] - self.mus), axis = 0) for i in range(len(grid))])
                output = self.Rescale_Grad(output, self.interpolation_ranges, self.integration_ranges)
            else:
                output = "not possible to provide gradient of E"
                print(output)
                
        elif self.method == "LJ" or self.method == "PoorMansIS":
            delta = 1e-3
            output = get_fprime(self.Predict, grid, delta, quantity)
            
        elif self.method is None:
            grid = self.rescale_points(grid, self.integration_ranges, self.ranges)
            output = self.mol.get_gradient_array(grid,quantity=quantity)
            output = self.Rescale_Grad(output, self.ranges, self.integration_ranges)
        
        return output
    
           
    
    def Rescale_Grad(self,grad_array,original_ranges, projection_ranges):
        return grad_array * (np.diff(original_ranges, axis = 1).T  / (np.diff(projection_ranges, axis = 1).T))
        
    
    def Return_Integral_Kernel(self):
        As = self.As
        cetas = np.sqrt(self.cetas[0]) * (np.diff(self.interpolation_ranges) / np.diff(self.integration_ranges)).reshape(self.dim)
        mus = self.rescale_points(self.mus, self.interpolation_ranges, self.integration_ranges)
        return [As,cetas,mus]
        
    
    def Generate_2D_slice(self,beta,mol,dir1,dir2,restcoords,n, fun = None, method = "rbf"):
        x = np.linspace(self.ranges[dir1][0],self.ranges[dir1][1],n[0])
        y = np.linspace(self.ranges[dir2][0],self.ranges[dir2][1],n[1])
        X,Y = np.meshgrid(x,y)
        
        grid_reducedD = np.zeros([np.size(X),self.dim])
        grid_reducedD[:,dir1] = X.reshape(X.size)
        grid_reducedD[:,dir2] = Y.reshape(Y.size)
        restdir = np.arange(self.dim)
        restdir = np.delete(restdir, [dir1,dir2])
        grid_reducedD[:,restdir] = np.matlib.repmat(restcoords.reshape(1,len(restcoords)),X.size,1)
        grid = grid_reducedD
        #grid = self.get_6D_points(grid_reducedD)
        print(grid)
        
        if fun is not None:
            Z = fun(self.rescale_points(grid, self.ranges, self.integration_ranges),quantity = method).reshape(np.shape(X))
        elif method == "rbf":
            Z = self.Predict(self.rescale_points(grid, self.ranges, self.interpolation_ranges)).reshape(np.shape(X))
        elif method == "w":
            Z = mol.get_w(grid).reshape(np.shape(X))
        elif method == "E":
            Z = mol.get_energies(grid).reshape(np.shape(X))
        return X,Y,Z
    
    
    def Plot_2D_slice(self,X,Y,Z):
        plt.figure()
        contour_num = 100
        plt.contour(X,Y,Z, contour_num, linewidths=0.5, colors='k')               #contour plot with contour_num contour lines
        plt.contourf(X,Y,Z, contour_num)
        plt.colorbar()
        plt.xlabel("coord 1 / Angstroem")
        plt.ylabel("coord 2 / Angstroem")
        plt.title("Planar slice of PES")
        pass