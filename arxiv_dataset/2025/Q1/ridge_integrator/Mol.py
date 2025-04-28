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
import scipy.optimize
from scipy import special
import bisect as bisect
import math as math
import numpy.matlib
from ase.optimize import BFGS
from ase.constraints import Hookean
from ase.constraints import FixAtoms
from ase.constraints import FixedPlane
from ase import units
import itertools


#%% functions


def energy(Pore,mol,atom_num_pore,atom_num_mol):
    pos = Pore.get_positions()
    pos[atom_num_pore:atom_num_pore + atom_num_mol,:] = mol.reshape(atom_num_mol,3)
    Pore.set_positions(pos)
    E = Pore.get_potential_energy()
    return E


def translate_molecule(reference_mol,x,y,z):
    mol = np.zeros(np.shape(reference_mol))
    mol[:,0] = reference_mol[:,0] + x
    mol[:,1] = reference_mol[:,1] + y
    mol[:,2] = reference_mol[:,2] + z
    return mol 


def center(pos,masses):
    com = np.sum((pos * masses[:,None])/np.sum(masses), axis=0).reshape(3)
    return com 


def put_center_to_origin(pos,masses):
    com = center(pos,masses)
    pos = np.subtract(pos,com)
    return pos


#%% Define all the functions

class Mol:

    def __init__(self, beta, Pore, atom_num_mol, atom_num_pore, coordinates, fixed_coordinates, fixed_coordinate_values, 
                 relax = True, pos_information = [np.zeros(3),[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]], linear = False): 
        
        self.linear = linear
        self.beta = beta
        self.coordinates = coordinates
        
        self.atom_num_mol = atom_num_mol
        self.atom_num_pore = atom_num_pore
        self.atom_num_total = atom_num_mol + atom_num_pore
        
        self.mol_masses = Pore.get_masses()[self.atom_num_pore:atom_num_pore + atom_num_mol]
        self.pore_masses = Pore.get_masses()[0:self.atom_num_pore]
        self.total_masses = Pore.get_masses()
        
        self.Pore = Pore
        self.Set_pore_to_origin()
        self.Set_mol_to_origin()
        
        self.box_location = pos_information[0]
        self.c1, self.c2, self.c3 = pos_information[1][0], pos_information[1][1], pos_information[1][2]
        
        self.initial_point = Pore.get_positions()[-atom_num_mol:None,:]
        self.dim = len(coordinates)
        self.mol_reference = self.Set_mol_angles_zero(self.initial_point, self.mol_masses, self.atom_num_mol) if atom_num_mol > 1 else self.initial_point
        if relax == True: self.Relax_Pore_fixed_mol(fmax = 0.05) 
        self.E_ref =  energy(self.Pore, translate_molecule(self.mol_reference,0,0,6), atom_num_pore, atom_num_mol)
        
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
        
        self.pore_position = self.getPore_positions()
        
        pass 
    
        
    def get_6D_points(self,points):
        points_6D = np.zeros([int(points.size/len(self.coordinate_numbers)),6])
        points = points.reshape(1,points.size) if len(points.shape) == 1 else points
        if points.shape[1] != 6: 
            points_6D[:,self.coordinate_numbers] = points
            points_6D[:,self.fixed_coordinate_numbers] = np.ones([int(points.size/self.dim),len(self.fixed_coordinate_numbers)]) * self.fixed_coordinate_values
        else:
            points_6D = points
        return points_6D
        
    
    def get_energy(self,mol):
        pos = self.Pore.get_positions()
        pos[self.atom_num_pore:self.atom_num_total,:] = mol.reshape(self.atom_num_mol,3)
        self.Pore.set_positions(pos)
        E = self.Pore.get_potential_energy()
        return E


    def setPore_positions(self, pore_positions):
        pos = self.Pore.get_positions()
        pos[0:self.atom_num_pore,:] = pore_positions.reshape(self.atom_num_pore,3)
        self.Pore.set_positions(pos)
        return
    
    
    def getPore_positions(self):
        return self.Pore.get_positions()[0:self.atom_num_pore]
    
    
    def setMol_positions(self, mol_positions):
        pos = self.Pore.get_positions()
        pos[-self.atom_num_mol:None,:] = mol_positions.reshape(self.atom_num_mol,3)
        self.Pore.set_positions(pos)
        return
    
    
    def getMol_positions(self):
        return self.Pore.get_positions()[self.atom_num_pore:self.atom_num_total]
        
    
    def center(self,pos,masses):
        com = np.sum((pos * masses[:,None])/np.sum(masses), axis=0).reshape(3)
        return com 
    
    
    def plot_Pore(self,atoms):
        plt.figure()
        plt.scatter(atoms[:,0],atoms[:,1])
        return           
    
    
    def put_center_to_origin(self,pos,masses):
        com = self.center(pos,masses)
        pos = np.subtract(pos,com)
        return pos
    
    
    def Set_mol_angles_zero(self,mol,mol_masses,atom_num_mol):
        mol_centered = self.put_center_to_origin(mol, mol_masses)
        angles = self.get_angles(mol_centered)
        mol_reference_lin = self.rot_theta(self.rot_phi(mol_centered, -angles[1]), -angles[0])
        if np.any(np.array(self.coordinates) == "xi"):
            mol_reference = self.rot_xi(mol_reference_lin,-angles[2])
        else:
            mol_reference = mol_reference_lin
        self.setMol_positions(mol_reference)
        return mol_reference
    
    
    def Set_pore_to_origin(self):
        pore_centered = self.put_center_to_origin(self.Pore.get_positions()[0:self.atom_num_pore], self.pore_masses[0:self.atom_num_pore])
        self.setPore_positions(pore_centered)
        pass
    
    
    def Set_mol_to_origin(self):
        pore_centered = self.put_center_to_origin(self.Pore.get_positions()[self.atom_num_pore:self.atom_num_total], self.mol_masses)
        self.setMol_positions(pore_centered)
        pass
        
    
    def energy(self,mol):
        pos = self.Pore.get_positions()
        pos[self.atom_num_pore:self.atom_num_pore + self.atom_num_mol,:] = mol.reshape(self.atom_num_mol,3)
        self.Pore.set_positions(pos)
        E = self.Pore.get_potential_energy()
        return E
    
    
    def force(self,mol):
        pos = self.Pore.get_positions()
        pos[self.atom_num_pore:self.atom_num_pore + self.atom_num_mol,:] = mol.reshape(self.atom_num_mol,3)
        self.Pore.set_positions(pos)
        f = self.Pore.get_forces()[self.atom_num_pore:self.atom_num_pore + self.atom_num_mol,:] 
        f = f.reshape(self.atom_num_mol*3)
        return f
    
    
    def force_2D_array(self,mol):
        pos = self.Pore.get_positions()
        pos[self.atom_num_pore:self.atom_num_pore + self.atom_num_mol,:] = mol.reshape(self.atom_num_mol,3)
        self.Pore.set_positions(pos)
        f = self.Pore.get_forces()[self.atom_num_pore:self.atom_num_pore + self.atom_num_mol,:] 
        return f
    
    
    def Hess(self,pos,delta):
        x = np.zeros(self.atom_num_mol*3)
        x[:] = pos.reshape(self.atom_num_mol*3)
        Hess = np.zeros([self.dim,self.dim])
        for i in range(self.dim):
            Hess[:,i] = -(self.force(x + np.eye(self.dim)[:,i]*delta) - self.force(x))/delta
        Hess = 0.5*(np.triu(Hess)+np.triu(Hess).transpose()+np.tril(Hess)+np.tril(Hess).transpose() - 2*np.eye(self.dim)*np.diag(Hess))
        masses_long = np.matlib.repmat(self.mol_masses.reshape(self.atom_num_mol,1),1,3).reshape(self.atom_num_mol*3)
        Hess_mass = Hess / np.sqrt(masses_long[:,None]) / np.sqrt(masses_long[:,None].transpose())
        return Hess_mass, Hess
    
    
    def force_total(self,pos):
        pos = pos.reshape(self.atom_num_total,3)
        self.Pore.set_positions(pos)
        f = self.Pore.get_forces().reshape(self.atom_num_total*3)
        return f
                
    
    def Hess_total(self,pos,delta,masses):
        x = np.zeros(self.atom_num_total*3)
        x[:] = pos.reshape(self.atom_num_total*3)
        Hess = np.zeros([self.atom_num_total*3,self.atom_num_total*3])
        for i in range(self.atom_num_total*3):
            Hess[:,i] = -(self.force_total(x + np.eye(self.atom_num_total*3)[:,i]*delta) - self.force_total(x))/delta
        
        Hess = 0.5*(np.triu(Hess)+np.triu(Hess).transpose()+np.tril(Hess)+np.tril(Hess).transpose() - 2*np.eye(self.atom_num_total*3)*np.diag(Hess))
        masses_long = np.matlib.repmat(masses.reshape(self.atom_num_total,1),1,3).reshape(self.atom_num_total*3)
        Hess_mass = Hess/np.sqrt(masses_long[:,None]) / np.sqrt(masses_long[:,None].transpose())
        
        Hess_pore_only = Hess[0:self.atom_num_pore*3,0:self.atom_num_pore*3]
        return Hess_mass, Hess, Hess_pore_only
    
    
    def rot_molecule(self,mol,theta,phi,xi):
        mol_xi = self.rot_xi(mol,xi)
        mol_theta = self.rot_theta(mol_xi,theta)
        mol_rotated = self.rot_phi(mol_theta,phi)
        return mol_rotated
    
    
    def translate_molecule(self,reference_mol,x,y,z):
        mol_translated = np.zeros(np.shape(reference_mol))
        trans_vec = np.array([x,y,z])
        mol_translated[:,:] = reference_mol + trans_vec
        return mol_translated 
    
    
    def transrot_mol(self,mol,theta,phi,xi,c1,c2,c3 = 0):
        rotated_mol = self.rot_molecule(mol, theta, phi, xi)
        if not self.cart:
            trans_vec = self.box_location + self.c1 * c1*np.cos(c2) + self.c2 * c1*np.sin(c2) + self.c3 * c3
            translated_mol = self.translate_molecule(rotated_mol, *trans_vec)
        if self.cart:
            trans_vec = self.box_location + c1 * self.c1 + c2 * self.c2 + c3 * self.c3
            translated_mol = self.translate_molecule(rotated_mol, *trans_vec)
        return translated_mol
    
    
    def get_angles(self,mol):
        mol = self.put_center_to_origin(mol,self.mol_masses)
        ax_vec = (mol[1,:])/np.linalg.norm(mol[1,:])
        theta = np.arccos(ax_vec[2])
        phi = np.mod(math.atan2(ax_vec[1],ax_vec[0]),np.pi*2)
        
        mol_phi_inv = self.rot_phi_inv(mol,phi)
        mol_rotated = self.rot_theta_inv(mol_phi_inv,theta)
        
        if np.any(np.array(self.coordinates) == "xi"):
            xi = np.mod(math.atan2(mol_rotated[2,1],mol_rotated[2,0]),np.pi*2) 
        else:
            xi = 0
        return theta,phi,xi 
    
    
    def rot_phi(self,mol,phi):
        rot_mat = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
        mol_rot = np.matmul(mol,rot_mat)
        return mol_rot
    
    
    def rot_phi_inv(self,mol,phi):
        rot_mat = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
        mol_rot = np.matmul(mol,rot_mat)
        return mol_rot
    
    
    def rot_theta(self,mol,theta):
        rot_mat = np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
        mol_rot = np.matmul(mol,rot_mat)
        return mol_rot
    
    
    def rot_theta2(self,mol,theta,phi):
        rot_mat = np.array([[np.cos(phi),0,np.sin(phi)],[0,0,0],[-np.sin(phi),0,np.cos(phi)]])
        mol_phi_redo = self.rot_phi_inv(mol,phi)
        mol_rot_theta = np.matmul(rot_mat,mol_phi_redo)
        mol_final_theta = self.rot_phi(mol_rot_theta,phi)
        return mol_final_theta
    
    
    def rot_theta_inv(self,mol,theta):
        rot_mat = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        mol_rot_theta = np.matmul(mol,rot_mat)
        return mol_rot_theta
    
    
    def rot_xi(self,mol,xi):
        rot_mat = np.array([[np.cos(xi),np.sin(xi),0],[-np.sin(xi),np.cos(xi),0],[0,0,1]])
        mol_rot = np.matmul(mol,rot_mat)
        return mol_rot
    
    
    def zscan(self,delta_z,length,start_pos,start_z):
        energies = np.zeros(length)
        pos = np.zeros(np.shape(start_pos))
        pos[:,:] = start_pos 
        pos[:,2] = pos[:,2] + start_z
        for i in range(length):
            pos[:,2] = pos[:,2] + delta_z     
            E2 = self.energy(pos)
            energies[i] = E2
        return np.linspace(self.center(start_pos,self.mol_masses)[2]+start_z,self.center(start_pos,self.mol_masses)[2] + length*delta_z + start_z,length),energies
    
    
    def generate_angle_points(self,xi_num,theta_num,phi_num):
        xis = np.linspace(np.pi/xi_num,2*np.pi- np.pi/xi_num,xi_num)
        thetas = np.linspace(np.pi/(2*theta_num),np.pi - np.pi/(2*theta_num),theta_num) 
        phis = np.linspace(np.pi/phi_num,2*np.pi - np.pi/phi_num,phi_num)
        Mask = np.zeros([xi_num,theta_num,phi_num])
        return xis, thetas, phis, Mask
        
    
    def get_rotational_unit_vectors(self,mol,delta,xi,theta,phi):
        Xi_hat = (self.rot_molecule(mol,theta,phi,xi) - self.rot_molecule(mol,theta,phi,xi + delta))/delta * np.sqrt(self.mol_masses[:,None])
        phi_hat = (self.rot_molecule(mol,theta,phi,xi) - self.rot_molecule(mol,theta,phi + delta,xi))/delta * np.sqrt(self.mol_masses[:,None])
        theta_hat = (self.rot_molecule(mol,theta,phi,xi) - self.rot_molecule(mol,theta + delta,phi,xi))/delta * np.sqrt(self.mol_masses[:,None])
        return Xi_hat, phi_hat, theta_hat
    
    
    def get_basis_vectors(self,mol,theta,phi,xi,c1,c2,z,delta = 1e-4):
        theta_hat = np.sqrt(self.mol_masses[:,None]) * (self.transrot_mol(mol,theta + delta,phi,xi,c1,c2,z) - self.transrot_mol(mol,theta,phi,xi,c1,c2,z))/delta 
        phi_hat = np.sqrt(self.mol_masses[:,None]) * (self.transrot_mol(mol,theta,phi + delta,xi,c1,c2,z) - self.transrot_mol(mol,theta,phi,xi,c1,c2,z))/delta 
        xi_hat = np.sqrt(self.mol_masses[:,None]) * (self.transrot_mol(mol,theta,phi,xi + delta,c1,c2,z) - self.transrot_mol(mol,theta,phi,xi,c1,c2,z))/delta 
        c1_hat = np.sqrt(self.mol_masses[:,None]) * (self.transrot_mol(mol,theta,phi,xi,c1 + delta,c2,z) - self.transrot_mol(mol,theta,phi,xi,c1,c2,z))/delta 
        c2_hat = np.sqrt(self.mol_masses[:,None]) * (self.transrot_mol(mol,theta,phi,xi,c1,c2 + delta,z) - self.transrot_mol(mol,theta,phi,xi,c1,c2,z))/delta 
        z_hat = np.sqrt(self.mol_masses[:,None]) * (self.transrot_mol(mol,theta,phi,xi,c1,c2,z + delta) - self.transrot_mol(mol,theta,phi,xi,c1,c2,z))/delta 
        return theta_hat, phi_hat, xi_hat, c1_hat, c2_hat, z_hat
    
    
    def get_jacobian(self,point):
        
        basis_vecs = self.get_basis_vectors(self.mol_reference,*self.get_6D_points(point)[0])
        if not self.linear and self.atom_num_mol > 2:
            theta_norm = basis_vecs[0] / np.linalg.norm(basis_vecs[0])
            phi_norm = basis_vecs[1] / np.linalg.norm(basis_vecs[1])
            theta_gram_schmidt = basis_vecs[0]
            phi_gram_schmidt = basis_vecs[1] - np.sum(theta_norm*basis_vecs[1]) * theta_norm
            phi_gram_schmidt_norm = phi_gram_schmidt / np.linalg.norm(phi_gram_schmidt)
            xi_gram_schmidt = basis_vecs[2] - np.sum(theta_norm*basis_vecs[2]) * theta_norm - np.sum(phi_gram_schmidt_norm*basis_vecs[2]) * phi_gram_schmidt_norm
            Jacobian_rot = np.linalg.norm(theta_gram_schmidt) * np.linalg.norm(phi_gram_schmidt) * np.linalg.norm(xi_gram_schmidt)
        elif self.linear:
            Jacobian_rot = np.linalg.norm(basis_vecs[0]) * np.linalg.norm(basis_vecs[1])
        elif self.atom_num_mol == 1:
            Jacobian_rot = 1
           
        if not np.any(np.array(self.coordinates) == "z"):
            Jacobian_trans = np.linalg.norm(basis_vecs[3]) * np.linalg.norm(basis_vecs[4])
        else:
            Jacobian_trans = np.linalg.norm(basis_vecs[3]) * np.linalg.norm(basis_vecs[4]) * np.linalg.norm(basis_vecs[5])
            
        return Jacobian_rot * Jacobian_trans
    
    
    def get_jacobians(self,points):
        return np.array([self.get_jacobian(points[i]) for i in range(len(points))])
    
    
    def get_max_jac_basis(self):
        all_basis_vecs = np.array([*self.get_basis_vectors(self.mol_reference, np.pi/2, 0, 0, 1, 0, 0)]) * np.sqrt(self.mol_masses[None,:,None])
        basis_vecs = all_basis_vecs[self.coordinate_numbers,:,:]
        lengths_basis_vecs = np.sqrt((basis_vecs**2).sum(axis = (1,2)))
        return np.prod(lengths_basis_vecs)
    
    
    def get_gradient(self,internal_coords_point):
        internal_coords_point = internal_coords_point.reshape(int(self.dim)) if len(internal_coords_point.shape) == 2 else internal_coords_point
        f = np.nan
        while np.any(np.isnan(f)):
            f = self.force(self.transrot_mol(self.mol_reference, *(internal_coords_point + 1e-2 * np.random.rand(len(internal_coords_point))))).reshape(self.atom_num_mol,3)
        xi_hat, phi_hat, theta_hat, r_hat, chi_hat, z_hat = self.get_unit_vectors(self.mol_reference,*internal_coords_point)
        vec_list = [theta_hat, phi_hat, xi_hat, r_hat, chi_hat, z_hat]
        grad = -np.array([np.sum(vec_list[i] * f) for i in range(len(vec_list))]) 
        print("f =",f)
        return grad
    
    
    def get_gradient_w(self,internal_coords_point):
        internal_coords_point = internal_coords_point.reshape(int(self.dim)) if len(internal_coords_point.shape) == 2 else internal_coords_point
        E = self.get_energies(internal_coords_point)
        J = np.abs(np.sin(internal_coords_point[0])) * internal_coords_point[3]
        grad_E = self.get_gradient(internal_coords_point)
        grad_J = np.zeros(6)
        if not self.cart:
            grad_J[0] = np.cos(internal_coords_point[0]) * internal_coords_point[3]
            grad_J[3] = np.abs(np.sin(internal_coords_point[0]))
        else:
            grad_J[0] = np.cos(internal_coords_point[0]) 
        
        grad_w = -self.beta*np.exp(-self.beta*E) * grad_E * J + np.exp(-self.beta*E) * grad_J
        return grad_w[self.coordinate_numbers]
        
    
    def get_gradient_array(self,internal_coords,quantity = "E"):
        grads = np.zeros_like(internal_coords)
        for i in range(len(internal_coords)):
            grads[i] = self.get_gradient_w(self.get_6D_points(internal_coords[i])[0]) if quantity == "w" else self.get_gradient(self.get_6D_points(internal_coords[i])[0])[self.coordinate_numbers]
        return grads
    
    
    def check_grad(self,TR_point,k,beta,fun,ranges,interpolation_ranges,Int,delta = 1e-3):
        point = self.transrot_mol(self.mol_reference,*self.get_6D_points(TR_point)[0])
        E1 = self.energy(point) - self.E_ref
        E1_rbf = fun(Int.rescale_points(TR_point, ranges, interpolation_ranges).reshape(len(TR_point))) 
        
        TR_point[k] = TR_point[k] + delta
        point = self.transrot_mol(self.mol_reference,*self.get_6D_points(TR_point)[0])
        E2 = self.energy(point) - self.E_ref
        E2_rbf = fun(Int.rescale_points(TR_point, ranges, interpolation_ranges).reshape(len(TR_point))) 

        num_grad = (E2 - E1) / delta
        num_grad_rbf = (E2_rbf - E1_rbf) / delta
        return num_grad, num_grad_rbf
    
    
    def check_rbf_grad(self,point,k,beta,fun,delta = 1e-3):
        E1_rbf = fun(point) 
        point2 = np.copy(point)
        point2[:,k] = point2[:,k] + delta
        E2_rbf = fun(point2)
        num_grad_rbf = (E2_rbf - E1_rbf) / delta
        return num_grad_rbf
    
    
    def boxsize(self,Xi_hat, phi_hat, theta_hat):
        theta_norm = theta_hat / np.sqrt((theta_hat**2).sum())                          # norm theta
        phi_norm = phi_hat / np.sqrt((phi_hat**2).sum())
        phi_hat_ortho = phi_hat - (theta_norm*phi_hat).sum() * theta_hat
        xi_hat_ortho = Xi_hat - (theta_norm*Xi_hat).sum() * theta_norm - (phi_norm*Xi_hat).sum() * phi_norm
        
        VOLUME = np.sqrt((theta_hat**2).sum() * (phi_hat_ortho**2).sum() * (xi_hat_ortho**2).sum())
        return VOLUME
    
    
    def MakeMask(self,xi_num, theta_num, phi_num):
        xis, thetas, phis, Mask = self.generate_angle_points(xi_num,theta_num,phi_num)
        for i in range(len(xis)):
            for j in range(len(thetas)):
                for k in range(len(phis)):
                    Xi_hat, phi_hat, theta_hat = self.get_rotational_unit_vectors(self.mol_reference,1e-6,xis[i], thetas[j], phis[k])
                    VOL = self.boxsize(Xi_hat, phi_hat, theta_hat)
                    Mask[i,j,k] = VOL
        return Mask, xis, thetas, phis
    
    
    def draw_angle_pos(self,Mask, xis, thetas, phis):
        WEIGHTS = Mask
        WEIGHTS_vec = WEIGHTS.reshape(Mask.size)
        WEIGHTS_vec_cum = np.cumsum(WEIGHTS_vec)
        WEIGHTS_vec_cum_normed = WEIGHTS_vec_cum/WEIGHTS_vec_cum[-1]
        
        r = np.random.rand()
        
        vec_pos = bisect.bisect_left(WEIGHTS_vec_cum_normed,r)
        
        index_Mask = np.unravel_index(np.ravel((vec_pos), order = "C"), Mask.shape, order = "C")
        print(index_Mask)
        xi = float(xis[index_Mask[0]])
        theta = float(thetas[index_Mask[1]])
        phi = float(phis[index_Mask[2]])
        
        return xi, theta, phi 
        
    
    def getrandpos(self,mol,interval_x,interval_y,interval_z):
        x = np.random.rand(1) * interval_x - interval_x/2 
        y = np.random.rand(1) * interval_y - interval_y/2
        z = np.random.rand(1) * interval_z - interval_z/2
        phi = np.random.rand(1) * 2*np.pi
        theta = np.arccos(-2*np.random.rand(1)+1)
        xi = np.random.rand(1) * 2*np.pi
        #xi, theta, phi = self.draw_angle_pos(Mask, xis, thetas, phis)
        mol = self.rot_molecule(self.mol_reference,theta,phi,xi)
        mol = self.translate_molecule(mol,x,y,z)
        return mol
    
    
    def get_phase_spacevol(self,interval_x, interval_y, interval_z, Mask):
        dphi_dtheta_dxi = 4*np.pi**3 / np.size(Mask)
        vol_rotational = dphi_dtheta_dxi * np.sum(Mask)
        vol_translational = np.sqrt(np.sum(self.mol_masses))**3 * interval_x * interval_y * interval_z
        return vol_rotational * vol_translational, vol_rotational
        
    
    def getV(self,molpos1,molpos2):
        V = np.zeros(np.size(molpos1,0))
        for i in range(np.size(molpos1,0)):
            V[i] = self.energy(np.array([molpos1[i][0],molpos1[i][1],molpos1[i][2],molpos2[i][0],molpos2[i][1],molpos2[i][2]]))
        return V
    
    
    def xi_theta_phi_scan(self,Mask,x,y,z):
        xis, thetas, phis, Mask = self.generate_angle_points(np.shape(Mask)[0],np.shape(Mask)[1],np.shape(Mask)[2])
        Energies = np.zeros(np.shape(Mask))
        for i in range(len(xis)):
            for j in range(len(thetas)):
                for k in range(len(phis)):
                    Pos = self.translate_molecule(self.rot_molecule(self.mol_reference,thetas[j],phis[k],xis[i]),x,y,z)
                    Energies[i,j,k] = self.energy(Pos)
        print('one angle cycle done')
        return Energies
    
    
    def Energy_scan(self,x_interval,y_interval,x_num,y_num,Mask):
        Energy_scan = np.zeros([x_num,y_num,np.shape(Mask)[0],np.shape(Mask)[1],np.shape(Mask)[2]])
        Supermask = np.zeros([x_num,y_num,np.shape(Mask)[0],np.shape(Mask)[1],np.shape(Mask)[2]])
        x_values = np.linspace(-x_interval/2,x_interval/2,x_num)
        y_values = np.linspace(-y_interval/2,y_interval/2,y_num)
        for i in range(x_num):
            for j in range(y_num):
                Energy_scan[i,j,:,:,:] = self.xi_theta_phi_scan(Mask,x_values[i],y_values[j], 0)
                Supermask[i,j,:,:,:] = Mask
        return Energy_scan, Supermask
    
    
    def get_Z(self,center_of_Box, beta, number_of_samples,boxvol,Mask):
        Z = 0
        Z_vec_x = np.zeros(number_of_samples)
        Z_v = (2*np.pi / beta)**3                                                #velocity part of partition sum
        W_vec = np.zeros(number_of_samples)
        Poses = list()
        for i in range(number_of_samples):
            Position = self.getrandpos(self.mol_reference,boxvol[0],boxvol[1],boxvol[2])
            E = self.energy(Position)
            #print(E - self.E_ref)
            W = np.exp(-beta*(E-self.E_ref))   
            Z = Z + W
            Z_vec_x[i] = Z
            W_vec[i] = W
            Poses.append(Position)
        Z_vec = Z_vec_x*Z_v/np.linspace(1,number_of_samples,number_of_samples) * self.get_phase_spacevol(boxvol[0], boxvol[1], boxvol[2], Mask)[0]
        Z = Z_vec[-1]
        return Z, Poses, Z_v
    
    
    def get_velocity(self,masses,beta):
        rands = np.random.rand(3,3)
        vel = special.erfinv(rands)*np.sqrt(2/(masses[:,None]*beta))
        signs = 2*np.random.randint(0,2,size=(3,3))-1
        vel = vel * signs
        return vel
    
    
    def calc_k(self,Mask_E, Scan_E, x_num, y_num, interval_x_scan, interval_y_scan,Z,Z_v, beta):
        dphi_dtheta_dxi_dx_dy = 4*np.pi**3 /np.size(Mask_E) *(x_num * y_num) * (interval_x_scan/x_num)*(interval_y_scan/y_num) * np.sqrt(np.sum(self.mol_masses))**2
        k = 1/(Z / Z_v) * (2*np.pi)**(-1/2)*(beta)**(-1/2) * np.sum(Mask_E*np.exp(-beta*(Scan_E-self.E_ref))) * dphi_dtheta_dxi_dx_dy *  2 *units.fs * 1e6
        return k
    
    
    def calc_k_with_partsum_monte(self,beta, Mask, total_boxvol,center_of_Box,boxvol, monte_carlo_num, monte_carlo_num_ridge):
        Z,dummy,Z_v = self.get_Z(center_of_Box, beta, monte_carlo_num, total_boxvol, Mask)
        k = 1/Z  * (2*np.pi)**(-1/2)*(beta)**(-1/2) * self.get_Z(center_of_Box, beta, monte_carlo_num_ridge, boxvol, Mask)[0]  / (np.sqrt(np.sum(self.mol_masses)) * boxvol[2]) *  2 *units.fs * 1e6
        return k 
    
    
    def calcEyring_k(self,beta,Z,saddle_point):
        H = self.Hess(saddle_point, 0.0001)
        Hess_saddle_eigvals = np.linalg.eig(H[0])[0]
        print(Hess_saddle_eigvals)
        Hess_saddle_positive_nonvib_eigvals = Hess_saddle_eigvals[Hess_saddle_eigvals > 0]
        Hess_saddle_positive_nonvib_eigvals = Hess_saddle_positive_nonvib_eigvals[Hess_saddle_positive_nonvib_eigvals < 1]
        print(Hess_saddle_positive_nonvib_eigvals)
        k = np.exp(-beta * (self.get_energy(saddle_point) - self.E_ref)) * 1/Z *(2*np.pi)**(10/2)*beta**(-12/2)/(np.sqrt(np.product(Hess_saddle_positive_nonvib_eigvals))) * 2 *units.fs * 1e6 
        return k
    
    
    def get_wavenumbers(self,saddle_point, min_point):
        H_min = self.Hess(min_point, 0.0001)[0]
        H_saddle = self.Hess(saddle_point, 0.0001)[0]
        Hess_saddle_eigvals = np.linalg.eig(H_saddle)[0]   
        Hess_min_eigvals = np.linalg.eig(H_min)[0]
        return np.sqrt(Hess_saddle_eigvals) * units.s / (1e2 * 3*10**8 * 2* np.pi), np.sqrt(Hess_min_eigvals) * units.s / (1e2 * 3*10**8 * 2* np.pi)
    
    
    def conv_to_wavenumbers(self,w):
        return w  * units.s / (1e2 * 3*10**8 * 2* np.pi)
    
    
    def calcEyring_poormans(self,beta,saddle_point,Min_pos):
        H_min = self.Hess(Min_pos, 0.0001)[0]
        H_saddle = self.Hess(saddle_point, 0.0001)[0]
        
        Hess_saddle_eigvals = np.linalg.eig(H_saddle)[0]
        Hess_saddle_positive_eigvals = np.sort(Hess_saddle_eigvals)[1:None]     
        Hess_min_eigvals = np.linalg.eig(H_min)[0]

        print(self.get_energy(saddle_point) - self.get_energy(Min_pos))
        k = np.exp(-beta * (self.get_energy(saddle_point) - self.get_energy(Min_pos))) * np.sqrt(np.product(Hess_min_eigvals))*(2*np.pi)**(-1) / np.sqrt(np.product(Hess_saddle_positive_eigvals)) * 2 *units.fs * 1e6 
        return k
    
    
    def calcEyring_Grimme(self,beta,saddle_point,Min_pos,I_av,n = 3):
        
        H_min = self.Hess(Min_pos, 0.0001)[0]
        H_saddle = self.Hess(saddle_point, 0.0001)[0]
        
        Saddle_eigvals = np.linalg.eig(H_saddle)[0]
        Saddle_vib_eigvals = np.sort(Saddle_eigvals)[(n+1):None]  
        Saddle_rot_eigvals = np.sort(Saddle_eigvals)[1:n+1] 
        
        Min_eigvals = np.linalg.eig(H_min)[0]
        Min_vib_eigvals = np.sort(Min_eigvals)[n:None]  
        Min_rot_eigvals = np.sort(Min_eigvals)[0:n] 
        
        k_without_grimme = np.exp(-beta * (self.get_energy(saddle_point) - self.get_energy(Min_pos))) * np.sqrt(np.product(Min_vib_eigvals))*(2*np.pi)**(-1) / np.sqrt(np.product(Saddle_vib_eigvals)) * 2 *units.fs * 1e6  
        
        mus_saddle = 1.055 * 1e-34 / ( 2 * np.sqrt(Saddle_rot_eigvals) * units.s)
        mus_min = 1.055 * 1e-34 / ( 2 * np.sqrt(Saddle_rot_eigvals) * units.s)
        I_prime_saddle = mus_saddle * I_av / (mus_saddle + I_av)
        I_prime_min = mus_min * I_av / (mus_min + I_av)
        weights_saddle = 1/(1 - self.conv_to_wavenumbers(np.sqrt(Saddle_rot_eigvals))/100)
        weights_min = 1/(1 - self.conv_to_wavenumbers(np.sqrt(Min_rot_eigvals))/100)
        Z_grimmecorr_saddle = np.exp(weights_saddle * np.log(np.prod(np.sqrt(Saddle_rot_eigvals) / beta)) + (1 - weights_saddle) * np.log((8*np.pi * I_prime_saddle / beta)**(n/2)))
        Z_grimmecorr_min = np.exp(weights_saddle * np.log(np.prod(np.sqrt(Min_rot_eigvals) / beta)) + (1 - weights_min) * np.log((8*np.pi * I_prime_min / beta)**(n/2)))
        grimme_corr = Z_grimmecorr_saddle / Z_grimmecorr_min
        
        return k_without_grimme * grimme_corr
    
    
    def calcEyring_Grimme_min(self,beta,saddle_point,Min_pos,n = 3):
        H_min = self.Hess(Min_pos, 0.00001)[0]
        H_saddle = self.Hess(saddle_point, 0.00001)[0]
        
        Hess_saddle_eigvals = np.linalg.eig(H_saddle)[0]
        Hess_saddle_positive_eigvals = np.sort(Hess_saddle_eigvals)[(n+1):None]   
        Hess_min_eigvals = np.linalg.eig(H_min)[0]
        Hess_min_eigvals = np.sort(Hess_min_eigvals)[n:None] 
        k = np.exp(-beta * (self.get_energy(saddle_point) - self.get_energy(Min_pos))) * np.sqrt(np.product(Hess_min_eigvals))*(2*np.pi)**(-1) / np.sqrt(np.product(Hess_saddle_positive_eigvals)) * 2 *units.fs * 1e6 
        return k
        
    
    def calcEyring_total(self,beta,Z,saddle_point,Min_pos):
        H_min = self.Hess_total(Min_pos, 0.0001)
        H_saddle = self.Hess_total(saddle_point, 0.0001)
        
        Hess_saddle_eigvals = np.linalg.eig(H_saddle[0])[0]
        Hess_saddle_positive_eigvals = Hess_saddle_eigvals[Hess_saddle_eigvals > 1e-4]    
        Hess_min_eigvals = np.linalg.eig(H_min[0])[0]
        Hess_min_vib_eigvals = Hess_min_eigvals[Hess_min_eigvals > 1e-4]
        
        Z_x = np.sqrt(np.product(2*np.pi / Hess_min_vib_eigvals / beta))
        print(Hess_saddle_eigvals, Hess_min_eigvals)
        k = 1/Z_x*(2*np.pi)**(-1/2)*beta**(-1/2) * np.sqrt(np.product((2*np.pi / beta / Hess_saddle_positive_eigvals)))
        return k
                                        
    
    def find_saddle_point_in_plane(self,Hess, x_int,x_num,y_int,y_num,z_range,z_num, angle_num):
        info_list = list()
        saddle_list = list()
        z_s = np.linspace(-z_range/2,z_range/2,z_num)
        for i in range(z_num):
            mol = self.translate_molecule(self.mol_reference, 0, 0, z_s[i])
            
            xs = np.linspace(-x_int/2,x_int/2,x_num)
            ys = np.linspace(-y_int/2,y_int/2,y_num)
        
            Mask, xis, thetas, phis = self.MakeMask(xi_num = 12, theta_num = 12, phi_num = 12)
            Scan_E_int, Mask_E_int = self.Energy_scan(x_int,y_int,x_num,y_num,Mask)
        
            a = np.where(Scan_E_int == Scan_E_int.min())
            x = xs[a[0]][0]
            y = ys[a[1]][0]
            xi = xis[a[2]][0]
            theta = thetas[a[3]][0]
            phi = phis[a[4]][0]
            rotated_mol = self.rot_molecule(mol,theta,phi,xi)
            saddle_point_found = self.translate_molecule(rotated_mol,x,y,0)
            info_list.append(np.linalg.eig(Hess[0](saddle_point_found, 1e-4))[0])
            saddle_list.append(saddle_point_found)
        return info_list, saddle_list  
    
    
    def Relax_Pore_fixed_mol(self, fmax = 0.05):
        #c2 = FixedPlane(list(np.arange(self.atom_num_pore)), self.c3)
        c1 = FixAtoms(list(np.arange(self.atom_num_mol) + self.atom_num_pore))
        self.Pore.set_constraint([c1])
        opt_Pore = BFGS(self.Pore)
        opt_Pore.run(fmax = fmax)
        self.Pore.set_constraint([])
        return self.Pore.get_positions()[-self.atom_num_mol:None,:]
    
    
    def get_grid(self,ranges,rescaled_ranges,gridnum = 5 * np.ones(5)):
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
    
    
    def get_energies(self,gridpoints):
        gridpoints = gridpoints.reshape(1,len(gridpoints)) if len(gridpoints.shape) == 1 else gridpoints 
        energies = np.zeros(len(gridpoints))
        for i in range(len(energies)):
            mol = self.transrot_mol(self.mol_reference, *self.get_6D_points(gridpoints[i])[0]) #self.translate_molecule(rotated_mol,gridpoints[i,3] * np.cos(gridpoints[i,4]), gridpoints[i,3] * np.sin(gridpoints[i,4]), 0)
            energies[i] = self.energy(mol) - self.E_ref
        return energies
    
    
    def get_similarity_fun(self,angles1,angles_diff):
        point_base = np.hstack((angles1,np.array([0,0,0])))
        point_rot = np.hstack((angles1 + angles_diff,np.array([0,0,0])))
        mol_base = self.transrot_mol(self.mol_reference, *point_base)
        mol_rot = self.transrot_mol(self.mol_reference, *point_rot)
        simfun = np.einsum("ijk,ij->",1/((mol_rot[None,:,:] - mol_base[:,None,:])**2 + 1e-5), (self.mol_masses[:,None] - self.mol_masses[None,:]) == 0)
        return simfun
    
    
    def get_bravais_lattice(self,n):
        
        base_angles = np.ones(3) * 1.4
        
        def simfun(angle_array):
            return -self.get_similarity_fun(base_angles,angle_array)
        
        lattice_points = np.zeros([n,3])
        simfun_vals = np.zeros(n)
        start_points = np.random.rand(n,3) * 2*np.pi 
        positions = list()
        for i in range(n):
            opt_object = scipy.optimize.minimize(simfun, start_points[i], method = "BFGS")
            lattice_points[i,:] = opt_object.x
            simfun_vals[i] = simfun(opt_object.x)
            positions.append(self.transrot_mol(self.mol_reference, *np.hstack((base_angles + opt_object.x,np.array([0,0,0])))))
        return lattice_points, simfun_vals, positions
    
    
    def get_w(self,gridpoints):
        gridpoints = gridpoints.reshape(1,int(self.dim)) if len(gridpoints.shape) == 1 else gridpoints
        energies = np.zeros(len(gridpoints))
        for i in range(len(energies)):
            mol = self.transrot_mol(self.mol_reference, *self.get_6D_points(gridpoints[i])[0])
            energies[i] = self.energy(mol) - self.E_ref
            
        Jac = self.get_jacobians(gridpoints)
        
        # if not self.cart and self.atom_num_mol > 1: 
        #     r_arg = np.where(self.coordinate_numbers == 3)[0][0]
        #     theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
        #     Jac = np.abs(np.sin(gridpoints[:,theta_arg])) * gridpoints[:,r_arg]
        # elif self.cart and self.atom_num_mol > 1:
        #     theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
        #     Jac = np.abs(np.sin(gridpoints[:,theta_arg]))
        # elif not self.cart and self.atom_num_mol == 1:
        #     r_arg = np.where(self.coordinate_numbers == 3)[0][0]
        #     Jac = gridpoints[:,r_arg]
        # elif self.cart and self.atom_num_mol == 1:
        #     Jac = np.ones(len(gridpoints))
       
        w = np.exp(-self.beta * energies) * Jac
        
        return w
    
    
    def get_w_from_E(self,Es,gridpoints):
        gridpoints = gridpoints.reshape(1,int(self.dim)) if len(gridpoints.shape) == 1 else gridpoints
        
        Jac = self.get_jacobians(gridpoints)
        
        # if not self.cart and self.atom_num_mol > 1: 
        #     r_arg = np.where(self.coordinate_numbers == 3)[0][0]
        #     theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
        #     Jac = np.abs(np.sin(gridpoints[:,theta_arg])) * gridpoints[:,r_arg]
        # elif self.cart and self.atom_num_mol > 1:
        #     theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
        #     Jac = np.abs(np.sin(gridpoints[:,theta_arg]))
        # elif not self.cart and self.atom_num_mol == 1:
        #     r_arg = np.where(self.coordinate_numbers == 3)[0][0]
        #     Jac = gridpoints[:,r_arg]
        # elif self.cart and self.atom_num_mol == 1:
        #     Jac = np.ones(len(gridpoints))
       
        w = np.exp(-self.beta * Es) * Jac
        
        return w
    
    
    def get_grad_J(self,gridpoints):
        grad_J = np.zeros_like(gridpoints)
        theta_arg = np.where(self.coordinate_numbers == 0)[0][0]
        r_arg = np.where(self.coordinate_numbers == 3)[0][0]
        grad_J[:,theta_arg] = np.cos(gridpoints[:,theta_arg]) * gridpoints[:,r_arg]
        grad_J[:,r_arg] = np.abs(np.sin(gridpoints[:,theta_arg]))
        
        return grad_J
    
    
    def get_energies_and_forces(self,gridpoints):
        energies = np.zeros(len(gridpoints))
        #grad = np.zeros([len(gridpoints),5])
        for i in range(len(energies)):
            rotated_mol = self.rot_molecule(self.mol_reference,gridpoints[i,0],gridpoints[i,1],gridpoints[i,2])
            mol = self.translate_molecule(rotated_mol,gridpoints[i,3] * np.cos(gridpoints[i,4]), gridpoints[i,3] * np.sin(gridpoints[i,4]), 0)
            energies[i] = self.energy(mol) - self.E_ref
            #print(gridpoints[i])
            #grad[i,:] = self.get_gradient(gridpoints[i],level = "ll")

        return energies #, grad
    
    
    def get_energies_for_internal_coords(self,beta,int_coord_array):
        energies = np.zeros(len(int_coord_array))
        for i in range(len(energies)):
            rotated_mol = self.rot_molecule(self.mol_reference,int_coord_array[i,0],int_coord_array[i,1],int_coord_array[i,2])
            mol = self.translate_molecule(rotated_mol,int_coord_array[i,3] * np.cos(int_coord_array[i,4]), int_coord_array[i,3] * np.sin(int_coord_array[i,4]), 0)
            energies[i] = self.energy(mol) - self.E_ref
        return energies
    
    
    def get_I_average(self):
        I = np.einsum("i,ij,ik->jk",self.mol_masses,self.mol_reference,self.mol_reference)
        return np.sum(np.linalg.eig(I)[0])/3
            
