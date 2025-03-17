import numpy as np
from numpy import sin, cos, abs, exp, sqrt, pi
import torch
from random import random

class Operations:
    """

    This class creates the quantum operations that will be used.
    
    ...

    Attributes
    ----------
    qr : class
        The quantum register created using the QuantumRegister class.

    Methods
    -------
    ·XXZ_Hamiltonian:
        Parametrized XXZ Hamiltonian
    ·Global_XX_Layer:
        Layer formed by single-qubit rotations around the z axis acting on all qubits, and an entangling XX global gate.
    ·Rz_Layer:
        Layer of single-qubit rotations around the z axis acting on all qubits.
    ·Ent_XX_Layer:
        Global XX entangling gate.
    ·Ent_XX_Hamiltonian:
        Entangling XX Hamiltonian.
    """

    def __init__(self,qr):
        self.qr = qr 
        self.size = qr.size

    def XXZ_Hamiltonian(self, Delta):
        '''
        Parametrized XXZ Hamiltonian
        H = \sum_i \sigma^i_x \sigma^{i+1}_x + \sigma^i_y \sigma^{i+1}_y - \Delta * \sigma^i_z \sigma^{i+1}_z

        Parameters:
        -----------
        Delta : float
            Parameter describing the adiabatic path.
        '''
        H = 0
        for i in range(0, self.size-1):
            H += 2*(self.qr.Sp_i(i)@self.qr.Sm_i(i+1) + self.qr.Sm_i(i)@self.qr.Sp_i(i+1)) + Delta*self.qr.Z_i(i)@self.qr.Z_i(i+1)

        return H

    def Global_XX_Layer(self, Thetas, gamma, xi, Sp_list, Sm_list, Z_list):
        '''
        Layer formed by single-qubit rotations around the z axis acting on all qubits, and an entangling XX global gate.

        Parameters:
        -----------
        Thetas : list
            Rotation angles
        gamma : tensor
            Interaction strength
        xi : tensor
            Interaction range
        Sp_list : list
            List of \sigma_+ operators acting locally on each qubit
        Sm_list : list
            List of \sigma_- operators acting locally on each qubit
        Z_list : list
            List of \sigma_z operators acting locally on each qubit

        '''
        Rz_layer = self.Rz_Layer(Thetas, Z_list)
        ent_layer = self.Ent_XX_Layer(gamma, xi, Sp_list, Sm_list)

        tot_layer =  Rz_layer @ ent_layer

        return tot_layer

    def Rz_Layer(self, Thetas, Z_list):
        '''
        Layer of single-qubit rotations around the z axis acting on all qubits.

        Parameters:
        -----------
        Thetas : list
            Rotation angles
        Z_list : list
            List of \sigma_z operators acting locally on each qubit
                
        '''        
        M = torch.complex(torch.einsum('i,ikl->kl', Thetas, torch.real(Z_list)), torch.zeros((2**self.size, 2**self.size), dtype=torch.float64))
        rot_layer = torch.diag(torch.exp(torch.diag(-1j*M/2)))

        return rot_layer

    def Ent_XX_Layer(self, gamma, xi, Sp_list, Sm_list):
        '''
        Global XX entangling gate given by U = exp(-iHt), where H = \sum_{ij} \gamma \exp{-|r_i - r_j| / \xi} \sigma_{+} \sigma_{-} + H.c.      

        Parameters:
        -----------
        gamma : tensor
            Interaction strength
        xi : tensor
            Interaction range
        Sp_list : list
            List of \sigma_+ operators acting locally on each qubit
        Sm_list : list
            List of \sigma_- operators acting locally on each qubit           
        '''

        H = self.Ent_XX_Hamiltonian(gamma, xi, Sp_list, Sm_list)
        U = torch.linalg.matrix_exp(-1j*H*0.25)
        
        return U

    def Ent_XX_Hamiltonian(self, gamma, xi, Sp_list, Sm_list):
        '''
        Entangling XX Hamiltonian: H = \sum_{ij} \gamma \exp{-|r_i - r_j| / \xi} \sigma_{+} \sigma_{-} + H.c.   

        Parameters:
        -----------
        gamma : tensor
            Interaction strength
        xi : tensor
            Interaction range
        Sp_list : list
            List of \sigma_+ operators acting locally on each qubit
        Sm_list : list
            List of \sigma_- operators acting locally on each qubit   

        This method can be employed for the all-to-all global entangling gate by making the substitution
        exp_i = torch.exp(x_i / xi**2) ==> torch.exp(x_i * 0.0)
        exp_j = torch.exp(-x_j / xi**2) ==> torch.exp(-x_j * 0.0)
        i.e. by considering an infinite xi

        '''

        # Prefactor E_ij
        x_i = torch.linspace(0, self.size-1, self.size, dtype=torch.complex128).reshape((self.size, 1))
        x_j = torch.linspace(0, self.size-1, self.size, dtype=torch.complex128).reshape((1, self.size))
        exp_i = torch.exp(x_i / xi**2)
        exp_j = torch.exp(-x_j / xi**2)
        prefactor = exp_i @ exp_j
        prefactor = torch.triu(prefactor) - torch.eye(self.size)

        # Tensor contraction
        H = gamma*torch.einsum('ij,ikm,jml->kl', prefactor, Sp_list, Sm_list)
        H += torch.transpose(H.conj(), -1, 0)

        return H

