import numpy as np
from numpy import sin, cos, abs, exp, sqrt, pi
import torch
from random import random

class QuantumRegister:
    """

    This class creates a quantum register, providing all the desired
    operations.
    
    ...

    Attributes
    ----------
    size : int
        The total number of qubits in the register.

    Methods
    -------
    ·Sm():
        sigma_- matrix (single qubit operator).
    ·Sp():
        sigma_+ matrix (single qubit operator).
    ·X():
        X Pauli matrix (single qubit operator).
    ·Y():
        Y Pauli matrix (single qubit operator).
    ·Z():
        Z Pauli matrix (single qubit operator).
    ·H():
        Hadamard matrix (single qubit operator).
    ·proj0():
        projector over the |0> state (single qubit operator).
    ·proj1():
        projector over the |1> state (single qubit operator).
    ·CNOT():
        Pairwise CNOT gate.
    ·Rx(theta):
        Rotation of angle theta around the x axis
        (single qubit operator).
    ·Ry(theta):
        Rotation of angle theta around the y axis
        (single qubit operator).
    ·Rz(theta):
        Rotation of angle theta around the z axis
        (single qubit operator).
    ·local_operator(operator, i):
        Creates a local operator acting on the ith qubit in the
        Hilbert space spanned by the total number of qubits.
    ·Sm_i(index):
        sigma_- matrix
        (acting on the index-th qubit of the register).
    ·Sp_i(index):
        sigma_+ matrix 
        (acting on the index-th qubit of the register).
    ·X_i(index):
        X Pauli matrix
        (acting on the index-th qubit of the register).
    ·Y_i(index):
        Y Pauli matrix
        (acting on the index-th qubit of the register).
    ·Z_i(index):
        Z Pauli matrix
        (acting on the index-th qubit of the register).
    ·H_i(index):
        Hadamard matrix
        (acting on the index-th qubit of the register).
    ·CNOT_i():
        Pairwise CNOT gate, acting with control as the ith qubit and
        target as the i+1.
    ·CNOT_ij():
        CNOT gate acting with ith qubit as control and jth as target.
    ·CZ_ij(self, i, j):
        CZ gate acting with ith qubit as control and jth as target.
    ·Rx_i(index):
        Rotation of angle theta around the x axis
        (acting on the index-th qubit of the register).
    ·Ry_i(index):
        Rotation of angle theta around the y axis
        (acting on the index-th qubit of the register).
    ·Rz_i(index):
        Rotation of angle theta around the z axis
        (acting on the index-th qubit of the register).
    """

    def __init__(self,size):
        '''Create a quantum register with size = number of qubits'''
        self.size = size

    def Sm(self):
        '''Local sigma_- operator'''
        Sp = torch.complex(torch.tensor([[0., 1.],[0., 0.]], dtype=torch.float64), torch.zeros((2, 2), dtype=torch.float64))
        return Sp
    
    def Sp(self):
        '''Local sigma_+ operator'''
        Sm = torch.complex(torch.tensor([[0., 0.],[1., 0.]], dtype=torch.float64), torch.zeros((2, 2), dtype=torch.float64))
        return Sm
    
    def X(self):
        '''Local sigma_x operator'''
        Sx = torch.complex(torch.tensor([[0.,1.],[1.,0.]], dtype=torch.float64), torch.zeros((2, 2), dtype=torch.float64))
        return Sx
    
    def Y(self):
        '''Local sigma_y operator'''
        Sy = torch.tensor([[0.,-1.j],[1.j,0.]], dtype=torch.complex128)
        return Sy
    
    def Z(self):
        '''Local sigma_z operator'''
        Sz = torch.complex(torch.tensor([[1.,0.],[0.,-1.]], dtype=torch.float64), torch.zeros((2, 2), dtype=torch.float64))
        return Sz

    def H(self):
        '''Local Hadamard operator'''
        H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex128)/sqrt(2)
        return H

    def proj0(self):
        '''Local projector over the |0><0| state'''
        proj0 = torch.tensor([[1,0],[0,0]])
        return proj0

    def proj1(self):
        '''Local projector over the |1><1| state'''
        proj1 = torch.tensor([[0,0],[0,1]])
        return proj1

    def CNOT(self):
        '''Two-qubits CNOT operator (4x4 matrix)'''
        cnot = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
                            ],
        dtype=torch.complex128)
        return cnot

    
    def Rx(self,theta):
        '''
        Local Rx rotation. If we define it like
        this, the operator is still sparse (as it can be easily
        checked). This is because, although it comes from the exp
        of an Y matrix, this exponential only acts locally.
        ...

        Parameters
        -------
        theta : float
            The angle theta of rotation.
        '''

        Rx =  torch.linalg.matrix_exp(-1j*self.X()*theta/2)

        return Rx
    
    def Ry(self,theta):
        '''
        Local Ry rotation. If we define it like
        this, the operator is still sparse (as it can be easily
        checked). This is because, although it comes from the exp
        of an Y matrix, this exponential only acts locally.
        ...

        Parameters
        -------
        theta : float
            The angle theta of rotation.
        '''
        
        Ry =  torch.linalg.matrix_exp(-1j*self.Y()*theta/2)
        
        return Ry
        
    def Rz(self, theta):
        '''
        Local R_z rotation.
        ···

        Parameters
        ----------
        theta : float
            Rotation angle.
        '''

        Rz =  torch.linalg.matrix_exp(-1j*self.Z()*theta/2)

        return Rz

    def local_operator(self,operator,i):
        '''
        Creates an operator 1 \otimes ... operator \otimes 1
        where operator acts on the ith qubit.
        ...

        Parameters
        -------
        operator : tensor
            The operator acting locally on a single qubit.
        i : int
            The ith qubit in which operator acts. The first is i=0.
        '''

        return torch.kron(torch.eye(2**i), torch.kron(operator, torch.eye(2**(self.size-i-1))))

    def Sp_i(self,index):
        '''
        sigma_+ operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator(self.Sp(),index)

    def Sm_i(self,index):
        '''
        sigma_- operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator(self.Sm(),index)

    def X_i(self,index):
        '''
        X operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator(self.X(),index)

    def Y_i(self,index):
        '''
        Y operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator(self.Y(),index)

    def Z_i(self,index):
        '''
        Z operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator(self.Z(),index)

    def H_i(self,index):
        '''
        Hadamard gate acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        '''

        return self.local_operator(self.H(),index)

    def CNOT_i(self, i):
        '''
        CNOT gate acting locally over the indices (i, i+1)
        ...

        Parameters
        ----------
        i : int
            The ith qubit (control qubit)
            i+1 is the target qubit
        '''

        if i < self.size-1:
            cnot = torch.kron(torch.eye(2**i), torch.kron(self.CNOT(), torch.eye(2**(self.size-i-2))))
            return cnot
        else:
            raise ValueError("ERROR: i is bound to values between 0 and N-2")

    def CNOT_ij(self, i, j):
        '''
        CNOT gate acting over the indices (i,j). It uses the following decomposition
        for the gate (assuming i<j):
            CNOT_ij = 1_2^(i) x |0><0| x 1_2^(N-1-i) 
                    + 1_2^(i) x |1><1| x 1_2^(j-i-1) x X x 1_(N-j-1)
        ...

        Parameters
        ----------
        i : int
            The ith qubit (control qubit)
        j : int 
            The jth qubit (target qubit)
        '''
        if j > self.size:
            raise ValueError("ERROR: j is bound to values between 0 and N-2")
        if i>j:
            raise ValueError("ERROR: the control qubit has to precede the target one. To modify this, apply Hadamards (Nielsen & Chuan, page 179)")
        
        cnot = torch.kron(torch.eye(2**(i)), 
            torch.kron(self.proj0(), torch.eye(2**(self.size-i-1))))+torch.kron(torch.kron(torch.kron(torch.eye(2**(i)),
            torch.kron(self.proj1(), torch.eye(2**(j-i-1)))), self.X()), torch.eye(2**(self.size-j-1)))
        
        return cnot

    def CZ_ij(self, i, j):
        '''
        CZ gate acting over the indices (i,j). It uses the following decomposition
        for the gate (assuming i<j):
            CZ_ij = 1_2^(i) x |0><0| x 1_2^(N-1-i) 
                    + 1_2^(i) x |1><1| x 1_2^(j-i-1) x Z x 1_(N-j-1)
        ...

        Parameters
        ----------
        i : int
            The ith qubit (control qubit)
        j : int 
            The jth qubit (target qubit)
        '''

        if j > self.size:
            raise ValueError("ERROR: j is bound to values between 0 and N-2")
        if i>j:
            raise ValueError("ERROR: the control qubit has to precede the target one. To modify this, apply Hadamards (Nielsen & Chuan, page 179)")
        
        cz = torch.kron(torch.eye(2**(i)), 
            torch.kron(self.proj0(), torch.eye(2**(self.size-i-1))))+torch.kron(torch.kron(torch.kron(torch.eye(2**(i)),
            torch.kron(self.proj1(), torch.eye(2**(j-i-1)))), self.Z()), torch.eye(2**(self.size-j-1)))
        
        return cz
        

    def Rx_i(self,index,theta):
        '''
        Rotation around x operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        theta : float
            The angle of rotation.
        '''

        return self.local_operator(self.Rx(theta),index)
    
    def Ry_i(self,index,theta):
        '''
        Rotation around y operator acting locally over the #index qubit
        ...

        Parameters
        -------
        index : int
            The ith qubit in which operator acts.
        theta : float
            The angle of rotation.
        '''

        return self.local_operator(self.Ry(theta),index)

    def Rz_i(self, index, theta):
        '''
        Rotation around the z axis acting locally over the index qubit
        ···

        Parameters
        ----------

        index : int
            The ith qubit in which the operator acts
        theta : float
            Rotation angle
        '''

        return self.local_operator(self.Rz(theta), index)


