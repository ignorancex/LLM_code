'''
This program implements the variational quantum eigensolver (VQE) algorithm based on Bayesian optimisation (BO) with different types of information sharing allowing "immediate sharing".
Copyright (C) 2024  Milena Röhrs

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

This code is written for the bachelor's thesis 
"Bayesian Optimisation (BO) based VQE for Quantum Chemistry Applications:
Experiments on Information Sharing and the BO's Surrogate Model" by Milena Röhrs.
The purpose of this code is to demonstrate the application of BO to optimise the parameters of a quantum circuit in VQE.
The code was inspired by https://github.com/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_modular_bayesian_optimization.ipynb
as well as the Paper "Variational quantum algorithm with information sharing" by Self et al. 2021.
The documentation of the code was created with the help of the LLM Perplexity (https://www.perplexity.ai/).
'''
import xyz_py as xyzp
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import GPy
from GPy.kern import Kern

import matplotlib.pyplot as plt
import sys
import os
import json
import re
import itertools
import pdb

import numpy as np
from numpy.random import multivariate_normal

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit_nature.second_q.circuit.library.ansatzes import UCCSD
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import HartreeFock
from qiskit.primitives import Estimator as PrimEstimator
from qiskit.algorithms.optimizers import SLSQP
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeLima, FakeKolkata
from qiskit_aer.noise import NoiseModel

from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session
from qiskit_ibm_runtime import Estimator as RtEstimator

from contextlib import ExitStack, nullcontext
import pauli_handling as ph

from qiskit_nature import settings
settings.use_pauli_sum_op = False  # necessary due to impending deprication


def H2O_internal_to_mlkl(internals):
    """
    Takes internal coordinates for the water molecule and returns a MoleculeInfo Object that can be processed by the PYSCF-Driver.

        Parameters:
            internals (numpy array): [angle,bondl1,bondl2]

        Returns:
    	    molecule
    """
    if len(internals) == 2:
        [angle,bondl] = internals
        lcord = np.array([[0,0,0],[0,0,bondl],[0,np.sin(angle)*bondl,np.cos(angle)*bondl]])
    elif len(internals) == 3:
        [angle,bondl1,bondl2] = internals
        lcord = np.array([[0,0,0],[0,0,bondl1],[0,np.sin(angle)*bondl2,np.cos(angle)*bondl2]])
    else:
        raise NotImplementedError
    lsym = ["O", "H", "H"]
    mlkl = MoleculeInfo(charge=0, multiplicity=1, symbols=lsym, coords=lcord)
    mlkl.internal_coords = internals

    return mlkl


def H2_internal_to_mlkl(internals):
    """
    Creates a MoleculeInfo object for a given bond length of a hydrogen molecule (H2).

        Parameters:
            internals (list): A list containing the internal coordinates for the hydrogen molecule. The list should have one element, which is the bond length (in Angstroms).

        Returns:
            MoleculeInfo: A MoleculeInfo object representing the hydrogen molecule, with the provided internal coordinates.
    """

    [bondl1] = internals
    lcord = np.array([[0,0,0],[0,0,bondl1]])
    lsym = ["H", "H"]
    mlkl = MoleculeInfo(charge=0, multiplicity=1, symbols=lsym, coords=lcord)
    mlkl.internal_coords = internals

    return mlkl


def write_xyz_file(mlkl, output_path, output_name, comment=""):
    """
    Writes the coordinates and symbols of a molecular structure to an XYZ file.

        Parameters:
            mlkl (object): An object containing the molecular structure information, including the symbols and coordinates of the atoms.
            output_path (str): The path to the directory where the XYZ file will be saved.
            output_name (str): The name of the XYZ file (without the .xyz extension).
            comment (str, optional): A comment to be included in the XYZ file, typically a description of the molecular structure. Defaults to an empty string.

        Returns:
            bool: True if the file was successfully written, False otherwise.
    """

    # if not existent create output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # write to output_name.xyz

    geofile_name = "{0}/{1}.xyz".format(output_path,output_name)

    if not os.path.isfile(geofile_name):
        with open(geofile_name, mode="a") as geo_file:
            geo_file.write(str(len(mlkl.symbols)))
            geo_file.write("\n")
            geo_file.write(comment+"\n")
            for idx,atm in enumerate(mlkl.symbols):
                geo_file.write("{0} {1} {2} {3}\n".format(atm, mlkl.coords[idx][0], mlkl.coords[idx][1], mlkl.coords[idx][2]))
        geo_written = True
    else:
        geo_written = False

    return geo_written


def grid_to_molecules(grid_specification, molecule_name):
    """
    Generates a list of molecular objects (mlkl) based on a grid specification and a molecule name.
    This function supports the generation of molecular objects for two types of molecules: H2 and H2O. 
    For H2, the function generates a list of molecular objects with evenly spaced bond lengths. 
    For H2O, the function generates a list of molecular objects with varying bond lengths and bond angles.
    When the grid specification contains only two values, it is assumed that wo use fixed symmetry which means
    both bond length in the water molecule are the same.

        Parameters:
            grid_specification (list): A list of integers specifying the number of points along each dimension of the grid. The length of the list depends on the number of internal coordinates required to define the molecule.
            molecule_name (str): The name of the molecule to be generated. Currently, only "H2" and "H2O" are supported.

        Returns:
            list: A list of molecular objects (mlkl) representing the molecules generated based on the grid specification.

        Raises:
            NotImplementedError: If the `molecule_name` is not "H2" or "H2O", or if the `grid_specification` has an unsupported length.
    """

    list_of_mlkls = []
    if molecule_name == "H2O":
        if len(grid_specification) == 2:
            list1 = np.linspace(np.pi/2, np.pi, grid_specification[0])
            list2 = np.linspace(0.8, 1, grid_specification[1])
            combinations = list(itertools.product(list1, list2))
        elif len(grid_specification) == 3:
            list1 = np.linspace(1.7, 1.9, grid_specification[0])
            list2 = np.linspace(0.8, 1, grid_specification[1])
            list3 = np.linspace(0.8, 1, grid_specification[2])
            combinations = list(itertools.product(list1, list2, list3))
        else:
            raise NotImplementedError
        list_of_mlkls = list(map(H2O_internal_to_mlkl, combinations))
    elif molecule_name == "H2":
        evenly_spaced = np.linspace(0.3, 1, grid_specification[0])
        arr_of_internals = np.expand_dims(evenly_spaced, axis=1)
        for i in arr_of_internals:
            list_of_mlkls.append(H2_internal_to_mlkl(i))
    else:
        print("The code currently only implemented for H2 and H2O.\n To generalise the code it is recommended to find a python package that transforms internal to cartesian coordinated and write a general function internal_to_mlkl(internals).")
        raise NotImplementedError

    return list_of_mlkls


def molecule_to_problem(molecule, basis="sto3g"):
    """
    Generates an electronic structure problem for a given molecule using the PySCF library.

        Parameters:
            molecule (molecule object): The molecular structure to be used for the electronic structure calculation.
            basis (str, optional): The basis set to be used for the calculation. Defaults to "sto3g".

        Returns:
            full_problem (electronic_structure_problem): An object that contains the Restricted Hartree-Fock (RHF) result for the input molecule.
    """

    driver = PySCFDriver.from_molecule(molecule=molecule, basis=basis)
    full_problem = driver.run()

    return full_problem


def problem_to_pauli_expansion(full_problem, mapper):
    '''
    Returns the qubit operator corresponding to the molecules Hamiltonian.

        Parameters:
                full_problem (electronic_structure_problem): object that contains the RHF 
                                                                result for the molecule
                mapper (FermionicOp Mapper): converts fermionic operator to qubit operator
        Returns:
                qubit_op (qubit operator): contains expansion in Pauli strings
    '''
    
    fermionic_op = full_problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)
    
    return qubit_op


def H2_ansatz():
    """
    Returns the parameterised ansatz circuit used for H_2 in the paper
    "Variational quantum algorithm with information sharing" by Self et al. 2021.

            Parameters: -
            Returns:
                    u_theta (parameterised circuit): ansatz circuit U(theta)
    """

    pt = ParameterVector("theta", 6)
    u_theta = QuantumCircuit(2)
    u_theta.ry(pt[0], 0)
    u_theta.ry(pt[1], 1)
    u_theta.cx(0,1)
    u_theta.ry(pt[2], 0)
    u_theta.ry(pt[3], 1)
    u_theta.cx(0,1)
    u_theta.ry(pt[4], 0)
    u_theta.ry(pt[5], 1)

    return u_theta


def extract_initp_from_outfile(reference_folder, repetition_nr):
    """
    Extracts the initial points from an output file such that experiments with a differen setup still use the same initial points if 
    the repetition number is the same.

        Parameters:
            reference_folder (str): The path to the reference folder.
            repetition_nr (int): The repetition number.

        Returns:
            X_init (numpy.ndarray): An array of initial points.
    """

    reference_file = reference_folder+"/outdata_rep{0}".format(repetition_nr)
    with open(reference_file, 'r') as file:
        data = file.read()
        result = re.search(r'restarts(.*?)Details', data, re.DOTALL)
        if result:
            extracted_text = result.group(1)

    t = extracted_text.split(":")[2].split("]\n [")

    for i in range(len(t)):
        t[i]=t[i].strip(" ").strip("[").strip("\n").strip("]").strip(" ").split(" ")
        t[i] = [x for x in t[i] if x != '']

    for i in range(len(t)):
        for j in range (len(t[i])):
                try:
                    t[i][j] = float(t[i][j])
                except:
                    pass

    X_init = np.array(t)

    return X_init


def evaluate_paulis_outside(paulislist, ansatz, estimator, param_vals, cfo, shots=None):
    """
    Evaluates the expectation values of a list of Pauli operators for a given quantum circuit ansatz and parameter values 
    outside of the cost function object.

    Args:
        paulislist (list): A list of Pauli operator strings (e.g., ['X', 'YZ', 'I']) to be evaluated.
        ansatz (QuantumCircuit): The quantum circuit ansatz to be used for the evaluation.
        estimator (QuantumEstimator): The quantum estimator object used to run the circuits and measure the observables.
        param_vals (numpy.ndarray): The parameter values to be assigned to the circuit ansatz. Differnt dimensions are accepted for compatibility with the GPyOpt module.
        cfo (object): Const function object in which the number of job evaluations needs to be updated.
        shots (int, optional): The number of shots to use for the quantum circuit execution. If not provided, the estimator's default shots will be used.

    Returns:
        list: A list of the expected values of the Pauli operators for the given parameter values.
    """

    if param_vals.ndim == 2:
        paradic = {ansatz.parameters[i]: param_vals[:,i][0] for i in range(len(self.ansatz.parameters))}
    else:
        assert len(param_vals) == len(ansatz.parameters)  
        paradic = {ansatz.parameters[i]: param_vals[i] for i in range(len(param_vals))}

    state = ansatz.assign_parameters(paradic)

    observables = []
    for i in range(len(paulislist)):
        observables.append(SparsePauliOp(paulislist[i]))

    observables_tuple=tuple(observables)
    circuits = tuple([state for i in range(len(observables))])

    if shots == None:
        result = estimator.run(circuits, observables_tuple).result()
    else:
        result = estimator.run(circuits, observables_tuple, shots=shots).result()

    additional_job_evals = len(observables)
    cfo.num_job_evals += additional_job_evals
    exp_vals = result.values.tolist()

    return exp_vals


class ObjectiveFunction:
    """
    A class to represent the VQE objective function of the hydrogen molecule with defined geometry as a function of the ansatz parameters.

        Attributes:
        -----------
        - name: str, the name of the geometries xyz-file
        - molecule: object, the molecule associated with the objective function
        - mapper: FermionicOp Mapper object, the mapper used for transformation of the fermionic problem to the qubit space
        - ansatz: circuit object, parametrised ansatz circuit
        - estimator: estimator, the estimator used for evaluating the quantum state
        - basis: str, the basis set for RHF reference 
        - sharing_mode: str, the information sharing mode of the Bayesian optimisation with information sharing
        - reference: object, the reference for the objective function
        - full_problem: electronic_structure_problem object, contains the RHF result
        - e_nuc: float, internuclei interaction energy (needs to be added to electronic energy)
        - op: qubit operator, Hamiltonian in qubit space in form of its expansion in the Pauli basis
        - pauli_evaluations: list of dicts, lists all past estimations for Pauli expectation values in dictionary form
        - num_pauli_evals, int: total number of function evaluations of this objective excecuted via evaluate_with_paulis-method
        - num_job_evals, int: total number of quantum jobs performed to evaluate this objective function excecuted via evaluate_with_paulis-method
        - reference, float: stores exact evaluation reference
        - converged_at_iter, int: stores in which iteration the convergence criterion has been reached for the first time, None if not converged, yet

        Methods:
        --------
        - remodel_parameters: Reshapes the given Ansatz parameters in a form accepted by the assign_parameters method of a quantum circuit
        - evaluate_with_paulis: Evaluates the objective function by calculating the expectation values for the individual 
                                Pauli operators and summing them with the correct coefficients
    """

    def __init__(self, name, molecule, mapper, ansatz, estimator, active_space=None, basis = "sto3g", sharing_mode = "no_sharing", reference = None):

        self.name = name
        self.molecule = molecule
        self.mapper = mapper
        self.ansatz = ansatz
        self.estimator = estimator
        self.basis = basis
        self.full_problem = molecule_to_problem(molecule = molecule, basis = basis)
        self.active_space = active_space
        if active_space!=None:
            (active_electrons, active_spacial_orbitals) = active_space
            as_transformer = ActiveSpaceTransformer(active_electrons, active_spacial_orbitals)
            self.full_problem = as_transformer.transform(self.full_problem)
        self.e_nuc = self.full_problem.nuclear_repulsion_energy  # TODO: checken if AS transformer noch zusätzlichen shift hat
        self.op = problem_to_pauli_expansion(self.full_problem, mapper)
        self.params = ansatz.parameters
        self.results = []
        self.sharing_mode = sharing_mode
        self.pauli_evaluations = []
        self.num_pauli_evals = 0
        self.num_job_evals = 0
        self.reference = reference
        self.converged_at_iter = None
        self.sharing_partners = []
        self.relevant_paulis = self.op.paulis.to_labels()

    def add_sharing_partner(self, other):
        """
        Add a new sharing partner to the current objective function. 
        This method updates the `relevant_paulis` and `sharing_partners` attributes of the current objective function object.

            Parameters:
                other (ObjectiveFunction): The new sharing partner to be added.

            Returns:
                None
        """

        self.relevant_paulis, pauli_positions = ph.twopaulis(self.relevant_paulis, other.cost_function_object.op.paulis.to_labels())
        self.sharing_partners.append((other, pauli_positions))

    def remodel_parameters(self, param_vals):
        """
        Brings parameters in a shape such that they can be assigned to the parameterised ansatz.

            Parameters:
                param_vals (numpy array): the parameter values to be remodeled

            Returns:
                paradic (dict): a dictionary whose keys are the parameters of the ansatz and values are the corresponding elements of param_vals
        """

        if param_vals.ndim == 2:
            paradic = {self.ansatz.parameters[i]: param_vals[:,i][0] for i in range(len(self.ansatz.parameters))}
        else:
            assert len(param_vals) == len(self.ansatz.parameters)  
            paradic = {self.ansatz.parameters[i]: param_vals[i] for i in range(len(param_vals))}
        
        return paradic

    def evaluate_with_paulis(self, param_vals, shots=None):
        """
        Evaluate the expectation value of a Hamiltonian with respect to the ansatz state by estimating 
        the expectation values of the individual Pauli strings. The results are stored in the Pauli-evaluations
        attribute of the optimisation object.  
        
            Parameters:
                param_vals (array_like): The parameter values for the ansatz.
                shots (int, optional): The number of shots for evaluating the circuits. 
                                    If None, the current shot settings are not overwritten.

            Returns:
                E (float): Objective function value for given parameter set. 
        """
        

        paradic = self.remodel_parameters(param_vals)
        state = self.ansatz.assign_parameters(paradic)

        observables = []
        for i in range(len(self.relevant_paulis)):
            observables.append(SparsePauliOp(self.relevant_paulis[i]))

        observables_tuple=tuple(observables)
        circuits = tuple([state for i in range(len(observables))])

        if shots == None:
            self.results.append(self.estimator.run(circuits, observables_tuple))
        else:
            self.results.append(self.estimator.run(circuits, observables_tuple, shots=shots))

        self.num_job_evals += len(observables)
        result = self.results[-1].result()
        bracket_values = result.values.tolist()
        lin_comb_coeffs = self.op.coeffs
        ltemp = len(lin_comb_coeffs)
        max_len = max(len(bracket_values), ltemp)
        if max_len > ltemp:
            add_zero_coeffs = np.array([0]*(max_len-ltemp))
            lin_comb_coeffs = np.concatenate((lin_comb_coeffs,add_zero_coeffs))
        E = sum(bracket_values*lin_comb_coeffs)

        self.pauli_evaluations.append({
            "param_vals": param_vals,
            "pauli_observables": observables,
            "pauli_expectations": bracket_values
            })  # store everything we need to know about the Paulis

        self.num_pauli_evals += 1

        return np.array(E)

    def __call__(self, param_vals):
 
        paradic = self.remodel_parameters(param_vals)
        state = self.ansatz.assign_parameters(paradic)  # create trial state circuit for certain set of parameters
        self.results.append(self.estimator.run(state, self.op))
        E = self.results[-1].result().values

        return np.array(E)


def iBOIS(molecule_name, molecules, kernel_name, long_out, sharing_mode, iterations, num_initp, k_0, decr_rate, name, number, service, backend_str, backend, using_fake_backend, runtime_options, eps=10**(-3), load_initp = False, active_space = None):
    """
    Performs interconnected Bayesian optimisations for H_2 and H2_O.

    Parameters:
        molecule_name (string)
        molecules (list): List of the molecules investigated.
        kernel_name (string): Name of the kernel used.
        long_out (Boolean): Determines whether long output should be printed.
        sharing_mode (string): Mode of information sharing.
        iterations (int): Maximum number of BO iterations.
        num_initp (int): Number of initial points.
        k_0 (float): Initial value of the acquisition weight.
        decr_rate (float): Decrease rate factor of the acquisition weight.
        name (string): Name of the calculation.
        number (int): Repetition number of the calculation.
        backend_str (string): IBM Backend.
        backend (IBM backend): The backend for the quantum calculation. None if local calculation is performed. Else a simulator or quantum computer.
        using_fake_backend (Boolean): Whether a fake backend is being used.
        noise_model (string): Noise model.
        runtime_options (string): Runtime options.
        eps (float, optional): Relative tolerance of derivation from reference for convergence. Defaults to 10**(-3).
        load_initp (Boolean, optional): Should initial points be loaded? Defaults to False.
        active_space (tuple or None): (n,m), with number of active electrons n and number of active orbitals m.

    Returns:
        E (float): Electronic energy calculated via BO-VQE.
        E_exact (float): Result from exact diagonalisation.
    """

    # writing output

    directory = "results/"+name
    print(f"Writing output to {directory}.")
    isexist = os.path.exists(directory)
    if not isexist:
        os.makedirs(directory)
    
    conv_dir = directory+"/CONV"
    if not os.path.exists(conv_dir):
        os.makedirs(conv_dir)


    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        f.write("\n\n>>>Results of repetition {0}:<<<\n".format(number))
        f.write("Geometries {0}\nKernel: {1}\nSharing mode: {2}\nIterations: {3}\nBackend: {4}\nNumber of initial points: {5}\nInitial acquisition weight: {6}\nAcqisition decrease rate: {7}\nDevice: {8}\nConv. Crit.: {9}\n".format(len(molecules), kernel_name, sharing_mode, iterations, backend, num_initp, k_0, decr_rate, backend_str, eps))
    
    print("Output file created.")


    # prepare geometry

    opt_objs = []
    lsym = molecules[0].symbols

    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        f.write("Involved atoms: {0}\n".format(lsym))

    
    # write QC calculation details

    using_runtime = (backend_str != "no")
    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        if using_runtime:
            f.write("Using IBM Runtime.\n")
        else:
            f.write("Local Calculation.\n")


    # Specify ansatz

    dummy = molecule_to_problem(molecule = molecules[0], basis = "sto3g")
    if molecule_name == "H2":
        ansatz = H2_ansatz()
        mapper = ParityMapper(num_particles=dummy.num_particles)
    elif molecule_name == "H2O":
        print(dummy.num_particles)
        if active_space != None:
            (active_electrons, active_orbitals) = active_space
            spin_up_spin_down = (round(active_electrons/2),round(active_electrons/2))
            assert spin_up_spin_down[0]+spin_up_spin_down[1] == active_electrons
            print(spin_up_spin_down)
            mapper = ParityMapper(num_particles=spin_up_spin_down )  
            ansatz = RealAmplitudes(6, reps=2)  # TODO: Remove hardcoded qubit number
        else:
            mapper = ParityMapper(num_particles=dummy.num_particles)
            ansatz = UCCSD(num_spatial_orbitals=dummy.num_spatial_orbitals, num_particles=dummy.num_particles, qubit_mapper=mapper)
    else:
        raise NotImplementedError

    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        f.write("Ansatz:\n{0}".format(ansatz.decompose().draw(output='text')))


    # BO VQE implementation

    maxiter = iterations


    # Model parameters

    if kernel_name == "Matern52":
        kernel = GPy.kern.Matern52(input_dim=len(ansatz.parameters))  # see https://andrewcharlesjones.github.io/journal/matern-kernels.html
    elif kernel_name == "RBF":
        kernel = GPy.kern.RBF(input_dim=len(ansatz.parameters))
    elif kernel_name == "VQE-kernel":
        kernel = VQEKern(input_dim=len(ansatz.parameters)) 
    else:
        print("No valid kernel chosen.")
        sys.exit()


    # Hyperparameter optimisation    
        
    optimizer = 'bfgs'  
    max_iters = 1000
    optimize_restarts = 5  # number of random restarts (random initial values, maximize likelihood of the data), best one chosen
    sparse = False  # makes sense with in case of many observations
    num_inducing = 10  # number of inducing points if a sparse GP is used
    verbose = False
    ARD = False


    # BO noise-mode

    noise_var = None
    if not using_runtime:
        exact_feval = True  # if we are in noiseless scenario
    else:
        exact_feval = False  # TODO: No-noise case of runtime

    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        f.write("\n\nMODEL PARAMETETERS:\n\nKernel optimizer: {0}\nKernel optimization iterations: {1}\nKernel optimization restarts: {2}\n".format(optimizer, max_iters, optimize_restarts))


    # Instanciate model

    model = GPyOpt.models.gpmodel.GPModel(kernel, noise_var, exact_feval,
                                          optimizer, max_iters,
                                          optimize_restarts, sparse,
                                          num_inducing, verbose=verbose, ARD=ARD)


    # Defining feasible set

    bounds = [{'name': "param_"+str(i), 'type': 'continuous', 'domain': (0,2*np.pi)}
              for i in range(len(ansatz.parameters))]
    feasible_set = GPyOpt.Design_space(space = bounds, constraints = None)
    if sharing_mode != "no":
        if load_initp:
            print(load_initp)
            reference_folder = "..."
            with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                f.write(f"\nInitial points loaded from {reference_folder} repetition {number}.\n")
            X_init = extract_initp_from_outfile(reference_folder, number)
            assert len(X_init) == num_initp
        else:
            X_init = GPyOpt.experiment_design.initial_design('random', feasible_set, num_initp)

        with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
            f.write("initial points: \n{0}\n".format(X_init))


    # Create optimisation object for each geometry

    cost_fun_list = []
    E_exact = []


    with Session(service=service, backend=backend) if using_runtime else nullcontext() as session:
        print("Initialization of Optimizations")

        if not using_runtime:
            estimator = PrimEstimator()
        else:
            estimator = RtEstimator(session=session, options=runtime_options)

        for d_indx, mlkl in enumerate(molecules):


            # Objective function
            
            print("Calculation of initial points for internal coordinates {0} ...".format(mlkl.internal_coords), end="")
            opt_name = "Optimizer_"+str(d_indx)  #+"_at_distance_"+str(dist)
            cost_fun_list.append(ObjectiveFunction(opt_name ,mlkl, mapper, ansatz, estimator, active_space=active_space, sharing_mode=sharing_mode))
            with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                f.write("New objective function for Hamiltonian {0} created.\n".format(cost_fun_list[-1].op)) 
            write_xyz_file(cost_fun_list[-1].molecule, "{0}/geometries".format(directory), cost_fun_list[-1].name, comment=cost_fun_list[-1].name)


            # Reference exact result

            full_problem = cost_fun_list[-1].full_problem
            numpy_solver = NumPyMinimumEigensolver()
            calc_exact = GroundStateEigensolver(mapper, numpy_solver)
            res_exact = calc_exact.solve(full_problem)
            if long_out:
                with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                    f.write("\n\n______________________________\nExact: {0}\n".format(res_exact))
            reference = res_exact.eigenvalues[0]
            E_exact.append(reference)
            cost_fun_list[-1].reference = reference


            # Reference VQE result

            if long_out:
                local_vqe = VQE(PrimEstimator(), ansatz, SLSQP())
                calc_ref = GroundStateEigensolver(mapper, local_vqe)
                res_ref = calc_ref.solve(full_problem)
                with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                    f.write("SLSQP-Reference: {0}\n_________________________\n\n".format(res_ref))
                
            objective = GPyOpt.core.task.objective.SingleObjective(cost_fun_list[-1])  # transform into form required by GPyOpt
            with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                f.write("\n\nDetails about {0}:\n".format(cost_fun_list[-1].name))
                f.write("Expansion of Hamiltonian: {0}\n".format(cost_fun_list[-1].op))
                f.write("Nuclear Energy: {0}\n\n".format(cost_fun_list[-1].e_nuc))


            # Acquisition function and it's optimisation

            aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space = feasible_set,
                                                                        optimizer = "lbfgs")
            acquisition = GPyOpt.acquisitions.LCB.AcquisitionLCB(model, feasible_set,
                        optimizer=aquisition_optimizer,
                        cost_withGradients=None, exploration_weight=2) #TODO: make weight iteration dependent
            evaluator = GPyOpt.core.evaluators.Sequential(acquisition)


            # Initial points
            
            if using_runtime:
                def h(x):
                    E = cost_fun_list[-1].evaluate_with_paulis(x, shots=8192)
                    return E

            else:
                def h(x):
                    E = cost_fun_list[-1].evaluate_with_paulis(x)
                    return E

            if sharing_mode == "no":
                if load_initp:
                    reference_folder = "..."
                    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                        f.write(f"Initial points loaded from {reference_folder} repetition {number}.")
                    X_init = extract_initp_from_outfile(reference_folder, number)
                    assert len(X_init) == num_initp
                else:
                    X_init = GPyOpt.experiment_design.initial_design('random', feasible_set, num_initp)
                with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                    f.write("\nInitial points for optimizer {1} (Geometry {2}):\n {0}\n".format(X_init, d_indx, mlkl.internal_coords))

                Y_init=np.array(list(map(h, X_init)))
                with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                    f.write("Corresponding Energies: {0}\n".format(Y_init))

                Y_init = Y_init.reshape(-1,1)

            else:
                if d_indx == 0:

                    Y_init = np.array(list(map(h,X_init)))
                    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                        f.write("\nEnergies at initial points for optimizer {1} (Geometry {2}): \n{0}\n".format(Y_init, d_indx, mlkl.internal_coords))

                    paulis_till_now = cost_fun_list[0].op.paulis.to_labels()
                    expectation_values_till_now = []
                    for i in cost_fun_list[0].pauli_evaluations:
                        expectation_values_till_now.append(i["pauli_expectations"])
                    expectation_values_till_now = np.array(expectation_values_till_now)   
                    Y_init = Y_init.reshape(-1,1)

                else:
                    # compare known Paulis with new ones
                    if using_runtime:
                        shots = 8192
                    else:
                        shots = None
                    list3, mapping = ph.twopaulis(paulis_till_now, cost_fun_list[-1].op.paulis.to_labels())
                    if len(list3) > len(paulis_till_now):
                        missing_paulis = list3[len(paulis_till_now):len(list3)]
                        with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                            f.write("The shared information till now lacks the evaluation of the Paulis:\n{0}\nCalculation of the missing info is started.\n".format(missing_paulis))
                            print("The shared information till now lacks the evaluation of the Paulis:\n{0}\nCalculation of the missing info is started.\n".format(missing_paulis))

                        
                        # evaluate missing Paulis

                        add_exp_val_list = []
                        for paramvals in X_init:
                            add_exp_val_list.append(evaluate_paulis_outside(missing_paulis, ansatz, estimator, paramvals, cost_fun_list[-1], shots))
                        add_exp_vals = np.array(add_exp_val_list)
                        paulis_till_now = list3
                        expectation_values_till_now = np.concatenate((expectation_values_till_now,add_exp_vals), axis=1)
                        
                        
                    # include mapping
                    
                    assert len(mapping) == len(cost_fun_list[-1].op.coeffs)
                    
                    paulis_init_weighted = np.zeros((expectation_values_till_now.shape[0],len(mapping)))
                    print(" evaluation includes ", end="")
                    print(len(mapping), end="")
                    print(" Paulis ... ", end="")
                    for p_idx, p in enumerate(cost_fun_list[-1].op.coeffs):
                        paulis_init_weighted[:,p_idx] = p * expectation_values_till_now[:,mapping[p_idx]]

                    Y_init = np.sum(paulis_init_weighted, axis=1)
                    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                        f.write("\nEnergies at initial points for optimizer {1} (Geometry {2}): \n{0}\n".format(Y_init, d_indx, mlkl.internal_coords))
                    Y_init = Y_init.reshape(-1,1)

            
            # Initialise optimisation object

            opt_objs.append(GPyOpt.methods.modular_bayesian_optimization.ModularBayesianOptimization(
                            model, feasible_set, objective, 
                            acquisition, evaluator, X_init, Y_init, cost=None,
                            normalize_Y=True, model_update_interval=1, de_duplication=False)
                        )
            opt_objs[-1].intgeo = mlkl.internal_coords
            opt_objs[-1].e_nuc = cost_fun_list[-1].e_nuc
            opt_objs[-1].cost_function_object = cost_fun_list[-1]
            opt_objs[-1].indx = d_indx

            rel_initial_errors = abs(Y_init-reference)/abs(reference)
            if np.min(rel_initial_errors) <= eps:  # check if initial points fulfil convergence criterion
                opt_objs[-1].cost_function_object.converged_at_iter = 0

            opt_objs[-1].exact_feval =  "TODO"
            opt_objs[-1].constraints =  "TODO"
            opt_objs[-1].model_type =  "TODO"
            opt_objs[-1].acquisition_type = "TODO"
            print("--> Optimizer created.")


        # Main BO-loop

        if (sharing_mode == "all-to-all-sharing") or (sharing_mode == "immediate_aa"):
            print("Specification of sharing-partners (all-to-all) running ...")
            for c_idx, current in enumerate(opt_objs):
                for o_idx, other in enumerate(opt_objs):
                    if (c_idx != o_idx):
                        current.cost_function_object.add_sharing_partner(other)
        elif (sharing_mode == "NN") or (sharing_mode == "immediate_NN"):
            raise NotImplementedError("Next-neighbor sharing is not implemented for this Codeversion. Refer to the Code vqe_with_bo_for_H2 for NN sharing in H2.")
        elif (sharing_mode == "no"):
            pass
        else:
            raise ValueError("Invalid case encountered. You probably requested an illdefiined sharing-mode.")


        jobs_after_it = np.zeros(maxiter)
        n = 0
        all_converged = False
        while ((not all_converged) or (using_runtime)) and (n<maxiter):  # in presence of noise don't stop at convergence
            try:
                print("Iteration {0}".format(n+1))
                with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                    f.write("\n>>Iteration {0}<<\n\n".format(n+1))

                for my_indx, myBopt in enumerate(opt_objs):


                    # update surrogate model and acquisition weight

                    myBopt._update_model(myBopt.normalization_type)  
                    myBopt.acquisition.exploration_weight = max(0.000001, k_0 * (1-(n*decr_rate)/maxiter))
                    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                            f.write("Acquisition weight: {0}\n\n".format(myBopt.acquisition.exploration_weight))
                    
                    
                    # get next parameter(s) by optimising acquisition function and evaluate objective
                    
                    theta_new = myBopt._compute_next_evaluations()
                    if (n > maxiter-6) and (using_runtime):  # perform last 5 evaluations with more shots
                        E_theta_new = myBopt.cost_function_object.evaluate_with_paulis(theta_new, shots=8192)
                        with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                            f.write("---energy evaluation with {0} shots---\n".format(8192))
                    else:
                        E_theta_new = myBopt.cost_function_object.evaluate_with_paulis(theta_new)
                    assert theta_new[0][0] == myBopt.cost_function_object.pauli_evaluations[-1]["param_vals"][0][0]  # sanity-test, no guarentee
                    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                            f.write("\n{0}:\n".format(myBopt.cost_function_object.name))
                            f.write("Newly investigated parameters: {0}\n".format(theta_new))
                            f.write("Newly evaluated energy: {0:.5f} = ".format(E_theta_new.real))
                            for e_indx, coeff in enumerate(myBopt.cost_function_object.op.coeffs):
                                f.write(" + ({0:.5f} * {1:.5f})".format(coeff.real, myBopt.cost_function_object.pauli_evaluations[-1]["pauli_expectations"][e_indx].real))
                            f.write("\n")


                            # Update convergence status
                            
                            temp_ref = myBopt.cost_function_object.reference
                            temp_reldev = abs(E_theta_new-temp_ref)/abs(temp_ref)
                            f.write("Relative daviation from reference: {0}\n".format(temp_reldev))
                            if (myBopt.cost_function_object.converged_at_iter == None) and (temp_reldev <= eps):
                                myBopt.cost_function_object.converged_at_iter = n + 1
                                f.write("   => {0} reached convergence at iteration {1}!\n".format(myBopt.cost_function_object.name, n+1))


                    # Immediate information sharing (if sharing-mode requires)

                    if (sharing_mode == "immediate_aa") or ("immediate_NN"):

                        def immediate_info_sharing(myBopt, otherBopt, mapping=None):
                            """Shares info just evaluated from myBopt to otherBopt"""
                            with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                                f.write("\n---> Sharing with {0}:\n".format(otherBopt.cost_function_object.name))
                                f.write("     Parameters: {0}\n".format(theta_new))
                                otherBopt.X = np.vstack((otherBopt.X, theta_new)) 
                                f.write("     Energy: ")
                                if mapping==None:
                                    E_foreign = sum(myBopt.cost_function_object.pauli_evaluations[-1]["pauli_expectations"]*otherBopt.cost_function_object.op.coeffs)
                                    for e_indx, coeff in enumerate(otherBopt.cost_function_object.op.coeffs):
                                        f.write(" + ({0:.5f} * {1:.5f})".format(coeff.real, myBopt.cost_function_object.pauli_evaluations[-1]["pauli_expectations"][e_indx].real))
                                    f.write("\n")
                                else:
                                    temp = 0
                                    assert len(otherBopt.cost_function_object.op.coeffs) == len(mapping)
                                    for j, h_j in enumerate(otherBopt.cost_function_object.op.coeffs):
                                        p = myBopt.cost_function_object.pauli_evaluations[-1]["pauli_expectations"][mapping[j]]
                                        temp += h_j * p
                                        if j == 0:
                                            f.write(" ({0:.5f} * {1:.5f})".format(h_j.real, p.real))
                                        else:
                                            f.write(" + ({0:.5f} * {1:.5f})".format(h_j.real, p.real))
                                    E_foreign = temp
                                f.write("\n             = {0:.5f}\n".format(E_foreign.real))
                                otherBopt.Y = np.vstack((otherBopt.Y, E_foreign))        

                                temp_ref = otherBopt.cost_function_object.reference
                                if (otherBopt.cost_function_object.converged_at_iter == None) and (abs(E_foreign-temp_ref)/abs(temp_ref) <= eps):
                                    otherBopt.cost_function_object.converged_at_iter = n + 1
                                    f.write("   => {0} reached convergence at iteration {1}!\n".format(otherBopt.cost_function_object.name, n+1))


                        if sharing_mode == "immediate_aa":
                            for partner in myBopt.cost_function_object.sharing_partners:
                                (sp, pauli_positions) = partner
                                immediate_info_sharing(myBopt, sp, pauli_positions)

                        elif sharing_mode == "immediate_NN":
                            raise NotImplementedError


                    # give results to optimiser object

                    myBopt.X = np.vstack((myBopt.X, theta_new))  # add new parameters to array
                    myBopt.Y = np.vstack((myBopt.Y, E_theta_new))  # add corresponding energy to array
                    myBopt.next_param = theta_new  # use that Python is dynamic and we can use as dict

                    jobs_after_it[n] += myBopt.cost_function_object.num_job_evals


                # Non-immediate information sharing

                if (sharing_mode == "all-to-all-sharing") or (sharing_mode == "NN"):

                    raise NotImplementedError

                
                # Update convergence monitoring parameters

                all_converged = all(myBopt.cost_function_object.converged_at_iter is not None for my_indx, myBopt in enumerate(opt_objs))
                n += 1
                
            except Exception as err:
                print("An error occured in iteration {0}".format(n+1))
                print(f"Unexpected {err=}, {type(err)=}\n")
                with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
                    f.write("An error occured in iteration {0}: ".format(n+1))
                    f.write(f"Unexpected {err=}, {type(err)=}\n")
                    break


    assert len(cost_fun_list) == len(opt_objs)
    E = []
    total_job_evals = 0 


    # Finalise optimisation and write output

    position = []
    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        largest_conv_iter = -1
        for idx, myBopt in enumerate(opt_objs):

            myBopt.initial_design_numdata = num_initp
            myBopt.objective_name = myBopt.cost_function_object.name
            report_file = None
            myBopt.run_optimization(max_iter = 0, eps = 0, report_file = report_file)  # generate conclusions from data
            total_job_evals += myBopt.cost_function_object.num_job_evals
            
            f.write("\n\n\n")
            f.write("="*100)
            f.write("\nVQE with Bayesian Optimization (Geometry: {0}):\n".format(myBopt.intgeo))
            f.write("="*100)  # https://www.blopig.com/blog/wp-content/uploads/2019/10/GPyOpt-Tutorial1.html
            f.write("\nParameter that minimise the objective:"+str(myBopt.x_opt))
            f.write("\n\n                                    BO            exact")
            f.write(f"\nMinimum value of the objective: {myBopt.fx_opt:.6f}     {E_exact[idx]:.6f}")  # Electronic energy
            E.append(myBopt.fx_opt+cost_fun_list[idx].e_nuc)
            E_exact[idx] += cost_fun_list[idx].e_nuc
            f.write(f"\nWith energy shift applied:      {E[-1]:.6f}     {E_exact[idx]:.6f}")  # Total energy 

            f.write("\n\n>>> Model/Kernel details: <<<\n")
            print(myBopt.model.model)
            model_dict = myBopt.model.model.to_dict(save_data=False) 
            json.dump(model_dict, f, indent=4)

            f.write("\n")
            f.write("="*100)
            f.write("\n\n")
            f.write("All Energies:\n{0}\n".format(myBopt.Y.T))
            f.write("Best Energies:\n{0}\n".format(myBopt.Y_best))
            ref = myBopt.cost_function_object.reference


            # Evaluate point of convergence

            f.write("Reference: {0}\n".format(ref))
            rel_err = abs((np.array(myBopt.Y_best)-ref)/ref)
            f.write("Relative Error:\n{0}\n".format(rel_err))
            conv_crit = lambda x: x < eps
            position.append(next((i+1 for i, x in enumerate(rel_err) if conv_crit(x)), None))
            if position[-1] != None:
                if sharing_mode == None:
                    conv_iter = position[-1] - num_initp
                else:
                    conv_iter = myBopt.cost_function_object.converged_at_iter
                    largest_conv_iter = np.max([largest_conv_iter,conv_iter])

                f.write("Converged in the {0} iteration.\n".format(conv_iter))
            else:
                f.write("Not converged to the desired accuracy.")
             
            myBopt.plot_convergence("{0}/conv_of_Opt{1}_rep{2}.png".format(conv_dir, myBopt.indx, number))
        

        f.write("Total number of job evaluations: {0}\n".format(total_job_evals))
        if None in position:
            f.write("Not all points have converged:\n")
        else:
            needed_jobs = 0
            if sharing_mode == "no":
                for i, pos in enumerate(position):
                    needed_jobs += len(opt_objs[i].cost_function_object.op.coeffs)*pos
                f.write("Jobs performed until convergence: {0}\n\n".format(needed_jobs))
                pos = max(position)
                c = pos - num_initp
                f.write("Converged after {0} iterations.\n".format(c))
            else:
                f.write("\n\nComplete PES converged after {0} iterations.\n".format(largest_conv_iter))
                f.write("Jobs performed until convergence: {0}\n\n".format(jobs_after_it[largest_conv_iter-1]))

    with open("{0}/outdata_rep{1}".format(directory,number), mode="a") as f:
        for idx, myBopt in enumerate(opt_objs):
            f.write("{0}: {1}\n".format(myBopt.cost_function_object.name,myBopt.cost_function_object.converged_at_iter))
        f.write(f"Jobs performed after Iteration ... : {jobs_after_it}\n")


    return E, E_exact, total_job_evals, myBopt.Y_best, reference


def setup_quantum_calculation(backend_str):
    """
    Set up a quantum calculation based on the specified backend.

        Parameters:
            backend_str (str): A string representing the backend for the quantum calculation. 

        Returns:
            service: The QiskitRuntimeService used for the calculation.
            backend: The backend for the quantum calculation. None if local calculation is performed. Else a simulator or quantum computer.
            using_fake_backend (bool): A flag indicating whether a fake backend is being used.
            noise_model: The noise model for the quantum calculation, if one is used.
            noise_mod (str): A string representing the type of noise model to later be used in naming the output storage.
            runtime_options: The options for the IBM runtime platform of the quantum calculation.
    """
 
    service = QiskitRuntimeService(channel="ibm_quantum", token='')
    if (backend_str == "FakeLima") or (backend_str == "FakeKolkata"):
        backend = service.get_backend("ibmq_qasm_simulator")
        using_fake_backend = True

        if (backend_str == "FakeLima"):    
            noise_model = NoiseModel.from_backend(FakeLima())
            noise_mod = FakeLima().backend_name
            basis_gates = FakeLima().configuration().basis_gates
            coupling_map = FakeLima().configuration().coupling_map
        elif (backend_str == "FakeKolkata"):
            noise_model = NoiseModel.from_backend(FakeKolkata())
            noise_mod = FakeKolkata().backend_name
            basis_gates = FakeKolkata().configuration().basis_gates
            coupling_map = FakeKolkata().configuration().coupling_map
        else: 
            NotImplementedError

        runtime_options = Options()
        runtime_options.execution.shots = 1000  # 8192
        runtime_options.optimization_level = 0
        runtime_options.resilience_level = 0
        runtime_options.simulator = {
            "noise_model": noise_model,
            "basis_gates": basis_gates,  # for V2 backend interface: fake_backend.operation_names
            "coupling_map": coupling_map,  # "coupling_map": fake_backend.coupling_map,
            "seed_simulator": None
            }
        
    elif (backend_str == "no"):
        backend = None
        noise_model = None
        noise_mod = "NoNoise"
        using_fake_backend = False
        runtime_options = None

    elif (backend_str in ["Brisbane", "Osaka", "Kyoto", "Ehningen"]):
        noise_model = None
        using_fake_backend = False
        if (backend_str == "Brisbane"):
            noise_mod = "Real_Brisbane"
            backend = service.get_backend('ibm_brisbane')
        if (backend_str == "Osaka"):
            noise_mod = "Real_Osaka"
            backend = service.get_backend('ibm_osaka')
        if (backend_str == "Kyoto"):
            noise_mod = "Real_Kyoto"
            backend = service.get_backend('ibm_kyoto')
        if (backend_str == "Ehningen"):
            noise_mod = "Ehningen"
            service = QiskitRuntimeService(channel = "ibm_quantum", url="https://auth.de.quantum-computing.ibm.com/api", token='', verify = True)
            backend = service.get_backend('ibmq_ehningen')
        runtime_options = Options()
        runtime_options.execution.shots = 1000  # 8192
        runtime_options.optimization_level = 0
        runtime_options.resilience_level = 0

    else:
        try:
            noise_mod = backend_str
            service = QiskitRuntimeService(channel="ibm_quantum", token='')
            backend = service.get_backend(backend_str)
            noise_model = None
            using_fake_backend = False
            runtime_options = Options()
            runtime_options.execution.shots = 1000
            runtime_options.optimization_level = 0
            runtime_options.resilience_level = 0
        except:
            print("The desired quantum computer could not be identified or you don't have access to it. Choose one of the following options:")
            service.backends()
            NotImplementedError

    return service, backend, using_fake_backend, noise_model, noise_mod, runtime_options


if __name__ == "__main__":
    print("BOIS-with-immediate-sharing Copyright (C) 2024  Milena Röhrs\nThis program comes with ABSOLUTELY NO WARRANTY; for details check out the LICENSE file.\nThis is free software, and you are welcome to redistribute it\nunder certain conditions; for details check out the LICENSE file.")
    print("Start calculation")
    repetition_nr = sys.argv[1]  
    sharing_mode= sys.argv[2] # choose from: "all-to-all-sharing", "no", "immediate_aa", "NN", "immediate_NN"
    kernel_name = sys.argv[3] # choose from: "RBF", "Matern52" 
    backend_str = sys.argv[4] # choose from: "no", "FakeLima", "FakeKolkata", "Brisbane", "Osaka", "Kyoto", "Ehningen"
    molecule_name = sys.argv[5]

    grid_specification = [2,2]
    mlkl_list = grid_to_molecules(grid_specification, molecule_name)

    service, backend, using_fake_backend, noise_model, noise_mod, runtime_options = setup_quantum_calculation(backend_str)
    print("Sharing Mode: {0}".format(sharing_mode))

    iterations = 10
    k_0 = 1
    decr_rate = 1  # Be careful: this is a factor modifying the decrease rate IN ADDITION to the decrease rate induced by the maximum number of iterations.

    num_initp = 30
    load_initp = False
    long_out = True
    eps = 2*10**(-3)

    if molecule_name == "H2":
        active_space = None
    elif molecule_name == "H2O":
        active_space = (4,4)  # "(n,m), where n is the number of active electrons and m is the number of active orbitals" (https://chemistry.stackexchange.com/questions/64483/what-are-complete-active-space-methods-and-how-are-such-spaces-defined-for-molec)
    else:
        raise NotImplementedError
    
    print(repetition_nr)
    series_name = molecule_name+"_PES_"+kernel_name+"_"+sharing_mode+"_"+noise_mod+"_convCrit"+str(eps)
    experiment_name = series_name+"_Rep"+repetition_nr
    PES, E_exact, jobs, a, b = iBOIS(molecule_name, mlkl_list, kernel_name, long_out, sharing_mode, iterations, num_initp, k_0, decr_rate, series_name, repetition_nr, service, backend_str, backend, using_fake_backend, runtime_options, eps, load_initp, active_space)
    print(PES)
