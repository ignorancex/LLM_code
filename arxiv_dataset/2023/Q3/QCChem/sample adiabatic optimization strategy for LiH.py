#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from qiskit import*
from qiskit.circuit import Parameter
import numpy as np
from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer



a=2.59

molecule = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1)
driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", 
                                           driver_type=ElectronicStructureDriverType.PYQUANTE )

es_problem = ElectronicStructureProblem(driver,[FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3,-2])])
second_q_op = es_problem.second_q_ops()

qubit_converter = QubitConverter(ParityMapper(), z2symmetry_reduction="auto")
qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles
                                  ,sector_locator=es_problem.symmetry_sector_locator)

op = qubit_op


#======================================================


from qiskit.algorithms import NumPyEigensolver


npe = NumPyEigensolver(k=1)
result = npe.compute_eigenvalues(op)

#print(op.to_matrix().real)
print("a:",a)
print("")
target_energy =min(result.eigenvalues)
print("")
print(f"exact energy: {target_energy:.5f}")
    
print("==========================***=========================")


from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance,algorithm_globals


quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))


from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal

seed = 50
algorithm_globals.random_seed = seed
    
num_qubits = 6

qr = QuantumRegister(6, 'q')    
ansatz=QuantumCircuit(6)
theta0 = Parameter('t0')
theta1 = Parameter('t1')
theta2=Parameter('t2')
theta3=Parameter('t3')
theta4=Parameter('t4')
theta5=Parameter('t5')
theta6=Parameter('t6')
theta7=Parameter('t7')
theta8=Parameter('t8')
theta9=Parameter('t9')
theta10=Parameter('t10')
theta11=Parameter('t11')
theta12=Parameter('t12')
theta13=Parameter('t13')
theta14=Parameter('t14')
theta15=Parameter('t15')
theta16=Parameter('t16')
theta17=Parameter('t17')
theta18=Parameter('t18')
theta19=Parameter('t19')
    
    
#===================================================================

layer_1 = [(0, 5),(1,4),(2,3)]
layer_2 = [(0, 3),(1,2),(4,5)]

ansatz1=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=4,insert_barriers=True, parameter_prefix = 'theta')
    
    
from qiskit.algorithms.optimizers import SLSQP
slsqp = SLSQP(maxiter=200)
vqe = VQE(ansatz1,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
   )

result1 = vqe.compute_minimum_eigenvalue(operator=op)


#print(res)
print("")
print(f" VQE Result: {result1.eigenvalue.real:.8f}")
print("")
print("The circuit depth is:", ansatz1.decompose().depth())
print(ansatz1.decompose().draw())
print("==========================***=========================")

#===================================================================

initial_pt = result1.optimal_point


# In[ ]:



from qiskit import*
from qiskit.circuit import Parameter
import numpy as np
from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer



a=2.6

molecule = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1)
driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", 
                                           driver_type=ElectronicStructureDriverType.PYQUANTE )

es_problem = ElectronicStructureProblem(driver,[FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3,-2])])
second_q_op = es_problem.second_q_ops()

qubit_converter = QubitConverter(ParityMapper(), z2symmetry_reduction="auto")
qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles
                                  ,sector_locator=es_problem.symmetry_sector_locator)

op = qubit_op


#======================================================


from qiskit.algorithms import NumPyEigensolver


npe = NumPyEigensolver(k=1)
result = npe.compute_eigenvalues(op)

#print(op.to_matrix().real)
print("a:",a)
print("")
target_energy =min(result.eigenvalues)
print("")
print(f"exact energy: {target_energy:.5f}")
    
print("==========================***=========================")


from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance,algorithm_globals


quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))


from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal

seed = 50
algorithm_globals.random_seed = seed
    
num_qubits = 6

qr = QuantumRegister(6, 'q')    
ansatz=QuantumCircuit(6)
theta0 = Parameter('t0')
theta1 = Parameter('t1')
theta2=Parameter('t2')
theta3=Parameter('t3')
theta4=Parameter('t4')
theta5=Parameter('t5')
theta6=Parameter('t6')
theta7=Parameter('t7')
theta8=Parameter('t8')
theta9=Parameter('t9')
theta10=Parameter('t10')
theta11=Parameter('t11')
theta12=Parameter('t12')
theta13=Parameter('t13')
theta14=Parameter('t14')
theta15=Parameter('t15')
theta16=Parameter('t16')
theta17=Parameter('t17')
theta18=Parameter('t18')
theta19=Parameter('t19')
    
    
#===================================================================

layer_1 = [(0, 5),(1,4),(2,3)]
layer_2 = [(0, 3),(1,2),(4,5)]

ansatz1=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=4,insert_barriers=True, parameter_prefix = 'theta')
    
    
from qiskit.algorithms.optimizers import SLSQP
slsqp = SLSQP(maxiter=200)
vqe = VQE(ansatz1,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
 initial_point=initial_pt   )

result1 = vqe.compute_minimum_eigenvalue(operator=op)


#print(res)
print("")
print(f" VQE Result: {result1.eigenvalue.real:.8f}")
print("")
print("The circuit depth is:", ansatz1.decompose().depth())
print(ansatz1.decompose().draw())
print("==========================***=========================")

#===================================================================

initial_pt = result1.optimal_point

calc = GroundStateEigensolver(qubit_converter, vqe)
res = calc.solve(es_problem)

print("a:",a)

print("Total ground state energy=",res.total_energies)


# In[ ]:


from qiskit import*
from qiskit.circuit import Parameter
import numpy as np
from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer



a=2.7

molecule = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1)
driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", 
                                           driver_type=ElectronicStructureDriverType.PYQUANTE )

es_problem = ElectronicStructureProblem(driver,[FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3,-2])])
second_q_op = es_problem.second_q_ops()

qubit_converter = QubitConverter(ParityMapper(), z2symmetry_reduction="auto")
qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles
                                  ,sector_locator=es_problem.symmetry_sector_locator)

op = qubit_op


#======================================================


from qiskit.algorithms import NumPyEigensolver


npe = NumPyEigensolver(k=1)
result = npe.compute_eigenvalues(op)

#print(op.to_matrix().real)
print("a:",a)
print("")
target_energy =min(result.eigenvalues)
print("")
print(f"exact energy: {target_energy:.5f}")
    
print("==========================***=========================")


from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance,algorithm_globals


quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))


from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal

seed = 50
algorithm_globals.random_seed = seed
    
num_qubits = 6

qr = QuantumRegister(6, 'q')    
ansatz=QuantumCircuit(6)
theta0 = Parameter('t0')
theta1 = Parameter('t1')
theta2=Parameter('t2')
theta3=Parameter('t3')
theta4=Parameter('t4')
theta5=Parameter('t5')
theta6=Parameter('t6')
theta7=Parameter('t7')
theta8=Parameter('t8')
theta9=Parameter('t9')
theta10=Parameter('t10')
theta11=Parameter('t11')
theta12=Parameter('t12')
theta13=Parameter('t13')
theta14=Parameter('t14')
theta15=Parameter('t15')
theta16=Parameter('t16')
theta17=Parameter('t17')
theta18=Parameter('t18')
theta19=Parameter('t19')
    
    
#===================================================================

layer_1 = [(0, 5),(1,4),(2,3)]
layer_2 = [(0, 3),(1,2),(4,5)]

ansatz1=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=4,insert_barriers=True, parameter_prefix = 'theta')
    
    
from qiskit.algorithms.optimizers import SLSQP
slsqp = SLSQP(maxiter=200)
vqe = VQE(ansatz1,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
 initial_point=initial_pt   )

result1 = vqe.compute_minimum_eigenvalue(operator=op)


#print(res)
print("")
print(f" VQE Result: {result1.eigenvalue.real:.8f}")
print("")
print("The circuit depth is:", ansatz1.decompose().depth())
print(ansatz1.decompose().draw())
print("==========================***=========================")

#===================================================================

initial_pt = result1.optimal_point

calc = GroundStateEigensolver(qubit_converter, vqe)
res = calc.solve(es_problem)

print("a:",a)

print("Total ground state energy=",res.total_energies)


from qiskit import*
from qiskit.circuit import Parameter
import numpy as np
from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer



a=2.9

molecule = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1)
driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", 
                                           driver_type=ElectronicStructureDriverType.PYQUANTE )

es_problem = ElectronicStructureProblem(driver,[FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3,-2])])
second_q_op = es_problem.second_q_ops()

qubit_converter = QubitConverter(ParityMapper(), z2symmetry_reduction="auto")
qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles
                                  ,sector_locator=es_problem.symmetry_sector_locator)

op = qubit_op


#======================================================


from qiskit.algorithms import NumPyEigensolver


npe = NumPyEigensolver(k=1)
result = npe.compute_eigenvalues(op)

#print(op.to_matrix().real)
print("a:",a)
print("")
target_energy =min(result.eigenvalues)
print("")
print(f"exact energy: {target_energy:.5f}")
    
print("==========================***=========================")


from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance,algorithm_globals


quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))


from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal

seed = 50
algorithm_globals.random_seed = seed
    
num_qubits = 6

qr = QuantumRegister(6, 'q')    
ansatz=QuantumCircuit(6)
theta0 = Parameter('t0')
theta1 = Parameter('t1')
theta2=Parameter('t2')
theta3=Parameter('t3')
theta4=Parameter('t4')
theta5=Parameter('t5')
theta6=Parameter('t6')
theta7=Parameter('t7')
theta8=Parameter('t8')
theta9=Parameter('t9')
theta10=Parameter('t10')
theta11=Parameter('t11')
theta12=Parameter('t12')
theta13=Parameter('t13')
theta14=Parameter('t14')
theta15=Parameter('t15')
theta16=Parameter('t16')
theta17=Parameter('t17')
theta18=Parameter('t18')
theta19=Parameter('t19')
    
    
#===================================================================

layer_1 = [(0, 5),(1,4),(2,3)]
layer_2 = [(0, 3),(1,2),(4,5)]

ansatz1=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=4,insert_barriers=True, parameter_prefix = 'theta')
    
    
from qiskit.algorithms.optimizers import SLSQP
slsqp = SLSQP(maxiter=200)
vqe = VQE(ansatz1,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
 initial_point=initial_pt   )

result1 = vqe.compute_minimum_eigenvalue(operator=op)


#print(res)
print("")
print(f" VQE Result: {result1.eigenvalue.real:.8f}")
print("")
print("The circuit depth is:", ansatz1.decompose().depth())
print(ansatz1.decompose().draw())
print("==========================***=========================")

#===================================================================

initial_pt = result1.optimal_point


calc = GroundStateEigensolver(qubit_converter, vqe)
res = calc.solve(es_problem)

print("a:",a)

print("Total ground state energy=",res.total_energies)


