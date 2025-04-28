#!/usr/bin/env python
# coding: utf-8

# In[1]:

# qiskit version: {'qiskit-terra': '0.19.1', 'qiskit-aer': '0.10.2', 'qiskit-ignis': '0.7.0',
#'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.1', 'qiskit-nature': '0.3.0', 
                #'qiskit-finance': '0.3.0', 'qiskit-optimization': None, 'qiskit-machine-learning': '0.3.0'}
        
#==============================================================================================================


from qiskit_nature.settings import settings
settings.dict_aux_operators = True
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


energy4=[]
par4=[]
spin4=[]
for a in np.arange(0.4,3.51,0.01, dtype=object):
    molecule = Molecule(
        geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(JordanWignerMapper())
    from qiskit.providers.aer import StatevectorSimulator
    
    from qiskit import Aer
    from qiskit.utils import QuantumInstance,algorithm_globals
    from qiskit_nature.algorithms import VQEUCCFactory

    quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
    vqe_solver = VQEUCCFactory(quantum_instance)

    from qiskit.algorithms import VQE
    from qiskit.circuit.library import TwoLocal

    seed = 50
    algorithm_globals.random_seed = seed
    
    num_qubits = 4

    
    layer_1 = [(0, 1),(2,3)]
    layer_2 = [(0, 3),(1,2)]
    
    ansatz=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=4,insert_barriers=True, parameter_prefix = 'theta')
    
  
    
    
    
    from qiskit.algorithms.optimizers import SLSQP
    slsqp = SLSQP(maxiter=200)
    another_solver = VQE(
        ansatz,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
    )
    from qiskit_nature.algorithms import GroundStateEigensolver

    calc = GroundStateEigensolver(qubit_converter, another_solver)
    res = calc.solve(es_problem)

    print("a:",a)
    print("")
    print(res)
    print("")
    print("Total ground state energy=",res.total_energies)
    print("")
    #print("absolute tolerance= ",  res.ABSOLUTE_TOLERANCE )
    print("=======================***************============================")
    energy4.append(res.total_energies[0].real)
    par4.append(res.num_particles)
    spin4.append(res.spin)
    
      
    



print("The circuit depth is:", ansatz.decompose().depth())
display(ansatz.decompose().draw(output = 'mpl'))




import numpy as np
exact=[]
for a in np.arange(0.4,3.51,0.01, dtype=object):
    from qiskit import Aer
    from qiskit_nature.drivers import UnitsType, Molecule
    from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
    )
    from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
    from qiskit_nature.converters.second_quantization import QubitConverter
    from qiskit_nature.mappers.second_quantization import JordanWignerMapper
    molecule = Molecule(
        geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(JordanWignerMapper())

###################
    from qiskit_nature.algorithms import GroundStateEigensolver
    from qiskit.algorithms import NumPyMinimumEigensolver

    numpy_solver = NumPyMinimumEigensolver()


#################


    calc = GroundStateEigensolver(qubit_converter, numpy_solver)
    res = calc.solve(es_problem)
    #print("a:",a)
    #print("")
    #print(res)
    #print("")
    #print("Total ground state energy=",res.total_energies)
    #print("")
    #print("=======================***************============================")
    exact.append(res.total_energies[0].real)

import matplotlib.pyplot as plt
x=np.arange(0.4,3.51,0.01)
y=exact
y1=energy4

 
plt.plot(x, y, color='g', linestyle='solid', linewidth = 1,label='exact')
plt.plot(x, y1, color='red', linestyle='solid', linewidth = 1,label='cluster')
 
plt.legend(loc="upper right") 
# naming the x axis
plt.xlabel('a')
plt.xlabel('Bond lenght (Angstrom)')
# naming the y axis
plt.ylabel('Energy (Ha)')
 
# giving a title to my graph
plt.title('')
 
# function to show the plot
#plt.show()


# In[2]:


import matplotlib.pyplot as plt
x=np.arange(0.4,3.51,0.01)
y=exact
y1=enrgy4

 
plt.plot(x, y, color='g', linestyle='solid', linewidth = 1,label='exact')
plt.plot(x, y1, color='red', linestyle='solid', linewidth = 1,label='cluster')
 
plt.legend(loc="upper right") 
# naming the x axis
plt.xlabel('a')
plt.xlabel('Bond lenght (Angstrom)')
# naming the y axis
plt.ylabel('Energy (Ha)')
 
# giving a title to my graph
plt.title('')
 


# In[3]:


print("reps=4" , enrgy4)


# In[5]:


print("exact:" ,exact)


# In[6]:


from qiskit_nature.settings import settings
settings.dict_aux_operators = True
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


energy5=[]
par5=[]
spin5=[]
for a in np.arange(0.4,3.51,0.01, dtype=object):
    molecule = Molecule(
        geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(JordanWignerMapper())
    from qiskit.providers.aer import StatevectorSimulator
    
    from qiskit import Aer
    from qiskit.utils import QuantumInstance,algorithm_globals
    from qiskit_nature.algorithms import VQEUCCFactory

    quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
    vqe_solver = VQEUCCFactory(quantum_instance)

    from qiskit.algorithms import VQE
    from qiskit.circuit.library import TwoLocal

    seed = 50
    algorithm_globals.random_seed = seed
    
    num_qubits = 4

    
    layer_1 = [(0, 1),(2,3)]
    layer_2 = [(0, 3),(1,2)]
    
    ansatz=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=5,insert_barriers=True, parameter_prefix = 'theta')
    
  
    
    
    
    from qiskit.algorithms.optimizers import SLSQP
    slsqp = SLSQP(maxiter=200)
    another_solver = VQE(
        ansatz,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
    )
    from qiskit_nature.algorithms import GroundStateEigensolver

    calc = GroundStateEigensolver(qubit_converter, another_solver)
    res = calc.solve(es_problem)

    print("a:",a)
    print("")
    print(res)
    print("")
    print("Total ground state energy=",res.total_energies)
    print("")
    #print("absolute tolerance= ",  res.ABSOLUTE_TOLERANCE )
    print("=======================***************============================")
    energy5.append(res.total_energies[0].real)
    par5.append(res.num_particles)
    spin5.append(res.spin)
    
      
    



print("The circuit depth is:", ansatz.decompose().depth())
display(ansatz.decompose().draw(output = 'mpl'))




# In[7]:


import matplotlib.pyplot as plt
x=np.arange(0.4,3.51,0.01)
y=exact
y1=energy5

 
plt.plot(x, y, color='g', linestyle='solid', linewidth = 1,label='exact')
plt.plot(x, y1, color='red', linestyle='solid', linewidth = 1,label='cluster')
 
plt.legend(loc="upper right") 
# naming the x axis
plt.xlabel('a')
plt.xlabel('Bond lenght (Angstrom)')
# naming the y axis
plt.ylabel('Energy (Ha)')
 
# giving a title to my graph
plt.title('')


# In[8]:


print("reps=5" , energy5)


