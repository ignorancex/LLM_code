#!/usr/bin/env python
# coding: utf-8

# In[2]:


# qiskit version: {'qiskit-terra': '0.19.1', 'qiskit-aer': '0.10.2', 'qiskit-ignis': '0.7.0',
#'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.1', 'qiskit-nature': '0.3.0', 
                #'qiskit-finance': '0.3.0', 'qiskit-optimization': None, 'qiskit-machine-learning': '0.3.0'}
        
#==============================================================================================================

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


energy1=[]
par1=[]
spin1=[]
for a in np.arange(0.4,3.51,0.01, dtype=object):
    molecule = Molecule(
        geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(ParityMapper(),two_qubit_reduction=True)
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
    
    num_qubits = 2
    
    tl_circuit = TwoLocal(num_qubits, 'ry', entanglement_blocks='cx',reps=1,entanglement='linear',skip_final_rotation_layer=True)

    from qiskit.algorithms.optimizers import SLSQP
    slsqp = SLSQP(maxiter=200)
    another_solver = VQE(
        ansatz=tl_circuit,optimizer=slsqp,
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
    energy1.append(res.total_energies[0].real)
    par1.append(res.num_particles)
    spin1.append(res.spin)
    
      
    




tl_circuit.decompose().draw(output = 'mpl')


# In[3]:


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


energymf=[]
parmf=[]
spinmf=[]
for a in np.arange(0.4,3.51,0.01, dtype=object):
    molecule = Molecule(
        geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(ParityMapper(),two_qubit_reduction=True)
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
    
    num_qubits = 2
    
    tl_circuit = TwoLocal(num_qubits, 'ry', entanglement_blocks='cx',reps=0,entanglement='linear',skip_final_rotation_layer=False)

    from qiskit.algorithms.optimizers import SLSQP
    slsqp = SLSQP(maxiter=200)
    another_solver = VQE(
        ansatz=tl_circuit,optimizer=slsqp,
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
    energymf.append(res.total_energies[0].real)
    parmf.append(res.num_particles)
    spinmf.append(res.spin)
    
      
    




tl_circuit.decompose().draw(output = 'mpl')


# In[4]:


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
import matplotlib.font_manager as font_manager

plt.rcParams['font.size'] = 16  
font = font_manager.FontProperties(family='Arial', style='normal', size=16)

#plt.legend(loc="upper right",fontsize = 30,prop={'family': 'Arial'})    

plt.legend(loc="best",prop=font) 

x=np.arange(0.4,3.51,0.01)
y=exact
y1=energy1
y2=energymf

 
plt.plot(x, y, color='g', linestyle='solid', linewidth = 1,label='exact')
plt.plot(x, y1, color='blue', linestyle='dashed', linewidth = 1,label='cluster')
plt.plot(x, y2, color='red', linestyle='solid', linewidth = 1,label='mean-field')



# naming the x axis
plt.xlabel('Bond lenght (Angstrom)',fontsize = 16,fontname='Arial ')
# naming the y axis
plt.ylabel('Energy (Ha)',fontsize = 16,fontname='Arial ')
 
# giving a title to my graph
plt.title('Exact')
 
# function to show the plot
plt.show()


# In[5]:


print("exact: ", exact)


# In[6]:


print("energy1: ", energy1)


# In[7]:


print("energymf: ", energymf)


# In[8]:


from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
import numpy as np


mf = []
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    op = a*((X ^I)+(I ^X))+ b*(Z^Z)
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    ansatz = TwoLocal(rotation_blocks='ry', reps=0)

    slsqp = SLSQP(maxiter=200)
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=op)
    #print(result)
    optimizer_evals = result.optimizer_evals
    print("a:",a)
    print(">")
    print("REsult:", result)
    print("")
    print("======================================***=============================================")
    mf.append(result.eigenvalue.real)

from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow import I, X, Z

exact=[]
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    c=9999
    op = a*((X ^I)+(I ^X))+ b*(Z^Z)
    npe = NumPyEigensolver(k=4)
    result = npe.compute_eigenvalues(op)
    num_qubits = 2
    #print(op.to_matrix().real)
    target_energy =min(result.eigenvalues)
    exact.append(target_energy.real)
    
print(ansatz.decompose().draw())


# In[9]:


print("MF Energy: ", mf)


# In[10]:


print("Exact Energy: ", exact)


# In[11]:


from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
import numpy as np


cluster = []
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    op = a*((X ^I)+(I ^X))+ b*(Z^Z)
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    ansatz = TwoLocal(2, 'ry', 'cx', 'linear', reps=1, insert_barriers=False,skip_final_rotation_layer=True)

    slsqp = SLSQP(maxiter=200)
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=op)
    #print(result)
    optimizer_evals = result.optimizer_evals
    print("a:",a)
    print(">")
    print("REsult:", result)
    print("")
    print("======================================***=============================================")
    cluster.append(result.eigenvalue.real)



# In[12]:


print("cluster: ", cluster)


# In[20]:


import numpy as np
my_list = [2,4,6,8,10]
my_array = np.array(my_list)
# printing my_array

# printing the type of my_array
print(my_array)


# In[22]:


x=np.arange(0,1.01,0.01)
y=np.array(exact)
y1=np.array(cluster)
y2=y1-y
import matplotlib.pyplot as plt
 
#plt.plot(x, y, color='red', linestyle='solid', linewidth = 1, label='cluster')
plt.plot(x, y2, color='green', linestyle='solid', linewidth = 1 , label='exact')
 
plt.legend(loc="upper right")    
    
#plt.ylim(-3,0)
#plt.xlim(0,1)
 
# naming the x axis
plt.xlabel('a')
# naming the y axis
plt.ylabel('Energy')
 
# giving a title to my graph
#plt.title('Exact Vs entanglement-free')
 
# function to show the plot
#plt.show()


# In[27]:


from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
import numpy as np

arteta0 =[]
arteta1=[]
ar = []
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    op = a*((X ^I)+(I ^X))+ b*(Z^Z)
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    ansatz = TwoLocal(rotation_blocks='ry', reps=0)

    slsqp = SLSQP(maxiter=200)
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=op)
    #print(result)
    optimizer_evals = result.optimizer_evals
    print("a:",a)
    print(">")
    #print("REsult:", result)
    print("")
    print("======================================***=============================================")
    ar.append(result.eigenvalue)
    arteta0.append(result.optimal_point[0])
    arteta1.append(result.optimal_point[1])


    
    
print(ansatz.decompose().draw())
x=np.arange(0,1.01,0.01)
y0=arteta0
y1=arteta1

import matplotlib.pyplot as plt
 
plt.plot(x, y0, color='red', linestyle='solid', linewidth = 1, label='\u03B8 [0]')
plt.plot(x, y1, color='blue', linestyle='solid', linewidth = 1, label='\u03B8 [1]') 
 
#plt.ylim(-3,0)
#plt.xlim(0,1)
 
# naming the x axis
plt.xlabel('a')
# naming the y axis
plt.ylabel('E')
 
# giving a title to my graph
#plt.title('Exact Vs entanglement-free')
 
# function to show the plot
plt.show()


# In[29]:


print("theta0: " , arteta0 )


# In[30]:


print("theta1: " , arteta1 )


# In[31]:


from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
import numpy as np

theta0 =[]
theta1=[]
ar = []
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    op = a*((X ^I)+(I ^X))+ b*(Z^Z)
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    ansatz = TwoLocal(2, 'ry', 'cx', 'linear', reps=1, insert_barriers=False,skip_final_rotation_layer=True)

    slsqp = SLSQP(maxiter=200)
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=op)
    #print(result)
    optimizer_evals = result.optimizer_evals
    print("a:",a)
    print(">")
    print("REsult:", result)
    print("")
    print("======================================***=============================================")
    ar.append(result.eigenvalue)
    theta0.append(result.optimal_point[0])
    theta1.append(result.optimal_point[1])




# In[32]:


x=np.arange(0,1.01,0.01)
y0=theta0
y1=theta1

import matplotlib.pyplot as plt
 
plt.plot(x, y0, color='red', linestyle='solid', linewidth = 1, label='\u03B8 [0]')
plt.plot(x, y1, color='blue', linestyle='solid', linewidth = 1, label='\u03B8 [1]') 
 
#plt.ylim(-3,0)
#plt.xlim(0,1)
 
# naming the x axis
plt.xlabel('a')
# naming the y axis
plt.ylabel('E')
 
# giving a title to my graph
#plt.title('Exact Vs entanglement-free')
 
# function to show the plot
plt.show()


# In[33]:


print("theta0 cc: " , theta0 )


# In[34]:


print("theta1 cc: " , theta1 )


# In[ ]:




