#!/usr/bin/env python
# coding: utf-8

# In[3]:


# qiskit version: {'qiskit-terra': '0.19.1', 'qiskit-aer': '0.10.2', 'qiskit-ignis': '0.7.0',
#'qiskit-ibmq-provider': '0.18.3', 'qiskit-aqua': '0.9.5', 'qiskit': '0.34.1', 'qiskit-nature': '0.3.0', 
                #'qiskit-finance': '0.3.0', 'qiskit-optimization': None, 'qiskit-machine-learning': '0.3.0'}
        
#==============================================================================================================



from qiskit_nature.settings import settings
settings.dict_aux_operators = True
from qiskit import*
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
    
import numpy as np
arexact=[]
for a in np.arange(1,3.1,0.1, dtype=object):
    from qiskit import Aer
    from qiskit_nature.drivers import UnitsType, Molecule
    from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
    )
    from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
    from qiskit_nature.converters.second_quantization import QubitConverter
    from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
    molecule = Molecule(
        geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver,[FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3,-2)])
    qubit_converter =  QubitConverter(ParityMapper(), z2symmetry_reduction="auto")

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
    arexact.append(res.total_energies[0].real)


# In[4]:


print(arexact)


# In[7]:


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
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer

enrgy5=[]
par5=[]
spin5=[]
for a in np.arange(1,3.1,0.1, dtype=object):
    molecule = Molecule(
        geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, a]]], charge=0, multiplicity=1
    )
    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYQUANTE
    )

    es_problem = ElectronicStructureProblem(driver,[FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3,-2])])
    
    second_q_op = es_problem.second_q_ops()
    qubit_converter = QubitConverter(ParityMapper(), z2symmetry_reduction="auto")
    
    

    from qiskit.providers.aer import StatevectorSimulator
    from qiskit import Aer
    from qiskit.utils import QuantumInstance,algorithm_globals
    from qiskit.algorithms import VQE
    from qiskit.circuit.library import TwoLocal

    seed = 50
    algorithm_globals.random_seed = seed
    
    num_qubits = 6

    
    layer_1 = [(0, 5),(1,4),(2,3)]
    layer_2 = [(0, 3),(1,2),(4,5)]

    ansatz=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=5,insert_barriers=True, parameter_prefix = 'theta')
    
   # change the " reps=5 " in the "ansatz" to 6 or 8 or...
    
    
    
    from qiskit.algorithms.optimizers import SLSQP
    slsqp = SLSQP(maxiter=200)
    vqe = VQE(
        ansatz,optimizer=slsqp,
        quantum_instance=QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed),
    )
    from qiskit_nature.algorithms import GroundStateEigensolver

    calc = GroundStateEigensolver(qubit_converter, vqe)
    res = calc.solve(es_problem)

    print("a:",a)
    print("")
    print(res)
    print("")
    print("Total ground state energy=",res.total_energies)
    print("")
    #print("absolute tolerance= ",  res.ABSOLUTE_TOLERANCE )
    print("=======================***************============================")
    enrgy5.append(res.total_energies[0].real)
    par5.append(res.num_particles)
    spin5.append(res.spin)
    
      
    



print("The circuit depth is:", ansatz.decompose().depth())
ansatz.decompose().draw(output = 'mpl')


# In[9]:


print("energy= ", enrgy5)


# In[13]:


print("particle number="   ,par5)


# In[14]:


print("spin= "  ,spin5)


# In[ ]:




