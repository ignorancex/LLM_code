#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
import numpy as np



energy = []
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    op = a*((X ^I ^ I ^ I ^ I ^ I)+(I ^X ^ I ^ I ^ I ^ I) +(I ^I ^ X ^ I ^ I ^ I)+ (I ^I ^ I ^ X^ I ^ I)+(I ^I ^ I ^ I^ X ^ I)+(I ^I ^ I ^ I^ I ^ X))+ b*((Z^Z^I^I^I^I)+(I^Z^Z^I^I^I) + (I^I^Z^Z^I^I)+(I^I^I^Z^Z^I)+(I^I^I^I^Z^Z)+(Z^I^I^I^I^Z))
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    layer_1 = [(0, 1),(2,3),(4,5)]
    layer_2 = [(1, 2),(3,4),(5,0)]
 
    
    from qiskit.circuit.library import TwoLocal
    num_qubits = 6

    ansatz=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2 ]
                      , reps=2,insert_barriers=True, parameter_prefix = 'theta')
    
    # you can change the " reps=2 " in the "ansatz" to 4,6,8
    
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
    energy.append(result.eigenvalue.real)
print(energy)

from qiskit.algorithms import NumPyEigensolver


exact=[]
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    c=9999
    op = a*((X ^I ^ I ^ I ^ I ^ I)+(I ^X ^ I ^ I ^ I ^ I) +(I ^I ^ X ^ I ^ I ^ I)+ (I ^I ^ I ^ X^ I ^ I)+(I ^I ^ I ^ I^ X ^ I)+(I ^I ^ I ^ I^ I ^ X))+ b*((Z^Z^I^I^I^I)+(I^Z^Z^I^I^I) + (I^I^Z^Z^I^I)+(I^I^I^Z^Z^I)+(I^I^I^I^Z^Z)+(Z^I^I^I^I^Z))
    npe = NumPyEigensolver(k=1)
    result = npe.compute_eigenvalues(op)
    num_qubits = 6
    #print(op.to_matrix().real)
    target_energy =min(result.eigenvalues)
    exact.append(target_energy.real)
    

x=np.arange(0,1.01,0.01)
y=arn6
y2=ar1

import matplotlib.pyplot as plt
 
plt.plot(x, y, color='red', linestyle='dashed', linewidth = 1)
plt.plot(x, y2, color='green', linestyle='solid', linewidth = 1)
 
# naming the x axis
plt.xlabel('a')
# naming the y axis
plt.ylabel('E')
 
# giving a title to my graph
plt.title('Exact Vs cluster A (reps=2)')
 
# function to show the plot
plt.show()

print("The circuit depth is:", ansatz.decompose().depth())
(ansatz.decompose().draw(output = 'mpl'))

