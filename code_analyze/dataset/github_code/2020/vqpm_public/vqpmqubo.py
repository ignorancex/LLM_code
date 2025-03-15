 '''
QPM converges the eigenvector of I+U with the maximum magnitude 
that is max |1+e^{i\lambda}|, where \lambda is the minimum eigenvalue of H.
The file includes the implementations of variational quantum power method (vqpm) applied to random QUBO, 
vqpmForQUBO: it finds the minimum eigenphase of u 

@author: adaskin, the last updated on June 30, 2021
'''

import numpy as np
import matplotlib.pyplot as plt


############################################
#################################################################
def calculate_block_prob(psi,qubits):
    '''NOT USED!! 
       computes probabilities for the states of a given qubit group.
      psi: a quantum state.
      qubits:the set of qubits such as [1 3 5]
    **the order of the qubits |1,2,3,4,..n>;
    '''
    
    n = len(psi)

    logn = int( np.log2(n));   
    nq = len(qubits)
    f = np.zeros(2**nq);
    
    for j in range(n):
        jbits = bin(j)[2:].zfill(logn)
        ind = 0
        for q in range(nq):
            if jbits[qubits[q]-1] == '1':
                ind += 2**(nq-q-1)
      

        f[ind] += np.real(np.abs(psi[j])**2)
    return f; 
############################################
#################################################################
def calculate_prob(psi,qubit):
    '''computes probabilities for the states of a given qubit.
     psi: a quantum state.
     qubit: index of the qubit, the order of the qubits |1,2,3,4,..n>;
     '''
    n = len(psi)

    logn = int( np.log2(n));   

    f = np.zeros(2);
    qmask = (1<<logn-qubit)
    #for k in range(0, len(psi)) #the columns of the input
    for j in range(n):
        
        ind = 0;
        index = ((j & qmask) != 0);
        ind = ind + index;   
       
        f[ind] = f[ind]+np.real(np.abs(psi[j])**2);
    return f; 
    
############################################
#################################################################
def prepareNewState(outVec, n,qStates, pdiff, precision):
    '''        
     based on probabilities of the qubits generates new state
     measures qubits 1 to n
     prepares a state by combining them
    ----------
    outVec: a quantum state
    n: #qubits
    qStates: given qubit states
    pdiff: necessary prob diff between 1 and 0 to assume qubit 1 or 0
    Returns a state, and new qStates
    
    '''
    p = precision      #precision in probs
 
  
    state = [complex(1)]
    
    for q in range(1,n+1):
        if q in qStates:
            stateQ = qStates[q]
        else:
            stateQ = calculate_prob(outVec, q)
           
            stateQ = np.round(stateQ,p);
        
            if(stateQ[0]>stateQ[1]+pdiff):
                stateQ[1]= 0
                stateQ[0] = 1
                qStates[q]=stateQ
                
            elif stateQ[0]+pdiff<stateQ[1]:
                stateQ[0]=0
                stateQ[1] = 1
                
                qStates[q]=stateQ
            else:
                stateQ = np.sqrt(stateQ)
      
        print("q:", q, stateQ)
        state = np.kron(state, stateQ)    
    return state,qStates  
############################################


def applyUH(psi, u):
    '''
    NOT USED!!
    applies CU and Hadamard
    '''
    #apply Hadamard to 1st qubit in reg0
    s2 = 1/np.sqrt(2)
    D = int(len(psi)/2)
    
    for indx0 in range(D): #first qubit is in ket{0}
        indx1 = indx0+D  #1st qubit is '1'
        #apply Hadamard 
        temp0 = psi[indx0]*s2
        temp1 =  psi[indx1]*s2*u[indx0]
    
        psi[indx0] = temp0 + temp1 # where p0 in ket{0}
        psi[indx1] = temp0- temp1  # where p1 in ket{1}
    return psi

   


def vqpmForQubo(u, n, maxiter, iexpected,  pdiff, precision):
    '''
    measures the state at each iteration
    '''
    Pimin = np.zeros((maxiter,), float);
    N = 2 ** n;
    s2 = 1 / np.sqrt(2); 
    inVec = np.ones((N, ), dtype=complex) / np.sqrt(N); 
    
    psi1 = np.zeros((N, ), dtype=complex);#first qubit in \ket{1}
    numOfIter = maxiter;

    qStates={}    
    for j in range(0, maxiter):
        
        #inVec is the state after R_z gates
        # Apply 1st Hadamard to the first qubit
        psi0 = inVec * s2; #first qubit is 0
        # psi1 = inVec * s2; #first qubit is 1
        
        # print(np.abs(u))
        # print(np.abs(psi1))
        
        # apply CU
        for k in range(0, N):
            psi1[k] = u[k] * psi0[k];
        
        
        # Apply 2nd Hadamard to the first qubit
        psi0final = np.add(s2 * psi0, psi1 * s2); #we use I+U
        # psi1final = snp.add(s2 * psi0, -s2*psi1) 
        
        p0 = np.linalg.norm(psi0final);
        # p1 = np.linalg.norm(psi1final);
        
        psi0final = psi0final/p0
        
        Pimin[j] = np.abs(psi0final[iexpected]) ** 2;
        # print("iter: %d probO Of 1st qubit: %f prob Of Expected State: %lf" % (j, p0, Pimin[j]));
        if (Pimin[j] >= 0.5):
            numOfIter = j + 1;
            break;
            
        inVec,qStates = prepareNewState(psi0final,n,qStates,  pdiff, precision)
   
        

    final_probs =  np.power(np.abs(psi0final), 2);
    stateProb = np.max(final_probs)
    foundState =np.argmax(final_probs)  
    
    
    return   foundState, stateProb, qStates, p0, numOfIter,  Pimin[0:numOfIter]
  


  
def unitaryForQubo(n, Q):
    '''
    creates a vector which represents the diagonal of U
    '''    
    
    # x {0, +1}
    # for given coefficients c and number of parameters n, it creates a diagonal
    # unitary whose elements encode the solution space. it returns a vector.
    N = 2 ** n;
    u = np.ones((N,), complex)
    # \sum qiixi +   # \sum qij xixj , i<= j
  
    for k in range(0, N):
        b1 = bin(k)[2:].zfill(n);
        for i in range(0, n):
            for j in range(i, n ):
                qij = np.exp(1j * Q[i][j] );
                if  (b1[i] == '1') and (b1[j] == '1'):
                    u[k] = u[k] * qij;

    return u

def testUForQubo(n, Q):
    '''
    creates a vector which represents the solution space for qubo
    '''    
    
  
    N = 2 ** n;
    #u = np.ones((N,), complex)
    utest = np.zeros((N,), float)
    # \sum qiixi +   # \sum qij xixj , i<= j
  
    for k in range(0, N):
        b1 = bin(k)[2:].zfill(n);
        for i in range(0, n):
            for j in range(i, n ):
                #qij = np.exp(1j * Q[i][j] );
                if  (b1[i] == '1') and (b1[j] == '1'):
                   # u[k] = u[k] * qij;
                    utest[k] += Q[i][j]
    return  utest

def randomQ(n):
    '''for agiven n parameters it generates a random coefficient matrix Q
    its diagonal should be considered as c
    '''
    Q = np.random.randn(n, n); Q = Q + np.transpose(Q);
    return Q;



##############################################################################
##############################################################################
def uForMinNegativePhase(n,Q):
    '''prepares u for min negative value 
     the sum in [0, pi/2] always positive
    '''
    maxQ = np.sum(np.sum(np.abs(np.triu(Q,0))));
    Q1 = Q / (maxQ); #sum in [-1 1]
    
    Q1 = Q1*np.pi/4; #the sum in [-pi/4, pi/4]
   

    u = unitaryForQubo(n, Q1);
    u = np.exp(1j * np.pi/4) * u;  # the sum in [0, pi/2] always positive
    lu = np.real(np.log(u) / (1j));
    expectedState = np.argmin(lu);
    return Q1, u, lu, expectedState

def uForMinNegativePhase2(n,Q):
    '''NOT USED!!
    this prepares u for min negative value
    the sum in [|negativesum|, pi/4+|negativesum|]
    '''
    
    maxQ = np.sum(np.sum(np.abs(np.triu(Q,0))));
    Q1 = Q / (maxQ); # sum in [-1, 1]
    
    Q1 = Q1*np.pi/4; #the sum in [-pi/4, pi/4]
    
    negativesum = 0  #possible min negative value
    for row in Q1:
        for e in row:
            if e<0:
                negativesum += e
                

    u = unitaryForQubo(n,  Q1);
    lu = np.real(np.log(u) / (1j));

    u = np.exp(-1j * negativesum) * u;  # the sum in [|negativesum|, pi/4+|negativesum|]
    lu = np.real(np.log(u) / (1j));

    expectedState = np.argmin(lu);
    return Q1, u, lu, expectedState
    
if __name__ == '__main__':
    ###PARAMETERS
    pdiff = 0.01 #necessary prob diff to assume a qubit 1 or 0
    precision = 4 #precision of measurement outcome
    ########################################################
    ##RANDOM Q
    n = 10;  #the number of parameters and qubits
    # Q=np.array([[ 4.02377326, -1.06286586,  0.49009314,  0.95332512],
    #    [-1.06286586,  1.4338403 , -1.4136876 ,  0.29605018],
    #    [ 0.49009314, -1.4136876 , -3.60973431, -0.7966874 ],
    #    [ 0.95332512,  0.29605018, -0.7966874 , -0.52469588]])
    Q = randomQ(n) 
    Qscaled, u, lu, expectedState = uForMinNegativePhase(n,Q)
    ########################################################

    

    
    ############################################
    ###RUN VQPM
    foundState, stateProb, qStates, p0, numOfIter,  Pimin = vqpmForQubo(u, n, 50, expectedState,  pdiff, precision);
    
    ####################################################################
    ### print/plot results
    print("-------------------------------------")
    print("Eigenvalues in [%f, %f]" % (np.min(lu), np.max(lu)))
    print("Expected and Found Eigenvalues")
    
    print("expected  value:%f, state:%d" % (lu[expectedState], expectedState))
    print("found     value:%f, state:%d with prob:%f" % (lu[foundState], foundState, stateProb))
    sortedlu = np.sort(np.abs(lu));
    print("eigengap: ", sortedlu[1]-sortedlu[0])
    print(sortedlu)
    
    fig = plt.figure()
    plt.plot(range(numOfIter), Pimin,'b.')
    plt.ylabel('iteration')
    
    plt.xlabel('the success probability (the eigengap: %f, 1/2^n: %f)' 
               %(sortedlu[1]-sortedlu[0], 1/2**n))
    plt.show()
    #fig.savefig('destination_path.eps', format='eps', dpi=1000)
    # utest0 = testUForQubo(n, Q)
    # utest = testUForQubo(n, Qscaled)
    
    # for i in range(2**4):
    #     print(bin(i)[2:].zfill(4),"&", round(utest0[i],5),"&", round(utest[i],5),\
    #           "&", round(lu[i],5), "\\\\")  


