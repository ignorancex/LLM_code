import numpy as np


# QUBO to Dwave Ising
def QUBO_to_Ising(Q):
    q = np.diagonal(Q)
    QD = np.copy(Q)
    for i in range(len(QD)):
        QD[i, i] = 0.0
    QQ = np.copy(QD + np.transpose(QD))
    J = np.triu(QQ) / 4.0
    uno = np.ones(len(QQ))
    h = q / 2 + np.dot(QQ, uno) / 4
    g = np.dot(uno, np.dot(QD, uno)) / 4.0 + np.dot(q, uno) / 2.0
    return (J, h, g)


# Dwave Ising to QUBO
def Ising_to_QUBO(J, h):
    Q = 4.0*J+2.0*np.diag(h-(np.sum(J,axis=0)+np.sum(J,axis=1)))
    return Q


# Dwave Ising to simulator Ising
# 0-1 basis transform
def Ising_Dwave_to_Simulator(J, h):
    return (J, (-1)*h)


# unit test: randomly generate 100 qubo, transform to Ising and transform back and print difference

for i in range(100):

    n=np.random.choice(10)+2
    Q=np.random.rand(n,n)
    for i in range(n):
        for j in range(i):
            Q[i,j]=0.0

    #print("Q")
    #print(Q)

    (J,h,g)=QUBO_to_Ising(Q)
    Q2=Ising_to_QUBO(J, h)

    #print("Q2")
    #print(Q2)
    #print("difference",np.amax(np.absolute(Q-Q2)))


