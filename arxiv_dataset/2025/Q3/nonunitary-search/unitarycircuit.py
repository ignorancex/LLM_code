"""
it gives the unitary circuit simulation of C given in the paper,
and can be used to generate figures 
with amplitude amplification used in the paper.
You can change the value of a and the number of iterations and qubits
in the main function at the bottom.
ammar daskin, Decempber 2024.
"""
import numpy as np
import random
import scipy
import scipy.linalg as la
import matplotlib.pyplot as plt


############################################
#################################################################
def prob_of_a_qubit(psi, qubit):
    """taken from
    https://github.com/adaskin/a-simple-quantum-simulator.git
    computes probabilities in a quantum state for a given qubit.
    Parameters
    ----------
    psi: numpy 1 dimensional row vector
        representing a quantum state
    qubit:  int
        an integer number
        - the order of the qubits |0,1,..n-1>

    Returns
    -------
    numpy ndarray
        a vector that represents probabilities.
    """
    N = len(psi)

    n = int(np.log2(N))

    f = np.zeros(2)
    qshift = n - qubit - 1
    for j in range(N):
        # jbits = bin(j)[2:].zfill(n)
        # qbitval1 = int(jbits[qubit])
        qbitval = (j >> qshift) & 1

        # print(qbitval,qbitval1)
        f[qbitval] += np.real(np.abs(psi[j]) ** 2)
    return f


####################################################


def matrixC(nqubits, a):
    J = 1 / 2 * np.array([[1, 1], [1, 1]])
    M = np.array([[a, -1], [-1, a]])
    C = M
    for i in range(nqubits - 1):
        C = np.kron(J, C)
    return C


def matrixSquareRoot(nqubits, a):
    H = 1 / np.sqrt(2) * np.array([[1, -1], [1, 1]])
    Ht = 1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]])
    J = 1 / 2 * np.array([[1, 1], [1, 1]])
    M = np.array([[a, -1], [-1, a]])

    Dj = np.array([[0, 0], [0, 1]])
    Dm = np.array([[a + 1, 0], [0, a - 1]])
    I = np.eye(2**nqubits)

    print(Dm)

    S = (Dm / (a + 1)) ** 2
    Uh = H
    Uht = Ht

    for i in range(nqubits - 1):
        S = np.kron(Dj, S)
        Uh = np.kron(H, Uh)
        Uht = np.kron(Ht, Uht)

    S = Uht @ np.sqrt(I - S) @ Uh

    return S, Uht, Uh


def unitaryC(n, k=1, a=0.8):
    ntotal = n + k + 1
    N = 2**n
    K = 2**k

    # smaller a makes ket0 higher,
    # but marked smaller
    # a = 0.8

    # circuit
    # initial Hadamards on n+1 qubits and marked
    psi = 1 / np.sqrt(2 * N) * np.ones((2 * N, 1))

    # mark element on the second half i.e. psi=[v0;v1]
    imark = random.randint(N, 2 * N - 1)
    psi[imark] = -psi[imark]

    ancilla0 = np.array([[1], [0]])  # \ket{0}

    K2 = int(K / 2)  # excluding first and last qubits in the ancilla
    ancilla1 = 1 / np.sqrt(K2) * np.ones((K2, 1))

    ancilla = np.kron(ancilla0, ancilla1)

    # the whole quantum state before C
    psi1 = np.kron(ancilla, psi)

    C = matrixC(k, a)

    # the eigenvalues of  must be 0<= eig(C) <=1
    kappa_c = la.norm(C)
    print("kappac vs a+1", kappa_c, a + 1)
    C = C / (a + 1)

    S, Uht, Uh = matrixSquareRoot(k, a)

    # unitary C
    Uc = np.zeros((2 * K, 2 * K))
    Uc[0:K, 0:K] = C.copy()
    Uc[0:K, K : 2 * K] = -S.copy()
    Uc[K : 2 * K, 0:K] = S.copy()
    Uc[K : 2 * K, K : 2 * K] = C.copy()

    # np.abs(C@C.transpose()+S@S.transpose())
    # Uc@Uc.transpose()

    # global operator with the system register
    U = np.kron(Uc, np.eye(N))

    # apply to the state
    psi2 = U @ psi1

    return psi2, imark


def statsforstate(n, psi, imark):
    N = 2**n
    print("1/N:{}, 1/n:{}".format(1 / N, 1 / n))
    probs0 = prob_of_a_qubit(psi, 0)
    print("probabilities of the first qubit\n", probs0)

    [psirow, psicol] = psi.shape

    # when first qubit \ket{0}
    q0ket0 = psi[0 : int(psirow / 2)]

    normq0 = la.norm(q0ket0)
    q0ket0 = q0ket0 / la.norm(q0ket0)
    print("probability of the q0-ket0 state:", normq0**2)

    # for k=1 we have two marked state,
    print("probabilities where we have marked state")
    markstart = imark - N
    pmarked_total = 0
    for i in range(imark - N, q0ket0.shape[0], N):
        pmarked_total += np.abs(q0ket0[i]) ** 2
        print("marked: ", q0ket0[i] ** 2)
        print("Pmarked total: ", pmarked_total)

    return probs0[0], pmarked_total


def aa(psi, n1, n2, iter=1):
    """for a given two registers of n1 and n2 qubits,
    it amplifies the prob-0 of state in the first register
    """
    N1 = 2**n1
    N2 = 2**n2
    I1 = np.eye(N1)
    I2 = np.eye(N2)
    I = np.eye(N1 * N2)

    # first register ket{0}
    s = np.eye(N1, 1)
    # marking gate for the first register
    Us = I1 - 2 * s @ s.transpose()
    # global unitary
    U1 = np.kron(Us, I2)

    U2 = 2 * psi @ psi.transpose() - I

    # apply AA
    temp1 = psi.copy()
    for i in range(iter):
        temp2 = U1 @ temp1
        psi2 = U2 @ temp2
        temp1 = psi2

    return psi2


if __name__ == "__main__":
    n = 11
    N = 2**n
    # a = 0.75
    runs = np.zeros((n, 8))
    for i in range(1, n + 1):
        a = (i - 1) / (i)
        psi_beforeaa, imark = unitaryC(i, 1, a=a)
        beforeaa = statsforstate(i, psi_beforeaa, imark)

        psi_afteraa = aa(psi_beforeaa, 1, i + 1, iter=i)
        afteraa = statsforstate(i, psi_afteraa, imark)
        runs[i - 1][0] = i
        runs[i - 1][1] = 2**i
        runs[i - 1][2] = 1 / i
        runs[i - 1][3] = 1 / (2**i)
        runs[i - 1][4] = beforeaa[0]  # prob0
        runs[i - 1][5] = beforeaa[1]  # pmarked
        runs[i - 1][6] = afteraa[0]  # prob0
        runs[i - 1][7] = afteraa[1]  # pmarked

    # plt.plot(runs[1:,2],runs[1:,3],'o--', color='grey',label='1/N')
    plt.plot(runs[1:, 0], np.sqrt(runs[1:, 3]), "x--", label="$1/\sqrt{N}$")
    plt.plot(runs[1:, 0], runs[1:, 2], "x--", label="1/n")

    plt.plot(runs[1:, 0], np.abs(runs[1:, -4]), "o:", label="$P_0$ before AA")
    plt.plot(runs[1:, 0], np.abs(runs[1:, -2]), "*:", label="$P_0$ after AA")

    plt.plot(runs[1:, 0], np.abs(runs[1:, -3]), "^-", label="$P_{marked}$ before AA ")
    plt.plot(runs[1:, 0], np.abs(runs[1:, -1]), "*-", label="$P_{marked}$ after AA ")
    plt.xticks(ticks=runs[1:, 0], labels=range(2, n + 1))

    plt.xlabel("the value of $n$")
    plt.ylabel("Magnitude")
    plt.legend()
    # axs[1].set_xticks(RE.keys())
    # figname = "a-{}.png".format(a)
    # plt.savefig(figname)
    plt.show()
