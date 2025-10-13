"""
it aplies a not unitary gate 
to construct the final state 
in the Grover's search algorithm
@adaskin, 2024
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def with_nonunitary2(n):
    N = 2**n

    I = np.eye(N)
    psi = 1 / np.sqrt(2 * N) * np.ones((2 * N, 1))

    # mark element on the second half
    imark = random.randint(N, 2 * N - 1)

    psi[imark] = -psi[imark]

    a = (n - 2) / n  # you can try poly(n) e.g n**2 too.
    b = 1
    M = np.array([[-a, +b], 
                  [+1, -1]])
    # M = np.array([[-1, 1], [1, -1]])

    U = np.kron(M, I)
    psi2 = U @ psi
    # we look at imark-N because it is in the upper part..
    print("elements p> 1/2N:\n", np.argwhere(np.abs(psi2[:]) > 1 / (2 * N)))
    print("norm of out state:", np.linalg.norm(psi2))
    print("px, pmax: ", psi2[imark - N] ** 2, np.max(psi2) ** 2)
    print("1/N:", 1 / N)
    print("1/n:", (1 / n) ** 2)

    psi3 = psi2 / np.linalg.norm(psi2)
    print("norm of out state:", np.linalg.norm(psi3))
    print("px, pmax: ", psi3[imark - N] ** 2, np.max(psi3) ** 2)
    print("1/N:", 1 / N)
    print("1/n:", (1 / n))
    return n, 1 / n, N, 1 / N, psi2[imark - N], psi3[imark - N]


if __name__ == "__main__":
    n = 11
    runs = np.zeros((n, 6))
    for i in range(1, n + 1):
        r = with_nonunitary2(i)
        for j in range(len(r)):
            runs[i - 1][j] = r[j]

    plt.plot(runs[1:, 0], np.sqrt(runs[1:, 3]), "x--", label="$1/\sqrt{N}$")
    plt.plot(runs[1:, 0], runs[1:, 1], "o--", label="1/n")
    plt.plot(
        runs[1:, 0],
        np.abs(runs[1:, -2]),
        "o:",
        label="absolute value of marked-unnormalized",
    )
    plt.plot(
        runs[1:, 0],
        np.abs(runs[1:, -1]),
        "^-",
        label="absolute value of marked-normalized",
    )
    plt.xticks(ticks=runs[1:, 0], labels=range(2, n + 1))

    plt.xlabel("the value of $n$")
    plt.ylabel("Magnitude")
    plt.legend()

    plt.show()
    #plt.savefig("foo.png", bbox_inches="tight")
