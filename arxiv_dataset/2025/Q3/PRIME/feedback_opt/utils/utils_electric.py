import numpy as np

# Code from Nullspace Power Balance and Nullspace Power Flow, Keith


def diag(x):
    diagmat = np.asmatrix(np.diagflat(x))
    return diagmat


def pointybracket(M):
    """
    from Bolognani fast power system analysis via implicit linearization
    """
    return np.asmatrix(
        np.vstack((np.hstack((np.real(M), -np.imag(M))), np.hstack((np.imag(M), np.real(M)))))
    )


def pointybracket_forVector(x):
    return np.asmatrix(np.vstack((np.real(x), np.imag(x))))


def makeN(dimN):
    """
    from Bolognani fast power system analysis via implicit linearization
    """
    return np.asmatrix(np.diag(np.hstack((np.ones(int(dimN / 2)), -np.ones(int(dimN / 2))))))
    # left multiplying by N multiplies the bottom half entries by -1
    # right multiplying by N multiplies the right half entries by -1
    # pointybracket(np.conj(M)) = N*pointybracket(M)*N
    # pointybracket_forvector(np.conj(x)) = N*pointybracket(x)


def R(ucomp):
    """
    Sensitivity of real and reactive portion of a vector to the mag and angle of the vector
    from Bolognani fast power system analysis via implicit linearization
    """
    theta = np.angle(ucomp)
    v = np.absolute(ucomp)
    return np.asmatrix(
        np.vstack(
            (
                np.hstack(
                    (np.diagflat(np.cos(theta)), -np.diagflat(v) * np.diagflat(np.sin(theta)))
                ),
                np.hstack(
                    (np.diagflat(np.sin(theta)), np.diagflat(v) * np.diagflat(np.cos(theta)))
                ),
            )
        )
    )


def mag_sens_toRealAndImag(xcomp):
    """
    see lineflow sensitivity notes
    """
    theta = np.angle(xcomp)
    return np.asmatrix(np.hstack((np.diagflat(np.cos(theta)), np.diagflat(np.sin(theta)))))


def cart2pol_lists(x_list, y_list):
    assert len(x_list) == len(y_list)
    rho_list = []
    phi_list = []
    for i, x in enumerate(x_list):
        y = y_list[i]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        rho_list.append(rho)
        phi_list.append(phi)
    return rho_list, phi_list


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def get_sens_powerInjections_to_voltage(Y, U):
    """
    DESCRIPTION
    Returns the sensitivity of the power injection to changes in the network voltages for the network described by Y at voltage U.

    INPUTS
    - Y: The dense bus admittance matrix. Complex valued.
    - U: 1-D array of complex voltages of all the nodes on the network in nodeorder. This could presumably be relaxed to some subset of all of the nodes.

    OUTPUTS
    - Gamma_powerInjections_to_voltage: the sensitivity matrix for voltage mag and angle to power injections

    NOTES
    - for transformers, Y_ij != Y_ji. We assume that if one said of the transformer does not have flow constraint violation, then neither does the other side.
    - This code is an adapted version of GeminiCode's calculate_sensitivity()
    - It can handle power balancing (see Keith's dissertation) but that is not necessary for OPF
    """

    n_nodes = int(np.shape(Y)[0])
    assert n_nodes == len(np.ravel(U))
    U = np.asmatrix(np.reshape(U, (n_nodes, 1)))

    N = makeN(n_nodes * 2)

    Gamma_allNodes = (
        pointybracket(diag(np.conj(Y * U))) + pointybracket(diag(U)) * N * pointybracket(Y)
    ) * R(U)

    return Gamma_allNodes
