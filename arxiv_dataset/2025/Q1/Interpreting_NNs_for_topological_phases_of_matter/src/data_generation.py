import concurrent.futures
import ctypes
import pathlib
import pickle
import re
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns
import sympy as sp
from matplotlib.colors import Normalize
from numba import njit
from numba.typed import Dict, List
from tqdm.auto import tqdm, trange

"""
The following code is an implementation of the exact diagonalization of the
1D Bose-Hubbard model, following the method described in the paper by Zhang
and Dong (2010), DOI:10.1088/0143-0807/31/3/016, adapted to the SSH model.

"""


def calc_Dim(M, N, statistic="b"):
    """
    A function for calculation of Fock basis dimensions, depending on the
    system's statistic.

    Parameters
    ----------
    M : int
        Number of sites in the 1D chain.
    N : int
        Number of particles in the system.
    statistic : str, optional
        Statistic the system's particles follow. Allowed values are 'f' for
        Fermions, and 'b' for Bosons. The default is "b".

    Raises
    ------
    ValueError
        This error is raised if incorrect statistic alias is entered.

    Returns
    -------
    dim : int
        Fock basis dimension.

    """
    if statistic == "b":
        return int(np.math.gamma(N + M) / (np.math.gamma(N + 1) * np.math.gamma(M)))
    elif statistic == "f":
        # return np.int64(
        #     (np.math.gamma(M + 1)) / (np.math.gamma(N + 1) * np.math.gamma(M - N + 1))
        # )
        return np.int64(sc.special.comb(M, N, exact=True))
    else:
        raise ValueError("Incorrect statistic")


def calc_dim_tab(M, N_tab, stat_tab):
    """
    Calculate the dimension table for given parameters.

    Parameters:
    - M (int): The total number of sites.
    - N_tab (list): A list of integers representing the fermion numbers.
    - stat_tab (list): A list of strings representing the statistics of the fermions.

    Returns:
    - int: The product of all the dimensions in the dimension table.

    Raises:
    - ValueError: If the statistic is incorrect or the fermion number is non-physical.
    """

    dim_arr = np.array([1.0], dtype=np.float64)
    for (n_i, stat_i) in zip(N_tab, stat_tab):
        if stat_i == "b":
            D = calc_Dim(M, n_i, stat_i)
            dim_arr = np.append(dim_arr, [D])
        elif stat_i == "f":
            if n_i > M:
                raise ValueError("Non-physical fermion number")
            dim_F = calc_Dim(M, n_i, stat_i)
            dim_arr = np.append(dim_arr, [dim_F])
        else:
            raise ValueError("Incorrect statistic")
    return int(dim_arr.prod())


def get_tensor_basis(M, N, statistic=["b"], component_count=1, verb=0):
    """
    A function for generation of Fock basis for 1 or 2 component systems of
    either Fermi-Dirac or Bose-Einstein statistic.

    Parameters
    ----------
    M : int
        Number of sites in 1D chain.
    N : np.array
        An array containing particle counts in either of the components.
        Its length should match the component_count parameter.
    statistic : np.array, optional
        An array containing particle statistics in either of the components.
        Possible values are 'f' for fermions, and 'b' for bosons.
        Its length should match the component_count parameter.
        The default is ["b"].
    component_count : int, optional
        Number of Fock basis components. For composite/spinful systems it
        should be 2, for simple ones, it should be 1. The default is 1.
    verb : int, optional
        If verb > 0m then technical prints will show up. The default is 0.

    Raises
    ------
    ValueError
        Raises this error if 'N' is not a NumPy array, or if incorrect
        statistic is passed.

    Returns
    -------
    A : np.array
        An array containing Fock basis vectors as rows sorted lexicographically.

    """
    if len(N) == 1:
        N_val = N[0]
        stat_val = statistic[0]
        vec_num = 0
        if stat_val == "b":
            D = calc_Dim(M, N_val, "b")
            A = np.zeros(shape=(D**2, M * component_count), dtype=np.int64)
        elif stat_val == "f":
            if N_val > M:
                raise ValueError("Non-physical fermion number")
            dim_F = calc_Dim(M, N_val, stat_val)
            A = np.zeros(shape=(dim_F**2, M * component_count), dtype=np.int64)
        else:
            raise ValueError("Incorrect statistic")
        if component_count == 1:
            return get_basis(M, N_val, statistic=stat_val, verb=verb)
        else:
            bases = {}
            for comp_num in range(0, component_count):
                if verb > 0:
                    print("Generating basis for component no.", comp_num)
                bases[comp_num] = get_basis(M, N_val, statistic=stat_val, verb=verb)

            for ket_1 in bases[0]:
                for ket_2 in bases[1]:
                    A[vec_num] = np.concatenate((ket_1, ket_2))
                    vec_num += 1
        return A
    elif len(N) > 1:
        N_tab = N
        stat_tab = statistic

        dim_prod = calc_dim_tab(M, N_tab, stat_tab)
        A = np.zeros(shape=(dim_prod, M * component_count), dtype=np.int64)

        vec_num = 0
        if component_count == 1:
            return get_basis(M, N_tab[0], statistic=statistic, verb=verb)
        else:
            bases = {}
            for (n_i, stat_i, comp_num) in zip(N, stat_tab, range(0, component_count)):
                if verb > 0:
                    print("Generating basis for component no.", comp_num)
                bases[comp_num] = get_basis(M, n_i, statistic=stat_i, verb=verb)

            for ket_1 in bases[0]:
                for ket_2 in bases[1]:
                    A[vec_num] = np.concatenate((ket_1, ket_2))
                    vec_num += 1
        return A
    else:
        raise ValueError("Please pass 'N' as a list")


def get_basis(M, N, statistic="b", verb=0):
    """
    Generate basis vectors for a given system size and statistic.

    Parameters:
    - M (int): The total number of sites in the system.
    - N (int): The number of particles in the system.
    - statistic (str, optional): The statistic of the particles. Can be "b" for bosons or "f" for fermions. Defaults to "b".
    - verb (int, optional): Verbosity level. Set to 0 for no output, 1 for printing the basis vectors. Defaults to 0.

    Returns:
    - A (ndarray): The basis vectors for the given system size and statistic.

    Raises:
    - ValueError: If an incorrect statistic is provided.

    """
    # Defining empty array for the basis vectors
    if statistic == "b":
        D = calc_Dim(M, N, "b")
        A = np.zeros(shape=(D, M), dtype=np.int64)
    elif statistic == "f":
        if N > M:
            raise ValueError("Non-physical fermion number")
        D = calc_Dim(M, N, "b")
        A = np.zeros(shape=(D, M), dtype=np.int64)
        # dim_F = math.comb(M, N)
        dim_F = calc_Dim(M, N, statistic)
        A_F = np.zeros(shape=(dim_F, M), dtype=np.int64)
        a_f_ind = 0
    else:
        raise ValueError("Incorrect statistic")

    # Setting up the basis vectors
    for num, row in enumerate(A):
        if num != 0:
            row_old = A[num - 1].copy()
            row_new = row.copy()
            k_ind = np.max(np.nonzero(row_old[: M - 1])[0])
            if row_old[k_ind + 1] != 0:
                k_ind = np.max(np.nonzero(A[num - 2][: M - 1])[0])
            # Setting the new row
            for i in range(0, k_ind):
                row_new[i] = row_old[i]
            row_new[k_ind] = row_old[k_ind] - 1
            existing_sum = 0
            for pos in range(len(row_new)):
                if pos >= k_ind + 3:
                    continue
                else:
                    existing_sum += row_new[pos]
            row_new[k_ind + 1] = N - existing_sum

            # Copying back the new row
            A[num] = row_new
            if statistic == "f":
                if np.max(row_new) < 2:
                    A_F[a_f_ind] = row_new
                    a_f_ind += 1
        else:
            row[0] = N
            if statistic == "f":
                if np.max(row) < 2:
                    A_F[a_f_ind] = row
                    a_f_ind += 1

    if verb > 0:
        if statistic == "b" or statistic == "f_2":
            print(A)
        elif statistic == "f":
            print(A_F)
        else:
            raise ValueError("Incorrect statistic")
    if statistic == "b":
        A[-1, -1] = N
        return A
    elif statistic == "f":
        A_F[-1, -1] = 1
        return A_F
    else:
        raise ValueError("Incorrect statistic")


@njit
def a_operator(
    i_ind,
    ket,
    kind="c",
    component_number=None,
    ket_size=None,
    statistic="b",
    pbc=0,
    component_count=1,
):
    """
    Apply the creation or annihilation operator to a ket vector.

    Parameters:
    - i_ind (int): The index of the site to be modified.
    - ket (ndarray): The ket vector to be modified.
    - kind (str, optional): The type of operator to apply. Default is "c" (creation operator).
    - component_number (int, optional): The component number of the ket vector. Default is None.
    - ket_size (int, optional): The size of each component in the ket vector. Default is None.
    - statistic (str, optional): The statistic of the ket vector. Default is "b" (bosonic).
    - pbc (int, optional): The boundary condition. Default is 0 (no periodic boundary condition).
    - component_count (int, optional): The number of components in the ket vector. Default is 1.

    Returns:
    - factor (float): The factor by which the ket vector is multiplied.
    - ket_new (ndarray): The modified ket vector.

    """

    # Setting which site should be modified based on boundary conditions.
    if pbc == 0:
        i_mod = i_ind
    else:
        i_mod = np.mod(i_ind, ket.shape[0])

    # Creation operator
    if kind == "c":
        if statistic == "b":
            if component_count == 1:
                factor = np.sqrt(ket[i_mod] + 1)
                ket_new = ket.copy()
                ket_new[i_mod] += 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                factor = np.sqrt(ket_to_mod[i_mod] + 1)
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] += 1
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                return factor, ket_out
        elif statistic == "f":
            if component_count == 1:
                if ket[i_mod] == 1:
                    return 0, ket
                factor = np.sqrt(ket[i_mod] + 1)
                ket_new = ket.copy()
                ket_new[i_mod] += 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                if ket_to_mod[i_mod] == 1:
                    return 0, ket
                minus_factor = np.count_nonzero(
                    ket[
                        component_number * ket_size : component_number * ket_size
                        + i_mod
                    ]
                )
                factor = np.sqrt(ket_to_mod[i_mod] + 1) * np.power(-1, minus_factor)
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] += 1
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                return factor, ket_out

    # Anihillation operator
    if kind == "a":
        if statistic == "b":
            if component_count == 1:
                if ket[i_mod] == 0:
                    return 0, ket
                factor = np.sqrt(ket[i_mod])
                ket_new = ket.copy()
                ket_new[i_mod] -= 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                if ket_to_mod[i_mod] == 0:
                    return 0, ket
                factor = np.sqrt(ket_to_mod[i_mod])
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] -= 1
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                return factor, ket_out
        elif statistic == "f":
            if component_count == 1:
                if ket[i_mod] == 0:
                    return 0, ket
                factor = np.sqrt(ket[i_mod])
                ket_new = ket.copy()
                ket_new[i_mod] -= 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                if ket_to_mod[i_mod] == 0:
                    return 0, ket
                minus_factor = np.count_nonzero(
                    ket[
                        component_number * ket_size : component_number * ket_size
                        + i_mod
                    ]
                )
                factor = np.sqrt(ket_to_mod[i_mod]) * np.power(-1, minus_factor)
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] -= 1
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                return factor, ket_out

    return factor, ket_new


@njit
def tag_func(v):
    """
    The tagging function, which hashes all the basis vectors with unique values, according to equation:
    T(v) = \sum_i \sqrt{p_i} v_i where i is the index of element in the vector

    Parameters:
    - v (ndarray): The basis vector to be tagged.

    Returns:
    - tag (float): The tag of the basis vector.
    """
    res_tab = np.zeros(len(v))
    for i in range(0, len(v)):
        res_tab[i] = np.sqrt(100 * (i + 1) + 3) * v[i]
    return np.sum(res_tab)


@njit
def find_orig_idx(ket, t_dict):
    """
    Function for finding which basis vector is the input ket
    in the original ordered basis of the Hilbert space.

    Parameters:
    - ket: Hashed ket to be searched in the original Hilbert space basis
    - t_dict: dictionary of hashed basis vectors

    Returns:
    - Index of input vector in the original ordered basis of the Hilbert space
    """
    ket_hash = tag_func(ket)
    try:
        a = t_dict[ket_hash]
        return int(a)
    except Exception:
        return -1


"""Construction of the Hamiltonian"""


@njit
def random_func(seed: int = 1234) -> float:
    """
    Generate a random float number between -0.5 and 0.5 using a seeded pseudo-random number generator.
    This is a Numba-compatible version of the function.

    Parameters:
    - seed (int): The seed value for the pseudo-random number generator. Default is 1234.

    Returns:
    - float: The randomly generated float number.
    """
    seeded_prng = np.random.seed(seed)
    random_computation = np.random.uniform(-0.5, 0.5)
    return random_computation


# Access the _PyTime_AsSecondsDouble and _PyTime_GetSystemClock functions from pythonapi
get_system_clock = ctypes.pythonapi._PyTime_GetSystemClock
as_seconds_double = ctypes.pythonapi._PyTime_AsSecondsDouble

# Set the argument types and return types of the functions
get_system_clock.argtypes = []
get_system_clock.restype = ctypes.c_int64

as_seconds_double.argtypes = [ctypes.c_int64]
as_seconds_double.restype = ctypes.c_double


@njit
def t_ns():
    """
    Get the current time in nanoseconds.

    Returns:
    - int: The current time in nanoseconds.
    """
    system_clock = get_system_clock()
    current_time = as_seconds_double(system_clock)
    return current_time


@njit
def get_kinetic_H_vw(
    A,
    M,
    J,
    t_dict,
    vw=[1, 1],
    disord_W=[0, 0],
    pbc=0,
    statistic="b",
    component_no=1,
    component_count=1,
):
    """
    A function calculating the kinetic Hamiltonian of a SSH system, defined with staggered hopping amplitudes v, w.

    Parameters
    ----------
    A : Numpy Array of int64
        Fock Basis on which we operate, either fermionic, or bosonic. Base's statistic should match the one given in 'statistic' parameter.
    M : int
        Number of sites in the system. A site is a building block of elementary cell.
    J : int
        Hopping scaling factor, as stated in Hamiltonian definition.
    t_dict : numba.typed.typedict.Dict
        A dictionary connecting hashed and sorted basis vectors with their original positions.
    vw : Numpy Array of float64, optional
        A vector containing hopping amplitudes v, w. The default is [1, 1].
    disord_W : Numpy Array of float64, optional
        An array setting the disorder amplitude on respectively inter and intra cell hopping. The default is [0, 0].
    pbc : int, optional
        Periodic (1) or open(0) boundary conditions imposed on SSH chain. The default is 0.
    statistic : Numpy Array of str32, optional
        Statistic describing the particles in the system 'b' - bosonic, 'f' - fermionic. The default is "b".
    component_no : int, optional
        Which part of Fock basis vector should be acted on. Only matters if presented with two-component Fock basis. The default is 1.
    component_count : int, optional
        Declaration of Fock basis components in the system. The default is 1.

    Returns
    -------
    values : numba.typed.typedlist.List
        A list of values to assign to sparse matrix - Hamiltonian.
    ket_idx_list : numba.typed.typedlist.List
        An row number list in the resulting sparse matrix, corresponding to the indices of original basis vectors.
    ket_tilde_tilde_idx_list : numba.typed.typedlist.List
        An column number list in the resulting sparse matrix, corresponding to the indices of original basis vectors.

    """

    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    ket_len = M

    if pbc == 0:
        range_end = M - 1
    else:
        range_end = M

    for ket in A:
        # We set independent random generator states for genrator
        # Seed we set is system time modulo max integer capacity
        r_1 = random_func(int(t_ns() * (10**9)) % (2**32 - 1))
        W_1_disord = disord_W[0] * r_1
        r_2 = random_func(int(t_ns() * (10**9)) % (2**32 - 1))
        W_2_disord = disord_W[1] * r_2

        # Hitchhiker's guide through hopping:
        # w + W_1 * r_n' <- Intercell Hopping (Between neighboring cells)
        # v + W_2 * r_n  <- Intracell Hopping (Within unit cell)

        stag_dict = {1: (vw[1] + W_1_disord), -1: (vw[0] + W_2_disord), 0: 0}

        for i in range(0, range_end):

            if pbc == 0:
                i_next = i + 1
            else:
                i_next = np.mod(i + 1, ket_len)

            for (anihillated, created) in [(i, i_next), (i_next, i)]:
                if component_count > 1:
                    multicomp_factor = component_no * ket_len
                else:
                    multicomp_factor = 0
                if ket[multicomp_factor + anihillated] > 0:

                    factor_1, ket_tilde = a_operator(
                        anihillated,
                        ket,
                        "a",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_tilde_tilde = a_operator(
                        created,
                        ket_tilde,
                        "c",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = int(find_orig_idx(ket, t_dict))
                    ket_tilde_tilde_idx = int(find_orig_idx(ket_tilde_tilde, t_dict))

                    val = (stag_dict[((-1) ** (i_next))]) * factor_1 * factor_2 * J

                    if ket_tilde_tilde_idx == -1 or np.abs(val) == 0:
                        continue
                    else:

                        values.append(val)

                        ket_idx_list.append(ket_idx)

                        ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

    return values, ket_idx_list, ket_tilde_tilde_idx_list


@njit
def get_kinetic_H_vw_both(
    A,
    M,
    J,
    t_dict,
    vw=[1, 1],
    disord_W=[0, 0],
    statistic="b",
    component_no=1,
    component_count=1,
):

    """
    A function calculating the kinetic Hamiltonian of a SSH system, defined with staggered hopping amplitudes v, w.

    Parameters
    ----------
    A : Numpy Array of int64
        Fock Basis on which we operate, either fermionic, or bosonic. Base's statistic should match the one given in 'statistic' parameter.
    M : int
        Number of sites in the system. A site is a building block of elementary cell.
    J : int
        Hopping scaling factor, as stated in Hamiltonian definition.
    t_dict : numba.typed.typedict.Dict
        A dictionary connecting hashed and sorted basis vectors with their original positions.
    vw : Numpy Array of float64, optional
        A vector containing hopping amplitudes v, w. The default is [1, 1].
    disord_W : Numpy Array of float64, optional
        An array setting the disorder amplitude on respectively inter and intra cell hopping. The default is [0, 0].
    statistic : Numpy Array of str32, optional
        Statistic describing the particles in the system 'b' - bosonic, 'f' - fermionic. The default is "b".
    component_no : int, optional
        Which part of Fock basis vector should be acted on. Only matters if presented with two-component Fock basis. The default is 1.
    component_count : int, optional
        Declaration of Fock basis components in the system. The default is 1.

    Returns
    -------
    values : numba.typed.typedlist.List
        A list of values to assign to sparse matrix - Hamiltonian.
    ket_idx_list : numba.typed.typedlist.List
        An row number list in the resulting sparse matrix, corresponding to the indices of original basis vectors.
    ket_tilde_tilde_idx_list : numba.typed.typedlist.List
        An column number list in the resulting sparse matrix, corresponding to the indices of original basis vectors.

    """

    # PBC lists

    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    # OBC lists
    values_obc = List()

    ket_idx_list_obc = List()

    ket_tilde_tilde_idx_list_obc = List()

    ket_len = M

    for ket in A:
        # We set independent random generator states for genrator
        # Seed we set is system time modulo max integer capacity
        r_1 = random_func(int(t_ns() * (10**9)) % (2**32 - 1))
        W_1_disord = disord_W[0] * r_1
        r_2 = random_func(int(t_ns() * (10**9)) % (2**32 - 1))
        W_2_disord = disord_W[1] * r_2

        # Hitchhiker's guide through hopping:
        # w + W_1 * r_n' <- Intercell Hopping (Between neighboring cells)
        # v + W_2 * r_n  <- Intracell Hopping (Within unit cell)

        stag_dict = {1: (vw[1] + W_1_disord), -1: (vw[0] + W_2_disord), 0: 0}

        for pbc in [1, 0]:
            if pbc == 0:
                range_end = M - 1
            else:
                range_end = M

            for i in range(0, range_end):

                if pbc == 0:
                    i_next = i + 1
                else:
                    i_next = np.mod(i + 1, ket_len)

                for (anihillated, created) in [(i, i_next), (i_next, i)]:
                    if component_count > 1:
                        multicomp_factor = component_no * ket_len
                    else:
                        multicomp_factor = 0
                    if ket[multicomp_factor + anihillated] > 0:

                        factor_1, ket_tilde = a_operator(
                            anihillated,
                            ket,
                            "a",
                            pbc=pbc,
                            statistic=statistic,
                            component_number=component_no,
                            ket_size=ket_len,
                            component_count=component_count,
                        )
                        factor_2, ket_tilde_tilde = a_operator(
                            created,
                            ket_tilde,
                            "c",
                            pbc=pbc,
                            statistic=statistic,
                            component_number=component_no,
                            ket_size=ket_len,
                            component_count=component_count,
                        )

                        ket_idx = int(find_orig_idx(ket, t_dict))
                        ket_tilde_tilde_idx = int(
                            find_orig_idx(ket_tilde_tilde, t_dict)
                        )

                        val = (stag_dict[((-1) ** (i_next))]) * factor_1 * factor_2 * J

                        if ket_tilde_tilde_idx == -1 or np.abs(val) == 0:
                            continue
                        else:
                            if pbc == 1:
                                values.append(val)

                                ket_idx_list.append(ket_idx)

                                ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)
                            else:
                                values_obc.append(val)

                                ket_idx_list_obc.append(ket_idx)

                                ket_tilde_tilde_idx_list_obc.append(ket_tilde_tilde_idx)

    return (values, ket_idx_list, ket_tilde_tilde_idx_list), (
        values_obc,
        ket_idx_list_obc,
        ket_tilde_tilde_idx_list_obc,
    )


def generate_dataset(
    nsamples=200,
    which_dataset="test",
    M=50,
    J=1,
    w_ssh=1,
    real_num=0,
    n_reals_tot=50,
    W=0,
    counter_start=0,
    component_count=1,
    override_save_loc=None,
):
    """
    Function for generating the dataset for the CNN training/validation/testing

    Parameters
    ----------
    nsamples : int, optional
        Number of samples in the dataset. The default is 200.
    M : int, optional
        Number of sites in the system. The default is 50.
    J : int, optional
        Hopping scaling factor. The default is 1.
    w_ssh : int, optional
        SSH intercell hopping amplitude. The default is 1.
    real_num : int, optional
        Number of the disorder realisation. The default is 0.
    n_reals_tot : int, optional
        Number of all computed disorder realisations. The default is 50.
    W : int, optional
        Disorder strength. The default is 0.
    counter_start : int, optional
        Starting number of the disorder realisation. The default is 0.
    component_count : int, optional
        Number of components in the basis vector. For spinless fermions it is 1. The default is 1.

    Returns
    -------
    None.
    """

    if override_save_loc is None:
        save_loc = pathlib.Path(
            f"./Datasets/M={M}/{n_reals_tot}-disorder-realisations/W={W}/"
        )
    else:
        save_loc = pathlib.Path(override_save_loc)
    save_loc.mkdir(exist_ok=True, parents=True)

    # Control parameters

    # SSH intercell hopping amplitude
    ww = w_ssh

    # =============================================================================
    # Split range for the validation and training datasets
    # =============================================================================

    # Training dataset values
    if which_dataset == "training":
        vv_tab = np.append(
            np.linspace(0, 0.8, nsamples // 2), np.linspace(1.2, 2, nsamples // 2)
        )

    # Validation dataset values
    elif which_dataset == "validation":
        vv_tab = np.append(
            np.linspace(0.001, 0.801, nsamples // 2),
            np.linspace(1.201, 2.001, nsamples // 2),
        )

    # Test dataset values
    elif which_dataset == "test":
        vv_tab = np.linspace(0.002, 2.002, nsamples)

    else:
        raise ValueError("Incorrect dataset type")

    if which_dataset != "test":
        dataset_designation = which_dataset
    else:
        dataset_designation = f"{which_dataset}_{counter_start+real_num}"

    # Initialization of lists for winding number and observables
    winding_number_tab = []
    observable_tab = []

    w_tab_data = []

    v_tab_data = []

    # N - Number of particles in a system
    N = np.array([1])  # If number - the same number for both systems

    # Statistic (bose/fermi)
    stat_vec = np.array(["f"])

    # =============================================================================
    #     In this file we will first calculate the winding number for the given v,
    #     using periodic boundary conditions, and then calculate the same system
    #     assuming open boundary conditions, as only in this scenario we can see
    #     the edge states, which we would like to investigate
    # =============================================================================

    for vv in tqdm(vv_tab, colour="cyan", unit="H", desc="Sample: ", leave=False):
        # =========================================================================
        #     Part responsible for calculating the winding number for the system
        # =========================================================================

        # For winding number calculation we set periodic bounary conditions (PBC)
        pbc = 1

        # Staggering tab - SSH parameter
        vw_tab = np.array([vv, ww])

        # Disorder Measure - W_1, W_2
        W_disorder_tab = np.array([0.5 * W, W])

        stat = stat_vec

        # D - Dimension of the final Hamiltonian matrix
        D = calc_dim_tab(M, N, stat_tab=stat)

        A = get_tensor_basis(
            M, N, statistic=stat, component_count=component_count, verb=0
        )

        # Getting the basis vectors hashed
        tab_T = np.array([tag_func(v) for v in A])
        # Preserving the original order of basis vectors
        ind = np.argsort(tab_T)
        # Sorting the new array for more efficient searching (by bisection)
        t_sorted = tab_T.copy()
        t_sorted.sort()
        t_dict = Dict()

        for key, val in zip(tab_T, np.arange(0, A.shape[0])):
            t_dict[key] = val

        # Initialization of lists for Hamiltonian matrices

        H_hop_pbc = [List(), List(), List()]
        H_hop_obc = [List(), List(), List()]

        # Getting the Hamiltonian matrices
        """
        We will calculate the Hamiltonian for both PBC and OBC.
        This way we have exactly the same system, the same random numbers,
        the same tunneling amplitudes, but different boundary conditions.
        """
        H_hop_pbc, H_hop_obc = get_kinetic_H_vw_both(
            A,
            M,
            J,
            t_dict,
            vw=vw_tab,
            disord_W=W_disorder_tab,
            statistic=stat[0],
            component_count=component_count,
            component_no=0,
        )

        # Converting the Hamiltonian to sparse format
        # PBC
        H_hop_pbc = sc.sparse.coo_matrix(
            (H_hop_pbc[0], (H_hop_pbc[1], H_hop_pbc[2])), shape=(D, D)
        )

        H_hop_pbc = sc.sparse.triu(H_hop_pbc)

        H_hop_pbc = (
            H_hop_pbc
            + H_hop_pbc.T
            - sc.sparse.diags(H_hop_pbc.diagonal(), format="coo")
        )

        H_dense_pbc = (J * H_hop_pbc).toarray()

        dvec = np.array(
            [(lambda snum: 1 if snum % 2 == 0 else -1)(snum) for snum in range(M)]
        )

        chiral = sc.sparse.coo_matrix(sc.sparse.diags(dvec))

        ch_evals, ch_evecs = np.linalg.eigh(chiral.toarray())

        sorted_evecs = np.zeros(shape=(D, D))
        sorted_evals = np.zeros(shape=(D,))

        for num, i in enumerate(
            np.repeat(
                np.linspace(0, D // 2, D // 2, endpoint=False, dtype=np.int64),
                2,
            )
        ):
            rownum = np.where(ch_evecs[:, -(num + 1)] == 1)[0][0]
            fact = 0 if num % 2 == 0 else int(D / 2)
            new_rownum = fact + i
            new_row = ch_evecs[rownum].copy()
            sorted_evecs[new_rownum] = new_row
            sorted_evals[new_rownum] = ch_evals[rownum].copy()

        ch_evecs = sorted_evecs
        ch_evals = sorted_evals

        H_new_basis = ch_evecs @ H_dense_pbc @ (np.linalg.inv(ch_evecs))

        u = H_new_basis[D // 2 :, : D // 2]

        v, w, k = sp.symbols("v w k", real=True)

        u = sp.Matrix(u)

        u[-1, 0] = u[-1, 0] * sp.exp(sp.I * k)

        dkh = sp.simplify(sp.Derivative(u, k).doit())

        hinv = sp.Inverse(u).doit()

        h1_fin = hinv @ dkh

        expr1 = sp.Trace(h1_fin).doit() / (2 * sp.pi * sp.I)

        integrand1 = sp.lambdify(k, expr1.doit(), "scipy")

        r1, r1err = sc.integrate.quad(integrand1, -np.pi, np.pi - np.finfo(float).eps)

        winding_number_tab.append(r1)

        del (
            r1,
            r1err,
            hinv,
            h1_fin,
            expr1,
            integrand1,
            u,
            H_new_basis,
            H_dense_pbc,
            H_hop_pbc,
        )

        # =========================================================================
        #     Part responsible for calculating the eigenvectors observables to feed
        #     the CNN with
        # =========================================================================

        # Now switch to open bounary conditions (OBC)

        # Converting the Hamiltonian to sparse format
        # OBC
        H_hop_obc = sc.sparse.coo_matrix(
            (H_hop_obc[0], (H_hop_obc[1], H_hop_obc[2])), shape=(D, D)
        )

        H_hop_obc = sc.sparse.triu(H_hop_obc)

        H_hop_obc = (
            H_hop_obc
            + H_hop_obc.T
            - sc.sparse.diags(H_hop_obc.diagonal(), format="coo")
        )

        H_obc = J * H_hop_obc

        H_tot_obc = H_obc

        H_dense_obc = H_tot_obc.toarray()

        evals_obc, evecs_obc = np.linalg.eigh(H_dense_obc)

        observable = evecs_obc**2

        observable_tab.append(observable)

        w_tab_data.append(W)

        v_tab_data.append(vv)

    res_dict = {
        "W": w_tab_data,
        "labels": np.round(np.abs(winding_number_tab)).astype(dtype=np.int64),
        "data": observable_tab,
        "v": v_tab_data,
    }

    res_file = save_loc.joinpath(f"{dataset_designation}_set.pickle")

    with open(res_file, "wb") as f:
        pickle.dump(res_dict, f)

    del (
        res_dict,
        winding_number_tab,
        w_tab_data,
        observable_tab,
        v_tab_data,
        H_dense_obc,
        observable,
        H_obc,
        H_hop_obc,
        H_tot_obc,
        evals_obc,
        evecs_obc,
        t_dict,
    )

    return


def symmetrical_colormap(cmap_settings, new_name=None):
    """
    This function takes a colormap and creates a new one, as the concatenation of itself by a symmetrical fold.
    """
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # this defined the roughness of the colormap, 128 fine
    n = 50

    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))  # take the standard colormap # 'right-part'
    colors_l = colors_r[
        ::-1
    ]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


if __name__ == "__main__":
    generate_dataset()
