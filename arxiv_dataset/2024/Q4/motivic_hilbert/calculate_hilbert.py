from numpy.polynomial import Polynomial
import argparse

def generate_delta_(delta, delete):
    """
    Removes all occurrences of a specified element from the list.

    Parameters:
    delta (list): The list from which to remove elements.
    delete (any): The element to remove from the list.

    Returns:
    list: A new list with the specified element removed.
    """
    delta = [x for x in delta if x != delete]
    return delta

def delta_min(delta, gamma):
    """
    Generates a list of elements from delta that do not have a residual in gamma when subtracted from any element in delta_min.

    Parameters:
    delta (list): The list of elements to process.
    gamma (list): The list of residuals to check against.

    Returns:
    list: A new list containing elements from delta that meet the criteria.
    """
    delta_min = []
    for i in range(len(delta)):
        d = delta[i]
        if i == 0:
            delta_min.append(d)
        else:
            for j in range(len(delta_min)):
                residual = d - delta_min[j]
                if residual in gamma:
                    break
            if residual not in gamma:
                delta_min.append(d)
    return delta_min

def generate_delta(delta_0, delete, gamma):
    """
    Generates a filtered list from delta_0 by removing a specified element and then processes it to generate delta_min.

    Parameters:
    delta_0 (list): The initial list of elements.
    delete (any): The element to remove from the list.
    gamma (list): The list of residuals to check against.

    Returns:
    tuple: A tuple containing the delta_min list and the filtered delta_pre list.
    """
    delta_pre = [x for x in delta_0 if x != delete]
    return delta_min(delta_pre, gamma), delta_pre

def convert_dict_list(dict):
    """
    Converts a dictionary with tuple keys into a list of lists.

    Parameters:
    dict (dict): The dictionary to convert, where keys are tuples.

    Returns:
    list: A list of lists, where each inner list corresponds to a key in the dictionary.
    """
    D_temp = []
    for level, value in dict.items():
        D_temp.append(list(level))
    return D_temp

def all_level_delta(gamma, c):
    """
    Obtains all levels of delta as a list of lists.

    Parameters:
    gamma (list): The list of residuals to check against.
    c (int): The number of levels to generate.

    Returns:
    list: A list of lists, where each inner list represents a level of delta.
    """
    D_list = []
    delta_0 = generate_delta_(gamma, 0)
    D_1 = [delta_0]
    D_list.append(D_1)
    k = 0
    while k <= c:
        D = {}
        for delta in D_list[k]:
            for min_gen in delta_min(delta, gamma):
                D[tuple(generate_delta_(delta, min_gen))] = None
                D_ = convert_dict_list(D)
        D_list.append(D_)
        k += 1
    return D_list

def get_max_number_of_complement(delta, gamma):
    """
    Finds the maximum number in gamma that is not in delta.

    Parameters:
    delta (list): The list of elements to check against.
    gamma (list): The list of residuals to check.

    Returns:
    int: The maximum number in gamma that is not in delta.
    """
    return max(set(gamma) - set(delta))

def get_m_delta(delta, number):
    """
    Adds a number to delta and returns the new sorted list.

    Parameters:
    delta (list): The original list of elements.
    number (int): The number to add to the list.

    Returns:
    list: A new sorted list with the added number.
    """
    m_delta = delta.copy()
    m_delta.append(number)
    m_delta = sorted(m_delta)
    return m_delta

def calculate_syzygy_delta(delta, gamma):
    """
    Calculates the syzygy delta for a given delta and gamma.

    Parameters:
    delta (list): The list of elements to process.
    gamma (list): The list of residuals to check against.

    Returns:
    list: A list of elements that form the syzygy delta.
    """
    syz_delta = []
    delta_min_ = delta_min(delta, gamma)
    for i in range(len(delta)):
        d = delta[i]
        count = 0
        for j in range(len(delta_min_)):
            residual = d - delta_min_[j]
            if residual in gamma:
                count += 1
            if count == 2:
                break
        if count >= 2:
            syz_delta.append(d)
    return syz_delta

def calculate_cardinality_syzygy(delta, gamma, gamma_index, syz_delta):
    """
    Calculates the cardinality of the syzygy delta.

    Parameters:
    delta (list): The list of elements to process.
    gamma (list): The list of residuals to check against.
    gamma_index (int): The index of the gamma element to check.
    syz_delta (list): The syzygy delta list.

    Returns:
    int: The cardinality of the syzygy delta.
    """
    delta_min_gen, _ = generate_delta(delta, None, gamma)
    cardinality = 0
    for d in delta_min_gen:
        if d < gamma_index:
            cardinality += 1

    syz_delta_min_gen, _ = generate_delta(syz_delta, None, gamma)
    cardinality_syzygy = 0
    for syz_mg in syz_delta_min_gen:
        for d in delta_min_gen:
            if (gamma_index - syz_mg + d) in gamma:
                if syz_mg < gamma_index:
                    cardinality_syzygy += 1
                    break
    return cardinality - cardinality_syzygy

def mononial_poly(b):
    """
    Generates a monomial polynomial of degree b.

    Parameters:
    b (int): The degree of the monomial polynomial.

    Returns:
    list: A list representing the monomial polynomial.
    """
    l = []
    for _ in range(b):
        l.append(0)
    l.append(1)
    return l

def all_level_hilb_delta(D_list, gamma, c):
    """
    Obtains all levels of the Hilbert scheme associated with delta.

    Parameters:
    D_list (list): The list of delta levels.
    gamma (list): The list of residuals to check against.
    c (int): The number of levels to generate.

    Returns:
    list: A list of Polynomial objects representing the Hilbert scheme.
    """
    k = 0
    L_list = []
    while k <= c:
        p = Polynomial([0])
        hilb_dict = {}
        for delta in D_list[k]:
            degree = calculate_cardinality_syzygy(delta, gamma, get_max_number_of_complement(delta, gamma), calculate_syzygy_delta(delta, gamma))
            m_delta = get_m_delta(delta, get_max_number_of_complement(delta, gamma))
            if k >= 1:
                degree += hilb_dict_last[tuple(m_delta)]
            hilb_dict[tuple(delta)] = degree
            p += Polynomial(mononial_poly(degree))
        L_list.append(p)
        hilb_dict_last = hilb_dict
        k += 1
    return L_list

def compute_numerical_semigroup(generators, limit=None):
    """
    Computes the numerical semigroup generated by a set of generators.

    Parameters:
    generators (list): The list of generators.
    limit (int, optional): The upper limit for the range to compute.

    Returns:
    set: A set containing the elements of the numerical semigroup.
    """
    semigroup = set()
    semigroup.add(0)  # The semigroup must contain 0

    if limit is None:
        limit = max(generators) * max(generators)

    for i in range(limit + 1):
        for g in generators:
            if i - g >= 0 and (i - g) in semigroup:
                semigroup.add(i)
                break
    return semigroup

def compute_gaps_and_frobenius(generators):
    """
    Computes the gaps and Frobenius number for a numerical semigroup.

    Parameters:
    generators (list): The list of generators.

    Returns:
    tuple: A tuple containing the Frobenius number and the list of gaps.
    """
    limit = max(generators) * max(generators)
    semigroup = compute_numerical_semigroup(generators, limit)
    gaps = [n for n in range(limit) if n not in semigroup]
    frobenius_number = max(gaps) if gaps else None
    return frobenius_number, gaps

def get_frobenius_number(list_all):
    """
    Gets the Frobenius number for a given list of generators.

    Parameters:
    list_all (list): The list of generators.

    Returns:
    int: The Frobenius number.
    """
    frobenius_number, gaps = compute_gaps_and_frobenius(list_all)
    return frobenius_number

def get_c(list_all):
    """
    Gets the value of c for a given list of generators.

    Parameters:
    list_all (list): The list of generators.

    Returns:
    int: The value of c.
    """
    frobenius_number, gaps = compute_gaps_and_frobenius(list_all)
    c = frobenius_number + 1
    return c

def get_gamma(list_all):
    """
    Gets the gamma list for a given list of generators.

    Parameters:
    list_all (list): The list of generators.

    Returns:
    list: The gamma list.
    """
    _, gaps = compute_gaps_and_frobenius(list_all)
    c = get_c(list_all)
    L = [x for x in range(0, c + 1)]
    gamma = [x for x in L if x not in gaps]
    for i in range(c + 1, (list_all[0] - 1) * c):
        gamma.append(i)
    return gamma

def get_polynomial(list_all):
    """
    Gets the polynomial list for a given list of generators.

    Parameters:
    list_all (list): The list of generators.

    Returns:
    list: The list of Polynomial objects.
    """
    gamma = get_gamma(list_all)
    c = get_c(list_all)
    D_list = all_level_delta(gamma, c)
    list_poly = all_level_hilb_delta(D_list, gamma, c)
    return list_poly

def get_poly_coef(list_all):
    """
    Gets the polynomial coefficients for a given list of generators.

    Parameters:
    list_all (list): The list of generators.

    Returns:
    list: A list of lists containing the polynomial coefficients.
    """
    gamma = get_gamma(list_all)
    c = get_c(list_all)
    D_list = all_level_delta(gamma, c)
    
    list_poly_coef = []
    for i in range(c):
        list_poly_coef.append(all_level_hilb_delta(D_list, gamma, c)[i].coef.tolist())
    return list_poly_coef


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument(
        "--list_all",
        nargs="*",
        type=int,
        default=[4, 6, 13],
    )

    args = args.parse_args()

    save_list = get_polynomial(args.list_all)    
    c = get_c(args.list_all)
    gamma = get_gamma(args.list_all)
    list_poly =  get_polynomial(args.list_all)
    list_poly_coef = get_poly_coef(args.list_all)
    D_list =  all_level_delta(gamma, c)
    print(args.list_all)
    for i in range(c):
        print(f'{i+1}-th', list_poly[i])
