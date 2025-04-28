import math
import numpy as np
from tabulate import tabulate


def print_homologies(force_cosheaf, constant_cosheaf, position_cosheaf):
    F = force_cosheaf
    J = constant_cosheaf
    P = position_cosheaf
    print("Homology space dimensions:")
    print("   Force  Constant  Position")
    print("H_2:", F.homology_dim(2), "------", J.homology_dim(2), "------", P.homology_dim(2))
    print("H_1:", F.homology_dim(1), "------", J.homology_dim(1), "------", P.homology_dim(1))
    print("H_0:", F.homology_dim(0), "------", J.homology_dim(0), "------", P.homology_dim(0))



def irreducible_pairs(force_cosheaf, constant_cosheaf, position_cosheaf):
    Fhomology_char = [force_cosheaf.homology_irred_char(i) for i in range(0, 3)]
    Jhomology_char = [constant_cosheaf.homology_irred_char(i) for i in range(0, 3)]
    Phomology_char = [position_cosheaf.homology_irred_char(i) for i in range(0, 3)]
    P_irred_basis = position_cosheaf.homology_irred_basis(2)
    char_dims = force_cosheaf.group_action.regular_char()

    for i in range(0, force_cosheaf.group_action.num_conjugacy_classes()):
        print("Irreducible number:", i, "Dimension:", char_dims[i])
        print("Dimensions of the irreducible cosheaf homology:")
        print("     F --R^2-- P")
        print("H_2:", int(Fhomology_char[2][i]), "--", int(Jhomology_char[2][i]), "--", int(Phomology_char[2][i]) )
        print("H_1:", int(Fhomology_char[1][i]), "--", int(Jhomology_char[1][i]), "--", int(Phomology_char[1][i]) )
        print("H_0:", int(Fhomology_char[0][i]), "--", int(Jhomology_char[0][i]), "--", int(Phomology_char[0][i]) )
        if bool(P_irred_basis):    #If there exists a reciprocal figure
            #Center the figure
            non_constant_pos = position_cosheaf.remove_constant_component(P_irred_basis[i])
            for j in range(0, non_constant_pos.shape[1]):
                posvec = non_constant_pos[:,j]
                position_cosheaf.plot_both(posvec)
        print("-------------------------------------------------------------------------------------------")



def irreducible_euler_char(cosheaf):
    homology_char = [cosheaf.homology_irred_char(i) for i in range(0, 3)]
    chain_char = [cosheaf.chain_irred_char(i) for i in range(0, 3)]
    char_dims = cosheaf.group_action.regular_char()

    for i in range(0, cosheaf.group_action.num_conjugacy_classes()):
        data= [None] * 5
        for dim in range(0, 3):
            data[dim]= ['char over C_' + str(dim) + ':', int(chain_char[dim][i]), '|',
                            'char over H_' + str(dim) + ':', int(homology_char[dim][i]) ]
        data[3] = ['', '--------', '', '--------']
        data[4] = ['alternating sum:', int(chain_char[2][i] - chain_char[1][i] + chain_char[0][i]), "|", 
                            '', int(homology_char[2][i] - homology_char[1][i] + homology_char[0][i]) ]

        data_header=['Chain Characters', '# copies of irriducible', '|', 
                        'Homology Characters', '# copies of irriducible']
        print("Irreducible number:", i, "Dimension:", char_dims[i])
        print()
        print(tabulate(data, headers=data_header, numalign="left" ))
        print('\n----------------------------------------------------------------------------------------\n')



def run_tests(force_cosheaf, constant_cosheaf, position_cosheaf):
    F = force_cosheaf
    J = constant_cosheaf
    P = position_cosheaf
    action = F.group_action

    print("Checking commutativity of cosheaf representation and boundary matrices\n")
    for generator_index in action.generator_indices():
        print("Group generator index:", generator_index)
        for dim in [1,2]:
            print("Force Cosheaf,    dim", dim, "error:", commutative_rep_bdd_test(F, dim, generator_index))
        for dim in [1,2]:
            print("Constant Cosheaf, dim", dim, "error:", commutative_rep_bdd_test(J, dim, generator_index))
        for dim in [1,2]:
            print("Position Cosheaf, dim", dim, "error:", commutative_rep_bdd_test(P, dim, generator_index))

    print("\n---------------------------------------\n")
    print("Check commutativity of cosheaf representation and cosheaf map matrices\n")
    for generator_index in action.generator_indices():
        print("Group generator index:", generator_index)
        for dim in [0,1,2]:
            print("Force Cosheaf,    dim", dim, "error:", commutative_rep_map_test(F, J, dim, generator_index))
        for dim in [0,1,2]:
            print("Constant Cosheaf, dim", dim, "error:", commutative_rep_map_test(J, P, dim, generator_index))

    print("\n---------------------------------------\n")
    print("Check commutativity of boundary and cosheaf map matrices\n")
    for dim in [1,2]:
        print("Force Cosheaf,    dim", dim, "error:", commutative_bdd_map_test(F, J, dim))
    for dim in [1,2]:
        print("Constant Cosheaf, dim", dim, "error:", commutative_bdd_map_test(J, P, dim))



def commutative_rep_bdd_test(Cosheaf, dim, index):
    #Check error in commutativity of representation matrices and boundary map of cosheaves
    A = np.dot(Cosheaf.representation(dim-1, index), Cosheaf.bdd(dim))
    B = np.dot(Cosheaf.bdd(dim), Cosheaf.representation(dim, index))
    c = np.absolute(A-B).round(3)
    if c.shape[0] > 0 and c.shape[1] > 0: #If the boundary matrix is non-trivial
        return c.max()
    else:
        return 0

def commutative_rep_map_test(Cosheaf_domain, Cosheaf_target, dim, index):
    #Check error in commutativity of representation matrices and cosheaf maps
    A = np.dot(Cosheaf_domain.to_next_cosheaf_map(dim), Cosheaf_domain.representation(dim, index))
    B = np.dot(Cosheaf_target.representation(dim, index), Cosheaf_domain.to_next_cosheaf_map(dim))
    c = np.absolute(A-B).round(3)
    if c.shape[0] > 0 and c.shape[1] > 0: #If the cosheaf map is non-trivial
        return c.max()
    else:
        return 0

def commutative_bdd_map_test(Cosheaf_domain, Cosheaf_target, dim):
    #Check error in commutativity of boundary maps and cosheaf maps
    A = np.dot(Cosheaf_domain.to_next_cosheaf_map(dim-1), Cosheaf_domain.bdd(dim))
    B = np.dot(Cosheaf_target.bdd(dim), Cosheaf_domain.to_next_cosheaf_map(dim))
    c = np.absolute(A-B).round(3)
    if c.shape[0] > 0 and c.shape[1] > 0: #If the cosheaf map and boundary map is non-trivial
        return c.max()
    else:
        return 0