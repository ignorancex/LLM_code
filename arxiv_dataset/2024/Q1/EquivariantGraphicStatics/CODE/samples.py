import math
import numpy as np
from CODE.cell_complex import *
from CODE.group_actions import *

def simple_triangle():
    print("Simple triangle sample\nDihedral symmetry order 2")
    coords = np.array([[0,0], [1,1], [-1,1]])
    E = {0:[0,1], 1:[1,2], 2:[2,0]}
    F = {0: [0,1,2]}
    R_perm = {0:0, 1:1, 2:2}
    F_perm = {0:0, 1:2, 2:1}
    R_order = 1
    cell_complex = cell_complex_2(coords, E, F)
    action = dihedral_group_action(coords, R_perm, R_order, F_perm)
    return cell_complex, action

def simple_trivial():
    print("Simple triangle sample\nDihedral symmetry order 2")
    coords = np.array([[0,0], [1,1], [-1,1]])
    E = {0:[0,1], 1:[1,2], 2:[2,0]}
    F = {0: [0,1,2]}
    R_perm = {0:0, 1:1, 2:2}
    F_perm = {0:0, 1:2, 2:1}
    R_order = 1
    cell_complex = cell_complex_2(coords, E, F)
    action = trivial_group_action(coords)
    return cell_complex, action

def simple_square():
    print("Simple square sample\nDihedral symmetry order 4")
    coords = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
    E = {0:[0,1], 1:[1,2], 2:[2,3], 3:[3,0]}
    F = {0: [0,1,2,3]}
    R_perm = {0:3, 1:0, 2:1, 3:2}
    F_perm = {0:1, 1:0, 2:3, 3:2}
    R_order = 4
    cell_complex = cell_complex_2(coords, E, F)
    action = cyclic_group_action(coords, R_perm, R_order)
    return cell_complex, action
    
def boxed():
    print("Box wheel with spokes sample\nDihedral symmetry order 4")
    coords = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, 0]])
    E = {0:[0,1], 1:[2,1], 2:[2,3], 3:[3,0], 4:[0,4], 5:[1,4], 6:[2,4], 7:[3,4]}
    F = {0: [0, 4, 1], 1: [1,4,2], 2: [2,4,3], 3: [3,4,0], 4: [0,1,2,3]}
    R_perm = {0:3, 1:0, 2:1, 3:2, 4:4}
    F_perm = {0:1, 1:0, 2:3, 3:2, 4:4} #yflip
    R_order = 4
    cell_complex = cell_complex_2(coords, E, F)
    action = dihedral_group_action(coords, R_perm, R_order, F_perm)
    return cell_complex, action
    
def boxed_reciprocal():
    print("Box wheel reciprocal sample\nDihedral symmetry order 4")
    coords = np.array([[-1,0], [0,1], [1,0], [0,-1], [0,0]])
    E = {0:[0,1], 1:[1,2], 2:[3,2], 3:[3,0], 4:[0,4], 5:[1,4], 6:[2,4], 7:[3,4]}
    F = {0:[0, 4, 1], 1:[1,2,4], 2:[2,4,3], 3:[3,4,0], 4:[0,1,2,3]}
    R_perm = {0:3, 1:0, 2:1, 3:2, 4:4}
    F_perm = {0:0, 1:3, 2:2, 3:1, 4:4} #yflip
    R_order = 4
    cell_complex = cell_complex_2(coords, E, F)
    action = dihedral_group_action(coords, R_perm, R_order, F_perm)
    return cell_complex, action

def tower():
    print("Tower sample\nMirror symmetry (Dihedral symmetry order 2)")
    coords = np.array([[-2, -1], [2, -1], [0,0], [0, 2], [1, 3], [-1, 3]])
    E = {0:[0,1], 1:[1,2], 2:[2,0], 3:[2,3], 4:[3,4], 5:[4,5], 6:[5,3], 7:[0,5], 8:[1,4]}
    F = {0:[0,1,2], 1:[3,4,5], 2:[0,2,3,5], 3:[1,4,3,2], 4:[0,5,4,1]}
    R_perm = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
    F_perm = {0:1, 1:0, 2:2, 3:3, 4:5, 5:4}
    R_order = 1
    cell_complex = cell_complex_2(coords, E, F)
    action = dihedral_group_action(coords, R_perm, R_order, F_perm)
    return cell_complex, action

def klein_four():
    print("Klein four sample \nDihedral symmetry order 4")
    R_order = 2
    coords = np.array([[-3,0], [0,-2]])
    E = {}
    F = {}
    #Coordinates, Edges, and Faces
    for j in range(0, 3): #ycoord
        for i in range(0, 5): #xcoord
            coords = np.append(coords, np.array([[i-2, j-1]]), axis = 0)
            #Current node number is i+5*j+2
            if i > 0:
                E[len(E)] = [i+5*j+1, i+5*j+2]
            if j > 0:
                E[len(E)] = [i+5*j-3, i+5*j+2]
                if i > 0:
                    F[len(F)] = [i+5*j-4, i+5*j-3, i+5*j+2, i+5*j+1]
    coords = np.append(coords, np.array([[0,2], [3,0]]), axis = 0)
    
    for i in range(0, 5):
        E[len(E)] = [12+i, 17]
        E[len(E)] = [1, i+2]
        if i > 0:
            F[len(F)] = [12+i-1, 12+i, 17]
            F[len(F)] = [1, i+2, i+1]
    for i in range(0, 3):
        E[len(E)] = [0, 5*i+2]
        E[len(E)] = [5*i+6, 18]
        if i > 0:
            F[len(F)] = [0, 5*i-3, 5*i+2]
            F[len(F)] = [5*i+1, 18, 5*i+6]
    F[len(F)] = [0,2,1,6,18,16,17,12]
    #Permutations
    R_perm = {}
    F_perm = {0:0, 1:17, 17:1, 18:18}
    for i in range(0, coords.shape[0]):
        R_perm[len(R_perm)] = 18 - i
        if i > 1 and i <= 6:
            F_perm[i] = i+10
        elif i > 6 and i <= 11:
            F_perm[i] = i
        elif i > 11 and i <= 16:
            F_perm[i] = i-10
    cell_complex = cell_complex_2(coords, E, F)
    action = dihedral_group_action(coords, R_perm, R_order, F_perm)
    return cell_complex, action

def simple_wedge():
    print("Simple triangular wedge to complex sample \nCyclic symmetry order 3")
    polw = np.array([[1,0]])
    Ew = {0:[-1,0], 1:[0,1]}
    Fw = {0:[-1,0,1]}
    outsidew = [0]
    insidew = []
    Mw = {0:1}
    num_wedge = 3
    cell_complex, action = wedge_to_complex(num_wedge, polw, Ew, Fw, outsidew, insidew, Mw)
    return cell_complex, action

def fancy_hexagon():
    print("Fancy hexagon sample \nDihedral symmetry order 6")
    polw = np.array([[math.sqrt(3), 0.0], [.67, math.pi/6]])
    Ew = {0:[-1,0], 1:[-1,1], 2:[0,1], 3:[0,2], 4:[1,2]}
    Fw = {0:[-1,0,1], 1:[-1,1,2], 2:[0,2,1]}
    outsidew = [0]
    insidew = []
    Mw = {0:2, 1:1}
    num_wedge = 6
    return wedge_to_complex(num_wedge, polw, Ew, Fw, outsidew, insidew, Mw)

def four_star():
    print("Four star sample \nDihedral symmetry order 4")
    polw = np.array([[4.0, 0.0], [2.0, math.pi/6], [1.3, math.pi/4], [2.0, math.pi/3]])
    Ew = {0:[-1,0], 1:[-1,1], 2:[-1,2], 3:[-1,3], 4:[0,1], 5:[1,2], 6:[1,3], 7:[2,3], 8:[3,4]}
    Fw = {0:[-1,0,1], 1:[-1,1,2], 2:[-1,2,3], 3:[-1,3,4], 4:[1,3,2]}
    outsidew = [0,1,3]
    insidew = []
    Mw = {0:4, 1:3, 2:2, 3:1}
    num_wedge = 4
    return wedge_to_complex(num_wedge, polw, Ew, Fw, outsidew, insidew, Mw)

def triangular():
    print("Triangular sample \nDihedral symmetry order 3")
    polw = np.array([[math.sqrt(3), 0.0], [.5, math.pi/3]])
    Ew = {0:[1,3], 1:[-1,1], 2:[0,1], 3:[0,2], 4:[1,2]}
    Fw = {0:[-1,1,3], 1:[1,2,3], 2:[0,2,1]}
    outsidew = [0]
    insidew = []
    Mw = {0:2, 1:1}
    num_wedge = 3
    return wedge_to_complex(num_wedge, polw, Ew, Fw, outsidew, insidew, Mw)
    
def flower():
    print("Flower sample \nCyclic symmetry order 6")
    c = 1.35 #Radius exponent base
    r = 1.7 #Turning speed to radius ratio
    s = 0.38 #Turning speed
    N = 6
    num_wedge = 5
    
    polw = np.zeros((0, 2), dtype = float)
    Ew = {0:[0,N]}
    Fw = {0:[0,1,2,N]}
    for i in range(0, N):
        polw = np.append(polw, np.array([[c**(r*s*i), s*i]]), axis = 0 )
        if i >= 1:
            Ew[len(Ew)] = [i-1, i]
        if i >= 2:
            Ew[len(Ew)] = [i, N+i-2]
        if i >= 3:
            Ew[len(Ew)] = [i, N+i-3]
            Fw[len(Fw)] = [N+i-3, i-1, i]
            Fw[len(Fw)] = [N+i-3, i, N+i-2]
    outsidew = [N-3, N-2, N-1]
    insidew = [0]
    return wedge_to_complex(num_wedge, polw, Ew, Fw, outsidew, insidew)

def pinecone():
    c = 1.41 #Radius exponent base
    r = 1.7 #Turning speed to radius ratio
    s = 0.3 #Turning speed
    N = 6
    num_wedge = 7
    
    polw = np.zeros((0, 2), dtype = float)
    Ew = {0:[0,N], 1:[1,N]}
    Fw = {0:[0,1,N], 1:[1,2,N]}
    for i in range(0, N):
        polw = np.append(polw, np.array([[c**(r*s*i), s*i]]), axis = 0 )
        if i >= 1:
            Ew[len(Ew)] = [i-1, i]
        if i >= 2:
            Ew[len(Ew)] = [i, N+i-2]
        if i >= 3:
            Fw[len(Fw)] = [N+i-3, i-1, i,N+i-2]
    Ew[len(Ew)] = [N-1, 2*N-2]
    Fw[len(Fw)] = [N-1, 2*N-2, 2*N-3]
    outsidew = [N-2, N-1]
    insidew = [0]
    return wedge_to_complex(num_wedge, polw, Ew, Fw, outsidew, insidew)