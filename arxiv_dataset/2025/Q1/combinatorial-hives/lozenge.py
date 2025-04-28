import argparse
import numpy as np
from scipy.stats import unitary_group
from tqdm import tqdm
import networkx as nx
import cvxpy as cp
import drawsvg as draw

def generate_GUE_matrix(size):
    # Real and imaginary parts of the off-diagonal elements
    real_part = np.random.normal(0, np.sqrt(1/2), (size, size))
    imag_part = np.random.normal(0, np.sqrt(1/2), (size, size))
    # Initialize the complex matrix
    matrix = real_part + 1j * imag_part
    # Adjust diagonal elements: real with variance 1
    for i in range(size):
        matrix[i, i] = np.random.normal(0, 1)
    # Make the matrix Hermitian
    gaussian_unitary_ensemble = (matrix + matrix.conj().T) / 2
    return gaussian_unitary_ensemble

def generate_Random_Projection(size):
    # Create diagonal matrix with half entries 1's and other half 0's
    D = np.diag(np.concatenate((np.ones(size//2),np.zeros(size-size//2))))
    # Create random unitary matrix
    Q = unitary_group.rvs(size)
    # Generate random projection matrix
    random_projection = Q @ D @ Q.conj().T
    return random_projection

def generate_Sequential_Spectrum_Matrix(size):
    # Create diagonal matrix with entries 0,1,2,...,size-1
    D = np.diag(np.arange(size))
    # Create random unitary matrix
    Q = unitary_group.rvs(size)
    # Generate sequential spectrum matrix
    sequential_spectrum_matrix = Q @ D @ Q.conj().T
    return sequential_spectrum_matrix

def generate_Alternating_Sequential_Spectrum_Matrix(size):
    # Create diagonal matrix with entries size,size-2,size-4,...,2,0,-2,-4,...,-(size-2)
    D = np.diag(np.arange(size,-size,-2))
    # Create random unitary matrix
    Q = unitary_group.rvs(size)
    # Generate sequential spectrum matrix
    alternating_sequential_spectrum_matrix = Q @ D @ Q.conj().T
    return alternating_sequential_spectrum_matrix

def octahedron_recurrence(X,Y,LA_diff=1000):
    # Generate the Gelfand-Tsetlin patterns
    GT_mu = {}
    GT_nu = {}
    GT_la = {}
    for i in range(N):
        Z = X[:N-i,:N-i]
        eigvals = np.linalg.eigvalsh(Z)[::-1]
        for j in range(N-i):
            GT_mu[(i,j)] = eigvals[j]
    for i in range(N):
        Z = Y[:N-i,:N-i]
        eigvals = np.linalg.eigvalsh(Z)[::-1]
        for j in range(N-i):
            GT_nu[(i,j)] = eigvals[j]
    for i in range(1):
        for j in range(N-i):
            GT_la[(i,j)] = LA_diff*(N-j)
    # Fill up the values in the double hive H using GT patterns
    H = {}
    H[(0,N)] = 0
    for i in range(1,N+1):
        H[(i,N)] = H[i-1,N] + GT_la[(0,i-1)]
    for i in range(N):
        for j in range(N-i):
            H[(i,N-j-1)] = H[(i,N-j)] + GT_mu[(i,j)]
    for j in range(N):
        for i in range(j,N):
            H[(i+1,j)] = H[(i,j)] + GT_nu[(j,i-j)]
    # Fill up the upper double hive on the tetrahedron T
    T = {}
    for key in H.keys():
        i = key[0]
        j = key[1]
        if i >= j:
            T[(0,i-j,j,N-i)] = H[key]
        else:
            T[(j-i,0,i,N-j)] = H[key]
    # Perform octahedron recurrence
    for i in range(N-1):
        for j in range(i+1):
            for k in range(N-i-1):
                x,y,z,w = (i-j,j,k,N-i-k-2)
                a = T[(x+1,y,z+1,w)]
                b = T[(x+1,y,z,w+1)]
                c = T[(x,y+1,z,w+1)]
                d = T[(x,y+1,z+1,w)]
                e = T[(x,y,z+1,w+1)]
                f = max(a + c, b + d) - e
                T[(x+1,y+1,z,w)] = f
    # Obtain the double hive on the bottom of the tetrahedron
    H_oct = {}
    for i in range(N+1):
        for j in range(N+1):
            if i+j <= N:
                H_oct[(i,j)] = T[(j,i,0,N-i-j)]
            else:
                H_oct[(i,j)] = T[(N-i,N-j,i+j-N,0)]
    # Obtain height map of double hive (top (S) and bottom (S_oct))
    S = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            S[i,j] = H[(i,j)]
    S_oct = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            S_oct[i,j] = H_oct[(i,j)]
    # Visualize only one triangle (set other triangle values to NAN)
    # for i in range(N+1):
    #     for j in range(N+1):
    #         if i+j > N:
    #             S_oct[i,j] = np.nan
    return H, H_oct, T, S, S_oct

def occupancy_excavation_hexagon(verts_hex, vert_query):
    A,B,C,D,E,F = verts_hex
    V = vert_query
    # Check if the point V lies inside or on the boundary of the hexagon
    BA = np.cross(B-A,V-A) <= 0
    CB = np.cross(C-B,V-B) <= 0
    DC = np.cross(D-C,V-C) <= 0
    ED = np.cross(E-D,V-D) <= 0
    FE = np.cross(F-E,V-E) <= 0
    AF = np.cross(A-F,V-F) <= 0
    # Check if the point V lies inside the square lattice of the double hive
    # inside_square = V[0] >=0 and V[1] >= 0 and V[0] <= N and V[1] <= N
    return BA and CB and DC and ED and FE and AF #and inside_square

if __name__ == '__main__':

    # Create argparse for the parameters
    parser = argparse.ArgumentParser(description='Generate Lozenge Tilings using Speyers Perfect Matching')
    parser.add_argument('--matrix_type', choices=['GUE','RP','SSM','ASSM'],default='GUE',help='Type of matrix to use for minor process')
    parser.add_argument('--N', type=int, default=100, help='Size of the double hive')
    parser.add_argument('--V_i', type=int, default=50, help='x-coordinate of the base point V in the interior of the double hive for which the lozenge tiling is to be generated.')
    parser.add_argument('--V_j', type=int, default=50, help='y-coordinate of the base point V in the interior of the double hive for which the lozenge tiling is to be generated.')
    parser.add_argument('--sig_X', type=float, default=1, help='Scaling factor for X')
    parser.add_argument('--sig_Y', type=float, default=1, help='Scaling factor for Y')
    parser.add_argument('--LA_diff', type=float, default=1000, help='Difference factor in large lambda')
    args = parser.parse_args()

    matrix_type = args.matrix_type
    N = args.N
    sig_X = args.sig_X
    sig_Y = args.sig_Y
    LA_diff = args.LA_diff
    V_i = args.V_i
    V_j = args.V_j

    generate_matrix_dict = {'GUE':generate_GUE_matrix, 
                            'RP':generate_Random_Projection, 
                            'SSM':generate_Sequential_Spectrum_Matrix,
                            'ASSM':generate_Alternating_Sequential_Spectrum_Matrix}
    generate_matrix_func = generate_matrix_dict[matrix_type]

    # Generate the matrices X and Y
    print('Beginning Octahedron recurrence')
    X = sig_X * generate_matrix_func(N)
    Y = sig_Y * generate_matrix_func(N)
    H, H_oct, T, S, S_oct = octahedron_recurrence(X,Y,LA_diff)
    print('Octahedron recurrence completed')

    # if the vertex V lies in T
    if V_i + V_j <= N:
        A = [0,0]
        B = [0, V_j]
        C = [V_i, V_i + V_j]
        D = [V_i + V_j, V_i + V_j]
        E = [V_i + V_j, V_j]
        F = [V_i, 0]
    # if the vertex V lies in T'
    else:
        A = [V_i + V_j - N, V_i + V_j - N]
        B = [V_i + V_j - N, V_j]
        C = [V_i, N]
        D = [N, N]
        E = [N, V_j]
        F = [V_i, V_i + V_j - N]
    
    verts_hex = np.array([A,B,C,D,E,F])
    # vert_query = np.array([N//2,N//2])
    # A,B,C,D,E,F = verts_hex

    print('Creating the graph for optimization')

    G = nx.Graph()
    face_edges = {}
    face_value = {}
    face_count = 0

    first = np.array(A)
    last = np.array(D)

    for i in tqdm(range(N+1),leave=False):
        for j in tqdm(range(N+1),leave=False):
            vert_query = np.array([i,j])
            occ = occupancy_excavation_hexagon(verts_hex, vert_query)
            if occ:
                # Inside the excavation hexagon
                face_count+=1
                face_edges[(i,j)] = []
                if i-j==first[0]-first[1]:
                    # Equator
                    # Create red Nodes
                    a = vert_query + np.array([-1/2,-1/2])
                    b = vert_query + np.array([-1/3,1/3])
                    c = vert_query + np.array([1/2,1/2])
                    d = vert_query + np.array([1/3,-1/3])
                    # Check occupancy of red nodes
                    occ_a = occupancy_excavation_hexagon(verts_hex, a)
                    occ_b = occupancy_excavation_hexagon(verts_hex, b)
                    occ_c = occupancy_excavation_hexagon(verts_hex, c)
                    occ_d = occupancy_excavation_hexagon(verts_hex, d)
                    # Multiply by 3 to avoid floating point issues
                    a = 3*vert_query + np.array([-3/2,-3/2])
                    b = 3*vert_query + np.array([-1,1])
                    c = 3*vert_query + np.array([3/2,3/2])
                    d = 3*vert_query + np.array([1,-1])
                    # Add red nodes to the graph
                    if occ_a: G.add_node(tuple(a))
                    if occ_b: G.add_node(tuple(b))
                    if occ_c: G.add_node(tuple(c))
                    if occ_d: G.add_node(tuple(d))
                    # Add red edges to the graph
                    if occ_a and occ_b: 
                        G.add_edge(tuple(a),tuple(b))
                        face_edges[(i,j)].append((tuple(a),tuple(b)))
                    if occ_b and occ_c: 
                        G.add_edge(tuple(b),tuple(c))
                        face_edges[(i,j)].append((tuple(b),tuple(c)))
                    if occ_c and occ_d: 
                        G.add_edge(tuple(c),tuple(d))
                        face_edges[(i,j)].append((tuple(c),tuple(d)))
                    if occ_d and occ_a: 
                        G.add_edge(tuple(d),tuple(a))
                        face_edges[(i,j)].append((tuple(d),tuple(a)))
                    if (vert_query[0]==first[0] and vert_query[1]==first[1]) or (vert_query[0]==last[0] and vert_query[1]==last[1]):
                        face_value[(i,j)] = 0
                    else:
                        face_value[(i,j)] = 1    
                else:
                    # Not Equator
                    # Create red Nodes
                    a = vert_query + np.array([-1/3,-2/3])
                    b = vert_query + np.array([-2/3,-1/3])
                    c = vert_query + np.array([-1/3,1/3])
                    d = vert_query + np.array([1/3,2/3])
                    e = vert_query + np.array([2/3,1/3])
                    f = vert_query + np.array([1/3,-1/3])
                    # Make special adjustments for the special cases
                    if i-j==first[0]-first[1]+1: c = vert_query + np.array([-1/2,1/2])
                    if i-j==first[0]-first[1]-1: f = vert_query + np.array([1/2,-1/2])
                    # Check occupancy of red nodes
                    occ_a = occupancy_excavation_hexagon(verts_hex, a)
                    occ_b = occupancy_excavation_hexagon(verts_hex, b)
                    occ_c = occupancy_excavation_hexagon(verts_hex, c)
                    occ_d = occupancy_excavation_hexagon(verts_hex, d)
                    occ_e = occupancy_excavation_hexagon(verts_hex, e)
                    occ_f = occupancy_excavation_hexagon(verts_hex, f)
                    # Multiply by 3 to avoid floating point issues
                    a = 3*vert_query + np.array([-1,-2])
                    b = 3*vert_query + np.array([-2,-1])
                    c = 3*vert_query + np.array([-1,1])
                    d = 3*vert_query + np.array([1,2])
                    e = 3*vert_query + np.array([2,1])
                    f = 3*vert_query + np.array([1,-1])
                    # Make special adjustments for the special cases
                    if i-j==first[0]-first[1]+1: c = 3*vert_query + np.array([-3/2,3/2])
                    if i-j==first[0]-first[1]-1: f = 3*vert_query + np.array([3/2,-3/2])
                    # Add red nodes to the graph
                    if occ_a: G.add_node(tuple(a))
                    if occ_b: G.add_node(tuple(b))
                    if occ_c: G.add_node(tuple(c))
                    if occ_d: G.add_node(tuple(d))
                    if occ_e: G.add_node(tuple(e))
                    if occ_f: G.add_node(tuple(f))
                    # Add red edges to the graph
                    if occ_a and occ_b: 
                        G.add_edge(tuple(a),tuple(b))
                        face_edges[(i,j)].append((tuple(a),tuple(b)))
                    if occ_b and occ_c: 
                        G.add_edge(tuple(b),tuple(c))
                        face_edges[(i,j)].append((tuple(b),tuple(c)))
                    if occ_c and occ_d: 
                        G.add_edge(tuple(c),tuple(d))
                        face_edges[(i,j)].append((tuple(c),tuple(d)))
                    if occ_d and occ_e: 
                        G.add_edge(tuple(d),tuple(e))
                        face_edges[(i,j)].append((tuple(d),tuple(e)))
                    if occ_e and occ_f: 
                        G.add_edge(tuple(e),tuple(f))
                        face_edges[(i,j)].append((tuple(e),tuple(f)))
                    if occ_f and occ_a: 
                        G.add_edge(tuple(f),tuple(a))
                        face_edges[(i,j)].append((tuple(f),tuple(a)))
                    count = int(occ_a) + int(occ_b) + int(occ_c) + int(occ_d) + int(occ_e) + int(occ_f)
                    if count == 6:
                        face_value[(i,j)] = 2
                    else:
                        face_value[(i,j)] = 1

    edge_index = {}
    index_edge = {}
    edge_count = 0
    for k,edge in enumerate(G.edges()):
        edge_index[edge] = k
        edge_index[edge[::-1]] = k
        index_edge[k] = edge
        edge_count += 1

    # Create the Face to Edge adjacency matrix
    hive_values = np.zeros(face_count)
    face_values = np.zeros(face_count)
    face_edge_adj = np.zeros((face_count,edge_count))
    for k,(i,j) in enumerate(face_value.keys()):
        hive_values[k] = H[(i,j)]
        face_values[k] = face_value[(i,j)]
        for edge in face_edges[(i,j)]:
            face_edge_adj[k,edge_index[edge]] = 1
        
    print('Graph created successfully')

    print('Beginning optimization for perfect matching on the graph')

    # Create the edge variables for optimization
    x = cp.Variable(edge_count, boolean=True)
    # Create the objective function
    objective = cp.Maximize(hive_values.T @ (face_values - (face_edge_adj @ x)))
    constraints = [x>=0]
    for node in G.nodes():
        edge_list = list(G.edges(node))
        edge_indices = [edge_index[edge] for edge in edge_list]
        constraints.append(cp.sum(x[edge_indices]) == 1)
    prob = cp.Problem(objective, constraints)

    opt_val = prob.solve()
    print("Optimal value:", opt_val)
    print('Optimization problem completed with status:', prob.status)

    print('Creating the lozenge tiling')

    opt_var = x.value
    match_ids = np.arange(edge_count)[opt_var>0.5]
    match_edges = [index_edge[k] for k in match_ids]
    match_edges = np.array(match_edges) 
    match_edges = match_edges/3
    
    for k,edge in enumerate(match_edges):
        u,v = edge
        if u[0]-u[1] == first[0]-first[1]:
            if v[0]-v[1] > first[0]-first[1]:
                match_edges[k][0][0] += 1/6 
                match_edges[k][0][1] += -1/6 
            elif v[0]-v[1] < first[0]-first[1]:
                match_edges[k][0][0] += -1/6 
                match_edges[k][0][1] += 1/6
        elif v[0]-v[1] == first[0]-first[1]:
            if u[0]-u[1] > first[0]-first[1]:
                match_edges[k][1][0] += 1/6 
                match_edges[k][1][1] += -1/6 
            elif u[0]-u[1] < first[0]-first[1]:
                match_edges[k][1][0] += -1/6 
                match_edges[k][1][1] += 1/6

    SQRT_3 = np.sqrt(3)

    EPS = 1e-3
    ROT_90 = np.array([[0,1],[-1,0]])
    i_cap = np.array([1/2,SQRT_3/2])
    j_cap = np.array([1/2,-SQRT_3/2])

    blue_upper_1 = np.array([np.cos(np.pi/6),np.sin(np.pi/6)])
    blue_upper_2 = -1 * blue_upper_1
    red_upper_1 = np.array([np.cos(5*np.pi/6),np.sin(5*np.pi/6)])
    red_upper_2 = -1 * red_upper_1
    green_upper_1 = np.array([np.cos(np.pi/2),np.sin(np.pi/2)])
    green_upper_2 = -1 * green_upper_1

    red_lower_1 = np.array([np.cos(np.pi/6),np.sin(np.pi/6)])
    red_lower_2 = -1 * red_lower_1
    blue_lower_1 = np.array([np.cos(5*np.pi/6),np.sin(5*np.pi/6)])
    blue_lower_2 = -1 * blue_lower_1
    green_lower_1 = np.array([np.cos(np.pi/2),np.sin(np.pi/2)])
    green_lower_2 = -1 * green_lower_1

    drawing = draw.Drawing(N, N, origin=(0,-N/2))

    for k,edge in enumerate(match_edges):
        u,v = edge
        u = u[0]*i_cap + u[1]*j_cap
        v = v[0]*i_cap + v[1]*j_cap
        a = (u+v)/2 - (SQRT_3/2) * ROT_90 @ (u-v)
        b = 2*u - v
        c = (u+v)/2 + (SQRT_3/2) * ROT_90 @ (u-v)
        d = 2*v - u

        delta = (u-v)/np.linalg.norm(u-v)

        # Since y-axis is flipped in the drawing
        a[-1] = -a[-1]
        b[-1] = -b[-1]
        c[-1] = -c[-1]
        d[-1] = -d[-1]

        if u[1]>0 and v[1]>0:
            # Upper hive
            if np.linalg.norm(delta - blue_upper_1) < EPS or np.linalg.norm(delta - blue_upper_2) < EPS:
                drawing.append(draw.Lines(*a,*b,*c,*d,*a, close=True, fill='blue'))
            elif np.linalg.norm(delta - red_upper_1) < EPS or np.linalg.norm(delta - red_upper_2) < EPS:
                drawing.append(draw.Lines(*a,*b,*c,*d,*a, close=True, fill='red'))
            elif np.linalg.norm(delta - green_upper_1) < EPS or np.linalg.norm(delta - green_upper_2) < EPS:
                drawing.append(draw.Lines(*a,*b,*c,*d,*a, close=True, fill='green'))
        elif u[1]<0 and v[1]<0:
            # Lower Hive
            if np.linalg.norm(delta - blue_lower_1) < EPS or np.linalg.norm(delta - blue_lower_2) < EPS:
                drawing.append(draw.Lines(*a,*b,*c,*d,*a, close=True, fill='blue'))
            elif np.linalg.norm(delta - red_lower_1) < EPS or np.linalg.norm(delta - red_lower_2) < EPS:
                drawing.append(draw.Lines(*a,*b,*c,*d,*a, close=True, fill='red'))
            elif np.linalg.norm(delta - green_lower_1) < EPS or np.linalg.norm(delta - green_lower_2) < EPS:
                drawing.append(draw.Lines(*a,*b,*c,*d,*a, close=True, fill='green'))

    drawing.save_svg('lozenge_{}_{}_{}.svg'.format(N,V_i,V_j))

    print('Lozenge tiling created and saved successfully as lozenge_{}_{}_{}.svg'.format(N,V_i,V_j))
