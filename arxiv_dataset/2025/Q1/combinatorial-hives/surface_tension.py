import argparse
import numpy as np
from scipy.stats import unitary_group
from tqdm import tqdm
import plotly.graph_objects as go
import cvxpy as cp
import math

# Compute triangle area given the three vertices
def triangle_area(A, B, C):
    return 0.5 * np.abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))

# Compute barycentric coordinates of a point P with respect to the triangle ABC
def barycentric_coordinates(A, B, C, P):
    # Calculate the area of the triangle ABC
    area_ABC = triangle_area(A, B, C)
    
    # Calculate the areas of the triangles PBC, PCA, PAB
    area_PBC = triangle_area(P, B, C)
    area_PCA = triangle_area(P, C, A)
    area_PAB = triangle_area(P, A, B)
    
    # Calculate the barycentric coordinates
    alpha = area_PBC / area_ABC
    beta = area_PCA / area_ABC
    gamma = area_PAB / area_ABC
    
    return alpha, beta, gamma

# Sample random barycentric coordinates with margin conditions
def sample_barycentric_coordinates(m):
    assert 0 <= m < 1/3, "m must be in the range [0, 1/3)"
    while True:
        # Sample random barycentric coordinates
        u, v = np.random.random(), np.random.random()
        # Ensure the point lies within the triangle
        if u + v > 1:
            u, v = 1 - u, 1 - v
        # Calculate barycentric coordinates
        a = u
        b = v
        c = 1 - u - v
        # Check if the coordinates satisfy the condition m < a, b, c < 1 - m
        if m < a < 1 - m and m < b < 1 - m and m < c < 1 - m:
            return a, b, c

# Compute the index of a vertex (i,j) inside a hive of size n
def vectorize(i,j,n):
    return int(i*(2*n + 1 - i)/2 + j)

# Compute the vertex (i,j) from the index id inside a hive of size n
def devectorize(id,n):
    i = n - math.ceil((-1 + math.sqrt(1 + 8*(n*(n+1)/2 - id))) / 2)
    j = int(id - i*(2*n + 1 - i)/2)
    return i,j

# Compute the log vandermonde determinant of a sequence seq
def compute_log_Vandermonde(seq):
    log_Vandermonde = 0
    for i in range(len(seq)):
        for j in range(i):
            log_Vandermonde += np.log(seq[j]-seq[i])
    return log_Vandermonde

if __name__ == '__main__':

    # Create argparse for the parameters
    parser = argparse.ArgumentParser(description='Numerical estimation of the surface tension of GUE hives')
    parser.add_argument('--N', type=int, default=250, help='Size of the hive')
    parser.add_argument('--N_hess', type=int, default=100, help='Size of discretization of the Hessian domain')
    parser.add_argument('--N_hess_samp', type=int, default=10000, help='Number of hives sampled on the Hessian domain')
    parser.add_argument('--margin', type=float, default=0.02, help='Margin for sampling from triangle interior')
    args = parser.parse_args()

    N = args.N
    N_hess = args.N_hess
    N_hess_samp = args.N_hess_samp
    margin = args.margin

    # Load the (1,0,1) GUE hive
    S_oct_a = np.load('{}_mean_S_oct_{:d}_sig_X_{:.3f}_sig_Y_{:.3f}.npy'.format('GUE',1000,1,0))

    # Rotate to get a (0,1,1) GUE hive
    S_oct_b = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            S_oct_b[i,j] = S_oct_a[N-i-j,i]

    # Rotate to get a (1,1,0) GUE hive
    S_oct_c = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            S_oct_c[i,j] = S_oct_a[j,N-i-j]

    # Set other panel to nan
    for i in range(N+1):
        for j in range(N+1):
            if i+j > N:
                S_oct_a[i,j] = np.nan
                S_oct_b[i,j] = np.nan
                S_oct_c[i,j] = np.nan

    # Define the vertices of the Hessian triangle S_0 + S_1 + S_2 = 1
    A = np.array([1,0,0])
    B = np.array([0,1,0])
    C = np.array([0,0,1])

    # Define the vertices of the Hessian triangular grid
    A_h = np.array([0,0])
    B_h = np.array([1,0])
    C_h = np.array([0,1])

    # Initialize the lists to store the weights, values and offsets
    Weights = []
    Values = []
    Offsets = []

    # Compute the constant term in the objective function
    # gamma = np.array([S_oct_a[0,j-1] - S_oct_a[0,j] for j in range(N,0,-1)])
    # tau = np.arange(N,0,-1) 
    # log_Van_gamma = compute_log_Vandermonde(gamma)
    # log_Van_tau = compute_log_Vandermonde(tau)
    # C_n = np.log(np.sqrt(N)) + (log_Van_gamma - log_Van_tau)/((N-1)*(N-2)/2)
    C_n = 5/4

    for _ in tqdm(range(N_hess_samp)):
        # Generate the hive
        SS_0, SS_1, SS_2 = sample_barycentric_coordinates(margin)
        S_oct = SS_0 * S_oct_a + SS_1 * S_oct_b + SS_2 * S_oct_c
        sig_X = SS_0 + SS_1 
        sig_Y = SS_0 + SS_2
        sig_Z = SS_1 + SS_2

        # Initialize the offset and the Hess dictionary to zero
        offset = 0
        Hess = {}
        for i in range(0,N_hess+1):
            for j in range(0,N_hess-i+1):
                Hess[(i,j)] = 0

        for i in range(1,N):
            for j in range(1,N-i):
                # Compute the vertices of the hexagon containing the point V_0
                V_0 = S_oct[i,j]
                V_1 = S_oct[i,j-1]
                V_2 = S_oct[i-1,j]
                V_3 = S_oct[i-1,j+1]
                V_4 = S_oct[i,j+1]
                V_5 = S_oct[i+1,j]
                V_6 = S_oct[i+1,j-1]
                # Compute the slacks and the point P on the Hessian triangle
                S_0 = (2*V_0 + V_3 + V_6 - V_1 - V_2 - V_4 - V_5)/2
                S_1 = (2*V_0 + V_2 + V_5 - V_1 - V_3 - V_4 - V_6)/2
                S_2 = (2*V_0 + V_1 + V_4 - V_2 - V_3 - V_5 - V_6)/2
                S_sum = S_0 + S_1 + S_2
                P = np.array([S_0/S_sum,S_1/S_sum,S_2/S_sum])
                # Update the offset
                offset += np.log(S_sum)
                # Get the vertices of the small triangle inside the Hessian grid containing the point P
                u = (C - A)
                v = (B - A)
                u_mag = np.linalg.norm(u)
                v_mag = np.linalg.norm(v)
                u_cap = u / u_mag
                v_cap = v / v_mag
                q = P - A
                q_dot_u = np.dot(q,u_cap)
                q_dot_v = np.dot(q,v_cap)
                u_dot_v = np.dot(u_cap,v_cap)
                alp = (q_dot_u - q_dot_v * u_dot_v)/(1 - u_dot_v**2)
                bet = (q_dot_v - q_dot_u * u_dot_v)/(1 - u_dot_v**2)
                alpha = N_hess * alp / u_mag
                beta = N_hess * bet / v_mag
                floor_alpha = np.floor(alpha)
                floor_beta = np.floor(beta)
                del_u = alpha - floor_alpha
                del_v = beta - floor_beta
                T_0 = np.array([alpha, beta])     
                T_2 = np.array([floor_alpha + 1, floor_beta])
                T_3 = np.array([floor_alpha, floor_beta + 1])
                if del_u + del_v <= 1:
                    T_1 = np.array([floor_alpha, floor_beta])
                else:
                    T_1 = np.array([floor_alpha + 1, floor_beta + 1])
                # Compute the barycentric coordinates of the point T_0 inside the triangle T_1 T_2 T_3
                bar_1, bar_2, bar_3 = barycentric_coordinates(T_1, T_2, T_3, T_0)
                # Update the Hess dictionary with the barycentric coordinates
                T_1 = np.round(T_1).astype(int)
                T_2 = np.round(T_2).astype(int)
                T_3 = np.round(T_3).astype(int)
                Hess[(T_1[0],T_1[1])] += bar_1
                Hess[(T_2[0],T_2[1])] += bar_2
                Hess[(T_3[0],T_3[1])] += bar_3

        # Update the Weights
        W_temp = []  
        for i in range(0,N_hess+1):
            for j in range(0,N_hess-i+1):
                W_temp.append(Hess[(i,j)]/((N-1)*(N-2)/2))
        Weights.append(W_temp)
        # Update the Values
        semi_per = (sig_X + sig_Y + sig_Z) / 2
        area_triangle = np.sqrt(semi_per * (semi_per - sig_X) * (semi_per - sig_Y) * (semi_per - sig_Z))
        Values.append(np.log((4*area_triangle**2)/(sig_X*sig_Y*sig_Z)))
        # Update the Offsets
        Offsets.append(offset/((N-1)*(N-2)/2))

    # Initialize the optimization variables
    x = cp.Variable((N_hess+2)*(N_hess+1)//2)
    # Initialize the constraints
    constraints = []

    # Convexity Constraints in the interior
    for i in range(1,N_hess):
        for j in range(1,N_hess-i):
            constraints.append(x[vectorize(i,j+1,N_hess+1)]+x[vectorize(i,j-1,N_hess+1)]>=2*x[vectorize(i,j,N_hess+1)])
            constraints.append(x[vectorize(i+1,j,N_hess+1)]+x[vectorize(i-1,j,N_hess+1)]>=2*x[vectorize(i,j,N_hess+1)])
            constraints.append(x[vectorize(i+1,j-1,N_hess+1)]+x[vectorize(i-1,j+1,N_hess+1)]>=2*x[vectorize(i,j,N_hess+1)])
    # Convexity Constraints on the boundary (x-axis)
    i = 0
    for j in range(1,N_hess):
        constraints.append(x[vectorize(i,j+1,N_hess+1)]+x[vectorize(i,j-1,N_hess+1)]>=2*x[vectorize(i,j,N_hess+1)])
    # Convexity Constraints on the boundary (y-axis)
    j = 0
    for i in range(1,N_hess):
        constraints.append(x[vectorize(i+1,j,N_hess+1)]+x[vectorize(i-1,j,N_hess+1)]>=2*x[vectorize(i,j,N_hess+1)])
    # Convexity Constraints on the boundary (antidiagonal)
    for i in range(1,N_hess):
        j = N_hess-i
        constraints.append(x[vectorize(i+1,j-1,N_hess+1)]+x[vectorize(i-1,j+1,N_hess+1)]>=2*x[vectorize(i,j,N_hess+1)])

    # Convert the lists to numpy arrays
    Weights = np.array(Weights)
    Values = np.array(Values)
    Offsets = np.array(Offsets)

    # Set up the convex optimization problem and solve it
    objective = cp.Minimize(cp.sum_squares(Weights @ x + Offsets + Values + C_n))
    prob = cp.Problem(objective, constraints)
    opt_val = prob.solve()
    print('Problem completed with status:', prob.status)
    print('Optimal value:', opt_val)
    opt_val_x = x.value

    # Save the surface tension in a numpy matrix, setting the other panel values to nan
    surface_tension = np.zeros((N_hess+1,N_hess+1))
    for i in range(N_hess+1):
        for j in range(N_hess+1):
            if i+j <= N_hess:
                surface_tension[i,j] = opt_val_x[vectorize(i,j,N_hess+1)]
            # elif i==0 or j==0 or i+j==N_hess:
            #     surface_tension[i,j] = np.nan
            else:
                surface_tension[i,j] = np.nan
    np.save("Surface_Tension_{}_{}_{}".format(N,N_hess,N_hess_samp),surface_tension)

    # Plot the surface tension
    fig = go.Figure(data=[go.Surface(z=surface_tension, x=np.arange(0,N_hess+1), y=np.arange(0,N_hess+1))])
    fig.update_layout(title='Surface Tension',
                        scene = dict(
                            xaxis_title='i',
                            yaxis_title='j',
                            zaxis_title='Surface Tension Value'),
                            autosize=False,
                            width=800, height=800,
                            margin=dict(l=65, r=50, b=65, t=90))
    # fig.show()
    fig.write_html("Surface_Tension_{}_{}_{}.html".format(N,N_hess,N_hess_samp))

    # Plot the exponential of the negation of the surface tension
    exp_surface_tension = np.exp(-surface_tension)
    fig = go.Figure(data=[go.Surface(z=exp_surface_tension, x=np.arange(0,N_hess+1), y=np.arange(0,N_hess+1))])
    fig.update_layout(title='Exponential of Negation of Surface Tension',
                        scene = dict(
                            xaxis_title='i',
                            yaxis_title='j',
                            zaxis_title='Surface Tension Value'),
                            autosize=False,
                            width=800, height=800,
                            margin=dict(l=65, r=50, b=65, t=90))
    fig.write_html("Exp_Surface_Tension_{}_{}_{}.html".format(N,N_hess,N_hess_samp))
