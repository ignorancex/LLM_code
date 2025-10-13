from dolfin import *
import meshio
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Make sure model parameters can be decided when running:
parser = argparse.ArgumentParser(description="run the thing")
parser.add_argument("-T", type=int, required=True)
parser.add_argument("-dt", type=float, required=True)
parser.add_argument("-E", type=float, required=True)
parser.add_argument("-D1", type=int, required=True)
parser.add_argument("-mesh_name", type=str, required=True)
parser.add_argument("-twoDstim", type=str, required=True)

args = parser.parse_args()
print(args)

# define parameters from given above
T = args.T
dt = args.dt
E = args.E
mesh_name = args.mesh_name
twoDstim = args.twoDstim
D1 = args.D1

# Model parameters
cf = 1
cr = 1
C = 3.25
gamma = 77.56
D2 = D1
D4 = 3

# Create meshes from gmsh files
mesh = Mesh() #define 
meshnr = 1

# write xdmf file from msh file (only need to do this once)
msh = meshio.read('../meshes/'+mesh_name+'.msh') #read msh file
meshio.write('../meshes/'+mesh_name+'.xdmf', meshio.Mesh(points=msh.points, cells={"tetra": msh.cells_dict["tetra"]})) #convert to xdmf file

with XDMFFile('../meshes/'+mesh_name+'.xdmf') as xdmf: 
    xdmf.read(mesh) #write xdmf file to mesh
File('../meshes/'+mesh_name+'.pvd').write(mesh) #save mesh to new pvd file

#find surface mesh
surface = BoundaryMesh(mesh, 'exterior') 

# Defining relationship between vertices of the two meshes
vs_2_vb = surface.entity_map(0)
#vb_2_v.vector()[j] is the number of vertex on
#domain_mesh, where j is the corresponding vertex on surface mesh

# Build function space with Lagrange multiplier
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
S1 = FiniteElement("Lagrange", surface.ufl_cell(), 1)
W = FunctionSpace(mesh, P1)
S = FunctionSpace(surface, S1)

#compute volume and surface area for other constants
vol1 = Constant('1')
vol = interpolate(vol1, W)
vl = assemble(vol*dx)

area1 = Constant('1')
area = interpolate(area1, S)
ar = assemble(area*dx)

#define nr
nr = vl/ar

if twoDstim == '2D':
    #make bigE a function for 2D stimulus
    # boundary conditions for linear elasticity:
    def Boundary(x, on_boundary):
        return on_boundary and near(x[2],0) #(-0.99 > x[0] > -1. or 1. > x[0] > 0.99) #((abs(x[0] - 0.9) < DOLFIN_EPS) or (abs(x[0] + 0.9) < DOLFIN_EPS)) #near(x[0],1) #(x[0] < -0.9 or x[0] > 0.9)  #near(x[0], 0)

    bcs = DirichletBC(W, Constant(E), Boundary)
    E2D = Function(W)
    bcs.apply(E2D.vector())

    k3 = nr*3.79*E2D/(C+E2D)

    #create subdomain for 2D stimulus
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2],0)
    bottom = Bottom()

    # Create mesh functions over the cell facets
    sub_domains = MeshFunction("size_t", surface, surface.topology().dim())

    # Mark all facets as sub domain 0 except for the bottom, which is marked as 1
    sub_domains.set_all(0)
    bottom.mark(sub_domains, 1)

    dsx = Measure('dx', domain=surface, subdomain_data=sub_domains, metadata={'quadrature_degree': 3})

    # Save sub domains to file
    #file = File("../results/subdomains.pvd")
    #file << sub_domains
    
elif twoDstim == '2xD':
    #make bigE a function for 2xD stimulus
    # boundary conditions for linear elasticity:
    def Boundary(x, on_boundary):
        return on_boundary and near(x[2],0) #(-0.99 > x[0] > -1. or 1. > x[0] > 0.99) #((abs(x[0] - 0.9) < DOLFIN_EPS) or (abs(x[0] + 0.9) < DOLFIN_EPS)) #near(x[0],1) #(x[0] < -0.9 or x[0] > 0.9)  #near(x[0], 0)

    bcs = DirichletBC(W, Constant(E), Boundary)
    E2D = Function(W)
    bcs.apply(E2D.vector())

    k3 = nr*3.79*E2D/(C+E2D)

else:
    k3 = nr*3.79*E/(C+E)

#define rest of constants
k1 = 0.35
k2 = nr*0.15   
k4 = 6.25
k5 = nr*0.168

# otherwise get the warning for too many integration points
q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})
ds = ds(metadata={'quadrature_degree': q_degree})

# define relationship between vertex and degrees of freedom of each finite element space
v2d_W = vertex_to_dof_map(W)
v2d_S = vertex_to_dof_map(S)

# Define trial and test functions
cd = TrialFunction(W)
vd = TestFunction(W)
ca = TrialFunction(W)
va = TestFunction(W)
pa = TrialFunction(S)
wa = TestFunction(S)

#define functions for p in bulk and u on surface
surf_ca = Function(S)

#initial conditions
#u0  = Expression("pow(x[0],2)", degree=3)
cd0 = Constant('0.7')
cd1 = interpolate(cd0, W)
ca0 = Constant('0.3')
ca1 = interpolate(ca0, W)

#p0  = Expression("pow(x[0],2)", degree=3)
pd0 = 1
pa0 = Constant('0.06')
pa1 = interpolate(pa0, S)

# Weak statement of the lhs
a_cd = cd*vd*dx + dt*D1*inner(grad(cd), grad(vd))*dx + dt*(k2+k3)*cd*vd*ds
a_ca = ca*va*dx + dt*D2*inner(grad(ca), grad(va))*dx + dt*k1*ca*va*dx

A_cd = assemble(a_cd)
A_ca = assemble(a_ca)

# Octput files that eventually create Figure 21 and 23
file_cd = File("../results/simulations/rhomodel"+twoDstim+"_cd_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E.pvd", "compressed")
file_ca = File("../results/simulations/rhomodel"+twoDstim+"_ca_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E.pvd", "compressed")
file_pa = File("../results/simulations/rhomodel"+twoDstim+"_p_reduced_dt="+str(dt)+"_T="+str(T)+"_D="+str(D1)+"_"+str(E)+"E.pvd", "compressed")

# initialize functions
cd_new = Function(W)
ca_new = Function(W)
pa_new = Function(S)

# define constants for time loop
N = int(np.floor(T/dt))
print(N, dt)
temp_der_stats = np.zeros((4,N)) #to store l2 norms in each timestep
temp_stats = np.zeros((4,N)) #to store l2 norms in each timestep
temp_stats_minmax = np.zeros((6,N))

t = dt
for n in range(0,N):

    # Weak statement of the rhs of c_i 
    L_cd = cd1*vd*dx + dt*cf*k1*ca1*vd*dx
    b_cd = assemble(L_cd)
    
    #now solve the systems
    begin('Solving ligand concentration')
    solve(A_cd, cd_new.vector(), b_cd)
    end()

    # Weak statement of the rhs of c_i 
    L_ca = ca1*va*dx + dt*(k2+k3)/cf*cd_new*va*ds
    b_ca = assemble(L_ca)
    
    #now solve the systems
    begin('Solving ligand concentration')
    solve(A_ca, ca_new.vector(), b_ca)
    end()

    # now get u1 from function in bulk to function on surface
    for i in range(surface.num_vertices()):
        j = vs_2_vb.array()[i]
        surf_ca.vector()[v2d_S[i]] = ca_new.vector()[v2d_W[j]]

    # weak statement of rhs of pa
    if twoDstim == '2D':
        a_pa = pa*wa*dsx(0) + dt*D4*inner(grad(pa), grad(wa))*dsx(0) + dt*k4*pa*wa*dsx(1) + dt*k5/cr*(gamma*surf_ca*surf_ca*surf_ca*surf_ca*surf_ca+1)/nr*pa*wa*dsx(1)
        A_pa = assemble(a_pa)
        
        # Weak statement of the rhs of p
        L_pa = pa1*wa*dsx(0) + dt*k5/cr*(gamma*surf_ca*surf_ca*surf_ca*surf_ca*surf_ca+1)*(pd0+pa0/nr)*wa*dsx(1)
        b_pa = assemble(L_pa)
    else:
        a_pa = pa*wa*dx + dt*D4*inner(grad(pa), grad(wa))*dx + dt*k4*pa*wa*dx + dt*k5/cr*(gamma*surf_ca*surf_ca*surf_ca*surf_ca*surf_ca+1)/nr*pa*wa*dx
        A_pa = assemble(a_pa)
        
        # Weak statement of the rhs of p
        L_pa = pa1*wa*dx + dt*k5/cr*(gamma*surf_ca*surf_ca*surf_ca*surf_ca*surf_ca+1)*(pd0+pa0/nr)*wa*dx
        b_pa = assemble(L_pa)

    # now solve for p
    begin('Solving co-receptor concentration')
    solve(A_pa, pa_new.vector(), b_pa)
    end()

    # compute l2 norms of discrete time derivatives
    ca_der = np.sqrt(assemble(((ca_new-ca1)/dt)**2*dx(mesh))/vl)
    cd_der = np.sqrt(assemble(((cd_new-cd1)/dt)**2*dx(mesh))/vl)
    pa_der = np.sqrt(assemble(((pa_new-pa1)/dt)**2*dx(surface))/ar)
    # another way to compute this norm: u_err = errornorm(ue, u, norm_type='L2', degree_rise=3)

    ca_temp = np.sqrt(assemble((ca1)**2*dx(mesh))/vl)
    cd_temp = np.sqrt(assemble((cd1)**2*dx(mesh))/vl)
    pa_temp = np.sqrt(assemble((pa1)**2*dx(surface))/ar)

    ca_min = np.min(ca1.compute_vertex_values())
    ca_max = np.max(ca1.compute_vertex_values())
    cd_min = np.min(cd1.compute_vertex_values())
    cd_max = np.max(cd1.compute_vertex_values())
    pa_min = np.min(pa1.compute_vertex_values())
    pa_max = np.max(pa1.compute_vertex_values())

    #store L2 norm and min/max at each time step
    temp_der_stats[:,n] = np.array([ca_der, cd_der, pa_der, t]) 
    temp_stats[:,n] = np.array([ca_temp, cd_temp, pa_temp, t])
    temp_stats_minmax[:,n] = np.array([ca_min, ca_max, cd_min, cd_max, pa_min, pa_max])

    # write all solutions to predetermined files
    file_cd.write(cd1,t)
    file_ca.write(ca1,t)
    file_pa.write(pa1,t)

    # renew u1 and next timestep
    cd1.assign(cd_new)
    ca1.assign(ca_new)
    pa1.assign(pa_new)
    t += dt

# save numpy arrays with L2 norm and min/max at each timestep
np.save('../results/temp/tempstats0_'+twoDstim+'-der-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E', temp_der_stats)
np.save('../results/temp/tempstats0_'+twoDstim+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E', temp_stats)
np.save('../results/temp/tempstats0_'+twoDstim+'-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E', temp_stats_minmax)
        
# plot the temporal statistics
fig, axs = plt.subplots(2, 2, figsize=(12,12))
axs = axs.flatten()
axs[0].plot(temp_der_stats[3,:], temp_der_stats[0,:])
axs[1].plot(temp_der_stats[3,:], temp_der_stats[1,:])
axs[2].plot(temp_der_stats[3,:], temp_der_stats[2,:])
for i in range(3):
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('L2-norm')
    axs[i].grid(alpha=0.3, which='both', linestyle=':')
axs[0].set_title('$||\partial_t c_a||_{L^2(Y)}$')
axs[1].set_title('$||\partial_t c_d||_{L^2(Y)}$')
axs[2].set_title('$||\partial_t p_a||_{L^2(\Gamma)}$')
plt.savefig('../results/figures/tempstats0_'+twoDstim+'-der-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.png')
#plt.show()
plt.close()

fig, axs = plt.subplots(2, 2, figsize=(12,12))
axs = axs.flatten()
axs[0].plot(temp_stats[3,:], temp_stats[0,:])
axs[1].plot(temp_stats[3,:], temp_stats[1,:])
axs[2].plot(temp_stats[3,:], temp_stats[2,:])
for i in range(3):
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('L2-norm')
    axs[i].grid(alpha=0.3, which='both', linestyle=':')
axs[0].set_title('$||c_a||_{L^2(Y)}$')
axs[1].set_title('$||c_d||_{L^2(Y)}$')
axs[2].set_title('$||p_a||_{L^2(\Gamma)}$')
plt.savefig('../results/figures/tempstats0_'+twoDstim+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_D='+str(D1)+'_'+str(E)+'E.png')
#plt.show()
plt.close()
