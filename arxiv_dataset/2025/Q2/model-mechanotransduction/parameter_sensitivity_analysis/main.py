from dolfin import *
import meshio
import numpy as np
from ufl import nabla_div, Max

T = 10
dt = 0.05
bigE = 7000000.0
mesh_name = 'cell_substrate'
coupling = 4
twoDstim = 'no'
partfixed = 'no'

#parameters changing in parameter sensitivy analysis
C1 = 1
k6 = 1
nu_ = 0.3
k7 = 0.2
p = 2.6
k8 = 10**(1/2.6)

for C1 in [0.8*C1, 1.2*C1]: 
#for C1 in [0.9*C1, 1.1*C1]: 
#for nu_ in [0.8*nu_, 1.2*nu_]: 
#for nu_ in [0.9*nu_, 1.1*nu_]: 
#for k7 in [0.8*k7, 1.2*k7]: 
#for k7 in [0.9*k7, 1.1*k7]: 
#for k8 in [0.8*k8, 1.2*k8]: 
#for k8 in [0.9*k8, 1.1*k8]: 
#for p in [0.8*p, 1.2*p]: 
#for p in [0.9*p, 1.1*p]: 
#for k6 in [0.8*k6, 1.2*k6]: 
#for k6 in [0.9*k6, 1.1*k6]: 

    #define variables that always stay the same
    cf = 1
    cr = 1
    C = 3.25
    gamma = 77.56
    D1 = 40
    D2 = 40
    D4 = 3
    # elasticity variables
    di = 3
    #define rest of constants
    k1 = 0.35
    k4 = 6.25
    if coupling == 1 or coupling == 3:
        k9 = Constant(0)
    else:
        k9 = Constant(C1)
        #k9 = Constant(1)

    # Define strain and stress
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
        #return sym(nabla_grad(u))

    def sigma(u):
        return lambda_*nabla_div(u)*Identity(di) + 2*mu*epsilon(u)

    # Create meshes from gmsh files
    if mesh_name == 'cell_substrate':
        meshnr = 1
    else:
        meshnr = 0
    mesh = Mesh() #define mesh

    msh = meshio.read('../meshes/'+mesh_name+'.msh') #read msh file
    meshio.write('../meshes/'+mesh_name+'.xdmf', meshio.Mesh(points=msh.points, cells={"tetra": msh.cells_dict["tetra"]})) #convert to xdmf file

    with XDMFFile('../meshes/'+mesh_name+'.xdmf') as xdmf: 
        xdmf.read(mesh) #write xdmf file to mesh
    File('../meshes/'+mesh_name+'.pvd').write(mesh) #save mesh to new pvd 

    #for loop over each mesh refinement
    surface = BoundaryMesh(mesh, 'exterior') #find surface mesh

    # Defining relationship between vertices of the two meshes
    vs_2_vb = surface.entity_map(0)

    # Build function space with Lagrange multiplier
    P0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)    
    P1 = VectorElement("Lagrange", mesh.ufl_cell(), 1, dim=3)
    R1 = VectorElement("Real", mesh.ufl_cell(), 0, dim=3)
    R2 = FiniteElement("Real", mesh.ufl_cell(), 0)
    PRR = MixedElement([P1,R1,R2,R2,R2])
    E = FunctionSpace(mesh, PRR)
    W = FunctionSpace(mesh, P0)
    P = E.sub(0).collapse()
    Z1 = E.sub(1).collapse()
    Z2 = E.sub(2).collapse()
    Z3 = E.sub(3).collapse()
    Z4 = E.sub(4).collapse()

    # need to define function assigner, so you can get u from the larger function space
    fa = FunctionAssigner([P,Z1,Z2,Z3,Z4],E)

    # including the surface ones
    S1 = FiniteElement("Lagrange", surface.ufl_cell(), 1)
    ES = VectorFunctionSpace(surface, S1, 1)
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
    k2 = nr*0.15
    k5 = nr*0.168
    def Boundary(x, on_boundary):
        return on_boundary and near(x[2],0) #(-0.99 > x[0] > -1. or 1. > x[0] > 0.99) #((abs(x[0] - 0.9) < DOLFIN_EPS) or (abs(x[0] + 0.9) < DOLFIN_EPS)) #near(x[0],1) #(x[0] < -0.9 or x[0] > 0.9)  #near(x[0], 0)


    if twoDstim == 'yes':
        #make bigE a function for 2D stimulus
        bcs = DirichletBC(W, Constant(bigE), Boundary)
        E2D = Function(W)
        bcs.apply(E2D.vector())
        k3 = nr*3.79*E2D/(C+E2D)
    else:
        k3 = nr*3.79*bigE/(C+bigE)

    # otherwise get the warning for too many integration points
    q_degree = 3
    dx = dx(metadata={'quadrature_degree': q_degree})
    ds = ds(metadata={'quadrature_degree': q_degree})

    # define relationship between vertex and degrees of freedom of each finite element space
    v2d_W = vertex_to_dof_map(W)
    v2d_S = vertex_to_dof_map(S)

    # Define trial and test functions for chemical equations
    cd = TrialFunction(W)
    vd = TestFunction(W)
    ca = TrialFunction(W)
    va = TestFunction(W)
    pa = TrialFunction(S)
    wa = TestFunction(S)

    #define elasticity variational problem 
    # let's define the cross products
    xe1 = Expression(('0','x[2]','-x[1]'), degree=3)
    xe2 = Expression(('-x[2]','0','x[0]'), degree=3)
    xe3 = Expression(('x[1]','-x[0]','0'), degree=3)

    # Define variational problem
    (u, l1, l2, l3, l4) = TrialFunctions(E)
    (v, d1, d2, d3, d4) = TestFunctions(E)

    #define functions for p in bulk and u on surface
    surf_ca = Function(S)
    bulk_pa1 = Function(W)

    #initial conditions
    cd0 = Constant('0.7')
    cd1 = interpolate(cd0, W)
    ca0 = Constant('0.3')
    ca1 = interpolate(ca0, W)

    pd0 = 1
    pa0 = Constant('0.06')
    pa1 = interpolate(pa0, S)

    u_bc = Constant(0)
    bc = DirichletBC(E.sub(0).sub(2), u_bc, Boundary)

    #define unit normal 
    nu = FacetNormal(mesh)

    # now for initial value of u1, solve the system for initial pa1
    for i in range(surface.num_vertices()): 
        j = vs_2_vb.array()[i] 
        bulk_pa1.vector()[v2d_W[j]] = pa1.vector()[v2d_S[i]] 

    Ec1 = Function(W)
    w1 = Function(E)
    u1 = Function(P)

    Ec1.interpolate(Expression('k7*(1+pow(k8*ca1,p))', ca1 = ca1, k7 = k7, k8 = k8, p = p, degree=3))
    Ec = Constant(0.6)
    #define lame parametersx
    if coupling == 3 or coupling == 4:
        mu = Ec1/(2*(1+nu_))
        lambda_ = Ec1*nu_/((1+nu_)*(1-2*nu_))
    else:
        mu = Ec/(2*(1+nu_))
        lambda_ = Ec*nu_/((1+nu_)*(1-2*nu_))

    a = inner(sigma(u), epsilon(v))*dx + (dot(l1,v) + dot(u,d1)+ dot(l2*xe1,v)+ dot(l3*xe2,v)+ dot(l4*xe3,v) + dot(u,d2*xe1) + dot(u,d3*xe2) + dot(u,d4*xe3))*dx
    L = k6*dot(bulk_pa1*nu, v)*ds

    if partfixed == 'yes':
        solve(a == L, w1,bcs=bc, solver_parameters={'linear_solver':'mumps'})
    else:
        solve(a == L, w1,solver_parameters={'linear_solver':'mumps'})

    fa.assign([u1,Function(Z1),Function(Z2),Function(Z3),Function(Z4)],w1) 

    # Output file
    if partfixed == 'yes':
        if twoDstim == 'yes':
            file_cd = File("../results/simulations/coupled"+str(coupling)+"_2D_cd_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_ca = File("../results/simulations/coupled"+str(coupling)+"_2D_ca_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_pa = File("../results/simulations/coupled"+str(coupling)+"_2D_p_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_u = File("../results/simulations/coupled"+str(coupling)+"_2D_u_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_ec = File("../results/simulations/coupled"+str(coupling)+"_2D_ec_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_dtu_div = File("../results/simulations/coupled"+str(coupling)+"_2D_dtudiv_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
        else:
            file_cd = File("../results/simulations/coupled"+str(coupling)+"_cd_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_ca = File("../results/simulations/coupled"+str(coupling)+"_ca_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_pa = File("../results/simulations/coupled"+str(coupling)+"_p_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_u = File("../results/simulations/coupled"+str(coupling)+"_u_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_ec = File("../results/simulations/coupled"+str(coupling)+"_ec_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_dtu_div = File("../results/simulations/coupled"+str(coupling)+"_dtudiv_partfixed_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
    else:       
        if twoDstim == 'yes':
            file_cd = File("../results/simulations/coupled"+str(coupling)+"_2D_cd_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_ca = File("../results/simulations/coupled"+str(coupling)+"_2D_ca_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_pa = File("../results/simulations/coupled"+str(coupling)+"_2D_p_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_u = File("../results/simulations/coupled"+str(coupling)+"_2D_u_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_ec = File("../results/simulations/coupled"+str(coupling)+"_2D_ec_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_dtu_div = File("../results/simulations/coupled"+str(coupling)+"_2D_dtudiv_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
        else:
            file_cd = File("../results/simulations/coupled"+str(coupling)+"_cd_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_ca = File("../results/simulations/coupled"+str(coupling)+"_ca_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_pa = File("../results/simulations/coupled"+str(coupling)+"_p_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", "compressed")
            file_u = File("../results/simulations/coupled"+str(coupling)+"_u_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_ec = File("../results/simulations/coupled"+str(coupling)+"_ec_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')
            file_dtu_div = File("../results/simulations/coupled"+str(coupling)+"_dtudiv_neumann_m"+str(meshnr)+"_dt="+str(dt)+"_T="+str(T)+"_k6="+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+"_"+str(bigE)+"E.pvd", 'compressed')


    # initialize functions
    cd_new = Function(W)
    ca_new = Function(W)
    pa_new = Function(S)
    w_new = Function(E)
    u_new = Function(P)
    Ec_new = Function(W)
    dtu_div = Function(W)
    u_div = Function(W)

    # define constants for time loop
    N = int(np.floor(T/dt))
    print('N =',N)
    temp_der_stats = np.zeros((5,N)) #to store l2 norms in each timestep
    temp_stats = np.zeros((5,N)) #to store l2 norms in each timestep
    temp_stats_minmax = np.zeros((8,N)) #to store l2 norms in each timestep
    variables = np.zeros((11,N)) #to store l2 norms in each timestep

    t = dt
    for n in range(0,N):
        print(n)

        # transform p on surface to p in bulk
        for i in range(surface.num_vertices()): 
            j = vs_2_vb.array()[i] 
            bulk_pa1.vector()[v2d_W[j]] = pa1.vector()[v2d_S[i]] 

        #define E_c
        Ec1.interpolate(Expression('k7*(1+pow(k8*ca1,p))', ca1 = ca1, k7 = k7, k8 = k8, p = p, degree=3))

        #define lame parametersx
        if coupling == 3 or coupling == 4:
            mu = Ec1/(2*(1+nu_))
            lambda_ = Ec1*nu_/((1+nu_)*(1-2*nu_))
        else:
            mu = Ec/(2*(1+nu_))
            lambda_ = Ec*nu_/((1+nu_)*(1-2*nu_))
        
        # define the rhs of the linear elasticity equation
        #a = inner(sigma(u), epsilon(v))*dx + (dot(l1,v) + dot(u,d1)+ dot(l2*xe1,v)+ dot(l3*xe2,v)+ dot(l4*xe3,v) + dot(u,d2*xe1) + dot(u,d3*xe2) + dot(u,d4*xe3))*dx
        L = k6*dot(bulk_pa1*nu, v)*ds

        # Compute solution
        if partfixed == 'yes':
            solve(a == L, w_new, bcs=bc, solver_parameters={'linear_solver':'mumps'})
        else:
            solve(a == L, w_new,solver_parameters={'linear_solver':'mumps'})

        # need to project u into functionspace P 
        fa.assign([u_new,Function(Z1),Function(Z2),Function(Z3),Function(Z4)],w_new)

        # Weak statement of both sides of activated c
        a_cd = cd*vd*dx + dt*D1*inner(grad(cd), grad(vd))*dx + dt*(k2+k3)*cd*vd*ds + dt*k9*Max(tr(sigma(u_new)),0)*cd*vd*dx
        L_cd = cd1*vd*dx + dt*cf*k1*ca1*vd*dx
        
        A_cd = assemble(a_cd)
        b_cd = assemble(L_cd)
        
        #now solve the systems
        begin('Solving ligand concentration')
        solve(A_cd, cd_new.vector(), b_cd)
        end()

        # Weak statement of both sides of deactivated c
        a_ca = ca*va*dx + dt*D2*inner(grad(ca), grad(va))*dx + dt*k1*ca*va*dx
        L_ca = ca1*va*dx + dt*(k2+k3)/cf*cd_new*va*ds + dt*k9*Max(tr(sigma(u_new)),0)*cd_new*va*dx
        
        A_ca = assemble(a_ca)
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
        u_der = np.sqrt(assemble(((u_new-u1)/dt)**2*dx(mesh))/vl)
        # another way to compute this norm: u_err = errornorm(ue, u, norm_type='L2', degree_rise=3)

        ca_temp = np.sqrt(assemble((ca1)**2*dx(mesh))/vl)
        cd_temp = np.sqrt(assemble((cd1)**2*dx(mesh))/vl)
        pa_temp = np.sqrt(assemble((pa1)**2*dx(surface))/ar)
        u_temp = np.sqrt(assemble((u1)**2*dx(mesh))/vl)

        ca_min = np.min(ca1.compute_vertex_values())
        ca_max = np.max(ca1.compute_vertex_values())
        cd_min = np.min(cd1.compute_vertex_values())
        cd_max = np.max(cd1.compute_vertex_values())
        pa_min = np.min(pa1.compute_vertex_values())
        pa_max = np.max(pa1.compute_vertex_values())
        u_min = np.min(u1.compute_vertex_values())
        u_max = np.max(u1.compute_vertex_values())

        temp_der_stats[:,n] = np.array([ca_der, cd_der, pa_der, u_der, t]) #store L2 norm
        temp_stats[:,n] = np.array([ca_temp, cd_temp, pa_temp, u_temp, t-dt])
        temp_stats_minmax[:,n] = np.array([ca_min, ca_max, cd_min, cd_max, pa_min, pa_max, u_min, u_max])

        trace_s = Function(W)
        trace_s.assign(project(tr(sigma(u_new)),W))
        traces_temp = np.sqrt(assemble((trace_s)**2*dx(mesh))/vl)
        traces_min = np.min(trace_s.compute_vertex_values())
        traces_max = np.max(trace_s.compute_vertex_values())
        dtu_div.assign(project((div(u_new)-div(u1))/dt,W))
        dtu_div_l2 = np.sqrt(assemble(((div(u_new)-div(u1))/dt)**2*dx(mesh))/vl)
        u_div.assign(project(div(u1),W))
        u_div_l2 = np.sqrt(assemble(u_div**2*dx(mesh))/vl)
        u_div_min = np.min(u_div.compute_vertex_values())
        u_div_max = np.max(u_div.compute_vertex_values())

        ec_temp = np.sqrt(assemble((Ec1)**2*dx(mesh))/vl)
        ec_min = np.min(Ec1.compute_vertex_values())
        ec_max = np.max(Ec1.compute_vertex_values())

        variables[:,n] = np.array([traces_min, traces_max, traces_temp, ec_min, ec_max, ec_temp, dtu_div_l2, u_div_min, u_div_max, u_div_l2, t]) #store L2 norm  

        # write all solutions to predetermined files
        file_cd.write(cd1,t-dt)
        file_ca.write(ca1,t-dt)
        file_pa.write(pa1,t-dt)
        file_ec.write(Ec1,t-dt)
        file_u.write(u1,t-dt)
        file_dtu_div.write(dtu_div, t-dt)

        # renew u1 and next timestep
        cd1.assign(cd_new)
        ca1.assign(ca_new)
        pa1.assign(pa_new)
        u1.assign(u_new)

        t += dt

    if partfixed == 'yes':
        if twoDstim == 'yes':
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed_2D-der-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_der_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed_2D-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed_2D-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats_minmax)
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed_2D-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', variables)
        else:
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed-der-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_der_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats_minmax)
            np.save('../results/temp/tempstats'+str(coupling)+'_partfixed-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', variables)
    else:
        if twoDstim == 'yes':
            np.save('../results/temp/tempstats'+str(coupling)+'_2D-der-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_der_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'_2D-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'_2D-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats_minmax)
            np.save('../results/temp/tempstats'+str(coupling)+'_2D-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', variables)
        else:
            np.save('../results/temp/tempstats'+str(coupling)+'-der-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_der_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats)
            np.save('../results/temp/tempstats'+str(coupling)+'-minmax-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', temp_stats_minmax)
            np.save('../results/temp/tempstats'+str(coupling)+'-oth-m'+str(meshnr)+'_dt='+str(dt)+'_T='+str(T)+'_k6='+str(k6)+"_C1="+str(C1)+"_k7="+str(k7)+"_k8="+str(round(k8, 2))+"_p="+str(p)+"_nuc="+str(nu_)+'_'+str(bigE)+'E', variables)
