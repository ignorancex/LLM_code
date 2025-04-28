#include "lpm.h"

/* definition of global variables */
/* int */
int ntype, particles_first_row, nparticle, rows, layers, nneighbors, nneighbors1, nneighbors2, dim;
int lattice, nneighbors_AFEM, nneighbors_AFEM1, nneighbors_AFEM2, plmode, eulerflag, nslip_face;
int max_nslip_vector, nslipSys, nbreak;

int *IK, *JK, *type, *dispBC_index, *fix_index, *nslip_vector, *lacknblist, pbc[NDIM], *pl_flag;
int *nb, *nb_initial, *nb_conn, *state_v;
int **neighbors, **neighbors1, **neighbors2, **neighbors_AFEM;
int **K_pointer, **conn, **nsign, **yield_bond, **cp_Jact, **damage_broken;

/* double precision float */
double radius, hx, hy, hz, neighbor1_cutoff, neighbor2_cutoff, angle1, angle2, angle3, particle_volume;
double R_matrix[NDIM * NDIM], box[2 * NDIM], cp_tau0[3], cp_taus[3], cp_eta, cp_p, cp_h0, cp_q, cp_maxloop;
double box_x, box_y, box_z, cp_q, dtime, cp_theta, J2_H, J2_xi, J2_C;
double damage_threshold, damageb_A, damagec_A, critical_bstrain;

double *K_global, *plastic_K_global, *residual, *Pin, *Pex, *Pex_temp, *disp, *sigmay, *cp_dA;
double *distort_Uy, *reaction_force, *distort_U, *J2_beta_eq, *damage_visual;
double *J2_dlambda, *J2_stresseq, *J2_stressm, *J2_triaxiality;

double **xyz, **xyz_initial, **xyz_temp, **distance, **distance_initial, **KnTve, **F, **csx, **csy, **csz;
double **dL_total, **TdL_total, **dLe_total, **TdLe_total, **csx_initial, **csy_initial, **csz_initial;
double **Ce, **Cp, **distort_F, **distort_backF, **eqdistort_U, **distort_backdL;
double **slip_normal, **schmid_tensor, **schmid_tensor_local, **cp_RSS, **stress_tensor;
double **cp_Cab, **strain_tensor, **dL_ave, **ddL_total, **TddL_total, **F_temp, **ddLp;
double **dL, **ddL, **dLe, **bond_stress, **damage_w;

double **Kn, **Tv, **Kne, **Tve, **J2_alpha, **damage_con, **cp_A, **cp_dgy, **cp_dA_single;
double ***slip_vector, ***dLp, ***J2_beta, ***damage_D, ***cp_gy, ***cp_A_single;

/************************************************************************/
/****************************** Main procedure **************************/
/************************************************************************/
int main(int argc, char *argv[])
{
    printf("\n==================================================\n");
    printf("            Nonlocal LPM Program in C             \n");
    printf("==================================================\n");

    const int nt = omp_get_max_threads(); /* maximum number of threads provided by the computer */
    const int nt_gene = 2;                /* number of threads for general calculation */

    printf("OpenMP with %d/%d threads for bond force calculation\n", nt_gene, nt);

    omp_set_num_threads(nt);
    char *paraFile;
    if (argc == 1)
        paraFile = "input.txt"; /* defalut input file name */
    else if (argc == 2)
        paraFile = *++argv;

    double start = omp_get_wtime(); // record the CPU time, begin

    // lattice type -> lattice number, int
    // square -> 0; hexagon -> 1; simple cubic -> 2; face-centered cubic -> 3
    // body-centered cubic with 1 types of slip systems -> 4
    // body-centered cubic with 2 types of slip systems -> 5
    // body-centered cubic with 3 types of slip systems -> 6
    lattice = 3;

    // dimensionality, int
    if (lattice < 2)
        dim = 2;
    else
        dim = 3;

    // particle radius, double
    radius = 0.3; //0.318; // 0.4; 0.18; 0.14

    // periodic boundary conditions
    // 1 with periodicity, 0 without periodicity
    pbc[0] = 0, pbc[1] = 0, pbc[2] = 0;

    // Euler angles setting for system rotation
    // flag is 0 ~ 2 for different conventions, int
    // // angle1, angle2 and an angle3 are Euler angles in degree, double
    eulerflag = 0; // Kocks convention
    angle1 = PI / 180.0 * 0.0;
    angle2 = PI / 180.0 * 0.0;
    angle3 = PI / 180.0 * 0.0;
    // eulerflag = 0; // direct rotation
    // angle1 = PI / 180.0 * 45.0;
    // angle2 = PI / 180.0 * 0.0;
    // angle3 = PI / 180.0 * 0.0;

    // simulation box size, double
    // xmin; xmax; ymin; ymax; zmin; zmax
    // box[0] = 0.0, box[1] = 25.0; // 3D
    // box[2] = 0.0, box[3] = 25.0;
    // box[4] = 0.0, box[5] = 50.0;
    box[0] = 0.0, box[1] = 10.0; // 3D plate
    box[2] = 0, box[3] = 10.0;
    box[4] = 0.0, box[5] = 10.0;
    // box[0] = 0.0, box[1] = 18.0; // 3D cylinder
    // box[2] = 0.0, box[3] = 18.0;
    // box[4] = 0.0, box[5] = 40.0;
    // box[0] = 0.0, box[1] = 50.0; // 3D plate, mm
    // box[2] = 0.0, box[3] = 48.0;
    // box[4] = 0.0, box[5] = 10.0;
    box_x = box[1] - box[0]; // box size
    box_y = box[3] - box[2];
    box_z = box[5] - box[4];
    createCuboid();

    // (if applicable) read particle information in lammps format
    // char tempFile[] = "Fe_supercell.lmp";
    // char tempSkip = 15; // skip line numbers
    // readLammps(tempFile, tempSkip);

    // move the particles coordinates
    double movexyz[] = {-0., -0., -0.};
    // double movexyz[] = {-0.0, 0., 0.};
    moveParticle(movexyz);

    // create a pre-existed crack
    // double ca1 = -10.0, ca2 = 2.5, ch = 10.0;
    // createCrack(ca1, ca2, ch);

    // modify the configuration
    // removeBlock(-10.0, 10.0, 35.0, 100.0, -100, 100);
    // removeBlock(-10.0, 10.0, -100.0, 13.0, -100, 100);
    // double pc1[3] = {10.0, 12.771, 0.0}, pc2[3] = {10.0, 35.161, 0.0};
    // removeCircle(pc1, 5.0, 'z'); // remove the particles inside the half circle
    // removeCircle(pc2, 5.0, 'z');

    // cut the cuboid and get a cylinder along z direction
    // double pc[3] = {12.5, 12.5, 25.0};
    // createCylinderz(pc, 12.5);
    // removeRingz(pc, 10.0, 5.0);

    // cut the 3D cuboid and get a dog-bone specimen
    // double pc[3] = {9.0, 9.0, 20.0};
    // createCylinderz(pc, 9.0);  // cylinder test specimen
    // removeRingz(pc, 9.0, 4.0); // remove a ring

    initMatrices();
    copyDouble2D(xyz_initial, xyz, nparticle, NDIM);

    printf("\nParticle number is %d\n", nparticle);

    // search neighbor
    searchNormalNeighbor();
    int len_K_global = searchAFEMNeighbor();

    //printf("Neighbor-searching finished, stiffness matrix size is %d\n", len_K_global);

    // assign types for particles located in different rectangular regions
    // xlo, xhi, ylo, yhi, zlo, zhi, type
    ntype = 0;
    type = allocInt1D(nparticle, ntype++); // set particle type as 0 in default
    // setTypeRect(-100.0, 100.0, 1.0 - 2 * radius, 100.0, -100.0, 100.0, ntype++); // top layer, type 1, 2D square, shear
    // setTypeRect(-100.0, 100.0, -100.0, 2 * radius, -100.0, 100.0, ntype++);      // bottom layer, type 2

    // setTypeCircle(pc1[0], pc1[1], 8, ntype++);
    // setTypeCircle(pc2[0], pc2[1], 8, ntype++);

    setTypeRect(-100.0, 100.0, -100.0, 100.0, 10 - 1.2 * radius, 100.0, ntype++);               // top layer, type 1, 3D crystal plasticity
    setTypeRect(-100.0, 1.2 * radius, -100.0, 100.0, 10 - 1.2 * radius, 100.0, ntype++);        // x line, type 2
    setTypeRect(-100.0, 100.0, -100.0, 1.2 * radius, 10 - 1.2 * radius, 100.0, ntype++);        // y line, type 3
    setTypeRect(-100.0, 1.2 * radius, -100.0, 1.2 * radius, 10 - 1.2 * radius, 100.0, ntype++); // top fix point, type 4
    setTypeRect(-100.0, 100.0, -100.0, 100.0, -100.0, 1.2 * radius, ntype++);                   // lower layer, type 5

    // setTypeRect(7, 13, 35, 40 + 4 * radius, -100.0, 100.0, ntype++);                                     // upper circle point, type 1, 3D Gaofeng Zhao
    // setTypeRect(7, 13, 8 - 5 * radius, 13, -100.0, 100.0, ntype++);                                      // lower circle point, type 2
    // setTypeRect(50.0 - 2 * radius, 100.0, 24.0 - 2 * radius, 24.0 + 2 * radius, -100.0, 100.0, ntype++); // side line, type 3
    // setTypeRect(-100.0, 2 * radius, 24.6, 24.7, -100.0, 100.0, ntype++);                                 // crack tip mouth1, type 4
    // setTypeRect(-100.0, 2 * radius, 23.4, 23.5, -100.0, 100.0, ntype++);                                 // crack tip mouth2, type 5

    // setTypeRect(10.0 - 2 * radius, 10.0 + 2 * radius, 40.0 - 2 * radius, 40.0 + 2 * radius, -100.0, 100.0, ntype++);          // upper circle point, type 1, 3D Gaofeng Zhao
    // setTypeRect(10.0 - 2 * radius, 10.0 + 2 * radius, 8.0 - 2 * radius, 8.0 + 2 * radius, -100.0, 100.0, ntype++);            // lower circle point, type 2
    // setTypeRect(50.0 - 2 * radius, 100.0, 24.0 - 2 * radius, 24.0 + 2 * radius, -100.0, 100.0, ntype++);                      // side line, type 3
    // setTypeRect(50.0 - 2 * radius, 100.0, 24.0 - 2 * radius, 24.0 + 2 * radius, 5.0 - 2 * radius, 5.0 + 2 * radius, ntype++); // side point, type 4
    // setTypeRect(-100.0, 1.5 * radius, 25.0, 25.5, -100.0, 100.0, ntype++);                                                    // crack tip mouth1, type 5
    // setTypeRect(-100.0, 1.5 * radius, 22.5, 23.0, -100.0, 100.0, ntype++);                                                    // crack tip mouth2, type 6

    // material elastic parameters setting, MPa
    double C11, C12, C44;
    // double E0 = 210e3, mu0 = 0.3;
    // double E0 = 115e3, mu0 = 0.28; // Gaofeng Zhao
    // double E0 = 31.4e9, mu0 = 0.4; // three point bending
    // double E0 = 146e3, mu0 = 0.3; // 2D plate
    // double E0 = 146e3, mu0 = 0.3;
    // double E0 = 69e3, mu0 = 0.3;
    // double E0 = 31.4e9, mu0 = 0.4; // three point bending
    // plane strain or 3D
    // C11 = E0 * (1.0 - mu0) / (1.0 + mu0) / (1.0 - 2.0 * mu0);
    // C12 = E0 * mu0 / (1.0 + mu0) / (1.0 - 2.0 * mu0);
    // C44 = E0 / 2.0 / (1.0 + mu0);
    // plane stress
    // C11 = E0 / (1.0 - mu0) / (1.0 + mu0);
    // C12 = E0 * mu0 / (1.0 - mu0) / (1.0 + mu0);
    // C44 = E0 / 2.0 / (1.0 + mu0);
    // 3D cases
    // C11 = 108.2e3; // in Hailong's paper
    // C12 = 61.3e3;
    // C44 = 28.5e3;
    C11 = 107.3e3; // Al
    C12 = 60.8e3;
    C44 = 28.3e3;
    // C11 = 236e3; // alpha-Fe
    // C12 = 134e3;
    // C44 = 119e3;
    Ce = allocDouble2D(ntype, 3, 0.);
    for (int i = 0; i < ntype; i++)
    {
        Ce[i][0] = C11; // C11
        Ce[i][1] = C12; // C12
        Ce[i][2] = C44; // C44
    }

    // elasticity or plasticity model settings
    // (0) stress-based J2 plasticity, mixed-linear hardening
    // plmode = 0;
    // sigmay = allocDouble1D(nparticle, 955); // initial yield stress, MPa
    // J2_xi = 0.0;                            // 0 for isotropic and 1 for kinematic hardening
    // J2_H = 0.0;                             // isotropic hardening modulus, MPa, (1007-955)/0.02165

    // (1) rate-dependent crystal plasticity based on Miehe 2001
    plmode = 1;
    cp_maxloop = 10;                                       // maximum loop number for updating active slip systems
    cp_tau0[0] = 1.6, cp_tau0[1] = 0.0, cp_tau0[2] = 0.0;  // tau0 for all slip systems
    cp_taus[0] = 30.0, cp_taus[1] = 0.0, cp_taus[2] = 0.0; // taus for all slip systems
    cp_h0 = 100.0;                                         // initial hardening modulus
    cp_p = 4.0;                                            // strain-rate-sensitivity exponent
    cp_q = 1.0;
    cp_eta = 1000.0; // viscosity parameter

    // (3) energy-based J2 plasticity using return-mapping
    // plmode = 3;
    // sigmay = allocDouble1D(nparticle, 200.); // initial yield stress, MPa
    // J2_xi = 0.0;                             // 0 for isotropic and 1 for kinematic hardening
    // J2_H = 38.714e3;                         // isotropic hardening modulus, MPa

    // (4) energy-based J2 plasticity using incremental updating (Haoyang's algorithm)
    // plmode = 4;
    // for (int i = 0; i < nparticle; i++)
    //     distort_Uy[i] = 200. * 200. / Ce[type[i]][2] / 6.0;
    // J2_xi = 1.0;     // 0 for isotropic and 1 for kinematic hardening
    // J2_H = 38.714e3; // isotropic hardening modulus, MPa

    // (5) stress-based ductile fracture simulation, nonlinear hardening
    // plmode = 5;
    // J2_C = 0.0; // kinematic hardening modulus, MPa

    // (6) elastic (brittle) material
    //plmode = 6;

    printf("Constitutive mode is %d\n", plmode);

    // damage settings
    nbreak = 100;              // limit the broken number of bonds in a single iteration, should be an even number
    critical_bstrain = 1.0e-2; // critical bond strain value at which bond will break
    damageb_A = 10.0;          // (ductile) damage parameter for bond-based damage evolution rule (ductile fracture)
    damagec_A = 200.0;         // (ductile) continuum damage evolution parameter, A;
    damage_threshold = 0.5;    // damage threshold, [0, 1), particle will be frozen after reach this value

    // define the crack
    // defineCrack(ca1, ca2, ch);

    // output parameters
    char aflag = 'z';      // output directional axis
    int dtype = 5;         // particle type used to output displacement or force
    int outdisp_flag = 1;  // output displacement, 0 or 1
    int outforce_flag = 1; // output force, 0 or 1
    int out_step = 1;      // print initial strain and dump info, output step is set as 1
    char dumpflag = 'm';   // 's' for single step; 'm' for multiple step

    char dispFile[] = "result_disp.txt";
    char dispFile1[] = "result_disp_CMOD1.txt";
    char dispFile2[] = "result_disp_CMOD2.txt";
    char forceFile[] = "result_force.txt";
    char stressFile[] = "result_stress.txt";
    char dumpFile[] = "result_position.dump";
    char actFile[] = "result_Jact.txt";
    char slipFile[] = "result_slipRSS.txt";
    char damageFile[] = "result_damage.txt";
    char bondFile[] = "result_brokenbonds.txt";
    char stretchFile[] = "result_stretch.txt";
    char dlambdaFile[] = "result_dlambda.txt";
    char neighborFile[] = "result_neighbor.txt";
    char cabFile[] = "result_Cab.txt";

    // boundary conditions and whole simulation settings
    int n_steps = 100;          // number of loading steps
    dtime = 0.1;                // time step, s
    double step_size = -2.0e-3; // total force or displacement loading
    //double step_size = -100.0; // step size for force or displacement loading
    // int n_steps = 10;        // number of loading steps
    // double step_size = -650; // step size for force or displacement loading

    int nbd = 0, nbf = 0;     // total number of disp or force boundary conditions
    char cal_method[] = "cg"; // calculation method, pardiso or conjugate gradient
    struct dispBCPara dBP[MAXLINE][MAXSMALL] = {0};
    struct forceBCPara fBP[MAXLINE][MAXSMALL] = {0};
    int load_indicator[MAXLINE] = {0}; // tension (1) or compressive (-1) loading condition for uniaxial loading

    // displace boundary conditions
    for (int i = 0; i < n_steps; i++)
    {
        nbd = 0;
        load_indicator[i] = 1;
        dBP[i][nbd].type = 1, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0; // 3D loading, tension
        dBP[i][nbd].type = 2, dBP[i][nbd].flag = 'x', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 2, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 3, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 3, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'x', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;
        dBP[i][nbd].type = 5, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = step_size;
    }
    // for (int i = 0; i < n_steps; i++)
    // {
    //     nbd = 0;
    //     load_indicator[i] = 1;
    //     dBP[i][nbd].type = 1, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = -step_size; // 3D loading
    //     dBP[i][nbd].type = 2, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = step_size;
    //     dBP[i][nbd].type = 3, dBP[i][nbd].flag = 'x', dBP[i][nbd++].step = 0.0;
    //     dBP[i][nbd].type = 3, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = 0.0;
    //     dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'x', dBP[i][nbd++].step = 0.0;
    //     dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = 0.0;
    //     dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;
    // }

    // force boundary conditions
    //for (int i = 0; i < n_steps; i++)
    //{
    //    load_indicator[i] = 1;

    //    nbd = 0;
    //    dBP[i][nbd].type = 1, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0; // 3D loading, tension
    //    dBP[i][nbd].type = 2, dBP[i][nbd].flag = 'x', dBP[i][nbd++].step = 0.0;
    //    dBP[i][nbd].type = 2, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;
    //    dBP[i][nbd].type = 3, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = 0.0;
    //    dBP[i][nbd].type = 3, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;
    //    dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'x', dBP[i][nbd++].step = 0.0;
    //    dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'y', dBP[i][nbd++].step = 0.0;
    //    dBP[i][nbd].type = 4, dBP[i][nbd].flag = 'z', dBP[i][nbd++].step = 0.0;

    //    nbf = 0;
    //    fBP[i][nbf].type = 5;
    //    fBP[i][nbf].flag1 = 'x', fBP[i][nbf].step1 = 0.0;
    //    fBP[i][nbf].flag2 = 'y', fBP[i][nbf].step2 = 0.0;
    //    fBP[i][nbf].flag3 = 'z', fBP[i][nbf++].step3 = step_size;
    //}

    /************************** Simulation begins *************************/
    // compute necessary into before loading starts
    calcKnTv();
    computedL();

    // crystal lattice settings
    slipSysDefine3D(); // define slip systems for cubic systems
    if (lattice == 3 || lattice == 4)
        computeCab();

    // output files
    if (outdisp_flag)
    {
        writeDisp(dispFile, aflag, dtype, 0);
        // writeDisp(dispFile1, 'y', 5, 0);
        // writeDisp(dispFile2, 'y', 6, 0);
    }
    if (outforce_flag)
    {
        writeStress(stressFile, 0, 0);
        writeReaction(forceFile, aflag, dtype, 0);
        // writeForce(forceFile, aflag, 0.3, 0); // lower half body force
    }
    writeDump(dumpFile, 0, dumpflag);

    //writeCab(cabFile, 0);
    // writeCab(cabFile, 288);
    // writeCab(cabFile, 4227);
    //writeCab(cabFile, 4228);
    //writeCab(cabFile, 4229);

    // compute the elastic stiffness matrix
    //omp_set_num_threads(nt_gene);
    //if (dim == 2)
    //    calcStiffness2DFiniteDifference(6);
    //else if (dim == 3)
    //    calcStiffness3DFiniteDifference(6);
    //omp_set_num_threads(nt);

    // omp_set_num_threads(nt_gene);
    double initrun = omp_get_wtime();
    printf("Initialization finished in %f seconds\n\n", initrun - start);

    // incremental loading procedure
    for (int i = 0; i < n_steps; i++)
    {
        int plmode4_flag = 0;
        double startrun = omp_get_wtime();

        printf("######################################## Loading step %d ######################################\n", i + 1);
        copyDouble2D(xyz_temp, xyz, nparticle, NDIM);
        copyDouble2D(F_temp, F, nparticle, nneighbors);
        copyDouble1D(Pex_temp, Pex, dim * nparticle);
        // double time_t1 = omp_get_wtime();
        // printf("Copy costs %f seconds\n", time_t1 - startrun);

        omp_set_num_threads(nt_gene);
        // compute the elastic stiffness matrix
        if (dim == 2)
            calcStiffness2DFiniteDifference(6);
        else if (dim == 3)
            calcStiffness3DFiniteDifference(6);

        double time_t1 = omp_get_wtime();
        printf("Stiffness matrix calculation costs %f seconds\n", time_t1 - startrun);

    label_wei:
        setDispBC(nbd, dBP[i]);  // update displacement BC
        setForceBC(nbf, fBP[i]); // update force BC

        computeBondForceGeneral(4, load_indicator[i]); // incremental updating
        omp_set_num_threads(nt);

    label_broken_bond:
        updateRR(); // residual force vector and reaction force (RR)

        //double time_t2 = omp_get_wtime();
        //printf("Incremental updating costs %f seconds\n", time_t2 - time_t1);

        // compute the Euclidean norm (L2 norm)
        double norm_residual = cblas_dnrm2(dim * nparticle, residual, 1);
        double norm_reaction_force = cblas_dnrm2(countNEqual(dispBC_index, nparticle * dim, 1), reaction_force, 1);
        double tol_multiplier = MAX(norm_residual, norm_reaction_force);
        char tempChar1[] = "residual", tempChar2[] = "reaction";
        printf("Norm of residual is %.5e, norm of reaction is %.5e, tolerance criterion is based on ", norm_residual, norm_reaction_force);
        if (norm_residual > norm_reaction_force)
            printf("%s force\n", tempChar1);
        else
            printf("%s force\n\n", tempChar2);

        // global Newton iteration starts
        int ni = 0;
        while (norm_residual > TOLITER * tol_multiplier && ni < MAXITER)
        {
            printf("Step-%d, iteration-%d: ", i + 1, ni);

            switchStateV(0); // copy the last converged state variable [1] into current one [0]

            // time_t1 = omp_get_wtime();
            // compute the stiffness matrix, then modify it for displacement boundary condition
            if (dim == 2)
            {
                // calcStiffness2DFiniteDifference(6);
                setDispBC_stiffnessUpdate2D();
            }
            else if (dim == 3)
            {
                // calcStiffness3DFiniteDifference(6);
                setDispBC_stiffnessUpdate3D();
            }
            // time_t2 = omp_get_wtime();
            // printf("Modify stiffness costs %f seconds\n", time_t2 - time_t1);

            // solve for the incremental displacement
            if (strcmp(cal_method, "pardiso") == 0)
                solverPARDISO();
            else if (strcmp(cal_method, "cg") == 0)
                solverCG();
            // time_t1 = omp_get_wtime();
            // printf("Solve the linear system costs %f seconds\n", time_t1 - time_t2);

            omp_set_num_threads(nt_gene);
            computeBondForceGeneral(plmode, load_indicator[i]); // update the bond force
            omp_set_num_threads(nt);

            // time_t2 = omp_get_wtime();
            // printf("Update bond force costs %f seconds\n", time_t2 - time_t1);

            // writeDlambda(dlambdaFile, 9039, 9047, i + 1, ni);

            updateRR(ni++); /* update the RHS risidual force vector */
            norm_residual = cblas_dnrm2(dim * nparticle, residual, 1);
            printf("Norm of residual is %.3e, residual ratio is %.3e\n", norm_residual, norm_residual / tol_multiplier);
        }

        // compute the distortional energy, check the yield function
        if (plmode == 4)
        {
            if (plmode4_flag == 0)
            {
                updateStateJ2energyWei(plmode4_flag); // compute temporary distortional energy, update Kn, Tv
                computedL();                          // update the Tcs_xxx
                copyDouble2D(xyz, xyz_temp, nparticle, NDIM);
                copyDouble1D(Pex, Pex_temp, dim * nparticle);
                plmode4_flag = 1;

                if (dim == 2)
                    calcStiffness2DFiniteDifference(6);
                else if (dim == 3)
                    calcStiffness3DFiniteDifference(6);

                goto label_wei;
            }
            else
                updateStateJ2energyWei(plmode4_flag); // update distortional energy and yield surface
        }

        /* accumulate damage, and break bonds when damage reaches critical values */
        // int broken_bond = updateDamageGeneral(bondFile, i + 1, plmode);
        // updateCrack();
        switchStateV(1); // copy current state variable [0] into last converged one [1]

        printf("Loading step %d has finished in %d iterations\n\nData output ...\n", i + 1, ni);

        /* ----------------------- data output setting section ------------------------ */

        // test the strain measure
        // int ii = 1429;
        // printDouble(F[ii], 1, nneighbors);
        // printDouble(bond_stress[ii], 1, nneighbors);
        // printDouble(stress_tensor[ii], 1, 2 * NDIM);
        // printDouble(dL[ii], 1, nneighbors);

        // double strain_tensor[2 * NDIM] = {0.0};
        // strain_tensor[0] = ((1 - mu0) * stress_tensor[ii][0] - mu0 * stress_tensor[ii][1]) * (1 + mu0) / E0;
        // strain_tensor[1] = (-mu0 * stress_tensor[ii][0] + (1 - mu0) * stress_tensor[ii][1]) * (1 + mu0) / E0;
        // strain_tensor[5] = stress_tensor[ii][5] * (1 + mu0) / E0;
        // double xi[8] = {0.0};
        // for (int j = 0; j < nb_initial[ii]; j++)
        // {
        //     xi[j] = distance_initial[ii][j] * (strain_tensor[0] * csx[ii][j] * csx[ii][j] +
        //                                        strain_tensor[1] * csy[ii][j] * csy[ii][j] +
        //                                        strain_tensor[2] * csz[ii][j] * csz[ii][j] +
        //                                        2 * strain_tensor[3] * csy[ii][j] * csz[ii][j] +
        //                                        2 * strain_tensor[4] * csx[ii][j] * csz[ii][j] +
        //                                        2 * strain_tensor[5] * csx[ii][j] * csy[ii][j]);
        // }
        // printDouble(xi, 1, nneighbors);

        if ((i + 1) % out_step == 0)
        {
            if (outdisp_flag)
            {
                writeDisp(dispFile, aflag, dtype, i + 1);
                // writeDisp(dispFile1, 'y', 5, i + 1);
                // writeDisp(dispFile2, 'y', 6, i + 1);
            }
            if (outforce_flag)
            {
                writeStress(stressFile, 0, i + 1);
                writeReaction(forceFile, aflag, dtype, i + 1);
                // writeForce(forceFile, aflag, 0.3, i + 1);
            }
            writeDump(dumpFile, i + 1, dumpflag); /* print particle position info */

            if (plmode == 1)
            {
                writeJact(actFile, i + 1); /* print active slip systems into file */
                writeRSS(slipFile, i + 1);
            }
        }

        // check if breakage happens, recompute stiffness matrix (this is same as a new loading step)
        // if (broken_bond > 0)
        // {
        //     // switchStateV(0); // copy the last converged state variable [1] into current one [0]
        //     // computeBondForceGeneral(plmode, load_indicator[i]); // update the bond force

        //     startrun = omp_get_wtime();
        //     omp_set_num_threads(nt_gene);
        //     // recompute stiffness matrix
        //     if (dim == 2)
        //         calcStiffness2DFiniteDifference(6);
        //     else if (dim == 3)
        //         calcStiffness3DFiniteDifference(6);
        //     omp_set_num_threads(nt);
        //     time_t1 = omp_get_wtime();
        //     printf("\nStiffness matrix calculation costs %f seconds\n", time_t1 - startrun);

        //     goto label_broken_bond;
        // }

        double finishrun = omp_get_wtime();
        printf("Time costed for step %d: %f seconds\n\n", i + 1, finishrun - startrun);
    }

    writeBondstretch(stretchFile, 0);
    writeNeighbor(neighborFile);

    double finish = omp_get_wtime();
    printf("Computation time for total steps: %f seconds\n\n", finish - start);

    // frees unused memory allocated by the Intel MKL Memory Allocator
    mkl_free_buffers();

    return 0;
}

/************************************************************************/
/*************************** End main procedures ************************/
/************************************************************************/
