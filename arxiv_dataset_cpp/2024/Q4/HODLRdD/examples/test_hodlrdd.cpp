#include <cstdlib>
#include "../include/HDD_matrix.hpp"
#include "../include/HDD_clusters.hpp"
#include "../include/kernel_function.hpp"
#include "../include/myHeaders.hpp"
#include "../include/points_dt.hpp"
#include "../include/LowRank_matrix.hpp"
#include "../include/user_kernel.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    std::vector<ptsnD>* gridPoints;
    if(argc>1){
        numPoints = atoi(argv[1]);
        INTERACTION_TYPE_ALLOWED = atoi(argv[2]);
    }
    else
        numPoints = 10;
    N = pow(numPoints, NDIM);
    // Bounding box of the kernel
    Eigen::VectorXd X(NDIM), Y(NDIM);
    for (int i = 0; i < NDIM; ++i)
    {
        X(i) = -1;
        Y(i) = 1;
    }

    kernel_4d_test *ker = new kernel_4d_test();
    ker->get_points(gridPoints);
    kernel_function<kernel_4d_test> *kernelfunc = new kernel_function<kernel_4d_test>(ker);
    HODLRdD_matrix Kmat = HODLRdD_matrix(ker, gridPoints, X, Y);
    Kmat.Assemble_matrix_operators();
    Vec b_test,b_true,x_test,x_true = Vec::Random(N);
    //std::string x_file_name = data_directory + "x_1overR2_" + std::to_string(N) + ".bin";
    //std::string rhs_file_name = data_directory + "rhs_1overR2_" + std::to_string(N) + ".bin";
    //x_true = storedata::load_vec(x_file_name);
    //b_true = storedata::load_vec(rhs_file_name);
    std::vector<size_t> v3;
    for (size_t i = 0; i < N; i++)
        v3.push_back(i);
    b_true = Vec::Zero(N, 1);
#pragma omp parallel num_threads(nThreads) shared(x_true, v3, kernelfunc, b_true, N)
    {
#pragma omp for schedule(dynamic, 10)
        for (size_t i = 0; i < N; i++)
            b_true(i) = kernelfunc->getRow(i, v3).dot(x_true);
    }
    std::cout << "The size of K matrix " << Kmat.get_size() << std::endl;
    
    b_test = Kmat * x_true; // * Operator
    
    x_test = Kmat.solve(b_true);
    Kmat.print_matrix_statistics();
    //Kmat.print_matrix_details();   // Uncomment if you want to see the details of the interactions
    
    std::cout << "Relative Error in matvec   ... " << Vec_ops::relative_error(b_true, b_test) << std::endl;
    std::cout << "Relative Error in solution ... " << Vec_ops::relative_error(x_true, x_test) << std::endl;

    return 0;
}

/*      points_dt Test case
******Headers******
#include "points_dt.hpp"
*******************
 // Verify the max_norm, 2-norm and the namespaces
    ptsnD A,B;
    A.x[0] = 0.5;
    A.x[1] = 0.5;
    B.x[0] = 0.75;
    B.x[1] = 0.75;

    cout << "Max norm .." << nd_points::max_norm_distance(A, B) << endl;
    cout << "2-norm .." << nd_points::euclidean_distance(A, B) << endl;

*/

/*      Interaction type of two boxes provided the center of the clusters
        ptsnD c1,c2;
        c1.x[0] = 0.5;
        c1.x[1] = 0.5;
        c1.x[2] = 0.5;
        c2.x[0] = 2.5;
        c2.x[1] = 1.5;
        c2.x[2] = 1.5;
        int tmp = INTERACTION_type(c1, c2, 1.0);
        cout << "Interaction type .. "<< tmp << std::endl;

int INTERACTION_type(ptsnD A, ptsnD B, double L)
{
    int type_sharing = NDIM;
    // This ensures whether cluster |A|  |B| in the ith dimension image
    double dp =  nd_points::euclidean_distance(A, B) / L;
    dp *= dp;
    //std::cout << "Frac : "<< dp <<std::endl;
    if (double(NDIM)+1 > dp)
        type_sharing -= std::round(dp);
    else
        type_sharing = -1;
    return type_sharing;
}
    */

/*      User Kernel made into one file


class userkernel{
 std::vector<ptsnD>* gridPoints = new std::vector<ptsnD>; // location of particles in the domain
public:
 userkernel()
 {
     VectorXd  loc_dir[NDIM];   // Xdir, Ydir has been replaced
     // std::cout << "Cheb Points" << std::endl;
     Eigen::VectorXd X(NDIM), Y(NDIM);
     for(int i=0; i<NDIM; ++i){
         X(i) = -1;
         Y(i) =  1;
     }
     int cts = 0;
     for(int i=0; i<NDIM; ++i){
         loc_dir[i] = uniform_nodes(X(i), Y(i), numPoints);
     }
     //std::cout << Xdir << std::endl;

     for (int i = 0; i < numPoints; i++){
         for (int j = 0; j < numPoints; j++){
            // for (int k = 0; k < numPoints; k++){
              //   for (int l = 0; l < numPoints; l++){
                     ptsnD temp;
                     temp.x[0] = loc_dir[0][i];
                     temp.x[1] = loc_dir[1][j];
                //     temp.x[2] = loc_dir[2][k];
             //    temp.x[3] = loc_dir[3][l];
                     temp.id = cts++;
                     gridPoints->push_back(temp);
             //    }
             //}
         }
     }
 }
 // The Green's function in 4D
 dtype_base Kernel_Fun(dtype_base x){
     double c = -1.0/(4*PI*PI);
     return c/(x);
 }

 dtype_base getMatrixEntry(int i, int j){
     double r;
     int n = numPoints;
     double h = 1.0/n;
     double h4 = pow(h,4.0);
     if(i==j){
         double* a = new double[4];
         double* b = new double[4];

         //a[0] = 0, a[1] = 0, a[2] = 0, a[3] = 0;
         //b[0] = h*0.5, b[1] = h*0.5, b[2] = h*0.5, b[3] = h*0.5;
         //r = 1.0 +  quadruple_integral(a,b);         // Second kind
         r = 500.0;

     }
     else{
         ptsnD a,b;
         a = gridPoints->at(i);
         b = gridPoints->at(j);
         r = nd_points::euclidean_distance(a, b);
         r = Kernel_Fun(r);                        // Kernel function
     }
     return r;
 }

 ~userkernel(){
 }
};


*/

/*      Preliminary Tests for cluster subdivision correctness
    std::vector<size_t> v1;
    std::vector<size_t> v2;
    std::vector<size_t> v3;
    // for(int i=0;i<50;i++)
    //     v1.push_back(i);
    // for (int i = 50; i < 100; i++)
    //     v2.push_back(i);
    // //std::iota(std::begin(v1), std::end(v1), 0); // Fill with 0, 1, ..., 99.

    //std::iota(std::begin(v2), std::end(v2), 50); // Fill with 0, 1, ..., 99.
    // std::cout << "Grid_points A" << std::endl;
    // cluster A(X,Y,gridPoints);
    // A.add_points(v1);
    // std::cout << "Grid_points B" << std::endl;
    // cluster B(X, Y, gridPoints);
    // B.add_points(v2);


    std::vector<cluster *> t;
    //A.print_cluster();
    // A.level_clustering(t);
    // for(int i=0;i<t.size();i++)
    //     t[i]->print_cluster();
*/

/*      ACA Test function
    // std::cout << kernelfunc->getRow(1, v2) << endl;
    // std::cout << kernelfunc->getCol(1,v1) << endl;
    // Mat K,L, R;

    // K = kernelfunc->getMatrix(v1,v2);
    // LowRankMat<userkernel> LR(kernelfunc, v1, v2, true);
    // bl = LR * x;
    // std::cout << K.rows() << "," << K.cols() << std::endl;
    // std::cout << x.size() << std::endl;

    // b = K*x;
    // std::cout << "Rank of ACA.. " << LR.rank() << std::endl;
    // std::cout << "Relative Error (ACA).." << Vec_ops::relative_error(b, bl) << std::endl;
    // kernelfunc->ACA_FAST(L,R,0.000001,v1,v2);
    // bl = L*(R.transpose()*x);
    // std::cout << "Relative Error (ACA).."<< Vec_ops::relative_error(b,bl) << std::endl;
*/

/* True solution test case

    if (N < 500000)
    {
        b2 = Vec::Zero(N, 1);
        std::vector<size_t> v3;
        for (size_t i = 0; i < N; i++)
            v3.push_back(i);
        for (size_t i = 0; i < N; i++)
            b2(i) = kernelfunc->getRow(i, v3).dot(x_test); // Exact value matrix_vector
        // b2 = Kmat * x_sol; // Exact value
        // std::cout << "x" << std::endl;
        // std::cout << x_test << std::endl;
        // std::cout << "x _ sol" << std::endl;
        // std::cout << x_sol << std::endl;
        std::cout << "Relative Error in hmatrix matvec ... " << Vec_ops::relative_error(b2, b1) << std::endl;
        std::cout << "Relative Error in GMRES solution ... " << Vec_ops::relative_error(x_sol, x_test) << std::endl;
    }
*/

// ?? Interaction list

// Statistics for the article
// 1) Compression ratio (Matrix)
// 2) Maximum rank of the Tree (Tree)
// Estimated memory vs the original memory
