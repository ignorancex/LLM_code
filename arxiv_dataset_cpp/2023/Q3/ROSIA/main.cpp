/////////////////////////////////////////////////////////////////////////////
//
//                          ROSIA
//
// This package contains the source code which implements the
// rotation-search-based star identification algorithm proposed in
//
// @article{chng2022rosia,
//  title={ROSIA: Rotation-Search-Based Star Identification Algorithm},
//  author={Chng, Chee-Kheng and Bustos, Alvaro Parra and McCarthy, Benjamin and Chin, Tat-Jun},
//  journal={arXiv preprint arXiv:2210.00429},
//  year={2022}
//}
//
// Copyright (c) 2023 Chee-Kheng Chng
// Australian Institute for Machine Learning (AIML)
// School of Computer Science, The University of Adelaide, Australia
// Please acknowledge the authors by citing the above paper in any academic
// publications that have made use of this package or part of it.
//
/////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <sys/resource.h>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>

#ifdef __cplusplus
extern "C"
{
#endif //__cplusplus
#include <cblas.h>
#ifdef __cplusplus
}
#endif //__cplusplus


#include <string>
//#include "mex.h"
//#include "matrix.h"
#include "reg_common.h"
#include "registration.h"

using namespace reg;
using namespace reg::search;


void catalog_reading(std::string filename, Matrix3X &star_vec, Matrix1X &star_mag, Matrix1X &hip_id, Matrix1X &cd1, Matrix1X &cd2) {

    std::ifstream myfile(filename, std::ios_base::in);  // open a file to perform write operation using file object
    int num_stars = 0;
    std::string line;
    if (myfile.is_open()) {
        while (myfile.peek() != EOF) {
            getline(myfile, line);
            num_stars++;
        }


        myfile.close();

        double *local_star_vec = (double *) malloc(3 * num_stars * sizeof(double));
        double *local_mag = (double *) malloc(1 * num_stars * sizeof(double));
        double *local_hip = (double *) malloc(1 * num_stars * sizeof(double));
        double *local_cd1 = (double *) malloc(1 * num_stars * sizeof(double));
        double *local_cd2 = (double *) malloc(1 * num_stars * sizeof(double));

        std::ifstream myfile(filename, std::ios_base::in);  // open a file to perform write operation using file object
        std::string line;


        int star_idx = 0;
        int cd_idx = 0;

        if (myfile.is_open()) {
            // Use the getline function to read the file line by line
            while (getline(myfile, line)) {
                // Declare a stringstream to extract the numbers from the line
                std::stringstream ss(line);
                for (int i = 0; i < 3; i++) {
                    ss >> local_star_vec[i + star_idx];
                }
                ss >> local_mag[cd_idx];
                ss >> local_hip[cd_idx];

                ss >> local_cd1[cd_idx];
                ss >> local_cd2[cd_idx];

                star_idx = star_idx + 3;
                cd_idx = cd_idx + 1;
            }
        }

        star_vec.n = (int) num_stars;
        star_vec.x = local_star_vec;

        star_mag.n = (int) num_stars;
        star_mag.x = local_mag;

        hip_id.n = (int) num_stars;
        hip_id.x = local_hip;

        cd1.n = (int) num_stars;
        cd1.x = local_cd1;

        cd2.n = (int) num_stars;
        cd2.x = local_cd2;

    }
}

void input_reading(std::string filename, Matrix3X &star_vec, Matrix1X &star_mag, Matrix1X &hip) {
    std::ifstream myfile(filename, std::ios_base::in);  // open a file to perform write operation using file object
    int num_stars = 0;
    std::string line;
    if (myfile.is_open()) {
        while (myfile.peek() != EOF) {
            getline(myfile, line);
            num_stars++;
        }


        myfile.close();

        double *local_star_vec = (double *) malloc(3 * num_stars * sizeof(double));
        double *local_mag = (double *) malloc(1 * num_stars * sizeof(double));
        double *local_hip = (double *) malloc(1 * num_stars * sizeof(double));


        std::ifstream myfile(filename, std::ios_base::in);  // open a file to perform write operation using file object
        std::string line;

        int star_idx = 0;
        int cd_idx = 0;

        if (myfile.is_open()) {
            // Use the getline function to read the file line by line
            while (getline(myfile, line)) {
                // Declare a stringstream to extract the numbers from the line
                std::stringstream ss(line);
                for (int i = 0; i < 3; i++) {
                    ss >> local_star_vec[i + star_idx];
                }
                ss >> local_mag[cd_idx];
                ss >> local_hip[cd_idx];

                star_idx = star_idx + 3;
                cd_idx = cd_idx + 1;
            }
        }
        star_vec.n = (int) num_stars;
        star_vec.x = local_star_vec;


        star_mag.n = (int) num_stars;
        star_mag.x = local_mag;

        hip.n = (int) num_stars;
        hip.x = local_hip;

    }
}


void write_output(std::string R_filename, std::string q_filename, std::string time_filename, const Matrix3 matrixR, const int q){
    std::fstream file;
    file.open(R_filename, std::ios::out | std::ios::app);
    file<<matrixR.m[0]<<" "<<matrixR.m[1]<<" "<<matrixR.m[2]<<"\n";
    file<<matrixR.m[3]<<" "<<matrixR.m[4]<<" "<<matrixR.m[5]<<"\n";
    file<<matrixR.m[6]<<" "<<matrixR.m[7]<<" "<<matrixR.m[8]<<"\n";
    file.close();

    file.open(q_filename, std::ios::out | std::ios::app);
    file<<q<<"\n";
    file.close();

//    file.open(time_filename, std::ios::out | std::ios::app);
//    file<<time<<"\n";
//    file.close();
}


struct MyVector {
    double x;
    double y;
    double z;

    MyVector(double x, double y, double z) : x(x), y(y), z(z) {}

    double dot(const MyVector& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    double magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};

double computeAngularDistance(const MyVector& v1, const MyVector& v2) {
    double dotProduct = v1.dot(v2);
    double magnitudes = v1.magnitude() * v2.magnitude();

    if (magnitudes == 0) {
        throw std::invalid_argument("One of the vectors is a zero vector");
    }

    double cosineOfAngle = dotProduct / magnitudes;

    // Clamp the value between -1 and 1 in case of numerical errors
    cosineOfAngle = std::max(-1.0, std::min(1.0, cosineOfAngle));

    double angleInRadians = std::acos(cosineOfAngle);

    return angleInRadians;
}


std::vector<std::vector<double>> computeAllAngularDistances(const std::vector<MyVector>& vectors) {
    size_t N = vectors.size();
    std::vector<std::vector<double>> angularDistances(N, std::vector<double>(N, 0.0));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            angularDistances[i][j] = computeAngularDistance(vectors[i], vectors[j]);
            angularDistances[j][i] = angularDistances[i][j]; // the matrix is symmetric
        }
    }

    return angularDistances;
}


std::vector<double> findTwoSmallest(const std::vector<double>& vec) {
    std::priority_queue<double, std::vector<double>, std::greater<double>> pq(vec.begin(), vec.end());

    std::vector<double> result;
    result.push_back(pq.top()); pq.pop();
    result.push_back(pq.top()); pq.pop();

    return result;
}

std::vector<unsigned int> getSortingIndices(Matrix1X* matrix) {
    size_t n = matrix->cols();

    // Create a vector of indices
    std::vector<unsigned int> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort the indices based on the values of the matrix
    std::sort(indices.begin(), indices.end(),
              [matrix](unsigned int i1, unsigned int i2) { return (*matrix)(i1) < (*matrix)(i2); });

    return indices;
}

int main(int argc, char* argv[]){
    const double dist_th = std::stod(argv[1]);
    const double mag_th = std::stod(argv[2]);

    std::cout<<"Angular distance threshold: "<<dist_th<<std::endl;
    std::cout<<"Magnitude threshold: "<<mag_th<<std::endl;

    std::string catalog_filename = argv[3];
    std::string out_dir = argv[4];
    std::string file_dir = argv[5];

    const int i = std::stoi(argv[6]);

    Matrix3X catalog_vec;
    Matrix1X catalog_mag;
    Matrix1X catalog_hip_id;
    Matrix1X catalog_cd1;
    Matrix1X catalog_cd2;

    catalog_reading(catalog_filename, catalog_vec, catalog_mag, catalog_hip_id, catalog_cd1, catalog_cd2);


    std::string str_i = std::to_string(i);
    std::string curr_scene_filename = file_dir + str_i + ".txt";
    std::string curr_output_filename = out_dir + str_i + ".txt";

    Matrix3X input_vec;
    Matrix1X input_mag;
    Matrix1X input_cd1;
    Matrix1X input_cd2;
    Matrix1X input_hip;

    
    input_reading(curr_scene_filename, input_vec, input_mag, input_hip);

    //////////////////////////////// sort based on magnitudes ////////////////////////////////
    std::vector<unsigned int> indices = getSortingIndices(&input_mag);

    input_mag.sort(indices.data());
    input_vec.sort(indices.data());
    input_hip.sort(indices.data());

//    for (size_t i = 0; i < input_vec.n; i++){
//        std::cout<<input_vec.x[i*3]<<" ";
//        std::cout<<input_vec.x[i*3+1]<<" ";
//        std::cout<<input_vec.x[i*3+2]<<" ";
//        std::cout<<input_hip(i)<<std::endl;
//    }

    //////////////////////////////// compute two closest angular distances ////////////////////////////////

    std::vector<MyVector> vectors;
    for (int j = 0; j < input_vec.n; ++j) {
        vectors.push_back(MyVector(input_vec.x[j*3], input_vec.x[j*3+1], input_vec.x[j*3+2]));
    }

    std::vector<std::vector<double>> angularDistances = computeAllAngularDistances(vectors);
    double *tmp_cd1 = (double *) malloc(1 * input_vec.n * sizeof(double));
    double *tmp_cd2 = (double *) malloc(1 * input_vec.n * sizeof(double));

    for (size_t j = 0; j < vectors.size(); ++j) {
        // Exclude the angle to itself (which is always zero) by setting it to infinity
        angularDistances[j][j] = std::numeric_limits<double>::infinity();

        std::vector<double> twoSmallest = findTwoSmallest(angularDistances[j]);

        tmp_cd1[j] = twoSmallest[0];
        tmp_cd2[j] = twoSmallest[1];

    }


    input_cd1.n = (int)input_vec.n;
    input_cd1.x = tmp_cd1;

    input_cd2.n = (int)input_vec.n;
    input_cd2.x = tmp_cd2;

    const int buckets = (int)catalog_vec.n/10;

    size_t q;
    AxisAngle resp;

    const int gap = 0;
    const int lwbnd = 0;

    //////////////////////////////// BnB starts////////////////////////////////

    q = bnb_rsearch_3dof_mcirc_ml(input_vec, catalog_vec, input_mag, catalog_mag, input_cd1, input_cd2,
                                  catalog_cd1, catalog_cd2,
                                  dist_th, mag_th, gap, lwbnd, buckets, resp);

//    std::cout<<q<<std::endl;

    /* if timing is interested, this is where to time the algorithm. The star ID should be provided here in a complete implementation.
     * Only in this prototyping code, the star ID is retrieved from the following code.*/


    Matrix3 R;
    fromAxisAngle(R, resp);

    std::vector<MyVector> cat_vectors;
    for (int j = 0; j < catalog_vec.n; ++j) {
        cat_vectors.push_back(MyVector(catalog_vec.x[j*3], catalog_vec.x[j*3+1], catalog_vec.x[j*3+2]));
    }

    for (size_t i = 0; i < input_vec.n; i++){

        Vector3 result = multiply(R,input_vec.x+3*i);
        std::vector<MyVector> rot_input_vectors;
        rot_input_vectors.push_back(MyVector(result[0], result[1], result[2]));

        std::vector<double> angularDistances;
        for (size_t j = 0; j < catalog_vec.n; j++) {
            angularDistances.push_back(computeAngularDistance(rot_input_vectors[0], cat_vectors[j]));
        }

        auto smallest = std::min_element(angularDistances.begin(), angularDistances.end());
        int smallestIndex = std::distance(angularDistances.begin(), smallest);
        double smallestValue = *smallest;

        std::ofstream outfile(curr_output_filename, std::ios::app);
        if (smallestValue <= dist_th){
            // write the HIP ID result
            if (outfile.is_open()) {
                outfile << input_vec.x[i*3]<<" ";
                outfile << input_vec.x[i*3+1]<<" ";
                outfile << input_vec.x[i*3+2]<<" ";
                outfile << catalog_hip_id(smallestIndex) << "\n";
                outfile.close();
            } else {
                std::cerr << "Unable to open file: " << curr_output_filename << std::endl;
            }
        }

    }

    std::ofstream outfile(curr_output_filename, std::ios::app);
    // write the rotation result
    if (outfile.is_open()) {
        outfile<<R.m[0]<<" "<<R.m[1]<<" "<<R.m[2]<<"\n";
        outfile<<R.m[3]<<" "<<R.m[4]<<" "<<R.m[5]<<"\n";
        outfile<<R.m[6]<<" "<<R.m[7]<<" "<<R.m[8]<<"\n";
    } else {
        std::cerr << "Unable to open file: " << curr_output_filename << std::endl;
    }

    input_vec.x = NULL;
    input_mag.x = NULL;
    input_cd1.x = NULL;
    input_cd2.x = NULL;


    
}


