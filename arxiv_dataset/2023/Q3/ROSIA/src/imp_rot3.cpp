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

#include "reg_common.h"
#include "search.h"
#include "registration.h"
#include "state_priority_hashtable.h"
#include "mcirc_indexation.h"
#include <iostream>
#include <chrono>

using namespace reg::search;

int reg::search::bnb_rsearch_3dof_mcirc(
        const Matrix3X &X, const Matrix3X &Y,
        const Matrix1X &X_mag, const Matrix1X &Y_mag,
        const Matrix1X &IN_d1, const Matrix1X &IN_d2,
        const Matrix1X &CAT_d1, const Matrix1X &CAT_d2,
        const double dist_th, const double mag_th,
        int gap, int lwbnd, int buckets,
        AxisAngle &result)
{
//    mxAssert(th>0,"invalid threshold");

    typedef RotationSearchSpaceRegion3DOFS8 SSR;

    SSR guessAndResult(-PI,-PI,-PI,PI,PI,PI);

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    DataIndexation<SSR> *dsi = new MCircIndexation<SSR>(X, Y, X_mag, Y_mag, IN_d1, IN_d2, CAT_d1, CAT_d2, dist_th, mag_th);
    dsi->sweep();
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "myFunction() took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    const int qual = bnb_search_table<SSR, 8> (
            *dsi, IN_d1, IN_d2, CAT_d1, CAT_d2, mag_th, lwbnd, gap, buckets, guessAndResult);

    delete dsi;

    result = centre(guessAndResult);
    return qual;
}

//int reg::search::bnb_rsearch_3dof_mcirc(
//        const Matrix3X &X, const Matrix3X &Y, double th,
//        int gap, int lwbnd, int buckets,
//        AxisAngle &result)
//{
//    mxAssert(th>0,"invalid threshold");
//
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;
//
//    SSR guessAndResult(-PI,-PI,-PI,PI,PI,PI);
//
//    DataIndexation<SSR> *dsi = new MCircIndexation<SSR>(X, Y, th);
//    dsi->sweep();
//
//    const int qual = bnb_search_table<SSR, 8> (
//            *dsi, lwbnd, gap, buckets, guessAndResult);
//    delete dsi;
//
//    result = centre(guessAndResult);
//    return qual;
//}

int reg::search::bnb_rsearch_3dof_mcirc_ml(
        const Matrix3X &X, const Matrix3X &Y,
        const Matrix1X &X_mag, const Matrix1X &Y_mag,
        const Matrix1X &IN_d1, const Matrix1X &IN_d2,
        const Matrix1X &CAT_d1, const Matrix1X &CAT_d2,
        const double dist_th, const double mag_th,
        int gap, int lwbnd, int buckets,
        AxisAngle &result)
{
//    mxAssert(th>0,"invalid threshold");

    typedef RotationSearchSpaceRegion3DOFS8 SSR;

    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
    DataIndexation<SSR> *dsi = new MCircIndexation<SSR>(X, Y, X_mag, Y_mag, IN_d1, IN_d2, CAT_d1, CAT_d2, dist_th, mag_th);
    dsi->sweep();

    const int qual = bnb_search_ml_table<SSR, 8> (*dsi, IN_d1, IN_d2, CAT_d1, CAT_d2, mag_th, lwbnd, gap, buckets, guessAndResult);
    delete dsi;

    result = centre(guessAndResult);
    return qual;
}



//Split by 2
//int reg::search::bnb_rsearch_3dof_s2_mcirc(const Matrix &M, const Matrix &B,
//                                          double th, int gap, int known_sol_qual,
//                                          int buckets, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS2 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new CircDataSetIndexation<SSR>(M, B, th);
//    dsi->sweep();

//    const int qual = bnb_search_table<SSR, 2> (
//                *dsi, known_sol_qual, gap,
//                buckets,
//                guessAndResult);
//    delete dsi;

//    result = Eigen::AngleAxisd(ssrAngle(guessAndResult), ssrAxis(guessAndResult));
//    return qual;
//}



//int reg::search::bnb_rsearch_3dof_s2_1kdt(const Matrix &M, const Matrix &B,
//                                          double th, int gap, int known_sol_qual,
//                                          int buckets, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS2 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new Rot1KDTDataSetIndexationRangeSearch<SSR>(M, B, th, true);
//    dsi->sweep();

//    const int qual = bnb_search_table<SSR, 2> (
//                *dsi, known_sol_qual, gap, buckets,
//                guessAndResult);
//    delete dsi;

//    //result = Eigen::AngleAxisd(ssrAngle(guessAndResult), ssrAxis(guessAndResult));
//    result = centre(guessAndResult);
//    return qual;
//}



// ------------------------------------
// -------- 1KDT ----------------------
// ------------------------------------

//int reg::search::bnb_rsearch_3dof_s8_1kdtnn(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new Rot1KDTDataSetIndexationNNSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_table<SSR, 8> (
//                *dsi, known_sol_qual, gap, buckets,
//                guessAndResult);
//    delete dsi;
//    result = centre(guessAndResult);
//    return qual;
//}

//int reg::search::bnb_rsearch_3dof_s8_1kdtnn_ml(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new Rot1KDTDataSetIndexationNNSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_ml_table<SSR, 8> (*dsi, known_sol_qual, gap,
//                                                  buckets, guessAndResult);
//    delete dsi;
//    result = centre(guessAndResult);
//    return qual;
//}


//int reg::search::bnb_rsearch_3dof_s8_1kdtrange(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int knownSolQual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new Rot1KDTDataSetIndexationRangeSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_table<SSR, 8> (
//                *dsi, knownSolQual, gap, buckets,
//                guessAndResult);
//    delete dsi;
//    result = centre(guessAndResult);
//    return qual;
//}

//int reg::search::bnb_rsearch_3dof_s8_1kdtrange_ml(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new Rot1KDTDataSetIndexationRangeSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_ml_table<SSR, 8> (
//                *dsi, known_sol_qual, gap,
//                buckets, guessAndResult);
//    delete dsi;
//    result = centre(guessAndResult);
//    return qual;
//}



// ------------------------------------
// -------- MKDT ----------------------
// ------------------------------------

//int reg::search::bnb_rsearch_3dof_s8_mkdtnn(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new RotMKDTDataSetIndexationNNSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_table<SSR, 8> (
//                *dsi, known_sol_qual, gap, buckets,
//                guessAndResult);
//    delete dsi;

//    result = centre(guessAndResult);
//    return qual;
//}

//int reg::search::bnb_rsearch_3dof_s8_mkdtnn_ml(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new RotMKDTDataSetIndexationNNSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_ml_table<SSR, 8> (*dsi, known_sol_qual, gap,
//                                                  buckets, guessAndResult);
//    delete dsi;

//    result = centre(guessAndResult);
//    return qual;
//}

//int reg::search::bnb_rsearch_3dof_s8_mkdtrange(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new RotMKDTDataSetIndexationRangeSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_table<SSR, 8> (
//                *dsi, known_sol_qual, gap, buckets,
//                guessAndResult);
//    delete dsi;

//    result = centre(guessAndResult);
//    return qual;
//}

//int reg::search::bnb_rsearch_3dof_s8_mkdtrange_ml(
//        const Matrix &M, const Matrix &B,
//        double th, int gap, int known_sol_qual,
//        int buckets, bool bo, Eigen::AngleAxisd &result)
//{
//    typedef RotationSearchSpaceRegion3DOFS8 SSR;

//    SSR guessAndResult(-PI, -PI, -PI, PI, PI, PI);
//    DataSetIndexation<SSR> *dsi = new RotMKDTDataSetIndexationRangeSearch<SSR>(M, B, th, bo);
//    dsi->sweep();

//    const int qual = bnb_search_ml_table<SSR, 8> (
//                *dsi, known_sol_qual, gap,
//                buckets, guessAndResult);
//    delete dsi;

//    result = centre(guessAndResult);
//    return qual;
//}
