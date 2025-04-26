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
#include "registration.h"
#include "state.h"
#include "state_priority_hashtable.h"
#include "data_indexation.h"
#include "reg_common.h"
#include "search.h"
#include "translation_kdtree_data_set_indexation.h"

using namespace reg;
using namespace reg::search;

typedef TranslationSearchSpaceRegion3DOF SSR;

typedef KDTreeSingleIndexAdaptor<
L2_Simple_Adaptor<double, PointCloud > ,
PointCloud, 3 > KdTree;


int tsearch( Matrix4X *M, TranslationKDTreeDataSetIndexation<SSR> &dsi,
            int lwbnd, int buckets, SSR ssr,
            Vector3 &result)
{
    SSR guessAndResult = ssr;
    dsi.setM(M);
    const int qual = reg::search::bnb_search_table<SSR, 8> (
                dsi, lwbnd, 0, buckets, guessAndResult);
    
    result = ssrCentre(guessAndResult);
    return qual;
}


//int evalQual(const pcl::KdTreeFLANN<pcl::PointXYZ> &treeB, const Matrix &M, double th)

//int evalQual(const KdTree &treeB, const Matrix3X &M, double th)
//{
//    mxAssert(th>0, "invalid th");
//  
//    int qual=0;
//    
//    std::vector<std::pair<size_t,double> > indices_dists;
//    RadiusResultSet<double,size_t> resultSet(th*th,indices_dists);
//    
//
//    const int msize = M.cols();
//
//    for (int i=0; i<msize; i++)
//    {
//        double *p = M.x + 3*i;
//        
////        Eigen::Vector3d ep = M.col(i);
////        pcl::PointXYZ p(ep(0), ep(1), ep(2));
////        if (treeB.radiusSearch(p, th, nbIdxB, nbSqrDistB, 1))
//        if (treeB.findNeighbors(resultSet, p, nanoflann::SearchParams(32,0,false)) ) //32 is the default value
//        {
//            qual++;
//        }
//    }
//    return qual;
//}


Matrix4X transform( Matrix4X &x, Transform3 &tform)
{
    //TM = guessAndResult * TM;
    
    //y = tform*x;
    

    char *chn = (char *)"N";
    double alpha = 1.0, beta = 0.0;
    ptrdiff_t m = 4;
    ptrdiff_t n =x.cols();
    
//    mxAssert(n>0,"");
    
    Matrix4X y(n);
    
    
    dgemm(chn, chn, &m, &n, &m, &alpha, tform.x, &m, x.x, &m, &beta, y.x, &m);

    
    return y;
}


int reg::search::local_search_6dof(
        const Matrix3X &M, const Matrix3X &B,
        double th, int known_sol_qual, SSR tr_box,
        int(*bnb_rsearch_3dof)(const Matrix3X&, const Matrix3X&,
                    double,int, int, int, AxisAngle&),
        Transform3 &guessAndResult)
{
    int qual;
    int rqual, tqual;
    const int msize = (int)M.cols();
    const int buckets = MAX((int)msize/10,10);

    //Create DSI
    TranslationKDTreeDataSetIndexation<SSR> dsi(M, B, th);
    dsi.sweep();

    Transform3 transf_t;
    Transform3 transf_r;

    int num_of_iter=0;

    // Apply Initial guess
    Matrix4X TM(msize);
    for(size_t i=0; i<msize; i++)
    {
        TM.x[4*i]   = M.x[3*i];
        TM.x[4*i+1] = M.x[3*i+1];
        TM.x[4*i+2] = M.x[3*i+2];
        TM.x[4*i+3] = 0;
    }
    
//    std::cout<<"TM = [...\n";
//    for(size_t i; i<msize; i++)
//    {
//        std::cout<<TM.x[4*i] <<" "<< TM.x[4*i+1] <<" "<< TM.x[4*i+2] <<" "<< TM.x[4*i+3]<<std::endl;
//    }
//    std::cout<<" ]\n";
//    
//    std::cout<<"G = [...\n";
//    for(size_t i; i<msize; i++)
//    {
//        std::cout<<TM.x[4*i] <<" "<< TM.x[4*i+1] <<" "<< TM.x[4*i+2] <<" "<< TM.x[4*i+3]<<std::endl;
//    }
//    std::cout<<" ]\n";
//    
    
    

    
    //TM = guessAndResult * TM;
    TM = transform(TM,guessAndResult);
    
    //TODO: avoid to use aux. matrix TM3
    Matrix3X TM3(msize);
    
    transf_t = Transform3();
    transf_r = Transform3();
    qual = known_sol_qual;

    // ---------------------------------------
    // Do R->T iterations
    // ---------------------------------------
    while(true)
    {
        // ----------------------------
        // Rotation
        // -----------------------------
        AxisAngle rsearch_sol;
        
        //Copy TM to 3 dim...
        for(size_t i=0; i<msize; i++)
        {
            TM3.x[3*i]   = TM.x[4*i];
            TM3.x[3*i+1] = TM.x[4*i+1];
            TM3.x[3*i+2] = TM.x[4*i+2];
        }
        
        rqual = bnb_rsearch_3dof(TM3, B, th, 0, qual, buckets, rsearch_sol);
     
        std::cout<< "local method: current qual = "<< rqual <<std::endl;
//        std::cout<< "axis = "<< rsearch_sol.axis().transpose() << std::endl;
//        std::cout<< "angle = "<<rsearch_sol.angle()<<std::endl;

        //update TM
        Matrix3 R;
        fromAxisAngle(R,rsearch_sol); //rsearch_sol.toRotationMatrix();
        
        Transform3 tformR(R);
        
        //TM = tformR * TM;
        TM = transform(TM, tformR);
        transf_r = tformR*transf_t;

        // ----------------------------
        // Translation
        // -----------------------------
        Vector3 tsearch_sol(0,0,0);
        tqual = tsearch(&TM, dsi, rqual, buckets, tr_box, tsearch_sol);
        if(!(tqual>qual))
        {
            break;
        }

        qual = tqual;
        //Update TM
        Transform3 tr (tsearch_sol);
        //tr = tsearch_sol;
        //TM = tr*TM;
        TM=transform(TM,tr);
        transf_t = tr*transf_r;

        std::cout<<"local search: "<< num_of_iter <<":  rqual " << rqual << "  tqual "<< tqual <<std::endl;
        num_of_iter++;
    }

    guessAndResult = transf_t*guessAndResult;

    std::cout<< num_of_iter <<":  Final qual = " << qual <<std::endl;
    return qual;
}
