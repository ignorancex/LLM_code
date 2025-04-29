/*
 * This file is part of litchi, a lightweight C++ library
 * for Minkowski analysis
 * 
 * Copyright (C) 2021-2023 Caroline Collischon <caroline.collischon@fau.de>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */


#include "tensorOperations.hpp"
#include "litchi_eat.hpp"

#include "eigen/Eigen/Dense"

#include "healpix_cxx/healpix_map.h"
#include "healpix_cxx/healpix_map_fitsio.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
int main ()
{
    
    const double pi = 3.1415926535897962643;

    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    
    Eigen::Matrix3d mehrabadimatrix{
        {3,1,0},
        {0,2,0},
        {0,0,7}
    };
    Eigen::EigenSolver<Eigen::Matrix3d> solver(mehrabadimatrix,false);
    auto EVvec = solver.eigenvalues();
    cout << EVvec << endl;
    
    cout << "Is minktensorStack nothrow move constructible? " << (std::is_nothrow_move_constructible<minkTensorStack>::value ? "yes" : "no") << endl;
    
    pointing A(pi/2,0);
    pointing B(pi/4,0);
    pointing C(pi/2,pi/4);
    pointing D(pi/2,pi/2);
    pointing E(pi/2,pi);
    
    pointing r(0.5, 0);
    pointing n(0.5, pi);
    
    minkTensorIntegrand asdf(0,2,0,A,B);
    Healpix_Map<double> map = read_Healpix_map_from_fits<double>("./COM_CMB_IQU-smica_2048_R3.00_hm1.fits", 1, 2);
    //Test ispolar
    minkmapSphere testmink(map,0,0,1,0.);
    normalHealpixInterface interface(testmink);
    fix_arr<int, 4> neighbors;
    fix_arr<double, 4> weight;
    pointing north(0,0);
    try {
        map.get_interpol(north,neighbors,weight);
    } catch (...) {
        cerr << "Interpolation of north pole RING failed in Healpix!\n";
        return 1;
    }
    for(uint i=0;i<neighbors.size();i++)
    {
        assert(interface.ispolar(neighbors[i])==1 && "ispolar RING north broken!");
    }
    pointing south(pi,0);
    south.normalize();
    try {
        map.get_interpol(south,neighbors,weight);
    } catch (...) {
        cerr << "Interpolation of south pole RING failed in Healpix!\n";
        return 1;
    }
    for(uint i=0;i<neighbors.size();i++)
    {
        assert(interface.ispolar(neighbors[i])==2 && "ispolar RING south broken!");
    }
    pointing random(4.76,1.345);
    random.normalize();
    try {
        map.get_interpol(random,neighbors,weight);
    } catch (...) {
        cerr << "Interpolation of random point RING failed in Healpix!\n";
        return 1;
    }
    for(uint i=0;i<neighbors.size();i++)
    {
        assert(!interface.ispolar(neighbors[i]) && "ispolar RING broken (false-positive)!");
    }
    
    
    try {
        map.get_interpol(north,neighbors,weight);
    } catch (...) {
        cerr << "Interpolation of north pole NEST failed in Healpix!\n";
        return 1;
    }
    for(uint i=0;i<neighbors.size();i++)
    {
        assert(interface.ispolar(neighbors[i])==1 && "ispolar NEST north broken!");
    }
    try {
        map.get_interpol(south,neighbors,weight);
    } catch (...) {
        cerr << "Interpolation of south pole NEST failed in Healpix!\n";
        return 1;
    }
    for(uint i=0;i<neighbors.size();i++)
    {
        assert(interface.ispolar(neighbors[i])==2 && "ispolar NEST south broken!");
    }
    random.normalize();
    try {
        map.get_interpol(random,neighbors,weight);
    } catch (...) {
        cerr << "Interpolation of random point NEST failed in Healpix!\n";
        return 1;
    }
    for(uint i=0;i<neighbors.size();i++)
    {
        assert(!interface.ispolar(neighbors[i]) && "ispolar NEST broken (false-positive)!");
    }
    
    
    //Test tensor functionality
    pointing nr(0.45*pi, 0.5*pi);
    minkTensorIntegrand mytensor(0,2,0,r,n);
    minkTensorIntegrand mytensor2(0,2,0,nr,nr);
    minkTensorIntegrand mytensor3(1,1,0,r,nr);
    //tensor2D testtensor(mytensor);
    assert( abs(mytensor.accessElement({1,1})-(pi*pi)) < 1e-13 && "minkTensorIntegrand::accessElement() broken!" );
    assert( abs(mytensor2.accessElement({0,0})-(0.45*pi*0.45*pi)) < 1e-13 && "minkTensorIntegrand::accessElement() broken!" );
    assert( abs(mytensor3.accessElement({1,0})-(0.5* (0.5*0.5*pi + 0*0.45*pi))) < 1e-13 && "minkTensorIntegrand::accessElement() broken!" );
    //assert( abs(testtensor.accessElement({0,1}) - (0.5*pi)) < 1e-13 && "tensor2D::accessElement() or assign() broken!");
    
    //testtensor = mytensor2;
    minkTensorStack zahlenfriedhof (mytensor+mytensor2);
    mytensor2.moveTo(r);
    assert( abs(zahlenfriedhof.accessElement({1,1})-mytensor.accessElement({1,1})-mytensor2.accessElement({1,1}))<1e-13 && "minkTensorStack sum broken!" );
    assert( abs(zahlenfriedhof.accessElement({0,0})-mytensor.accessElement({0,0})-mytensor2.accessElement({0,0}))<1e-13 && "minkTensorStack sum broken!" );
    zahlenfriedhof = mytensor + 3*mytensor2;
    assert( abs(zahlenfriedhof.accessElement({1,1})-mytensor.accessElement({1,1})-3*mytensor2.accessElement({1,1}))<1e-13 && "minkTensorStack product or sum broken!" );
    
    
    int drei = 3;
    minkTensorIntegrand mytensor4(3,4,0,r,n);
    minkTensorIntegrand mytensor5(3,4,0,nr,nr);
    auto newtensor = drei*(3*(mytensor4 + mytensor5));
    mytensor5.moveTo(r);
    assert( abs( newtensor.accessElement({1,1,0,1,1,0,0}) -(9*(mytensor4.accessElement({1,1,0,1,1,0,0}) + mytensor5.accessElement({1,1,0,1,1,0,0}))) ) < 1e-13 && "'auto newtensor = drei*(3*(mytensor4 + mytensor5));' does not work! " );
    
    /*
    tensor2D testTensor(2,1,0);
    testTensor.writeElement({0,0,1},5);
    assert(abs(testTensor.accessElement({0,0,1})-5)<1e-14 && "testTensor::writeElement broken!");
    */
    
    zahlenfriedhof = mytensor + mytensor2;
    cout << "trace additivity check deactivated" << '\n';
    //assert( abs(trace(zahlenfriedhof) - (trace(mytensor) +trace(mytensor2))) < 1e-13 && "trace not additive!" );
    assert( abs(trace(zahlenfriedhof) - (zahlenfriedhof.accessElement({1,1})*pow(sin(zahlenfriedhof.r.theta),2) + zahlenfriedhof.accessElement({0,0}))) < 1e-13 && "trace not working! (check normalization?)" );
    
    minkTensorIntegrand myVectorTheta(0,1,1,pointing(0.5*pi,0),pointing(1,0));
    assert(abs(eigenVecDir(myVectorTheta))<1e-12 && "eigenVecDir not working (theta) for rank 1!");
    minkTensorIntegrand myVectorPhi  (0,1,1,pointing(0.5*pi,0),pointing(0,1));
    assert(abs(eigenVecDir(myVectorPhi)-pi/2)<1e-12 && "eigenVecDir not working (phi, equator) for rank 1!");
    myVectorPhi.moveTo(pointing(1.0,0));
    assert(abs(eigenVecDir(myVectorPhi)-pi/2)<1e-4 && "eigenVecDir not working (phi, lat) for rank 1!");
    minkTensorIntegrand myVectorMix  (0,1,1,pointing(0.5*pi,0),pointing(1,1));
    assert(abs(eigenVecDir(myVectorMix)-pi/4)<1e-12 && "eigenVecDir not working (mix, equator) for rank 1!");
    myVectorMix.moveTo(pointing(1.25,0));
    assert(abs(eigenVecDir(myVectorMix)-pi/4)<1e-4 && "eigenVecDir not working (mix, lat) for rank 1!");
    
    
    //Test minkTensorStack
    minkTensorStack mystack(mytensor4);
    auto otherStack = mystack + minkTensorStack(mytensor5);
    assert( abs( otherStack.accessElement({1,1,0,1,1,0,0}) -(mystack.accessElement({1,1,0,1,1,0,0}) + mytensor5.accessElement({1,1,0,1,1,0,0})) ) < 1e-13 && "minkTensorStack addition does not work! " );
    
    auto otherStack2 = drei*(3*(mystack + minkTensorStack(mytensor5)));
    assert( abs( otherStack2.accessElement({1,1,0,1,1,0,0}) -(9*(mystack.accessElement({1,1,0,1,1,0,0}) + mytensor5.accessElement({1,1,0,1,1,0,0}))) ) < 1e-13 && "'auto otherStack2 = drei*(3*(mystack + minkTensorStack(mytensor5)));' does not work! " );
    
    
    minkTensorIntegrand asdfTensor (0,1,1,pointing(pi/2,0.01) ,pointing(1,1));
    minkTensorIntegrand asdfTensor2(0,1,1,pointing(pi/2,2*pi-0.01),pointing(1,1));
    auto asdfSum = asdfTensor2 + asdfTensor;
    assert( abs(asdfSum.accessElement({0}) - asdfSum.accessElement({1}))<1e-4 && "sum of two tensors left and right of meridian broken" );
    
    //Test vector evq, should be length
    pointing n1(0.1,0.1), n2(-0.1,-0.1);
    normalizeVectorOnSphere(n1,pi/3);
    normalizeVectorOnSphere(n2,pi/3);
    minkTensorIntegrand integ1(0,1,1,pointing(pi/3,0.1),n1),integ2(0,1,1,pointing(pi/3,0.1),n2);
    minkTensorStack vectorstack = integ1*0.1;
    assert( abs(eigenValueQuotient(vectorstack)-0.1) < 1e-12 && "vector evq length calculation broken 1");
    vectorstack = integ2*0.2;
    assert( abs(eigenValueQuotient(vectorstack)-0.2) < 1e-12 && "vector evq length calculation broken 2");
    vectorstack = integ1 + integ2;
    assert( abs(eigenValueQuotient(vectorstack)) < 1e-12 && "sum of vector evq length calculation broken");
    
    
    
    
    //Test actual map, trace of W^(0,2)_1 should be equal to boundary length
    Healpix_Map<double> degradedMap(128, map.Scheme(), SET_NSIDE);
    degradedMap.Import_degrade(map);
    map = degradedMap;
    
    std::vector<double> thresholds = makeIntervals_lin(0, 1e-5, 3);
    std::vector<minkmapSphere> mapsScalar;
    std::vector<minkmapSphere> mapsTensor;
    for(double thresh : thresholds)
    {
        mapsScalar.push_back(minkmapSphere(map, 0, 0, 1, thresh));
        mapsTensor.push_back(minkmapSphere(map, 0, 2, 1, thresh));
    }
    
    std::vector<int> pixnums {68, 1933, 1235, 563, 1023, 585, 7169, 1123};
    for( int pixnum : pixnums)
    {
        minkmapStack sumOfMapsS(mapsScalar);
        minkmapStack sumOfMapsT(mapsTensor);
        
        double pix1S = mapsScalar.at(0).at(pixnum).accessElement({});
        minkTensorStack pix1T = mapsTensor.at(0).at(pixnum);
        cout << "Scalar   " << "tr(tensor) " << "relative difference, pixnum: " << pixnum << endl;
        cout << pix1S << " " << trace(pix1T) << " " << abs(pix1S-trace(pix1T))/max(pix1S, trace(pix1T) ) << endl;
        if(max(pix1S, trace(pix1T)) ) assert( abs(pix1S-trace(pix1T))/max(pix1S, trace(pix1T) )<1e-12 && "Trace of (0,2,1) and boundary different at thresh 0" );
        else assert( abs(pix1S-trace(pix1T))<1e-12 && "Trace of (0,2,1) and boundary different at thresh 0" );
        
        double pix2S = mapsScalar.at(1).at(pixnum).accessElement({});
        minkTensorStack pix2T = mapsTensor.at(1).at(pixnum);
        cout << pix2S << " " << trace(pix2T) << " " << abs(pix2S-trace(pix2T))/max(pix2S, trace(pix2T) ) << endl;
        if(max(pix2S, trace(pix2T)) ) assert(  abs(pix2S-trace(pix2T))/max(pix2S, trace(pix2T) )<1e-12 && "Trace of (0,2,1) and boundary different at thresh > 0" );
        else assert( abs(pix2S-trace(pix2T))<1e-12 && "Trace of (0,2,1) and boundary different at thresh > 0" );
        
        double pix3S = sumOfMapsS.at(pixnum).accessElement({});
        minkTensorStack pix3T = sumOfMapsT.at(pixnum);
        cout << pix3S << " " << trace(pix3T) << " " << abs(pix3S-trace(pix3T))/max(pix3S, trace(pix3T) ) << endl;
        if(max(pix3S, trace(pix3T)) ) assert( abs(pix3S-trace(pix3T))/max(pix3S, trace(pix3T) )<1e-12 && "Trace of (0,2,1) and boundary different in sum of maps" );
        else assert( abs(pix3S-trace(pix3T))<1e-12 && "Trace of (0,2,1) and boundary different in sum of maps" );
        
        
        const normalHealpixInterface interfaceS(sumOfMapsS);
        const normalHealpixInterface interfaceT(sumOfMapsT);
        double pix4S = interfaceS.at(pixnum).accessElement({});
        minkTensorStack pix4T = interfaceT.at(pixnum);
        //tensor2D pix4Tfail = (tensor2D) sumOfMapsT.at(67);
        
        cout << pix4S << " " << trace(pix4T) << " " << abs(pix4S-trace(pix4T))/max(pix4S, trace(pix4T) ) << endl;
        if(max(pix4S, trace(pix4T)) ) assert( abs(pix4S-trace(pix4T))/max(pix4S, trace(pix4T) )<1e-12 && "Trace of (0,2,1) and boundary different in Interface" );
        else assert( abs(pix4S-trace(pix4T))<1e-12 && "Trace of (0,2,1) and boundary different in interface" );
        
        auto neighbors = mapsScalar.at(0).minkmapPixelNeighbors(pixnum);
        double maxdist = 2*map.max_pixrad();
        assert(arclength(map.pix2ang(neighbors.at(0)),map.pix2ang(neighbors.at(1))) < maxdist && "Distance between neighbors too large!" );
        assert(arclength(map.pix2ang(neighbors.at(0)),map.pix2ang(neighbors.at(2))) < maxdist && "Distance between neighbors too large!" );
        assert(arclength(map.pix2ang(neighbors.at(0)),map.pix2ang(neighbors.at(3))) < maxdist && "Distance between neighbors too large!" );
        
        cout << "Testing EVQ \n";
        auto test = eigenValueQuotient(pix1T);
        test = eigenValueQuotient(pix2T);
        test = eigenValueQuotient(pix3T);
        test = eigenValueQuotient(pix4T);
        
        
        cout << "Testing EVD \n";
        test = eigenVecDir(pix1T);
        test = eigenVecDir(pix2T);
        test = eigenVecDir(pix3T);
        test = eigenVecDir(pix4T);
    }
    
    paramStruct testParams;
    testParams.forceOutname = true;
    testParams.mint = -1e-4, testParams.maxt = 1e-4, testParams.numt = 13;
    testParams.rankA = 0, testParams.rankB = 2, testParams.curvIndex = 1;
    testParams.Nside = 128, testParams.smooth = 8;
    makeMinkmap("./COM_CMB_IQU-smica_2048_R3.00_hm1.fits",testParams,"testmap.fits");
    
    return 0;
}
