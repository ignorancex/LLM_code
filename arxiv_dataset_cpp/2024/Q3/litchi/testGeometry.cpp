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


#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>
#include <limits>
typedef std::numeric_limits< double > dbl;

#include "geometryHelpers.hpp"

using namespace std;
int main ()
{
    
    const double pi = 3.1415926535897962643;
    //Test midpoint, geometry helpers
    pointing A(pi/2,0);
    pointing B(pi/4,0);
    pointing C(pi/2,pi/4);
    pointing D(pi/2,pi/2);
    pointing E(pi/2,pi);
    
    
    assert( abs(midpoint(A,D).theta-pi/2)<1e-12 && abs(midpoint(A,D).phi-pi/4)<1e-12 && "midpoint broken!");
    assert( abs(midpoint(C,C).theta-C.theta)<1e-13 && abs(midpoint(C,C).phi-C.phi)<1e-13 && "midpoint broken when entering same vector twice!");
    assert( abs(midpoint(A,B).theta*(1/pi)-0.375)<1e-13 && abs(midpoint(A,B).phi)<1e-13 && "midpoint broken!");
    
    assert( abs(arclength(A,B)-pi/4)<1e-13 && abs(arclength(A,C)-pi/4)<1e-13 && "arclength broken!" );
    
    
    double theta = pi/2, phi = 0, epsilon = 1e-5;
    assert( abs(arclength(pointing(theta,phi),pointing(theta+epsilon,phi)) - epsilon)<1e-12 && "arclength broken! (small distance theta equator)" );
    assert( abs(arclength(pointing(theta,phi),pointing(theta,phi+epsilon)) - epsilon)<1e-12 && "arclength broken! (small distance phi equator)" );
    
    theta = pi/6, phi = 0.2*pi/180;
    cout.precision(dbl::max_digits10);
    assert( abs(arclength(pointing(theta,phi),pointing(theta,phi+epsilon))/ sin(theta) - epsilon )<1e-12 && "arclength broken! (small distance phi pi/8)" );
    assert( abs(arclength(pointing(theta,phi),pointing(theta+epsilon,phi)) - epsilon)<1e-12 && "arclength broken! (small distance theta pi/8)" );
    
    
    
    flipPointing(A);
    assert( abs(A.theta-pi/2) < 1e-12 && abs(A.phi-pi) < 1e-12 && "flipPointing broken!" );
    flipPointing(A);
    pointing n1 = getN_cartesian(B,A);
    assert( abs(n1.theta) < 1e-12 &&  "getN_cartesian broken!" );
    
    assert( abs(sphereArea(A, B, C) - sphereArea(A,C,B)) < 1e-13 && abs(sphereArea(A, B, C) - sphereArea(B,C,A)) <1e-13 && abs(sphereArea(A,B,C)-sphereArea(B,A,C))<1e-13 && "sphereArea not commutative!");
    assert( abs(sphereArea(A,D,pointing(0,0))/pi - 0.5) < 1e-12 && "sphereArea broken!" );
    vec3 axis(1,0,0);
    vec3 input(0,1,0);
    vec3 testvec = rotateAroundAxis(axis, input, pi/2);
    assert((abs(testvec.x)<1e-14 && abs(testvec.y)<1e-14 && abs(testvec.z-1)<1e-14) && "rotateAroundAxis() broken!");
    
    
    //Test more pointing-related functions
    pointing r(0.5, 0);
    pointing n(0.5, pi);
    
    pointing interpolation  = interpPointing(r, 1.5, n, 0.5, 1);
    assert(abs(interpolation.theta)<1e-12 && "interpPointing() broken!");
    interpolation  = interpPointing(n, 0.5, r, 1.5, 1);
    assert(abs(interpolation.theta)<1e-12 && "interpPointing() broken!");
    interpolation = interpPointing(pointing(pi/2,-0.01), 0.5, pointing(pi/2, 0.01), 1.5, 1);
    assert(abs(interpolation.phi)<1e-12 && "interpPointing() meridian negative phi broken!");
    interpolation = interpPointing(pointing(pi/2,2*pi-0.01), 0.5, pointing(pi/2, 0.01), 1.5, 1);
    assert(abs(interpolation.phi)<1e-12 && "interpPointing() meridian 2pi broken!");
    interpolation = interpPointing( pointing(pi/2, 0), 0.5, pointing(pi/2-1,0), 2, 1 );
    assert( abs(interpolation.theta-(pi/2-(1./3)) )<1e-12 && "interpPointing() 1/3 broken" );
    
    theta = pi/6, phi = 0.1*pi/180;
    interpolation = interpPointing( pointing(theta,phi), 1.5, pointing(theta+2*epsilon,phi), 0.5, 1 );
    assert( abs(interpolation.theta - (theta+epsilon))<1e-12 && "interpPointing() pi/6 small distance theta");
    interpolation = interpPointing( pointing(theta,phi), 1.5, pointing(theta,phi+2*epsilon), 0.5, 1 );
    cout << (interpolation.phi - (phi+epsilon)) << endl;
    assert( abs(interpolation.phi - (phi+epsilon))<1e-12 && "interpPointing() pi/6 small distance phi");
    
    
    pointing eq1(pi/2,0);
    pointing eq2(pi/2+0.01, 0.01);
    pointing eqn = getN_rotation(eq1,eq2);
    
    pointing pole1(rotateAroundAxis(vec3(0,1,0),eq1.to_vec3(),pi/2-0.01));
    pointing pole2(rotateAroundAxis(vec3(0,1,0),eq2.to_vec3(),pi/2-0.01));
    pointing polen1 = parallelTransport(eq1,pole1,eqn);
    pointing polen2 = getN_rotation(pole1,pole2);
    //Parallel transport with one step not that accurate, but phi here is >67. With recursion transported vector gets more similar to newly calculated one
    assert( abs(polen1.theta-polen2.theta)<1e-4 && abs(polen1.phi-polen2.phi)<1e-4  && "getN_rotation or parallelTransport broken!");
    
    
    pointing leftOfZero(pi*0.4,0.01);
    pointing leftBelow(pi*0.4+0.01,-0.0);
    leftOfZero.normalize();
    leftBelow.normalize();
    pointing rightOfZero(pi*0.4,-0.01);
    pointing rightBelow(pi*0.4+0.01,-0.02);
    //rightOfZero.normalize();
    //rightBelow.normalize();
    pointing leftn = getN_rotation(leftOfZero,leftBelow);
    pointing transpn = parallelTransport(leftOfZero,rightOfZero,leftn);
    pointing rightn = getN_rotation(rightOfZero,rightBelow);
    cout << " left " << leftn;
    cout << "transp " << transpn;
    cout << "right " << rightn;
    
    
    //Test parallel transport of vectors
    pointing myN(1,0);
    pointing myNphi (0,1);
    pointing moved = parallelTransport(pointing(pi/2,0), pointing(pi/2,-pi/4), myN);
    pointing movedphi = parallelTransport(pointing(pi/2,0.01), pointing(pi/2,-0.01), myNphi);
    assert( abs(moved.theta-1) < 1e-6 && abs(moved.phi) < 1e-3 && "parallelTransport thetavector broken (phi direction)!" );
    assert( abs(movedphi.theta) < 1e-6 && abs(movedphi.phi-1) < 1e-3 && "parallelTransport phivector broken (phi direction)!" );
    
    moved = parallelTransport(pointing(pi/2,0), pointing(pi/4,0), myN);
    movedphi = parallelTransport(pointing(pi/2,0), pointing(pi/4,0), myNphi);
    assert( abs(moved.theta-1) < 1e-12 && abs(moved.phi) < 1e-3 && "parallelTransport thetavector broken (theta direction)!" );
    assert( abs(movedphi.theta) < 1e-4 && abs( movedphi.phi-1/(pow(sin(pi/4),1)) ) < 1e-6 && "parallelTransport phivector broken (theta direction)!" );
    
    cout << "Is pair of double and pointing nothrow move constructible? " << (std::is_nothrow_move_constructible<std::pair<pointing,double>>::value ? "yes" : "no") << endl;
    
    return 0;
}
