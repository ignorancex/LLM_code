//
//  truncated_exponential.cpp
//  truncated exponential sampling
//
//  Created by Emanuel Casiano-Diaz on 8/22/20.
//  Copyright © 2020 Emanuel Casiano-Díaz. All rights reserved.
//

// #include "pimc.hpp"
// #include "cxxopts.hpp"
// #include <assert.h>
// #include "uuid.hpp"

// #ifndef pimc_hpp
// #define pimc_hpp

#include <stdio.h>
#include<iostream>
#include<vector>
#include<boost/random.hpp>
#include<cmath>
#include<chrono>
#include<iomanip>  // for std::setprecision
#include <fstream>
#include <cstdlib> // for exit function
// #include "RNG.h"
#include<sstream>
#include<string.h>

#include <boost/math/special_functions/lambert_w.hpp>
using boost::math::lambert_w0;
using boost::math::lambert_wm1;

using namespace std;
using namespace std::chrono;

/*-------------------------- Function Definitions ----------------------------*/

double infer_tau1(double tau_old,double a, double b, double c, 
                    boost::random::mt19937 &rng){

    /* samples tau1 from marginalized distribution P(tau1), which is obtained by
    integrating tau2 dependence from joint truncated exponential distribution
    P(tau1,tau2). Constraint: a < tau1 < tau2 < b */

    double u,Z,P,arg,atol,tau,B,y,tau_wm1,tau_w0;

    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);
    y = rnum(rng);

    // Compute normalization of truncated exponential dist.
    Z = (1/(c*c)) * (exp(-c*(b-a)) - 1) + (1/c) * (b - a);
    B = -exp(-c*b);
    u = y*c*c*Z + B*exp(c*a) + c*a;

    // Determine LambertW branch & compute tau
    // arg = max(-1/exp(1), A*c*exp(c*y));
    arg = max(-1/exp(1), B*exp(u));
    arg = B*exp(u);
    // cout << -1/exp(1) << " " << A*c*exp(c*y) << endl;
    if (c > 0){ // k = 0 branch
        tau = (1/c)*(u-lambert_w0(arg));
    }
    else {      // k = -1 branch
        tau = (1/c)*(u-lambert_wm1(arg));
    }

    // Check with specific x values
    // double F;
    // F = (1/Z) * ((1/c) * (exp(c*(b-a))-exp(c*(b-tau)))-(tau-a));
    // atol = 1.0e-10;
    // assert(abs(y-(A*exp(-c*tau)-tau)) <= atol);
    // assert(abs(F-x) <= atol);
    // assert(a-atol <= tau <= b+atol);

    return tau;
}

/*----------------------------------------------------------------------------*/

double infer_tau1_with_rejection(double tau_old,double a, double b, double c, 
                    boost::random::mt19937 &rng){

    /* samples tau1 from marginalized distribution P(tau1), which is obtained by
    integrating tau2 dependence from joint truncated exponential distribution
    P(tau1,tau2). Constraint: a < tau1 < tau2 < b */

    double y,Z,P,arg,atol,tau,A,x,Pmax,tau_new,x1,x2,u;

    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);
    x1 = rnum(rng);
    x2 = rnum(rng);

    // Propose new random time in [a,b)
    tau_new = a + x1*(b-a);

    // Compute normalization of marginalized distribution P(x1)
    Z = (1/(c*c)) * (exp(-c*(b-a)) - 1) + (1/c) * (b - a);

    // Compute marginalized probability density P(x1)
    P = (1/(c*Z))*(1-exp(-c*(b-tau_new)));
    
    // Get maximum value of distribution (to rescale uniform dist.)
    // Pmax = (1/(c*Z))*(exp(c*(b-a))-1);
    Pmax = (1/(c*Z))*(1-exp(-c*(b-a)));

    // Rescaled uniform number
    u = x2*Pmax;

    if (u <= P){
        // cout << "infer tau1 accepted" << endl;
        return tau_new;
    }
    else {
        // cout << "infer tau1 rejected" << endl;
        return tau_old;
    }
}

/*----------------------------------------------------------------------------*/

double shifted_infer_tau1(double x,double a, double b, double c){

    double y,Z,P,arg,atol,tau,A,a_pre_shift,b_pre_shift,inverse_c;

    /* samples tau1 from marginalized distribution P(tau1), which is obtained by
    integrating tau2 dependence from joint truncated exponential distribution
    P(tau1,tau2). Constraint: a < tau1 < tau2 < b */

    // Save original values of the upper bounds
    a_pre_shift = a;
    b_pre_shift = b;

    // Shift lower and upper bounds
    a = a-b;
    b = 0;

    // Compute normalization of truncated exponential dist.
    inverse_c = 1/c;
    Z = inverse_c * (exp(c*(-a)) - 1) + a;
    
    //
    y = Z*x - inverse_c*exp(-c*a) - a;

    //
    A = -inverse_c;

    // Determine LambertW branch & compute tau
    arg = max(-1/exp(1), A*c*exp(c*y)); // avoids arg<-1/e due to finite prec.
    if (c < 0){ // k = 0 branch
        tau = inverse_c*lambert_w0(arg)-y;
    }
    else {      // k = -1 branch
        tau = inverse_c*lambert_wm1(arg)-y;
    }

    tau += b_pre_shift;
    // cout << tau << endl;

    // Unit test.
    // Check with specific x values
    // double F;
    // F = (1/Z) * ((1/c) * (exp(c*(b-a))-exp(c*(b-tau)))-(tau-a));
    // atol = 1.0e-10;
    // assert(abs(y-(A*exp(-c*tau)-tau)) <= atol);
    // assert(abs(F-x) <= atol);
    // assert(a-atol <= tau);
    // assert(tau <= b+atol);

    return tau;
}

/*----------------------------------------------------------------------------*/

double infer_tau2_with_rejection(double tau_old,
                                 double a, double b, double c,
                                 boost::random::mt19937 &rng){

    /* Sample tau2 from simple truncated exponential distribution */

    double Z,P,Pmax,tau_new,x1,x2,u;

    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);

    x1 = rnum(rng);
    x2 = rnum(rng);

    // Propose new random time in [a,b)
    tau_new = a + x1*(b-a);

    // Compute normalization constant
    Z = (1-exp(-c*(b-a)))/c;

    // Compute probability density
    P = (1/Z)*exp(-c*(tau_new-a));

    // Get maximum value of truncexpon (to rescale uniform dist.)
    if(c>0)
        Pmax = (1/Z);
    else
        Pmax = (1/Z)*exp(-c*(b-a));
    // the conditions above are ok. I checked them for simple truncexpon.

    // Rescaled uniform number
    u = x2*Pmax;

    if (u <= P){
        // cout << "infer tau2 accepted" << endl;
        return tau_new;
    }
    else {
        // cout << "infer tau2 rejected" << endl;
        return tau_old;
    }
}

/*----------------------------------------------------------------------------*/

void infer_taus_with_rejection_bad(double &tau1_old, double &tau2_old,
                                 double a, double b, double c,
                                 boost::random::mt19937 &rng){

    /* Sample tau2 from simple truncated exponential distribution */

    double Z,P,Pmax,tau_new,x1,x2,u,tau1_new,tau2_new,Z1,Z2,P1,P2,x3;

    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);

    x1 = rnum(rng);
    x2 = rnum(rng);
    x3 = rnum(rng);

    // Propose new random time in [a,b)
    tau1_new = a + x1*(b-a);

    // Compute normalization of marginalized distribution P(x1)
    Z1 = (1/(c*c))*(exp(c*(b-a))-c*(b-a)-a);

    // Compute marginalized probability density P(x1)
    P1 = (1/(c*Z1))*(exp(c*(b-tau1_new))-1);

    // Propose new random time in [a,b)
    tau2_new = tau1_new + x2*(b-tau1_new);

    // Compute normalization constant
    Z2 = (exp(c*(b-tau1_new))-1)/c;

    // Compute probability density
    P2 = (1/Z2)*exp(c*(tau2_new-tau1_new));

    // Get maximum value of truncexpon (to rescale uniform dist.)
    if(c<0)
        Pmax = (1/Z);
    else
        Pmax = (1/Z)*exp(c*(b-a));
    // the conditions above are ok. I checked them for simple truncexpon.
    Pmax = 3;
    cout << P1*P2 << endl;

    // Rescaled uniform number
    u = x3*Pmax;

    if (u <= P1*P2){
        // cout << "infer tau2 accepted" << endl;
        tau1_old = tau1_new;
        tau2_old = tau2_new;
    }
    else {
        // cout << "infer tau2 rejected" << endl;
    }
}

/*----------------------------------------------------------------------------*/

void infer_taus_with_rejection(double &tau1_old, double &tau2_old,
                                 double a, double b, double c,
                                 boost::random::mt19937 &rng){

    /* Sample tau2 from simple truncated exponential distribution */

    double Z,P,Pmax,tau_new,x1,x2,u,tau1_new,tau2_new,Z1,Z2,P1,P2,x3;

    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);

    x1 = rnum(rng);
    x2 = rnum(rng);
    x3 = rnum(rng);

    // Propose new random time in [a,b)
    tau1_new = a + x1*(b-a);
    tau2_new = tau1_new + x2*(b-tau1_new);

    // Compute normalization of joint distribution P(x1,x2)
    Z = (1/(c*c))*(exp(c*(b-a))-c*(b-a)-1);

    // Compute joint probability density P(x1,x2)
    double factor;
    if(tau1_new<tau2_new){factor=1;}
    else{factor=0;}
    P = factor*(1/Z)*(exp(c*(tau2_new-tau1_new)));

    // Get maximum value of truncexpon (to rescale uniform dist.)
    if(c<0)
        Pmax = (1/Z);
    else
        Pmax = (1/Z)*exp(c*(b-a));
    // the conditions above are ok. I checked them for simple truncexpon.
    Pmax = 3;
    cout << P << endl;

    // Rescaled uniform number
    u = x3*Pmax;

    if (u <= P){
        // cout << "infer tau2 accepted" << endl;
        tau1_old = tau1_new;
        tau2_old = tau2_new;
    }
    else {
        // cout << "infer tau2 rejected" << endl;
    }
}

/*----------------------------------------------------------------------------*/

double infer_tau2(double tau_old,
                                 double a, double b, double c,
                                 boost::random::mt19937 &rng){

    /* Sample tau2 from simple truncated exponential distribution */

    double tau,Z,P,Pmax,tau_new,x,u;

    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);
    x = rnum(rng);

    /* ---- */
    // Sample the new time of the worm end from truncated exponential dist.
    /*:::::::::::::::::::: Truncated Exponential RVS :::::::::::::::::::::::::*/
    Z = (1-exp(-c*(b-a)))/c;
    tau = a - log(1-Z*x*c)  / c;
    // cout << Z << endl;
    /*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/

    return tau;
}

/*----------------------------------------------------------------------------*/
// Main
int main(int argc, char** argv){

    double a,a_new,b,b_new,c,x,x2,tau1,tau2,arg1,arg2,arg3,arg0,tau,u,
    tau1_new,tau2_new;
    int constexpr num_samples = 1'000'000;

    int w0_ctr = 0;
    int wm1_ctr = 0;

    // Initialize vectors that will store tau2 samples from simple dist.
    vector<double> samples0(num_samples,0);
    vector<double> samples1(num_samples,0);


    // Initialize vectors that will store tau1,tau2 samples from joint dist.
    vector<double> samples2(num_samples,0);
    vector<double> samples3(num_samples,0);

    vector<double> samples4(num_samples,0);
    vector<double> samples5(num_samples,0);

    // Initialize a Mersenne Twister RNG
    int seed = 2001;
    boost::random::mt19937 rng(seed);

    // Create a uniform distribution with support: [0.0,1.0)
    boost::random::uniform_real_distribution<double> rnum(0.0,1.0);

    // Declare variables related 
    auto start = high_resolution_clock::now();
    auto end = high_resolution_clock::now();
    auto elapsed_time = duration_cast<nanoseconds>(end - start);
    double duration = elapsed_time.count() * 1e-9;

    // set parameters
    a = 0.1;  // lower bound
    b = 1.5;  // upper bound
    c = 2.0; // exponential decay

    // Part I: Sampling from simple (1d) truncated exponential distribution

    // Direct
    tau = a + rnum(rng)*(b-a); // Initialize tau to random value in interval
    for (int i=0;i<num_samples;i++){
        tau = infer_tau2_with_rejection(tau,a,b,c,rng);
        samples0[i] = tau;
    }

    // Rejection
    tau = a + rnum(rng)*(b-a); // Initialize tau to random value in interval
    for (int i=0;i<num_samples;i++){
        tau = infer_tau2(tau,a,b,c,rng);
        samples1[i] = tau;
    }

    // Part II: Sampling from joint (2d) truncated exponential distribution

    // Direct
    tau1 = a + rnum(rng)*(b-a);    // Initialize tau to random value in interval
    tau2 = tau1 + rnum(rng)*(b-tau1);
    for (int i=0;i<num_samples;i++){

        tau1 = infer_tau1(tau1,a,b,c,rng);
        tau2 = infer_tau2(tau2,tau1,b,c,rng);

        // MAYBE THIS IS WRONG?

        samples2[i] = tau1;
        samples3[i] = tau2;
    }

    // Rejection
    tau1 = a + rnum(rng)*(b-a); // Initialize tau to random value in interval
    tau2 = tau1 + rnum(rng)*(b-tau1);
    for (int i=0;i<num_samples;i++){

        // // Propose pair of new values for the random variates
        tau1_new = infer_tau1_with_rejection(tau1,a,b,c,rng);
        tau2_new = infer_tau2(tau2,tau1_new,b,c,rng);

        // infer_taus_with_rejection_bad(tau1,tau2,a,b,c,rng);  
    
        // if(tau2_new>tau1_new){
        if(1){
            tau1=tau1_new;
            tau2=tau2_new;
            }
    
        if(tau2<tau1){
            cout << "WHAT THE HELL" << endl;
            cout << tau2 << " " << tau1 << endl;
            }

        samples4[i] = tau1;
        samples5[i] = tau2;
    }

    // // Generate samples and/or time execution of infer_tau1()
    // double sum;
    // sum = 0;
    // // Time execution of sampling tau1 and tau2"
    // start = high_resolution_clock::now();
    // for (int i=0;i<num_samples;i++){
    //     /* sample a random x value from U(0,1)*/
    //     x = rnum(rng);
    //     tau1 = infer_tau1(x,a,b,c);
    //     samples[i] = tau1;

    //     x = rnum(rng);
    //     a_new = tau1;
    //     tau2 = infer_tau2(x,a_new,b,c);

    //     samples2[i] = tau2;
    //     sum+=tau2;
    // }

    //  end = high_resolution_clock::now();
    //  cout << "\nsum: " << sum << endl;
    //  elapsed_time = duration_cast<nanoseconds>(end - start);
    //  duration = elapsed_time.count() * 1e-9;

    // cout << "------------------------------------" << endl;
    // cout << "sampling tau1,tau2 | " << duration/num_samples <<endl;
    // cout << endl;

    /* ------ All of this is related to saving to file ------- */

    // Create file name
    string filename0,filename1,filename2,filename3;

    filename0="../data/"+to_string(a)+"_"+to_string(b)+"_"+to_string(c)+
    "_simpleTruncexpon_rejection.dat";

    filename1="../data/"+to_string(a)+"_"+to_string(b)+"_"+to_string(c)+
    "_simpleTruncexpon_direct.dat";

    filename2="../data/"+to_string(a)+"_"+to_string(b)+"_"+to_string(c)+
    "_jointTruncexpon_direct.dat";

    filename3="../data/"+to_string(a)+"_"+to_string(b)+"_"+to_string(c)+
    "_jointTruncexpon_rejection.dat";

    // Open files
    ofstream samples0_file,samples1_file,samples2_file,samples3_file;
    samples0_file.open(filename0);
    samples1_file.open(filename1);
    samples2_file.open(filename2);
    samples3_file.open(filename3);

    // Write sampled numbers to file
    for (int i=0; i<num_samples; i++){
        samples0_file<<fixed<<setprecision(17)<<samples0[i]<<endl;
        samples1_file<<fixed<<setprecision(17)<<samples1[i]<<endl;
        samples2_file<<fixed<<setprecision(17)<<samples2[i]<<" "<<samples3[i]<<endl;
        samples3_file<<fixed<<setprecision(17)<<samples4[i]<<" "<<samples5[i]<<endl;
    }
    
    // Close file
    samples0_file.close();
    samples1_file.close();
    samples2_file.close();
    samples3_file.close();

    // sum = 0;
    // // Time execution of sampling shifted tau1 and tau2"
    // start = high_resolution_clock::now();
    // for (int i=0;i<num_samples;i++){
    //     /* sample a random x value from U(0,1)*/
    //     x = rnum(rng);
    //     tau1 = shifted_infer_tau1(x,a,b,c);
    //     samples[i] = tau1;

    //     x = rnum(rng);
    //     a_new = tau1;
    //     tau2 = infer_tau2(x,a_new,b,c);

    //     samples2[i] = tau2;
    //     sum+=tau1; 
    // }
    // cout << "sum: " << sum << endl;
    
    //  end = high_resolution_clock::now();
    //  elapsed_time = duration_cast<nanoseconds>(end - start);
    //  duration = elapsed_time.count() * 1e-9;

    // cout << "------------------------------------" << endl;
    // cout << "shifted  tau1,tau2 | " << duration/num_samples <<endl;
    // cout << endl;

    // Test issue where argument of lambert W -1 branch is illegal
    // long double z = -6.3801721902898252e-309;
    // double result;
    // result = lambert_wm1(z);

    // long double inverse_e = -1/exp(1);
    // cout << setprecision(24) << inverse_e << endl;
    // cout << setprecision(24) << -boost::math::constants::exp_minus_one<long double>() << endl;

    // // test if we can perform comparison where one number is underflowed
    // double underflowed_num,legal_num;
    // underflowed_num = -2.569141358374482e-322;
    // legal_num = 1.01;
    // cout << underflowed_num/100 << " " << legal_num << endl;
    // cout << (underflowed_num < legal_num) << endl;
    // if (abs(underflowed_num) < 1e-20 ){cout << "VEGETA!!!!" << endl;} 
    // cout << lambert_wm1(-1e-300) << " " << lambert_wm1(0) << " " << lambert_wm1(-1e-100) << endl;
    // cout << lambert_w0(-1e-100) << " " << lambert_w0(0) << " " << lambert_w0(-1e-14) << endl;

    return 0;
}