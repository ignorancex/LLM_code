#include "Langevin.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "qhat.h"

// two step Langevin: step 1, giving (temp, drag, kpara, kperp), return the first step (xi_Tx, xi_Ty, xi_L)
// E is the HQ energy, M is HQ mass, temp is the local temperature
// drag, kperp, kpara have the unit of [GeV^3]
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::normal_distribution<double> distribution(0.0, 1.0);
        
void Langevin_pre(double p_length, double M, double temp, double drag, double kperp, double kpara, double deltat_lrf, std::vector<double> &pre_result)
{
         // sample normal distribution with mean=0.0, width=1.0

        // save the values of rho as those will carry on to the post_point scheme second step
        std::vector<double> rho_xi(3, 0.0);
        for (size_t i=0; i< 3; ++i)
                rho_xi[i] = distribution(generator);


        std::vector<double> xi(3, 0.0);
        xi[0] = std::sqrt(kperp/deltat_lrf) * rho_xi[0];
        xi[1] = std::sqrt(kperp/deltat_lrf) * rho_xi[1];
        xi[2] = std::sqrt(kpara/deltat_lrf) * rho_xi[2];

        
        std::vector<double> new_p(4, 0.0);
        new_p[1] = xi[0] * deltat_lrf;
        new_p[2] = xi[1] * deltat_lrf;
        new_p[3] = p_length + (-drag * p_length + xi[2]) *deltat_lrf;
        new_p[0] = std::sqrt(M*M + new_p[1]*new_p[1] + new_p[2]*new_p[2] + new_p[3]*new_p[3]);

        pre_result={new_p[0], new_p[1], new_p[2], new_p[3], rho_xi[0], rho_xi[1], rho_xi[2]};


}


// for the two step post-point scheme, between step 1 (pre) and step 2(post), there are a few steps missing 
// the pre-step1 returns new_p, with new_p, we can update kpara and kperp 

void Langevin_post(double p_length, double M, double temp, double drag, double kperp, double kpara, double deltat_lrf, std::vector<double> const& pre_result, std::vector<double> & post_result)
{
        // use the pre-point stored rho
//        std::cout << "Langevin_post: " << pre_result[0] << " " << pre_result[1] << " " << pre_result[2] << " " << pre_result[3] <<" " << pre_result[4] << " " << pre_result[5] << " " << pre_result[6] << std::endl;

        std::vector<double> rho={pre_result[4], pre_result[5], pre_result[6]};
        std::vector<double> xi(3, 0.0);
        xi[0] = std::sqrt(kperp/deltat_lrf) * rho[0];
        xi[1] = std::sqrt(kperp/deltat_lrf) * rho[1];
        xi[2] = std::sqrt(kpara/deltat_lrf) * rho[2];

		post_result.resize(4);
        post_result[1] = xi[0] * deltat_lrf;
        post_result[2] = xi[1] * deltat_lrf;
        post_result[3] = p_length + (-drag * p_length + xi[2]) * deltat_lrf;
        post_result[0] = std::sqrt(M*M + post_result[1]*post_result[1] + post_result[2]*post_result[2] + post_result[3]*post_result[3]);

}

