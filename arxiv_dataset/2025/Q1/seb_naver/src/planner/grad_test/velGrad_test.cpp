#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
using namespace std;
double Fun(double ds, Eigen::Vector2d vel){   
    double inv_cos_vphix = 1.0;
    double vmax = 15;

    double vel_norm = vel.norm();
    double vx = ds * vel.norm() * inv_cos_vphix;
    double vioVel = vx * vx - vmax * vmax;
    double cost = vioVel;
    return cost;
}
// Function to compute the analytical derivative with respect to vel
Eigen::Vector2d analytical_derivative_vel(double ds, Eigen::Vector2d vel) {
    double inv_cos_vphix = 1.0;
    double vel_norm = vel.norm();
    double vx = ds * vel_norm * inv_cos_vphix;
    Eigen::Vector2d gradViolaVv = 2 * vx * ds * inv_cos_vphix * vel / vel_norm;
    return gradViolaVv;
}

// Function to compute the analytical derivative with respect to ds
double analytical_derivative_ds(double ds, Eigen::Vector2d vel) {
    double inv_cos_vphix = 1.0;
    double vel_norm = vel.norm();
    double vx = ds * vel_norm * inv_cos_vphix;
    double gradViolaVds = 2 * vx * vel_norm * inv_cos_vphix;
    return gradViolaVds;
}
// Function to compute the numerical derivative with respect to vel
Eigen::Vector2d numerical_derivative_vel(Eigen::Vector2d vel, double ds, double h = 1e-5) {
    Eigen::Vector2d grad;
    for (int i = 0; i < vel.size(); ++i) {
        Eigen::Vector2d vel_plus = vel;
        Eigen::Vector2d vel_minus = vel;
        vel_plus(i) += h;
        vel_minus(i) -= h;
        grad(i) = (Fun(ds, vel_plus) - Fun(ds, vel_minus)) / (2 * h);
    }
    return grad;
}

// Function to compute the numerical derivative with respect to ds
double numerical_derivative_ds(Eigen::Vector2d vel, double ds, double h = 1e-5) {
    return (Fun(ds + h, vel) - Fun(ds - h, vel)) / (2 * h);
}
int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::Vector2d vel = Eigen::Vector2d::Random().cwiseAbs() * 9 + Eigen::Vector2d::Ones();
    double ds = 0.1 + static_cast<double>(std::rand()) / (static_cast<double>(RAND_MAX / (2.0 - 0.1)));
    
    // Define lambda functions for numerical derivatives
    // std::Fun<double(Eigen::Vector2d)> func_vel = [&](Eigen::Vector2d v) { return Fun(ds, v); };
    // std::Fun<double(double)> func_ds = [&](double d) { return Fun(d, vel); };

    // Compute numerical derivatives
    Eigen::Vector2d num_deriv_vel = numerical_derivative_vel(vel,ds);
    double num_deriv_ds = numerical_derivative_ds(vel, ds);

    // Print numerical derivatives
    std::cout << "Numerical derivative with respect to vel: " << num_deriv_vel.transpose() << std::endl;
    std::cout << "Numerical derivative with respect to ds: " << num_deriv_ds<<std::endl;

    // Compute analytical derivatives
    Eigen::Vector2d anal_deriv_vel = analytical_derivative_vel(ds, vel);
    double anal_deriv_ds = analytical_derivative_ds(ds, vel);

    // Print analytical derivatives
    std::cout << "Analytical derivative with respect to vel: " << anal_deriv_vel.transpose() << std::endl;
    std::cout << "Analytical derivative with respect to ds: " << anal_deriv_ds << std::endl;
    
    return 0;
}