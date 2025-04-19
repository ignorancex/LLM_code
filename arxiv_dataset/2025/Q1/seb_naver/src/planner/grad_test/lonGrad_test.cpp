#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
using namespace std;

double delta_signl = 1e-5;
double calAx_body(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc){
    double gravity = 9.81;
    double sin_phix = 0.0;
    double inv_cos_vphix = 1.0;
    Eigen::Vector2d vel_local_dt = vel * ds ;
    double vel_local_norm = vel_local_dt.norm() + delta_signl;
    
    Eigen::Vector2d acc_local_dt = acc * ds * ds + vel * dds;
    double ax_body = acc_local_dt.dot(vel_local_dt) / vel_local_norm * inv_cos_vphix + gravity * sin_phix;
    return ax_body;
}
double Fun(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc){ 
    double lonAccmax = 1.0;
    double ax_body = calAx_body(ds, dds, vel, acc);
    double vioLonAcc = ax_body * ax_body - lonAccmax * lonAccmax;
}
// Function to compute the analytical derivative with respect to vel
Eigen::Vector2d analytical_derivative_vel(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc) {
    double inv_cos_vphix = 1.0;
    double vel_norm = vel.norm() + delta_signl;
    double ax_body = calAx_body(ds, dds, vel, acc);
    Eigen::Vector2d gradViolaLonAv = 2 * ax_body * ((ds * ds * acc + 2 * dds * vel) / vel_norm * inv_cos_vphix 
                                    - vel * (ds * ds * vel.dot(acc) + dds * vel.dot(vel)) / pow( vel_norm,3) * inv_cos_vphix); 

    return gradViolaLonAv;
}

// Function to compute the analytical derivative with respect to ds
double analytical_derivative_ds(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc) {
    double inv_cos_vphix = 1.0;
    double vel_norm = vel.norm() + delta_signl;
    double ax_body = calAx_body(ds, dds, vel, acc);
    double gradViolaLonAds = 2.0 * ax_body * (2 * ds * acc.dot(vel) /  vel_norm) * inv_cos_vphix;
    return gradViolaLonAds;
}
// Function to compute the analytical derivative with respect to acc
Eigen::Vector2d analytical_derivative_acc(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc) {
    double inv_cos_vphix = 1.0;
    double vel_norm = vel.norm()+ delta_signl;
    double ax_body = calAx_body(ds, dds, vel, acc);
    Eigen::Vector2d gradViolaLonAa = 2 * ax_body * ds * ds * vel / vel_norm * inv_cos_vphix;
    return gradViolaLonAa;
}

// Function to compute the analytical derivative with respect to dds
double analytical_derivative_dds(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc) {
    double inv_cos_vphix = 1.0;
    double vel_norm = vel.norm()+ delta_signl;
    double ax_body = calAx_body(ds, dds, vel, acc);
    double gradViolaLonAdds = 2.0 * ax_body * vel.dot(vel) / vel_norm * inv_cos_vphix;
    return gradViolaLonAdds;
}

// Function to compute the numerical derivative with respect to acc
Eigen::Vector2d numerical_derivative_acc(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc, double h = 1e-5) {
    Eigen::Vector2d grad;
    for (int i = 0; i < acc.size(); ++i) {
        Eigen::Vector2d acc_plus = acc;
        Eigen::Vector2d acc_minus = acc;
        acc_plus(i) += h;
        acc_minus(i) -= h;
        grad(i) = (Fun(ds, dds, vel, acc_plus) - Fun(ds, dds, vel, acc_minus)) / (2 * h);
    }
    return grad;
}

// Function to compute the numerical derivative with respect to dds
double numerical_derivative_dds(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc, double h = 1e-5) {
    return (Fun(ds, dds + h, vel, acc) - Fun(ds, dds - h, vel, acc)) / (2 * h);
}
// Function to compute the numerical derivative with respect to vel
Eigen::Vector2d numerical_derivative_vel(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc, double h = 1e-5) {
    Eigen::Vector2d grad;
    for (int i = 0; i < vel.size(); ++i) {
        Eigen::Vector2d vel_plus = vel;
        Eigen::Vector2d vel_minus = vel;
        vel_plus(i) += h;
        vel_minus(i) -= h;
        grad(i) = (Fun(ds, dds, vel_plus, acc) - Fun(ds, dds, vel_minus, acc)) / (2 * h);
    }
    return grad;
}

// Function to compute the numerical derivative with respect to ds
double numerical_derivative_ds(double ds, double dds, Eigen::Vector2d vel, Eigen::Vector2d acc, double h = 1e-5) {
    return (Fun(ds + h, dds, vel, acc) - Fun(ds - h, dds, vel, acc)) / (2 * h);
}
int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    // Eigen::Vector2d vel = Eigen::Vector2d::Random().cwiseAbs() * 9 + Eigen::Vector2d::Zero();
    Eigen::Vector2d vel = Eigen::Vector2d::Random().cwiseAbs() * 1e-3;
    double ds = 0.0 + static_cast<double>(std::rand()) / (static_cast<double>(RAND_MAX / (2.0 - 0.1)));
    Eigen::Vector2d acc = Eigen::Vector2d::Random().cwiseAbs() * 9 + Eigen::Vector2d::Zero();
    double dds = 0.0 + static_cast<double>(std::rand()) / (static_cast<double>(RAND_MAX / (2.0 - 0.1)));
    std::cout<<"vel "<<vel.transpose()<<" acc "<<acc.transpose()<<" ds "<<ds<<" dds "<<dds<<std::endl;
    // Compute numerical derivatives
    Eigen::Vector2d num_deriv_vel = numerical_derivative_vel(ds, dds, vel, acc);
    double num_deriv_ds = numerical_derivative_ds(ds, dds, vel, acc);

    // Print numerical derivatives
    std::cout << "Numerical derivative with respect to vel: " << num_deriv_vel.transpose() << std::endl;
    std::cout << "Numerical derivative with respect to ds: " << num_deriv_ds<<std::endl;

    // Compute analytical derivatives
    Eigen::Vector2d anal_deriv_vel = analytical_derivative_vel(ds, dds, vel, acc);
    double anal_deriv_ds = analytical_derivative_ds(ds, dds, vel, acc);

    // Print analytical derivatives
    std::cout << "Analytical derivative with respect to vel: " << anal_deriv_vel.transpose() << std::endl;
    std::cout << "Analytical derivative with respect to ds: " << anal_deriv_ds << std::endl;

    // Compute numerical derivatives for dds and acc
    double num_deriv_dds = numerical_derivative_dds(ds, dds, vel, acc);
    Eigen::Vector2d num_deriv_acc = numerical_derivative_acc(ds, dds, vel, acc);

    // Print numerical derivatives for dds and acc
    std::cout << "Numerical derivative with respect to dds: " << num_deriv_dds << std::endl;
    std::cout << "Numerical derivative with respect to acc: " << num_deriv_acc.transpose() << std::endl;

    // Compute analytical derivatives for dds and acc
    double anal_deriv_dds = analytical_derivative_dds(ds, dds, vel, acc);
    Eigen::Vector2d anal_deriv_acc = analytical_derivative_acc(ds, dds, vel, acc);

    // Print analytical derivatives for dds and acc
    std::cout << "Analytical derivative with respect to dds: " << anal_deriv_dds << std::endl;
    std::cout << "Analytical derivative with respect to acc: " << anal_deriv_acc.transpose() << std::endl;
    
    return 0;
}