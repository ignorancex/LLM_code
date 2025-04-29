// Std includes
#include <cmath>
#include <iostream>
#include <memory>
// Thirdparties includes
#include <Eigen/Dense>
// Lib includes
#include "s0s/runge_kutta_fehlberg.h"
#include "sl0/point.h"
#include "sl0/chain/static.h"
// Simple includes
#include "flow.h"

using TypeScalar = double;
// State
template<int StateSize>
using TypeState = Eigen::Matrix<TypeScalar, StateSize, 1>;
using TypeStateDynamic = Eigen::Matrix<TypeScalar, Eigen::Dynamic, 1>;
// Space
constexpr unsigned int DIM = 3;
using TypeVector = Eigen::Matrix<TypeScalar, DIM, 1>;
// Ref and View
template<typename ...Args>
using TypeRef = Eigen::Ref<Args...>;
template<typename ...Args>
using TypeView = Eigen::Map<Args...>;
// Group Parameters
using TypeStepPoint = sl0::StepPoint<TypeState, DIM, TypeRef, TypeView, Flow>;
// Solver
using TypeSolver = s0s::SolverRungeKuttaFehlberg;

const unsigned int np = 10;

int main () { 
    // Parameters
    TypeVector x0 = TypeVector({0.0, 0.0, 0.0});
    TypeScalar t0 = 0.0;
    TypeScalar dt = 1e-3;
    unsigned int nt = std::round(1.0 / dt);
    double l = 1.0;
    // Create chain
    std::shared_ptr<TypeStepPoint> sStepPoint = std::make_shared<TypeStepPoint>(std::make_shared<Flow>());
    sl0::ChainStatic<TypeState, DIM, TypeRef, np, TypeView, TypeStepPoint, TypeSolver> chain(sStepPoint, l, 4);
    // Init
    for(std::size_t i = 0; i < np; i++) {
        sStepPoint->x(chain.sStep->memberState(chain.state, i)) = x0;
        sStepPoint->x(chain.sStep->memberState(chain.state, i))[0] += i * l/double(np-1);
    }
    std::cout << "Init Length : " << "\n" << chain.sStep->actualLength(chain.state) << "\n";
    std::cout << "Init State : " << "\n" << chain.state << "\n";
    chain.t = t0;
    // Computation
    std::cout << "Computing" << "\n";
    for(std::size_t i = 0; i < nt; i++) {
        chain.update(dt);
    }
    // out
    std::cout << "\n";
    std::cout << "Chain advected following an exponential flow, exp(" << chain.t << ") = " << "\n";
    std::cout << "\n";
    std::cout << "Final State : " << "\n" << chain.state << "\n";
    std::cout << "Final Length : " << "\n" << chain.sStep->actualLength(chain.state) << "\n";
    std::cout << std::endl;
}

