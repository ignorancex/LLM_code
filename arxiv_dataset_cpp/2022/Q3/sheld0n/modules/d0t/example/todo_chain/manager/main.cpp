// Std includes
#include <cmath>
#include <iostream>
#include <memory>
// Thirdparties includes
#include <Eigen/Dense>
// Lib includes
#include "s0s/runge_kutta_fehlberg.h"
#include "sl0/point.h"
#include "sl0/chain/dynamic.h"
#include "sl0/chain/manager.h"
// Simple includes
#include "flow.h"

using TypeScalar = double;
// State
template<int Size>
using TypeVector = Eigen::Matrix<TypeScalar, Size, 1>;
// Space
constexpr unsigned int DIM = 2;
using TypeSpaceVector = Eigen::Matrix<TypeScalar, DIM, 1>;
// Ref and View
template<typename ...Args>
using TypeRef = Eigen::Ref<Args...>;
template<typename ...Args>
using TypeView = Eigen::Map<Args...>;
// Chain manager Parameters
using TypeStepPoint = sl0::StepPoint<TypeVector, DIM, TypeView, Flow>;
using TypeStepChainDynamic = sl0::StepChainDynamic<TypeVector, DIM, TypeView, TypeRef, TypeStepPoint>;
// Solver
using TypeSolver = s0s::SolverRungeKuttaFehlberg<TypeVector<-1>, TypeView>;

const unsigned int np = 11;

int main () {
    // Parameters
    TypeSpaceVector x0 = TypeSpaceVector::Zero();
    TypeScalar t0 = 0.0;
    TypeScalar dt = 1e-3;
    unsigned int nt = std::round(1.0 / dt);
    double dl = 0.1;
    double l = 1.0;
    // Create chain
    std::shared_ptr<TypeStepPoint> sStepPoint = std::make_shared<TypeStepPoint>(std::make_shared<Flow>());
    sl0::ChainManager<TypeVector, DIM, TypeView, TypeRef, TypeStepPoint, TypeSolver> chain(sStepPoint, dl, 4);
    // Init
    chain.sStep->addManaged(chain.states);
    for(std::size_t i = 0; i < np; i++) {
        chain.sStep->sManagedSteps[0]->addMember(chain.states[0]);
        sStepPoint->x(chain.sStep->sManagedSteps[0]->memberState(chain.states[0].data(), i)) = x0;
        sStepPoint->x(chain.sStep->sManagedSteps[0]->memberState(chain.states[0].data(), i))[0] += i * dl;
    }
    std::cout << "Init Length : " << "\n" << chain.sStep->sManagedSteps[0]->length(chain.states[0].data()) << "\n";
    std::cout << "Init Size : " << "\n" << chain.sStep->sManagedSteps[0]->size() << "\n";
    chain.t = t0;
    // Computation
    std::cout << "Computing" << "\n";
    for(std::size_t i = 0; i < nt; i++) {
        chain.update(dt);
    }
    // out
    std::cout << "\n";
    std::cout << "Chain advected following a an exponential flow, exp(" << chain.t << ") = " << "\n";
    std::cout << "\n";
    std::cout << "Final Length : " << "\n" << chain.sStep->sManagedSteps[0]->length(chain.states[0].data()) << "\n";
    std::cout << "Final Size : " << "\n" << chain.sStep->sManagedSteps[0]->size() << "\n";
    std::cout << std::endl;
}

