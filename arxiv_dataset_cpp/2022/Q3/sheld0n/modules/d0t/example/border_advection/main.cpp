// std includes
#include <cmath>
#include <iostream>
// thirdparties includes
#include <Eigen/Dense>
// lib includes
#include "s0s/runge_kutta_fehlberg.h"
// testing lib include
#include "d0t/variable.h"
#include "d0t/variables/vector.h"
#include "d0t/variables/border.h"
#include "d0t/equation.h"
#include "d0t/equations/g.h"
// simple includes
#include "flow.h"

using tScalar = double;
// linear Algebra
template<int Size>
using tVector = Eigen::Matrix<tScalar, Size, 1>;
// space
constexpr unsigned int Dim = 2;
using tSpaceVector = Eigen::Matrix<tScalar, Dim, 1>;
// ref and view
template<typename ...Args>
using tView = Eigen::Map<Args...>;
// solver
using tSolver = s0s::SolverRungeKuttaFehlberg<tVector<Eigen::Dynamic>, tView>;
// flow
using tFlow = Flow<tSpaceVector, tView>;

// creating variable, equation and solution
class Point : public d0t::VariableComposed<d0t::VariableVector<tVector, tView, Dim>> {
    public:
        using tBase = d0t::VariableComposed<d0t::VariableVector<tVector, tView, Dim>>;
        using tStateVectorDynamic = typename tBase::tStateVectorDynamic;
    public:
        using tPosition = tVariableComponent<0>::type;
        using tSpaceVector = typename tPosition::tSpaceVector;
        
        static constexpr auto statePosition = state<0>;
        static constexpr auto cStatePosition = cState<0>;
        
        static tView<tSpaceVector> position(double* pState) {
            return tPosition::get(statePosition(pState));
        }
        static tView<const tSpaceVector> cPosition(const double* pState) {
            return tPosition::cGet(cStatePosition(pState));
        }
};
constexpr const double BorderDensity = 2.0;
constexpr const bool BorderClosed = true;
constexpr const unsigned int BorderInterpolationOrder = 4;
class Border : public d0t::VariableBorder<Point, BorderDensity, BorderClosed, BorderInterpolationOrder> {
};
constexpr const double BorderLocalBurningVelocity = 1.0;
using tBorderEquation = d0t::EquationG<Border::tCurve, tFlow, BorderLocalBurningVelocity>;
using tSolution = d0t::SolutionGroups<tSolver, tBorderEquation, Border>;

// initial state parameters
constexpr const double initRadius = 1.0;
constexpr unsigned int initPointNumber = 16;

int main () { 
    tSpaceVector x0 = tSpaceVector::Zero();
    double dt = 1e-3;
    double tEnd = 1.0;
    unsigned int nt = std::round(tEnd / dt);
    // Create equation and solution
    tSolution solution;
    // TODO: ----------------------------------------------------------------------------- add some members
    // Set initial state
    Border::pushBackGroup(solution.states);
    for(unsigned int memberIndex = 0; memberIndex < initPointNumber; memberIndex++) {
        Border::tCurve::pushBackMember(Border::state(solution.states, 0));
        Point::position(
            Border::tCurve::state(Border::state(solution.states, 0).data(), memberIndex)
        ) = x0 + initRadius * tSpaceVector({std::cos(2.0 * M_PI * memberIndex / initPointNumber), std::sin(2.0 * M_PI * memberIndex / initPointNumber)});
    }
    std::cout << "\n";
    std::cout << "Points Initial Positions : " << "\n\n";
    for(unsigned int curveIndex = 0; curveIndex < Border::groupNumber(solution.states.size()); curveIndex++) {
        std::cout << "Curve index : " << curveIndex << "\n\n";
        for(double s = 0; s <= 1.0; s+=0.1) {
            std::cout << "s = " << s << "\n";
            std::cout << Border::tCurve::cPositionInterpolated(Border::cState(solution.states, curveIndex).data(), Border::cState(solution.states, curveIndex).size(), s) << "\n\n";
        }
    }
    std::cout << std::endl;
    // Computation
    for(unsigned int i = 0; i < nt; i++) {
        solution.step(dt);
    }
    // Output
    std::cout << "\n";
    std::cout << "Points advected following a an exponential flow, exp(" << solution.t << ") = " << "\n";
    std::cout << "\n";
    std::cout << "Points Final Positions : " << "\n\n";
    for(unsigned int curveIndex = 0; curveIndex < Border::groupNumber(solution.states.size()); curveIndex++) {
        std::cout << "Curve index : " << curveIndex << "\n\n";
        for(double s = 0; s <= 1.0; s+=0.1) {
            std::cout << "s = " << s << "\n";
            std::cout << Border::tCurve::cPositionInterpolated(Border::cState(solution.states, curveIndex).data(), Border::cState(solution.states, curveIndex).size(), s) << "\n\n";
        }
    }
    std::cout << std::endl;
}
