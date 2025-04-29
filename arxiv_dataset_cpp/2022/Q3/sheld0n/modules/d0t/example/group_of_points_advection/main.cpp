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
#include "d0t/equation.h"
#include "d0t/equations/advection.h"
// simple includes
#include "flow.h"

using tScalar = double;
// linear Algebra
template<int Size>
using tVector = Eigen::Matrix<tScalar, Size, 1>;
// space
constexpr unsigned int Dim = 3;
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
class GroupOfPoints : public d0t::VariableGroupStatic<Point, 10> {
};
using tSubEquation = d0t::EquationAdvection<Point, tFlow>;
using tEquation = d0t::EquationGroupStatic<GroupOfPoints, tSubEquation>;
using tSolution = d0t::SolutionStatic<tSolver, tEquation>;

int main () { 
    tSpaceVector x0 = tSpaceVector::Constant(1.0);
    double dt = 1e-3;
    double tEnd = 1.0;
    unsigned int nt = std::round(tEnd / dt);
    // Create equation and solution
    tSolution solution;
    // Set initial state
    for(unsigned int memberIndex = 0; memberIndex < GroupOfPoints::GroupSize; memberIndex++) {
    	Point::position(GroupOfPoints::state(solution.state.data(), memberIndex)) = x0;
    }
    // Computation
    for(unsigned int i = 0; i < nt; i++) {
        solution.step(dt);
    }
    // out
    std::cout << "\n";
    std::cout << "Points advected following a an exponential flow, exp(" << solution.t << ") = " << "\n";
    std::cout << "\n";
    std::cout << "Points Final Positions : " << "\n";
    for(unsigned int memberIndex = 0; memberIndex < GroupOfPoints::GroupSize; memberIndex++) {
    	std::cout << Point::tPosition::cGet(Point::statePosition(GroupOfPoints::state(solution.state.data(), memberIndex))) << "\n";
    }
    std::cout << std::endl;
}
