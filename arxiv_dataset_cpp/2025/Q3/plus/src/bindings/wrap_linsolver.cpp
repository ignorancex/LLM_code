#include "linsolver.hpp"
#include "wrappers.hpp"

void wrap_linsolver(py::module& m) {
  py::class_<LinSolver>(m, "LinSolver")
      .def("step", [](LinSolver* solver) { solver->stepper()->step(); })
      .def("restart", [](LinSolver* solver) { solver->stepper()->restart(); })
      .def("max_norm_residual", &LinSolver::residualMaxNorm)
      .def("set_method", static_cast<void (LinSolver::*)(const std::string&)>(
                             &LinSolver::setMethod))  // cast needed because
                                                      // setMethod is overloaded
      .def("solve", &LinSolver::solve)
      .def_readwrite("max_iter", &LinSolver::max_iter)
      .def_readwrite("tol", &LinSolver::tol);
}
