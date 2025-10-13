#include "poissonsystem.hpp"
#include "wrappers.hpp"

void wrap_poissonsolver(py::module& m) {
  py::class_<PoissonSystem>(m, "PoissonSystem")
      .def("init", &PoissonSystem::init)
      .def("solve", [](PoissonSystem* p) { return fieldToArray(p->solve()); })
      .def_property_readonly("solver", &PoissonSystem::solver);
}
