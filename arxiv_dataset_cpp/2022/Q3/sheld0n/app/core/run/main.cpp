// app includes
// // env
#include "param/parameters.h"
#include "core/core.h"
// // run
#include "param/run/parameters.h"
#include "core/run/core.h"
// python
#include "pybind11/embed.h"
namespace py = pybind11;

int main () {
	// python support
	py::scoped_interpreter python;
	py::gil_scoped_release release;
	// run
	c0p::Run<c0p::RunParameters> run;
}
