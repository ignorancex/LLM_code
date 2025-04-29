// app includes
// // env
#include "param/parameters.h"
#include "core/core.h"
// // learn
#include "core/learn/core.h"
// python
#include "pybind11/embed.h"
namespace py = pybind11;

int main () {
	// python
	py::scoped_interpreter python;
	py::gil_scoped_release release;
	// learn
	c0p::Learn::init();
	c0p::Learn::run();
	c0p::Learn::save();
}
