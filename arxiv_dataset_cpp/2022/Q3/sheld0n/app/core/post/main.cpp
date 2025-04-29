// app includes
// // env
#include "param/parameters.h"
#include "core/core.h"
// // solutions
#include "param/solutions/parameters.h"
#include "core/solutions/core.h"
// // post
#include "param/post/parameters.h"
#include "core/post/core.h"
// python
#include "pybind11/embed.h"
namespace py = pybind11;

int main () {
	py::scoped_interpreter python;
	py::gil_scoped_release release;

	c0p::Post<c0p::PostParameters> post;
}
