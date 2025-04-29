#include <pybind11/pybind11.h>

#include "symmetric/symmetric.h"

PYBIND11_MODULE(_C, mod) {
  QUIK::symmetric::buildSubmodule(mod);
}
