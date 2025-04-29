/*
 * This file is part of litchi, a lightweight C++ library
 * for Minkowski analysis
 * 
 * Copyright (C) 2021-2023 Caroline Collischon <caroline.collischon@fau.de>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */


#include <pybind11/pybind11.h>
#include "litchi_eat.hpp"

namespace py = pybind11;


//Nside, rankA, rankB, curvIndex, mint, maxt, numt, smooth, smoothRad, NsideOut, linThresh, function, forceOutname, sequence, maskname, maskThresh
PYBIND11_MODULE(litchieat, m) {
    py::class_<paramStruct>(m, "paramStruct")
        .def(py::init<>())
        .def_readwrite("Nside", &paramStruct::Nside)
        .def_readwrite("rankA", &paramStruct::rankA)
        .def_readwrite("rankB", &paramStruct::rankB)
        .def_readwrite("curvIndex", &paramStruct::curvIndex)
        .def_readwrite("mint", &paramStruct::mint)
        .def_readwrite("maxt", &paramStruct::maxt)
        .def_readwrite("numt", &paramStruct::numt)
        .def_readwrite("smooth", &paramStruct::smooth)
        .def_readwrite("smoothRad", &paramStruct::smoothRad)
        .def_readwrite("NsideOut", &paramStruct::NsideOut)
        .def_readwrite("linThresh", &paramStruct::linThresh)
        .def_readwrite("function", &paramStruct::function)
        .def_readwrite("forceOutname", &paramStruct::forceOutname)
        .def_readwrite("sequence", &paramStruct::sequence)
        .def_readwrite("maskname", &paramStruct::maskname)
        .def_readwrite("maskThresh", &paramStruct::maskThresh)
        .def("__copy__",[](const paramStruct& self){
            return paramStruct(self);
        })
        .def("__deepcopy__",[](const paramStruct& self, py::dict){
            return paramStruct(self);
        });
    m.def("makeSingleMinkmap", &makeSingleMinkmap, "Can be replaced with makeMinkmap. Reads file given by inname, creates single Minkmap according to params, writes to outname", py::arg("inname"), py::arg("params"), py::arg("outname"));
    m.def("makeSequence", &makeSequence, "Can be replaced with makeMinkmap. Reads file given by inname, creates threshold sequence of Minkmaps according to params, writes to outname", py::arg("inname"), py::arg("params"), py::arg("outname"));
    m.def("makeMinkmap", &makeMinkmap, "Reads file given by inname, creates Minkmap or sequence of Minkmap according to params, writes to outname", py::arg("inname"), py::arg("params"), py::arg("outname"));
}
