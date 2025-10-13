#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
} AbsInfObject;

static PyObject *
AbsInf_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return type->tp_alloc(type, 0);
}

static PyObject *
AbsInf_richcompare(PyObject *self, PyObject *other, int op) {
    switch (op) {
        case Py_GT:  // self > other → always True
            Py_RETURN_TRUE;
        case Py_LT:  // self < other → always False
            Py_RETURN_FALSE;
        default:
            Py_RETURN_NOTIMPLEMENTED;
    }
}

static PyObject *
AbsInf_add(PyObject *self, PyObject *other) {
    Py_INCREF(self);
    return self;
}

static PyNumberMethods AbsInf_as_number = {
    .nb_add = AbsInf_add,
};

static PyTypeObject AbsInfType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "absinf.AbsInf",
    .tp_basicsize = sizeof(AbsInfObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = AbsInf_new,
    .tp_richcompare = AbsInf_richcompare,
    .tp_as_number = &AbsInf_as_number,
    .tp_doc = "AbsInf: always greater than any number",
};

static PyModuleDef absinfmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "absinf",
    .m_doc = "Abstract infinity object that is greater than any number.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_absinf(void) {
    PyObject *m;
    if (PyType_Ready(&AbsInfType) < 0)
        return NULL;

    m = PyModule_Create(&absinfmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&AbsInfType);
    if (PyModule_AddObject(m, "AbsInf", (PyObject *)&AbsInfType) < 0) {
        Py_DECREF(&AbsInfType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
