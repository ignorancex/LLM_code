from distutils.core import setup, Extension
from os import getenv
import sys
import os
from distutils import sysconfig

def hasNumpy():
  NUMPY='NONUMPY'
  NUMPYPATH='.'
  try:
    import numpy as np
    NUMPYPATH= "/".join(np.__file__.split("/")[0:-1])
    NUMPY='NUMPY'
  except:
    pass
  return NUMPY, NUMPYPATH

NUMPY,NUMPYPATH = hasNumpy()
print(NUMPY, NUMPYPATH)
windir = getenv('windir')
if windir == None:
  WIN32 = 'NOWIN32'
  LPSOLVE55 = '/home/mac9908/clara/lp_solve_5.5/lpsolve55/bin/ux64'
else:
  WIN32 = 'WIN32'
  LPSOLVE55 = '../../lpsolve55/bin/win32'
setup (name = "lpsolve55",
       version = "5.5.2.11",
       description = "Linear Program Solver, Interface to lpsolve",
       author = "Peter Notebaert",
       author_email = "lpsolve@peno.be",
       url = "http://www.peno.be/",
       py_modules=['lp_solve', 'lp_maker'],
       ext_modules = [Extension("lpsolve55",
				["lpsolve.c", "hash.c", "pythonmod.c"],
                                define_macros=[('PYTHON', '1'), (WIN32, '1'), ('NODEBUG', '1'), ('DINLINE', 'static'), (NUMPY, '1'), ('_CRT_SECURE_NO_WARNINGS', '1')],
                                include_dirs=['../..', NUMPYPATH, '/home/mac9908/clara/lp_solve_dev'],
                                library_dirs=[LPSOLVE55],
				libraries = ["lpsolve55"])
		      ]
)
