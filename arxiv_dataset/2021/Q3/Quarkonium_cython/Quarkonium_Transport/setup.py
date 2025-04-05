from setuptools import setup, Extension
from Cython.Build import cythonize
from glob import glob
import os
import numpy

#libs=[path for path in os.environ['LD_LIBRARY_PATH'].split(':') if path]

#------ Dissociation and Recombination Module ------
filesDisRec = [	'cython/DisRec.pyx', 
				'src/DisRec.cxx']
#------ Lorentz and Rotation transformations ------
filesLorRot = [	'cython/LorRot.pyx',
				'src/TransLorentzRotate.cxx' ]
modules = [
        Extension('DisRec',
        		 sources=filesDisRec, 
        		 language="c++", 
        		 #library_dirs=libs,
                 include_dirs=[numpy.get_include()],
        		 extra_compile_args=["-stdlib=libc++", '-march=native', '-fPIC'],
        		 #extra_compile_args=["-std=c++11", '-march=native', '-fPIC'],
        		 # use -stdlib=libc++ on MaxOS otherwise -std=c++11
        		 libraries=["m", "gsl", "gslcblas"]),
        Extension('LorRot',
        		 sources=filesLorRot, 
        		 language="c++", 
        		 #library_dirs=libs,
                 include_dirs=[numpy.get_include()],
        		 extra_compile_args=["-stdlib=libc++", '-march=native', '-fPIC'],
        		 #extra_compile_args=["-std=c++11", '-march=native', '-fPIC'],
        		 # use -stdlib=libc++ on MaxOS otherwise -std=c++11
        		 libraries=["m", "gsl", "gslcblas"]),
]


setup(
        ext_modules=cythonize(modules)
)
