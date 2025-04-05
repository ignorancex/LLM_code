from distutils.core import setup, Extension
import os

os.environ["CC"]="gcc"

module1=Extension('cl2wth',
                  libraries=['gsl','gslcblas'],
                  library_dirs=['/users/damonge/lib'],
                  include_dirs=['/users/damonge/include'],
                  sources=['cl2wth.c'],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-lgomp']);


#    long_description = file.read()
setup(name='Cl2Wth',
      version='0.0',
      description='Cl2Wth',
#      author='David Alonso',
#      author_email='david.alonso@astro.ox.ac.uk',
#      url='http://members.ift.uam-csic.es/dmonge/Software.html',
#      license='GPL v3.0',
#      long_description=long_description,
      ext_modules=[module1])
