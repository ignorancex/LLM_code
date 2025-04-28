
from distutils.core import setup
import warnings

setup(name='hytoperm',
   version='0.1.0',
   python_requires='>=3.8',
   description='Hybrid Trajectory Optimizer for Persistent Monitoring',
   author='Jonas Hall',
   author_email='hall.f.jonas@gmail.com',
   license='MIT',
   include_package_data = True,
   py_modules=[],
   setup_requires=['setuptools_scm'],
   install_requires=[
      'PySimpleGUI==4.60.5',
      'matplotlib<3.9',             # cmap
      'numpy>=1.24.4,<2.0.0',
      'casadi>=3.6.3',
      'pdfCropMargins>=2.1.1',
      'scipy',
      'python-tsp>=0.4.0',
      'networkx>=2.8.8'
   ]
)

warnings.filterwarnings("always")
