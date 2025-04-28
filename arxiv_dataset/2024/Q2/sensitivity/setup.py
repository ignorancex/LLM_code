from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="primary_sensitivity",
        sources=["sensitivity/sensitivity.pyx", "sensitivity/students_t.c", "sensitivity/primary_sensitivity_impl.c"],
        include_dirs=["."],
    )
]

setup(
    name="primary_sensitivity",
    version="0.1",
    description="Computes sensitivity of Pearson Correlation Coefficient and its p-value",
    author="Marc Harary",
    author_email="marc@ds.dfci.harvard.edu",
    ext_modules=cythonize(ext_modules)
)
