# Setup file for Lipschitz Driven Inference

from setuptools import setup, find_packages

setup(
    name='lipschitz_driven_inference',
    version='0.1',
    description='Construct bias-corrected confidence intervals for Lipschitz-driven linear regression',
    author='David R. Burt and Renato Berlinghieri',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'cvxpy',
        'POT',
        'gpflow',
        'statsmodels',
        'pandas',
        'json_tricks',
        'joblib',
        'shapely',
        'geopandas',
        ],
)
