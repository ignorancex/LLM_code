from setuptools import setup, find_packages

setup(
    name='inventory_pricing_management',
    version='0.1.0',
    description='Inventory and Pricing Management Environment under Censored and Dependent Demand',
    author='anonymous',
    author_email='anonymous@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'scikit-survival',
        'gym',
        'or-gym',
        'stable-baselines3'
    ],
)
