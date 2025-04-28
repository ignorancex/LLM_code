import setuptools

setuptools.setup(
    name='reglyman',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.14,<2.0',
        'scipy>=1.1,<2.0',
        'matplotlib',
        'regpy',
	'nbodykit',
    'astropy',
    'pmesh'
    ],
    python_requires='>=3.6,<4.0',
)
