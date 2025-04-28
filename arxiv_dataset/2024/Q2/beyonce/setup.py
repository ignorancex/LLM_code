from setuptools import setup

setup(
    name="beyonce",
    version="1.0.0",    
    description="",
    url="https://github.com/dmvandamt/beyonce",
    author="Dirk van Dam",
    author_email="dmvandam@strw.leidenuniv.nl",
    license="BSD 2-clause",
    packages=["beyonce"],
    install_requires=["matplotlib",
                      "numpy",
                      "pyppluss",
                      "pytest",
                      "tqdm",      
                      ],

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux", 
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Typing :: Stubs Only",
    ],
)