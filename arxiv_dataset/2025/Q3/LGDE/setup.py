"""Setup."""

from setuptools import setup, find_packages

install_requires = [
    "matplotlib>=3.5.1",
    "networkx>=2.6.3",
    "numpy>=1.21.5",
    "pandas>=1.5.3",
    "python_igraph>=0.10.2",
    "scipy>=1.7.3",
    "tqdm>=4.64.1",
    "severability @ git+https://github.com/barahona-research-group/severability",
]


setup(
    name="LGDE",
    version="1.1.0",
    description="Local Graph-based Dictionary Expansion Python package",
    url="https://github.com/barahona-research-group/LGDE",
    author="Juni Schindler",
    author_email="juni.schindler19@imperial.ac.uk",
    long_description=open("README.md", "r", encoding="UTF-8").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10.14",
    ],
)
