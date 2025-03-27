from setuptools import setup, find_packages

setup(
    name="helix",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy~=1.26",
        "networkx~=3.2.1",
        "matplotlib~=3.8.2",
        "gurobipy~=11.0.0",
    ],
    author="Yixuan Mei",
    author_email="yixuanm@andrew.cmu.edu",
    description="Official implementation of Helix.",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Thesys-lab/Helix-ASPLOS25",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)