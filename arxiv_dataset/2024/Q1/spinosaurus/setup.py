import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spinosaurus",
    version="1.0",
    author="Stephen Chen and Nick Kokron",
    author_email="sfschen@ias.edu",
    description="Code for perturbation theory of galaxy shapes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sfschen/spinosaurus",
    packages=['spinosaurus','spinosaurus/Utils'],
             #setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy','pyfftw'],
)