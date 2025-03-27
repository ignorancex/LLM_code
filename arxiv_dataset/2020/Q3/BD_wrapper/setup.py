import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="BD_wrapper",  # Replace with your own username
    version="0.1.0",
    author="Manuel Riener",
    author_email="riener@mpia-hd.mpg.de",
    description="Python wrapper for the Bayesian Distance Calculator (BD_wrapper)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mriener/BD_wrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
