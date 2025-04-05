import setuptools

from glob import glob


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randomised_filtering",
    version="0.0.1",
    author="Antonia Hain, Daniel Jorgens",
    author_email="antoniabhain@gmail.com, djoerch@gmail.com",
    description="Randomised tractogram filtering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="github.com/djoerch/randomised_filtering",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "dipy>=1.3.0",
        "nibabel>=3.0.2",
        "numpy>=1.18.0",
        "tqdm",
        "matplotlib",
    ],
    scripts=glob("scripts/*.py"),
)
