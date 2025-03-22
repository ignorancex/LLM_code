import setuptools
import os

# read requirements.txt
this_folder = os.path.dirname(os.path.realpath(__file__))
req_path = this_folder + "/requirements.txt"
install_requires = []
if os.path.isfile(req_path):
    with open(req_path, "r") as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="zero-shot-mt",
    version="0.0.1",
    maintainer="Anonymous EACL Author",
    maintainer_email="",
    description="Zero-Shot MT system",
    url="",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=["Programming Language :: Python :: 3",],
    python_requires=">=3.7",
    install_requires=install_requires
)
