from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


licenses = {
    "apache2": (
        "Apache Software License 2.0",
        "OSI Approved :: Apache Software License",
    ),
    "mit": ("MIT License", "OSI Approved :: MIT License"),
    "gpl2": (
        "GNU General Public License v2",
        "OSI Approved :: GNU General Public License v2 (GPLv2)",
    ),
    "gpl3": (
        "GNU General Public License v3",
        "OSI Approved :: GNU General Public License v3 (GPLv3)",
    ),
    "bsd3": ("BSD License", "OSI Approved :: BSD License"),
}

py_versions = "3.8 3.9 3.10 3.11".split()

setup(
    name="diffvox",  
    license="MIT",
    version="0.1.0",  
    author="Hossein Momeni",
    author_email="momeni.hossein80@gmail.com",
    description="Voxel grid optimization in 3D reconstruction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hossein-momeni/DiffVox", 
    packages=find_packages(where="."),  # Search for packages in the current directory
    package_dir={'diffvox': 'diffvox'},  # Map 'diffvox' to the inner directory
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ]
    + [
        f"Programming Language :: Python :: {py}"
        for py in py_versions
    ],
    python_requires=">=3.8",  # Specify Python version compatibility
    install_requires=parse_requirements("requirements.txt"),
    include_package_data=True,
    keywords="3D reconstruction voxel grid optimization",
)
