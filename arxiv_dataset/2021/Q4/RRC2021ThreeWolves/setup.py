import os
import setuptools

PACKAGE_NAME = "three_wolves"

setuptools.setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    # Packages to export
    packages=setuptools.find_packages(),
    data_files=[
        # Install "marker" file in package index
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        # Include our package.xml file
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
    ],
    # This is important as well
    install_requires=["setuptools"],
    zip_safe=True,
    author="Jilong Wang",
    author_email="42jaylonw@gmail.com",
    maintainer="Jilong Wang",
    maintainer_email="42jaylonw@gmail.com",
    description="Three Wolves package for the Real Robot Challenge Submission System.",
    license="BSD 3-clause",
    # Like the CMakeLists add_executable macro, you can add your python
    # scripts here.
    entry_points={
        "console_scripts": [
            "sim_cube_trajectory = three_wolves.scripts.sim_cube_trajectory:main",
            "real_cube_trajectory = three_wolves.scripts.real_cube_trajectory:main",
        ],
    },
)
