
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lila",
    version="0.0.0",
    author="Mahdi Qezlou, Simeon Bird, Adam Lidz, Guochao Sun, Andrew B. Newman",
    author_email="mahdi.qezlou@email.ucr.edu",
    description="Line Intensity map X Ly-Alpha forest forecast (LILA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahdiqezlou/lali",
    project_urls={
        "Bug Tracker": "https://github.com/mahdiqezlou/lali",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires="<=3.9",
)
