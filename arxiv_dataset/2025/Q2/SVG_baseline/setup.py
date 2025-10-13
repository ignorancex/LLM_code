from setuptools import setup, find_packages

setup(
    name="va_core",
    version="0.0.0",
    description="Core modules for V&A generation research.",
    author="A. Hayakawa and M. Ishii",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)