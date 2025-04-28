from setuptools import setup, find_packages

setup(
    name='MW-Inner-DM-Profile',
    version='0.0.1',  # Replace with your project's version
    description='Data and code for the paper "Journey to the Center of the Milky Way: Constraints on the Inner Dark Matter Distribution" (Hussein et al. 2025)',
    author='Abdelaziz Hussein',
    author_email='abdelh@mit.edu',
    url='https://github.com/abdelazizhussein/MW-Inner-DM-Profile',  # Update with your GitHub URL
    packages=find_packages(),
    install_requires=[
        'astropy',
        'numpy',
        'matplotlib',
        'scipy',
    ],
)
