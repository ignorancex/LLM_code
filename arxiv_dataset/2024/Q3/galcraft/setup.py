from setuptools import setup, find_packages
import os

def readme():
    with open('README.md') as file:
        return(file.read())

def versionNumber():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GalCraft/_version.py')) as versionFile:
        return(versionFile.readlines()[-1].split()[-1].strip("\"'"))

setup(name='GalCraft',
      version=versionNumber(),
      description='GalCraft: Building integral-field spectrograph data cubes of the Galaxy',
      long_description_content_type="text/markdown",
      long_description=readme(),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
      ],
      url='https://github.com/purmortal/galcraft',
      author='Zixian Wang (Purmortal)',
      author_email='wang.zixian.astro@gmail.com',
      packages=find_packages(),
      package_data={
          '': ['**/*']
      },
      install_requires=[
          'astropy',
          'numpy',
          'scipy',
          'matplotlib',
          'ebfpy',
          'ephem',
          'spectres',
      ],
      python_requires='>=3.6',
      entry_points={
          'console_scripts': [
              'GalCraft        = GalCraft.MainProcess:main'
          ],
      },
      include_package_data=True,
      zip_safe=False)
