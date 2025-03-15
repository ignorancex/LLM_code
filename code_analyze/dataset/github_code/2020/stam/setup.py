from distutils.core import setup
import setuptools

setup(name='stam',
      version='0.2.3',
      author="Na'ama Hallakoun",
      author_email='naama.hallakoun@weizmann.ac.il',
      description='Stellar-Track-based Assignment of Mass',
      long_description=open('README.md').read(),
      url='https://github.com/naamach/stam',
      license='LICENSE.txt',
      packages=setuptools.find_packages(),
      install_requires=['astropy', 'configparser', 'numpy', 'scipy', 'tqdm', 'geomdl', 'matplotlib'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Astronomy']
      )
