from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aif_model'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install the resource folder to the share directory
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
        (os.path.join('share', package_name), ['package.xml']),
        # Install launch folder to the share directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tin',
    maintainer_email='misic.tin@gmail.com',
    description='Active inference model for visual attention',
    license='Apache2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'act_inf = aif_model.act_inf:main',
            'auto_trial = aif_model.auto_trial:main'
        ],
    },
)
