from setuptools import find_packages, setup

package_name = 'camera_orientation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tin',
    maintainer_email='misic.tin@gmail.com',
    description='Package used to change camera orientation in gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turn_cam = camera_orientation.turn_cam:main',
            'test_orientation_publisher = camera_orientation.test_orientation_publisher:main'
        ],
    },
)
