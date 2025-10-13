from setuptools import find_packages, setup

package_name = 'simple_sim'

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
    description='Simple simulation in gazebo with randomly spawning objects',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_spawner = simple_sim.object_spawner:main',
            'projection_publisher = simple_sim.projection_publisher:main',
            'needs_publisher = simple_sim.needs_publisher:main'
        ],
    },
)
