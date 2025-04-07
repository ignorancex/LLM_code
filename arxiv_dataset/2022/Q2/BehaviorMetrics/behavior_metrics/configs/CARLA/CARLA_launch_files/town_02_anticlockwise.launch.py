import os
import launch
from ament_index_python.packages import get_package_share_directory

from utils.logger import logger


def generate_launch_description():
    ld = launch.LaunchDescription([
        # Declare launch arguments
        launch.actions.DeclareLaunchArgument(
            name='host',
            default_value='localhost'
        ),
        launch.actions.DeclareLaunchArgument(
            name='port',
            default_value='2000'
        ),
        launch.actions.DeclareLaunchArgument(
            name='timeout',
            default_value='10'
        ),
        launch.actions.DeclareLaunchArgument(
            name='role_name',
            default_value='ego_vehicle'
        ),
        launch.actions.DeclareLaunchArgument(
            name='vehicle_filter',
            default_value='vehicle.*'
        ),
        launch.actions.DeclareLaunchArgument(
            name='spawn_point',
            default_value='10.0, -307.0, 1.37, 0.0, 0.0, 0.0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='town',
            default_value='Town02'
        ),
        launch.actions.DeclareLaunchArgument(
            name='passive',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='synchronous_mode_wait_for_vehicle_control_command',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='fixed_delta_seconds',
            default_value='0.05'
        ),
        # Declare launch argument for quality (e.g., Low, Epic, High, etc.)
        launch.actions.DeclareLaunchArgument(
            name='quality',
            default_value='High'
        ),

        # Set the environment variable 'QUALITY' based on the 'quality' launch argument.
        # This ensures that the launched processes can access the value via os.environ.get('QUALITY')
        launch.actions.SetEnvironmentVariable(
            name='QUALITY',
            value=launch.substitutions.LaunchConfiguration('quality')
        ),
        

        # Log information for debugging purposes
        launch.actions.LogInfo(
            msg=['Spawn point: ', launch.substitutions.LaunchConfiguration('spawn_point')]
        ),
        launch.actions.LogInfo(
            msg=['Quality: ', launch.substitutions.LaunchConfiguration('quality')]
        ),

        # Include the carla_ros_bridge launch file
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('carla_ros_bridge'),
                    'carla_ros_bridge.launch.py'
                )
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'town': launch.substitutions.LaunchConfiguration('town'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'passive': launch.substitutions.LaunchConfiguration('passive'),
                'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command'),
                'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds'),
                'quality': launch.substitutions.LaunchConfiguration('quality')
            }.items()
        ),

        # Include the carla_spawn_objects launch file
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('carla_spawn_objects'),
                    'carla_example_ego_vehicle.launch.py'
                )
            ),
            launch_arguments={
                'vehicle_filter': launch.substitutions.LaunchConfiguration('vehicle_filter'),
                'role_name': launch.substitutions.LaunchConfiguration('role_name'),
                'spawn_point': launch.substitutions.LaunchConfiguration('spawn_point')
            }.items()
        ),

        # Include the carla_manual_control launch file
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('carla_manual_control'),
                    'carla_manual_control.launch.py'
                )
            ),
            launch_arguments={
                'role_name': launch.substitutions.LaunchConfiguration('role_name')
            }.items()
        )
    ])
    return ld
