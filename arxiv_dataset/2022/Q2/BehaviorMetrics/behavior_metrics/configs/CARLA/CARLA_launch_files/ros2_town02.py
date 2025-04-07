import os
import launch
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = launch.LaunchDescription([
        # Basic launch arguments
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
            default_value='None'
        ),
        launch.actions.DeclareLaunchArgument(
            name='town',
            default_value='Town01'
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
        # Weather condition launch arguments
        launch.actions.DeclareLaunchArgument(
            name='cloudiness',
            default_value='30.0'  # Default cloudiness level
        ),
        launch.actions.DeclareLaunchArgument(
            name='precipitation',
            default_value='30.0'  # Default precipitation level
        ),
        launch.actions.DeclareLaunchArgument(
            name='sun_altitude_angle',
            default_value='10.0'  # Default sun altitude angle
        ),
        launch.actions.DeclareLaunchArgument(
            name='sun_azimuth_angle',
            default_value='90.0'  # Default sun azimuth angle
        ),
        launch.actions.DeclareLaunchArgument(
            name='precipitation_deposits',
            default_value='20.0'  # Default precipitation deposits level
        ),
        launch.actions.DeclareLaunchArgument(
            name='wind_intensity',
            default_value='0.0'  # Default wind intensity
        ),
        launch.actions.DeclareLaunchArgument(
            name='fog_density',
            default_value='1.0'  # Default fog density
        ),
        launch.actions.DeclareLaunchArgument(
            name='wetness',
            default_value='0.0'  # Default wetness level
        ),
        # Log weather information for debugging
        launch.actions.LogInfo(
            msg=['Cloudiness: ', launch.substitutions.LaunchConfiguration('cloudiness')]
        ),
        launch.actions.LogInfo(
            msg=['Precipitation: ', launch.substitutions.LaunchConfiguration('precipitation')]
        ),
        launch.actions.LogInfo(
            msg=['Sun Altitude Angle: ', launch.substitutions.LaunchConfiguration('sun_altitude_angle')]
        ),
        launch.actions.LogInfo(
            msg=['Sun Azimuth Angle: ', launch.substitutions.LaunchConfiguration('sun_azimuth_angle')]
        ),
        launch.actions.LogInfo(
            msg=['Precipitation Deposits: ', launch.substitutions.LaunchConfiguration('precipitation_deposits')]
        ),
        launch.actions.LogInfo(
            msg=['Wind Intensity: ', launch.substitutions.LaunchConfiguration('wind_intensity')]
        ),
        launch.actions.LogInfo(
            msg=['Fog Density: ', launch.substitutions.LaunchConfiguration('fog_density')]
        ),
        launch.actions.LogInfo(
            msg=['Wetness: ', launch.substitutions.LaunchConfiguration('wetness')]
        ),
        # Include the carla_ros_bridge launch file
        # Propagate both the basic and weather parameters.
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('carla_ros_bridge'),
                             'carla_ros_bridge.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'town': launch.substitutions.LaunchConfiguration('town'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'passive': launch.substitutions.LaunchConfiguration('passive'),
                'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command'),
                'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds'),
                # Pass weather parameters to carla_ros_bridge (if supported)
                'cloudiness': launch.substitutions.LaunchConfiguration('cloudiness'),
                'precipitation': launch.substitutions.LaunchConfiguration('precipitation'),
                'sun_altitude_angle': launch.substitutions.LaunchConfiguration('sun_altitude_angle'),
                'sun_azimuth_angle': launch.substitutions.LaunchConfiguration('sun_azimuth_angle'),
                'precipitation_deposits': launch.substitutions.LaunchConfiguration('precipitation_deposits'),
                'wind_intensity': launch.substitutions.LaunchConfiguration('wind_intensity'),
                'fog_density': launch.substitutions.LaunchConfiguration('fog_density'),
                'wetness': launch.substitutions.LaunchConfiguration('wetness')
            }.items()
        ),
        # Include the carla_spawn_objects launch file
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('carla_spawn_objects'),
                             'carla_example_ego_vehicle.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'vehicle_filter': launch.substitutions.LaunchConfiguration('vehicle_filter'),
                'role_name': launch.substitutions.LaunchConfiguration('role_name'),
                'spawn_point': launch.substitutions.LaunchConfiguration('spawn_point')
            }.items()
        ),
        # Include the carla_manual_control launch file
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('carla_manual_control'),
                             'carla_manual_control.launch.py')
            ),
            launch_arguments={
                'role_name': launch.substitutions.LaunchConfiguration('role_name')
            }.items()
        )
    ])
    return ld

if __name__ == '__main__':
    generate_launch_description()

