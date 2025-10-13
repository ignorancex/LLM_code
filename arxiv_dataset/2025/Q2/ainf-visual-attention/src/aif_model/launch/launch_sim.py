import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the Gazebo launch file
    another_launch_path = os.path.join(
        get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py'
    )

    return launch.LaunchDescription([
        # Include Gazebo launch description with a world argument
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(another_launch_path),
            launch_arguments={'world': 'worlds/static_test.world'}.items(),
        ),
        # Execute the process to publish to the /needs topic
        ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '/needs', 'std_msgs/msg/Float32MultiArray', '{data: [0.0, 0.0, 1.0]}'],
            output='log'
        ),
        # Launch camera_orientation node
        launch_ros.actions.Node(
            package='camera_orientation',
            executable='turn_cam',
            name='turn_cam_node',
            output='log'
        ),
        # Launch aif_model node
        launch_ros.actions.Node(
            package='aif_model',
            executable='act_inf',
            name='act_inf_node',
            output='screen'
        )
    ])
