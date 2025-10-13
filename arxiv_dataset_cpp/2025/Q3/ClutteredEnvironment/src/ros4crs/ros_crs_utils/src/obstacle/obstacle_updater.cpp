#include <Eigen/Core>

#include <ros_crs_utils/obstacle/obstacle_updater.h>

namespace message_conversion
{
using crs_controls::ObstacleVector;
using crs_controls::Rhombus;

ObstacleUpdater::ObstacleUpdater(ros::NodeHandle nh, Callback callback)
{
  nh_ = nh;
  sub_ = nh.subscribe("/mocap/all_obstacles", 10, &ObstacleUpdater::obstacleUpdateCallback, this);
  callback_ = callback;
  updated_ = false;
  std::cout << "created ObstacleUpdater" << std::endl;
}

void ObstacleUpdater::obstacleUpdateCallback(const geometry_msgs::PoseArray& msg)
{
  if (updated_)
  {
    return;
  }

  std::unique_ptr<ObstacleVector> obstacles = std::make_unique<ObstacleVector>();
  for (const geometry_msgs::Pose& obstacle_pose : msg.poses)
  {
    // We only support reading rhombuses, for now
    double xCenter = obstacle_pose.position.x;
    double yCenter = obstacle_pose.position.y;

    // Compute the Euler rotation
    // First, create an Eigen quaternion from the pose orientation
    const geometry_msgs::Quaternion& msg_quaternion = obstacle_pose.orientation;
    Eigen::Quaterniond eigen_quaternion = { msg_quaternion.w, msg_quaternion.x, msg_quaternion.y, msg_quaternion.z };
    auto euler_angles = eigen_quaternion.toRotationMatrix().eulerAngles(0, 1, 2);
    double yaw = euler_angles[2];

    obstacles->emplace_back(new Rhombus(xCenter, yCenter, yaw));
  }

  callback_(std::move(obstacles));
  updated_ = true;
}

}  // namespace message_conversion
