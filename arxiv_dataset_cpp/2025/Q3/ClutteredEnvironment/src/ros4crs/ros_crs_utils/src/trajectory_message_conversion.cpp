#include "ros_crs_utils/trajectory_message_conversion.h"
#include <planning/cartesian_reference_point.h>
#include <planning/multi_car_cartesian_reference_point.h>
#include <vector>

namespace message_conversion
{
// TODO, we are really just hijakcing the joint trajectory format. There must either be a better suited message or we
// should define our own
template <>
trajectory_msgs::JointTrajectory
convertToRosTrajectory(const std::vector<crs_planning::cartesian_reference_point>& trajectory,
                       std::string car_name /* = "" */)
{
  trajectory_msgs::JointTrajectory ros_trajectory;
  ros_trajectory.joint_names.push_back(car_name);
  for (const auto& pt : trajectory)
  {
    trajectory_msgs::JointTrajectoryPoint ros_pt;
    ros_pt.positions.push_back(pt.x);
    ros_pt.positions.push_back(pt.y);
    ros_pt.positions.push_back(0);  // z = 0;
    ros_trajectory.points.push_back(ros_pt);
  }
  return ros_trajectory;
}

template <>
trajectory_msgs::JointTrajectory
convertToRosTrajectory(const std::vector<crs_planning::multi_car_cartesian_reference_point>& trajectory,
                       std::string car_name /* = "" */)
{
  trajectory_msgs::JointTrajectory ros_trajectory;
  ros_trajectory.joint_names.push_back(car_name);

  int car_index = 0;
  if (!trajectory.empty())
  {
    for (car_index = 0; car_index < trajectory[0].namespaces.size(); car_index++)
    {
      if (trajectory[0].namespaces[car_index] == car_name)
        break;
    }
  }

  for (const auto& pt : trajectory)
  {
    trajectory_msgs::JointTrajectoryPoint ros_pt;
    ros_pt.positions.push_back(pt.points[car_index].x);
    ros_pt.positions.push_back(pt.points[car_index].y);
    ros_pt.positions.push_back(0);  // z = 0;
    ros_trajectory.points.push_back(ros_pt);
  }
  return ros_trajectory;
}

template <>
trajectory_msgs::JointTrajectory convertToRosTrajectory(const std::vector<std::vector<double>>& trajectory,
                                                        std::string car_name /* = "" */)
{
  // We continue with the abuse of the joint trajectory message here:
  // For each index in the horizon, the position is a vector [x, y, z]
  // and the velocities are [vx, vy, vz] in body frame.
  trajectory_msgs::JointTrajectory ros_trajectory;
  ros_trajectory.joint_names.push_back(car_name);

  for (const auto& pt : trajectory)
  {
    trajectory_msgs::JointTrajectoryPoint ros_pt;
    ros_pt.positions.push_back(pt[0]);
    ros_pt.positions.push_back(pt[1]);
    ros_pt.positions.push_back(0);  // z = 0

    ros_pt.velocities.push_back(pt[2]);
    ros_pt.velocities.push_back(pt[3]);
    ros_pt.velocities.push_back(0);  // vz = 0

    ros_trajectory.points.push_back(ros_pt);
  }

  return ros_trajectory;
}

std::vector<std::vector<double>> convertFromRosTrajectory(const trajectory_msgs::JointTrajectory& ros_trajectory)
{
  // Note: ideally we'd have a different version for each type of controller trajectory

  std::vector<std::vector<double>> trajectory;
  for (const auto& pt : ros_trajectory.points)
  {
    double x = pt.positions[0];
    double y = pt.positions[1];
    double vx = pt.velocities[0];
    double vy = pt.velocities[1];
    trajectory.push_back({ x, y, vx, vy });
  }
  return trajectory;
}

geometry_msgs::PolygonStamped convertToRosVoronoi(const std::vector<double> vor_edges_x,
                                                  const std::vector<double> vor_edges_y)
{
  geometry_msgs::PolygonStamped poly_msg;
  poly_msg.header.frame_id = "crs_frame";
  for (int i = 0; i < vor_edges_x.size(); i++)
  {
    geometry_msgs::Point32 point;
    point.x = vor_edges_x[i];
    point.y = vor_edges_y[i];
    poly_msg.polygon.points.push_back(point);
  }
  return poly_msg;
}

}  // namespace message_conversion
