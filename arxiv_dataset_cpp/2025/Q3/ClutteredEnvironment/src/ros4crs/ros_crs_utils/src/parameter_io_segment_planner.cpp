
#include <ros_crs_utils/parameter_io.h>
#include <segment_planner/segment_planner.h>

namespace parameter_io
{
template <typename T>
static void getParam(const ros::NodeHandle& nh, std::string key, T& param)
{
  if (!nh.getParam(key, param))
  {
    ROS_WARN_STREAM(" getConfig: did not load " + key);
  }
}

template <>
crs_planning::SegmentPlanner::Config getConfig<crs_planning::SegmentPlanner::Config>(const ros::NodeHandle& nh)
{
  crs_planning::SegmentPlanner::Config config;

  getParam(nh, "controller_node/solver_params/num_segments", config.num_segments);
  getParam(nh, "controller_node/solver_params/euclidean_offset_cost", config.euclidean_offset_cost);
  getParam(nh, "planner_node/planner_params/goal_reached_buffer", config.goal_reached_buffer);

  getParam(nh, "controller_node/model_bounds/x_min", config.bounds.x_min);
  getParam(nh, "controller_node/model_bounds/x_max", config.bounds.x_max);
  getParam(nh, "controller_node/model_bounds/y_min", config.bounds.y_min);
  getParam(nh, "controller_node/model_bounds/y_max", config.bounds.y_max);

  if (config.euclidean_offset_cost)
  {
    ROS_ASSERT(config.num_segments == 1);
  }

  return config;
}

}  // namespace parameter_io
