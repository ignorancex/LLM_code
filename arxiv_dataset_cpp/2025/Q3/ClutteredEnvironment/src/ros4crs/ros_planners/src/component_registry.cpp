#include "ros_planners/component_registry.h"

#ifdef kinematic_model_FOUND
#include <kinematic_model/kinematic_car_state.h>
#include <kinematic_model/kinematic_car_input.h>
#include "kinematic_model/kinematic_discrete.h"
#endif

#include <crs_msgs/car_state_cart.h>
#include <crs_msgs/car_input.h>
#include <ros_crs_utils/parameter_io.h>
#include <ros_crs_utils/obstacle/obstacle_updater.h>

#include <commons/obstacle.h>
#include <planning/cartesian_reference_point.h>
#include <ros_planners/ros_planner.h>
#include <segment_planner/segment_planner.h>

#include "ros_planners/visualizers/segment_planner_visualizer.h"

// Don't let this object go out of scope
message_conversion::ObstacleUpdater* obstacle_updater;

/**
 * @brief This file loads the specific controller implementation and wraps it inside a ros controller object
 *
 */
namespace ros_planner
{
template <>
ros_planner::RosPlanner<crs_msgs::car_state_cart, crs_models::kinematic_model::kinematic_car_state,
                        crs_planning::cartesian_reference_point>*
resolvePlanner(ros::NodeHandle& nh, ros::NodeHandle& nh_private, const std::string& planner_type)
{
  // Currently, the segment planner only works kbm
  using StateMsg = crs_msgs::car_state_cart;
  using StateType = crs_models::kinematic_model::kinematic_car_state;
  using TrajectoryType = crs_planning::cartesian_reference_point;
  using ModelParams = crs_models::kinematic_model::kinematic_params;

  // Load the model parameters

  ModelParams params;
  parameter_io::getModelParams<ModelParams>(ros::NodeHandle(nh, "model/model_params/"), params);

  auto model = std::make_unique<crs_models::kinematic_model::DiscreteKinematicModel>(params);

  // Create Planner
  SegmentPlanner::Config planner_config = parameter_io::getConfig<SegmentPlanner::Config>(nh);
  auto planner_ptr = std::make_shared<crs_planning::SegmentPlanner>(
      std::move(model), std::make_unique<SegmentPlanner::Config>(planner_config));

  // Downcast to BasePlanner type
  auto derived_ptr = std::dynamic_pointer_cast<crs_planning::BasePlanner<
      crs_planning::cartesian_reference_point, crs_models::kinematic_model::kinematic_car_state>>(planner_ptr);

  auto visualizer_ptr = std::make_shared<SegmentPlannerVisualizer>(nh_private, planner_ptr);
  auto derived_visualizer_ptr = std::dynamic_pointer_cast<BasePlannerVisualizer>(visualizer_ptr);

  parameter_io::Coords target_coords = parameter_io::getCoords(nh, "planner_node/initial_target");
  bool start_immediately = false;
  nh.getParam("planner_node/start_immediately", start_immediately);
  DynamicTarget::Params target_params = { start_immediately, target_coords.x, target_coords.y };

  ros_planner::RosPlanner<StateMsg, StateType, TrajectoryType>::SubscriptionInfo subscriptions = {
    .initial_state = true,
    .initial_obstacles = true,
    .trajectory = true,
    .target_info = { target_params },
  };

  return new ros_planner::RosPlanner<StateMsg, StateType, TrajectoryType>(nh, nh_private, derived_ptr, subscriptions,
                                                                          derived_visualizer_ptr);
}

}  // namespace ros_planner
