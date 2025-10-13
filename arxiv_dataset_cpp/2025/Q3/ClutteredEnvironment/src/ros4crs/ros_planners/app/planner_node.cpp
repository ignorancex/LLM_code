#include "ros_planners/component_registry.h"
#include <planning/cartesian_reference_point.h>
#include <planning/multi_car_cartesian_reference_point.h>
#include <interactive_markers/interactive_marker_server.h>
#include <ros/ros.h>

#include <crs_msgs/car_input.h>
#include <crs_msgs/car_state_cart.h>

// Required reference so object does not go out of scope and gets dereferenced
void* planner_ptr;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ros_planner");
  ros::NodeHandle nh = ros::NodeHandle("");           // /<NAMESPACE>/*
  ros::NodeHandle nh_private = ros::NodeHandle("~");  // /<NAMESPACE>/ros_planner/*

  std::string planner_type;
  std::string state_type;
  nh_private.getParam("planner_type", planner_type);  // UNUSED CURRENTLY
  nh_private.getParam("state_type", state_type);

#ifdef kinematic_model_FOUND
  if (state_type == "kinematic_car")
  {
    if (planner_type == "segment_planner")
    {
      using StateMsg = crs_msgs::car_state_cart;
      using StateType = crs_models::kinematic_model::kinematic_car_state;
      using TrajectoryType = crs_planning::cartesian_reference_point;
      planner_ptr =
          (void*)ros_planner::resolvePlanner<StateMsg, StateType, TrajectoryType>(nh, nh_private, planner_type);
      ROS_INFO("loaded segment planner");
    }
  }
#endif

  if (!planner_ptr)
  {
    ROS_ERROR_STREAM("Unknown state type: " << state_type << ". Aborting!");
    return 1;
  }
  ros::MultiThreadedSpinner spinner(2);  // Use 1 threads
  spinner.spin();                        // spin() will not return until the node has been shutdown
  return 0;
}
