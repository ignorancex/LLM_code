#include <ros_crs_utils/parameter_io.h>
#include "mpc_solvers/kinematic_nonconvex_mpc_solver.h"

#include <mpc_controller/kinematic_controller/tracking_mpc_kinematic_config.h>
#include <mpc_controller/kinematic_controller/tracking_mpc_kinematic_nonconvex_config.h>

namespace parameter_io
{
template <>
crs_controls::tracking_mpc_kinematic_config
getConfig<crs_controls::tracking_mpc_kinematic_config>(const ros::NodeHandle& nh)
{
  crs_controls::tracking_mpc_kinematic_config params;

  if (!nh.getParam("Q1", params.Q1))
    ROS_WARN_STREAM(" getConfig<crs_controls::tracking_mpc_kinematic_config>: did not load Q1");
  if (!nh.getParam("Q2", params.Q2))
    ROS_WARN_STREAM(" getConfig<crs_controls::tracking_mpc_kinematic_config>: did not load Q2");
  if (!nh.getParam("R1", params.R1))
    ROS_WARN_STREAM(" getConfig<crs_controls::tracking_mpc_kinematic_config>: did not load R1");
  if (!nh.getParam("R2", params.R2))
    ROS_WARN_STREAM(" getConfig<crs_controls::tracking_mpc_kinematic_config>: did not load R2");
  if (!nh.getParam("lag_compensation_time", params.lag_compensation_time))
    ROS_WARN_STREAM(" getConfig<crs_controls::tracking_mpc_kinematic_config>: did not load lag_compensation_time");
  if (!nh.getParam("solver_type", params.solver_type))
    ROS_WARN_STREAM(" getConfig<crs_controls::tracking_mpc_kinematic_config>: did not load solver_type");

  return params;
}

template <typename T>
static void getParam(const ros::NodeHandle& nh, std::string key, T& param)
{
  if (!nh.getParam(key, param))
  {
    ROS_WARN_STREAM(" getConfig: did not load " + key);
  }
}

template <>
crs_controls::tracking_mpc_kinematic_nonconvex_config
getConfig<crs_controls::tracking_mpc_kinematic_nonconvex_config>(const ros::NodeHandle& nh)
{
  mpc_solvers::kinematic_solvers::mpc_nonconvex_solver_config solver_config;

  getParam(nh, "model_bounds/x_min", solver_config.xmin);
  getParam(nh, "model_bounds/x_max", solver_config.xmax);
  getParam(nh, "model_bounds/y_min", solver_config.ymin);
  getParam(nh, "model_bounds/y_max", solver_config.ymax);
  getParam(nh, "solver_params/N", solver_config.horizon_length);
  getParam(nh, "solver_params/num_obstacles", solver_config.num_obstacles);
  getParam(nh, "solver_params/num_segments", solver_config.num_segments);
  getParam(nh, "solver_params/euclidean_offset_cost", solver_config.euclidean_offset_cost);

  crs_controls::tracking_mpc_kinematic_nonconvex_config config;

  getParam(nh, "controller_params/Q1", config.Q1);
  getParam(nh, "controller_params/Q2", config.Q2);
  getParam(nh, "controller_params/R1", config.R1);
  getParam(nh, "controller_params/R2", config.R2);
  getParam(nh, "controller_params/lag_compensation_time", config.lag_compensation_time);
  getParam(nh, "controller_params/solver_type", config.solver_type);
  config.solver_config = solver_config;

  return config;
}

}  // namespace parameter_io
