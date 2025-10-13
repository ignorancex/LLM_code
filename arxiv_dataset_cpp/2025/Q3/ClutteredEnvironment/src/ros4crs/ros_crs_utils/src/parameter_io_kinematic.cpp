#include <ros_crs_utils/parameter_io.h>
#include <kinematic_model/kinematic_params.h>
#include <kinematic_model/kinematic_car_input.h>
#include <kinematic_model/kinematic_car_state.h>
namespace parameter_io
{

template <>
crs_models::kinematic_model::kinematic_car_state getState(const ros::NodeHandle& nh)
{
  crs_models::kinematic_model::kinematic_car_state state;
  std::vector<double> state_as_vec;
  if (!nh.getParam("value", state_as_vec))
  {
    ROS_WARN_STREAM("Could not load initial state from parameters.");
  }
  else
  {
    state.pos_x = state_as_vec[0];
    state.pos_y = state_as_vec[1];
    state.yaw = state_as_vec[2];
    state.velocity = state_as_vec[3];
  }
  return state;
}

template <>
crs_models::kinematic_model::kinematic_car_input getInput(const ros::NodeHandle& nh)
{
  crs_models::kinematic_model::kinematic_car_input input;
  std::vector<double> input_as_vec;
  if (!nh.getParam("value", input_as_vec))  // Load initial input from params
                                            // (pacejka_car_ekf.yaml)
  {
    ROS_WARN_STREAM("Could not load initial input!");
  }
  else
  {
    input.steer = input_as_vec[0];
    input.torque = input_as_vec[1];
  }
  return input;
}

template <>
void getModelParams<crs_models::kinematic_model::kinematic_params>(
    const ros::NodeHandle& nh, crs_models::kinematic_model::kinematic_params& params, bool verbose)
{
  if (!nh.getParam("lf", params.lf) && verbose)
    ROS_WARN_STREAM(" getModelParams<crs_models::kinematic_model::kinematic_params>: did not load lf. Namespace: "
                    << nh.getNamespace());
  if (!nh.getParam("lr", params.lr) && verbose)
    ROS_WARN_STREAM(
        " getModelParams<crs_models::kinematic_model::kinematic_params>: did not load lr. Ns:" << nh.getNamespace());
  if (!nh.getParam("tau", params.tau) && verbose)
    ROS_WARN_STREAM(
        " getModelParams<crs_models::kinematic_model::kinematic_params>: did not load tau. Ns:" << nh.getNamespace());

  if (!nh.getParam("a", params.a) && verbose)
    ROS_WARN_STREAM(
        " getModelParams<crs_models::kinematic_model::kinematic_params>: did not load a. Ns:" << nh.getNamespace());
  if (!nh.getParam("b", params.b) && verbose)
    ROS_WARN_STREAM(
        " getModelParams<crs_models::kinematic_model::kinematic_params>: did not load b. Ns:" << nh.getNamespace());

  // Optional parameters
  if (!nh.getParam("width", params.car_width) && verbose)
    ROS_INFO_STREAM(" getModelParams<crs_models::kinematic_model::kinematic_params>: did not load car_width. Ns:"
                    << nh.getNamespace());
  if (!nh.getParam("length", params.car_length) && verbose)
    ROS_INFO_STREAM(" getModelParams<crs_models::kinematic_model::kinematic_params>: did not load car_length. Ns:"
                    << nh.getNamespace());

  // TODO: These values probably shouldn't be associated with the model, but instead with the controller
  if (!nh.getParam("min_dist_to_obstacle", params.min_dist_to_obstacle) && verbose)
    ROS_INFO_STREAM(
        " getModelParams<crs_models::kinematic_model::kinematic_params>: did not load min_dist_to_obstacle. Ns:"
        << nh.getNamespace());
  if (!nh.getParam("additional_buffer", params.additional_buffer) && verbose)
    ROS_INFO_STREAM(
        " getModelParams<crs_models::kinematic_model::kinematic_params>: did not load min_dist_to_obstacle. Ns:"
        << nh.getNamespace());
}

}  // namespace parameter_io
