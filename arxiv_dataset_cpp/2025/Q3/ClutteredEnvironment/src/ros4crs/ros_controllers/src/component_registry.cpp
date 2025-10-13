#include <functional>

#include "ros_controllers/component_registry.h"
#include "ros_controllers/ros_controller.h"
#include <ros/ros.h>
#include <ros_crs_utils/parameter_io.h>

#include <commons/base_trajectory.h>
#include <commons/dynamic_point_trajectory.h>

#ifdef mpc_controller_FOUND
#ifdef kinematic_model_FOUND
#include <mpc_controller/kinematic_controller/tracking_mpc_kinematic_controller.h>
#include <mpc_controller/kinematic_controller/tracking_mpc_kinematic_nonconvex_controller.h>
#include <kinematic_model/kinematic_car_input.h>
#include <kinematic_model/kinematic_car_state.h>
#endif
#endif

#include "ros_controllers/visualizers/mpc_multiphase_controller_visualizer.h"
#include "ros_crs_utils/obstacle/obstacle_updater.h"

#include <crs_msgs/car_state_cart.h>
#include <crs_msgs/car_input.h>

message_conversion::ObstacleUpdater* obstacle_updater = nullptr;

/**
 * @brief This file loads the specific controller implementation and wraps it inside a ros controller object
 *
 */
namespace ros_controllers
{
template <typename ModelType, typename StateType, typename InputType>
std::unique_ptr<BaseControllerVisualizer<StateType, InputType>>
loadControllerVisualizer(ros::NodeHandle nh,
                         std::shared_ptr<crs_controls::BaseController<StateType, InputType>> controller)
{
  std::string type;
  if (!nh.getParam("type", type))
  {
    ROS_WARN("No Type specified for visualizer! No visualizer loaded for controller");
    return std::unique_ptr<BaseControllerVisualizer<StateType, InputType>>(nullptr);
  }

  if (type == "mpc_multiphase")
  {
    return std::make_unique<MpcMultiphaseControllerVisualizer<ModelType, StateType, InputType>>(
        nh, std::dynamic_pointer_cast<crs_controls::MpcController<ModelType, StateType, InputType>>(controller));
  }

  ROS_WARN_STREAM("No Visualizer found for type: " << type << ". No visualizer loaded for controller!");
  return std::unique_ptr<BaseControllerVisualizer<StateType, InputType>>(nullptr);
};

#ifdef kinematic_model_FOUND
// Shorten type names
typedef crs_msgs::car_state_cart ros_car_state;
typedef crs_msgs::car_input ros_car_input;
typedef crs_models::kinematic_model::kinematic_car_state kinematic_state;
typedef crs_models::kinematic_model::kinematic_car_input kinematic_input;
typedef crs_models::kinematic_model::kinematic_params kinematic_model_params;
typedef RosController<ros_car_state, ros_car_input, kinematic_state, kinematic_input> kinematic_ros_controller;

#ifdef mpc_controller_FOUND

inline std::shared_ptr<crs_models::kinematic_model::DiscreteKinematicModel>
getKinematicModel(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
{
  // Load model from params
  kinematic_model_params kinematic_params;

  // First load generic, gt model
  parameter_io::getModelParams<kinematic_model_params>(ros::NodeHandle(nh, "model/model_params"), kinematic_params);

  // Patch certain params
  parameter_io::getModelParams<kinematic_model_params>(ros::NodeHandle(nh_private, "model/model_params"),
                                                       kinematic_params, false);

  // Load model from params
  return std::make_shared<crs_models::kinematic_model::DiscreteKinematicModel>(kinematic_params);
}

inline kinematic_ros_controller* getKinematicTrackingMPCController(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
                                                                   void*& dynamic_callback_allocator, bool nonconvex)
{
  std::shared_ptr<crs_models::kinematic_model::DiscreteKinematicModel> kinematic_model =
      getKinematicModel(nh, nh_private);

  // start track points
  /* std::vector<double> x_start = { 0.3 };
  std::vector<double> y_start = { 0.0 }; */

  std::shared_ptr<crs_controls::StaticTrackTrajectory> static_track =
      parameter_io::loadTrackDescriptionFromParams(ros::NodeHandle(nh, "track"));
  std::cout << "loading track" << std::endl;

  std::cout << "loading reference" << std::endl;

  // Ptr to store either the Nonconvex or the normal tracking mpc controller
  std::shared_ptr<crs_controls::MpcController<crs_models::kinematic_model::DiscreteKinematicModel, kinematic_state,
                                              kinematic_input>>
      derived_ptr = nullptr;

  // std::shared_ptr<crs_controls::DynamicPointTrajectory> dynamic_ref =
  //     parameter_io::loadReferenceFromParams(ros::NodeHandle(nh, "reference"));

  // TODO: make config option for reading dynamic point trajectory from a file or initializaing
  std::shared_ptr<crs_controls::DynamicPointTrajectory> dynamic_ref =
      std::make_shared<crs_controls::DynamicPointTrajectory>();

  if (nonconvex)
  {
    // Read the config
    crs_controls::tracking_mpc_kinematic_nonconvex_config cfg =
        parameter_io::getConfig<crs_controls::tracking_mpc_kinematic_nonconvex_config>(nh_private);

    // Create the controller, and downcast to BaseController type
    auto ptr =
        std::make_shared<crs_controls::kinematic_controllers::NonconvexController>(cfg, kinematic_model, dynamic_ref);
    derived_ptr =
        std::dynamic_pointer_cast<crs_controls::MpcController<crs_models::kinematic_model::DiscreteKinematicModel,
                                                              kinematic_state, kinematic_input>>(ptr);

    message_conversion::ObstacleUpdater::Callback callback =
        boost::bind(&crs_controls::kinematic_controllers::NonconvexController::setObstacles, ptr, _1);
    obstacle_updater = new message_conversion::ObstacleUpdater(nh, callback);
  }
  else
  {
    // Read the config
    crs_controls::tracking_mpc_kinematic_config cfg =
        parameter_io::getConfig<crs_controls::tracking_mpc_kinematic_config>(
            ros::NodeHandle(nh_private, "controller_params"));

    // Create the controller, and downcast to BaseController type
    auto ptr = std::make_shared<crs_controls::KinematicTrackingMpcController>(cfg, kinematic_model, dynamic_ref);
    derived_ptr =
        std::dynamic_pointer_cast<crs_controls::MpcController<crs_models::kinematic_model::DiscreteKinematicModel,
                                                              kinematic_state, kinematic_input>>(ptr);
  }

  auto visualizer_ptr =
      loadControllerVisualizer<crs_models::kinematic_model::DiscreteKinematicModel, kinematic_state, kinematic_input>(
          ros::NodeHandle(nh_private, "visualizer"), derived_ptr);

  return new kinematic_ros_controller(nh, nh_private, std::move(visualizer_ptr), derived_ptr);
}

#endif  // mpc_controller_FOUND
#endif  // kinematic_model_FOUND

#ifdef kinematic_model_FOUND
template <>
kinematic_ros_controller* resolveController<ros_car_state, ros_car_input, kinematic_state, kinematic_input>(
    ros::NodeHandle& nh, ros::NodeHandle& nh_private, const std::string& controller_type,
    void*& dynamic_callback_allocator)
{
#ifdef mpc_controller_FOUND
  if (controller_type == "KINEMATIC_TRACKING_MPC")
  {
    return getKinematicTrackingMPCController(nh, nh_private, dynamic_callback_allocator, false);
  }
  else if (controller_type == "KINEMATIC_NONCONVEX_MPC")
  {
    return getKinematicTrackingMPCController(nh, nh_private, dynamic_callback_allocator, true);
  }
#endif  // mpc_controller_FOUND

  assert(true && "Did not find registered controller for specified controller type.");
  return nullptr;
}
#endif  // kinematic_model_FOUND

}  // namespace ros_controllers
