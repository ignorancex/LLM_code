#include "ros_simulator/component_registry.h"
#include <ros_crs_utils/parameter_io.h>

#ifdef kinematic_model_FOUND
#include "ros_simulator/ros_kinematic_simulator.h"
#include <kinematic_sensor_model/mocap_sensor_model.h>
#endif

namespace ros_simulator
{
Simulator* resolveSimulator(ros::NodeHandle& nh, ros::NodeHandle& nh_private, const std::string& state_type,
                            const std::string& input_type, const std::vector<std::string>& sensors_to_load)
{
#ifdef kinematic_model_FOUND
  if (state_type == "kinematic_car" && input_type == "kinematic_car")
  {
    // Create kinematic_simulator simulator
    auto* kinematic_simulator = new ros_simulator::KinematicSimulator(nh, nh_private);
    for (auto sensor_name : sensors_to_load)  // Parse sensor models
    {
      std::string sensor_key;
      if (!nh_private.getParam("sensors/" + sensor_name + "/key", sensor_key))  // Load sensor key from params
                                                                                // (pacejka_car_simulator.yaml)
        sensor_key = sensor_name;
      // Measurement Delay
      double delay = 0.0;
      nh_private.getParam("sensors/" + sensor_name + "/delay", delay);

      if (sensor_key == crs_sensor_models::kinematic_sensor_models::MocapSensorModel::SENSOR_KEY)
      {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        parameter_io::getMatrixFromParams<3, 3>(ros::NodeHandle(nh_private, "sensors/" + sensor_name + "/R"),
                                                R);  // Load R from params (pacejka_car_simulator.yaml)

        // Create mocap sensor model using R
        std::shared_ptr<crs_sensor_models::kinematic_sensor_models::MocapSensorModel> mocap_sensor_model =
            std::make_shared<crs_sensor_models::kinematic_sensor_models::MocapSensorModel>(R);

        // Sensor used for simulation
        kinematic_simulator->registerSensorModel(mocap_sensor_model, delay);
      }
      else
      {
        ROS_WARN_STREAM("Unknown sensor model " << sensor_name << ". Sensor model will not be loaded!");
      }
    }
    return kinematic_simulator;
  }
#endif

  return nullptr;
}
}  // namespace ros_simulator
