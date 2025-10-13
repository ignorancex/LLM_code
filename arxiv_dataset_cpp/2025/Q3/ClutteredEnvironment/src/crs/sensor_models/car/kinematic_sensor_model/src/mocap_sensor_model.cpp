
#include "kinematic_sensor_model/mocap_sensor_model.h"
#include <dynamic_models/utils/data_conversion.h>

namespace crs_sensor_models
{
namespace kinematic_sensor_models
{
// Option 2: Create an object and specify the process noise covariance matrix Q yourself.
MocapSensorModel::MocapSensorModel(const Eigen::Matrix3d& R)
  : SensorModel(3, MocapSensorModel::SENSOR_KEY)  // Measurement dimension is three
{
  std::vector<casadi::MX> measured_states_mx = { state_mx[0], state_mx[1], state_mx[2] };
  measurement_function = casadi::Function("applyMeasurementModel", state_and_input_mx, measured_states_mx);

  setR(R);
}

const std::string MocapSensorModel::SENSOR_KEY = "mocap";

}  // namespace kinematic_sensor_models
}  // namespace crs_sensor_models
