// Copyright 2018 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <string>
#include <algorithm>
#include <limits>
#include <memory>
#include <cmath>
#include <filesystem>

#include <boost/make_shared.hpp>
#include <boost/variant.hpp>

#include <gazebo/transport/transport.hh>
#include <gazebo/common/Exception.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/HingeJoint.hh>
#include <gazebo/sensors/SensorTypes.hh>
#include <gazebo/sensors/RaySensor.hh>

#include <gazebo_ros/conversions/sensor_msgs.hpp>
#include <gazebo_ros/node.hpp>
#include <gazebo_ros/utils.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/time.hpp>

#include <ignition/math/Rand.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Matrix3.hh>

#include <predict/predict.h>
#include <gnss_multipath_plugin/multipath_sensor.hpp>
#include "gnss_multipath_plugin/msg/gnss_multipath_fix.hpp"
#include "gnss_multipath_plugin/geodetic_conv.hpp"


namespace gazebo_plugins
{

class GazeboRosMultipathSensorPrivate
{
public:
  /// Node for ROS communication.
  gazebo_ros::Node::SharedPtr ros_node_;

  // Aliases
  using LaserScan = sensor_msgs::msg::LaserScan;
  using LaserScanPub = rclcpp::Publisher<LaserScan>::SharedPtr;
 
  /// Publisher of output
  /// \todo use std::variant one c++17 is supported in ROS2
  boost::variant<LaserScanPub> pub_;

  /// TF frame output is published in
  std::string frame_name_;
  
  /// Subscribe to gazebo's laserscan, calling the appropriate callback based on output type
  void SubscribeGazeboLaserScan();

  /// Publish a sensor_msgs/LaserScan message from a gazebo laser scan
  void PublishLaserScan(ConstLaserScanStampedPtr & _msg);

  /// Gazebo transport topic to subscribe to for laser scan
  std::string sensor_topic_;

  /// Minimum intensity value to publish for laser scan / pointcloud messages
  double min_intensity_{0.0};

  /// Gazebo node used to subscribe to laser scan
  gazebo::transport::NodePtr gazebo_node_;

  /// Gazebo subscribe to parent sensor's laser scan
  gazebo::transport::SubscriberPtr laser_scan_sub_;
  gazebo::sensors::RaySensorPtr parent_ray_sensor_;
  gazebo::physics::WorldPtr world_;
  gazebo::physics::EntityPtr parent_entity_;

  // Time information to get the satellite position
  struct tm *timeinfo_;

  
  bool disable_noise_;
  geodetic_converter::GeodeticConverter g_geodetic_converter;
  
  int num_sat_;
  // TlE parameters to store the satellite's orbital parameters
  const char **tle_lines_;

  // Satellite orbital parameters to predict satellite's position
  predict_orbital_elements_t **sat_orbit_elements_;

  // Satellite position information at a given time
  struct predict_position *sat_position_;

  void SetRayAngles(std::vector<double> _azimuth, std::vector<double> _elevation); 
  bool GetLeastSquaresEstimate(std::vector<double> _meas, std::vector<std::vector<double>> _sat_ecef, Eigen::Vector3d & _rec_ecef);
  void CalculateDOP(std::vector<std::vector<double>> _sat_ecef, 
                    Eigen::Vector3d _rec_ecef, std::vector<double> &_dop);
  void ParseSatelliteTLE();
  void GetSatellitesInfo(struct tm *_timeinfo, std::vector<double> _rec_lla,
      std::vector<std::vector<double>> &_sat_ecef, std::vector<double> &_true_range, std::vector<double> &_azimuth,
      std::vector<double> &_elevation);
  time_t MakeTimeUTC(const struct tm* timeinfo_utc);
  double origin_lat_, origin_lon_, origin_alt_;
  
  //Offset publisher to publish multipath offset
  rclcpp::Publisher<gnss_multipath_plugin::msg::GNSSMultipathFix>::SharedPtr gnss_multipath_fix_publisher_;
};

GazeboRosMultipathSensor::GazeboRosMultipathSensor()
: impl_(std::make_unique<GazeboRosMultipathSensorPrivate>())
{
}

GazeboRosMultipathSensor::~GazeboRosMultipathSensor()
{
  // Must release subscriber and then call fini on node to remove it from topic manager.
  impl_->laser_scan_sub_.reset();
  if (impl_->gazebo_node_) {
    impl_->gazebo_node_->Fini();
  }
  impl_->gazebo_node_.reset();
}

void GazeboRosMultipathSensor::Load(gazebo::sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
{
  // Initialize time data structure
  impl_->timeinfo_ = (struct tm*)calloc(1, sizeof(struct tm));
  // Create ros_node configured from sdf
  impl_->ros_node_ = gazebo_ros::Node::Get(_sdf);
  impl_->parent_ray_sensor_ =
    std::dynamic_pointer_cast<gazebo::sensors::RaySensor>(_sensor);
  std::string worldName = _sensor->WorldName();
  impl_->world_ = gazebo::physics::get_world(worldName);
  std::string parent_entity_name = impl_->parent_ray_sensor_->ParentName();
  impl_->parent_entity_ = impl_->world_->EntityByName("laser_0");
  //RCLCPP_INFO(impl_->ros_node_->get_logger(), parent_entity_name);
  // Get QoS profiles
  const gazebo_ros::QoS & qos = impl_->ros_node_->get_qos();

  // Get QoS profile for the publisher
  rclcpp::QoS pub_qos = qos.get_publisher_qos("~/out", rclcpp::SensorDataQoS().reliable());

  // Get tf frame for output
  impl_->frame_name_ = gazebo_ros::SensorFrameID(*_sensor, *_sdf);

  // Get output type from sdf if provided
  if (!_sdf->HasElement("output_type")) {
    RCLCPP_WARN(
      impl_->ros_node_->get_logger(), "missing <output_type>, defaults to sensor_msgs/LaserScan");
      impl_->pub_ = impl_->ros_node_->create_publisher<sensor_msgs::msg::LaserScan>(
        "~/out", pub_qos);
  } else {
    std::string output_type_string = _sdf->Get<std::string>("output_type");
    if (output_type_string == "sensor_msgs/LaserScan") {
      impl_->pub_ = impl_->ros_node_->create_publisher<sensor_msgs::msg::LaserScan>(
        "~/out", pub_qos);
    } 
    else {
      RCLCPP_ERROR(
        impl_->ros_node_->get_logger(), "Invalid <output_type> [%s]", output_type_string.c_str());
      return;
    }
  }
  
  if (!_sdf->HasElement("min_intensity")) {
    RCLCPP_DEBUG(
      impl_->ros_node_->get_logger(), "missing <min_intensity>, defaults to %f",
      impl_->min_intensity_);
  } else {
    impl_->min_intensity_ = _sdf->Get<double>("min_intensity");
  }
  if (!_sdf->HasElement("disableNoise"))
  {
    RCLCPP_DEBUG(impl_->ros_node_->get_logger(), "Default to disable gaussian noise");
    impl_->disable_noise_ = true;
  }
  else
  {
    impl_->disable_noise_ = _sdf->Get<bool>("disableNoise");
    RCLCPP_INFO(impl_->ros_node_->get_logger(), "disable noise is %d\n", impl_->disable_noise_);

  }
  if(!_sdf->HasElement("origin_latitude"))
  {
    RCLCPP_DEBUG(impl_->ros_node_->get_logger(),  "Must provide origin latitutde");
  }
  else
  {
    std::string origin_lat = _sdf->Get<std::string>("origin_latitude");
    std::istringstream ss(origin_lat);
    double origin_lat_;
    ss >> origin_lat_;
    impl_->origin_lat_ = origin_lat_;
  }
  if(!_sdf->HasElement("origin_longitude"))
  {
    RCLCPP_DEBUG(impl_->ros_node_->get_logger(),  "Must provide origin longitude");
  }
  else
  {
    std::string origin_lon = _sdf->Get<std::string>("origin_longitude");
    std::istringstream ss(origin_lon);
    double origin_lon_;
    ss >> origin_lon_;
    impl_->origin_lon_ = origin_lon_;
  }

  if(!_sdf->HasElement("origin_altitude"))
  {
    RCLCPP_DEBUG(impl_->ros_node_->get_logger(),  "Must provide origin altitude");
  }
  else
  {
    std::string origin_alt = _sdf->Get<std::string>("origin_altitude");
    std::istringstream ss(origin_alt);
    double origin_alt_;
    ss >> origin_alt_;
    impl_->origin_alt_ = origin_alt_;
  }

  // if(!_sdf->HasElement("date_time"))
  // {
  //   RCLCPP_DEBUG(impl_->ros_node_->get_logger(),  "Must provide date and time");
  // }
  // else
  // {
  //   std::string date_time_ = _sdf->Get<std::string>("date_time");
  //   std::istringstream ss(date_time_);
  //   std::string dateTimeFormat{"%d/%m/%Y %H:%M:%S"};
  //   struct tm timeinfo_;
  //   ss >> std::get_time( &timeinfo_, dateTimeFormat.c_str());
  //   impl_->timeinfo_ = &timeinfo_;
  // }
  // Initializing the converter with the geodetic location of the origin.
  impl_->g_geodetic_converter.initialiseReference(impl_->origin_lat_, impl_->origin_lon_, impl_->origin_alt_);
  RCLCPP_INFO(impl_->ros_node_->get_logger(), "Origin Lat Lon Alt:%f %f %f",impl_->origin_lat_, impl_->origin_lon_, impl_->origin_alt_);
  RCLCPP_INFO(impl_->ros_node_->get_logger(), "Num sat:%d", impl_->num_sat_);
  //RCLCPP_INFO(impl_->ros_node_->get_logger(), std::filesystem::current_path());
  impl_->ParseSatelliteTLE();
  // Create gazebo transport node and subscribe to sensor's laser scan
  impl_->gazebo_node_ = boost::make_shared<gazebo::transport::Node>();
  impl_->gazebo_node_->Init(_sensor->WorldName());

  // TODO(ironmig): use lazy publisher to only process laser data when output has a subscriber
  impl_->sensor_topic_ = _sensor->Topic();
  impl_->gnss_multipath_fix_publisher_ = impl_->ros_node_->create_publisher<gnss_multipath_plugin::msg::GNSSMultipathFix>("gnss_multipath_fix", 10); 
  impl_->SubscribeGazeboLaserScan();
}

void GazeboRosMultipathSensorPrivate::SubscribeGazeboLaserScan()
{
  if (pub_.type() == typeid(LaserScanPub)) {
    laser_scan_sub_ = gazebo_node_->Subscribe(
      sensor_topic_, &GazeboRosMultipathSensorPrivate::PublishLaserScan, this);
  } 
  else {
    RCLCPP_ERROR(ros_node_->get_logger(), "Publisher is an invalid type. This is an internal bug.");
  }
}

void GazeboRosMultipathSensorPrivate::PublishLaserScan(ConstLaserScanStampedPtr & _msg)
{
  //rclcpp::Time t1 = rclcpp::Clock{}.now();
  std::vector<double> sat_true_range_;
  std::vector<std::vector<double>> sat_ecef_;
  std::vector<double> elevation_;
  std::vector<double> azimuth_; 
  std::vector<double> ray_elevation_;
  std::vector<double> ray_azimuth_;
  std::vector<double> rec_lla_{0,0,0};
  // Reciever's pose in ENU world coordinates
  ignition::math::Vector3d true_rec_world_pos_ = parent_entity_->WorldPose().Pos();
  // Reciever's position in geodetic (LLA) coordinates
  g_geodetic_converter.enu2Geodetic(true_rec_world_pos_[0], true_rec_world_pos_[1],
                              true_rec_world_pos_[2], &rec_lla_[0], &rec_lla_[1], &rec_lla_[2]);
  
  //RCLCPP_INFO(ros_node_->get_logger(), "time :%d:%d:%d", timeinfo_->tm_year, timeinfo_->tm_mon,timeinfo_->tm_mday );
  // Constructing a time structure for getting satellites position
  timeinfo_->tm_year = 2021 - 1900;
  timeinfo_->tm_mon = 5 - 1;
  timeinfo_->tm_mday = 15;
  timeinfo_->tm_hour = 1;
  timeinfo_->tm_min = 55;
  timeinfo_->tm_sec = 0;
  timeinfo_->tm_isdst = 0;
  // Collect positive elevation satellites ECEF, true ranges and angles 
  GetSatellitesInfo(timeinfo_,rec_lla_, sat_ecef_, sat_true_range_, azimuth_, elevation_ );
  int vis_num_sat_ = sat_true_range_.size();
  //RCLCPP_INFO(ros_node_->get_logger(), "num sat:%d", vis_num_sat_);
  for ( int i = 0; i < vis_num_sat_; i++ )
  {
    // Setting the azimuth and elevation angles for the satellite ray and the reflected ray for each satellite.
    ray_azimuth_.push_back(azimuth_[i]);
    ray_azimuth_.push_back(azimuth_[i]+ M_PI); // assume reflection is 180 deg azimuth from the direct ray
    ray_elevation_.push_back(elevation_[i]);
    ray_elevation_.push_back(elevation_[i]); // assume reflection has the same elevation as the direct ray
    //RCLCPP_INFO(ros_node_->get_logger(), "sat angle:%f %f", azimuth_[i]*180/M_PI, elevation_[i]*180/M_PI);
  }
  //Updating the ray angles for sateliite and reflected ray.
  SetRayAngles(ray_azimuth_, ray_elevation_);

  //Ray_ranges stores the ranges of both satellite and reflected rays.
  std::vector<double> ray_ranges_; 
  ray_ranges_.resize(_msg->scan().count());
  std::copy(_msg->scan().ranges().begin(),
            _msg->scan().ranges().end(),
            ray_ranges_.begin());
                        
  
  //Publishing the range offset and satellites blocked
  gnss_multipath_plugin::msg::GNSSMultipathFix gnss_multipath_fix_msg;
  gnss_multipath_fix_msg.header.stamp = rclcpp::Time(_msg->time().sec(), _msg->time().nsec());
  gnss_multipath_fix_msg.header.frame_id = frame_name_;
  gnss_multipath_fix_msg.navsatfix.header.stamp = rclcpp::Time(_msg->time().sec(), _msg->time().nsec());
  gnss_multipath_fix_msg.navsatfix.header.frame_id = frame_name_;
  gnss_multipath_fix_msg.enu_true.resize(3);
  gnss_multipath_fix_msg.enu_gnss_fix.resize(3);
  gnss_multipath_fix_msg.dop.resize(5);
  int num_sat_blocked_ = 0;
  std::vector<double> visible_sat_range_meas;
  std::vector<std::vector<double>> visible_sat_ecef;
  
  // Iterate for all the visible satellites (positive elevation angles)
  for (int i = 0; i < vis_num_sat_; i++)
  {
    // noise
    double noise_ = 0.0;
    if (!disable_noise_)
    {  
        noise_ = ignition::math::Rand::DblNormal(0.0, 1.0);
    }
    if (ray_elevation_[2*i] > (15*M_PI)/180)
    {
      // ray is obstructed
      if (ray_ranges_[2*i] < _msg->scan().range_max()) 
      {
        // check range of the mirror ray corresponding to the satellite ray 
        double mir_ray_range_ = ray_ranges_[2*i+1];

        // check if mirror ray also is obstructed, providing a reflection
        if (mir_ray_range_ < _msg->scan().range_max())
        {
          double range_offset =  mir_ray_range_ * ( 1 + sin(M_PI/2.0 - 2*ray_elevation_[2*i]));
          // RCLCPP_INFO(ros_node_->get_logger(), "range:%f %f %f ", range_offset,mir_ray_range_, ray_elevation_[2*i] );
          // Ignore the offset if the multipath range is too high
          if (range_offset < 50)
          {
            //RCLCPP_INFO(ros_node_->get_logger(), "offset:%d %f", i, range_offset);
            visible_sat_range_meas.push_back(sat_true_range_[i] + range_offset +  noise_);
            visible_sat_ecef.push_back(sat_ecef_[i]);
          }
          else
          {
            // The multipath distance is large, signal intensity will be degraded significantly, so ignored.
            num_sat_blocked_++;
          }
        } 
        else 
        {
          // otherwise no line of sight to any satellites and no reflections, no reading
          num_sat_blocked_++;
        }
      } 
      else 
      {
        // unobstructed sat reading
        visible_sat_range_meas.push_back(sat_true_range_[i] + noise_);
        visible_sat_ecef.push_back(sat_ecef_[i]);
      }
    }
    else
    {
      // low elevation sat reading
      num_sat_blocked_++;
    }
  }
  if ((vis_num_sat_ - num_sat_blocked_) > 4)
  {
    // Calculate the reciever's position using the satellite's predicted coordinates and the pseudo-ranges.
    Eigen::Vector3d rec_ecef(0,0,0);              
    GetLeastSquaresEstimate(visible_sat_range_meas,  visible_sat_ecef, rec_ecef);
    std::vector<double> dop;
    CalculateDOP(visible_sat_ecef, rec_ecef, dop);
    double enu_x,enu_y,enu_z, latitude=0, longitude=0, altitude=0;
    g_geodetic_converter.ecef2Geodetic(rec_ecef(0), rec_ecef(1),rec_ecef(2), &latitude,
                      &longitude, &altitude);
    
    g_geodetic_converter.geodetic2Enu(latitude,
                      longitude, altitude, &enu_x,&enu_y,&enu_z);
    gnss_multipath_fix_msg.navsatfix.status.status = 1;
    
    gnss_multipath_fix_msg.navsatfix.latitude = latitude;
    gnss_multipath_fix_msg.navsatfix.longitude = longitude;
    gnss_multipath_fix_msg.navsatfix.altitude = altitude;

    gnss_multipath_fix_msg.enu_gnss_fix[0] = enu_x;
    gnss_multipath_fix_msg.enu_gnss_fix[1] = enu_y;
    gnss_multipath_fix_msg.enu_gnss_fix[2] = enu_z;
    
    //Copy DOP values to the message.
    std::copy(dop.begin(),
            dop.end(),
            gnss_multipath_fix_msg.dop.begin());
  }
  else 
  {
    // RCLCPP_INFO(ros_node_->get_logger(), "No Fix");
    gnss_multipath_fix_msg.navsatfix.status.status = 0;
    gnss_multipath_fix_msg.enu_gnss_fix[0] = NAN;
    gnss_multipath_fix_msg.enu_gnss_fix[1] = NAN;
    gnss_multipath_fix_msg.enu_gnss_fix[2] = NAN;
  }
  gnss_multipath_fix_msg.enu_true[0] = true_rec_world_pos_[0];
  gnss_multipath_fix_msg.enu_true[1] = true_rec_world_pos_[1];
  gnss_multipath_fix_msg.enu_true[2] = true_rec_world_pos_[2];
  gnss_multipath_fix_msg.visible_sat_count = vis_num_sat_ - num_sat_blocked_;
  gnss_multipath_fix_publisher_->publish(gnss_multipath_fix_msg);

  // Convert Laser scan to ROS LaserScan
  auto ls = gazebo_ros::Convert<sensor_msgs::msg::LaserScan>(*_msg);
  // Set tf frame
  ls.header.frame_id = frame_name_;
  // Publish output
  boost::get<LaserScanPub>(pub_)->publish(ls);
}

void GazeboRosMultipathSensorPrivate::SetRayAngles(std::vector<double> _azimuth, std::vector<double> _elevation) 
{ 
  gazebo::physics::MultiRayShapePtr laser_shape_ = parent_ray_sensor_->LaserShape();
  ignition::math::Vector3d start, end, axis, end_scan;
  ignition::math::Quaterniond ray;
  for(unsigned int i=0; i < _elevation.size(); i++ ) 
  {
    //RCLCPP_INFO(ros_node_->get_logger(), "ray angle:%f %f", _azimuth[i]*180/M_PI, _elevation[i]*180/M_PI);
    double yaw_angle = _azimuth[i];
    double pitch_angle = _elevation[i];
    // since we're rotating a unit x vector, a pitch rotation will now be
    // around the negative y axis
    ray.Euler(ignition::math::Vector3d(0.0, -pitch_angle, yaw_angle));
    axis = parent_entity_->WorldPose().Rot().Inverse() * parent_ray_sensor_->Pose().Rot() * ray * ignition::math::Vector3d::UnitX;
    start = (axis * laser_shape_->GetMinRange()) + parent_ray_sensor_->Pose().Pos();
    end = (axis * laser_shape_->GetMaxRange()) + parent_ray_sensor_->Pose().Pos();
    laser_shape_->SetRay(i, start, end);
  }
}

bool GazeboRosMultipathSensorPrivate::GetLeastSquaresEstimate(std::vector<double> _meas,
                 std::vector<std::vector<double>> _sat_ecef, Eigen::Vector3d &_rec_ecef)
{
  const int nsat = _meas.size();
  Eigen::MatrixXd A, b;
  A.resize(nsat, 4);
  b.resize(nsat, 1);
  Eigen::Matrix<double, 4, 1> dx;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver;
  double cdt = 0;
  int iter = 0;
  do
  {
    iter++;
    int i = 0;
    for (int i = 0 ; i < nsat; i++)
    {
      Eigen::Vector3d sat_pos(_sat_ecef[i][0], _sat_ecef[i][1], _sat_ecef[i][2]);
      double dist = (_rec_ecef - sat_pos).norm();
      A.block<1,3>(i,0) = (_rec_ecef - sat_pos).normalized();
      b(i) = _meas[i] - (dist + cdt);
      A(i,3) = -1;
      
    }
    solver.compute(A);
    dx = solver.solve(b);
    _rec_ecef += dx.topRows<3>();
  } while (dx.norm() > 1e-6 && iter < 10);
  return iter < 10;
}

void GazeboRosMultipathSensorPrivate::CalculateDOP(std::vector<std::vector<double>> _sat_ecef, 
                    Eigen::Vector3d _rec_ecef, std::vector<double> &_dop)
{
  const int nsat = _sat_ecef.size();
 
  Eigen::MatrixXd A, Q, Q_local;
  A.resize(nsat, 4);
  for (int i = 0 ; i < nsat; i++)
  {
    Eigen::Vector3d sat_pos(_sat_ecef[i][0], _sat_ecef[i][1], _sat_ecef[i][2]);
    A.block<1,3>(i,0) = (_rec_ecef - sat_pos).normalized();
    A(i,3) = -1;
  }
  Q = (A.transpose()*A).inverse();
  //GDOP
  _dop.push_back(std::sqrt(Q.trace()));
  //PDOP
  _dop.push_back(std::sqrt((Q.block<3,3>(0,0)).trace()));
	//TDOP
  _dop.push_back(std::sqrt(Q(3,3)));
  double latitude = 0.0, longitude=0.0, altitude=0.0;
  g_geodetic_converter.ecef2Geodetic(_rec_ecef(0), _rec_ecef(1),_rec_ecef(2), &latitude,
                      &longitude, &altitude);
	double phi = latitude; 			
	double lambda = longitude;	

  double sl = sin(lambda);
  double cl = cos(lambda);
  double sp = sin(phi);
  double cp = cos(phi);
  
  Eigen::MatrixXd Rot_matrix(4, 4);
  Rot_matrix(0, 0) = -cl*sp;
  Rot_matrix(0, 1) = -sl*sp;
  Rot_matrix(0, 2) = cp;
  Rot_matrix(0, 3) = 0;

  Rot_matrix(1, 0) = -sl;
  Rot_matrix(1, 1) = cl;
  Rot_matrix(1, 2) = 0;
  Rot_matrix(1, 3) = 0;

  Rot_matrix(2, 0) = cl*cp;
  Rot_matrix(2, 1) = cp*sl;
  Rot_matrix(2, 2) = sp;
  Rot_matrix(2, 3) = 0;

  Rot_matrix(3, 0) = 0;
  Rot_matrix(3, 1) = 0;
  Rot_matrix(3, 2) = 0;
  Rot_matrix(3, 3) = 1;

	// calculate the local cofactor matrix
	Q_local = Rot_matrix*Q*Rot_matrix.transpose();
  // calculate 'HDOP' and 'VDOP' 
	// HDOP
  _dop.push_back(std::sqrt(Q_local(0,0)*Q_local(1,1)));
	// VDOP
  _dop.push_back(std::sqrt(Q_local(2,2)));
  //RCLCPP_INFO(ros_node_->get_logger(), "gdop:%f pdop:%f tdop:%f hdop:%f vdop:%f", _dop[0],_dop[1],_dop[2],_dop[3], _dop[4]);
}

time_t GazeboRosMultipathSensorPrivate::MakeTimeUTC(const struct tm* timeinfo_utc)
{
	time_t curr_time = time(NULL);
	int timezone_diff = 0; //deviation of the current timezone from UTC in seconds

	//get UTC time, interpret resulting tm as a localtime
	struct tm timeinfo_gmt;
	gmtime_r(&curr_time, &timeinfo_gmt);
	time_t time_gmt = mktime(&timeinfo_gmt);

	//get localtime, interpret resulting tm as localtime
	struct tm timeinfo_local;
	localtime_r(&curr_time, &timeinfo_local);
	time_t time_local = mktime(&timeinfo_local);

	//find the time difference between the two interpretations
	timezone_diff += difftime(time_local, time_gmt);

	//hack for preventing mktime from assuming localtime: add timezone difference to the input struct.
	struct tm ret_timeinfo;
	ret_timeinfo.tm_sec = timeinfo_utc->tm_sec + timezone_diff;
	ret_timeinfo.tm_min = timeinfo_utc->tm_min;
	ret_timeinfo.tm_hour = timeinfo_utc->tm_hour;
	ret_timeinfo.tm_mday = timeinfo_utc->tm_mday;
	ret_timeinfo.tm_mon = timeinfo_utc->tm_mon;
	ret_timeinfo.tm_year = timeinfo_utc->tm_year;
	ret_timeinfo.tm_isdst = timeinfo_utc->tm_isdst;
	return mktime(&ret_timeinfo);
}

void GazeboRosMultipathSensorPrivate::ParseSatelliteTLE()
{
  num_sat_ = 31;
  RCLCPP_INFO(ros_node_->get_logger(), "Num sat:%d", num_sat_);
  // Initialization of data structures for satellite prediction 
  tle_lines_ = (const char**) calloc(2*num_sat_, sizeof(const char*));
  sat_orbit_elements_ = (predict_orbital_elements_t**) calloc(num_sat_, sizeof(predict_orbital_elements_t*));
  sat_position_ = (struct predict_position*) calloc(num_sat_, sizeof(struct predict_position));
  for ( int i = 0; i < 2*num_sat_; i++ )
  {
      tle_lines_[i] = (const char*) calloc(100, sizeof(const char));
  }
  for ( int i = 0; i < num_sat_; i++ )
  {
      sat_orbit_elements_[i] = (predict_orbital_elements_t*) calloc(1, sizeof( predict_orbital_elements_t));
  }
  
  tle_lines_[0] = "1 24876U 97035A   23033.20764968 -.00000038  00000+0  00000+0 0  9994";
  tle_lines_[1] = "2 24876  55.5459 146.7067 0065794  52.6664 307.9046  2.00564086187267";
 
  tle_lines_[2] = "1 26360U 00025A   23033.06431156 -.00000048  00000+0  00000+0 0  9992";
  tle_lines_[3] = "2 26360  54.2496  69.3979 0045611 194.8807 151.7827  2.00555919166578";

  tle_lines_[4] = "1 27663U 03005A   23033.62997381  .00000047  00000+0  00000+0 0  9997";
  tle_lines_[5] = "2 27663  55.3642 263.9131 0131905  42.8180 326.5456  2.00551168146626";

  tle_lines_[6] = "1 27704U 03010A   23033.55723828 -.00000091  00000+0  00000+0 0  9994";
  tle_lines_[7] = "2 27704  55.0899  13.9998 0249685 312.4590 223.4527  2.00565951145429";

  tle_lines_[8] = "1 28190U 04009A   23034.15509573 -.00000085  00000+0  00000+0 0  9992";
  tle_lines_[9] = "2 28190  55.8516 324.9248 0088848 128.9769  57.1221  2.00572247138304";

  tle_lines_[10] = "1 28474U 04045A   23033.60153542 -.00000090  00000+0  00000+0 0  9993";
  tle_lines_[11] = "2 28474  55.4150  14.2264 0202589 283.4591  74.0418  2.00562393133746";

  tle_lines_[12] = "1 28874U 05038A   23033.61665581 -.00000084  00000+0  00000+0 0  9995";
  tle_lines_[13] = "2 28874  55.9094 322.3891 0139570 278.1790 267.2996  2.00560021127153";

  tle_lines_[14] = "1 29486U 06042A   23033.78720829  .00000046  00000+0  00000+0 0  9993";
  tle_lines_[15] = "2 29486  54.6970 200.1224 0106835  27.4847 197.4334  2.00565983119862";

  tle_lines_[16] = "1 29601U 06052A   23032.92873802  .00000048  00000+0  00000+0 0  9990";
  tle_lines_[17] = "2 29601  55.3786 262.8889 0086684  75.9810 285.0549  2.00564342118716";
    
  tle_lines_[18] = "1 32260U 07047A   23033.23014611 -.00000051  00000+0  00000+0 0  9995";
  tle_lines_[19] = "2 32260  53.3839 131.0174 0145380  67.1246 294.4003  2.00563380112126";

  tle_lines_[20] = "1 32384U 07062A   23033.85587446 -.00000084  00000+0  00000+0 0  9994";
  tle_lines_[21] = "2 32384  56.0351 323.1715 0020102 142.8969  77.6124  2.00574229110895";

  tle_lines_[22] = "1 32711U 08012A   23032.64592938  .00000056  00000+0  00000+0 0  9990";
  tle_lines_[23] = "2 32711  54.4610 199.0507 0168701 232.8355 125.5954  2.00572261109059";

  tle_lines_[24] = "1 35752U 09043A   23034.15624287 -.00000035  00000+0  00000+0 0  9992";
  tle_lines_[25] = "2 35752  55.2549  76.2529 0055618  63.8864 319.4178  2.00571659 98673";

  tle_lines_[26] = "1 36585U 10022A   23033.21070384  .00000056  00000+0  00000+0 0  9997";
  tle_lines_[27] = "2 36585  54.6695 258.2376 0109114  58.2673 124.2846  2.00550759 92904";
  
  tle_lines_[28] = "1 37753U 11036A   23033.10118579 -.00000081  00000+0  00000+0 0  9997";
  tle_lines_[29] = "2 37753  56.6933  19.7748 0121088  52.9316 129.8426  2.00566862 84583";

  tle_lines_[30] = "1 38833U 12053A   23033.33039722  .00000038  00000+0  00000+0 0  9995";
  tle_lines_[31] = "2 38833  53.5190 193.9407 0135373  49.9651 311.1852  2.00564336 74752";

  tle_lines_[32] = "1 39166U 13023A   23033.18918996 -.00000082  00000+0  00000+0 0  9995";
  tle_lines_[33] = "2 39166  55.5293 318.6885 0111059  38.9232 321.8895  2.00565200 71193";

  tle_lines_[34] = "1 39533U 14008A   23033.19159522  .00000049  00000+0  00000+0 0  9992";
  tle_lines_[35] = "2 39533  53.6091 199.4781 0062019 209.5871 150.0417  2.00566418 65548";

  tle_lines_[36] = "1 39741U 14026A   23033.96686822 -.00000082  00000+0  00000+0 0  9994";
  tle_lines_[37] = "2 39741  56.6508  19.2656 0030744 308.2184  51.5125  2.00572503 63854";

  tle_lines_[38] = "1 39741U 14026A   23033.96686822 -.00000082  00000+0  00000+0 0  9994";
  tle_lines_[39] = "2 39741  56.6508  19.2656 0030744 308.2184  51.5125  2.00572503 63854";

  tle_lines_[40] = "1 40105U 14045A   23032.56334991 -.00000048  00000+0  00000+0 0  9995";
  tle_lines_[41] = "2 40105  54.7481 137.6538 0023243 114.6825 245.5467  2.00562646 61376";

  tle_lines_[42] = "1 40294U 14068A   23032.43441650 -.00000049  00000+0  00000+0 0  9997";
  tle_lines_[43] = "2 40294  55.9964  78.9035 0042465  56.8096 303.6631  2.00557360 60507";

  tle_lines_[44] = "1 40534U 15013A   23033.06804744  .00000058  00000+0  00000+0 0  9991";
  tle_lines_[45] = "2 40534  53.5692 255.1758 0077227  23.8247 336.6011  2.00567871 57134";

  tle_lines_[46] = "1 40730U 15033A   23033.72582779 -.00000076  00000+0  00000+0 0  9996";
  tle_lines_[47] = "2 40730  55.0217 317.3848 0082059  10.2474 349.9421  2.00566329 55333";

  tle_lines_[48] = "1 41019U 15062A   23033.28107185 -.00000041  00000+0  00000+0 0  9996";
  tle_lines_[49] = "2 41019  55.9834  78.7168 0085778 220.7636 138.6587  2.00562556 53141";

  tle_lines_[50] = "1 41328U 16007A   23033.36788295 -.00000045  00000+0  00000+0 0  9999";
  tle_lines_[51] = "2 41328  54.9593 138.3485 0067345 231.5397 127.8428  2.00559919 51156";

  tle_lines_[52] = "1 43873U 18109A   23032.57137684 -.00000044  00000+0  00000+0 0  9992";
  tle_lines_[53] = "2 43873  55.1375 140.8489 0023229 190.4825 204.0790  2.00558671 30373";

  tle_lines_[54] = "1 44506U 19056A   23033.40051584 -.00000083  00000+0  00000+0 0  9998";
  tle_lines_[55] = "2 44506  55.7692  19.9151 0029851 186.7930 347.2623  2.00561526 25382";

  tle_lines_[54] = "1 45854U 20041A   23034.26264002 -.00000033  00000+0  00000+0 0  9992";
  tle_lines_[55] = "2 45854  55.7045  77.1516 0029948 185.9714 192.1492  2.00567044 19363";

  tle_lines_[56] = "1 46826U 20078A   23032.06127720  .00000047  00000+0  00000+0 0  9997";
  tle_lines_[57] = "2 46826  54.4254 260.8119 0026417 188.1073  16.6882  2.00564630 16797";

  tle_lines_[58] = "1 48859U 21054A   23033.87596315 -.00000082  00000+0  00000+0 0  9990";
  tle_lines_[59] = "2 48859  55.2812  21.7836 0009051 219.9340  39.0694  2.00560997 12067";

  tle_lines_[60] = "1 55268U 23009A   23033.46134089  .00000045  00000+0  00000+0 0  9993";
  tle_lines_[61] = "2 55268  55.1007 196.8026 0008185  97.8399 262.2308  2.00570647   571";

  
  for ( int i = 0; i < num_sat_; i++ )
  {
      // Create orbit object
      sat_orbit_elements_[i] = predict_parse_tle(tle_lines_[2*i], tle_lines_[2*i+1]);
  }
}

void GazeboRosMultipathSensorPrivate::GetSatellitesInfo(struct tm *_timeinfo, std::vector<double> _rec_lla,
      std::vector<std::vector<double>> &_sat_ecef, std::vector<double> &_true_range, std::vector<double> &_azimuth,
      std::vector<double> &_elevation)
{
  // Predict the julian time using the current utc time
  predict_julian_date_t curr_time = predict_to_julian(MakeTimeUTC(_timeinfo));
  std::vector<double> ecef = {0,0,0};
  for ( int i = 0; i < num_sat_; i++ )
  {
    // Predict satellite's position based on the current time.
    predict_orbit(sat_orbit_elements_[i], &sat_position_[i], curr_time);
    g_geodetic_converter.geodetic2Ecef(sat_position_[i].latitude*180.0/M_PI, sat_position_[i].longitude*180.0/M_PI , 
                  sat_position_[i].altitude*1000 , &ecef[0], &ecef[1], &ecef[2]);
    predict_observer_t *obs = predict_create_observer("obs", _rec_lla[0]*M_PI/180.0, _rec_lla[1]*M_PI/180.0, _rec_lla[2]);
    struct predict_observation reciever_obs;
    predict_observe_orbit(obs, &sat_position_[i], &reciever_obs);
    if (reciever_obs.elevation > 0)
    {
      _sat_ecef.push_back(ecef);
      _true_range.push_back(reciever_obs.range*1000); //From Km to m  
      _azimuth.push_back(reciever_obs.azimuth);
      _elevation.push_back(reciever_obs.elevation);
    }
    
  }
}

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(GazeboRosMultipathSensor)
}  // namespace gazebo_plugins