#include "crs_msgs/car_state_cart.h"
#include "car_track_visualizer/CarTrackVisualizer.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <cmath>
#include <ros_crs_utils/obstacle_message_conversion.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <ros_crs_utils/parameter_io.h>
#include <ros_crs_utils/validation.h>

CarTrackVisualizer::CarTrackVisualizer(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
  : nh_(nh), nh_private_(nh_private)
{
  loadParameters();
  subscribeToState();
  setupPublisher();
  setupTrack();
  setupMarker();
  subscribeToObstacles();
  publishTrack();  // since the topic is latched, this only happens once
}

void CarTrackVisualizer::loadParameters()
{
  ROS_INFO("Visualizer: loading visualizer parameters");
  // load node parameters
  if (!nh_private_.getParam("node_rate", node_rate_))
    ROS_WARN_STREAM("Visualizer: did not load visualizer node_rate.");

  // load node parameters
  if (!nh_private_.getParam("show_track_angle", show_track_angle_))
    ROS_WARN_STREAM("Visualizer: did not load visualizer show_track_angle.");
  // load node parameters
  if (!nh_private_.getParam("point_downsampling_factor", point_downsampling_factor_))
    ROS_WARN_STREAM("Visualizer: did not load visualizer point_downsampling_factor.");

  if (!nh_private_.getParam("frame_name", frame_name_))
    ROS_WARN_STREAM("Did not load frame_name_, defaulting to " << frame_name_);

  // load node parameters
  if (!nh_private_.getParam("lloyd_flag", lloyd_flag_))
    ROS_WARN_STREAM("Visualizer: did not load visualizer lloyd_flag.");

  nh_private_.getParam("trajectory/show_past_gt_trajectory", show_past_gt_trajectory_);
  nh_private_.getParam("trajectory/show_past_est_trajectory", show_past_est_trajectory_);
  nh_private_.getParam("trajectory/number_of_past_samples", number_of_past_samples_);

  nh_private_.getParam("estimate/color/r", COLOR_CAR_EST_[0]);
  nh_private_.getParam("estimate/color/g", COLOR_CAR_EST_[1]);
  nh_private_.getParam("estimate/color/b", COLOR_CAR_EST_[2]);
  nh_private_.getParam("estimate/color/a", COLOR_CAR_EST_[3]);

  nh_private_.getParam("gt/color/r", COLOR_CAR_GT_[0]);
  nh_private_.getParam("gt/color/g", COLOR_CAR_GT_[1]);
  nh_private_.getParam("gt/color/b", COLOR_CAR_GT_[2]);
  nh_private_.getParam("gt/color/a", COLOR_CAR_GT_[3]);
  nh_private_.getParam("publish_track", publish_track_);

  publish_track_ = false;

  nh_private_.getParam("car_namespace", car_namespace_);

  static_track_trajectory_ = parameter_io::loadTrackDescriptionFromParams(ros::NodeHandle("track"));
}

void CarTrackVisualizer::subscribeToState()
{
  sub_gt_ = nh_private_.subscribe<crs_msgs::car_state_cart>(
      "car_state_gt", 10, boost::bind(&CarTrackVisualizer::stateCallback, this, _1, false));
  sub_est_ = nh_private_.subscribe<crs_msgs::car_state_cart>(
      "car_state_est", 10, boost::bind(&CarTrackVisualizer::stateCallback, this, _1, true));
}

void CarTrackVisualizer::subscribeToObstacles()
{
  sub_obstacles_ = nh_private_.subscribe("obstacles", 10, &CarTrackVisualizer::setupObstaclesCallback, this);
}

// float min_velocity_ = 1.5;
// float max_velocity_ = 2.5;
void CarTrackVisualizer::stateCallback(const boost::shared_ptr<crs_msgs::car_state_cart const> msg, bool is_estimation)
{
  auto& stamp = is_estimation ? last_est_callback_ : last_gt_callback_;
  if (msg->header.stamp.toSec() - stamp < 1 / getNodeRate())
    return;

  stamp = msg->header.stamp.toSec();

  if (is_estimation)
  {
    got_est_car_state_ = true;
    car_state_estimated_ = *msg;
  }
  else
  {
    got_car_state_ = true;
    car_state_ = *msg;
  }

  auto& trajectory_markers = is_estimation ? est_trajectory_ : gt_trajectory_;
  if ((show_past_est_trajectory_ && is_estimation) || (show_past_gt_trajectory_ && !is_estimation))
  {
    if (number_of_past_samples_ != -1 && trajectory_markers.points.size() > number_of_past_samples_)
    {
      trajectory_markers.points.erase(trajectory_markers.points.begin(),
                                      trajectory_markers.points.begin() +
                                          int(0.1 * number_of_past_samples_));  // remove first 10%
      trajectory_markers.colors.erase(trajectory_markers.colors.begin(),
                                      trajectory_markers.colors.begin() +
                                          int(0.1 * number_of_past_samples_));  // remove first 10%
    }
    geometry_msgs::Point p;
    p.x = msg->x;
    p.y = msg->y;
    p.z = 0;
    trajectory_markers.points.push_back(p);

    std_msgs::ColorRGBA color;
    double normalized_velocity =
        std::max(0.0, std::min(1.0, (msg->v_tot - min_velocity_) / (max_velocity_ - min_velocity_)));
    color.r = normalized_velocity < 0.5 ? normalized_velocity : 1 - normalized_velocity;
    color.g = normalized_velocity < 0.5 ? 0 : normalized_velocity;
    color.b = normalized_velocity < 0.5 ? 1 - normalized_velocity : 0;
    trajectory_markers.colors.push_back(color);
  }
}

void CarTrackVisualizer::setupPublisher()
{
  pub_ = nh_.advertise<visualization_msgs::Marker>("car_visualization", 100);
  // The track publisher latches the marker array, meaning it is only published once.
  static_track_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("track_topic", 1, true);
}

void CarTrackVisualizer::setupTrack()
{
  ROS_DEBUG("Visualizer: Setting up track message template...");

  // Three options: Either publish the track, without angles (default),
  // publish the track with angles (show_track_angle_ is true) or publish
  // the track boundary (if lloyd_flag_ is set).

  // Set up the markers for the track boundary
  visualization_msgs::Marker boundary;

  boundary.header.frame_id = frame_name_;
  boundary.header.stamp = ros::Time::now();
  boundary.ns = "track_boundary";
  boundary.action = visualization_msgs::Marker::ADD;
  boundary.pose.orientation.w = 1.0;
  boundary.type = visualization_msgs::Marker::LINE_STRIP;
  boundary.id = 2;
  boundary.scale.x = TRACK_SCALE_;
  boundary.scale.y = TRACK_SCALE_;
  boundary.color.r = BLACK_[0];
  boundary.color.g = BLACK_[1];
  boundary.color.b = BLACK_[2];
  boundary.color.a = BLACK_[3];

  const auto w = static_track_trajectory_->getWidth();
  const auto& center_line = static_track_trajectory_->getCenterLine();

  geometry_msgs::Point point;

  // Two separate for loops, since subsequent markers are connected.
  for (int i = 0; i < center_line.size(); i += point_downsampling_factor_)
  {
    point.x = center_line[i].x() + w / 2.0 * static_track_trajectory_->getRate(i).y();
    point.y = center_line[i].y() - w / 2.0 * static_track_trajectory_->getRate(i).x();

    boundary.points.push_back(point);
  }

  for (int i = 0; i < center_line.size(); i += point_downsampling_factor_)
  {
    point.x = center_line[i].x() - w / 2.0 * static_track_trajectory_->getRate(i).y();
    point.y = center_line[i].y() + w / 2.0 * static_track_trajectory_->getRate(i).x();
    boundary.points.push_back(point);
  }

  track_msg_.markers.push_back(boundary);

  if (!lloyd_flag_)
  {
    // Entire track is published, either with or without track angles
    if (!show_track_angle_)
    {
      // In this case, we only create one Marker message that contains all the points
      visualization_msgs::Marker marker;
      marker.header.frame_id = frame_name_;
      marker.header.stamp = ros::Time::now();
      marker.ns = "track_center";
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.type = visualization_msgs::Marker::LINE_STRIP;
      marker.id = 1;
      marker.scale.x = TRACK_SCALE_;
      marker.scale.y = TRACK_SCALE_;
      marker.scale.z = TRACK_SCALE_;
      marker.color.r = ORANGE_[0];
      marker.color.g = ORANGE_[1];
      marker.color.b = ORANGE_[2];
      marker.color.a = ORANGE_[3];

      geometry_msgs::Point temp_p;
      int idx = 0;
      for (const auto& pt : static_track_trajectory_->getCenterLine())
      {
        if (idx++ % point_downsampling_factor_ != 0)
          continue;
        temp_p.x = pt.x();
        temp_p.y = pt.y();
        temp_p.z = -0.01;
        marker.points.push_back(temp_p);
      }

      track_msg_.markers.push_back(marker);
    }
    else
    {
      // In this case, each point needs an orientation. We fill multiple markers into the array.
      visualization_msgs::Marker marker;

      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD;
      marker.scale.y = TRACK_SCALE_;
      marker.scale.x = TRACK_SCALE_ * 3;

      int idx = 0;
      marker.id = 0;
      for (const auto& pt : static_track_trajectory_->getCenterLine())
      {
        if (idx++ % (point_downsampling_factor_ * 3) != 0)
          continue;
        marker.id = marker.id + 1;
        double angle = static_track_trajectory_->getTrackAngle(idx);
        marker.pose.position.x = pt.x();
        marker.pose.position.y = pt.y();
        marker.pose.orientation.w = std::cos(angle * 0.5);
        marker.pose.orientation.x = 0;
        marker.pose.orientation.y = 0;
        marker.pose.orientation.z = std::sin(angle * 0.5);

        track_msg_.markers.push_back(marker);
      }
    }
  }
  ROS_DEBUG("Visualizer: Track message template set up.");
}

static void visualizeRectangle(crs_controls::Rectangle& rectangle, visualization_msgs::Marker& marker)
{
  marker.type = visualization_msgs::Marker::CUBE;
  marker.scale.x = rectangle.getWidth();
  marker.scale.y = rectangle.getHeight();
  marker.scale.z = .1;

  geometry::Vertex lower_left = rectangle.getLowerLeft();
  marker.pose.position.x = lower_left[0] + rectangle.getWidth() / 2;
  marker.pose.position.y = lower_left[1] + rectangle.getHeight() / 2;
  marker.pose.position.z = 0;
}

static void visualizeRhombus(crs_controls::Rhombus& rhombus, visualization_msgs::Marker& marker)
{
  // Visualize as a TriangleList. Don't know a better way.
  marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
  marker.scale.x = 1;
  marker.scale.y = 1;
  marker.scale.z = 1;

  std::vector<geometry::Vertex> vertices;
  rhombus.getVertices(vertices);

  std::vector<geometry_msgs::Point> ros_vertices;
  for (const geometry::Vertex& vertex : vertices)
  {
    geometry_msgs::Point pt;
    pt.x = vertex[0];
    pt.y = vertex[1];
    pt.z = 0;
    ros_vertices.push_back(pt);
  }

  marker.points = {
    ros_vertices[0], ros_vertices[1], ros_vertices[2], ros_vertices[2], ros_vertices[3], ros_vertices[0]
  };
}

void CarTrackVisualizer::setupObstaclesCallback(const geometry_msgs::PoseArray& msg)
{
  ROS_INFO("Visualizer: setting up Obstacles...");
  sub_obstacles_.shutdown();  // we no longer need to receive obstacle positions
  std::unique_ptr<const crs_controls::ObstacleVector> obstacles = std::move(message_conversion::convertObstacles(msg));

  int id = 0;
  for (const auto& obstacle : *obstacles)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_name_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacle";
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = id++;

    crs_controls::Rectangle* rectangle = NULL;
    if ((rectangle = dynamic_cast<crs_controls::Rectangle*>(obstacle.get())))
    {
      visualizeRectangle(*rectangle, marker);
    }

    crs_controls::Rhombus* rhombus = NULL;
    if ((rhombus = dynamic_cast<crs_controls::Rhombus*>(obstacle.get())))
    {
      visualizeRhombus(*rhombus, marker);
    }

    marker.color.r = 1;
    marker.color.g = .6;
    marker.color.b = .6;
    marker.color.a = 1;

    obstacles_.markers.push_back(marker);
  }

  // Also color in the background canvas
  // TODO: This should use the static environment
  double x_min = 0, x_max = 0, y_min = 0, y_max = 0;
  if (!nh_.getParam("/car_1/controller_node/model_bounds/x_min", x_min))
    ROS_WARN("Failed to get parameter: x_min");
  if (!nh_.getParam("/car_1/controller_node/model_bounds/x_max", x_max))
    ROS_WARN("Failed to get parameter: x_max");
  if (!nh_.getParam("/car_1/controller_node/model_bounds/y_min", y_min))
    ROS_WARN("Failed to get parameter: y_min");
  if (!nh_.getParam("/car_1/controller_node/model_bounds/y_max", y_max))
    ROS_WARN("Failed to get parameter: y_max");

  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_name_;
  marker.header.stamp = ros::Time::now();
  marker.ns = "obstacle";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = id++;

  crs_controls::Rectangle rectangle(x_min, y_min, x_max - x_min, y_max - y_min);
  visualizeRectangle(rectangle, marker);

  marker.pose.position.z = -.05;  // make behind the other obstacles
  marker.color.r = .5;
  marker.color.g = 0;
  marker.color.b = .5;
  marker.color.a = .1;

  obstacles_.markers.push_back(marker);
}

void CarTrackVisualizer::setupMarker()
{
  uint32_t shape = visualization_msgs::Marker::CUBE;

  gt_car_marker_.header.frame_id = frame_name_;
  gt_car_marker_.header.stamp = ros::Time::now();
  gt_car_marker_.ns = car_namespace_ + "_groundtruth";
  gt_car_marker_.id = 0;

  // Set the marker type.
  gt_car_marker_.type = visualization_msgs::Marker::MESH_RESOURCE;
  gt_car_marker_.mesh_resource = "package://car_track_visualizer/config/Basic_Beetle.stl";
  gt_car_marker_.action = visualization_msgs::Marker::ADD;
  gt_car_marker_.pose.position.x = 0;
  gt_car_marker_.pose.position.y = 0;
  gt_car_marker_.pose.position.z = 0;
  gt_car_marker_.pose.orientation.x = 0.0;
  gt_car_marker_.pose.orientation.y = 0.0;
  gt_car_marker_.pose.orientation.z = 0.0;
  gt_car_marker_.pose.orientation.w = 1.0;

  // Set the scale of the marker (side lenghts of cube)

  const float actual_length = .12824;
  const float actual_width = .07126;
  const float model_length = 172.227;
  const float model_width = 68.508;
  gt_car_marker_.scale.x = actual_width / model_width;
  gt_car_marker_.scale.y = actual_length / model_length;
  gt_car_marker_.scale.z = CAR_SCALE_ / 2;

  // Set the color -- be sure to set alpha to something non-zero!
  gt_car_marker_.color.r = COLOR_CAR_GT_[0];
  gt_car_marker_.color.g = COLOR_CAR_GT_[1];
  gt_car_marker_.color.b = COLOR_CAR_GT_[2];
  gt_car_marker_.color.a = COLOR_CAR_GT_[3];

  est_car_marker_.header.frame_id = frame_name_;
  est_car_marker_.header.stamp = ros::Time::now();
  est_car_marker_.ns = car_namespace_ + "_estimated";

  est_car_marker_.id = 0;

  // Set the marker type.
  est_car_marker_.type = visualization_msgs::Marker::MESH_RESOURCE;
  est_car_marker_.mesh_resource = "package://car_track_visualizer/config/Basic_Beetle.stl";
  est_car_marker_.action = visualization_msgs::Marker::ADD;
  est_car_marker_.pose.position.x = 0;
  est_car_marker_.pose.position.y = 0;
  est_car_marker_.pose.position.z = 0;
  est_car_marker_.pose.orientation.x = 0.0;
  est_car_marker_.pose.orientation.y = 0.0;
  est_car_marker_.pose.orientation.z = 0.0;
  est_car_marker_.pose.orientation.w = 1.0;

  // Set the scale of the marker (side lenghts of cube)
  est_car_marker_.scale.x = actual_width / model_width;
  est_car_marker_.scale.y = actual_length / model_length;
  est_car_marker_.scale.z = CAR_SCALE_ / 2;

  // Set the color -- be sure to set alpha to something non-zero!
  est_car_marker_.color.r = COLOR_CAR_EST_[0];
  est_car_marker_.color.g = COLOR_CAR_EST_[1];
  est_car_marker_.color.b = COLOR_CAR_EST_[2];
  est_car_marker_.color.a = COLOR_CAR_EST_[3];

  // ===============================================
  //============== Trajectory history ==============
  // ===============================================

  est_trajectory_.header.frame_id = frame_name_;
  est_trajectory_.header.stamp = ros::Time::now();
  est_trajectory_.ns = car_namespace_ + "_estimated_trajectory";
  est_trajectory_.id = 0;

  // Set the trajectory marker type.
  est_trajectory_.type = visualization_msgs::Marker::LINE_STRIP;
  est_trajectory_.action = visualization_msgs::Marker::ADD;
  est_trajectory_.pose.orientation.x = 0.0;
  est_trajectory_.pose.orientation.y = 0.0;
  est_trajectory_.pose.orientation.z = 0.0;
  est_trajectory_.pose.orientation.w = 1.0;

  // Set the scale of the marker (side lenghts of cube)
  est_trajectory_.scale.x = TRACK_SCALE_ * 0.5;
  est_trajectory_.scale.y = TRACK_SCALE_ * 0.5;
  est_trajectory_.scale.z = TRACK_SCALE_ * 0.5;

  // Set the color -- be sure to set alpha to something non-zero!
  est_trajectory_.color.r = 0;
  est_trajectory_.color.g = 1;
  est_trajectory_.color.b = 0;
  est_trajectory_.color.a = 1;

  gt_trajectory_.header.frame_id = frame_name_;
  gt_trajectory_.header.stamp = ros::Time::now();
  gt_trajectory_.ns = car_namespace_ + "_groundtruth_trajectory";
  gt_trajectory_.id = 0;

  // Set the trajectory marker type.
  gt_trajectory_.type = visualization_msgs::Marker::LINE_STRIP;
  gt_trajectory_.action = visualization_msgs::Marker::ADD;
  gt_trajectory_.pose.orientation.x = 0.0;
  gt_trajectory_.pose.orientation.y = 0.0;
  gt_trajectory_.pose.orientation.z = 0.0;
  gt_trajectory_.pose.orientation.w = 1.0;

  // Set the scale of the marker (side lenghts of cube)
  gt_trajectory_.scale.x = TRACK_SCALE_ * 0.5;
  gt_trajectory_.scale.y = TRACK_SCALE_ * 0.5;
  gt_trajectory_.scale.z = TRACK_SCALE_ * 0.5;

  // Set the color -- be sure to set alpha to something non-zero!
  gt_trajectory_.color.r = 0;
  gt_trajectory_.color.g = 1;
  gt_trajectory_.color.b = 0;
  gt_trajectory_.color.a = 1;
}

void CarTrackVisualizer::updateMarker()
{
  // sets the quaternion entries based on the yaw angle of the car
  q_.setRPY(0, 0, car_state_.yaw + M_PI / 2.0);

  gt_car_marker_.pose.position.x = car_state_.x;
  gt_car_marker_.pose.position.y = car_state_.y;
  gt_car_marker_.pose.orientation.x = q_[0];
  gt_car_marker_.pose.orientation.y = q_[1];
  gt_car_marker_.pose.orientation.z = q_[2];
  gt_car_marker_.pose.orientation.w = q_[3];

  est_car_marker_.pose.position.x = car_state_estimated_.x;
  est_car_marker_.pose.position.y = car_state_estimated_.y;
  q_.setRPY(0, 0, car_state_estimated_.yaw + M_PI / 2.0);

  est_car_marker_.pose.orientation.x = q_[0];
  est_car_marker_.pose.orientation.y = q_[1];
  est_car_marker_.pose.orientation.z = q_[2];
  est_car_marker_.pose.orientation.w = q_[3];
}

float CarTrackVisualizer::getNodeRate()
{
  return node_rate_;
}

void CarTrackVisualizer::publishMarker()
{
  if (got_car_state_ && is_valid_marker(gt_car_marker_))
  {
    pub_.publish(gt_car_marker_);
  }
  if (got_est_car_state_ && is_valid_marker(est_car_marker_))
  {
    pub_.publish(est_car_marker_);
  }

  if (show_past_gt_trajectory_ && is_valid_marker(gt_trajectory_))
  {
    pub_.publish(gt_trajectory_);
  }
  if (show_past_est_trajectory_ && is_valid_marker(est_trajectory_))
  {
    pub_.publish(est_trajectory_);
  }

  // TODO: Publish more like the other obstacles
  for (const auto& obst : obstacles_.markers)
  {
    pub_.publish(obst);
  }
}

void CarTrackVisualizer::publishTrack()
{
  // Create message from template and adjust timestamps
  visualization_msgs::MarkerArray track_msg = track_msg_;

  for (auto& p : track_msg.markers)
  {
    p.header.stamp = ros::Time::now();
  }

  static_track_pub_.publish(track_msg);
}

void CarTrackVisualizer::run()
{
  updateMarker();
  publishMarker();
}
