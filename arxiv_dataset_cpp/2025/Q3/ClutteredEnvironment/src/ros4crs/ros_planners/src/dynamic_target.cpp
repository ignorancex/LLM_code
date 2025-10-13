#include <ros_planners/dynamic_target.h>
#include <interactive_markers/interactive_marker_server.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <ros/publisher.h>
#include <ros/node_handle.h>
#include <cmath>
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/Vector3.h"
#include "visualization_msgs/InteractiveMarkerFeedback.h"
#include "visualization_msgs/Marker.h"

namespace ros_planner
{
struct DynamicTarget::Private
{
  Callback on_target_change_function_;
  interactive_markers::InteractiveMarkerServer server_;

  // Create markers for when we replay the rosbag, so we can visualize the current and new targets
  visualization_msgs::Marker current_target_clone_;
  visualization_msgs::Marker pending_target_clone_;

  ros::Publisher target_clone_publisher_;

  Private(ros::NodeHandle nh_private, Callback callback, Params params)
    : on_target_change_function_{ callback }
    , server_(interactive_markers::InteractiveMarkerServer("destination_marker"))
  {
    tf::Vector3 initial_position(params.initial_x, params.initial_y, .0001);
    createDestinationMarker(initial_position);

    // mark the initial target change
    if (params.start_immediately)
    {
      on_target_change_function_(params.initial_x, params.initial_y);
    }

    /// Initialize the current and pending clone
    // common scale and pose
    geometry_msgs::Vector3 scale;
    scale.x = 0.1;
    scale.y = 0.1;
    scale.z = 0.001;

    geometry_msgs::Quaternion orientation;
    orientation.w = 1;
    orientation.x = 0;
    orientation.y = 0;
    orientation.z = 0;

    current_target_clone_.header.frame_id = "crs_frame";
    current_target_clone_.header.stamp = ros::Time::now();
    current_target_clone_.id = 0;
    current_target_clone_.ns = "current";
    current_target_clone_.type = visualization_msgs::Marker::CYLINDER;
    current_target_clone_.action = visualization_msgs::Marker::ADD;

    current_target_clone_.color.r = 1.0;
    current_target_clone_.color.b = 0.0;
    current_target_clone_.color.g = 0.0;
    current_target_clone_.color.a = 1.0;
    current_target_clone_.scale = scale;
    current_target_clone_.pose.position.z = 0.0;  // xy-position specified in callback
    current_target_clone_.pose.orientation = orientation;

    pending_target_clone_.header.frame_id = "crs_frame";
    pending_target_clone_.header.stamp = ros::Time::now();
    pending_target_clone_.id = 1;
    pending_target_clone_.ns = "pending";
    pending_target_clone_.type = visualization_msgs::Marker::CYLINDER;
    pending_target_clone_.action = visualization_msgs::Marker::ADD;

    pending_target_clone_.color.r = .8;
    pending_target_clone_.color.b = .0;
    pending_target_clone_.color.g = .75;
    pending_target_clone_.color.a = .5;
    pending_target_clone_.scale = scale;
    pending_target_clone_.pose.position.z = -.0001;
    pending_target_clone_.pose.orientation = orientation;

    // create the subscribers for the marker clones
    target_clone_publisher_ = nh_private.advertise<visualization_msgs::Marker>("visualization/target_clone", 10);
  }

  void createDestinationMarker(const tf::Vector3& position)
  {
    // create an interactive marker for our server
    visualization_msgs::InteractiveMarker int_marker;
    int_marker.header.frame_id = "crs_frame";
    int_marker.header.stamp = ros::Time::now();
    tf::pointTFToMsg(position, int_marker.pose.position);
    int_marker.scale = .3;

    int_marker.name = "destination";
    int_marker.description = "2D Move destination";

    visualization_msgs::InteractiveMarkerControl control;

    control.orientation.w = 1 / std::sqrt(2);
    control.orientation.x = 0;
    control.orientation.y = 1 / std::sqrt(2);
    control.orientation.z = 0;
    control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_PLANE;
    int_marker.controls.push_back(control);

    // make a box which also moves in the plane
    control.markers.push_back(makeBox(int_marker));
    control.always_visible = true;
    int_marker.controls.push_back(control);

    auto callback = boost::bind(&DynamicTarget::Private::processFeedback, this, boost::placeholders::_1);
    server_.insert(int_marker, callback);

    // 'commit' changes and send to all clients
    server_.applyChanges();
  }

  void processFeedback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback)
  {
    std::ostringstream s;
    double x = feedback->pose.position.x;
    double y = feedback->pose.position.y;
    switch (feedback->event_type)
    {
      case visualization_msgs::InteractiveMarkerFeedback::MOUSE_UP: {
        // Update the current target clone
        current_target_clone_.pose.position.x = x;
        current_target_clone_.pose.position.y = y;
        current_target_clone_.header.stamp = ros::Time::now();
        target_clone_publisher_.publish(current_target_clone_);

        ROS_INFO_STREAM(s.str() << "publishing pt " << x << ", " << y);
        on_target_change_function_(x, y);
        break;
      }
      case visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE: {
        // Update the pending target clone
        pending_target_clone_.pose.position.x = x;
        pending_target_clone_.pose.position.y = y;
        pending_target_clone_.header.stamp = ros::Time::now();
        target_clone_publisher_.publish(pending_target_clone_);
      }
      default:
        break;
    }
  }

  visualization_msgs::Marker makeBox(visualization_msgs::InteractiveMarker& msg)
  {
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.scale.x = msg.scale;
    marker.scale.y = msg.scale;
    marker.scale.z = .001;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    return marker;
  }
};

DynamicTarget::DynamicTarget(ros::NodeHandle nh_private, Callback callback, Params params)
  : impl_(std::make_unique<Private>(nh_private, callback, params))
{
}

DynamicTarget::~DynamicTarget()
{
}

}  // namespace ros_planner
