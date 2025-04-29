#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <boost/bind.hpp>
#include <plan_manage/ego_replan_fsm.h>

using namespace ego_planner;

bool enable_ = false;



void enableCallback(const std_msgs::Bool::ConstPtr& msg, EGOReplanFSM* rebo_replan)
{
  enable_ = msg->data;
  ROS_INFO("get enable command");
  std::cout << enable_ << endl;
  if (enable_) 
    {
        // ROS_INFO("Enable ego-planner node");
        rebo_replan->reset();
        // EGOReplanFSM rebo_replan;
    }
    else 
    {
        // ROS_INFO("Disable ego-planner node");
        rebo_replan->reset();

    }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ego_planner_node");
  // ros::NodeHandle planner_settings;
  ros::NodeHandle nh("~");
  ros::NodeHandle node;
  // planner_settings = nh;
  // EGOReplanFSM rebo_replan;
  EGOReplanFSM rebo_replan;

  rebo_replan.init(nh);
  // cout << "create subscriber for /planning/enable" << endl;
  // ros::Subscriber enable_sub = nh.subscribe("/planning/enable", 10, enableCallback);
  ros::Subscriber enable_sub = nh.subscribe<std_msgs::Bool>("/planning/enable", 10, boost::bind(enableCallback, _1, &rebo_replan));
  // cout << "subsriber created" << endl;


  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}


