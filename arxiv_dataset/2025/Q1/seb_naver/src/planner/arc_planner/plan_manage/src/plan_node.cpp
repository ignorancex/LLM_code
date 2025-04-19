#pragma once
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <plan_manage/replan_fsm.h>

int main(int argc, char **argv)
{

  ros::init(argc, argv, "planner_node");
  ros::NodeHandle nh("~");

  ReplanFSM rebo_replan;

  rebo_replan.init(nh);

  ros::spin();

  return 0;
}

