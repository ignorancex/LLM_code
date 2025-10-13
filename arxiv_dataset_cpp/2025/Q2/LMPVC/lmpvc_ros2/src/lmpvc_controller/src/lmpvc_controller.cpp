#include <cstdio>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <pluginlib/class_loader.hpp>
#include "lmpvc_controller/moveit_controller.hpp"
#include "lmpvc_controller/moveit_services.hpp"
#include "lmpvc_gripper/gripper_base.hpp"

using namespace std::chrono_literals;

void test(std::shared_ptr<MoveitController>& controller, rclcpp::Node::SharedPtr& node){
  RCLCPP_INFO(node->get_logger(), "Starting test sequence");
  
  // Creating a pose goal
  geometry_msgs::msg::Pose target_pose;
  target_pose = controller->get_pose();
  target_pose.position.z += 0.05;

  // Create a carthesian path
  std::vector<geometry_msgs::msg::Pose> waypoints;
  target_pose = controller->get_pose();
  target_pose.position.y += 0.15;
  waypoints.push_back(target_pose);
  target_pose = controller->get_pose();
  target_pose.position.y -= 0.15;
  waypoints.push_back(target_pose);
  target_pose = controller->get_pose();
  waypoints.push_back(target_pose);

  controller->limit_cartesian_speed(0.1);
  controller->plan_cartesian(waypoints);
  controller->move(true);

  waypoints.clear();
  target_pose = controller->get_pose();
  target_pose.position.y += 0.15;
  waypoints.push_back(target_pose);
  target_pose = controller->get_pose();
  target_pose.position.y -= 0.15;
  waypoints.push_back(target_pose);
  target_pose = controller->get_pose();
  waypoints.push_back(target_pose);

  controller->limit_cartesian_speed(0.2);
  controller->plan_cartesian(waypoints);
  controller->move(true);

  RCLCPP_INFO(node->get_logger(), "Test sequence finished");
}

int main(int argc, char ** argv)
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);

  std::string node_name = "lmpvc_controller";
  rclcpp::NodeOptions node_options;

  std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>(node_name, node_options);
  std::shared_ptr<rclcpp::Node> move_group_node = std::make_shared<rclcpp::Node>("move_group", node_options);

  RCLCPP_INFO(node->get_logger(), "LMPVC Controller starting");

  // Executor to spin the node
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(move_group_node);
  std::thread([&executor]() { executor.spin(); }).detach();
  
  // Initialize the MoveIt controller, move group configurable by parameter
  std::string planning_group_name;
  node->declare_parameter("planning_group_name", "fr3_manipulator");
  node->get_parameter("planning_group_name", planning_group_name);

  RCLCPP_INFO(node->get_logger(), "Using Move Group: %s", planning_group_name.c_str());

  std::shared_ptr<MoveitController> controller = std::make_shared<MoveitController>(node, move_group_node, planning_group_name);

  // Start services
  MoveitServices services(node, controller);

  // Trying to load a gripper control plugin, configrable by parameter
  pluginlib::ClassLoader<lmpvc_gripper::BasicGripper> gripper_loader("lmpvc_gripper", "lmpvc_gripper::BasicGripper");
  std::string gripper_plugin;
  std::shared_ptr<lmpvc_gripper::BasicGripper> gripper;

  bool gripper_enabled;
  node->declare_parameter("gripper_enabled", false);
  node->get_parameter("gripper_enabled", gripper_enabled);
  
  if(gripper_enabled){

    node->declare_parameter("gripper_plugin", "lmpvc_gripper_dummy_plugins::DummyGripper");
    node->get_parameter("gripper_plugin", gripper_plugin);

    try{
      gripper = gripper_loader.createSharedInstance(gripper_plugin);
    }
    catch(pluginlib::PluginlibException& ex){
      RCLCPP_ERROR(node->get_logger(), "Gripper plugin failed to load. Error: %s\n", ex.what());
      rclcpp::shutdown();
    }

    gripper->init(node);
    gripper->init_default_services();
  }

  // Optional test, configurable by parameter
  bool tests_enabled = false;
  node->declare_parameter("tests_enabled", false);
  node->get_parameter("tests_enabled", tests_enabled);

  if(tests_enabled){
    RCLCPP_INFO(node->get_logger(), "LMPVC Controller running in test mode");
    test(controller, node);
  }

  RCLCPP_INFO(node->get_logger(), "LMPVC Controller ready");

  //controller->start_servo();

  while(rclcpp::ok()){
    rclcpp::sleep_for(1s);
  };

  // Shutdown ROS
  rclcpp::shutdown();

  
  gripper.reset();

  return 0;
}