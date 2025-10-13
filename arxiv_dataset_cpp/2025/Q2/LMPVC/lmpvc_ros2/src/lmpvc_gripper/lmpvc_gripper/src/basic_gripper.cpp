#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <lmpvc_gripper/gripper_base.hpp>

int main(int argc, char** argv){
  // Initialize ROS and create a Node
  rclcpp::init(argc, argv);

  std::string node_name = "lmpvc_gripper";
  rclcpp::NodeOptions node_options;

  std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>(node_name, node_options);

  pluginlib::ClassLoader<lmpvc_gripper::BasicGripper> poly_loader("lmpvc_gripper", "lmpvc_gripper::BasicGripper");

  try{
    std::shared_ptr<lmpvc_gripper::BasicGripper> gripper = poly_loader.createSharedInstance("lmpvc_gripper_dummy_plugins::DummyGripper");
    gripper->init(node);
    gripper->open();
    gripper->close();
    gripper->set_force(1.0);
  }
  catch(pluginlib::PluginlibException& ex){
    RCLCPP_ERROR(node->get_logger(), "Gripper plugin failed to load. Error: %s\n", ex.what());
  }

  return 0;
}