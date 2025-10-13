#include <rclcpp/logging.hpp>
#include "lmpvc_gripper_dummy_plugins/dummy_gripper.hpp"

namespace lmpvc_gripper_dummy_plugins
{
  void DummyGripper::init(std::shared_ptr<rclcpp::Node> node){
    node_ = node;
    auto logger = node_->get_logger();
    RCLCPP_INFO(logger, "Dummy Gripper Plugin Initialized!");
  }

  bool DummyGripper::open(){
    auto logger = node_->get_logger();
    RCLCPP_INFO(logger, "Dummy Grasp!");
    return true;
  }

  bool DummyGripper::close(){
    auto logger = node_->get_logger();
    RCLCPP_INFO(logger, "Dummy Release!");
    return true;
  }

  bool DummyGripper::set_force(double force){
    auto logger = node_->get_logger();
    force_ = force;
    RCLCPP_INFO(logger, "Dummy Force Change!");
    return true;
  }
}

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(lmpvc_gripper_dummy_plugins::DummyGripper, lmpvc_gripper::BasicGripper)