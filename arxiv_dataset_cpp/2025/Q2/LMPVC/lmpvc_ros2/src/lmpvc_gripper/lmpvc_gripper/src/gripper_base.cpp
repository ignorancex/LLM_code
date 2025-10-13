#include <functional>
#include "lmpvc_gripper/gripper_base.hpp"

namespace lmpvc_gripper
{
  void BasicGripper::init_default_services(){
    close_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerCloseHand>("controller_close_hand", 
      std::bind(&BasicGripper::close_hand_cb, this, std::placeholders::_1, std::placeholders::_2));
    
    open_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerOpenHand>("controller_open_hand", 
      std::bind(&BasicGripper::open_hand_cb, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(node_->get_logger(), "Gripper services created!");
  }

  void BasicGripper::close_hand_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerCloseHand::Request> request,
    std::shared_ptr<lmpvc_interfaces::srv::ControllerCloseHand::Response> response) {

    auto logger = node_->get_logger();
    RCLCPP_INFO(logger, "Request received: Close Gripper!");
    
    response->success = set_force(request->force);
    if(response->success && rclcpp::ok()) {
      std::thread{std::bind(&BasicGripper::close, this)}.detach();
      response->success = true;
    }
    else {
      RCLCPP_WARN(logger, "Close gripper: Failed to set force!");
    }
    
    if(response->success) {
      RCLCPP_INFO(logger, "Close gripper: Success!");
    }
    else {
      RCLCPP_WARN(logger, "Close gripper: Failed!");
    }
  }

  void BasicGripper::open_hand_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerOpenHand::Request> request,
    std::shared_ptr<lmpvc_interfaces::srv::ControllerOpenHand::Response> response) {
    (void)request;

    auto logger = node_->get_logger();
    bool success = false;
    RCLCPP_INFO_STREAM(logger, "Request received: Open gripper");
    
    if(rclcpp::ok()){
      std::thread{std::bind(&BasicGripper::open, this)}.detach();
      success = true;
    }
    
    response->success = success;
    RCLCPP_INFO(logger, "Close gripper: %s", response->success ? "SUCCESS" : "FAILED");
  }
} // namespace lmpvc_gripper