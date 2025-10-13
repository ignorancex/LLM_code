#include <functional>
#include "lmpvc_controller/moveit_services.hpp"

MoveitServices::MoveitServices(std::shared_ptr<rclcpp::Node>& node, std::shared_ptr<MoveitController>& controller){
    // Saving pointers to associated node and controller objects
    node_ = node;
    controller_ = controller;

    // Creating service on the received node
    // All the std::bind nonsense is required to use the member functions of this object instance as callbacks
    execute_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerExec>("controller_exec", 
                std::bind(&MoveitServices::execute_cb, this, std::placeholders::_1, std::placeholders::_2));
    
    get_pose_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerGetPose>("controller_get_pose", 
                std::bind(&MoveitServices::get_pose_cb, this, std::placeholders::_1, std::placeholders::_2));
    
    plan_cartesian_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerPlan>("controller_plan", 
                std::bind(&MoveitServices::plan_cartesian_cb, this, std::placeholders::_1, std::placeholders::_2));

    set_speed_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerSetSpeed>("controller_set_speed", 
                std::bind(&MoveitServices::set_speed_cb, this, std::placeholders::_1, std::placeholders::_2));  

    stop_service_ = node_->create_service<lmpvc_interfaces::srv::ControllerStop>("controller_stop", 
                std::bind(&MoveitServices::stop_cb, this, std::placeholders::_1, std::placeholders::_2));                
    joint_reset_service_ = node_->create_service<std_srvs::srv::Trigger>("controller_reset_joints", 
        std::bind(&MoveitServices::joint_reset_cb, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(node_->get_logger(), "Services created!");
}

void MoveitServices::execute_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerExec::Request> request,
                                std::shared_ptr<lmpvc_interfaces::srv::ControllerExec::Response> response){
    RCLCPP_INFO(node_->get_logger(), "Request received: Execute");
    // Unused parameter
    (void)request;

    response->success = controller_->move(true);

    RCLCPP_INFO(node_->get_logger(), "Request served: Stop");
}

void MoveitServices::get_pose_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerGetPose::Request> request,
                                std::shared_ptr<lmpvc_interfaces::srv::ControllerGetPose::Response> response){
    RCLCPP_INFO(node_->get_logger(), "Request received: Get Pose");
    // Unused parameter
    (void)request;

    response->pose = controller_->get_pose();

    RCLCPP_INFO(node_->get_logger(), "Request served: Get Pose");
}

void MoveitServices::plan_cartesian_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerPlan::Request> request,
                                        std::shared_ptr<lmpvc_interfaces::srv::ControllerPlan::Response> response){
    RCLCPP_INFO(node_->get_logger(), "Request received: Plan");

    response->success = controller_->plan_cartesian(request->waypoints);

    RCLCPP_INFO(node_->get_logger(), "Request served: Plan");
}

void MoveitServices::set_speed_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerSetSpeed::Request> request,
                            std::shared_ptr<lmpvc_interfaces::srv::ControllerSetSpeed::Response> response){
    RCLCPP_INFO(node_->get_logger(), "Request received: Set Speed");
    // Unused parameter

    controller_->limit_cartesian_speed(request->speed);
    response->success = true;

    RCLCPP_INFO(node_->get_logger(), "Request served: Set Speed");
}

void MoveitServices::stop_cb(const std::shared_ptr<lmpvc_interfaces::srv::ControllerStop::Request> request,
                            std::shared_ptr<lmpvc_interfaces::srv::ControllerStop::Response> response){
    RCLCPP_INFO(node_->get_logger(), "Request received: Stop");
    // Unused parameter
    (void)request;

    controller_->stop();
    response->success = true;

    RCLCPP_INFO(node_->get_logger(), "Request served: Stop");
}

void MoveitServices::joint_reset_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response){

    RCLCPP_INFO(node_->get_logger(), "Request received: Reset Joints");
    // Unused parameter
    (void)request;

    response->success = controller_->reset_joints();

    RCLCPP_INFO(node_->get_logger(), "Request served: Reset Joints");

}