#include <chrono>
#include <functional>
#include <string>

#include "rclcpp/logging.hpp"
#include "lmpvc_gripper_franka_plugins/franka_hand.hpp"

using namespace std::chrono_literals;

namespace lmpvc_gripper_franka_plugins
{
  void FrankaHand::init(std::shared_ptr<rclcpp::Node> node){

    // Setup
    this->node_ = node;
    this->force_ = 5.0;
    this->max_width_ = 0;

    auto logger = node_->get_logger();

    init_cli();

    // Homing procedure
    if(!homing()) { // Action callbacks set the result using homing_ready_
      RCLCPP_ERROR(logger, "Homing failed");
      rclcpp::shutdown();
    }

    RCLCPP_INFO(logger, "Franka Hand Plugin Initialized");
  }

  bool FrankaHand::stop(){
    std::string msg = "Unknown";
    auto logger = node_->get_logger();

    while (!stop_client_->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        msg = "Interrupted while waiting for the Stop service. Exiting.";
        RCLCPP_ERROR_STREAM(logger, "Interrupted while waiting for the Stop service. Exiting.");
        return false;
      }
      RCLCPP_INFO_STREAM(logger, "Stop service not available, waiting again...");
    }

    bool success = false;
    std::future_status status;
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto result_future = stop_client_->async_send_request(request);

    switch (status = result_future.wait_for(5s); status){
      case std::future_status::deferred:
        msg = "Did not receive a response from Stop service";
        RCLCPP_INFO_STREAM(node_->get_logger(), msg);
        success = false;
        break;

      case std::future_status::timeout:
        msg = "Did not receive a response from Stop service";
        RCLCPP_INFO_STREAM(node_->get_logger(), msg);
        success = false;
        break;

      case std::future_status::ready:
        auto result = result_future.get();
        success = result->success;
        if(success){
          msg = "Stop call successful, message:";
          RCLCPP_INFO_STREAM(node_->get_logger(), msg << result->message);
        }
        else{
          msg = "Stop call failed, message:";
          RCLCPP_INFO_STREAM(node_->get_logger(), msg << result->message);
        }
        break;
    }
  
    return success;
  }

  bool FrankaHand::close(){
    auto logger = node_->get_logger();

    if(!stop()){
      RCLCPP_ERROR_STREAM(logger, "Franka stop service call failed");
      return false;
    }

    auto grasp_msg = Grasp::Goal();
    grasp_msg.width = 0;
    grasp_msg.epsilon.outer = max_width_; // Grasp regardless of width
    grasp_msg.speed = speed_;
    grasp_msg.force = force_;

    auto goal_handle_future = grasp_client_->async_send_goal(grasp_msg, grasp_options_);

    bool success = false;

    // Wait for goal to be accepted
    for(int i=0; i<10; i++){
      RCLCPP_INFO(logger, "Waiting for response from /fr3_gripper/grasp...");

      if(goal_handle_future.wait_for(1s) == std::future_status::timeout) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(logger, "Interrupted while waiting for /fr3_gripper/grasp. Exiting.");
            return success;
        }
        else if (i>8){
            RCLCPP_ERROR(logger, "Timed out while waiting for /fr3_gripper/grasp. Exiting.");
            return success;
        }
      }
      else {
        break;
      }
    }

    auto result_future = grasp_client_->async_get_result(goal_handle_future.get());
  
    // Wait for result
    for(int i=0; i<10; i++){
      if(result_future.wait_for(1s) == std::future_status::timeout) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(logger, "Interrupted while waiting to get a result from /fr3_gripper/grasp. Exiting.");
            return success;
        }
        else if (i>8){
            RCLCPP_ERROR(logger, "Timed out while waiting to get a result from /fr3_gripper/grasp. Exiting.");
            return success;
        }
      }
      else {
        break;
      }
    }

    auto result = result_future.get();

    if(result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      success = result.result->success;

      if(result.result->success) {
        RCLCPP_INFO_STREAM(logger, "Moving succesful!");
      }
      else {
        RCLCPP_ERROR_STREAM(logger, "Moving failed: " << result.result->error);
      }
    }

    return success;
  }


  bool FrankaHand::open(){
    auto logger = node_->get_logger();

    if(!stop()){
      RCLCPP_ERROR_STREAM(logger, "Franka stop service call failed");
      return false;
    }

    auto move_msg = Move::Goal();
    move_msg.width = 0.98*max_width_; // Leaving a bit of margin to avoid errors
    move_msg.speed = speed_;

    auto goal_handle_future = move_client_->async_send_goal(move_msg, move_options_);

    bool success = false;

    // Wait for goal to be accepted
    for(int i=0; i<10; i++){
      RCLCPP_INFO(logger, "Waiting for response from /fr3_gripper/move...");

      if(goal_handle_future.wait_for(1s) == std::future_status::timeout) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(logger, "Interrupted while waiting for /fr3_gripper/move. Exiting.");
            return success;
        }
        else if (i>8){
            RCLCPP_ERROR(logger, "Timed out while waiting for /fr3_gripper/move. Exiting.");
            return success;
        }
      }
      else {
        break;
      }
    }

    auto result_future = move_client_->async_get_result(goal_handle_future.get());
  
    // Wait for result
    for(int i=0; i<10; i++){
      if(result_future.wait_for(1s) == std::future_status::timeout) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(logger, "Interrupted while waiting to get a result from /fr3_gripper/move. Exiting.");
            return success;
        }
        else if (i>8){
            RCLCPP_ERROR(logger, "Timed out while waiting to get a result from /fr3_gripper/move. Exiting.");
            return success;
        }
      }
      else {
        break;
      }
    }

    auto result = result_future.get();

    if(result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      success = result.result->success;

      if(result.result->success) {
        RCLCPP_INFO_STREAM(logger, "Moving succesful!");
      }
      else {
        RCLCPP_ERROR_STREAM(logger, "Moving failed: " << result.result->error);
      }
    }

    return success;
  }

  bool FrankaHand::set_force(double force){
    force_ = force;
    return true;
  }

  bool FrankaHand::homing(){
    auto logger = node_->get_logger();

    auto homing_msg = Homing::Goal();
    auto goal_handle_future = homing_client_->async_send_goal(homing_msg, this->homing_options_);

    bool success = false;

    // Wait for goal to be accepted
    for(int i=0; i<10; i++){
      RCLCPP_INFO(logger, "Waiting for response from /fr3_gripper/homing...");

      if(goal_handle_future.wait_for(1s) == std::future_status::timeout) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(logger, "Interrupted while waiting for /fr3_gripper/homing. Exiting.");
            return success;
        }
        else if (i>8){
            RCLCPP_ERROR(logger, "Timed out while waiting for /fr3_gripper/homing. Exiting.");
            return success;
        }
      }
      else {
        break;
      }
    }

    auto result_future = homing_client_->async_get_result(goal_handle_future.get());
  
    // Wait for result
    for(int i=0; i<10; i++){
      if(result_future.wait_for(1s) == std::future_status::timeout) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(logger, "Interrupted while waiting to get a result from /fr3_gripper/homing. Exiting.");
            return success;
        }
        else if (i>8){
            RCLCPP_ERROR(logger, "Timed out while waiting to get a result from /fr3_gripper/homing. Exiting.");
            return success;
        }
      }
      else {
        break;
      }
    }

    auto result = result_future.get();

    if(result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      success = result.result->success;

      if(result.result->success) {
        RCLCPP_INFO_STREAM(logger, "Homing succesful!");
      }
      else {
        RCLCPP_ERROR_STREAM(logger, "Homing failed: " << result.result->error);
      }
    }
    return success;
  }

  void FrankaHand::init_cli(){
    using namespace std::placeholders;

    // Stop service
    this->stop_client_ =
    this->node_->create_client<std_srvs::srv::Trigger>("/fr3_gripper/stop");

    // Homing action
    this->homing_client_ = rclcpp_action::create_client<Homing>(
      this->node_,
      "/fr3_gripper/homing");
    
    this->homing_options_ = rclcpp_action::Client<Homing>::SendGoalOptions();
    
    homing_options_.goal_response_callback =
      std::bind(&FrankaHand::homing_response_callback, this, _1);
    homing_options_.feedback_callback =
      std::bind(&FrankaHand::homing_feedback_callback, this, _1, _2);
    homing_options_.result_callback =
      std::bind(&FrankaHand::homing_result_callback, this, _1);

    // Grasp action
    this->grasp_client_ = rclcpp_action::create_client<Grasp>(
      this->node_,
      "/fr3_gripper/grasp");
    
    this->grasp_options_ = rclcpp_action::Client<Grasp>::SendGoalOptions();
    
    grasp_options_.goal_response_callback =
      std::bind(&FrankaHand::grasp_response_callback, this, _1);
    grasp_options_.feedback_callback =
      std::bind(&FrankaHand::grasp_feedback_callback, this, _1, _2);
    grasp_options_.result_callback =
      std::bind(&FrankaHand::grasp_result_callback, this, _1);

    // Move action
    this->move_client_ = rclcpp_action::create_client<Move>(
      this->node_,
      "/fr3_gripper/move");
    
    this->move_options_ = rclcpp_action::Client<Move>::SendGoalOptions();
    
    move_options_.goal_response_callback =
      std::bind(&FrankaHand::move_response_callback, this, _1);
    move_options_.feedback_callback =
      std::bind(&FrankaHand::move_feedback_callback, this, _1, _2);
    move_options_.result_callback =
      std::bind(&FrankaHand::move_result_callback, this, _1);
  }


  void FrankaHand::homing_response_callback(const HomingGoalHandle::SharedPtr & goal_handle){
    auto logger = node_->get_logger();
    if (!goal_handle) {
      RCLCPP_ERROR_STREAM(logger, "Homing goal was rejected by Franka Hand");
    } else {
      RCLCPP_INFO_STREAM(logger, "Homing goal accepted by Franka Hand, waiting for result");
    }
  }
  
  void FrankaHand::homing_feedback_callback(HomingGoalHandle::SharedPtr,
    const std::shared_ptr<const Homing::Feedback> feedback){
    max_width_ = (max_width_ < feedback->current_width) ? feedback->current_width : max_width_;
  }
  
  void FrankaHand::homing_result_callback(const HomingGoalHandle::WrappedResult & result){
    auto logger = node_->get_logger();

    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR_STREAM(logger, "Goal was aborted");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR_STREAM(logger, "Goal was canceled");
        break;
      default:
        RCLCPP_ERROR_STREAM(logger, "Unknown result code");
        break;
    }

  }
   
  void FrankaHand::grasp_response_callback(const GraspGoalHandle::SharedPtr & goal_handle){
    auto logger = node_->get_logger();
    if (!goal_handle) {
      RCLCPP_ERROR_STREAM(logger, "Grasp goal was rejected by Franka Hand");
    } else {
      RCLCPP_INFO_STREAM(logger, "Grasp goal accepted by Franka Hand, waiting for result");
    }
  }
      
  void FrankaHand::grasp_feedback_callback(GraspGoalHandle::SharedPtr,
    const std::shared_ptr<const Grasp::Feedback> feedback){
    (void)feedback;
  }

  void FrankaHand::grasp_result_callback(const GraspGoalHandle::WrappedResult & result){
    auto logger = node_->get_logger();

    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR_STREAM(logger, "Goal was aborted");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR_STREAM(logger, "Goal was canceled");
        break;
      default:
        RCLCPP_ERROR_STREAM(logger, "Unknown result code");
        break;
    }

  }

  void FrankaHand::move_response_callback(const MoveGoalHandle::SharedPtr & goal_handle){
    auto logger = node_->get_logger();
    if (!goal_handle) {
      RCLCPP_ERROR_STREAM(logger, "Move goal was rejected by Franka Hand");
    } else {
      RCLCPP_INFO_STREAM(logger, "Move goal accepted by Franka Hand, waiting for result");
    }
  }

  void FrankaHand::move_feedback_callback(MoveGoalHandle::SharedPtr,
    const std::shared_ptr<const Move::Feedback> feedback){
    (void)feedback;
  }

  void FrankaHand::move_result_callback(const MoveGoalHandle::WrappedResult & result){
    auto logger = node_->get_logger();

    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR_STREAM(logger, "Goal was aborted");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR_STREAM(logger, "Goal was canceled");
        break;
      default:
        RCLCPP_ERROR_STREAM(logger, "Unknown result code");
        break;
    }

  }

}

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(lmpvc_gripper_franka_plugins::FrankaHand, lmpvc_gripper::BasicGripper)