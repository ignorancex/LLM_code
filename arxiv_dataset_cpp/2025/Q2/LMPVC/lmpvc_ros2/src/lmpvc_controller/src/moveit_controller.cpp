#include <future>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include "lmpvc_controller/moveit_controller.hpp"
#include "lmpvc_controller/limit_cartesian_speed.hpp"

using namespace std::chrono_literals;

MoveitController::MoveitController(std::shared_ptr<rclcpp::Node>& node, std::shared_ptr<rclcpp::Node>& move_group_node, std::string group_name)
: move_group_(move_group_node, group_name)
{
    node_ = node;
    group_name_ = group_name;
    jump_threshold_ = 5.0;
    eef_step_ = 0.01;
    max_speed_ = 0.05;

    // Maximum joint velocity multiplier, 0 -> default value
    move_group_.setMaxVelocityScalingFactor(0);

    default_joint_values_ = {
                            0.0141198,
                            -0.688377,
                            -0.0150506,
                            -2.50402,
                            -0.0192199,
                            1.81296,
                            0.848094,
                            };

    // End effector can be specified manually if necessary
    //move_group_.setEndEffector("hand_tcp");
}

geometry_msgs::msg::Pose MoveitController::get_pose(){
    geometry_msgs::msg::Pose pose = move_group_.getCurrentPose().pose;
    return pose;
}

std::vector<double> MoveitController::get_joint_group_positions(){
    std::vector<double> joint_group_positions;
    const moveit::core::JointModelGroup* joint_model_group =
    move_group_.getCurrentState()->getJointModelGroup(group_name_);

    moveit::core::RobotStatePtr current_state = move_group_.getCurrentState(10);
    current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);

    return joint_group_positions;
}

bool MoveitController::plan_cartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints,
                                    std::string plan_owner){
    rclcpp::Time start = node_->get_clock()->now();

    // Stop robot and begin guarding the plan
    stop();
    std::lock_guard<std::mutex> lock(plan_mutex_);
    plan_owner_ = plan_owner;

    // Another planning option upcoming
    bool success = legacy_cartesian_planner(waypoints);

    return success;
}

bool MoveitController::plan_pose_goal(geometry_msgs::msg::Pose& pose, std::string plan_owner){

    move_group_.setPoseTarget(pose);

    std::lock_guard<std::mutex> lock(plan_mutex_);
    plan_owner_ = plan_owner;

    bool success = (move_group_.plan(plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(node_->get_logger(), "Planning to Pose Goal: %s", success ? "SUCCESS" : "FAILED");

    return success;
}

bool MoveitController::plan_joint_goal(std::vector<double> joint_group_positions, std::string plan_owner){
    std::lock_guard<std::mutex> lock(plan_mutex_);
    plan_owner_ = plan_owner;

    move_group_.setJointValueTarget(joint_group_positions);

    bool success = (move_group_.plan(plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(node_->get_logger(), "Planning to Joint Goal: %s", success ? "" : "FAILED");

    return success;
}

bool MoveitController::reset_joints(){
    bool success = false; 
    
    if(plan_joint_goal(default_joint_values_)){
        success = move(true);
    }
    
    RCLCPP_INFO(node_->get_logger(), "Returning joints to default values: %s", success ? "" : "FAILED");

    return success;
}

bool MoveitController::move(bool wait, std::string plan_owner){
    stop();
    std::lock_guard<std::mutex> lock(plan_mutex_);
    bool success = false;

    if(plan_owner != plan_owner_){
        RCLCPP_INFO(node_->get_logger(), "Plan owner did not match!");
        return success;
    }

    if(wait){
        RCLCPP_INFO(node_->get_logger(), "Execution requested");
        success = (move_group_.execute(plan_) == moveit::core::MoveItErrorCode::SUCCESS);
        RCLCPP_INFO(node_->get_logger(), "Execution finished: %s", success ? "SUCCESS" : "FAILED");
    }
    else{
        RCLCPP_INFO(node_->get_logger(), "Async Execution requested");
        success = (move_group_.asyncExecute(plan_) == moveit::core::MoveItErrorCode::SUCCESS);
        RCLCPP_INFO(node_->get_logger(), "Async Execution started: %s", success ? "SUCCESS" : "FAILED");
    }

    return success;
}

void MoveitController::limit_cartesian_speed(double max_speed){
    std::lock_guard<std::mutex> lock(max_speed_mutex_);
    max_speed_ = max_speed;
}

void MoveitController::stop(){
    RCLCPP_INFO(node_->get_logger(), "Stopping robot");
    move_group_.stop();
}

bool MoveitController::start_servo(){
    auto start_request = std::make_shared<std_srvs::srv::Trigger::Request>();
    int n = 0;
    bool success = true;

    while (!start_servo_client_->wait_for_service(1s) && n < 10) {
        ++n;
        RCLCPP_INFO(node_->get_logger(), "MoveIt Servo start service not available, waiting...");

        if(n>9){
            RCLCPP_INFO(node_->get_logger(), "Could not start MoveIt Servo, service not avaialble");
            success = false;
        }
    }

    if(success){
        std::future_status status;
        auto result = start_servo_client_->async_send_request(start_request);

        switch (status = result.wait_for(5s); status){
            case std::future_status::deferred:
                RCLCPP_INFO(node_->get_logger(), "Did not receive a response from MoveIt Servo");
                success = false;
                break;

            case std::future_status::timeout:
                success = false;
                RCLCPP_INFO(node_->get_logger(), "Did not receive a response from MoveIt Servo");
                break;

            case std::future_status::ready:
                success = result.get()->success;
                if(success){
                    RCLCPP_INFO(node_->get_logger(), "MoveIt Servo started");
                }
                else{
                    RCLCPP_INFO(node_->get_logger(), "MoveIt Servo failed to start");
                }
                break;
        }
    }

    return success;
}

bool MoveitController::legacy_cartesian_planner(const std::vector<geometry_msgs::msg::Pose>& waypoints){
    rclcpp::Time start = node_->get_clock()->now();
    moveit_msgs::msg::RobotTrajectory trajectory;

    double fraction = move_group_.computeCartesianPath(waypoints, eef_step_,
        jump_threshold_, trajectory);
    // In case of error, computeCartesianPath returns -1.0
    bool success = (fraction != -1.0);

    if(success){
        robot_trajectory::RobotTrajectory rt(move_group_.getCurrentState()->getRobotModel(), group_name_);
        rt.setRobotTrajectoryMsg(*(move_group_.getCurrentState()), trajectory);

        // Cartesian speed limiting!
        std::lock_guard<std::mutex> lock(max_speed_mutex_);
        lmpvc_trajectory_processing::limitMaxCartesianLinkSpeed(node_, rt, max_speed_);

        rt.getRobotTrajectoryMsg(trajectory);
    }

    RCLCPP_INFO(node_->get_logger(), "Planning Cartesian Path: %s (%.2f%% achieved)",
                success ? "SUCCESS" : "FAILED", fraction * 100.0);

    
    rclcpp::Time end = node_->get_clock()->now();

    plan_.trajectory_ = trajectory;
    plan_.planning_time_ = (end - start).seconds();
    
    return success;
}

bool MoveitController::pilz_industrial_motion_planner(const std::vector<geometry_msgs::msg::Pose>& waypoints,
                        std::vector<std::string> ids, double blend_radius){
    (void)waypoints;
    (void)ids;
    (void)blend_radius;
    bool success = true;
    return success;
}

void MoveitController::twist_cb(const geometry_msgs::msg::TwistStamped& msg) const{
    twist_pub_->publish(msg);
}