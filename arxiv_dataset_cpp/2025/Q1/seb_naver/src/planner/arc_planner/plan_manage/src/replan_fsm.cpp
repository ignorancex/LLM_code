#include <plan_manage/replan_fsm.h>
#include <chrono>

void ReplanFSM::init(ros::NodeHandle &nh)
{
    nh_ = nh;
    exec_state_ = ReplanFSM::FSM_EXEC_STATE::INIT;
    have_target_ = false;
    collision_with_obs_ = false;

    nh_.param("odometry_topic", odom_topic_, odom_topic_);
    nh_.param("vehicle/car_d_cr", car_d_cr_, 1.015);
    nh_.param("vehicle/car_id", car_id_, 0);
    nh_.param("fsm/target_x", target_x_, 0.0);
    nh_.param("fsm/target_y", target_y_, 0.0);
    nh_.param("fsm/target_yaw", target_yaw_, 0.0);
    nh_.param("fsm/in_swarm", in_swarm, true);
    nh_.param("safeMargin", safe_margin_, 0.5);

    se2_grid_ptr_.reset(new se2_grid::SE2Grid);
    sdf_grid_ptr_.reset(new se2_grid::SE2Grid);
    planner_.init(nh, se2_grid_ptr_, sdf_grid_ptr_);
    
    odom_sub_    = nh_.subscribe(odom_topic_, 100, &ReplanFSM::OdomCallback, this);
    parking_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &ReplanFSM::ParkingCallback, this);
    sdf_grid_sub_ = nh_.subscribe("/car_"+std::to_string(car_id_)+"_terrain_analyzer/sdf_map", 10, &ReplanFSM::MapCallback, this);
    se2_grid_sub_ = nh_.subscribe("/car_"+std::to_string(car_id_)+"_terrain_analyzer/fused_map", 10, &ReplanFSM::SEMapCallback, this);
    exec_timer_ = nh_.createTimer(ros::Duration(0.02), &ReplanFSM::execFSMCallback, this);
    safety_timer_ = nh_.createTimer(ros::Duration(0.01), &ReplanFSM::checkCollisionCallback, this);
}

void ReplanFSM::MapCallback(const se2_grid_msgs::SE2Grid::ConstPtr& msg)
{
    se2_grid::SE2GridRosConverter::fromMessage(*msg, *sdf_grid_ptr_);
    sdf_grid_ptr_->convertToDefaultStartIndex();
}

void ReplanFSM::SEMapCallback(const se2_grid_msgs::SE2Grid::ConstPtr& msg)
{
    se2_grid::SE2GridRosConverter::fromMessage(*msg, *se2_grid_ptr_);
    se2_grid_ptr_->convertToDefaultStartIndex();
}

void ReplanFSM::OdomCallback(const nav_msgs::Odometry& msg)
{
    Eigen::Vector3d center_pos(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z);
    Eigen::Vector3d pos2center(-car_d_cr_, 0, 0);

    Eigen::Quaterniond quaternion(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, 
                                  msg.pose.pose.orientation.y, msg.pose.pose.orientation.z);
    Eigen::Vector3d pos = center_pos;
    cur_pos_ = pos.head(2);
    cur_vel_ = msg.twist.twist.linear.x;
    cur_yaw_ = tf::getYaw(msg.pose.pose.orientation);
    planner_.updateOdom(cur_pos_(0), cur_pos_(1), cur_yaw_);
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(cur_pos_(0), cur_pos_(1), 0.0));
    tf::Quaternion q;
    q.setRPY(0, 0, cur_yaw_);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, msg.header.stamp, "map", "car_"+std::to_string(car_id_)+"_pos"));
}

void ReplanFSM::ParkingCallback(const geometry_msgs::PoseStamped &msg)
{
    std::cout<<"ParkingCallback "<<std::endl;
    init_state_.head(2) = cur_pos_;
    init_state_(2) = cur_yaw_;
    init_state_(3) = cur_vel_;
    end_pt_ << msg.pose.position.x, msg.pose.position.y, 
            2.0 * atan2(msg.pose.orientation.z, msg.pose.orientation.w), 1.0e-2;
    have_target_ = true;
    planner_.setTarget(msg.pose.position.x,msg.pose.position.y,end_pt_(3));
    changeFSMExecState(SEQUENTIAL_START, "FSM");

}

void ReplanFSM::execFSMCallback(const ros::TimerEvent &e)
{
    exec_timer_.stop();
    static int fsm_num = 0;
    fsm_num++;
    if (fsm_num == 50)
    {
      if (!have_target_)
      {
        std::cout << "wait for goal or trigger." << std::endl;
      }
      fsm_num = 0;
    }

    switch (exec_state_)
    {
        case INIT:
        {
            changeFSMExecState(WAIT_TARGET, "FSM");
            break;
        }

        case WAIT_TARGET:
        {
            if(!have_target_ /*|| !have_trigger_*/)
                goto force_return;

            else
            {
                changeFSMExecState(SEQUENTIAL_START, "FSM");
            }
            break;
        }

        case SEQUENTIAL_START:
        {
            init_state_ << cur_pos_, cur_yaw_, cur_vel_;

            double start_time = ros::Time::now().toSec() + TIME_BUDGET;
            start_world_time_ = start_time;
            
            ros::Time t1 = ros::Time::now();
            planner_.setInitStateAndInput(init_state_, start_time);
            planner_.setParkingEnd(end_pt_);
            if(!planner_.process()){
                break;
            }
            ros::Time t2 = ros::Time::now();
            double time_spent_in_planning = (t2 - t1).toSec();
            if(TIME_BUDGET > time_spent_in_planning)
                ros::Duration(TIME_BUDGET - time_spent_in_planning).sleep();
            else
                ROS_ERROR("Out of time budget! %f ",time_spent_in_planning);
            traj_start_time_ = ros::Time::now();
            planner_.publishSimTraj2Controller();
            changeFSMExecState(EXEC_TRAJ, "FSM");

            break;
        }

        case REPLAN_TRAJ:
        {
            // reach end
            if((cur_pos_ - end_pt_.head(2)).norm() < 1.0 && abs(cur_yaw_ - end_pt_(2) < 0.15 && abs(cur_vel_) < 0.05))
            {
                changeFSMExecState(WAIT_TARGET, "FSM");
                have_target_ = false;
                goto force_return;
            }
            ros::Time t_now = ros::Time::now();
            double replan_start_time = t_now.toSec() + TIME_BUDGET + 2.0;
            start_world_time_ = replan_start_time;
            ros::Time t1 = ros::Time::now();
            Eigen::Vector4d replan_init_state;
            replan_init_state << cur_pos_, cur_yaw_, cur_vel_;
            planner_.getPredictState(replan_start_time, replan_init_state);
            if((replan_init_state.head(2) - end_pt_.head(2)).norm() < 1.0)
            {
                changeFSMExecState(EXEC_TRAJ, "FSM");
                break;
            }
            planner_.setInitStateAndInput(replan_init_state, replan_start_time);
            planner_.setTrajStartTime(replan_start_time);
            planner_.setParkingEnd(end_pt_);
            if(!planner_.process())
            {
                while(true)
                {
                    double t_cur = ros::Time::now().toSec();
                    if(t_cur > replan_start_time + 0.1)
                    {
                        break;
                    }
                }
                break;
            }
            ros::Time t2 = ros::Time::now();
            double time_spent_in_planning = (t2 - t1).toSec();
            if(TIME_BUDGET > time_spent_in_planning)
            {
                while(true)
                {
                    double t_cur = ros::Time::now().toSec();
                    if(t_cur > replan_start_time)
                        break;
                }
            }
            else
            {
                ROS_ERROR("Out of time budget! %f ",time_spent_in_planning);
            }
            planner_.publishSimTraj2Controller();
            traj_start_time_ = ros::Time::now();
            changeFSMExecState(EXEC_TRAJ, "FSM");

            break;
        }

        case EXEC_TRAJ:
        {
            ros::Time t_now = ros::Time::now();
            // if(((cur_pos_ - init_state_.head(2)).norm() > 2.0 || (t_now.toSec() - start_world_time_) > 4.5) && (cur_pos_ - end_pt_.head(2)).norm() > 2.0)
            // {
            //     changeFSMExecState(REPLAN_TRAJ, "FSM");
            // }

            // if((collision_with_obs_) && t_now.toSec() - start_world_time_ > TIME_BUDGET) // make sure the new trajectory have been executed and then replan
            // {
            //     changeFSMExecState(REPLAN_TRAJ, "FSM");
            //     collision_with_obs_ = false;
            //     break;
            // }
            if((cur_pos_ - end_pt_.head(2)).norm() < 1.0 && abs(cur_yaw_ - end_pt_(2) < 0.15 && abs(cur_vel_) < 0.05))
            {
                changeFSMExecState(WAIT_TARGET, "FSM");
                have_target_ = false;
                goto force_return;
            }

            break;
        }
    }
    force_return:;
    exec_timer_.start();
}

void ReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, std::string pos_call)
{
    static std::string state_str[8] = {"INIT", "WAIT_TARGET", "REPLAN_TRAJ", "EXEC_TRAJ", "SEQUENTIAL_START"};
    int pre_s = int(exec_state_);

    exec_state_ = new_state;
    std::cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << std::endl;    
}

void ReplanFSM::checkCollisionCallback(const ros::TimerEvent &e)
{   
    Eigen::Vector4d state;
    double t_now = ros::Time::now().toSec();
    if(t_now - start_world_time_ < 0.1){
        return;
    }
    for (double t = t_now - start_world_time_; t < planner_.getDuration(); t += 0.1)
    {
        planner_.getPredictState(t, state);
        Eigen::Vector2d pos = state.head(2);
        if (sdf_grid_ptr_->getValue("sdf_cpu", state.head(3)) - safe_margin_)
        {
            collision_with_obs_ = true;
            return;
        }
    }
    return;
}