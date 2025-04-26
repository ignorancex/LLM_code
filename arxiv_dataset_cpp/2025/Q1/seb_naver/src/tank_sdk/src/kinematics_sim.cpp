#include <iostream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include "tank_sdk/TankCMD.h"

using namespace std;

// ros interface
ros::Subscriber command_sub;
ros::Publisher  odom_pub;
ros::Publisher  mesh_pub;
ros::Timer simulate_timer;
ros::Time get_cmdtime;
visualization_msgs::Marker marker;

// simulator variables
std::default_random_engine generator;
std::normal_distribution<double> distribution{0.0,1.0};
tank_sdk::TankCMD immediate_cmd;
vector<tank_sdk::TankCMD> cmd_buff;
double x = 0.0;
double y = 0.0;
double yaw = 0.0;
double vx = 0.0;
double vy = 0.0;
double w = 0.0;
bool rcv_cmd = false;

// simulator parameters
double init_x = 0.0;
double init_y = 0.0;
double init_yaw = 0.0;
double time_resolution = 0.01;
double max_velocity = 1.5;
double max_angle_velocity = 2.4;
double time_delay = 0.0;
double noise_std = 0.1;
Eigen::Quaterniond q_mesh;
Eigen::Vector3d pos_mesh;

// utils
void normYaw(double& th)
{
	while (th > M_PI)
		th -= M_PI * 2;
	while (th < -M_PI)
		th += M_PI * 2;
}

Eigen::Vector2d guassRandom2d(double std)
{
	return std * Eigen::Vector2d(distribution(generator), distribution(generator));
}

Eigen::Vector3d guassRandom3d(double std)
{
	return std * Eigen::Vector3d(distribution(generator), distribution(generator), distribution(generator));
}

double guassRandom(double std)
{
	return std * distribution(generator);
}

// callBackes
void rcvCmdCallBack(const tank_sdk::TankCMDConstPtr cmd)
{	
	if (rcv_cmd==false)
	{
		rcv_cmd = true;
		cmd_buff.push_back(*cmd);
		get_cmdtime = ros::Time::now();
	}
	else
	{
		cmd_buff.push_back(*cmd);
		if ((ros::Time::now() - get_cmdtime).toSec() > time_delay)
		{
			immediate_cmd = cmd_buff[0];
			cmd_buff.erase(cmd_buff.begin());
		}
	}
}

void simCallback(const ros::TimerEvent &e)
{
	nav_msgs::Odometry new_odom;

	new_odom.header.stamp    = ros::Time::now();
	new_odom.header.frame_id = "world";

    double v = max(min(immediate_cmd.velocity, max_velocity), -max_velocity) + guassRandom(noise_std);

    w = max(min(immediate_cmd.angle_velocity, max_angle_velocity), -max_angle_velocity) + guassRandom(noise_std);
    vx = v * cos(yaw);
    vy = v * sin(yaw);

	x = x + vx * time_resolution;
	y = y + vy * time_resolution;
	yaw = yaw + w * time_resolution;
	normYaw(yaw);

	new_odom.pose.pose.position.x  = x;
	new_odom.pose.pose.position.y  = y;
	new_odom.pose.pose.position.z  = 0;
	// new_odom.pose.pose.position.z  = 1.25;
	new_odom.pose.pose.orientation.w  = cos(yaw/2.0);
	new_odom.pose.pose.orientation.x  = 0;
	new_odom.pose.pose.orientation.y  = 0;
	new_odom.pose.pose.orientation.z  = sin(yaw/2.0);;
	new_odom.twist.twist.linear.x  = vx;
	new_odom.twist.twist.linear.y  = vy;
	new_odom.twist.twist.linear.z  = 0;
	new_odom.twist.twist.angular.x = 0;
	new_odom.twist.twist.angular.y = 0;
	new_odom.twist.twist.angular.z = w;

	Eigen::Quaterniond qyaw(cos(yaw/2.0), 0.0, 0.0, sin(yaw/2.0));
	Eigen::Quaterniond q = (qyaw * q_mesh).normalized();
	Eigen::Matrix3d R(qyaw);
	Eigen::Vector3d dp = R*pos_mesh;
	marker.pose.position.x = x - dp.x();
	marker.pose.position.y = y - dp.y();
	marker.pose.position.z = 0.0;
	// marker.pose.position.z = 1.25;
	marker.pose.orientation.w = q.w();
	marker.pose.orientation.x = q.x();
	marker.pose.orientation.y = q.y();
	marker.pose.orientation.z = q.z();

	odom_pub.publish(new_odom);
	mesh_pub.publish(marker);
}

// main loop
int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "kinematics_simulator_node");
    ros::NodeHandle nh("~");

	nh.getParam("simulator/init_x", init_x);
	nh.getParam("simulator/init_y", init_y);
	nh.getParam("simulator/init_yaw", init_yaw);
	nh.getParam("simulator/time_resolution", time_resolution);
	nh.getParam("simulator/max_velocity", max_velocity);
	nh.getParam("simulator/max_angle_velocity", max_angle_velocity);
	nh.getParam("simulator/time_delay", time_delay);
	nh.getParam("simulator/noise_std", noise_std);
	  
    command_sub  = nh.subscribe("cmd", 1000, rcvCmdCallBack);
    odom_pub  = nh.advertise<nav_msgs::Odometry>("odom", 10);
	mesh_pub = nh.advertise<visualization_msgs::Marker>("mesh", 10);

	immediate_cmd.velocity = 0.0;
	immediate_cmd.angle_velocity = 0.0;

	x = init_x;
	y = init_y;
	yaw = init_yaw;
	
	marker.header.frame_id = "world";
	marker.id = 0;
	marker.type = visualization_msgs::Marker::MESH_RESOURCE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = init_x;
	marker.pose.position.y = init_y;
	marker.pose.position.z = 0.0;
	// marker.pose.position.z = 1.25;
	marker.pose.orientation.w = 0.5;
	marker.pose.orientation.x = 0.5;
	marker.pose.orientation.y = 0.5;
	marker.pose.orientation.z = 0.5;
	marker.color.a = 1.0;
	marker.color.r = 0.5;
	marker.color.g = 0.5;
	marker.color.b = 0.5;
    marker.scale.x = 0.001;
    marker.scale.y = 0.001;
    marker.scale.z = 0.001;
    q_mesh = Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5);
    pos_mesh = Eigen::Vector3d(0.5, 0.5, 0.5);
    marker.mesh_resource = "package://tank_sdk/meshes/differential_model.STL";
	
    simulate_timer = nh.createTimer(ros::Duration(time_resolution), simCallback);

	ros::spin();

    return 0;
}