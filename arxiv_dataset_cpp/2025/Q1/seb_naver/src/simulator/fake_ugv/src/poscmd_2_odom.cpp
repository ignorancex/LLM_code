#include <iostream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>

#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "tank_sdk/TankCMD.h"

using namespace std;

ros::Subscriber tank_cmd_sub, traj_sub;
ros::Publisher  odom_pub, cloud_pub, marker_pub, traj_pub;
ros::Timer cloud_timer;

tank_sdk::TankCMD tank_cmd;
sensor_msgs::PointCloud2 world_cloud_msg;
nav_msgs::Odometry new_odom;
visualization_msgs::Marker diff_marker;
ros::Time begin_time;

pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXY>::Ptr world_cloud_plane(new pcl::PointCloud<pcl::PointXY>());
pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
pcl::KdTreeFLANN<pcl::PointXY> kd_tree_plane;
std::string map_name;
Eigen::Vector3d now_p;
Eigen::Matrix3d now_R;
double p_init_x, p_init_y, p_init_z, yaw_init, max_height;
double time_resolution = 0.01;
bool rcv_cmd = false;
double x = 0.0;
double y = 0.0;
double v = 0.0;
double steer = 0.0;
double yaw = 0.0;
double vx = 0.0;
double vy = 0.0;
double w = 0.0;

// simulator parameters
// double init_x = 0.0;
// double init_y = 0.0;
// double init_v = 0.0;
// double init_steer = 0.0;
// double init_yaw = 0.0;
double max_steer = 0.7;
double max_steer_vel = 7.0;
double time_delay = 0.0;
double wheel_base = 0.6;
// double noise_percent = 0.0;
// double noise_percent = 0.1;
Eigen::Quaterniond q_mesh;
Eigen::Vector3d pos_mesh;


Eigen::Vector4d getTPM(Eigen::Vector3d pos, vector<Eigen::Vector3d> points)
{
	Eigen::Vector4d tpm;

	Eigen::Vector3d mean_points = Eigen::Vector3d::Zero();
	for (size_t i=0; i<points.size(); i++)
		mean_points+=points[i];

	mean_points /= (double)points.size();

	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	for (size_t i=0; i<points.size(); i++)
	{
		Eigen::Vector3d v = points[i] - mean_points;
		cov += v * v.transpose();
	}
	cov /= (double)points.size();
	Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
	Eigen::Matrix<double, 3, 1> D = es.pseudoEigenvalueMatrix().diagonal();// eigenValue
	Eigen::Matrix3d V = es.pseudoEigenvectors();    // eigenVector
	Eigen::MatrixXd::Index evalsMax;
	D.minCoeff(&evalsMax);
	Eigen::Matrix<double, 3, 1> n = V.col(evalsMax);
	n.normalize();
	if (n(2, 0) < 0.0)
		n = -n;

	tpm[3] = D(evalsMax) / D.sum() * 3.0;
	if (isnan(tpm[3]))
		n = Eigen::Vector3d(1.0, 0.0, 0.0);
	tpm[3] = mean_points.z();
	tpm[0] = n(0, 0);
	tpm[1] = n(1, 0);
	tpm[2] = n(2, 0);

	return tpm;
}

void getSE3(const Eigen::Vector3d& se2_odom, 
			Eigen::Vector3d& p,
			Eigen::Matrix3d& R)
{
	p(0) = se2_odom(0);
	p(1) = se2_odom(1);
	vector<int> Idxs;
	vector<float> SquaredDists;
	pcl::PointXY pxy;
	pxy.x = se2_odom(0);
	pxy.y = se2_odom(1);
	if (kd_tree_plane.nearestKSearch(pxy, 1, Idxs, SquaredDists) > 0)
	{
		double may_z = world_cloud->points[Idxs[0]].z;
		if (fabs(may_z - p.z()) < 0.3)
		{
			p(2) = may_z;
		}
	}
	else
		ROS_ERROR("no points in the map");
	vector<Eigen::Vector3d> points;
	pcl::PointXYZ pt;
	pt.x = se2_odom(0);
	pt.y = se2_odom(1);
	pt.z = p(2);
	if (kd_tree.radiusSearch(pt, 0.5, Idxs, SquaredDists) > 0)
	{
		for (size_t i=0; i<Idxs.size(); i++)
			points.emplace_back(Eigen::Vector3d(world_cloud->points[Idxs[i]].x, \
												world_cloud->points[Idxs[i]].y, \
												world_cloud->points[Idxs[i]].z ));
	}
	if (points.empty())
	{	
		Eigen::Quaterniond q(cos(se2_odom(2)), 0.0, 0.0, sin(se2_odom(2)));
		R = q.toRotationMatrix();
	}
	else
	{
		Eigen::Vector4d tpm = getTPM(p, points);
		Eigen::Vector3d xyaw(cos(se2_odom(2)), sin(se2_odom(2)), 0.0);
		Eigen::Vector3d zb = tpm.head(3);
		Eigen::Vector3d yb = zb.cross(xyaw).normalized();
		Eigen::Vector3d xb = yb.cross(zb);
		R.col(0) = xb;
		R.col(1) = yb;
		R.col(2) = zb;
		p(2) = tpm(3);
	}
	return;
}

void initParams()
{
	// global map
	pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);   // hxlin fixed : change global map
	pcl::PLYReader ply_reader;
	pcl::PCDReader pcd_reader;
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(ros::package::getPath("kimatic_simulator")+"/maps/"+map_name, *world_cloud)==-1) // load global map
	{
		printf("\nCouldn't read %sfile.\n\n",map_name);
	}
	world_cloud->width = world_cloud->points.size();
	world_cloud->height = 1;
	world_cloud->is_dense = true;
	world_cloud->header.frame_id = "map";
	kd_tree.setInputCloud(world_cloud);
	pcl::toROSMsg(*world_cloud, world_cloud_msg);

	for (size_t i=0; i<world_cloud->points.size(); i++)
	{
		pcl::PointXY p;
		p.x = world_cloud->points[i].x;
		p.y = world_cloud->points[i].y;
		world_cloud_plane->points.push_back(p);
	}
	world_cloud_plane->width = world_cloud_plane->points.size();
	world_cloud_plane->height = 1;
	world_cloud_plane->is_dense = true;
	world_cloud_plane->header.frame_id = "map";
	kd_tree_plane.setInputCloud(world_cloud_plane);

	// find initial position
	now_p.z() = p_init_z;
	getSE3(Eigen::Vector3d(p_init_x, p_init_y, yaw_init), now_p, now_R);
	new_odom.header.stamp    = ros::Time::now();
	new_odom.header.frame_id = "map";
	Eigen::Vector3d lidar_odom = now_p + now_R.col(2) * max_height;
	new_odom.pose.pose.position.x = x = lidar_odom.x();
	new_odom.pose.pose.position.y = y = lidar_odom.y();
	new_odom.pose.pose.position.z = lidar_odom.z();

	yaw = yaw_init;
	Eigen::Quaterniond q(now_R);
	new_odom.pose.pose.orientation.w = q.w();
	new_odom.pose.pose.orientation.x = q.x();
	new_odom.pose.pose.orientation.y = q.y();
	new_odom.pose.pose.orientation.z = q.z();
	new_odom.twist.twist.linear.x = 0.0;
	new_odom.twist.twist.linear.y = 0.0;
	new_odom.twist.twist.linear.z = 0.0;
	new_odom.twist.twist.angular.x = 0.0;
	new_odom.twist.twist.angular.y = 0.0;
	new_odom.twist.twist.angular.z = 0.0;

	diff_marker.header.frame_id = "map";
	diff_marker.id = 0;
	diff_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
	diff_marker.action = visualization_msgs::Marker::ADD;
	diff_marker.pose = new_odom.pose.pose;
	diff_marker.color.a = 1.0;
	diff_marker.color.r = 0.5;
	diff_marker.color.g = 0.5;
	diff_marker.color.b = 0.5;
	diff_marker.scale.x = 0.35;
	diff_marker.scale.y = 0.35;
	diff_marker.scale.z = 0.35;
	diff_marker.mesh_resource = "package://kimatic_simulator/meshes/car.dae";
}

void rcvVelCmdCallBack(const tank_sdk::TankCMD cmd)
{	
	rcv_cmd 	= true;
	tank_cmd    = cmd;
}

void rcvTrajCallBack(const nav_msgs::Path& traj) {
	nav_msgs::Path path_msg;
	path_msg = traj;
	geometry_msgs::PoseStamped tmpPose;

	for(auto& pt: path_msg.poses){
		double x = pt.pose.position.x;
		double y = pt.pose.position.y;
		double yaw = pt.pose.position.z;
		getSE3(Eigen::Vector3d(x, y, yaw), now_p, now_R);
		Eigen::Vector3d update_p = now_p + now_R.col(2) * max_height;
		pt.pose.position.z = update_p.z();
	}
	
	path_msg.header.frame_id = "map";
	path_msg.header.stamp = ros::Time::now();
	traj_pub.publish(path_msg);
}

void normyaw(double& y)
{
  if (y > M_PI)
  {
    y-=2*M_PI;
  }
  else if (y < -M_PI)
  {
    y+=2*M_PI;
  }
}

double calYawFromR(Eigen::Matrix3d R)
{
	Eigen::Vector2d p(R(0, 2), R(1, 2));
	Eigen::Vector2d b(R(0, 0), R(1, 0));
	Eigen::Vector2d x = (Eigen::Matrix2d::Identity()+p*p.transpose()/(1.0-p.squaredNorm()))*b;
	return atan2(x(1), x(0));
}
void normYaw(double& th)
{
	while (th > M_PI)
		th -= M_PI * 2;
	while (th < -M_PI)
		th += M_PI * 2;
}
void odomPubCallBack(const ros::TimerEvent& event)
{

	if ((ros::Time::now() - begin_time).toSec() > 5.0)
		cloud_timer.stop();
	if (!rcv_cmd)
	{
		new_odom.header.stamp = ros::Time::now();
    	odom_pub.publish(new_odom);
		diff_marker.header.stamp = ros::Time::now();
		marker_pub.publish(diff_marker);
		return;
	}
	vx = v * cos(yaw);
	vy = v * sin(yaw);
	w = v * tan(steer) / wheel_base;
	x = x + vx * time_resolution;
	y = y + vy * time_resolution;
	yaw = yaw + w * time_resolution;
	v = tank_cmd.velocity;
	steer = tank_cmd.angle_velocity;
	normYaw(yaw);
	new_odom.header.stamp = ros::Time::now();
	new_odom.pose.pose.position.x  = x;
	new_odom.pose.pose.position.y  = y;

	/*hxlin debug*/
	getSE3(Eigen::Vector3d(x, y, yaw), now_p, now_R);
	Eigen::Vector3d update_p = now_p + now_R.col(2) * max_height;
	new_odom.pose.pose.position.z  = update_p.z();


	new_odom.pose.pose.orientation.w  = cos(yaw/2.0);
	new_odom.pose.pose.orientation.x  = 0;
	new_odom.pose.pose.orientation.y  = 0;
	new_odom.pose.pose.orientation.z  = sin(yaw/2.0);
	new_odom.twist.twist.linear.x  = vx;
	new_odom.twist.twist.linear.y  = vy;
	new_odom.twist.twist.linear.z  = 0;
	new_odom.twist.twist.angular.x = v;
	new_odom.twist.twist.angular.y = steer;
	new_odom.twist.twist.angular.z = w;

	Eigen::Quaterniond qyaw(cos(yaw/2.0), 0.0, 0.0, sin(yaw/2.0));
	Eigen::Quaterniond q = (qyaw * q_mesh).normalized();
	Eigen::Matrix3d R(qyaw);
	Eigen::Vector3d dp = R*pos_mesh;
	diff_marker.header.stamp = ros::Time::now();
	diff_marker.pose.position.x = x - dp.x();
	diff_marker.pose.position.y = y - dp.y();
	/*hxlin debug*/
	getSE3(Eigen::Vector3d(x, y, yaw), now_p, now_R);
	Eigen::Vector3d update_p_ = now_p + now_R.col(2) * max_height;
	diff_marker.pose.position.z = update_p_.z()-dp.z();
	diff_marker.pose.orientation.w = q.w();
	diff_marker.pose.orientation.x = q.x();
	diff_marker.pose.orientation.y = q.y();
	diff_marker.pose.orientation.z = q.z();
	
	odom_pub.publish(new_odom);
	marker_pub.publish(diff_marker);
}

void cloudCallBack(const ros::TimerEvent& event)
{
	cloud_pub.publish(world_cloud_msg);
}

int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "ugv_kinematic_model_node");
    ros::NodeHandle nh( "~" );
	
	/* Hxlin Debug */
    nh.param("max_height", max_height, 1.0);
    nh.param("init_x", p_init_x, 1.0);
    nh.param("init_y", p_init_y, 1.0);
    nh.param("init_z", p_init_z, 1.0);
	nh.param("init_yaw", yaw_init, 0.0);
	nh.param("map_name", map_name, std::string("quick.pcd"));

	initParams();

	traj_sub = nh.subscribe( "/visualization/refinedTraj_kinoastar", 1, rcvTrajCallBack );
	tank_cmd_sub = nh.subscribe( "command", 1, rcvVelCmdCallBack );
    odom_pub  = nh.advertise<nav_msgs::Odometry>("odometry", 1);
	traj_pub = nh.advertise<nav_msgs::Path>("traj_has_height",1);
	marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1);
    cloud_pub  = nh.advertise<sensor_msgs::PointCloud2>("/global_cloud", 1);
	cloud_timer = nh.createTimer(ros::Duration(2.0), cloudCallBack);
	ros::Timer odom_timer = nh.createTimer(ros::Duration(time_resolution), odomPubCallBack);
	begin_time = ros::Time::now();
	Eigen::Matrix3d R_mesh;
	R_mesh << 1, 0, 0,
			  0, 1, 0,
			  0, 0, 1;
	q_mesh = Eigen::Quaterniond{R_mesh};
	pos_mesh = Eigen::Vector3d(0, 0, 0.0);
	ros::spin();
    return 0;
}