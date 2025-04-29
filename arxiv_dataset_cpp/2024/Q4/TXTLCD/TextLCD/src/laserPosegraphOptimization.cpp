#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <map>
#include <regex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/String.h>

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>
#include <yaml-cpp/yaml.h>

#include "text_lcd/common.h"
#include "text_lcd/tic_toc.h"

#include "text_lcd/save_pose.h"
#include "text_lcd/save_map.h"

#include "txt_loop/txt_loop.hpp"
#include "clipper/clipper.h"

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;
std::queue<std::shared_ptr<std::string>> txtBuf;

std::mutex mBuf;
std::mutex mKF;
std::mutex txt_manager_lock_;
std::mutex txt_buffer_lock;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 

std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;


std::vector<double> plane_coefficients;
bool plane_initialized = false;

//jin: ceres
bool graph_created = false;
ceres::Problem problem;
std::vector<ceres::ResidualBlockId> odom_residual_ids;
std::vector<ceres::ResidualBlockId> loop_residual_ids;

struct Pose3d {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();

  Pose3d(){
    p = Eigen::Vector3d::Zero();;
    q = Eigen::Quaterniond::Identity();
  }

  Pose3d(const Pose3d& pose){
    p = pose.p;
    q = pose.q;
  }

  Pose3d inverse()const{
    Pose3d pose;
    pose.p = -(q.inverse().matrix() * p);
    pose.q = q.inverse();
    return pose;
  }

  Pose3d operator*(Pose3d input)const{
    Pose3d pose;
    pose.q = q * input.q;
    pose.p = q.matrix() * input.p + p;
    return pose;
  }

  Eigen::Matrix4d matrix()const{
    Eigen::Matrix4d m;
    m << q.matrix(), p,
          0.0, 0.0, 0.0, 1.0;
    return m;
  }

  // The name of the data type in the g2o file format.
  static std::string name() {
    return "VERTEX_SE3:QUAT";
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;
typedef std::shared_ptr<Pose3d> Pose3dPtr;

std::istream& operator>>(std::istream& input, Pose3d& pose) {
  input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >>
      pose.q.y() >> pose.q.z() >> pose.q.w();
  // Normalize the quaternion to account for precision loss due to
  // serialization.
  pose.q.normalize();
  return input;
}

typedef std::map<int, Pose3d, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
    MapOfPoses;//todo hash map

Pose3dPtr odom_pose_prev = nullptr;
Pose3dPtr odom_pose_curr = nullptr;
Pose3dPtr odom_pose_drift = nullptr;

std::vector<Pose3dPtr> keyframePoses;
std::vector<Pose3dPtr> keyframePosesUpdated;
std::vector<double> odometry_distance;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;

bool useGPS = true;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO, pubPlane;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;
ros::Publisher pubLoopConstraintEdge;
ros::Publisher pubTxtObjects;
ros::Publisher pubGraph;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
// std::fstream pgTimeSaveStream;

class PoseGraph3dErrorTerm {
 public:
  PoseGraph3dErrorTerm(const Pose3d& t_ab_measured,
                       const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                  const T* const p_b_ptr, const T* const q_b_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_a(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_b(q_b_ptr);

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>().transpose());//! this comes from slict

    return true;
  }

  static ceres::CostFunction* Create(
      const Pose3d& t_ab_measured,
      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
        new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(int index = 0; index < keyframeTimes.size(); ++index){
      const auto& pose = keyframePoses.at(index);
      Eigen::Matrix3d rot = pose->q.matrix();
      stream << std::fixed << std::setprecision(6) << keyframeTimes.at(index) << " "
              << rot(0, 0) << " " << rot(0, 1) << " " << rot(0, 2) << " " << pose->p.x() << " "
              << rot(1, 0) << " " << rot(1, 1) << " " << rot(1, 2) << " " << pose->p.y() << " "
              << rot(2, 0) << " " << rot(2, 1) << " " << rot(2, 2) << " " << pose->p.z() << std::endl;
    }
    stream.close();
}

void saveOptimizedVerticesKITTIformat(std::string _filename)
{
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(int index = 0; index < keyframeTimes.size(); ++index){
    // for(const auto& pose : keyframePosesUpdated){
      const auto& pose = keyframePosesUpdated.at(index);
      Eigen::Matrix3d rot = pose->q.matrix();
      stream << std::fixed << std::setprecision(6) << keyframeTimes.at(index) << " "
              << rot(0, 0) << " " << rot(0, 1) << " " << rot(0, 2) << " " << pose->p.x() << " "
              << rot(1, 0) << " " << rot(1, 1) << " " << rot(1, 2) << " " << pose->p.y() << " "
              << rot(2, 0) << " " << rot(2, 1) << " " << rot(2, 2) << " " << pose->p.z() << std::endl;
    }
    stream.close();
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

Pose3d getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    Pose3d pose;
    pose.p = Eigen::Vector3d(_odom->pose.pose.position.x, _odom->pose.pose.position.y, _odom->pose.pose.position.z);
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    pose.q = Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z);
    Eigen::Vector3d extri_lidar_imu(-0.011, -0.02329, 0.04412);//todo
    pose.p = pose.p + pose.q.matrix() * extri_lidar_imu;
    return pose; 
} // getOdom

Pose3d diffTransformation(const Pose3d& _p1, const Pose3d& _p2){
  Eigen::Affine3d SE3_p1(Eigen::Translation3d(_p1.p) * _p1.q);
  Eigen::Affine3d SE3_p2(Eigen::Translation3d(_p2.p) * _p2.q);
  Eigen::Affine3d delta = SE3_p1.inverse() * SE3_p2;
  Pose3d pose;
  pose.p = delta.translation().matrix();
  pose.q = Eigen::Quaterniond(delta.rotation());
  return pose;
}

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose3d& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    Eigen::Affine3f transCur(Eigen::Translation3f(tf.p.cast<float>()) * tf.q.cast<float>());
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}


void pubPath( void )
{
    // pub odom and path 
    // nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    mKF.lock(); 
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx+=10) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose3d& pose_est = *(keyframePosesUpdated.at(node_idx)); // upodated poses

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.p.x();
        odomAftPGOthis.pose.pose.position.y = pose_est.p.y();
        odomAftPGOthis.pose.pose.position.z = pose_est.p.z();
        odomAftPGOthis.pose.pose.orientation.w = pose_est.q.cast<float>().w();
        odomAftPGOthis.pose.pose.orientation.x = pose_est.q.cast<float>().x();
        odomAftPGOthis.pose.pose.orientation.y = pose_est.q.cast<float>().y();
        odomAftPGOthis.pose.pose.orientation.z = pose_est.q.cast<float>().z();

        // odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        // pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    // pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    // static tf::TransformBroadcaster br;
    // tf::Transform transform;
    // tf::Quaternion q;
    // transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    // q.setW(odomAftPGO.pose.pose.orientation.w);
    // q.setX(odomAftPGO.pose.pose.orientation.x);
    // q.setY(odomAftPGO.pose.pose.orientation.y);
    // q.setZ(odomAftPGO.pose.pose.orientation.z);
    // transform.setRotation(q);
    // br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
} // pubPath

void ComputeCeresCost(std::vector<ceres::ResidualBlockId> &res_ids,
                          double &cost, std::vector<double>& residuals, ceres::Problem &problem){
    if (res_ids.size() == 0)
    {
        cost = -1;
        return;
    }

    ceres::Problem::EvaluateOptions e_option;
    e_option.residual_blocks = res_ids;
    e_option.num_threads = omp_get_max_threads();
    problem.Evaluate(e_option, &cost, &residuals, NULL, NULL);
}

void runCeresOpt(void)
{
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = std::thread::hardware_concurrency();

    double odom_cost, loop_cost;
    std::vector<double> odom_residuals, loop_residuals;
    std::cout << "Odom errors: " << std::endl; 
    ComputeCeresCost(odom_residual_ids, odom_cost, odom_residuals, problem);
    // for(int index = 0; index < odom_residual_ids.size(); ++index){
    //   Eigen::Vector3d ep(odom_residuals[6*index+0], odom_residuals[6*index+1], odom_residuals[6*index+2]);
    //   Eigen::Vector3d er(odom_residuals[6*index+3], odom_residuals[6*index+4], odom_residuals[6*index+5]);
    //   std::cout << "index " << index << ": " << ep.norm() << ", " << er.norm() << std::endl;
    // }
    std::cout << "odom total cost: " << odom_cost << std::endl;
    std::cout << "Loop errors: " << std::endl; 
    ComputeCeresCost(loop_residual_ids, loop_cost, loop_residuals, problem);
    // for(int index = 0; index < loop_residual_ids.size(); ++index){
    //   Eigen::Vector3d ep(loop_residuals[6*index+0], loop_residuals[6*index+1], loop_residuals[6*index+2]);
    //   Eigen::Vector3d er(loop_residuals[6*index+3], loop_residuals[6*index+4], loop_residuals[6*index+5]);
    //   std::cout << "index " << index << ": " << ep.norm() << ", " << er.norm() << std::endl;
    // }
    std::cout << "loop total cost: " << loop_cost << std::endl;

    ceres::Solver::Summary summary;
    mKF.lock();// will change pose variables
    ceres::Solve(options, &problem, &summary);//! donnot neet to copy the optimizated values, but odometry and optimization are mutual locked
    int parameter_block_num = problem.NumParameterBlocks();
    int optimized_pose_num = int(0.5 * parameter_block_num);
    if(optimized_pose_num != keyframePosesUpdated.size()){
      std::cout << "param block num: " << parameter_block_num << std::endl;
      std::cout << "keyframePosesUpdated size: " << keyframePosesUpdated.size() << std::endl;
    }
    *odom_pose_drift = (*keyframePosesUpdated.at(optimized_pose_num - 1)) * (keyframePoses.at(optimized_pose_num - 1)->inverse());
    mKF.unlock(); 
    
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "++++++++++++++++After optimization: " << std::endl; 
    ComputeCeresCost(odom_residual_ids, odom_cost, odom_residuals, problem);
    std::cout << "odom total cost: " << odom_cost << std::endl;
    ComputeCeresCost(loop_residual_ids, loop_cost, loop_residuals, problem);
    std::cout << "loop total cost: " << loop_cost << std::endl;

    mtxRecentPose.lock();
    // recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;
    recentIdxUpdated = optimized_pose_num - 1;
    mtxRecentPose.unlock();
}

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& left_submap_size, const int& right_submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -left_submap_size; i <= right_submap_size; ++i) {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
            continue;

        mKF.lock(); 
        *nearKeyframes += * local2global((keyframeLaserClouds[keyNear]), (keyframePosesUpdated[root_idx]->inverse() * (*(keyframePosesUpdated[keyNear]))));
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud


bool doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4f& relative_pose )//! jin:可返回nullptr
{
    // parse pointclouds
    int historyKeyframeSearchNum = 10; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 5, 0, _curr_kf_idx); // use same root of loop kf idx //! maybe bug
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, historyKeyframeSearchNum, _loop_kf_idx); 

    // ICP Settings
    // pcl::IterativeClosestPoint<PointType, PointType> icp;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(0.6);
    icp.setMaximumIterations(300);
    icp.setTransformationEpsilon(1e-12);
    icp.setEuclideanFitnessEpsilon(1e-12);
    icp.setRANSACIterations(0);//100
    // icp.setRANSACOutlierRejectionThreshold(0.1);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result, (keyframePosesUpdated[_loop_kf_idx]->inverse() * (*(keyframePosesUpdated[_curr_kf_idx]))).matrix().cast<float>());
 
    float loopFitnessScoreThreshold = 0.1; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[txt loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this text loop." << std::endl;
        // return std::nullopt;
        return false;
    } else {
        std::cout << "[txt loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this text loop." << std::endl;
    }

    // {
    //   static int loop_index = 0;
    //   pcl::io::savePCDFileBinaryCompressed("/media/jin/T7Shield/txt/S1b1/0121/1/test/" + std::to_string(loop_index) + "_target.pcd", *targetKeyframeCloud);
    //   pcl::io::savePCDFileBinaryCompressed("/media/jin/T7Shield/txt/S1b1/0121/1/test/" + std::to_string(loop_index) + "_source.pcd", *unused_result);
    //   loop_index++;
    // }

    relative_pose = icp.getFinalTransformation();
    return true;
} // doICPVirtualRelative


void publishUpdatedPosesandTFRealTime(){
  const Pose3d& latest_pose_opt = *(keyframePosesUpdated.back());
  nav_msgs::Odometry odom_opt;
  odom_opt.header.frame_id = "camera_init";
  odom_opt.child_frame_id = "/aft_pgo";
  odom_opt.header.stamp = ros::Time().fromSec(keyframeTimes.back());
  odom_opt.pose.pose.position.x = latest_pose_opt.p.x();
  odom_opt.pose.pose.position.y = latest_pose_opt.p.y();
  odom_opt.pose.pose.position.z = latest_pose_opt.p.z();
  odom_opt.pose.pose.orientation.w = latest_pose_opt.q.cast<float>().w();
  odom_opt.pose.pose.orientation.x = latest_pose_opt.q.cast<float>().x();
  odom_opt.pose.pose.orientation.y = latest_pose_opt.q.cast<float>().y();
  odom_opt.pose.pose.orientation.z = latest_pose_opt.q.cast<float>().z();
  pubOdomAftPGO.publish(odom_opt);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odom_opt.pose.pose.position.x, odom_opt.pose.pose.position.y, odom_opt.pose.pose.position.z));
  q.setW(odom_opt.pose.pose.orientation.w);
  q.setX(odom_opt.pose.pose.orientation.x);
  q.setY(odom_opt.pose.pose.orientation.y);
  q.setZ(odom_opt.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odom_opt.header.stamp, "camera_init", "/aft_pgo"));
}

TxtManager::Ptr txt_manager;

void process_odometry() {
    while(1){
		  while (!odometryBuf.empty() && !fullResBuf.empty()){
        sensor_msgs::PointCloud2Ptr current_cloud_msg(new sensor_msgs::PointCloud2); 
			  mBuf.lock();       
        while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
            odometryBuf.pop();//! jin: drop old odo 
        if (odometryBuf.empty()){
          mBuf.unlock();
          break;
        }

        // Time equal check
        timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
        timeLaser = fullResBuf.front()->header.stamp.toSec();

        *current_cloud_msg = *fullResBuf.front();
        fullResBuf.pop();

        Pose3d pose_curr = getOdom(odometryBuf.front());
        odometryBuf.pop();

        mBuf.unlock();//todo: reduce time occupied

        pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*current_cloud_msg, *thisKeyFrame);

        *odom_pose_prev = *odom_pose_curr;//todo check if dtf threshold is not 0!
        *odom_pose_curr = pose_curr;
        Pose3d dtf = diffTransformation(*odom_pose_prev, *odom_pose_curr); // dtf means delta_transform

        //
        // Save data and Add consecutive node 
        //
        pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
        downSizeFilterScancontext.setInputCloud(thisKeyFrame);
        downSizeFilterScancontext.filter(*thisKeyFrameDS);

        mKF.lock(); 
        // std::cout << "this keyframe poinit num:" << thisKeyFrame->points.size() << std::endl;
        keyframeLaserClouds.push_back(thisKeyFrame);//! jin
        keyframePoses.push_back(Pose3dPtr(new Pose3d(pose_curr)));
        Pose3d corrected_odom_pose = (*odom_pose_drift)*(pose_curr);
        keyframePosesUpdated.push_back(Pose3dPtr(new Pose3d(corrected_odom_pose)));
          
        keyframeTimes.push_back(timeLaserOdometry);
        if(odometry_distance.empty()){
          odometry_distance.push_back(0);
        }else{
          odometry_distance.push_back(odometry_distance.back() + dtf.p.norm());
        }

        mKF.unlock();
        publishUpdatedPosesandTFRealTime();

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << timeLaser;
        Frame::Ptr frame(new Frame(keyframePoses.size() - 1));//todo pose
        frame->lidar_timestamp_tag_ = oss.str();//! id
        txt_manager_lock_.lock();
        txt_manager->add(frame);
        txt_manager_lock_.unlock(); 

        const int prev_node_idx = keyframePoses.size() - 2; 
        const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
        if(!graph_created){
          const int init_node_idx = 0; 
          mtxPosegraph.lock();
          {
            // prior factor 
            ceres::LocalParameterization* q_local = new ceres::EigenQuaternionParameterization();
            problem.AddParameterBlock(keyframePosesUpdated[0]->q.coeffs().data(), 4, q_local);
            problem.AddParameterBlock(keyframePosesUpdated[0]->p.data(), 3);
            problem.SetParameterBlockConstant(keyframePosesUpdated[0]->q.coeffs().data());
            problem.SetParameterBlockConstant(keyframePosesUpdated[0]->p.data());
          }   
          mtxPosegraph.unlock();

          graph_created = true;

          std::cout << "posegraph prior node " << init_node_idx << " added" << std::endl;
        } else {
          mtxPosegraph.lock();
          {        
            Eigen::Matrix<double, 6, 6> sqrt_information;
            sqrt_information << 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 10.0;
            ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(diffTransformation(*(keyframePoses.at(prev_node_idx)), *(keyframePoses.at(curr_node_idx))), sqrt_information.llt().matrixL());
            ceres::LossFunction* loss_function = NULL;//todo
            ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
            ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function,
                             keyframePosesUpdated.at(prev_node_idx)->p.data(),
                             keyframePosesUpdated.at(prev_node_idx)->q.coeffs().data(),
                             keyframePosesUpdated.at(curr_node_idx)->p.data(),
                             keyframePosesUpdated.at(curr_node_idx)->q.coeffs().data());
            odom_residual_ids.push_back(id);
            problem.AddParameterBlock(keyframePosesUpdated.at(curr_node_idx)->p.data(), 3);
            problem.AddParameterBlock(keyframePosesUpdated.at(curr_node_idx)->q.coeffs().data(), 4, quaternion_local_parameterization);
          }
          mtxPosegraph.unlock();

          if(curr_node_idx % 100 == 0)
            std::cout << "posegraph odom node " << curr_node_idx << " added." << std::endl;
          } 
          // pgTimeSaveStream << timeLaser << std::endl; // path 
        }
        // std::cout << "odometry buffer is empty" << std::endl;

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

std::vector<std::string> split_line(const std::string& s, const std::string& delimiter){
  int left = 0;
  int right = 0;
  std::vector<std::string> result;
  while((right = s.find(delimiter, left)) != std::string::npos){
    if(left != right){
      result.emplace_back(s.substr(left, right - left));
    }
    left = right + delimiter.size();
  }
  if(left != s.size()){
    result.emplace_back(s.substr(left));
  }
  return result;
}

bool isLegalSymbol(char ch){
  return (ch == 45) || (ch == 47);
}

bool containsNonAscii(const std::string& str) {
    for (char ch : str) {
        if(!isdigit(ch) && !isupper(ch) && !islower(ch) && !isLegalSymbol(ch)){
          return true;
        }
    }
    return false;  // All characters are valid ASCII
}

bool allDigits(const std::string& str){
  for (char ch : str) {
      if(!isdigit(ch)){
        return false;
      }
  }
  return true;
}

void parseOCRString(const std::string& ocr_string, Frame::Ptr& frame){
  std::string line, line2, line3;
  std::string timestamp;
  std::stringstream ss(ocr_string);
  getline(ss, timestamp);
  int object_id = 0;
      while(getline(ss, line)){
        getline(ss, line2);//not process
        getline(ss, line3);//  
        std::vector<std::string> confidences = split_line(line2, " ");
        double confidence = stod(confidences.back());
        if(confidence < 0.9){// todo
          continue;
        }
        if(containsNonAscii(line) || allDigits(line)){
          // std::cout << "Contains non-ascii character" << std::endl;
        }else{
          // std::cout << "All characters are right" << std::endl;
          // std::cout << "[Txt content]==================>" << line << std::endl;
          TxtObject::Ptr object(new TxtObject(line, frame->frame_id_, object_id));//todo
          if(line.length() < 4 || line.find("#") != std::string::npos){
            object->type_ = TxtObject::type::INVALID;
          }else if(line.find("EXIT") != std::string::npos){
          // }else if(line.find("EXIT") != std::string::npos || line.find("FIRE") != std::string::npos){
              // std::cout << "EXIT" << std::endl;
              object->type_ = TxtObject::type::EXIT;
          }else{
            // Define the pattern
            std::regex pattern("^[A-Z]\\d[A-Za-z]-.*$");
            // Check if the string matches the pattern
            if (std::regex_match(line, pattern)) {//!
              // std::cout << "The string follows the pattern." << std::endl;
              std::string second_section = line.substr(4);
              if(second_section.find('/', 0) != std::string::npos){
                  // std::cout << "may emerge more than once" << std::endl;
                  object->type_ = TxtObject::type::ORDINARY;
              }else{
                  if(second_section.length() == 2){// B3a-S1/E1/F1 B3a-01
                      if(std::isdigit(second_section[0]) && std::isdigit(second_section[1])){
                          // std::cout << "unique room number" << std::endl;
                          object->type_ = TxtObject::type::ROOMNUM;
                          // if(line != "B3C-10" && line != "B3c-10"){
                          // }
                      }else if(std::isalpha(second_section[0]) && std::isdigit(second_section[1])){
                          // std::cout << "unique S*" << std::endl;
                          object->type_ = TxtObject::type::ROOMNUM;
                      }else{
                          object->type_ = TxtObject::type::ORDINARY;//!
                          // std::cout << "others" << std::endl;
                      }
                  }else if(second_section.length() == 3){
                      if(std::isdigit(second_section[0]) && std::isdigit(second_section[1]) && std::isalpha(second_section[2])){
                          object->type_ = TxtObject::type::ROOMNUM;//B3a-01a
                      }else if(std::isdigit(second_section[0]) && std::isdigit(second_section[1]) && std::isdigit(second_section[2])){
                          object->type_ = TxtObject::type::ROOMNUM;//B1a-111
                      }else{
                          object->type_ = TxtObject::type::ORDINARY;
                      }
                  }else{
                      if(second_section.substr(0, 3) == "AHU" && std::isdigit(second_section[3])){
                          // std::cout << "AHU*" << std::endl;
                          object->type_ = TxtObject::type::ROOMNUM;
                      }else if(second_section.substr(0, 2) == "ME" && std::isdigit(second_section[2]) && std::isdigit(second_section[3])){
                          object->type_ = TxtObject::type::ROOMNUM;//B1a-ME19
                      }else{
                          // std::cout << "others" << std::endl;
                          object->type_ = TxtObject::type::ORDINARY;
                      }
                  }
              }
            } else {
                object->type_ = TxtObject::type::ORDINARY;//!
                // std::cout << "The string does not follow the pattern." << std::endl;
            }
          }

          if(object->type_ == TxtObject::type::INVALID){
            continue;
          }
          std::vector<std::string> results = split_line(line3, " ");
          object->u1_ = std::stoi(results[0]);
          object->v1_ = std::stoi(results[1]);
          object->u2_ = std::stoi(results[2]);
          object->v2_ = std::stoi(results[3]);
          for(int index = 0; 2 * index + 5 < results.size(); ++index){
            cv::Point p(std::stoi(results[index * 2 + 4]), std::stoi(results[index * 2 + 5]));
            object->polygon_.push_back(p);//todo remove this
          }
          if(std::min(std::min(object->u1_, object->v1_), std::min(object->u2_, object->v2_)) <= 0){
            //noting
            std::cout << "invalid uv" << std::endl;
          }else{
            // std::cout << "valid object type: " << object->type_ << ", content: " << object->content_ << std::endl;
            frame->objects_.push_back(object);
            // std::cout << "push a object into frame" << std::endl;
            object_id++;
          }
        }
      }
}

int gatherPastTextObjects(TxtObject::Ptr pivot_obj, std::vector<std::string>& content, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& poses, std::vector<bool>& unique_labels){
  content.clear();
  poses.clear();
  unique_labels.clear();
  int target_id = -1;
  int frame_id = pivot_obj->frame_id_;
  Frame::Ptr current_frame = txt_manager->getFrame(frame_id);
  Pose3dPtr current_pose = keyframePoses.at(frame_id);
  for(int index = 0; index < current_frame->objects_.size(); ++index){
    TxtObject::Ptr current_object = current_frame->objects_.at(index);
    if(current_object->isValid()){
      content.push_back(current_object->content_);
      poses.push_back(current_object->center_);
      unique_labels.push_back((current_object->type_ == 1));
      if(current_object->frame_id_ == pivot_obj->frame_id_ && current_object->object_id_ == pivot_obj->object_id_){
        target_id = content.size() - 1;
      }
    }
  }
  int index = 1;
  while((frame_id - index) >= 0){
    if((keyframePoses.at(frame_id - index)->p - keyframePoses.at(frame_id)->p).norm() > 35){
      break;// too far, stop tracking
    }
    Frame::Ptr histoy_frame = txt_manager->getFrame(frame_id - index);
    Pose3dPtr history_pose = keyframePoses.at(frame_id - index);
    for(int ob_id = 0; ob_id < histoy_frame->objects_.size(); ++ob_id){
      TxtObject::Ptr history_object = histoy_frame->objects_.at(ob_id);
      if(history_object->isValid()){
        content.push_back(history_object->content_);
        Pose3d delta_pose = current_pose->inverse() * (*history_pose);
        poses.push_back(delta_pose.q.matrix() * history_object->center_ + delta_pose.p);
        unique_labels.push_back((history_object->type_ == 1));
      }
    }
    index++;
  }
  //debug
  for(Eigen::Vector3d& pos : poses){
    pos = current_pose->q.matrix() * pos + current_pose->p;
  }
  return target_id;
}

int gatherSurroundingTextObjects(TxtObject::Ptr pivot_obj, const std::unordered_set<std::string>& candidate_contens, const int& max_index, std::vector<std::string>& content, std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& poses){
  content.clear();
  poses.clear();
  int target_id = -1;
  int frame_id = pivot_obj->frame_id_;
  Frame::Ptr current_frame = txt_manager->getFrame(frame_id);
  Pose3dPtr current_pose = keyframePoses.at(frame_id);
  for(int index = 0; index < current_frame->objects_.size(); ++index){
    TxtObject::Ptr current_object = current_frame->objects_.at(index);
    if(current_object->isValid() && candidate_contens.find(current_object->content_) != candidate_contens.end()){
      content.push_back(current_object->content_);
      poses.push_back(current_object->center_);
      if(current_object->frame_id_ == pivot_obj->frame_id_ && current_object->object_id_ == pivot_obj->object_id_){
        target_id = content.size() - 1;
      }
    }
  }
  int index = 1;
  while((frame_id - index) >= 0){
    if((keyframePoses.at(frame_id - index)->p - keyframePoses.at(frame_id)->p).norm() > 35){
      break;// too far, stop tracking
    }
    Frame::Ptr histoy_frame = txt_manager->getFrame(frame_id - index);
    Pose3dPtr history_pose = keyframePoses.at(frame_id - index);
    for(int ob_id = 0; ob_id < histoy_frame->objects_.size(); ++ob_id){
      TxtObject::Ptr history_object = histoy_frame->objects_.at(ob_id);
      if(history_object->isValid() && candidate_contens.find(history_object->content_) != candidate_contens.end()){
        content.push_back(history_object->content_);
        Pose3d delta_pose = current_pose->inverse() * (*history_pose);
        poses.push_back(delta_pose.q.matrix() * history_object->center_ + delta_pose.p);
      }
    }
    index++;
  }
  index = 1;
  while((frame_id + index) <= max_index){
    if((keyframePoses.at(frame_id + index)->p - keyframePoses.at(frame_id)->p).norm() > 35){
      break;// too far, stop tracking
    }
    Frame::Ptr histoy_frame = txt_manager->getFrame(frame_id + index);
    Pose3dPtr history_pose = keyframePoses.at(frame_id + index);
    for(int ob_id = 0; ob_id < histoy_frame->objects_.size(); ++ob_id){
      TxtObject::Ptr history_object = histoy_frame->objects_.at(ob_id);
      if(history_object->isValid() && candidate_contens.find(history_object->content_) != candidate_contens.end()){
        content.push_back(history_object->content_);
        Pose3d delta_pose = current_pose->inverse() * (*history_pose);
        poses.push_back(delta_pose.q.matrix() * history_object->center_ + delta_pose.p);
      }
    }
    index++;
  }
  //debug
  for(Eigen::Vector3d& pos : poses){
    pos = current_pose->q.matrix() * pos + current_pose->p;
  }
  return target_id;
}

int last_frame_num = -1;
bool loop_need_optimization = false;
std::map<int, int> loops_container;
ImgLidarAligner aligner;

int last_viz_path = -2;
void process_viz_path(void)
{
    float hz = 1.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        // if(recentIdxUpdated > 1) {
        //     pubPath();
        // }
        if(recentIdxUpdated > 1 && (last_viz_path == -2 || recentIdxUpdated - last_viz_path >= 100)) {
            pubPath();
            last_viz_path = recentIdxUpdated;
        }
    }
}

int last_residual_size = 0;
void process_pgo(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(graph_created){
          if(loop_residual_ids.size() + odom_residual_ids.size() - last_residual_size < 100 && !loop_need_optimization){
            continue;
          }
          std::cout << "new add edge size: " << loop_residual_ids.size() + odom_residual_ids.size() - last_residual_size << ", loop_need_optimization: " << loop_need_optimization << std::endl;
          loop_need_optimization = false;//todo thread safety
          mtxPosegraph.lock();
          runCeresOpt();
          std::cout << "running optimization finished" << std::endl;
          mtxPosegraph.unlock();
          last_residual_size = loop_residual_ids.size() + odom_residual_ids.size();
          
        }
    }
}

void pubMap(void)
{
    int SKIP_FRAMES = 50; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();

    mKF.lock();
    Pose3d last_pose; 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
        // if(counter % SKIP_FRAMES == 0) {
        //     *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], *(keyframePosesUpdated[node_idx]));
        // }
        // counter++;
        if(node_idx == 0 || (last_pose.inverse() * (*(keyframePoses.at(node_idx)))).p.norm() > 1){
          last_pose = *(keyframePoses.at(node_idx));
          *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], *(keyframePosesUpdated[node_idx]));
        }
    }
    mKF.unlock(); 

    // downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    // downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

void visualizeLoopClosure()
{
    std::string odometryFrame = "camera_init";

    if (loops_container.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // loop node
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = ros::Time().fromSec(keyframeTimes.at(recentIdxUpdated));
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edge
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = ros::Time().fromSec(keyframeTimes.at(recentIdxUpdated));
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 1.0;
    markerEdge.color.g = 0.0;
    markerEdge.color.b = 1.0;
    markerEdge.color.a = 1;

    // iterate loops
    for (auto it = loops_container.begin(); it != loops_container.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = keyframePoses[key_cur]->p.x();
        p.y = keyframePoses[key_cur]->p.y();
        p.z = keyframePoses[key_cur]->p.z();
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = keyframePoses[key_pre]->p.x();
        p.y = keyframePoses[key_pre]->p.y();
        p.z = keyframePoses[key_pre]->p.z();
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void visualizeTxtObject()
{
    std::string odometryFrame = "camera_init";

    const std::vector<Frame::Ptr, std::allocator<Frame::Ptr>>& frames = txt_manager->frames_;

    if (frames.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = ros::Time().fromSec(keyframeTimes.at(recentIdxUpdated));
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    markerNode.ns = "txt_objects";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.0;
    markerNode.scale.y = 0.0;
    markerNode.scale.z = 0.5;
    markerNode.color.r = 0;
    markerNode.color.g = 0;
    markerNode.color.b = 1;
    markerNode.color.a = 1;

    for(int frame_id = 0; frame_id < frames.size(); ++frame_id){
      const Frame& frame = *(frames.at(frame_id));
      geometry_msgs::Pose pose;
      pose.position.x = keyframePosesUpdated.at(frame_id)->p.x();
      pose.position.y = keyframePosesUpdated.at(frame_id)->p.y();
      pose.position.z = keyframePosesUpdated.at(frame_id)->p.z();
      markerNode.pose = pose;
      for(int txt_id = 0; txt_id < frame.objects_.size(); ++txt_id){
        if(frame.objects_.at(txt_id)->type_ == TxtObject::type::INVALID) continue;
        markerNode.text = frame.objects_.at(txt_id)->content_ + "@" + std::to_string(frame_id);
        // Eigen::Vector3d text_position = keyframePosesUpdated.at(frame_id)->q.matrix() * frame.objects_.at(txt_id)->center_ + keyframePosesUpdated.at(frame_id)->p; 
        // markerNode.pose.position.x = text_position.x();
        // markerNode.pose.position.y = text_position.y();
        // markerNode.pose.position.z = text_position.z();
        markerArray.markers.push_back(markerNode);
        markerNode.id = markerNode.id + 1;
        markerNode.pose.position.z = markerNode.pose.position.z + 0.3;
      }
    }
    pubTxtObjects.publish(markerArray);
}

int last_vis_map = -2;
void process_viz_map(void)
{
    float vizmapFrequency = 1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {//! if bag is stopped, donnot republish
            pubMap();//todo control frequency
        // }
        // if(recentIdxUpdated > 1 && (last_vis_map == -2 || recentIdxUpdated - last_vis_map >= 100)){//every 100 key frames
          // pubMap();
          visualizeLoopClosure();
          visualizeTxtObject();
          last_vis_map = recentIdxUpdated;
        }
    }
} // pointcloud_viz

ros::ServiceServer srvSavePose, srvSaveMap;
bool savePoseService(text_lcd::save_poseRequest& req, text_lcd::save_poseResponse& res){
    saveOptimizedVerticesKITTIformat(pgKITTIformat); // pose//todo
    saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
    std::cout << "poses has been saved" << std::endl;
    return true;
}

bool saveMapService(text_lcd::save_mapRequest& req, text_lcd::save_mapResponse& res){
  std::string base_path;
  if(!req.destination.empty()){
    base_path = req.destination;
  }else{
    base_path = save_directory;
  }
  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>);
  pcl::VoxelGrid<PointType> downSizeFilterMap;
  downSizeFilterMap.setLeafSize(0.2, 0.2, 0.2);
  // downSizeFilterMap.setInputCloud(keyframeLaserClouds.front());// this frame is set fixed
  // downSizeFilterMap.filter(*map_cloud);
  *map_cloud = *(keyframeLaserClouds.front());
  int current_num = keyframePosesUpdated.size();
  double last_distance = 0;
  for(int index = 1; index < current_num; ++index){
    if(odometry_distance.at(index) - last_distance < 2){
      continue;
    }
    pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>);
    // downSizeFilterMap.setInputCloud(keyframeLaserClouds.at(index));
    // downSizeFilterMap.filter(*tmp_cloud);
    *tmp_cloud = *(keyframeLaserClouds.at(index));
    if(req.optimized == 0){
      *map_cloud += *(local2global(tmp_cloud, *(keyframePoses.at(index))));
    }else{
      *map_cloud += *(local2global(tmp_cloud, *(keyframePosesUpdated.at(index))));
    }
  }
  // downSizeFilterMap.setInputCloud(map_cloud);
  // downSizeFilterMap.filter(*map_cloud);

  int ret;
  if(req.optimized == 0){
   ret = pcl::io::savePCDFileBinary(base_path + "/map_odo.pcd", *map_cloud);
  }else{
   ret = pcl::io::savePCDFileBinary(base_path + "/map_opt.pcd", *map_cloud);
  }
  res.success = (ret == 0);
  return true;
}

void OCRCallback(const std_msgs::String::ConstPtr& ocr_string){
    if(ocr_string->data.empty()) return;
    txt_buffer_lock.lock();
    txtBuf.push(std::make_shared<std::string>(std::string(ocr_string->data)));
    // std::cout << "++++++++++" << std::endl;
    // std::cout << "push " << ocr_string->data << std::endl;
    // std::cout << "After push size: " << txtBuf.size() << std::endl;
    // std::cout << "++++++++++" << std::endl;  
    txt_buffer_lock.unlock();
}

int initialize_count = 0, retrieval_count = 0;
double initialize_time = 0.0, retrieval_time = 0.0;

// 定义全局数组来存储相机内参和外参
float k[9];  // 相机内参矩阵 (3x3)
float r[9];  // 旋转矩阵 (3x3)
float t[3];  // 平移向量 (3x1)
float d[4];
int image_size[2]; // 图像宽高 [width, height]
// 从 YAML 文件中解析并赋值到数组中
void loadCameraParams(const std::string &yaml_file_path) {
    std::cout << "Yaml file path: " << yaml_file_path << std::endl;
    try {
        YAML::Node config = YAML::LoadFile(yaml_file_path);

        // 读取图像尺寸
        image_size[0] = config["image_size"]["width"].as<int>();
        image_size[1] = config["image_size"]["height"].as<int>();

        // 读取相机内参矩阵 k
        YAML::Node k_values = config["intrinsics"]["k"];
        for (int i = 0; i < 9; ++i) {
            k[i] = k_values[i].as<float>();
        }

        // 读取相机内参矩阵 k
        YAML::Node d_values = config["distortion"]["d"];
        for (int i = 0; i < 4; ++i) {
            d[i] = d_values[i].as<float>();
        }

        // 读取外参旋转矩阵 r
        YAML::Node r_values = config["extrinsics"]["rotation"];
        for (int i = 0; i < 9; ++i) {
            r[i] = r_values[i].as<float>();
        }

        // 读取外参平移向量 t
        YAML::Node t_values = config["extrinsics"]["translation"];
        for (int i = 0; i < 3; ++i) {
            t[i] = t_values[i].as<float>();
        }

        // 输出检查加载的值
        ROS_INFO("Image size: %dx%d", image_size[0], image_size[1]);
        ROS_INFO("Camera intrinsics (k):");
        for (int i = 0; i < 9; ++i) {
            std::cout << k[i] << " ";
            if (i % 3 == 2) std::cout << std::endl;
        }
        ROS_INFO("Camera distortion (d):");
        for (int i = 0; i < 4; ++i) {
            std::cout << k[i] << " ";
        }
        std::cout << std::endl;
        ROS_INFO("Rotation matrix (r):");
        for (int i = 0; i < 9; ++i) {
            std::cout << r[i] << " ";
            if (i % 3 == 2) std::cout << std::endl;
        }
        ROS_INFO("Translation vector (t):");
        for (int i = 0; i < 3; ++i) {
            std::cout << t[i] << " ";
        }
        std::cout << std::endl;

    } catch (const YAML::Exception &e) {
        ROS_ERROR("Failed to load YAML file: %s", e.what());
    }
}


void ocr_process(){
    // std::ofstream of(save_directory + "/lc.txt", std::ios::out);
    // std::ofstream of2(save_directory + "/relative pose.txt", std::ios::out);
    while(1){
        while(!txtBuf.empty()){
            txt_buffer_lock.lock();
            std::string ocr_result = *txtBuf.front();
            txtBuf.pop();
            txt_buffer_lock.unlock();

            std::stringstream ss(ocr_result);
            std::string line;
            getline(ss, line);
            // std::cout << "line: " << line << std::endl;
            double timestamp = std::stod(line);
            // std::cout << "timestamp: " << std::fixed << std::setprecision(12) << timestamp << std::endl;
            auto ite = std::upper_bound(keyframeTimes.begin(), keyframeTimes.end(), timestamp);
            if(ite == keyframeTimes.end() || ite == keyframeTimes.begin()){
              std::cout << "no later odo poses for interpolation, skip, keyframeTimes size: " << keyframeTimes.size() << std::endl;;
              continue;
            }
            int anchor_frame_id = ite - keyframeTimes.begin() - 1;
            // std::cout << "anchor_frame_id: " << anchor_frame_id << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            Pose3d delta_pose = keyframePoses.at(anchor_frame_id)->inverse() * (*(keyframePoses.at(anchor_frame_id + 1)));
            double anchor_delta_time = timestamp - keyframeTimes.at(anchor_frame_id);
            if(anchor_delta_time < 0 || anchor_delta_time > 0.2){
              std::cout << "anchor_delta_time is wrong: " << anchor_delta_time << std::endl;
              continue;
            }
            double anchor_time_ratio = (anchor_delta_time / (keyframeTimes.at(anchor_frame_id + 1) - keyframeTimes.at(anchor_frame_id)));
            if(anchor_time_ratio < 0 || anchor_time_ratio > 1){
              std::cerr << "erro anchor_time_ratio: " << anchor_time_ratio << std::endl;
            }
            Eigen::Vector3d anchor_delta_position = delta_pose.p * anchor_time_ratio;
            Eigen::Quaterniond anchor_delta_rot = Eigen::Quaterniond::Identity().slerp(anchor_time_ratio, delta_pose.q);

            //! may skip frames, as we only process the latest frame 
            Frame::Ptr current_frame = txt_manager->getFrame(anchor_frame_id);
            parseOCRString(ocr_result, current_frame);
            if(current_frame->objects_.empty()){
              continue;
            }
            // std::cout << "After parse ocr string" << std::endl;
            // compute SE3 pose
            pcl::PointCloud<PointType>::Ptr local_cloud(new pcl::PointCloud<PointType>);      
            for(int index = 0; index < 10; ++index){
              if(anchor_frame_id - index < 0){continue;}
              Pose3d relative_pose = diffTransformation(*(keyframePosesUpdated.at(anchor_frame_id)), *(keyframePosesUpdated.at(anchor_frame_id - index)));
              pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>);
              pcl::transformPointCloud<PointType>(*(keyframeLaserClouds.at(anchor_frame_id - index)), *tmp_cloud, relative_pose.p.cast<float>(), relative_pose.q.cast<float>());
              *local_cloud += *tmp_cloud;
            }
            pcl::transformPointCloud<PointType>(*local_cloud, *local_cloud, -anchor_delta_rot.inverse().matrix().cast<float>() * anchor_delta_position.cast<float>(), anchor_delta_rot.inverse().cast<float>());//! current lidar frame which have same timestamp with camera

            // cv::Mat tmp_mat = cv::Mat::zeros(630, 1920, CV_32FC1);//todo change also in ocr

            // float k[] = {616.625219502153, 0.0, 953.517490713636, 
            //           0.0, 616.2007856249934, 498.49245496387283,
            //           0.0, 0.0, 1.0};
            // float r[] = {0.00523589, -0.999986, 2.74153e-05,
            //         -0.0209421, -0.000137062, -0.999781,
            //         0.999767, 0.00523417, -0.0209425};
            // float t[] = {0.0, -0.025, -0.12};

            cv::Mat tmp_mat = cv::Mat::zeros(image_size[1], image_size[0], CV_32FC1);

            aligner.reset();
            aligner.load(tmp_mat, local_cloud);
            pcl::PointCloud<pcl::PointXYZI> tmp_cloud;
            for(const TxtObject::Ptr& obj : current_frame->objects_){
                std::cout << obj->content_ << std::endl;
                pcl::PointCloud<pcl::PointXYZI> cloud;
                // std::cout << "u: " << obj->u1_ << ", v: " << obj->v1_ << std::endl;

                pcl::PointCloud<pcl::PointXYZI> ttt_cloud;
                pcl::PointCloud<pcl::PointXYZI> expand_cloud;
                cv::Mat img;
                aligner.getCloudInPolygon(obj->polygon_, ttt_cloud, expand_cloud, img);
                std::cout << "clouds in mask: " << ttt_cloud.size() << std::endl;
                if(ttt_cloud.size() < 10){
                  obj->type_ = TxtObject::type::INVALID;
                  continue;
                }

                pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(ttt_cloud.makeShared()));
                pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(plane);
                ransac.setDistanceThreshold(0.01);
                ransac.setMaxIterations(200);
                ransac.setProbability(0.99);
                ransac.computeModel();
                std::vector<int> inliers; 
                ransac.getInliers(inliers);
                std::cout << "inlier size: " << inliers.size() << std::endl;
                if(inliers.size() < 10){
                  obj->type_ = TxtObject::type::INVALID;
                  continue;
                }
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::copyPointCloud<pcl::PointXYZI>(ttt_cloud, inliers, *cloud_plane);//! small cluster

                Eigen::Vector3f center = Eigen::Vector3f::Zero();
                Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
                for(const pcl::PointXYZI p : cloud_plane->points){
                    Eigen::Vector3f v(p.x, p.y, p.z);
                    center += v;
                    covariance += v * v.transpose();
                }
                center /= cloud_plane->size();
                // std::cout << "center in lidar frame: " << center << std::endl;
                covariance = covariance / cloud_plane->size() - center * center.transpose();
                Eigen::EigenSolver<Eigen::Matrix3f> solver(covariance);
                Eigen::Vector3f eigen_values = solver.eigenvalues().real();
                Eigen::Matrix3f eigen_vectors = solver.eigenvectors().real();
                Eigen::Vector3f::Index min_index, max_index;
                eigen_values.minCoeff(&min_index);
                // eigen_values.maxCoeff(&max_index);
                // int mid_index = 3 - max_index - min_index;
                // if(eigen_values[min_index] < std::min<float>(eigen_values(mid_index) * min_ratio_, min_eigen_value_)){
                Eigen::Vector3f norm(eigen_vectors(0, min_index), eigen_vectors(1, min_index), eigen_vectors(2, min_index));//todo: do more to check
                norm.normalize();
                if(center.dot(norm) < 0){
                    norm = -norm;
                }
                // std::cout << "tmp Norm: " << norm << std::endl;
                float dis_lidar = -norm.dot(center);

                // refine by expanded cloud//todo check the fov of cloud
                pcl::PointCloud<pcl::PointXYZI> inlier_cloud;
                for(int p_id = 0; p_id < expand_cloud.size(); ++p_id){
                  Eigen::Vector3f current_point(expand_cloud.points[p_id].x, expand_cloud.points[p_id].y, expand_cloud.points[p_id].z);
                  if(fabs(current_point.dot(norm) + dis_lidar) < 0.02){
                    inlier_cloud.push_back(expand_cloud.points[p_id]);
                  }
                }
                // std::cout << "inlier cloud size: " << inlier_cloud.size() << std::endl;
                center.setZero();
                covariance.setZero();
                for(const pcl::PointXYZI p : inlier_cloud.points){
                    Eigen::Vector3f v(p.x, p.y, p.z);
                    center += v;
                    covariance += v * v.transpose();
                }
                center /= inlier_cloud.size();
                // std::cout << "expanded center in lidar frame: " << center << std::endl;
                covariance = covariance / inlier_cloud.size() - center * center.transpose();
                Eigen::EigenSolver<Eigen::Matrix3f> solver2(covariance);
                eigen_values = solver2.eigenvalues().real();
                eigen_vectors = solver2.eigenvectors().real();
                eigen_values.minCoeff(&min_index);
                norm = Eigen::Vector3f(eigen_vectors(0, min_index), eigen_vectors(1, min_index), eigen_vectors(2, min_index));//todo: do more to check
                norm.normalize();//! lidar frame
                if(center.dot(norm) < 0){
                    norm = -norm;
                }
                // std::cout << "Final Norm: " << norm << std::endl;

                // norm.dot(center)+d=0 => d
                // K.inverse() * [u, v, 1] = [X, Y, 1]
                // s[X, Y, 1].dot(norm) + d = 0 => s = -d/([XY1].dot(norm))
                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> extri_r(r);//! lidar->camera
                Eigen::Matrix<float, 3, 1> extri_t(t);
                Eigen::Vector3f norm_cam = extri_r * norm;//! camera frame
                Eigen::Vector3f center_cam = extri_r * center + extri_t;
                float dis = -norm_cam.dot(center_cam);
                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> k_m(k);
                // std::cout << "k: " << k_m << std::endl;
                Eigen::Vector3f position_norm1 = k_m.inverse() * Eigen::Vector3f(obj->u1_, obj->v1_, 1);
                float depth1 = -dis / (position_norm1.dot(norm_cam));
                Eigen::Vector3f left_point = position_norm1 * depth1;
                Eigen::Vector3f position_norm2 = k_m.inverse() * Eigen::Vector3f(obj->u2_, obj->v2_, 1);
                float depth2 = -dis / (position_norm2.dot(norm_cam));
                Eigen::Vector3f right_point = position_norm2 * depth2;//! both are in camera frame
                if(depth1 < 0 || depth2 < 0){
                  std::cerr << "depth is negative!" << std::endl;
                }

                left_point = extri_r.inverse() * (left_point - extri_t);//! lidar frame
                right_point = extri_r.inverse() * (right_point - extri_t);

                //! all of the results are under current lidar frame (when the image is captured)
                //! transform to anchored lidar frame
                
                obj->center_ = anchor_delta_rot.matrix() * left_point.cast<double>() + anchor_delta_position;
                obj->norm_ = anchor_delta_rot.matrix() * norm.cast<double>();
                obj->main_direction_ = anchor_delta_rot.matrix() * (right_point - left_point).normalized().cast<double>();
            }
      
            // repetitive observations of the same text entities are set invalid
            for(const TxtObject::Ptr& obj : current_frame->objects_){
                const std::string& content = obj->content_;
                if(txt_manager->object_sets_.find(content) != txt_manager->object_sets_.end()){
                    const std::map<int, int>& text_set = txt_manager->object_sets_[content]->objects_;
                    for(auto it = text_set.rbegin(); it != text_set.rend(); ++it){
                        if(fabs(odometry_distance.at(it->first) - odometry_distance.at(anchor_frame_id)) > 5){// 里程计距离超过5m，不再继续回溯
                          break;
                        }
                        if(txt_manager->frames_.at(it->first)->objects_.at(it->second)->isValid()){
                            Eigen::Vector3d last_position = keyframePoses.at(it->first)->q.matrix() * txt_manager->frames_.at(it->first)->objects_.at(it->second)->center_ + keyframePoses.at(it->first)->p;
                            Eigen::Vector3d current_position = keyframePoses.at(anchor_frame_id)->q.matrix() * obj->center_ + keyframePoses.at(anchor_frame_id)->p;
                            // if(current_frame_num - 1 == 2651 && it->first == 2641 && content == "EXIT"){
                            //   std::cout << "last_position: " << last_position << std::endl;
                            //   std::cout << "current_position: " << current_position << std::endl;
                            //   std::cout << "last_center: " << txt_manager->frames_.at(it->first)->objects_.at(it->second)->center_ << std::endl;
                            //   std::cout << "current_center: " << obj->center_ << std::endl;
                            // }
                            Eigen::Vector3d last_norm = keyframePoses.at(it->first)->q.matrix() * txt_manager->frames_.at(it->first)->objects_.at(it->second)->norm_;
                            Eigen::Vector3d current_norm = keyframePoses.at(anchor_frame_id)->q.matrix() * obj->norm_;
                            double cos_angle = last_norm.dot(current_norm)/(last_norm.norm() * current_norm.norm());
                            if((last_position - current_position).norm() < 0.2 && cos_angle > cos(M_PI_4)){
                                obj->type_ = TxtObject::type::INVALID;
                            }
                        } 
                    }
              }
          }
      
            //log
            auto initialize_end_time = std::chrono::high_resolution_clock::now();
            auto duration_initialize = std::chrono::duration_cast<std::chrono::milliseconds>(initialize_end_time - start_time);
            initialize_time += duration_initialize.count();
            initialize_count++;
            std::cout << "===Mean initial time: " << initialize_time/double(initialize_count) << std::endl;


            //log
            auto retrieval_start_time = std::chrono::high_resolution_clock::now();

            if(current_frame->objects_.empty()){
              std::cout << "empty objects, continue" << std::endl;

              //log
              auto retrieval_end_time = std::chrono::high_resolution_clock::now();
              retrieval_time += std::chrono::duration_cast<std::chrono::milliseconds>(retrieval_end_time - retrieval_end_time).count();
              retrieval_count++;
              std::cout << "===Mean time for retrieval: " << retrieval_time / double(retrieval_count);

              continue;//! jin: from callback to new thread, change return to continue;
            }else{
              bool valid_txt_exist = false;
              for(const TxtObject::Ptr& obj : current_frame->objects_){
                  if(obj->type_ != TxtObject::type::INVALID){
                      valid_txt_exist = true;
                      break;
                  }
              }
              if(!valid_txt_exist){
                std::cout << "No valid txt exists!!!" << std::endl;
              }else{
                  txt_manager->refreshFrame(anchor_frame_id);
                  std::cout << "After refresh" << std::endl;
                  //! retrieve
                  bool loop_detected = false;
                  txt_manager_lock_.lock();//! may use pose of frame which is being changed in main thread in the future
                  //log only
                  int last_loop_index = -1;
                  for(const TxtObject::Ptr& obj : current_frame->objects_){
                      if(!(obj->isValid())){
                          continue;
                      }
                      std::vector<TxtObject::Ptr> loop;
                      txt_manager->retrieveObject(obj, loop);// find objects with the same txt, but double check by distance is not performed
                      if(!loop.empty()){
                        std::cout << "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]" << std::endl;
                        for(const auto& o : loop){
                          std::cout << o->frame_id_ << ", ";
                        }
                        std::cout << std::endl;
                        // loop_pair_mutex_.lock();
                        for(int index = 0; index < loop.size(); ++index){
                            std::cout << "loop index: " << index << std::endl;
                            const TxtObject::Ptr& loop_object = loop.at(index);
                            if(!(loop_object->isValid())){
                              continue;
                            }
                            if(odometry_distance.at(obj->frame_id_) - odometry_distance.at(loop_object->frame_id_) < 10){//todo
                              continue;//not make loop if odometry distance is too near, actually this situation could be avoided by redundant information removal before
                            }

                            Eigen::Matrix3d rotation;
                            rotation << loop_object->main_direction_, loop_object->norm_.cross(loop_object->main_direction_), loop_object->norm_;
                            Pose3d pre_observation;
                            pre_observation.q = Eigen::Quaterniond(rotation);
                            pre_observation.p = loop_object->center_;

                            rotation << obj->main_direction_, obj->norm_.cross(obj->main_direction_), obj->norm_;
                            Pose3d current_observation;
                            current_observation.q = Eigen::Quaterniond(rotation);
                            current_observation.p = obj->center_;

                            visualization_msgs::MarkerArray markerArray;
                            markerArray.markers.clear();
                            if(obj->type_ != TxtObject::type::ROOMNUM){
                                //todo judge by clipper
                                std::vector<std::string> contents;
                                std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> poses;
                                std::vector<bool> unique_labels;
                                int current_index = gatherPastTextObjects(obj, contents, poses, unique_labels);

                                std::unordered_set<std::string> candidate_content(contents.begin(), contents.end());
                                std::vector<std::string> history_contents;
                                std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> history_poses;
                                int history_index = gatherSurroundingTextObjects(loop_object, candidate_content, obj->frame_id_, history_contents, history_poses);
                      
                                clipper::invariants::EuclideanDistance::Params iparams;
                                iparams.epsilon = 0.1;
                                iparams.sigma = 0.2;
                                std::cout << "epsilon: " << iparams.epsilon << std::endl;
                                std::cout << "mindist: " << iparams.mindist << std::endl;
                                std::cout << "sigma: " << iparams.sigma << std::endl;
                                clipper::invariants::EuclideanDistancePtr invariant =
                                          std::make_shared<clipper::invariants::EuclideanDistance>(iparams);

                                clipper::Params params;
                                // params.
                                clipper::CLIPPER clipper(invariant, params);
                                Eigen::MatrixXd model(3, contents.size());
                                for(int tmp_index = 0; tmp_index < contents.size(); ++tmp_index){
                                  model.col(tmp_index) = poses.at(tmp_index);
                                }
                                Eigen::MatrixXd data(3, history_contents.size());
                                for(int tmp_index = 0; tmp_index < history_contents.size(); ++tmp_index){
                                  data.col(tmp_index) = history_poses.at(tmp_index);
                                }
                                // std::cout << "model: " << model << std::endl;
                                // std::cout << "data: " << data << std::endl;
                                std::vector<std::pair<int, int>> pairs;
                                pairs.push_back(std::make_pair(current_index, history_index));
                                for(int current_id = 0; current_id < contents.size(); ++current_id){
                                  if(current_id == current_index) continue;//once for current entity
                                  for(int history_id = 0; history_id < history_contents.size(); ++history_id){
                                    if(history_id == history_index) continue;//once for current entity
                                    if(contents.at(current_id) != history_contents.at(history_id)) continue;
                                    pairs.push_back(std::make_pair(current_id, history_id));
                                  }
                                }//
                                if(pairs.size() < 3){
                                  std::cout << "pair size is small" << std::endl;
                                  continue;
                                }
                                Eigen::MatrixXi association(pairs.size(), 2);
                                for(int tmp_index = 0; tmp_index < pairs.size(); ++tmp_index){
                                  association(tmp_index, 0) = pairs.at(tmp_index).first;
                                  association(tmp_index, 1) = pairs.at(tmp_index).second;
                                }
                                clipper.scorePairwiseConsistency(model, data, association);
                                const clipper::Affinity& M = clipper.getAffinityMatrix();
                                // std::cout << "M: " << M << std::endl;
                                const clipper::Association& initial_association = clipper.getInitialAssociations();
                                // std::cout << "initial association: " << initial_association << std::endl;
                                clipper.solve();
                                const clipper::Association& inliers = clipper.getSelectedAssociations();
                      
                                // std::cout << "Ainliers: " << inliers << std::endl;
                                const clipper::Solution& sol = clipper.getSolution();
                                std::cout << "inlier size: " << inliers.rows() << ", node size: " << sol.nodes.size() << std::endl;
                                if(sol.nodes.size() < 3){
                                  std::cout << "Result association size is too small: " << sol.nodes.size() << std::endl;
                                  continue;
                                }
                      
                                //check if all inliers are from one notice
                                std::cout << "inliers: " << std::endl;
                                Eigen::Vector3d current_text_position = model.col(current_index);
                                double max_distance = -1;
                                for(int tmp_id = 0; tmp_id < inliers.rows(); ++tmp_id){
                                  Eigen::Vector3d neighbor_text_position = model.col(inliers(tmp_id, 0));
                                  double distance = (neighbor_text_position - current_text_position).norm();
                                  if(distance > max_distance){
                                    max_distance = distance;
                                  }
                                  std::cout << contents.at(inliers(tmp_id, 0)) << ", ";
                                }
                                std::cout << std::endl;
                                std::cout << "Max distance: " << max_distance << std::endl;
                                if(max_distance < 4){
                                  std::cout << "Inliers are too close to each other, skip" << std::endl;
                                  continue;
                                }

                                // //at least one unique txt should be inlier
                                // bool has_unique = false;
                                // for(int tmp_id = 0; tmp_id < inliers.rows(); ++tmp_id){
                                //   has_unique = has_unique || unique_labels.at(inliers(tmp_id, 0));
                                // }
                                // if(!has_unique){
                                //   std::cout << "No unique text, skip" << std::endl;
                                //   continue;
                                // }

                                bool included = false;
                                for(int index = 0; index < sol.nodes.size(); ++index){
                                  if(sol.nodes.at(index) == 0){
                                    included = true;
                                    break;
                                  }
                                }
                                if(!included){
                                  std::cout << "candidate associated is not included!" << std::endl;
                                  continue;
                                }

                                {
                                  std::string odometryFrame = "camera_init";
                                  // visualization_msgs::MarkerArray markerArray;
                                  // 闭环边
                                  visualization_msgs::Marker markerEdge;
                                  markerEdge.header.frame_id = odometryFrame;
                                  markerEdge.header.stamp = ros::Time().fromSec(keyframeTimes.at(recentIdxUpdated));
                                  markerEdge.action = visualization_msgs::Marker::ADD;
                                  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
                                  markerEdge.ns = "loop_edges";
                                  markerEdge.id = 1;
                                  markerEdge.pose.orientation.w = 1;
                                  markerEdge.scale.x = 0.1;
                                  markerEdge.color.r = 1.0;
                                  markerEdge.color.g = 0.0;
                                  markerEdge.color.b = 0.0;
                                  markerEdge.color.a = 1;
                                  for(int tmp_index = 0; tmp_index < poses.size(); ++tmp_index){
                                    if(tmp_index == current_index) continue;
                                    geometry_msgs::Point p;
                                    p.x = poses.at(current_index).x();
                                    p.y = poses.at(current_index).y();
                                    p.z = poses.at(current_index).z();
                                    markerEdge.points.push_back(p);
                                    p.x = poses.at(tmp_index).x();
                                    p.y = poses.at(tmp_index).y();
                                    p.z = poses.at(tmp_index).z();
                                    markerEdge.points.push_back(p);
                                  }
                                  markerArray.markers.push_back(markerEdge);
                                  markerEdge.id = 2;
                                  markerEdge.color.r = 0.0;
                                  markerEdge.color.g = 0.0;
                                  markerEdge.color.b = 1.0;
                                  markerEdge.color.a = 1;
                                  markerEdge.points.clear();
                                  for(int tmp_index = 0; tmp_index < history_poses.size(); ++tmp_index){
                                    if(tmp_index == history_index) continue;
                                    geometry_msgs::Point p;
                                    p.x = history_poses.at(history_index).x();
                                    p.y = history_poses.at(history_index).y();
                                    p.z = history_poses.at(history_index).z();
                                    markerEdge.points.push_back(p);
                                    p.x = history_poses.at(tmp_index).x();
                                    p.y = history_poses.at(tmp_index).y();
                                    p.z = history_poses.at(tmp_index).z();
                                    markerEdge.points.push_back(p);
                                  }
                                  // markerArray.markers.push_back(markerNode);
                                  markerArray.markers.push_back(markerEdge);
                                  // pubGraph.publish(markerArray);
                                }

                                std::cout << "==================Right graph by clipper: " << contents.at(current_index) << "!!!" << std::endl;
                            }
                    
                            Pose3d relative_pose = pre_observation * current_observation.inverse();

                            Eigen::Matrix4f refine_relative_pose = relative_pose.matrix().cast<float>();
                            if(!doICPVirtualRelative(loop_object->frame_id_, obj->frame_id_, refine_relative_pose)){//refinement by ICP
                              std::cout << "skip this loop" << std::endl;
                              continue;
                            }

                            // {
                            //   Eigen::Matrix4d odometry_relative_pose = ((keyframePoses.at(loop_object->frame_id_)->inverse()) * (*keyframePoses.at(obj->frame_id_))).matrix();
                            //   Eigen::Matrix4d initial_relative_pose = relative_pose.matrix();
                            //   of2 << std::fixed << std::setprecision(6) << 
                            //         keyframeTimes.at(obj->frame_id_) << " " << keyframeTimes.at(loop_object->frame_id_)  << " " << 
                            //         std::fixed << std::setprecision(3) << 
                            //         odometry_relative_pose(0, 0) << " " << odometry_relative_pose(0, 1) << " " << odometry_relative_pose(0, 2) << " " << odometry_relative_pose(0, 3) << " " << 
                            //         odometry_relative_pose(1, 0) << " " << odometry_relative_pose(1, 1) << " " << odometry_relative_pose(1, 2) << " " << odometry_relative_pose(1, 3) << " " << 
                            //         odometry_relative_pose(2, 0) << " " << odometry_relative_pose(2, 1) << " " << odometry_relative_pose(2, 2) << " " << odometry_relative_pose(2, 3) << " " << 
                            //         initial_relative_pose(0, 0) << " " << initial_relative_pose(0, 1) << " " << initial_relative_pose(0, 2) << " " << initial_relative_pose(0, 3) << " " << 
                            //         initial_relative_pose(1, 0) << " " << initial_relative_pose(1, 1) << " " << initial_relative_pose(1, 2) << " " << initial_relative_pose(1, 3) << " " << 
                            //         initial_relative_pose(2, 0) << " " << initial_relative_pose(2, 1) << " " << initial_relative_pose(2, 2) << " " << initial_relative_pose(2, 3) << " " << 
                            //         refine_relative_pose(0, 0) << " " << refine_relative_pose(0, 1) << " " << refine_relative_pose(0, 2) << " " << refine_relative_pose(0, 3) << " " << 
                            //         refine_relative_pose(1, 0) << " " << refine_relative_pose(1, 1) << " " << refine_relative_pose(1, 2) << " " << refine_relative_pose(1, 3) << " " << 
                            //         refine_relative_pose(2, 0) << " " << refine_relative_pose(2, 1) << " " << refine_relative_pose(2, 2) << " " << refine_relative_pose(2, 3) << " " << 
                            //         std::endl;
                            // }

                            if(!markerArray.markers.empty()){
                              pubGraph.publish(markerArray);
                            }

                            std::cout << "||+++++++++++++++++++++loop between frame " << loop.at(index)->frame_id_ << " vs " << obj->frame_id_ << ", type: " << obj->type_ << ", content: " << obj->content_ << "++++++++++++++++++++++||" << std::endl;

                            relative_pose.q = Eigen::Quaternionf(refine_relative_pose.topLeftCorner<3, 3>()).cast<double>();
                            relative_pose.p = Eigen::Vector3f(refine_relative_pose.topRightCorner<3, 1>()).cast<double>();

                            Eigen::Matrix<double, 6, 6> sqrt_information;
                            sqrt_information << 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 1000.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 1000.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 1000.0;

                            ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(relative_pose, sqrt_information);
                            ceres::LossFunction* loss_function = NULL;//todo
                            ceres::LocalParameterization* quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
                            mtxPosegraph.lock();
                            ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function,
                                            keyframePosesUpdated.at(loop.at(index)->frame_id_)->p.data(),
                                            keyframePosesUpdated.at(loop.at(index)->frame_id_)->q.coeffs().data(),
                                            keyframePosesUpdated.at(obj->frame_id_)->p.data(),
                                            keyframePosesUpdated.at(obj->frame_id_)->q.coeffs().data()
                                            );
                            mtxPosegraph.unlock();
                            loop_residual_ids.push_back(id);

                            loop_detected = true;
                            loops_container.insert(std::make_pair(obj->frame_id_, loop.at(index)->frame_id_));
                            last_loop_index = loop.at(index)->frame_id_;//log only
                        }
                        std::cout << "All loop is inspected" << std::endl;
                    }
                  }
                  txt_manager_lock_.unlock();
                  loop_need_optimization = loop_detected;
                  // {
                  //     double current_timestamp = keyframeTimes.at(anchor_frame_id);
                  //     if(last_loop_index != -1){
                  //         of << std::fixed << std::setprecision(6) << current_timestamp << " " << keyframeTimes.at(last_loop_index) << std::endl;
                  //     }else{
                  //         of << std::fixed << std::setprecision(6) << current_timestamp << " " << last_loop_index << std::endl;
                  //     }
                  // } 
              }//end of have valid txt
            }
            auto retrieval_end_time = std::chrono::high_resolution_clock::now();
            retrieval_time += std::chrono::duration_cast<std::chrono::milliseconds>(retrieval_end_time - retrieval_start_time).count();
            retrieval_count++;
            std::cout << "===Mean time for retrieval: " << retrieval_time / double(retrieval_count) << std::endl;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Loop closure time cost: " << duration.count() << " ms." << std::endl;
        }//end of one process
        std::chrono::milliseconds dura(30);
        std::this_thread::sleep_for(dura);
    }  
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserPGO");
    ros::NodeHandle nh;

    std::string yaml_file;
    nh.getParam("yaml_file", yaml_file);

    loadCameraParams(yaml_file);

    nh.param<std::string>("save_directory", save_directory, "/"); 
    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";
    // pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    // pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    // pgScansDirectory = save_directory + "Scans/";
    // auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    // unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

    srvSavePose  = nh.advertiseService("/save_pose" ,  &savePoseService);
    srvSaveMap  = nh.advertiseService("/save_map" ,  &saveMapService);

    ceres::Problem::Options option;// todo: configure params
    problem = ceres::Problem(option);

    odom_pose_curr.reset(new Pose3d);
    odom_pose_prev.reset(new Pose3d);
    odom_pose_drift.reset(new Pose3d);

    float filter_size = 0.1; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);// todo
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    double mapVizFilterSize;
    nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_lidar", 100, laserCloudFullResHandler);//! 原始数据放在队列里，等待pg线程读取,imu坐标系
    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/Odometry", 100, laserOdometryHandler);
  	ros::Subscriber OCRSub = nh.subscribe<std_msgs::String>("/ocr_string", 100, OCRCallback);


    pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
    pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
    pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
    pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
    // pubPlane = nh.advertise<sensor_msgs::PointCloud2>("/initial_plane", 100);

    pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
    pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    //!jin

    cv::Mat K(3, 3, CV_32FC1, k);
    cv::Mat D(4, 1, CV_32FC1, d);
    cv::Mat R(3, 3, CV_32FC1, r);
    cv::Mat T(3, 1, CV_32FC1, t);
    aligner.initialize(K, D, R, T);

    txt_manager.reset(new TxtManager());
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);
    pubTxtObjects = nh.advertise<visualization_msgs::MarkerArray>("/txt_objects", 1);
    pubGraph = nh.advertise<visualization_msgs::MarkerArray>("/text_object_graph", 1);

	  std::thread process_odometry_thread {process_odometry};
    std::thread ocr_process_thread {ocr_process};
    std::thread process_pgo_thread {process_pgo};
    std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
    std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

    ros::spin();

    return 0;
}
