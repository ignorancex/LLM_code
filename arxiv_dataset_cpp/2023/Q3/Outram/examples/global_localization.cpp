/*****************************************************************
 *
 * Copyright (c) 2023, Nanyang Technological University, Singapore
 *
 * Authors: Pengyu Yin
 * Contact: pengyu001@e.ntu.edu.sg
 *
 ****************************************************************/

#include <eigen_conversions/eigen_msg.h>
#include <nano_gicp/nano_gicp.hpp>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>

#include <locale>

#include "cluster_manager.hpp"
#include "dataio.hpp"
#include "desc/STDesc.h"
#include "eval.hpp"
#include "semantic_teaser.h"

pcl::PointCloud<PointType>::ConstPtr getCloud(std::string filename);
void setParams(int semantic_class, double cluster_distance_threshold,
               int minNum, int maxNum, clusterManager::ClusterParams &params,
               clusterManager::DCVCParam &seg_param);
void merge_label(const string label_file_path,
                 pcl::PointCloud<PointType>::Ptr raw_pc,
                 pcl::PointCloud<PointL>::Ptr semantic_pc,
                 double label_deter_rate);
void apply_color_mapping_spvnas(int label, int &r, int &g, int &b);
void apply_color_mapping_chz(int label, int &r, int &g, int &b);
void color_pc(const pcl::PointCloud<PointL>::Ptr semantic_cloud,
              pcl::PointCloud<PointRGB>::Ptr colored_cloud);
void setCovMatsMarkers(
    visualization_msgs::MarkerArray &markerArray,
    const pcl::PointCloud<PointType>::Ptr cloud,
    const std::vector<Eigen::Matrix3d,
                      Eigen::aligned_allocator<Eigen::Matrix3d>> &covariances,
    const std::vector<float> rgb_color, int id);
pcl::PointCloud<PointL>::Ptr
random_downsample_pl(pcl::PointCloud<PointL>::Ptr cloud_ori, int ratio);
void ds_point_cloud(pcl::PointCloud<PointType>::Ptr &pc_in,
                    pcl::PointCloud<PointType>::Ptr &pc_out);

const char separator = ' ';
const int nameWidth = 22;
const int numWidth = 8;

int main(int argc, char **argv) {
  ros::init(argc, argv, "global_localizaiton");
  ros::NodeHandle nh;

  std::string outram_path = ros::package::getPath("outram");
  std::cout << outram_path << std::endl;
  // ******************** cluster params ****************************
  int car_class_num;
  double car_min_cluster_dist;
  int car_min_point_num, car_max_point_num;
  bool use_car;
  bool use_DCVC_car;
  int car_minSeg;

  nh.param<int>("/car_param/class_num", car_class_num, 0);
  nh.param<double>("/car_param/min_dist", car_min_cluster_dist, 0.5);
  nh.param<int>("/car_param/min_num", car_min_point_num, 5);
  nh.param<int>("/car_param/max_num", car_max_point_num, 200);
  nh.param<bool>("/car_param/use_car", use_car, false);
  nh.param<bool>("/car_param/use_DCVC", use_DCVC_car, false);
  nh.param<int>("/car_param/DCVC_min_num", car_minSeg, 0);

  // trunk cluster params
  int trunk_class_num;
  double trunk_min_cluster_dist;
  int trunk_min_point_num, trunk_max_point_num;
  bool use_trunk;
  bool use_DCVC_trunk;
  int trunk_minSeg;

  nh.param<int>("/trunk_param/class_num", trunk_class_num, 0);
  nh.param<double>("/trunk_param/min_dist", trunk_min_cluster_dist, 0.5);
  nh.param<int>("/trunk_param/min_num", trunk_min_point_num, 5);
  nh.param<int>("/trunk_param/max_num", trunk_max_point_num, 200);
  nh.param<bool>("/trunk_param/use_trunk", use_trunk, false);
  nh.param<bool>("/trunk_param/use_DCVD", use_DCVC_trunk, false);
  nh.param<int>("/trunk_param/DCVC_min_num", trunk_minSeg, 0);

  // pole cluster params
  int pole_class_num;
  double pole_min_cluster_dist;
  int pole_min_point_num, pole_max_point_num;
  bool use_pole;
  bool use_DCVC_pole;
  int pole_minSeg;

  nh.param<int>("/pole_param/class_num", pole_class_num, 0);
  nh.param<double>("/pole_param/min_dist", pole_min_cluster_dist, 0.5);
  nh.param<int>("/pole_param/min_num", pole_min_point_num, 5);
  nh.param<int>("/pole_param/max_num", pole_max_point_num, 200);
  nh.param<bool>("/pole_param/use_pole", use_pole, false);
  nh.param<bool>("/pole_param/use_DCVD", use_DCVC_pole, false);
  nh.param<int>("/pole_param/DCVC_min_num", pole_minSeg, 0);

  // DCVC segmentation params
  double startR, deltaR, deltaP, deltaA;
  int minSeg;
  nh.param<double>("/DCVC_param/startR", startR, 0.0);
  nh.param<double>("/DCVC_param/deltaR", deltaR, 0.0);
  nh.param<double>("/DCVC_param/deltaP", deltaP, 0.0);
  nh.param<double>("/DCVC_param/deltaA", deltaA, 0.0);
  nh.param<int>("/DCVC_param/minSeg", minSeg, 0);

  // registration parmas
  double noise_level;
  nh.param<double>("/noise_level", noise_level, 0.06);
  double distribution_noise_level;
  nh.param<double>("/distribution_noise_level", distribution_noise_level, 0.06);

  bool solving_w_cov;
  nh.param<bool>("/solving_w_cov", solving_w_cov, true);
  bool solving_all2all;
  nh.param<bool>("/solving_all2all", solving_all2all, true);
  bool ds_all2all;
  nh.param<bool>("/dsample", ds_all2all, true);

  std::string gt_file_path;
  nh.param<std::string>("/gt_file_path", gt_file_path, "");
  gt_file_path = outram_path + gt_file_path;

  std::string scan_path, label_path;
  nh.param<std::string>("/scan_path", scan_path, "");
  scan_path = outram_path + scan_path;

  nh.param<std::string>("/label_path", label_path, "");
  label_path = outram_path + label_path;

  std::string viz_map_file_path;
  nh.param<std::string>("/viz_map_file_path", viz_map_file_path, "");
  viz_map_file_path = outram_path + viz_map_file_path;

  std::string cluster_map_file_path;
  nh.param<std::string>("/cluster_map_path", cluster_map_file_path, "");
  cluster_map_file_path = outram_path + cluster_map_file_path;

  std::string cluster_map_cov_file_path;
  nh.param<std::string>("/cluster_map_cov_file_path", cluster_map_cov_file_path,
                        "");
  cluster_map_cov_file_path = outram_path + cluster_map_cov_file_path;

  bool enable_visualization;
  nh.param<bool>("/enable_visualization", enable_visualization, true);

  bool step_stop;
  nh.param<bool>("/step_stop", step_stop, false);
  bool use_semantic;
  nh.param<bool>("/use_semantic", use_semantic, true);

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
      GT_list_camera;
  GT_list_camera = load_poses_from_transform_matrix(gt_file_path);

  std::vector<std::string> scanFiles;
  batch_read_filenames_in_folder(scan_path, "_filelist.txt", ".pcd", scanFiles);
  std::vector<std::string> labelFiles;
  batch_read_filenames_in_folder(label_path, "_filelist.txt", ".label",
                                 labelFiles);

  int begin_index, eva_frame_num;
  nh.param<int>("/begin_index", begin_index, 0);
  nh.param<int>("/eva_frame_num", eva_frame_num, 0);

  int total_frame;
  if (step_stop) {
    total_frame = begin_index + eva_frame_num;
  } else {
    total_frame = scanFiles.size();
  }

  std::vector<double> calib_vec;
  Eigen::Matrix4d calib_mat;
  nh.getParam("calibration_matrix/data", calib_vec);

  if (!vec2calib(calib_vec, calib_mat)) {
    return 0;
  } else {
    std::cout << "Mat Tr read complete." << std::endl;
  }

  // publishers
  ros::Publisher SrcPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/source", 100);
  ros::Publisher SrcColoredPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/srccolored", 100);
  ros::Publisher SrcCarCenterPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/src_car_nodes", 100);
  ros::Publisher SrcBuildingCenterPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/src_building_nodes", 100);
  ros::Publisher SrcVegCenterPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/src_veg_nodes", 100);
  ros::Publisher SrcTrunkPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/src_trunk_nodes", 100);
  ros::Publisher SrcPolePublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/src_pole_nodes", 100);
  ros::Publisher SrcCovPublisher =
      nh.advertise<visualization_msgs::MarkerArray>("/src_cov", 100);

  ros::Publisher TgtCarCenterPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/tgt_car_nodes", 100);
  ros::Publisher TgtTrunkPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/tgt_trunk_nodes", 100);
  ros::Publisher TgtPolePublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/tgt_pole_nodes", 100);
  ros::Publisher TgtCovPublisher =
      nh.advertise<visualization_msgs::MarkerArray>("/tgt_cov", 100);
  ros::Publisher MapVizPcPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/map_pure", 100);

  ros::Publisher InlierCorrPublisher =
      nh.advertise<visualization_msgs::Marker>("/inlierCorres", 100);
  ros::Publisher InitalCorrPublisher =
      nh.advertise<visualization_msgs::Marker>("/initCorres", 100);
  ros::Publisher pubMapSTD =
      nh.advertise<visualization_msgs::MarkerArray>("/stds_map", 10);
  ros::Publisher pubScanSTD =
      nh.advertise<visualization_msgs::MarkerArray>("/stds_scan", 10);

  ros::Publisher TransformedPublisher =
      nh.advertise<sensor_msgs::PointCloud2>("/transformed_pc", 100);

  // Generate instance map
  clusterManager::DCVCParam car_DCVC_param;
  car_DCVC_param.startR = startR;
  car_DCVC_param.deltaR = deltaR;
  car_DCVC_param.deltaP = deltaP;
  car_DCVC_param.deltaA = deltaA;
  car_DCVC_param.minSeg = car_minSeg;

  clusterManager::DCVCParam trunk_DCVC_param;
  trunk_DCVC_param.startR = startR;
  trunk_DCVC_param.deltaR = deltaR;
  trunk_DCVC_param.deltaP = deltaP;
  trunk_DCVC_param.deltaA = deltaA;
  trunk_DCVC_param.minSeg = trunk_minSeg;

  clusterManager::DCVCParam pole_DCVC_param;
  pole_DCVC_param.startR = startR;
  pole_DCVC_param.deltaR = deltaR;
  pole_DCVC_param.deltaP = deltaP;
  pole_DCVC_param.deltaA = deltaA;
  pole_DCVC_param.minSeg = pole_minSeg;

  clusterManager::ClusterParams car_params;
  setParams(car_class_num, car_min_cluster_dist, car_min_point_num,
            car_max_point_num, car_params, car_DCVC_param);
  clusterManager::ClusterParams trunk_params;
  setParams(trunk_class_num, trunk_min_cluster_dist, trunk_min_point_num,
            trunk_max_point_num, trunk_params, trunk_DCVC_param);
  clusterManager::ClusterParams pole_params;
  setParams(pole_class_num, pole_min_cluster_dist, pole_min_point_num,
            pole_max_point_num, pole_params, pole_DCVC_param);

  // load colored global map for visualization
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapVizPc(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(viz_map_file_path, *mapVizPc) ==
      -1) {
    ROS_ERROR("Couldn't read VIZ map file. \n");
    return (-1);
  } else {
    ROS_INFO("Viz Map loaded. \n");
  }

  std::vector<pcl::PointCloud<PointType>> tgt_sem_vec;
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      tgt_covariances;

  pcl::PointCloud<pcl::PointXYZLNormal>::Ptr mapFeatCloud(
      new pcl::PointCloud<pcl::PointXYZLNormal>);

  pcl::PointCloud<PointType>::Ptr tgtCarCloud(new pcl::PointCloud<PointType>);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      tgt_car_covariances;
  pcl::PointCloud<pcl::PointXYZLNormal>::Ptr mapCarCloudPtr(
      new pcl::PointCloud<pcl::PointXYZLNormal>);
  pcl::PointCloud<PointType>::Ptr tgtTrunkCloud(new pcl::PointCloud<PointType>);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      tgt_trunk_covariances;
  pcl::PointCloud<pcl::PointXYZLNormal>::Ptr mapTrunkCloudPtr(
      new pcl::PointCloud<pcl::PointXYZLNormal>);
  pcl::PointCloud<PointType>::Ptr tgtPoleCloud(new pcl::PointCloud<PointType>);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      tgt_pole_covariances;
  pcl::PointCloud<pcl::PointXYZLNormal>::Ptr mapPoleCloudPtr(
      new pcl::PointCloud<pcl::PointXYZLNormal>);

  std::ifstream fin(cluster_map_cov_file_path, std::ios::binary);

  pcl::io::loadPCDFile<pcl::PointXYZLNormal>(cluster_map_file_path,
                                             *mapFeatCloud);
  if (solving_w_cov) {
    if (!read_covariance_vec(fin, tgt_covariances)) {
      ROS_ERROR("Loading cov mat vector failed!");
    }
  }

  // split to different semantic clouds
  for (auto p : mapFeatCloud->points) {
    if (p.label == car_params.semanticLabel) {
      mapCarCloudPtr->points.push_back(p);
    } else if (p.label == trunk_params.semanticLabel) {
      mapTrunkCloudPtr->points.push_back(p);
    } else if (p.label == pole_params.semanticLabel) {
      mapPoleCloudPtr->points.push_back(p);
    }
  }
  pcl::copyPointCloud(*mapCarCloudPtr, *tgtCarCloud);
  pcl::copyPointCloud(*mapTrunkCloudPtr, *tgtTrunkCloud);
  pcl::copyPointCloud(*mapPoleCloudPtr, *tgtPoleCloud);
  tgt_sem_vec.emplace_back(*tgtCarCloud);
  tgt_sem_vec.emplace_back(*tgtTrunkCloud);
  tgt_sem_vec.emplace_back(*tgtPoleCloud);

  // generate triangulated 3D scene graph
  ConfigSetting config_setting;
  read_parameters(nh, config_setting);
  STDescManager *std_manager = new STDescManager(config_setting);

  std::vector<STDesc> map_stds_vec;
  if (solving_w_cov) {
    std_manager->BuildMapCovSTD(mapFeatCloud, tgt_covariances, map_stds_vec);
  } else {
    std_manager->BuildMapSTD(mapFeatCloud, map_stds_vec);
  }

  int success_count = 0;
  ros::Rate loop_rate(10);
  ros::Rate slow_loop(500);
  for (int index = begin_index; index < total_frame; index += 1) {
    std::cout << "*********  Current frame index:  " << index << "  *********"
              << std::endl;

    std::vector<double> temp_time_vec;

    // load cloud
    pcl::PointCloud<PointType>::Ptr srcRaw(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr srcGT(new pcl::PointCloud<PointType>);

    // int index = source_scan_index;
    std::string curPcPath = scanFiles[index];
    std::string curlabelPath = labelFiles[index];

    *srcRaw = *getCloud(curPcPath);

    Eigen::Matrix4d gt_lidar = GT_list_camera[index] * calib_mat;

    Eigen::Matrix4d vis_mat = Eigen::MatrixXd::Identity(4, 4);
    vis_mat(2, 3) += 50; // levitate for visualization

    vis_mat = gt_lidar * vis_mat;
    pcl::transformPointCloud(*srcRaw, *srcGT, vis_mat);

    // concatenate labels with point clouds
    pcl::PointCloud<PointL>::Ptr tempSemanticPc(new pcl::PointCloud<PointL>);
    pcl::PointCloud<PointL>::Ptr srcSemanticPc(new pcl::PointCloud<PointL>);
    merge_label(curlabelPath, srcRaw, srcSemanticPc, -1);

    std::vector<pcl::PointCloud<PointType>> src_sem_vec;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        src_covariances;
    pcl::PointCloud<PointType>::Ptr tCloud1(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tCloud2(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tCloud3(new pcl::PointCloud<PointType>);

    // src cloud nodes
    pcl::PointCloud<PointType>::Ptr srcCarCloud(new pcl::PointCloud<PointType>);
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        src_car_covariances;

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr srcCarCloudPtr(
        new pcl::PointCloud<pcl::PointXYZLNormal>);
    if (use_car) {
      clusterManager src_car_node;
      src_car_node.reset(car_params);
      src_car_node.segmentPointCloud(srcSemanticPc);
      src_car_node.computeCloudwNormal(srcCarCloudPtr, src_car_covariances);
      pcl::copyPointCloud(*srcCarCloudPtr, *tCloud1);
      if (ds_all2all) {
        ds_point_cloud(tCloud1, srcCarCloud);
      } else {
        pcl::copyPointCloud(*tCloud1, *srcCarCloud);
      }
      src_sem_vec.emplace_back(*srcCarCloud);
      std::copy(std::begin(src_car_covariances), std::end(src_car_covariances),
                std::back_inserter(src_covariances));
    }

    pcl::PointCloud<PointType>::Ptr srcTrunkCloud(
        new pcl::PointCloud<PointType>);
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        src_trunk_covariances;
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr srcTrunkCloudPtr(
        new pcl::PointCloud<pcl::PointXYZLNormal>);
    if (use_trunk) {
      clusterManager src_trunk_node;
      src_trunk_node.reset(trunk_params);
      src_trunk_node.segmentPointCloud(srcSemanticPc);
      src_trunk_node.computeCloudwNormal(srcTrunkCloudPtr,
                                         src_trunk_covariances);
      pcl::copyPointCloud(*srcTrunkCloudPtr, *tCloud2);
      if (ds_all2all) {
        ds_point_cloud(tCloud2, srcTrunkCloud);
      } else {
        pcl::copyPointCloud(*tCloud2, *srcTrunkCloud);
      }
      src_sem_vec.emplace_back(*srcTrunkCloud);
      std::copy(std::begin(src_trunk_covariances),
                std::end(src_trunk_covariances),
                std::back_inserter(src_covariances));
    }

    pcl::PointCloud<PointType>::Ptr srcPoleCloud(
        new pcl::PointCloud<PointType>);
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        src_pole_covariances;
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr srcPoleCloudPtr(
        new pcl::PointCloud<pcl::PointXYZLNormal>);
    if (use_pole) {
      clusterManager src_pole_node;
      src_pole_node.reset(pole_params);
      src_pole_node.segmentPointCloud(srcSemanticPc);
      src_pole_node.computeCloudwNormal(srcPoleCloudPtr, src_pole_covariances);
      pcl::copyPointCloud(*srcPoleCloudPtr, *tCloud3);
      if (ds_all2all) {
        ds_point_cloud(tCloud3, srcPoleCloud);
      } else {
        pcl::copyPointCloud(*tCloud3, *srcPoleCloud);
      }
      src_sem_vec.emplace_back(*srcPoleCloud);
      std::copy(std::begin(src_pole_covariances),
                std::end(src_pole_covariances),
                std::back_inserter(src_covariances));
    }

    // extract triangulated scene graph for current frame
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr srcFeatCloud(
        new pcl::PointCloud<pcl::PointXYZLNormal>);
    *srcFeatCloud = *srcCarCloudPtr + *srcTrunkCloudPtr;
    *srcFeatCloud += *srcPoleCloudPtr;

    std::vector<STDesc> current_stds;
    if (solving_w_cov) {
      std_manager->BuildSingleScanCovSTD(srcFeatCloud, src_covariances,
                                         current_stds);
    } else {
      std_manager->BuildSingleScanSTD(srcFeatCloud, current_stds);
    }

    // exhaustively searching correspondences
    if (use_semantic) {
      std_manager->SearchCorresSemSTD(current_stds, map_stds_vec);
    } else {
      std_manager->SearchCorresGeoSTD(current_stds, map_stds_vec);
    }

    semanticTeaser::Params params;
    params.teaser_params.noise_bound = noise_level;
    params.teaser_params.distribution_noise_bound = distribution_noise_level;

    semanticTeaser semSolver(params);

    Eigen::Matrix4d solution = Eigen::Matrix4d::Identity();

    if (std_manager->src_points_.cols() == 0 ||
        std_manager->tgt_points_.cols() == 0 ||
        srcFeatCloud->points.size() == 0) {
    } else {
      if (solving_all2all) {
        semSolver.solve_for_multiclass(src_sem_vec, tgt_sem_vec);
      } else if (solving_w_cov) {
        semSolver.solveSTDCovCorres(std_manager->src_points_,
                                    std_manager->tgt_points_,
                                    std_manager->src_covariances_matched_,
                                    std_manager->tgt_covariances_matched_);
      } else {
        semSolver.solveSTDcorres(std_manager->src_points_,
                                 std_manager->tgt_points_);
      }
      solution = semSolver.get_solution();
    }

    Eval eval;
    double translation_error;
    double rotation_error;
    eval.compute_adj_rpe(gt_lidar, solution, translation_error, rotation_error);

    std::cout << setprecision(4) << "\033[1;32m";
    std::cout << "Outram Translation Error = " << translation_error << " meter."
              << std::endl;
    std::cout << "       Rotational  Error = " << rotation_error
              << " deg.\033[0m" << std::endl;

    bool suc_reg_flag = false;
    if (translation_error < 5.0 && rotation_error < 10.0) {
      suc_reg_flag = true;
      success_count++;
    }

    pcl::PointCloud<PointType>::Ptr transformed_cloud(
        new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointRGB>::Ptr transformed_cloud_colored(
        new pcl::PointCloud<PointRGB>);
    Eigen::Matrix4d viz_transform;
    std::vector<int> pc_color;
    if (suc_reg_flag) {
      viz_transform = solution;
      pc_color = {0, 255, 0};
    } else {
      viz_transform = gt_lidar;
      pc_color = {255, 0, 0};
    }
    pcl::transformPointCloud(*srcRaw, *transformed_cloud, viz_transform);
    color_point_cloud(transformed_cloud, pc_color, transformed_cloud_colored);

    if (enable_visualization) {
      // for visualization
      pcl::PointCloud<PointRGB>::Ptr srcColoredRaw(
          new pcl::PointCloud<PointRGB>);
      color_pc(srcSemanticPc, srcColoredRaw);

      pcl::transformPointCloud<PointRGB>(*srcColoredRaw, *srcColoredRaw,
                                         vis_mat);
      pcl::transformPointCloud<pcl::PointXYZLNormal>(*srcCarCloudPtr,
                                                     *srcCarCloudPtr, vis_mat);
      pcl::transformPointCloud<pcl::PointXYZLNormal>(
          *srcTrunkCloudPtr, *srcTrunkCloudPtr, vis_mat);
      pcl::transformPointCloud<pcl::PointXYZLNormal>(*srcPoleCloudPtr,
                                                     *srcPoleCloudPtr, vis_mat);

      sensor_msgs::PointCloud2 SrcMsg = cloud2msg(*srcRaw);
      sensor_msgs::PointCloud2 SrcColoredMsg = cloud2msg(*srcColoredRaw);
      sensor_msgs::PointCloud2 SrcCarCenterMsg = cloud2msg(*srcCarCloudPtr);
      sensor_msgs::PointCloud2 SrcTrunkMsg = cloud2msg(*srcTrunkCloudPtr);
      sensor_msgs::PointCloud2 SrcPoleMsg = cloud2msg(*srcPoleCloudPtr);

      sensor_msgs::PointCloud2 TgtCarCenterMsg = cloud2msg(*mapCarCloudPtr);
      sensor_msgs::PointCloud2 TgtTrunkMsg = cloud2msg(*mapTrunkCloudPtr);
      sensor_msgs::PointCloud2 TgtPoleMsg = cloud2msg(*mapPoleCloudPtr);

      sensor_msgs::PointCloud2 TransformedMsg =
          cloud2msg(*transformed_cloud_colored);

      sensor_msgs::PointCloud2 VizMapMsg = cloud2msg(*mapVizPc);

      // correspondence visualization
      pcl::PointCloud<PointType> srcMaxClique;
      pcl::PointCloud<PointType> tgtMaxClique;
      semSolver.getMaxCliques(srcMaxClique, tgtMaxClique);
      pcl::transformPointCloud<PointType>(srcMaxClique, srcMaxClique, vis_mat);
      visualization_msgs::Marker inlierCorrMarker;
      std::vector<float> mc_color;
      if (suc_reg_flag) {
        mc_color = {0.0, 1.0, 0.0};
      } else {
        mc_color = {1.0, 0.0, 0.0};
      }

      setCorrespondenceMarker(srcMaxClique, tgtMaxClique, inlierCorrMarker, 0.5,
                              mc_color, 0);

      pcl::PointCloud<PointType> srcMatched;
      pcl::PointCloud<PointType> tgtMatched;
      semSolver.getInitCorr(srcMatched, tgtMatched);
      pcl::transformPointCloud<PointType>(srcMatched, srcMatched, vis_mat);
      visualization_msgs::Marker initalCorrMarker;
      setCorrespondenceMarker(srcMatched, tgtMatched, initalCorrMarker, 0.08,
                              {1.0, 0.0, 0.0}, 1);

      publish_map_std(std_manager->matched_stds_, pubMapSTD);
      publish_scan_std(current_stds, pubScanSTD);

      SrcPublisher.publish(SrcMsg);
      SrcColoredPublisher.publish(SrcColoredMsg);
      SrcCarCenterPublisher.publish(SrcCarCenterMsg);
      SrcTrunkPublisher.publish(SrcTrunkMsg);
      SrcPolePublisher.publish(SrcPoleMsg);

      TgtCarCenterPublisher.publish(TgtCarCenterMsg);
      TgtTrunkPublisher.publish(TgtTrunkMsg);
      TgtPolePublisher.publish(TgtPoleMsg);
      MapVizPcPublisher.publish(VizMapMsg);

      InlierCorrPublisher.publish(inlierCorrMarker);
      InitalCorrPublisher.publish(initalCorrMarker);

      TransformedPublisher.publish(TransformedMsg);
    }

    loop_rate.sleep();
    if (step_stop) {
      getchar();
    }
  }
  return 0;
}

void setParams(int semantic_class, double cluster_distance_threshold,
               int minNum, int maxNum, clusterManager::ClusterParams &params,
               clusterManager::DCVCParam &seg_param) {
  params.semanticLabel = semantic_class;
  params.clusterTolerance = cluster_distance_threshold;
  params.minClusterSize = minNum;
  params.maxClusterSize = maxNum;

  params.startR = seg_param.startR;
  params.deltaR = seg_param.deltaR;
  params.deltaP = seg_param.deltaP;
  params.deltaA = seg_param.deltaA;
  params.minSeg = seg_param.minSeg;
}

pcl::PointCloud<PointType>::ConstPtr getCloud(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  if (!file) {
    std::cerr << "error: failed to load point cloud " << filename << std::endl;
    return nullptr;
  }

  std::vector<float> buffer(1000000);
  size_t num_points = fread(reinterpret_cast<char *>(buffer.data()),
                            sizeof(float), buffer.size(), file) /
                      4;
  fclose(file);

  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
  cloud->resize(num_points);

  for (int i = 0; i < num_points; i++) {
    auto &pt = cloud->at(i);
    pt.x = buffer[i * 4];
    pt.y = buffer[i * 4 + 1];
    pt.z = buffer[i * 4 + 2];
    // Intensity is not in use
    //         pt.intensity = buffer[i * 4 + 3];
  }

  return cloud;
}

/**
 * @brief      Merge cloud and label to semantic_pc.
 * @param[in]  label_file_path; raw_point_cloud; out_semantic_pc
 * @return     None.
 */
void merge_label(const string label_file_path,
                 pcl::PointCloud<PointType>::Ptr raw_pc,
                 pcl::PointCloud<PointL>::Ptr semantic_pc,
                 double label_deter_rate) {
  // read label file
  std::ifstream in_stream(label_file_path.c_str(),
                          std::ios::in | std::ios::binary);
  vector<uint16_t> cloud_label_vec;
  cloud_label_vec.reserve(1000000);

  if (in_stream.is_open()) {
    // with 16 lower bit semantic label, 16 higher bit instance label
    uint32_t cur_whole_label;
    uint16_t cur_sem_label;

    while (in_stream.read((char *)&cur_whole_label, sizeof(cur_whole_label))) {
      cur_sem_label = cur_whole_label & 0xFFFF;
      cloud_label_vec.emplace_back(cur_sem_label);
    }
  } else {
    std::cerr << "error: failed to load label " << label_file_path << std::endl;
    return;
  }

  // sanity check for point cloud size
  if (raw_pc->points.size() != cloud_label_vec.size()) {
    std::cerr << "error: Point cloud size != label size" << std::endl;
    std::cout << "Point cloud size: " << raw_pc->points.size() << std::endl;
    std::cout << "Label size      : " << cloud_label_vec.size() << std::endl;
    return;
  }

  for (int i = 0; i < cloud_label_vec.size(); i++) {
    double cur_rand = (double)rand() / (RAND_MAX);
    if (cur_rand <= label_deter_rate) {
      cloud_label_vec[i] = 20;
    }
  }

  for (int i = 0; i < raw_pc->points.size(); i++) {
    PointL tmpL;
    tmpL.x = raw_pc->points[i].x;
    tmpL.y = raw_pc->points[i].y;
    tmpL.z = raw_pc->points[i].z;
    tmpL.label = cloud_label_vec[i];
    semantic_pc->points.push_back(tmpL);
  }

  semantic_pc->width = semantic_pc->points.size();
  semantic_pc->height = 1;
}

void apply_color_mapping_chz(int label, int &r, int &g, int &b) {
  switch (label) {
  case 0: // car
  {
    r = 100;
    g = 150;
    b = 245;
    break;
  }
  case 1: // truck
  {
    r = 80;
    g = 30;
    b = 180;
    break;
  }
  case 2: // bicycle
  {
    r = 30;
    g = 60;
    b = 150;
    break;
  }
  case 3: // person
  {
    r = 255;
    g = 30;
    b = 30;
    break;
  }
  case 4: // road
  {
    r = 255;
    g = 0;
    b = 255;
    break;
  }
  case 5: // parking
  {
    r = 255;
    g = 150;
    b = 255;
    break;
  }
  case 6: // sidewalk
  {
    r = 75;
    g = 0;
    b = 75;
    break;
  }
  case 7: // building
  {
    r = 255;
    g = 200;
    b = 0;
    break;
  }
  case 8: // vegetation
  {
    r = 0;
    g = 175;
    b = 0;
    break;
  }
  case 9: // pole
  {
    r = 255;
    g = 240;
    b = 150;
    break;
  }
  case 10: // trunk
  {
    r = 139;
    g = 69;
    b = 19;
    break;
  }
  default: // moving objects
  {
    r = 0;
    g = 0;
    b = 0;
  }
  }
}

void apply_color_mapping_spvnas(int label, int &r, int &g, int &b) {
  switch (label) {
  case 0: // car
  {
    r = 100;
    g = 150;
    b = 245;
    break;
  }
  case 1: // bicycle
  {
    r = 100;
    g = 230;
    b = 245;
    break;
  }
  case 2: // motorcycle
  {
    r = 30;
    g = 60;
    b = 150;
    break;
  }
  case 3: // truck
  {
    r = 80;
    g = 30;
    b = 180;
    break;
  }
  case 4: // other-vehicle
  {
    r = 0;
    g = 0;
    b = 255;
    break;
  }
  case 5: // person
  {
    r = 255;
    g = 30;
    b = 30;
    break;
  }
  case 6: // bicyclist
  {
    r = 255;
    g = 40;
    b = 200;
    break;
  }
  case 7: // motorcyclist
  {
    r = 150;
    g = 30;
    b = 90;
    break;
  }
  case 8: // road
  {
    r = 255;
    g = 0;
    b = 255;
    break;
  }
  case 9: // parking
  {
    r = 255;
    g = 150;
    b = 255;
    break;
  }
  case 10: // sidewalk
  {
    r = 75;
    g = 0;
    b = 75;
    break;
  }
  case 11: // other-ground
  {
    r = 175;
    g = 0;
    b = 75;
    break;
  }
  case 12: // building
  {
    r = 255;
    g = 200;
    b = 0;
    break;
  }
  case 13: // fence
  {
    r = 255;
    g = 120;
    b = 50;
    break;
  }
  case 14: // vegetation
  {
    r = 0;
    g = 175;
    b = 0;
    break;
  }
  case 15: // trunk
  {
    r = 135;
    g = 60;
    b = 0;
    break;
  }
  case 16: // terrain
  {
    r = 150;
    g = 240;
    b = 80;
    break;
  }
  case 17: // pole
  {
    r = 255;
    g = 240;
    b = 150;
    break;
  }
  case 18: // traffic-sign
  {
    r = 255;
    g = 0;
    b = 0;
    break;
  }
  default: // moving objects
  {
    r = 0;
    g = 0;
    b = 0;
  }
  }
}

/**
 * @brief      Color point cloud according to per point semantic labels.
 * @param[in]  semantic_cloud: input semantic cloud ptr (with label)
 * @param[in]  colored_cloud:  colored cloud ptr
 */
void color_pc(const pcl::PointCloud<PointL>::Ptr semantic_cloud,
              pcl::PointCloud<PointRGB>::Ptr colored_cloud) {
  int r, g, b;
  uint16_t temp_label;
  PointRGB temp_pt;
  for (int i = 0; i < semantic_cloud->points.size(); ++i) {
    temp_pt.x = semantic_cloud->points[i].x;
    temp_pt.y = semantic_cloud->points[i].y;
    temp_pt.z = semantic_cloud->points[i].z;
    temp_label = semantic_cloud->points[i].label;
    apply_color_mapping_chz((int)temp_label, r, g, b);
    temp_pt.r = r;
    temp_pt.g = g;
    temp_pt.b = b;
    colored_cloud->points.push_back(temp_pt);
  }
}

void setCovMatsMarkers(
    visualization_msgs::MarkerArray &markerArray,
    const pcl::PointCloud<PointType>::Ptr cloud,
    const std::vector<Eigen::Matrix3d,
                      Eigen::aligned_allocator<Eigen::Matrix3d>> &covariances,
    const std::vector<float> rgb_color = {0.0, 0.0, 0.0}, int id = 0) {
  // int id = 1;
  Eigen::EigenSolver<Eigen::Matrix3d> es;
  for (int i = 0; i < covariances.size(); ++i) {
    visualization_msgs::Marker covMarker;

    covMarker.header.frame_id = "map";
    covMarker.header.stamp = ros::Time();
    covMarker.ns = "my_namespace";
    covMarker.id = id; // To avoid overlap
    covMarker.type = visualization_msgs::Marker::CYLINDER;
    covMarker.action = visualization_msgs::Marker::ADD;

    PointType tempP = cloud->points[i];
    covMarker.pose.position.x = tempP.x;
    covMarker.pose.position.y = tempP.y;
    covMarker.pose.position.z = tempP.z;

    es.compute(covariances[i], true);
    covMarker.scale.x = sqrt(es.eigenvalues()(0).real());
    covMarker.scale.y = sqrt(es.eigenvalues()(1).real());
    covMarker.scale.z = sqrt(es.eigenvalues()(2).real());

    covMarker.color.r = rgb_color[0];
    covMarker.color.g = rgb_color[1];
    covMarker.color.b = rgb_color[2];
    covMarker.color.a = 1.0; // Don't forget to set the alpha!

    Eigen::Matrix3d eigen_mat = es.eigenvectors().real();
    Eigen::Matrix3d rot_mat = eigen_mat.transpose();
    // eigen_mat.normalize();
    Eigen::Quaterniond quat(rot_mat);
    quat.normalize();

    geometry_msgs::Quaternion geo_quat;
    tf::quaternionEigenToMsg(quat, geo_quat);

    covMarker.pose.orientation.x = geo_quat.x;
    covMarker.pose.orientation.y = geo_quat.y;
    covMarker.pose.orientation.z = geo_quat.z;
    covMarker.pose.orientation.w = geo_quat.w;

    markerArray.markers.push_back(covMarker);
    id++;
  }
}

pcl::PointCloud<PointL>::Ptr
random_downsample_pl(pcl::PointCloud<PointL>::Ptr cloud_ori, int ratio) {
  pcl::PointCloud<PointL>::Ptr sampled_pc(new pcl::PointCloud<PointL>);

  for (int i = 0; i < cloud_ori->points.size(); ++i) {
    if (i % ratio == 0) {
      sampled_pc->points.push_back(cloud_ori->points[i]);
    }
  }

  return sampled_pc;
}

void ds_point_cloud(pcl::PointCloud<PointType>::Ptr &pc_in,
                    pcl::PointCloud<PointType>::Ptr &pc_out) {
  int N = std::min(int(pc_in->points.size()), 3);
  for (int index = 0; index < N; index++) {
    pc_out->points.push_back(pc_in->points[index]);
  }
}