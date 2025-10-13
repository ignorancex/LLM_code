/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>

#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#include "xfeat.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#define SKIP_FIRST_CNT 10
using namespace std;
using namespace cv;


queue<sensor_msgs::ImageConstPtr> image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
// std::mutex m_buf;
// std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;
int total_frame_ = 0;
std::unordered_map<size_t, std::pair<Eigen::Vector3d,Eigen::Vector3d>> active_tracks_feature_;
XFeat xfeat_;
Ptr<ORB> orb = ORB::create();

cv::Mat prev_mkpts0_raw_, prev_feats0_raw_, prev_sc0_raw_;
cv::Mat mkpts0_raw_, feats0_raw_, sc0_raw_;
cv::Mat last_image_;

vector<KeyPoint> orb_keypoints1,orb_keypoints2;
Mat orb_descriptors1,orb_descriptors2;

int LOAD_PREVIOUS_POSE_GRAPH;
const std::string prefix = "frame";
const int length = 4; // 序号长度

struct ImageInfo{
    size_t image_id_;
    Eigen::Vector4d q_;
    Eigen::Vector3d t_;
    cv::Mat Image_;
    std::vector<cv::Point2f> features;
    std::vector<double> feat_ids;
};

std::vector<ImageInfo> active_tracks_image_;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_odometry_rect;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

ros::Publisher pub_point_cloud, pub_margin_cloud;

std::string generateFileName(const std::string& prefix, int index, int length) {

    std::ostringstream oss;

    oss << prefix << std::setw(length) << std::setfill('0') << index << ".jpg";

    return oss.str();

}
void ransac_run(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> queryKeypoint, std::vector<cv::KeyPoint> objectKeypoint, std::vector<cv::DMatch> &matches_ransac)
{
    std::vector<cv::Point2f> srcPoints(matches.size()),dstPoints(matches.size());

    for(int i=0;i<matches.size();i++){
        srcPoints[i]=queryKeypoint[matches[i].queryIdx].pt;
        dstPoints[i]=objectKeypoint[matches[i].trainIdx].pt;
    }

    std::vector<int> inliersMask(srcPoints.size());

    cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 5, inliersMask);

    for(int i = 0; i < inliersMask.size(); i++){
        if(inliersMask[i]) matches_ransac.push_back(matches[i]);
    }

    std::cout << "inliers count = " << matches_ransac.size() << std::endl;
    std::cout << "all count = " << matches.size() << std::endl;
    std::cout << "keypoint count = " << queryKeypoint.size() << std::endl;
}


void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
   
                cv_bridge::CvImageConstPtr ptr;
                cv_bridge::CvImageConstPtr ptr0;
                if (image_msg->encoding == "8UC1")
                {
                    sensor_msgs::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                    ptr0 = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
                }
                else
                {
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                    ptr0 = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
                }

                if(total_frame_==0)
                {
                    xfeat_.detectAndCompute(ptr0->image, prev_mkpts0_raw_, prev_feats0_raw_, prev_sc0_raw_);
                    
                    
                    KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                     prev_mkpts0_raw_, prev_feats0_raw_, 0, sequence);   
                    // KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image, 0, sequence);   
                    keyframe->color_img_ = ptr0->image.clone();
                    start_flag = 1;
                    cout<<"addKeyframe"<<endl;
                    posegraph.addKeyFrame(keyframe, 1);
                    frame_index++;
                    total_frame_++;
                    last_image_ = ptr->image.clone();
                    
                }else{      
                    // std::cout<<"total_frame_: "<<total_frame_<<std::endl;
                    int step = 5;
                    if(LOAD_PREVIOUS_POSE_GRAPH)
                        step = 40;

                    if(total_frame_%step==0)
                    {
                            cout<<"detectAndCompute Xfeat"<<endl;
                            xfeat_.detectAndCompute(ptr0->image, mkpts0_raw_, feats0_raw_, sc0_raw_);
                            
                            if(LOAD_PREVIOUS_POSE_GRAPH)
                            {
                                // KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                // 0, sequence);   
                                KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                    mkpts0_raw_, feats0_raw_, 0, sequence);   
                                keyframe->color_img_ = ptr0->image.clone();
                                start_flag = 1;
                                cout<<"addKeyframe Directly"<<endl;
                                posegraph.addKeyFrame(keyframe, 1);
                            
                                frame_index++;
                                
                            }else{
                            double error = 0;
                            double flow = 0;
                            bool use_xfeat = true;
                            if(use_xfeat){
                                    std::vector<cv::DMatch> matches2;
                                    std::vector<cv::Point2f> points1;
                                    std::vector<cv::Point2f> points2;
                                    xfeat_.match_reshaped(prev_mkpts0_raw_, prev_feats0_raw_, mkpts0_raw_, feats0_raw_, matches2, 200.0, points1, points2);
                                    std::vector<cv::KeyPoint> keypoints1, keypoints2;
                                    std::vector<cv::DMatch> matches_ransac;
                                        for (size_t i = 0; i < points1.size(); i++) {
                                            keypoints1.emplace_back(points1[i], 0);
                                            keypoints2.emplace_back(points2[i], 0);		
                                        }

                                    ransac_run(matches2, keypoints1, keypoints2, matches_ransac);
                                    for (size_t i = 0; i < matches_ransac.size(); i++)
                                    {
                                        cv::Point2f pt1 = keypoints1[matches_ransac[i].queryIdx].pt;
                                        cv::Point2f pt2 = keypoints2[matches_ransac[i].trainIdx].pt;
                                        error += sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
                                        flow += pt2.x - pt1.x;
                                    }
                                    error /= (double)matches_ransac.size();
                                    flow /= (double)matches_ransac.size();
                                    std::cout<<"using xfeat===error and flow: "<<error<<" "<<flow<<std::endl;
                            }else
                            {
                                   orb->detectAndCompute(last_image_, noArray(), orb_keypoints1, orb_descriptors1);
                                   orb->detectAndCompute(ptr->image, noArray(), orb_keypoints2, orb_descriptors2);

                                    // 使用BFMatcher进行特征匹配（Hamming距离）
                                    BFMatcher matcher(NORM_HAMMING);
                                    vector<vector<DMatch>> knnMatches;
                                    matcher.knnMatch(orb_descriptors1, orb_descriptors2, knnMatches, 2);

                                    // 应用Lowe's ratio test筛选匹配
                                    vector<DMatch> goodMatches;
                                    const float ratio_thresh = 0.75f;
                                    for (size_t i = 0; i < knnMatches.size(); i++) {
                                        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                                            goodMatches.push_back(knnMatches[i][0]);
                                        }
                                    }
                                    for (size_t i = 0; i < goodMatches.size(); i++)
                                    {
                                        cv::Point2f pt1 = orb_keypoints1[goodMatches[i].queryIdx].pt;
                                        cv::Point2f pt2 = orb_keypoints2[goodMatches[i].trainIdx].pt;
                                        error += sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
                                        flow += pt2.x - pt1.x;
                                    }
                                    error /= (double)goodMatches.size();
                                    flow /= (double)goodMatches.size();

                                    // Mat imgMatches;
                                    // drawMatches(last_image_, orb_keypoints1, ptr->image, orb_keypoints2, goodMatches, imgMatches,
                                    //         Scalar::all(-1), Scalar::all(-1), vector<char>(),
                                    //         DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                                    // 显示结果
                                    // imshow("ORB Feature Matches", imgMatches);
                                    // waitKey(0);


                                    std::cout<<"using orb===error and flow: "<<error<<" "<<flow<<std::endl;

                            }
                            
                            if(error>5)
                            {
                                std::cout<<"Get keyframe: "<<error<<" "<<flow<<std::endl;
                                KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                    mkpts0_raw_, feats0_raw_, flow, sequence);   
                                // KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image, flow, sequence);   
                                keyframe->color_img_ = ptr0->image.clone();
                                cout<<"addKeyframe "<<frame_index<<endl;
                                posegraph.addKeyFrame(keyframe, 0);
                
                                frame_index++;

                                last_image_ = ptr->image.clone();

                                // orb_keypoints1.swap(orb_keypoints2);
                                // orb_keypoints1 = orb_keypoints2;
                                // orb_descriptors1.release();
                                // orb_descriptors1 = orb_descriptors2.clone();
                                
                                prev_mkpts0_raw_.release();
                                prev_feats0_raw_.release();

                                prev_mkpts0_raw_ = mkpts0_raw_.clone();
                                prev_feats0_raw_ = feats0_raw_.clone();

                                
                            }
                        }
                    }
                    total_frame_++;
                }
                
                
            
            
    //ROS_INFO("image_callback!");
    // m_buf.lock();
    // image_buf.push(image_msg);
    // m_buf.unlock();
    // printf(" image time %f \n", image_msg->header.stamp.toSec());

    // detect unstable camera stream

}







void process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;

        if(!image_buf.empty())
        {
            image_msg = image_buf.front();
            image_buf.pop();
            cout<<"build keyframe "<<total_frame_<<endl;
            if (image_msg != NULL)
            {
                cv_bridge::CvImageConstPtr ptr;
                cv_bridge::CvImageConstPtr ptr0;
                if (image_msg->encoding == "8UC1")
                {
                    sensor_msgs::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                    ptr0 = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
                }
                else
                {
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                    ptr0 = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
                }
                if(total_frame_==0)
                {
                    //xfeat_.detectAndCompute(ptr0->image, prev_mkpts0_raw_, prev_feats0_raw_, prev_sc0_raw_);
                    
                    orb->detectAndCompute(ptr->image, noArray(), orb_keypoints1, orb_descriptors1);
                    // KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                    //                 orb_keypoints1, orb_descriptors1, 0, sequence);   
                    KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image, 0, sequence);   
                    // keyframe->color_img_ = ptr0->image.clone();
                    start_flag = 1;
                    cout<<"addKeyframe"<<endl;
                    posegraph.addKeyFrame(keyframe, 1);
                    frame_index++;
                    total_frame_++;
                    last_image_ = ptr->image.clone();
                    
                }else{      
                    
                    if(total_frame_%10==0)
                    {
                            // cout<<"detectAndCompute Xfeat"<<endl;
                            // xfeat_.detectAndCompute(ptr0->image, mkpts0_raw_, feats0_raw_, sc0_raw_);
                            
                            if(LOAD_PREVIOUS_POSE_GRAPH)
                            {
                                KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                0, sequence);   
                                // KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                //     orb_keypoints1, orb_descriptors1, 0, sequence);   
                                // keyframe->color_img_ = ptr0->image.clone();
                                start_flag = 1;
                                cout<<"addKeyframe Directly"<<endl;
                                posegraph.addKeyFrame(keyframe, 1);
                                frame_index++;
                                
                            }else{
                            double error = 0;
                            double flow = 0;
                            bool use_xfeat = false;
                            if(use_xfeat){
                                    std::vector<cv::DMatch> matches2;
                                    std::vector<cv::Point2f> points1;
                                    std::vector<cv::Point2f> points2;
                                    xfeat_.match(prev_mkpts0_raw_, prev_feats0_raw_, mkpts0_raw_, feats0_raw_, matches2, 200.0, points1, points2);
                                    std::vector<cv::KeyPoint> keypoints1, keypoints2;
                                    std::vector<cv::DMatch> matches_ransac;
                                        for (size_t i = 0; i < points1.size(); i++) {
                                            keypoints1.emplace_back(points1[i], 0);
                                            keypoints2.emplace_back(points2[i], 0);		
                                        }

                                    ransac_run(matches2, keypoints1, keypoints2, matches_ransac);
                                    for (size_t i = 0; i < matches_ransac.size(); i++)
                                    {
                                        cv::Point2f pt1 = keypoints1[matches_ransac[i].queryIdx].pt;
                                        cv::Point2f pt2 = keypoints2[matches_ransac[i].trainIdx].pt;
                                        error += sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
                                        flow += pt2.x - pt1.x;
                                    }
                                    error /= (double)matches_ransac.size();
                                    flow /= (double)matches_ransac.size();
                                    std::cout<<"using xfeat===error and flow: "<<error<<" "<<flow<<std::endl;
                            }else
                            {
                                   orb->detectAndCompute(last_image_, noArray(), orb_keypoints1, orb_descriptors1);
                                   orb->detectAndCompute(ptr->image, noArray(), orb_keypoints2, orb_descriptors2);

                                    // 使用BFMatcher进行特征匹配（Hamming距离）
                                    BFMatcher matcher(NORM_HAMMING);
                                    vector<vector<DMatch>> knnMatches;
                                    matcher.knnMatch(orb_descriptors1, orb_descriptors2, knnMatches, 2);

                                    // 应用Lowe's ratio test筛选匹配
                                    vector<DMatch> goodMatches;
                                    const float ratio_thresh = 0.75f;
                                    for (size_t i = 0; i < knnMatches.size(); i++) {
                                        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                                            goodMatches.push_back(knnMatches[i][0]);
                                        }
                                    }
                                    for (size_t i = 0; i < goodMatches.size(); i++)
                                    {
                                        cv::Point2f pt1 = orb_keypoints1[goodMatches[i].queryIdx].pt;
                                        cv::Point2f pt2 = orb_keypoints2[goodMatches[i].trainIdx].pt;
                                        error += sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
                                        flow += pt2.x - pt1.x;
                                    }
                                    error /= (double)goodMatches.size();
                                    flow /= (double)goodMatches.size();

                                    // Mat imgMatches;
                                    // drawMatches(last_image_, orb_keypoints1, ptr->image, orb_keypoints2, goodMatches, imgMatches,
                                    //         Scalar::all(-1), Scalar::all(-1), vector<char>(),
                                    //         DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                                    // 显示结果
                                    // imshow("ORB Feature Matches", imgMatches);
                                    // waitKey(0);


                                    std::cout<<"using orb===error and flow: "<<error<<" "<<flow<<std::endl;

                            }
                            
                            if(error>3)
                            {
                                std::cout<<"Get keyframe: "<<error<<" "<<flow<<std::endl;
                                // KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image,
                                //     orb_keypoints2, orb_descriptors2, flow, sequence);   
                                KeyFrame* keyframe = new KeyFrame(image_msg->header.stamp.toSec(), frame_index, ptr->image, flow, sequence);   
                                // keyframe->color_img_ = ptr0->image.clone();
                                start_flag = 1;
                                cout<<"addKeyframe "<<frame_index<<endl;
                                posegraph.addKeyFrame(keyframe, 0);
                                frame_index++;

                                last_image_ = ptr->image.clone();

                                // orb_keypoints1.swap(orb_keypoints2);
                                // orb_keypoints1 = orb_keypoints2;
                                // orb_descriptors1.release();
                                // orb_descriptors1 = orb_descriptors2.clone();
                                
                                // prev_mkpts0_raw_.release();
                                // prev_feats0_raw_.release();

                                // prev_mkpts0_raw_ = mkpts0_raw_.clone();
                                // prev_feats0_raw_ = feats0_raw_.clone();

                                
                            }
                        }
                    }
                    total_frame_++;
                }
                
                
            }
            }
        std::chrono::milliseconds dura(50);
        std::this_thread::sleep_for(dura);
    }
    // m_process.lock();
    // posegraph.savePoseGraph();
    // m_process.unlock();
    // printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
    // printf("program shutting down...\n");
}

void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            posegraph.savePoseGraph();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("program shutting down...\n");
            ros::shutdown();
        }

       
        if (c == 'p')
        {
            
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_fusion");
    ros::NodeHandle n("~");
    posegraph.registerPub(n);
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0.05; 
    active_tracks_feature_.clear();
    active_tracks_image_.clear();

    if(argc != 2)
    {
        printf("please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
               "for example: rosrun loop_fusion loop_fusion_node "
               "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 0;
    }
    printf("1\n");
    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);
    printf("2\n");
    cv::FileStorage fsSettings;
    
    try{
         fsSettings.open(config_file, cv::FileStorage::READ);
    }catch(cv::Exception ex)
    {
        return 1;
    }
    //  = cv::FileStorage(config_file, cv::FileStorage::READ);
    printf("3");
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE_TOPIC;
    

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    std::string pkg_path = ros::package::getPath("loop_fusion");
    string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
    //string vocabulary_file = pkg_path + "/support_files/ORBvoc.bin";
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph.loadVocabulary(vocabulary_file);
    
    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    printf("cam calib path: %s\n", cam0Path.c_str());
    // m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

    fsSettings["image0_topic"] >> IMAGE_TOPIC;        
    fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
    fsSettings["output_path"] >> VINS_RESULT_PATH;
    fsSettings["save_image"] >> DEBUG_IMAGE;

    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
    VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();


    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        posegraph.loadPoseGraph();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }
    /*
    //msckf
    ros::Subscriber sub_vio = n.subscribe("/firefly_sbx/vio/odom", 2000, vio_callback);
    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);
    ros::Subscriber sub_pose = n.subscribe("/firefly_sbx/vio/keyframe_pose", 2000, pose_callback);
    ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);
    ros::Subscriber sub_point = n.subscribe("/firefly_sbx/vio/keyframe_point", 2000, point_callback);
    ros::Subscriber sub_margin_point = n.subscribe("/firefly_sbx/vio/keyframe_point", 2000, margin_point_callback);
    */
   //openvins
    // ros::Subscriber sub_vio = n.subscribe("/ov_msckf/loop_pose", 2000, vio_callback);
    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 1000, image_callback); //  /ov_msckf/cam0/image_track
    // ros::Subscriber sub_pose = n.subscribe("/ov_msckf/loop_pose", 2000, pose_callback);
    // ros::Subscriber sub_extrinsic = n.subscribe("/ov_msckf/loop_extrinsic", 2000, extrinsic_callback);
    // ros::Subscriber sub_point = n.subscribe("/ov_msckf/loop_feats", 2000, point_callback);
    // ros::Subscriber sub_margin_point = n.subscribe("/ov_msckf/loop_feats", 2000, margin_point_callback);
    
    Eigen::Matrix4d T;
    /*
     //TUM-VI
    T<<-0.99952504, 0.02961534, -0.00852233,0.04727988,
          0.00750192, -0.03439736, -0.99938008, -0.04744323,
          -0.02989013, -0.99896935, 0.0340608, 0.03415885,
          0., 0., 0., 1. ;*/
    //    mymeyeOurs dataset
    /*
    T<<0.999891081429097,   0.014716783869332,  0.001114249117033,  -0.051901130404660,
   0.014713729579154,  -0.999888097994365,   0.002701416495616,   0.002650006340521,
   0.001153880593029,  -0.002684727501000,  -0.999995730389796,  -0.018337737428579,
                   0.0,                   0.0,                   0.0,   1.000000000000000;*/
  //Euroc dataset
  /*
    T<<0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
           0.999557249008, 0.0149672133247, 0.025715529948,  -0.064676986768,
           -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
           0, 0, 0, 1.;*/
    /*
    tic = T.inverse().block<3,1>(0,3);//相机和IMU之间的位姿变换
    qic = T.inverse().block<3,3>(0,0);*/

    

    // pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
    // pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    // pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud_loop_rect", 1000);
    // pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud_loop_rect", 1000);
    // pub_odometry_rect = n.advertise<nav_msgs::Odometry>("odometry_rect", 1000);

    // std::thread measurement_process;
    std::thread keyboard_command_process;

    // measurement_process = std::thread(process);
    keyboard_command_process = std::thread(command);
    
    ros::spin();

    return 0;
}
