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

#include "pose_graph.h"
// #include <filesystem>
 
// namespace fs = std::filesystem;

PoseGraph::PoseGraph()
{
    
    global_index = 0;

    global_loop_index = -1;

    load_previous_pose_graph = false;

    used_times = 0;
    lost_times = 0;

}

PoseGraph::~PoseGraph()
{
    // t_optimization.detach();
}

void PoseGraph::registerPub(ros::NodeHandle &n)
{
    // pub_pg_path = n.advertise<nav_msgs::Path>("pose_graph_path", 1000);
    // pub_base_path = n.advertise<nav_msgs::Path>("base_path", 1000);
    // pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
    pub_goal_point = n.advertise<geometry_msgs::PointStamped>("/clicked_point", 100);
    // pub_goal_point_list = n.advertise<sensor_msgs::PointCloud2>("/goal_points", 100);
    // for (int i = 1; i < 10; i++)
    //     pub_path[i] = n.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
}

void PoseGraph::setIMUFlag(bool _use_imu)
{
    use_imu = false;

}

void PoseGraph::loadVocabulary(std::string voc_path)
{
     
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void ransac_run1(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> queryKeypoint, std::vector<cv::KeyPoint> objectKeypoint, std::vector<cv::DMatch> &matches_ransac)
{
    std::vector<cv::Point2f> srcPoints(matches.size()),dstPoints(matches.size());
    std::vector<uchar> inliersMask(srcPoints.size());
    for(int i=0;i<matches.size();i++){
        srcPoints[i]=queryKeypoint[matches[i].queryIdx].pt;
        dstPoints[i]=objectKeypoint[matches[i].trainIdx].pt;
        // if(abs(srcPoints[i].y - dstPoints[i].y) < 20)
            // inliersMask[i] = 1;
      
    }

    

    // cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 5, inliersMask);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 716.4751880465594, 0, 650.8935779727392, 0, 716.7212329272082, 368.76884482637536, 0, 0, 1);  // 示例内参矩阵，需要根据实际情况调整
    cv::Mat essential_mat = cv::findEssentialMat(srcPoints, dstPoints, cameraMatrix, cv::RANSAC, 0.9, 1.0, inliersMask);
    for(int i = 0; i < inliersMask.size(); i++){
        if(inliersMask[i]) matches_ransac.push_back(matches[i]);
    }

    std::cout << "inliers count = " << matches_ransac.size() << std::endl;
    std::cout << "all count = " << matches.size() << std::endl;
    std::cout << "keypoint count = " << queryKeypoint.size() << std::endl;
}

void PoseGraph::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    // CYQ 250224
    geometry_msgs::PointStamped current_goal_stamped; // 当前目标点，初始(0,0,0)
    current_goal_stamped.point.x = 0;
    current_goal_stamped.point.y = 0;
    current_goal_stamped.point.z = 0; 
    current_goal_stamped.header.stamp = ros::Time::now();
    current_goal_stamped.header.frame_id = "global";    // 与openvins一致 global world
    //shift to base frame
    Vector3d vio_P_cur;
    Matrix3d vio_R_cur;

    global_index++;
	int loop_index = -1;


    if (load_previous_pose_graph) //检测闭环  and  cur_kf->index-100>0
    {
        printf("used_times: %d lost times: %d\n",used_times,lost_times);
        TicToc tmp_t;
            // global_loop_index = loop_index;
        if(global_loop_index==-1 )
        {
            //  
            loop_index = -1;  
            global_loop_index = 0;
            cout<<"First loop index is "<<loop_index<<" "<<cur_kf->index<<endl;
        }else{
            loop_index = -1;
        }
        if(used_times > 2)
        {
            last_global_loop_index = global_loop_index;
            loop_index = detectLoop(cur_kf, cur_kf->index); 
            global_loop_index = loop_index;
            used_times = 0;
        }
        // if(lost_times > 1)
        // {
        //     loop_index = -1;  
        //     global_loop_index = last_global_loop_index++;
        // }

            
    }
    else
    {
        addKeyFrameIntoVoc(cur_kf); //如果不检测闭环，那就把当前帧加入到词袋当中去，前100帧可以不检测闭环
        keyframelist.push_back(cur_kf);
        return;
    }

	if(loop_index == -1)
    {
        candidate_index_set_.clear();
        printf("======found no loop, use neighbor frames as loop====== \n");
        // for(int i = global_loop_index; i < global_loop_index+5; i++)
        // {

        //     candidate_index_set_.push_back(i);
        // }
        candidate_index_set_.push_back(global_loop_index);
        candidate_index_set_.push_back(global_loop_index+3);
        candidate_index_set_.push_back(global_loop_index+4);
        candidate_index_set_.push_back(global_loop_index+5);
    }
        
    

        printf(" %d detect loop with %d, candidate size %d\n", cur_kf->index, loop_index, candidate_index_set_.size());
        printf("candidates are %d %d %d %d  \n", candidate_index_set_[0],candidate_index_set_[1],candidate_index_set_[2],candidate_index_set_[3]);
        // KeyFrame* old_kf = getKeyFrame(loop_index); //得到闭环帧
        bool getlooped = false;
        
        if(load_previous_pose_graph)
        {
            // if(cur_kf->findConnection(old_kf))
            // {
            //      getlooped = true;
            // }
            //         
            int max_inlier_size = 0;
            std::vector<double> candi_flow_set_(0);
            int inlier_id = 0;
            int bestloopindex = global_loop_index;
            for(int i  = 0; i <  std::min((int)candidate_index_set_.size(),2); i++)
            {
                    //这里使用的是旧的帧到当前帧的投影，是之前设计的，具体motivation已经忘了
                    KeyFrame* cand_kf = getKeyFrame(candidate_index_set_[i]); //得到闭环帧

                    std::vector<cv::DMatch> matches2;
                    std::vector<cv::Point2f> points1;
                    std::vector<cv::Point2f> points2;

                    //wjk new added 0226     match_reshaped中200改为100
                    xfeats0.match_reshaped(cur_kf->xfeat_pts_, cur_kf->xfeat_des_, cand_kf->xfeat_pts_, cand_kf->xfeat_des_, matches2, 100.0, points1, points2);
                    std::vector<cv::KeyPoint> keypoints1, keypoints2;
                    std::vector<cv::DMatch> matches_ransac;
                    for (size_t j = 0; j < points1.size(); j++) {
                            keypoints1.emplace_back(points1[j], 0);
                            keypoints2.emplace_back(points2[j], 0);		
                    }

                    ransac_run1(matches2, keypoints1, keypoints2, matches_ransac); //wjk new added 0226
                    if(matches_ransac.size() > max_inlier_size)
                    {
                        max_inlier_size = matches_ransac.size();
                         bestloopindex= candidate_index_set_[i];
                         inlier_id = i;

                         cur_kf->vis_keypoints1_ = keypoints1;
                         cur_kf->vis_keypoints2_ = keypoints2;
                         cur_kf->vis_matches_ransac_ = matches_ransac;
                    }
                    double flow = 0;
                    int ind = 0;
                        for (size_t k = 0; k < matches_ransac.size(); k++)
                        {
                                cv::Point2f pt1 = keypoints1[matches_ransac[k].queryIdx].pt;
                                cv::Point2f pt2 = keypoints2[matches_ransac[k].trainIdx].pt;
                                flow += pt2.x - pt1.x;
                                ind++;
                                
                        }
                        // flow /= (double)matches_ransac.size();
                        flow /= (double)ind;
                        candi_flow_set_.push_back(flow);
            }
            if(max_inlier_size>20)
            {
                lost_times = 0;
                    // cv::Mat imgMatches;
                    // cv::drawMatches(color_img_, keypoints1, old_kf->color_img_, keypoints2, matches_ransac, imgMatches, cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    
                    // cv::imwrite("/home/wjk/catkin_flowfusion/output/example.png", imgMatches);
                    //wjk new added 0226 如下代码整体替换
                    printf("%d candi match size %d with curr %d id \n", global_loop_index, max_inlier_size, cur_kf->index);
                    if(bestloopindex == global_loop_index)
                    {
                        used_times++;
                    }
                    double flow;
                    flow = candi_flow_set_[inlier_id];
                        global_loop_index = bestloopindex;
                        // double flow  = candi_flow_set_[inlier_id];
                        cur_kf->loop_index = global_loop_index;
                        cur_kf->loop_flow = flow;

                        printf("===flow is==== %f\n",flow);
                        
                        getlooped = true;
                        
                        //分别计算直行，左转，右转的概率
                        double  move_s_prob = 1.0;
                        double  move_left_prob = 1.0;
                        double  move_right_prob = 1.0;
                        double sigma0 = 20;
                        double sigma1 = 30;

                        std::vector<double> flow_set(0);
                        std::vector<Eigen::Vector2f> goal_set;
                        goal_set.clear();
                        goal_set.push_back(Eigen::Vector2f(1,0));
                        goal_set.push_back(Eigen::Vector2f(1,-1));
                        goal_set.push_back(Eigen::Vector2f(1,1));

                        flow_set.push_back(flow);
                        //use local information
                        for(int k = 1; k < 2; k++)
                        {
                                break;
                                KeyFrame* next_kf = getKeyFrame(cur_kf->loop_index+3*k); //得到闭环帧
                                if(next_kf  != nullptr)
                                {
                                    std::vector<cv::DMatch> matches2;
                                    std::vector<cv::Point2f> points1;
                                    std::vector<cv::Point2f> points2;
                                        matches2.clear();
                                        points1.clear();
                                        points2.clear();
                    
                                        xfeats0.match_reshaped(cur_kf->xfeat_pts_, cur_kf->xfeat_des_, next_kf->xfeat_pts_, next_kf->xfeat_des_, matches2, 100.0, points1, points2);
                                        std::vector<cv::KeyPoint> keypoints1, keypoints2;
                                        std::vector<cv::DMatch> matches_ransac;
                                        keypoints1.clear();
                                        keypoints2.clear();
                                        matches_ransac.clear();
                                        
                                        for (size_t j = 0; j < points1.size(); j++) {
                                                keypoints1.emplace_back(points1[j], 0);
                                                keypoints2.emplace_back(points2[j], 0);		
                                        }

                                        ransac_run1(matches2, keypoints1, keypoints2, matches_ransac); //wjk new added 0226
                                        flow = 0;
                                        for (size_t k = 0; k < matches_ransac.size(); k++)
                                        {
                                                cv::Point2f pt1 = keypoints1[matches_ransac[k].queryIdx].pt;
                                                cv::Point2f pt2 = keypoints2[matches_ransac[k].trainIdx].pt;
                                                flow += pt2.x - pt1.x;
                                        }
                                        flow /= (double)matches_ransac.size();

                                        
                                            flow_set.push_back(flow);
                                            break;
                                }else
                                      break;
                        }
                        /*
                        for(int k = 0; k < flow_set.size(); k++)
                        {
                            double flows = flow_set[k];
                             move_s_prob *= exp(-pow(flows,2)/(2*pow(sigma0,2)));
                             move_left_prob *= exp(-pow(flows-50,2)/(2*pow(sigma1,2))) + exp(-pow(flows-100,2)/(2*pow(sigma1,2)))
                             + exp(-pow(flows-150,2)/(2*pow(sigma1,2))) + exp(-pow(flows-200,2)/(2*pow(sigma1,2)));
                             move_right_prob *= exp(-pow(flows+50,2)/(2*pow(sigma1,2))) +  exp(-pow(flows+100,2)/(2*pow(sigma1,2)))
                             + exp(-pow(flows+150,2)/(2*pow(sigma1,2))) +  exp(-pow(flows+200,2)/(2*pow(sigma1,2)));
                        }*/
                        for(int k = 0; k < flow_set.size(); k++)
                        {
                            double w_ = exp(-pow(k,2)/(2*4));
                            double flows = flow_set[k];
                             move_s_prob += w_ * exp(-pow(flows,2)/(2*pow(sigma0,2)));
                             if(flows > 0)
                                    move_left_prob += w_ * (1 - exp(-pow(flows,2)/(2*pow(sigma1,2))));
                            else
                                    move_right_prob += w_ * (1 - exp(-pow(flows,2)/(2*pow(sigma1,2))));
                        }

                        

                        std::vector<double> vec;
                        vec.push_back(move_s_prob);
                        vec.push_back(move_left_prob);
                        vec.push_back(move_right_prob);
                        auto it = max_element(vec.begin(), vec.end()); // 找到最大元素的迭代器
                        int direction = distance(vec.begin(), it); //0: 直行；1: 左传；2: 右转

                        std::cout<<"=====get gaols========== "<<goal_set[direction].transpose()<<std::endl;
                        // CYQ 250224
                        current_goal_stamped.point.x = goal_set[direction][0];
                        current_goal_stamped.point.y = -goal_set[direction][1];
                        current_goal_stamped.header.stamp = ros::Time::now();
                        pub_goal_point.publish(current_goal_stamped); // 循环发布目标点

                        
                        if(cur_kf->index%1==0 and true)
                        {
                                //显示闭环检测以及特征匹配结果
                                cv::Mat imgMatches;
                                KeyFrame* matched_kf = getKeyFrame(global_loop_index); //得到闭环帧
                                cv::drawMatches(cur_kf->color_img_, cur_kf->vis_keypoints1_, matched_kf->color_img_, cur_kf->vis_keypoints2_, cur_kf->vis_matches_ransac_, imgMatches, cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                                // imshow("Xfeat Feature Matches", imgMatches);
                                std::string image_savepath = "/home/luo/cyq/OFVTR/VINS/output/nav/";

                                // double image_time =ros::Time::now().toSec(); 
 
                                // std::string image_time_=to_string(image_time);

                                // image_savepath += image_time_;
                                // image_savepath += "/";



                                // if (fs::exists(image_savepath) && fs::is_directory(image_savepath)) {
                                //     fs::remove_all(image_savepath);
                                // }
                        
                                // // 创建新目录
                                // fs::create_directory(image_savepath);
                                // std::cout << "Directory created successfully." << std::endl;

                                image_savepath += to_string(cur_kf->index);
                                image_savepath += "_";
                                image_savepath += to_string(cur_kf->loop_index);
                                image_savepath += "_";
                                image_savepath += to_string(direction);
                            
                                image_savepath += ".png";
                                cv::imwrite(image_savepath, imgMatches);

                                ofstream outfile;
                                string traj_file = "/home/luo/cyq/OFVTR/VINS/output/trackingdata.txt";
                            
                                //cout<<"traj_file: "<<traj_file.c_str()<<endl;
                                outfile.open(traj_file.c_str(),ios::app);
                                outfile<<cur_kf->index<<" "<<cur_kf->loop_index<<" "<<direction<<" "<<max_inlier_size<<" "<<flow_set[0]<<std::endl;
                                outfile.close();
                                // waitKey(0);
                        }

                    }else
                    {
                        used_times = 10;
                        lost_times++;
                        // current_goal_stamped.point.x = 0.0;
                        // current_goal_stamped.point.y = 0.0;
                        // current_goal_stamped.header.stamp = ros::Time::now();
                        // pub_goal_point.publish(current_goal_stamped); // 循环发布目标点
                        // exit(0);
                    }
                    if(abs(global_loop_index-map_size_)<10)
                    {
                        current_goal_stamped.point.x = 0.0;
                        current_goal_stamped.point.y = 0.0;
                        current_goal_stamped.header.stamp = ros::Time::now();
                        pub_goal_point.publish(current_goal_stamped); // 循环发布目标点
                        exit(0);
                    }
                        
            }
            //TODO 如果找到了闭环，那么就把当前的flow，和后续的帧的flow都发布出去。然后导航根据收到的信息来选择最优的第一步。导航可以晚上做。
           

        
	
        
	printf("next frame!");
}


void PoseGraph::loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    
    addKeyFrameIntoVoc(cur_kf);
    
    keyframelist.push_back(cur_kf);

}

KeyFrame* PoseGraph::getKeyFrame(int index)
{
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index)
{
    TicToc tmp_t;
    // std::cout<<"first query; then add this frame into database!"<<endl;
    QueryResults ret;
    TicToc t_query;

    db.query_length_ = map_size_ - 1;
    int max_id;
    //max_id = map_size_; //wjk new added 0226
    max_id = global_loop_index+30;//wjk new added 0226
    db.min_id_ = max(0,global_loop_index - 5);
    db.query(keyframe->brief_descriptors, ret, 4, max_id);
    
        
    printf("query time: %f\n", t_query.toc());

    TicToc t_add;
    
    printf("add feature time: %f\n", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
 

    // a good match with its nerghbour
    std::cout<<"ret ID"<<ret[0].Id<<" "<<ret[1].Id<<" "<<ret[2].Id<<" "<<ret[3].Id<<endl;
    std::cout<<"ret Score"<<ret[0].Score<<" "<<ret[1].Score<<" "<<ret[2].Score<<" "<<ret[3].Score<<endl;
    if (ret.size() >= 1 &&ret[0].Score > 0.018) //0.05
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {          
                find_loop = true;
                int tmp_index = ret[i].Id;
                
            }

        }

    if (find_loop) // && frame_index > 50
    {
        candidate_index_set_.clear();
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if(ret[i].Score > 0.015)
               candidate_index_set_.push_back(ret[i].Id);
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }

        return min_index;
    }
    else
        return -1;

}

void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
    db.add(keyframe->brief_descriptors);
}



void PoseGraph::savePoseGraph()
{
    TicToc tmp_t;
    FILE *pFile;
     if(load_previous_pose_graph)
         POSE_GRAPH_SAVE_PATH = POSE_GRAPH_SAVE_PATH + "expanded/";
    printf("pose graph path: %s\n",POSE_GRAPH_SAVE_PATH.c_str());
    printf("pose graph saving... %d keyframes \n", keyframelist.size());
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    string feature_path;
    pFile = fopen (file_path.c_str(),"w");
    //fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
     list<KeyFrame*>::iterator it;
        // string traj_file1 = "/home/wjk/catkin_ws_vinsfusion/output/eulerangle6DOF.txt";
        // outfile.open(traj_file1.c_str()); //, ios::app
         for (it = keyframelist.begin(); it != keyframelist.end(); it++)
        {
        
            std::string image_path, descriptor_path, brief_path, keypoints_path;
            if (true)
            {
                image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.png";
                imwrite(image_path.c_str(), (*it)->image);
                image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.jpg";
                imwrite(image_path.c_str(), (*it)->color_img_);
                // printf("%d -th image: (%d,%d)\n",(*it)->index,(*it)->raw_image.cols,(*it)->raw_image.rows);
            }
            //! 这里要注意，特征点都是在里程计坐标系下的，所以要保留最原始的里程计位姿，才能使用特征点。
            
            cv::Mat feats0_reshaped = (*it)->xfeat_des_.clone();
            cv::Mat mkpts0_reshaped = (*it)->xfeat_pts_.clone();
     
            feature_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "pts.txt";
            cv::FileStorage fs_pt(feature_path.c_str(), cv::FileStorage::WRITE);
            fs_pt  << "pt"<< mkpts0_reshaped;
            fs_pt.release();

            feature_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "descs.txt";
            cv::FileStorage fs_des(feature_path.c_str(), cv::FileStorage::WRITE);
            fs_des  << "des"<< feats0_reshaped;
            fs_des.release();
        
             
            fprintf (pFile, " %d %f %f %f %d %d\n",(*it)->index, (*it)->time_stamp, 
                                        (*it)->flow_,(*it)->loop_flow, (*it)->loop_index, 
                                        (int)(*it)->keypoints.size());
        }
    fclose(pFile);

    printf("save pose graph time: %f s\n", tmp_t.toc() / 1000);
}
void PoseGraph::loadPoseGraph()
{
    load_previous_pose_graph = true;
    TicToc tmp_t;
    FILE * pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    printf("load pose graph from: %s \n", file_path.c_str());
    printf("pose graph loading...\n");
    pFile = fopen (file_path.c_str(),"r");
    if (pFile == NULL)
    {
        printf("load previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
        return;
    }
    int index;
    double time_stamp;
    double flow, loop_flow;
    int loop_index;
    int keypoints_num;
    int cnt = 0;
    
    string feature_path;
    while (fscanf(pFile,"%d %lf %lf %lf  %d %d", &index, &time_stamp, 
                                    &flow, &loop_flow,  
                                    &loop_index, &keypoints_num) != EOF) 
    {
        cv::Mat image,color_image;
        std::string image_path, descriptor_path;
        if (true)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
            image = cv::imread(image_path.c_str(), 0);
            image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.jpg";
            color_image = cv::imread(image_path.c_str(), 1);
        }
        
        cv::Mat mkpts0_raw, feats0_raw, sc0_raw;
        //  xfeats0.detectAndCompute(color_image, mkpts0_raw, feats0_raw, sc0_raw);
        feature_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "pts.txt";
        cv::FileStorage fs_pt(feature_path.c_str(), cv::FileStorage::READ);
        fs_pt ["pt"]>>mkpts0_raw;
        fs_pt.release();

        feature_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "descs.txt";
        cv::FileStorage fs_des(feature_path.c_str(), cv::FileStorage::READ);
        fs_des["des"]>> feats0_raw;
        fs_des.release();

        
        KeyFrame* keyframe = new KeyFrame(time_stamp, index, image, mkpts0_raw, feats0_raw, flow, 0);
        keyframe->color_img_ = color_image.clone();
        keyframe->loop_flow = loop_flow;
        keyframe->loop_index = loop_index;
        loadKeyFrame(keyframe, 0);
        
        cnt++;
    }
    // pose_cloud->width = pose_cloud->points.size();
    // pose_cloud->height = 1; // 表示这是一个无序点云
    // pose_cloud->is_dense = true; // 如果没有NaN或Inf值，则设置为true
    fclose (pFile);
    map_size_ = keyframelist.size();
    printf("load pose graph time:%d  global_keyframe %d keyframes %f s\n", map_size_,global_index, tmp_t.toc()/1000);
    
    base_sequence = 0;
}

