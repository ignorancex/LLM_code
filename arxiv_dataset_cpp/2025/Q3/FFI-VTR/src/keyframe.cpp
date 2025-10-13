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

#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


KeyFrame::KeyFrame(double _time_stamp, int _index,  cv::Mat _image,
			cv::Mat mkpts0_raw_, cv::Mat feats0_raw_, double flow, int _sequence)
{
	  time_stamp = _time_stamp;
	  index = _index;
	  image = _image.clone();
	  xfeat_pts_ = mkpts0_raw_.clone();
	  xfeat_des_ = feats0_raw_.clone();
	  sequence = _sequence;
	  flow_ =  flow;
	  computeBRIEFPoint(); //提取描述子用于场景闭环检测
}


KeyFrame::KeyFrame(double _time_stamp, int _index,  cv::Mat _image,std::vector<KeyPoint> keypoints_orb,
	cv::Mat descriptors_orb, double flow, int _sequence)
{
      time_stamp = _time_stamp;
	  index = _index;
	  image = _image.clone();
	  keypoints_orb_ = keypoints_orb;
	  descriptors_orb_ = descriptors_orb.clone();
	  sequence = _sequence;
	  flow_ =  flow;
	  computeBRIEFPoint(); //提取描述子用于场景闭环检测

 }

 KeyFrame::KeyFrame(double _time_stamp, int _index,  cv::Mat _image, double flow, int _sequence)
{
      time_stamp = _time_stamp;
	  index = _index;
	  image = _image.clone();
	  sequence = _sequence;
	  flow_ =  flow;
	  computeBRIEFPoint(); //提取描述子用于场景闭环检测
 }





void KeyFrame::getCameraPose(Eigen::Vector3d &T_w_c, Eigen::Matrix3d &R_w_c)
{
	//   R_w_c = qic * origin_vio_R.inverse();
	//   T_w_c = -qic * origin_vio_R.inverse()*origin_vio_T - qic * tic;
	R_w_c = qic * R_w_i.transpose();
	T_w_c = -qic * R_w_i.transpose()*T_w_i - qic * tic;
}

void KeyFrame::computeWindowBRIEFPoint()
{
	// !计算当前帧的有效三维特征点的描述子
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
    /*
	std::vector<int> levels0(point_2d_uv.size(), 0);
     cv::Mat currDescriptors0;
    currORBDescriptor0_ptr->computeDescriptors(point_2d_uv, levels0, windowOrbDescriptors);*/
}

void KeyFrame::computeBRIEFPoint()
{
	//! 在当前帧重新提取密集的描述子
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 5; // corner detector response threshold
	if(1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}

	extractor(image, keypoints, brief_descriptors);

}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 716.47;  // 460.0
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}


void ransac_run0(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> queryKeypoint, std::vector<cv::KeyPoint> objectKeypoint, std::vector<cv::DMatch> &matches_ransac)
{
    std::vector<cv::Point2f> srcPoints(matches.size()),dstPoints(matches.size());
    std::vector<int> inliersMask(srcPoints.size());
    for(int i=0;i<matches.size();i++){
        srcPoints[i]=queryKeypoint[matches[i].queryIdx].pt;
        dstPoints[i]=objectKeypoint[matches[i].trainIdx].pt;
        if(abs(srcPoints[i].y - dstPoints[i].y) < 20)
            inliersMask[i] = 1;
        else
            inliersMask[i] = 0;
    }

    

    // cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 5, inliersMask);

    for(int i = 0; i < inliersMask.size(); i++){
        if(inliersMask[i]) matches_ransac.push_back(matches[i]);
    }

    std::cout << "inliers count = " << matches_ransac.size() << std::endl;
    std::cout << "all count = " << matches.size() << std::endl;
    std::cout << "keypoint count = " << queryKeypoint.size() << std::endl;
}

//wjk added 0226
// double KeyFrame::computeflow(KeyFrame* old_kf)
// {
// 	std::vector<cv::DMatch> matches2;
//     std::vector<cv::Point2f> points1;
//     std::vector<cv::Point2f> points2;
//     xfeats_.match_reshaped(xfeat_pts_, xfeat_des_, old_kf->xfeat_pts_, old_kf->xfeat_des_, matches2, 100.0, points1, points2);

//     std::vector<cv::KeyPoint> keypoints1, keypoints2;
//     std::vector<cv::DMatch> matches_ransac;
//     for (size_t i = 0; i < points1.size(); i++) {
//             keypoints1.emplace_back(points1[i], 0);
//             keypoints2.emplace_back(points2[i], 0);		
//     }

//     ransac_run0(matches2, keypoints1, keypoints2, matches_ransac);
//     double flow = 0;
//     for (size_t i = 0; i < matches_ransac.size(); i++)
//     {
//          cv::Point2f pt1 = keypoints1[matches_ransac[i].queryIdx].pt;
//         cv::Point2f pt2 = keypoints2[matches_ransac[i].trainIdx].pt;
//         flow += pt2.x - pt1.x;
//     }
//         flow /= (double)matches_ransac.size();
//         return flow;
// }

bool KeyFrame::findConnection(KeyFrame* old_kf)
{
        Ptr<ORB> orb1= ORB::create();
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        orb1->detectAndCompute(image, noArray(), keypoints1, descriptors1);
        orb1->detectAndCompute(old_kf->image, noArray(), keypoints2, descriptors2);

        BFMatcher matcher(NORM_HAMMING);
        vector<vector<DMatch>> knnMatches;
        matcher.knnMatch(descriptors1,  descriptors2, knnMatches, 2);

        // 应用Lowe's ratio test筛选匹配
        vector<DMatch> goodMatches;
        const float ratio_thresh = 0.75f;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }
        if(goodMatches.size() > MIN_LOOP_NUM )
        {
            double flow = 0;
            for (size_t i = 0; i < goodMatches.size(); i++)
            {
                    cv::Point2f pt1 = keypoints1[goodMatches[i].queryIdx].pt;
                    cv::Point2f pt2 = keypoints2[goodMatches[i].trainIdx].pt;
                    if(abs(pt2.y - pt1.y)<5)
                        flow += pt2.x - pt1.x;
             }
            flow /= (double)goodMatches.size();
            loop_index = old_kf->index;
            loop_flow = flow;
            printf("flow is %f\n",flow);
            Mat imgMatches;
            drawMatches(image, keypoints1, old_kf->image, keypoints2, goodMatches, imgMatches,
                    Scalar::all(-1), Scalar::all(-1), vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            // 显示结果
            imshow("ORB Feature Matches", imgMatches);
            waitKey(0);
            
            return true;
        }
        else
            return false;

    /*
	std::vector<cv::DMatch> matches2;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    xfeats_.match(xfeat_pts_, xfeat_des_, old_kf->xfeat_pts_, old_kf->xfeat_des_, matches2, 200.0, points1, points2);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches_ransac;
    for (size_t i = 0; i < points1.size(); i++) {
            keypoints1.emplace_back(points1[i], 0);
            keypoints2.emplace_back(points2[i], 0);		
    }

    ransac_run0(matches2, keypoints1, keypoints2, matches_ransac);
    
	// cv::Mat imgMatches;
	// cv::drawMatches(color_img_, keypoints1, old_kf->color_img_, keypoints2, matches_ransac, imgMatches, cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // cv::imwrite("/home/wjk/catkin_flowfusion/output/example.png", imgMatches);

	if(matches_ransac.size() > MIN_LOOP_NUM )
	{
		  double flow = 0;
           for (size_t i = 0; i < matches_ransac.size(); i++)
            {
                cv::Point2f pt1 = keypoints1[matches_ransac[i].queryIdx].pt;
                cv::Point2f pt2 = keypoints2[matches_ransac[i].trainIdx].pt;
                flow += pt2.x - pt1.x;
            }
        flow /= (double)matches_ransac.size();
		loop_index = old_kf->index;
		loop_flow = flow;
        


		return true;
	}
	else
	    return false;*/
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
	
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopCamRelativeQ()
{
    return Eigen::Quaterniond(loop_cam_info(3), loop_cam_info(4), loop_cam_info(5), loop_cam_info(6));
}

Eigen::Vector3d KeyFrame::getLoopCamRelativeT()
{
    return Eigen::Vector3d(loop_cam_info(0), loop_cam_info(1), loop_cam_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}


double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


