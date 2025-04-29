//
// Created by Chen Tian on 2022/8/8.
//

#include <string>
#include "tinyxml2.h"
#include "camera.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace tinyxml2;
using namespace std;
using namespace cv;


void get_VSTcamera_attribute(const char* calibrate_xml, const char* camera_name, Camera& camera) {
	XMLDocument doc;
	doc.LoadFile(calibrate_xml);
	if (doc.ErrorID() != 0) {
		std::cout << "Load File Error: " << doc.ErrorID() << std::endl;
	}
	const char * cam_name;
	XMLElement* camera_element = doc.FirstChildElement("DeviceConfiguration")->FirstChildElement("Camera");
	XMLElement* camera_element_end = doc.FirstChildElement("DeviceConfiguration")->LastChildElement("Camera");
	std::cout << "camera name: " << camera_name << std::endl;
	for (XMLElement* iter = camera_element; iter != camera_element_end->NextSiblingElement(); iter = iter->NextSiblingElement()) {
		iter->QueryAttribute("cam_name", &cam_name);
		XMLElement* calibrate_element = iter->FirstChildElement("Calibration");
		XMLElement* rig_element = iter->FirstChildElement("Rig");
		if (std::strcmp(cam_name, camera_name) == 0) {
			const char* size;
			calibrate_element->QueryAttribute("size", &size);
			camera.set_size(size); // 设置图片大小
			const char*  focal_length; const char* principal_point;
			calibrate_element->QueryAttribute("focal_length", &focal_length);
			calibrate_element->QueryAttribute("principal_point", &principal_point);
			camera.set_intrinsic(focal_length, principal_point);
			std::cout << "intrinsic: " << camera.intrinsic << std::endl;
			const char* radial_distortion;
			calibrate_element->QueryAttribute("radial_distortion", &radial_distortion);
			camera.set_radial_distortion(radial_distortion);
			std::cout << "distortion: " << camera.distortion.t() << std::endl;
			const char* translation; const char* rowMajorRotationMat;
			rig_element->QueryAttribute("translation", &translation);
			rig_element->QueryAttribute("rowMajorRotationMat", &rowMajorRotationMat);
			camera.set_transform(translation, rowMajorRotationMat);
			std::cout << "transform: " << camera.transform << std::endl;
		}
	}
	std::cout << "--------------------------------------" << std::endl;
}

void get_img_lists(const char* img_info_xml, std::vector<std::string> &lists) {
	XMLDocument doc;
	doc.LoadFile(img_info_xml);
	if (doc.ErrorID() != 0) {
		std::cout << "Load File Error: " << doc.ErrorID() << std::endl;
	}
	XMLElement* frame_element = doc.FirstChildElement("Sequence")->FirstChildElement("Frameset")->FirstChildElement("Frame");
	XMLElement* frame_element_end = doc.FirstChildElement("Sequence")->FirstChildElement("Frameset")->LastChildElement("Frame");
	for (XMLElement* iter = frame_element; iter != frame_element_end->NextSiblingElement(); iter = iter->NextSiblingElement()) {
		const char *filename;
		iter->QueryAttribute("filename", &filename);
		lists.push_back(std::string(filename));
	}

}

int main(int argc, char **argv) {
	std::string  data_path,data_type;
	data_path = string(argv[1]);
	data_type = string(argv[2]);
	std::string  calibrate_xml = data_path + "/device_calibration.xml";
	fstream _file;
	_file.open(calibrate_xml, ios::in);
	if (!_file)
	{
		cout << calibrate_xml << "file not exist,data path error" << endl;
		return -1;
	}
    
	std::string img_info_xml="";;
	string left_name, right_name,cameraID;
    bool is_tracking = true;
	if (data_type.compare("tracking") == 0)
	{
		img_info_xml = data_path + "/Camera8/MetaInfo.xml";
		left_name = "trackingA";
		right_name = "trackingB";
		cameraID = "Camera8";
        is_tracking = true;
	}
	if (data_type.compare("ctr-tracking") == 0)
	{
		img_info_xml = data_path + "/Camera9/MetaInfo.xml";
		left_name = "ctr-trackingA";
		right_name = "ctr-trackingB";
		cameraID = "Camera9";
        is_tracking = true;
	}
	if (data_type.compare("rgb") == 0)
	{
		img_info_xml = data_path + "/Camera6/MetaInfo.xml";
		left_name = "rgb-left";
		right_name = "rgb-right";
		cameraID = "Camera6";
        is_tracking = false;
	}
	if ("" == img_info_xml)
	{
		cout << "input data type name error" << endl;
		return -1;
	}
    if ((is_tracking==false&&cameraID!="Camera6")||(is_tracking==true&&(cameraID!="Camera8"&&cameraID!="Camera9")))
    {
        cout << "is_tracking camera_type not matching,is_tracking:" <<is_tracking<<",cameraID:"<<cameraID<< endl;
		return -1;
    }

	std::string command = "mkdir -p " + data_path + "/left/ " + data_path + "/right/ " + data_path + "/merge/ ";
	system(command.c_str());
    cout<<"calibrate_xml:"<<calibrate_xml<<endl;
    cout<<"image path:"<<data_path + "/"+ cameraID<<endl;
	Camera camera_left(is_tracking), camera_right(is_tracking);
	get_VSTcamera_attribute(calibrate_xml.c_str(), left_name.c_str(), camera_left);
	get_VSTcamera_attribute(calibrate_xml.c_str(), right_name.c_str(), camera_right);
	cv::Mat T = camera_right.transform*camera_left.transform.inv();
	cv::Mat R = T.rowRange(0, 3).colRange(0, 3).clone();
	cv::Mat t = T.rowRange(0, 3).colRange(3, 4).clone();
	cv::Mat Rl, Rr, Pl, Pr, Q, mapLx, mapLy, mapRx, mapRy;
	if(!is_tracking)
	{
		cv::stereoRectify(camera_left.intrinsic, camera_left.distortion, camera_right.intrinsic, camera_right.distortion,
			camera_left.camera_size, R, t, Rl, Rr, Pl, Pr, Q);
		cv::initUndistortRectifyMap(camera_left.intrinsic, camera_left.distortion, Rl, Pl, camera_left.camera_size, CV_32FC1, mapLx, mapLy);
		cv::initUndistortRectifyMap(camera_right.intrinsic, camera_right.distortion, Rr, Pr, camera_right.camera_size, CV_32FC1, mapRx, mapRy);
	}
	else
	{
		cv::fisheye::stereoRectify(camera_left.intrinsic, camera_left.distortion, camera_right.intrinsic, camera_right.distortion,
			camera_left.camera_size, R, t, Rl, Rr, Pl, Pr, Q, fisheye::CALIB_ZERO_DISPARITY, camera_left.camera_size);
		cv::fisheye::initUndistortRectifyMap(camera_left.intrinsic, camera_left.distortion, Rl, Pl, camera_left.camera_size, CV_32FC1, mapLx, mapLy);
		cv::fisheye::initUndistortRectifyMap(camera_right.intrinsic, camera_right.distortion, Rr, Pr, camera_right.camera_size, CV_32FC1, mapRx, mapRy);
	}
	std::vector<std::string> lists;
	get_img_lists(img_info_xml.c_str(), lists);
	for (int i = 0; i < lists.size(); ++i) {
		cout << data_path << " process " << i << " frame" << endl;
		std::string path = data_path + "/"+ cameraID +"/" + lists[i];
		cv::Mat stereo = cv::imread(path);
		cv::Mat left = stereo.colRange(0, stereo.cols / 2).clone();

		cv::Mat left_calib, right_calib, left_c, right_c;
        if(!is_tracking)
        {
            cv::undistort(left, left_c, camera_left.intrinsic, camera_left.distortion);
        }
        else
        {
		    cv::fisheye::undistortImage(left, left_c, camera_left.intrinsic, camera_left.distortion, camera_left.intrinsic, camera_left.camera_size);
        }


		cv::Mat right = stereo.colRange(stereo.cols / 2, stereo.cols).clone();

        if(!is_tracking)
        {
		    cv::undistort(right, right_c, camera_right.intrinsic, camera_right.distortion);
        }
        else
        {
		    cv::fisheye::undistortImage(right, right_c, camera_right.intrinsic, camera_right.distortion, camera_right.intrinsic, camera_right.camera_size);
        }

		cv::remap(left, left_calib, mapLx, mapLy, cv::INTER_LINEAR);
		cv::remap(right, right_calib, mapRx, mapRy, cv::INTER_LINEAR);
		std::stringstream ss_left, ss_right, ss_merge;
		ss_left << data_path + "/left/" << std::setw(8) << std::setfill('0') << i << ".png";
		ss_right << data_path + "/right/" << std::setw(8) << std::setfill('0') << i << ".png";
		if(is_tracking)
		{
			flip(left_calib, left_calib, 0);
			flip(left_calib, left_calib, 1);
			flip(right_calib, right_calib, 0);
			flip(right_calib, right_calib, 1);
		}
		cv::imwrite(ss_left.str(), left_calib);
		cv::imwrite(ss_right.str(), right_calib);

		cv::Mat show, show1;
		cv::hconcat(left_calib, right_calib, show);
		for (int i = 0; i < show.rows; i += 32)
			line(show, Point(0, i), Point(show.cols, i), Scalar(0, 255, 0), 1, 8);
		cv::Mat show_resize;
		cv::resize(show, show_resize, Size(1000, 500), 0, 0, INTER_CUBIC);
		ss_merge << data_path + "/merge/" << std::setw(8) << std::setfill('0') << i << ".png";
		cv::imwrite(ss_merge.str(), show_resize);

	}
	std::cout << "calibrate left image save to " << data_path + "/left/" << std::endl;
	std::cout << "calibrate right image save to " << data_path + "/right/" << std::endl;

	return 0;
}