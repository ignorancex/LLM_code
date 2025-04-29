//
// Created by Chen Tian on 2022/8/8.
//

#include "camera.hpp"

std::vector<std::string> Camera::split(std::string str, std::string pattern)
{
    std::vector<std::string> ret;
    if (pattern.empty()) return ret;
    size_t start = 0, index = str.find_first_of(pattern, 0);
    while (index != str.npos)
    {
        if (start != index)
            ret.push_back(str.substr(start, index - start));
        start = index + 1;
        index = str.find_first_of(pattern, start);
    }
    if (!str.substr(start).empty())
        ret.push_back(str.substr(start));
    return ret;
}


void Camera::set_size(const char* size_string) {
    char* size = (char* )size_string;
    size = std::strtok(size, " ");
    if(size){
        camera_size.width = std::atoi(size);
    }else{
        std::cout << "Can not get camera size" << std::endl;
        return;
    }
    size = std::strtok(NULL, " ");
    if(size){
        camera_size.height = std::atoi(size);
    }else{
        std::cout << "Can not get camera size" << std::endl;
        return;
    }
}

void Camera::set_intrinsic(const char *focal_length_string, const char *principal_point_string) {
    char* focal_length_str = (char*) focal_length_string;
    char* principal_point_str = (char*) principal_point_string;
    focal_length_str = std::strtok(focal_length_str, " ");
    if(focal_length_str){
        intrinsic.at<double>(0,0) = std::stod(std::string(focal_length_str));
    }else{
        std::cout << "Can not get focal length! " << std::endl;
        return;
    }
    focal_length_str = std::strtok(NULL, " ");
    if(focal_length_str){
        intrinsic.at<double>(1,1) = std::stod(std::string(focal_length_str));
    }else{
        std::cout << "Can not get focal length! " << std::endl;
        return;
    }

    principal_point_str = std::strtok(principal_point_str, " ");
    if(principal_point_str){
        intrinsic.at<double>(0,2) = std::stod(std::string(principal_point_str));
    }else{
        std::cout << "Can not get principal point! " << std::endl;
        return;
    }
    principal_point_str = std::strtok(NULL, " ");
    if(principal_point_str){
        intrinsic.at<double>(1,2) = std::stod(std::string(principal_point_str));
    }else{
        std::cout << "Can not get principal point! " << std::endl;
        return;
    }
}

void Camera::set_radial_distortion(const char *radial_distortion_string) {
    std::string radial_distortion_str = std::string(radial_distortion_string);
    std::vector<std::string> ks = split(radial_distortion_str, " ");
    if(ks.size() != 6){
        std::cout << "Can not get radial distortion! " << std::endl;
    }
    if(!m_is_tracking)
    {
        for (int i = 0; i < 2; ++i) {
            distortion.at<double>(i,0) = std::stod(ks[i]);
        }
        for (int i = 4; i < 8; ++i) {
            distortion.at<double>(i,0) = std::stod(ks[i - 2]);
        }
    }
    else
    {
        for (int i = 0; i < 4; ++i) {
            distortion.at<double>(i, 0) = std::stod(ks[i]);
        }
    }

}

void Camera::set_transform(const char *translation_string, const char *rotation_string) {
    std::string translation_str = std::string(translation_string);
    std::string rotation_str = std::string(rotation_string);
    std::vector<std::string> ts = split(translation_str, " ");
    std::vector<std::string> rs = split(rotation_str, " ");
    if(ts.size() != 3 || rs.size() != 9){
        std::cout << "Can not get Rig data! " << std::endl;
    }
    for (int y = 0; y < 3; ++y) {
        translation.at<double>(y,0) = std::stod(ts[y]);
        transform.at<double>(y,3) = std::stod(ts[y]);
        for (int x = 0; x < 3; ++x) {
            rotation.at<double>(y,x) = std::stod(rs[y*3+x]);
            transform.at<double>(y,x) = std::stod(rs[y*3+x]);
        }
    }
}