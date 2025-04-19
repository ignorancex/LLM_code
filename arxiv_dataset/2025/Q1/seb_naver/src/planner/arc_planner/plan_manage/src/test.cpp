#include <memory>
#include <set>
#include <string>
#include <thread>
#include <iostream>
#include <sstream>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <cmath>
#include <algorithm>
#include <ros/console.h>
#include <tools/tic_toc.hpp>
Eigen::Matrix<double, 25, 25> xl = Eigen::MatrixXd::Random(25,25);
Eigen::Matrix<double, 25, 25> yl = Eigen::MatrixXd::Random(25,25);
void fnp(std::vector<Eigen::MatrixXd> x, Eigen::Vector2d inis, Eigen::MatrixXd env){
    Eigen::Vector2d lasts = inis;
    Eigen::Matrix<double, 100,2> opStates;
    opStates.row(0) = inis;
    for(int i = 0; i < 98; i++){
        Eigen::Vector2d lastgrid;
        lastgrid.array()  =  floor((lasts.array()+10.0) / 0.1);
        Eigen::MatrixXd maskemap = env.block<25,25>(int(lastgrid[0])-12,int(lastgrid[1])-12);
        Eigen::MatrixXd prbmap = x[i] * maskemap;
        double maxitem = prbmap.maxCoeff();
        prbmap = prbmap.array() - maxitem;
        prbmap = prbmap.array().exp();
        double norm = prbmap.sum();
        prbmap = prbmap / norm;
		prbmap.array() += 1.0e-16;
        opStates(i+1,0)=(prbmap * xl).mean();
		opStates(i+1,1)=(prbmap * yl).mean();
		lasts = opStates.row(i+1);
    }
    Eigen::Matrix<double, 102,2> labelTable;
    labelTable.row(0) = opStates.row(0);
    labelTable.block<100,2>(1,0) = opStates;
    labelTable.row(101) = opStates.row(99);
    for(int i = 0; i < 98; i++){
        Eigen::MatrixXd poses = labelTable.block<5,2>(i,0);
        opStates.row(i+1) = poses.colwise().mean();
    }
    return;

}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh("~");
    std::vector<Eigen::MatrixXd> x;
    Eigen::Vector2d inis = Eigen::Vector2d::Random();
    Eigen::MatrixXd env = Eigen::MatrixXd::Random(200,200);
    for(int i = 0; i < 98; i++){
        Eigen::MatrixXd xitem = Eigen::MatrixXd::Random(25,25);
        x.push_back(xitem);
    }
    TicToc time_profile_tool_;
    time_profile_tool_.tic();
    for(int i = 0; i <100; i++){
        double t1 = ros::Time::now().toSec();
        fnp(x, inis, env);
        double t2 = ros::Time::now().toSec();
        std::cout << "runTime: " << 1000.0*(t2-t1) << " ms" << std::endl;
    }



    ros::spin();
    return 0;
}
