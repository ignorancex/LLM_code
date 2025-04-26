#include <plan_manage/plan_manage.h>
#include <tf/tf.h>
#include <tools/tic_toc.hpp>
#include <fstream>
#include <random>
#include <thread>
using namespace plan_manage;
using namespace std;
void PlanManager::init(ros::NodeHandle& nh, std::shared_ptr<se2_grid::SE2Grid> se2_grid_map, std::shared_ptr<se2_grid::SE2Grid> sdf_grid_map){
    vis_tool.reset(new visualization::Visualization(nh));
    config_.reset(new Config(nh));
    se2_grid_map_ = se2_grid_map;
    sdf_grid_map_ = sdf_grid_map;
    hasTarget = false;
    hasOdom = false;
    cudaWarmTimer = nh.createTimer(ros::Duration(0.01), &PlanManager::warm,this);
    vis_tool->registe<nav_msgs::Path>("/visualization/kinoPath");
    vis_tool->registe<nav_msgs::Path>("/visualization/AstarPath");
    vis_tool->registe<nav_msgs::Path>("/visualization/optTraj");
    vis_tool->registe<nav_msgs::Path>("/visualization/optTraj_kino");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/debugTraj");
    vis_tool->registe<nav_msgs::Path>("/visualization/NewTraj");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/waittoRefine");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/optArrowTraj");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/optArrowTraj2");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/start_goal");
    vis_tool->registe<decomp_ros_msgs::PolyhedronArray>("/visualization/sfc");
    vis_tool->registe<visualization_msgs::Marker>("/visualization/robot_once");  
    vis_tool->registe<visualization_msgs::Marker>("/visualization/robots_front_end");  
    vis_tool->registe<decomp_ros_msgs::PolyhedronArray>("/visualization/sfc_once");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj2");
    vis_tool->registe<nav_msgs::Path>("/visualization/debugRefinedTraj2");
    vis_tool->registe<nav_msgs::Path>("/visualization/debugSe2Traj");
    vis_tool->registe<visualization_msgs::MarkerArray>("/visualization/arrowTraj");
    vis_tool->registe<sensor_msgs::PointCloud2>("/visualization/wapoints_nn");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_nn");
    vis_tool->registe<visualization_msgs::Marker>("/visualization/fullshapeTraj_nn");
    vis_tool->registe<sensor_msgs::PointCloud2>("/visualization/wapoints_kinoastar");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_kinoastar");
    vis_tool->registe<nav_msgs::Path>("/visualization/refinedTraj_dftpav");
    vis_tool->registe<visualization_msgs::Marker>("/visualization/fullshapeTraj_kinoastar");
    pieceTime = config_->pieceTime;
    kino_path_finder_.reset(new path_searching::KinoAstar);
    kino_path_finder_->init(config_, nh);  
    kino_path_finder_->intialMap(sdf_grid_map_);
    Eigen::Vector2d map_XY{config_->mapX, config_->mapY};
    trajCmdPub = nh.advertise<mpc_controller::DPtrajContainer>("/planner/dptrajectory", 1);  
}
void PlanManager::warm(const ros::TimerEvent &){
    return;
}
bool PlanManager::process(){
    if(!hasTarget){
        std::cout<<"No odom or target "<<" target "<<hasTarget<<" odom "<<hasOdom<<std::endl;
        return false;
    } 
    ROS_WARN("Triggering------------------------------------ we begin to plan a trajectory!");
    hasTarget = false;
    begin_time = ros::Time::now();
    vector<Eigen::Vector3d> sampleTraj;
    path_searching::KinoTrajData kino_trajs_;
    std::vector<Eigen::Vector3d> visKinoPath;

    kino_path_finder_->reset();
    Eigen::Vector4d iniFs, finFs;
    iniFs = start_state_;
    finFs = end_state_;
    std::cout<<"start_state_ "<<start_state_.transpose()<<std::endl;
    std::cout<<"end_state_ "<<end_state_.transpose()<<std::endl;
    std::vector<Eigen::Vector4d> start_goal;
    start_goal.push_back(iniFs);
    start_goal.push_back(finFs);
    vis_tool->visualize_balls_wxx(start_goal, "/visualization/start_goal");
    double model_time;
    int status = kino_path_finder_->search(iniFs, Eigen::Vector2d::Zero(), finFs, model_time, true);
    if(status == 0){
        ROS_WARN("No path found!");
        return false;
    }
        
    std::vector<Eigen::Vector3d> ts = kino_path_finder_->getRoughSamples();
    sampleTraj.insert(sampleTraj.end(), ts.begin(), ts.end());
    kino_path_finder_->getKinoNode(kino_trajs_, sampleTraj);
    for(int i = 0; i < sampleTraj.size(); i++){
        Eigen::Vector3d pos;
        pos.head(2) = sampleTraj[i].head(2);
        pos[2] = 0.2;
        visKinoPath.push_back(pos);
    }
    vis_tool->visualize_path(visKinoPath, "/visualization/kinoPath");
    int segnum = kino_trajs_.size();    
    std::vector<int> refined_singuals; refined_singuals.resize(segnum);
    Eigen::VectorXd refined_rt;
    refined_rt.resize(segnum);
    std::vector<Eigen::MatrixXd> refined_inPs_container;
    refined_inPs_container.resize(segnum);
    std::vector<Eigen::Vector2d> refined_gearPos;
    std::vector<double> refined_angles; 
    refined_gearPos.resize(segnum - 1); refined_angles.resize(segnum - 1);
    double basetime = 0.0;
    std::vector<int> pnums;
    for(int i = 0; i < segnum; i++){
        double timePerPiece = pieceTime;
        path_searching::FlatTrajData kino_traj = kino_trajs_.at(i);
        refined_singuals[i] = kino_traj.singul;
        std::vector<Eigen::Vector3d> pts = kino_traj.traj_pts;
        int piece_nums;
        double initTotalduration = 0.0;
        for(const auto pt : pts){
            initTotalduration += pt[2];
        }
        piece_nums = std::max(int(initTotalduration / timePerPiece + 0.5),1);
        double dt = initTotalduration / piece_nums; 
        refined_rt[i] = (initTotalduration / piece_nums);
        pnums.push_back(piece_nums);

        refined_inPs_container[i].resize(2, piece_nums - 1);
        for(int j = 0; j < piece_nums - 1; j++ ){
            double t = basetime + (j+1)*dt;
            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(t);
            refined_inPs_container[i].col(j) = pos.head(2);
        }
        if(i >=1){
            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(basetime);
            refined_gearPos[i-1] = pos.head(2);
            refined_angles[i-1] = pos[2];
        }
        basetime += initTotalduration;
    }
    iniState2d << iniFs[0], refined_singuals[0] * cos(iniFs[2]), 0.0,
                iniFs[1], refined_singuals[0] * sin(iniFs[2]), 0.0;
    finState2d << finFs[0], refined_singuals[segnum-1] * cos(finFs[2]), 0.0,
                finFs[1], refined_singuals[segnum-1] * sin(finFs[2]), 0.0;
    PolyTrajOpt::TrajOpt refinedOpt;
    int flagSs = refinedOpt.OptimizeSe2Trajectory(
    iniState2d, finState2d, refined_rt,
    refined_inPs_container, refined_gearPos,
    refined_angles, se2_grid_map_, sdf_grid_map_,  config_, refined_singuals, vis_tool,"kinoastar");
    if(!flagSs){
        ROS_ERROR("OptimizeSe2Trajectory failed");
        return false;
    }
    double max_cur = refinedOpt.getMaxCur();
    optTraj = refinedOpt.getOptTraj();
    optTraj.start_time = start_time_;
    ROS_WARN_STREAM("dptraj total arc: "<<optTraj.getTotalArc());
    ROS_WARN_STREAM("dptraj traj time: "<<optTraj.getTotalDuration());
    return true;
}
void PlanManager::publishSimTraj2Controller() {
    {
        trajmsg.start_time.fromSec(optTraj.start_time);
        trajmsg.traj_container.clear();
        trajmsg.reverse.clear();
        for(int i = 0; i < optTraj.getSegNum(); i++){
            int singual = optTraj.etas[i];
            mpc_controller::PolyTrajAC trajSegment;
            for(int j = 0; j < optTraj.Traj_container[i].getPieceNum(); j++){
                mpc_controller::SinglePolyAC piece;
                piece.dt = optTraj.Traj_container[i].tTraj[j].getDuration();
                piece.ds = optTraj.Traj_container[i].posTraj[j].getDuration();
                Eigen::Matrix<double, 2, 6> c = optTraj.Traj_container[i].posTraj[j].getCoeffMat();
                Eigen::Matrix<double, 1, 6> c1 = optTraj.Traj_container[i].tTraj[j].getCoeffMat(); 
                for (int k=0; k<6; k++)
                {
                    piece.coef_x.push_back(c(0, k));
                    piece.coef_y.push_back(c(1, k));
                    piece.coef_s.push_back(c1(0, k));
                }
                trajSegment.trajs.push_back(piece);
            }
            trajmsg.traj_container.push_back(trajSegment);
            if(singual > 0) trajmsg.reverse.push_back(false);
            else trajmsg.reverse.push_back(true);
        }
        trajCmdPub.publish(trajmsg);
    }
}
void PlanManager::setInitStateAndInput(const Eigen::Vector4d& state, const double& start_time)
{
  start_state_ = state;
  start_time_ = start_time;
  Eigen::Vector2d init_ctrl(0.0, 0.0);
  start_ctrl_ = init_ctrl;
}
void PlanManager::setTrajStartTime(const double& traj_start_time)
{
  start_time_ = traj_start_time;
}
void PlanManager::setParkingEnd(Eigen::Vector4d end_pt){
    end_state_ = end_pt;
    hasTarget = true;
}
bool PlanManager::getPredictState(const double pre_t, Eigen::Vector4d& state){
    double t = pre_t - optTraj.start_time;
    if(optTraj.Traj_container.empty() || t > optTraj.getTotalDuration()){
        std::cout<<"No trajectory"<<std::endl;
        return false;
    }
    state.head(2) = optTraj.getPos(t);
    state(2) = optTraj.getYaw(t);
    state(3) = optTraj.getVelItem(t);
    return true;
}