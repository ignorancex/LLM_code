#include "mpc/mpc.h"
 
using namespace std;
 
void MPC::init(ros::NodeHandle &nh)
{
    vector<double> Q_ = {10, 10, 0.5};
    vector<double> R_ = {10, 10, 0.5};
    vector<double> Rd_ = {10, 10, 0.5};
    nh.param("mpc/dt", dt, -1.0);
    nh.param("mpc/predict_steps", Npre, -1);
    nh.param("mpc/max_steer", max_steer, -1.0);
    nh.param("mpc/max_dsteer", max_dsteer, -1.0);
    nh.param("mpc/max_speed", max_speed, -1.0);
    nh.param("mpc/min_speed", min_speed, -1.0);
    nh.param("mpc/max_accel", max_accel, -1.0);
    nh.param("mpc/wheel_base", wheel_base, -1.0);
    nh.param("mpc/delay_num", delay_num, -1);
    nh.param("mpc/test_mpc", test_mpc, false);
    nh.param<std::vector<double>>("mpc/matrix_q", Q_, std::vector<double>());
    nh.param<std::vector<double>>("mpc/matrix_r", R_, std::vector<double>());
    nh.param<std::vector<double>>("mpc/matrix_rd", Rd_, std::vector<double>());

    Q = diag(DM({Q_[0], Q_[1], Q_[2]}));
    R = diag(DM({R_[0], R_[1]}));
    Rd = diag(DM({Rd_[0], Rd_[1]}));
    U_min = DM({min_speed, -max_steer});
    U_max = DM({max_speed, max_steer});
    x_0.resize(3, 1);
    x_0 = {0, 0, 0};

    u_0.resize(2, 1);
    u_0 = {0, 0};
    X_sol = repmat(x_0, 1, Npre);
    U_sol = repmat(u_0, 1, Npre);

    nh.param("mpc/bk_mode", bk_mode, false);
    nh.param<string>("mpc/traj_file", traj_file, "xxx");

    has_odom = false;
    receive_traj = false;
    for (int i=0; i<delay_num; i++)
        output_buff.push_back(Eigen::Vector2d::Zero());
    cmd.velocity = 0.0;
    cmd.angle_velocity = 0.0;

    pos_cmd_pub_ = nh.advertise<tank_sdk::TankCMD>("cmd", 200);
    vis_pub = nh.advertise<visualization_msgs::Marker>("/following_path", 10);
    predict_pub = nh.advertise<visualization_msgs::Marker>("/predict_path", 10);
    ref_pub = nh.advertise<visualization_msgs::Marker>("/reference_path", 10);
    desiredS_pub_ = nh.advertise<mpc_controller::desiredState>("/desired_state",10);
    cmd_timer_ = nh.createTimer(ros::Duration(0.05), &MPC::cmdCallback, this);
    odom_sub_ = nh.subscribe("odom", 1, &MPC::rcvOdomCallBack, this);
    traj_sub_1 = nh.subscribe("se2traj", 1, &MPC::rcvTrajCallBack, this);
    traj_sub_2 = nh.subscribe("traj", 1, &MPC::rcvFlatTrajCallBack, this);
    traj_sub_3 = nh.subscribe("dptraj", 1, &MPC::rcvDPTrajCallBack, this);
    err_pub = nh.advertise<std_msgs::Float64>("/track_err", 10);
    if (test_mpc)
    {
        trigger_sub_ = nh.subscribe("/move_base_simple/goal", 1, &MPC::rcvTriggerCallBack, this);
    }

    errs.clear();
    vels.clear();
    angle_vels.clear();
    jerks.clear();
    errs_yaw.clear();
    errs_lon.clear();
    errs_lat.clear();

    trajectory_length = 0.0;

    if (bk_mode)
    {
        string o = traj_file;
        outfile.open(o.insert(o.find("trajs"), "err_"), std::ofstream::out);
        outfile.clear();
    }
}

void MPC::rcvTriggerCallBack(const geometry_msgs::PoseStamped msg)
{
    receive_traj = true;
    traj_analyzer.setTestTraj(max_speed*0.5);
    eight_path = traj_analyzer.getTajWps(0.1);
}

void MPC::rcvTrajCallBack(mpc_controller::SE2TrajConstPtr msg)
{
    traj_analyzer.setTraj(msg);
    receive_traj = true;
}

void MPC::rcvFlatTrajCallBack(mpc_controller::PolyTrajConstPtr msg)
{
    traj_analyzer.setTraj(msg);
    receive_traj = true;
    has_output = false;
}
void MPC::rcvDPTrajCallBack(mpc_controller::DPtrajContainerConstPtr msg){
    new_traj_analyzer.setTraj(msg);
    receive_traj = true;
    has_output = false;
}
void MPC::rcvOdomCallBack(nav_msgs::OdometryPtr msg)
{
    has_odom = true;
    trajectory_length += Eigen::Vector2d(now_state.x-msg->pose.pose.position.x, now_state.y-msg->pose.pose.position.y).norm();
    now_state.x = msg->pose.pose.position.x;
    now_state.y = msg->pose.pose.position.y;
    now_state.theta = 2.0 * atan2(msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
}
void MPC::cmdCallback(const ros::TimerEvent &e)
{
    drawFollowPath();

    if (!has_odom || !receive_traj){
        return;
    }
    ros::Time begin = ros::Time::now();
    if(begin > new_traj_analyzer.start_time){
        traj_analyzer = new_traj_analyzer;
    }
    xref = traj_analyzer.getRefPoints(Npre, dt);
    if (!xref.empty())
    {
        TrajPoint p_now = traj_analyzer.getRefPoint();
        Eigen::Vector2d err_vec(p_now.x - now_state.x, p_now.y - now_state.y);
        std_msgs::Float64 err_msg;
        if(err_vec.norm()<1.0&&err_vec.norm()>0.001){
            errs.push_back(err_vec.norm());
            err_msg.data = errs.back();
            err_pub.publish(err_msg);
        }
        {
            TrajPoint tp = traj_analyzer.getRefPoint();
            mpc_controller::desiredState ds;
            ds.px = tp.x;
            ds.py = tp.y;
            ds.yaw = tp.theta;
            ds.stamp = ros::Time::now();
            Eigen::Vector2d err_vec(tp.x - now_state.x, tp.y - now_state.y);
            ds.tkerror = err_vec.norm();
            double err_yaw = tp.theta - now_state.theta;
            if(ds.tkerror > 1.0){
                ds.tkerror = 0.0;
                err_yaw = 0.0;
            }
            else{
                while (err_yaw > M_PI)
                    err_yaw -= 2.0 * M_PI;
                while (err_yaw < -M_PI)
                    err_yaw += 2.0 * M_PI;
                Eigen::Vector2d dir,n;
                dir << cos(tp.theta), sin(tp.theta);
                n << -sin(tp.theta), cos(tp.theta);
            }
            ds.tkyaw = err_yaw;
            errs_yaw.push_back(ds.tkyaw);
            desiredS_pub_.publish(ds);
        }
    }
    //publish desired state
    if (traj_analyzer.at_goal)
    {
        cmd.velocity = 0.0;
        cmd.angle_velocity = 0.0;

        if (bk_mode)
        {
            Eigen::Vector3d initp = Eigen::Vector3d::Zero();
            double mean_err = std::accumulate(errs.begin(), errs.end(), 0.0) / (1.0 * errs.size());
            mean_err_all += mean_err;
            outfile << mean_err << std::endl;
            if (initp == Eigen::Vector3d::Zero())
            {
                outfile << "all_mean_track_err: " << mean_err_all <<std::endl;
                outfile.close();
                ROS_WARN("all_mean_track_err: %lf", mean_err_all);
                ROS_WARN("Benchmark Done.");
                receive_traj = false;
                return;
            }
            geometry_msgs::Point init_point;

            init_point.x = initp.x();
            init_point.y = initp.y();
            init_point.z = initp.z();

            receive_traj = false;
            errs.clear();
            errs_yaw.clear();
            ros::Duration(1.0).sleep();
            eight_path = traj_analyzer.getTajWps(0.1);
            receive_traj = true;
        }

        double mean_err = std::accumulate(errs.begin(), errs.end(), 0.0) / (1.0 * errs.size());
        for (int i=0; i<vels.size(); i++)
            vels[i] = fabs(vels[i]);

        if (!has_output)
        {
            std::cout<<"\033[31m <<<Statistical Datas>>>:"<<std::endl;
            std::cout <<"max tracking-error: "<<*max_element(errs.begin(),errs.end())<<std::endl;
            std::cout <<"max tkyaw-error: "<<*max_element(errs_yaw.begin(),errs_yaw.end())<<std::endl;
            std::cout<<"mean tracking-error: "<<mean_err<<std::endl;
            std::cout<<"trajectory length  : "<<trajectory_length<<std::endl;
            std::cout<<"end position err   : "<<errs.back()<<std::endl;
            

            TrajPoint p_now = traj_analyzer.getRefPoint();
            double yaw_err = p_now.theta - now_state.theta;
            while (yaw_err > M_PI)
                yaw_err -= 2.0 * M_PI;
            while (yaw_err < -M_PI)
                yaw_err += 2.0 * M_PI;
            std::cout<<"end yaw err        : "<<fabs(yaw_err)<<"\033[0m"<<std::endl;

            errs.clear();
            vels.clear();
            angle_vels.clear();
            trajectory_length = 0.0;
            has_output = true;
        }
    }
    else
    {
        smooth_yaw(xref);
        ros::Time t1 = ros::Time::now();
        getCmd();
        ros::Time t2 = ros::Time::now();
        vels.push_back(cmd.velocity);
        angle_vels.push_back(fabs(cmd.angle_velocity));
    }
    pos_cmd_pub_.publish(cmd);
}

void MPC::getCmd(void)
{
    nlp = casadi::Opti();
    casadi::Dict options;
    casadi::Dict qp_options;
    options["print_status"] = false;
    options["print_time"] = false;
    options["print_header"] = false;
    options["print_iteration"] = false;
    options["verbose"] = false;
    options["verbose_init"] = false;
    qp_options["printLevel"] = "none";
    qp_options["sparse"] = true;
    qp_options["error_on_fail"] = false;
    options["qpsol_options"] = qp_options;
        
    X = nlp.variable(3, Npre);
    U = nlp.variable(2, Npre);
    J = 0;
    MX X_0 = nlp.parameter(3, 1);
    MX x_next;
    Slice all;
    
    for (int i = 0; i < Npre; i++)
    {
        MX ref_x = xref[i].x;
        MX ref_y = xref[i].y;
        MX ref_theta = xref[i].theta;
        MX ref_state = vertcat(ref_x, ref_y, ref_theta);
        MX ref_v = 0;
        MX ref_steer = 0;
        MX ref_u = vertcat(ref_v, ref_steer);

        if (i==0)
            x_next = stateTrans(X_0, U(all, i));
        else
            x_next = stateTrans(X(all, i-1), U(all, i));
        nlp.subject_to(X(all, i) == x_next);
        nlp.subject_to(U_min <= U(all, i) <= U_max);
        
        MX delta_x = ref_state - X(all, i);
        MX delta_u = ref_u - U(all, i);
        
        MX du;
        if (i > 0)
            du = U(all, i) - U(all, i - 1);
        else
            du = U(all, i) - u_0;

        J = J + mtimes(delta_x.T(), mtimes(Q, delta_x));
        J = J + mtimes(delta_u.T(), mtimes(R, delta_u));
        J = J + mtimes(du.T(), mtimes(Rd, du));
    }
    x_0 = {now_state.x, now_state.y, now_state.theta};
    nlp.set_value(X_0, x_0);
    nlp.set_initial(X, X_sol);
    nlp.set_initial(U, U_sol);
    nlp.solver("sqpmethod", options);
    nlp.minimize(J);


    try
    {
        const casadi::OptiSol sol = nlp.solve();

        X_sol = sol.value(X);
        U_sol = sol.value(U);
        DM cmd_0 = U_sol(all, 0);
        u_0 = cmd_0;
        cmd.velocity = (double)cmd_0(0, 0);
        cmd.angle_velocity = (double)cmd_0(1, 0);
    }
    catch(const std::exception& e)
    {
        // ROS_INFO_STREAM("Initial or symbolic values of X:\n" << nlp.debug().value(X));
        // ROS_INFO_STREAM("Initial or symbolic values of U:\n" << nlp.debug().value(U));
        // ROS_WARN_STREAM("Failed with initial X: " << X_sol);
        // ROS_WARN_STREAM("Failed with initial U: " << U_sol);
        // ROS_WARN("solver error, but we ignore.");
    }
    
    drawRefPath();
    drawPredictPath();

    if (delay_num>0)
    {
        output_buff.erase(output_buff.begin());
        output_buff.push_back(Eigen::Vector2d(cmd.velocity, cmd.angle_velocity));
    }
}
