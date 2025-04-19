#include <ros/ros.h>
#include "tank_sdk/TankCMD.h"


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "acc_test_node");
    ros::NodeHandle nh("~");
    
    ros::Publisher pub_tank_cmd = nh.advertise<tank_sdk::TankCMD>("cmd", 5);

    double v0, v1;
    nh.param("v0", v0, 0.0);
    nh.param("v1", v1, 0.0);
    double dur0, dur1;
    nh.param("dur0", dur0, 3.0);
    nh.param("dur1", dur1, 3.0);

    tank_sdk::TankCMD msg;
    msg.angle_velocity = 0.0;
    msg.velocity = 0.0;

    ros::Rate loop_rate(100);
    auto t1 = ros::Time::now();
    
    bool change = false;
    
    while (ros::ok()) {
        auto t2 = ros::Time::now();
        double dur = (t2-t1).toSec();
        if (dur < dur0) {
            // velocity -> 1.5
            // msg.velocity = v0;
            msg.velocity = v0 * dur/dur0;
        } else if (dur < dur0 + dur1) {
            // velocity -> 0
            // msg.velocity = v1;
            msg.velocity = v1 * (dur-dur0)/dur1 + v0 * (dur1-dur+dur0)/dur1;
            if (!change) {
               std::cout << "desired velocity: " << v0 << "->" << v1 << std::endl;
               change = true;
            }            
        } else {
            msg.velocity = 0.0;
        }
        pub_tank_cmd.publish(msg);
        loop_rate.sleep();
    }
    return 0;
}