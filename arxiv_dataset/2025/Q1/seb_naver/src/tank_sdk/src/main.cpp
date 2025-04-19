#include "ros/ros.h"
#include "tank_send.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "tank_sdk");
    ros::NodeHandle nh("~");

    TankSend tank_send(&nh);


    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}
