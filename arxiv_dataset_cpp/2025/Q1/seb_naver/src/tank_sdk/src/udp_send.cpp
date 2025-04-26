#include <iostream>
#include <thread>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>
#include <unistd.h>

#include "ros/ros.h"
#include "ackermann_msgs/AckermannDrive.h"
#include "tank_sdk/TankFeedback.h"
#include "tank_sdk/TankCMD.h"
#include "nav_msgs/Odometry.h"

#pragma pack(1)

struct HEADER
{
    int16_t mFrameHeader;
    int16_t mFrameType;
    uint8_t mFrameLength;
};

struct UdpControlCommand
{
    HEADER mHeader;
    int16_t vel;
    int16_t cur;
    int16_t pitch;
    uint8_t checkSum;
};

struct UdpRecvCommand
{
    HEADER mHeader;
    int16_t vel;
    int16_t cur;
    int16_t wheelVelLeft;
    int16_t wheelVelRight;
    int16_t reserve;
    uint8_t checkSum;
};

#pragma pack()
uint8_t sendBuffer[32];
uint8_t recvBuffer[32];

const int SEND_PORT = 8021;         // 发送数据的UDP端口
const int RECEIVE_PORT = 40021;      // 接收数据的UDP端口
const char *SERVER_IP = "192.2.2.21"; // 目标服务器IP地址
const char *RECEIVER_IP = "192.2.2.22"; // 车IP

int sendSockfd, receiveSockfd;
struct sockaddr_in serverAddr, receiveAddr;

ros::Subscriber cmd_sub_;
ros::Subscriber sb_sub_;
ros::Publisher feedback_pub_;
ros::Timer feedback_timer_;

void TankCmdCallback(const tank_sdk::TankCMD::ConstPtr &msg)
{
    // auto t1 = ros::Time::now();
    UdpControlCommand dataToSend;
    dataToSend.mHeader.mFrameHeader = 0xffcc;
    dataToSend.mHeader.mFrameType = 0x11;
    // dataToSend.mHeader.mFrameLength = 12;
    // old. work
    dataToSend.mHeader.mFrameLength = 16;

    if (fabs(msg->velocity) < 0.05)
    {
        // ???? 1000.0 ????
        dataToSend.vel = 0;
        dataToSend.cur = (int16_t) (msg->angle_velocity * 1000.0);
    }
    else
    {
        dataToSend.vel = (int16_t) (msg->velocity * 100.0);
        dataToSend.cur = (int16_t) (msg->angle_velocity  / msg->velocity * 1000.0);
        // dataToSend.cur = (int16_t) (msg->angle_velocity * 1.8 / msg->velocity * 1000.0);
    }
   
    dataToSend.pitch = 30 * 0.01;

    // std::cout<<"vel: "<<dataToSend.vel<<" cur: "<<dataToSend.cur<<std::endl;
    
    uint16_t checkSum = 0;

    memset(&sendBuffer[0], 0, sizeof(sendBuffer));
    memcpy(&sendBuffer[0], &dataToSend, 11);

    for (int i = 0; i < 11; i++)
    {
        checkSum += sendBuffer[i];
    }

    sendBuffer[11] = checkSum & 0xff;

    sendto(sendSockfd, &sendBuffer[0], 12, 0, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
    // auto t2 = ros::Time::now();
    // std::cout << "time: " << (t2-t1).toSec() * 1e3 << "ms" << std::endl;
}

void feedbackCallback(const ros::TimerEvent &e)
{
    struct sockaddr_in clientAddr;
    socklen_t addr_len = sizeof(clientAddr);

    memset(&recvBuffer[0], 0, sizeof(recvBuffer));
    recvfrom(sendSockfd, &recvBuffer[0], 16, 0, (struct sockaddr *)&clientAddr, &addr_len);

    uint16_t checkSum = 0;
    for (int i = 0; i < 15; i++)
    {
        checkSum += recvBuffer[i];
    }
    checkSum=checkSum&0xff;

    // std::cout << "/* message */" <<checkSum<< std::endl;
    if ((int)recvBuffer[15] == (int)(checkSum ))
    {

        UdpRecvCommand receivedData;
        memcpy(&receivedData, &recvBuffer[0], 16);
        tank_sdk::TankFeedback fb_msg;
        fb_msg.velocity = (int)(receivedData.vel) * 0.1;
        if (fabs(fb_msg.velocity) < 0.05)
        {
            fb_msg.yaw_velocity = (int)(receivedData.cur);
        }
        else
        {
            fb_msg.yaw_velocity = fb_msg.velocity * (int)(receivedData.cur) * 1e-3;
        }
        fb_msg.wheel_speed_left = (int)(receivedData.wheelVelLeft);
        fb_msg.wheel_speed_right = (int)(receivedData.wheelVelRight);
        feedback_pub_.publish(fb_msg);
        // std::cout << "Received Data->vel: " << receivedData.vel << " cur: " << receivedData.wheelVelLeft << std::endl;
    }
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "udp_send");
    ros::NodeHandle nh("~");

    // 创建UDP socket用于发送数据
    sendSockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sendSockfd < 0)
    {
        perror("Error in socket creation for sending");
        exit(1);
    }

    // 创建UDP socket用于接收数据
    receiveSockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (receiveSockfd < 0)
    {
        perror("Error in socket creation for receiving");
        exit(1);
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SEND_PORT);
    serverAddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    receiveAddr.sin_family = AF_INET;
    receiveAddr.sin_port = htons(RECEIVE_PORT);
    // receiveAddr.sin_addr.s_addr = INADDR_ANY;
    receiveAddr.sin_addr.s_addr = inet_addr(RECEIVER_IP); // xdl

    //绑定接收端口
    if (bind(sendSockfd, (struct sockaddr *)&receiveAddr, sizeof(receiveAddr)) < 0)
    {
        perror("Error in binding for sending");
        exit(1);
    }

    // ROS
    cmd_sub_ = nh.subscribe("cmd", 5, &TankCmdCallback, ros::TransportHints().tcpNoDelay());
    feedback_pub_ = nh.advertise<tank_sdk::TankFeedback>("feed_back", 5);
    feedback_timer_ = nh.createTimer(ros::Duration(0.01), feedbackCallback);

    // ros::spin();
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
