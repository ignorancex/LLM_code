#include "tank_send.h"


struct HEADER
{
    uint16_t mFrameHeader;
    uint8_t mFrameType;
    uint8_t mFrameLength;
};

struct CMD_CHASSIS
{
    HEADER mHeader;
    uint8_t mModeSwitch;
    uint8_t mAngleVelocity; //
    uint8_t mVelocity;
    uint8_t mHorValLeftJoy;
    uint8_t mVerValLeftJoy;
    uint8_t mDriveGear;
    uint8_t mChassisControl;
    uint8_t mEquipControl;
    uint16_t mLoadControl;
    uint8_t mCRC16Low;
    uint8_t mCRC16High;
    uint8_t mFrameEnd;
};

struct OCU_CHASSIS
{
    HEADER mHeader;
    uint8_t mModeSwitch;
    uint8_t mHorValRightJoy;
    uint8_t mVerValRightJoy;
    uint8_t mHorValLeftJoy;
    uint8_t mVerValLeftJoy;
    uint8_t mDriveGear;
    uint8_t mChassisControl;
    uint8_t mEquipControl;
    uint16_t mLoadControl;
    uint8_t mCRC16Low;
    uint8_t mCRC16High;
    uint8_t mFrameEnd;
};
/*
帧头	uint16
帧类型	uchar
帧长	uint
操控模式切换（互斥）	uchar
操纵摇杆水平方向值（右摇杆）	uchar
操作摇杆垂直方向值（右摇杆）	uchar
操纵摇杆水平方向值（左摇杆）	uchar
操作摇杆垂直方向值（左摇杆）	uchar
驱动档位和负载档位
(速度档包括前、后退)
（主履带速度档位互斥、摆臂速度档位互斥）	uchar
底盘控制(默认全置1)	uchar
设备控制(默认全置1)	uchar
载荷控制(默认全置1)	uchar
CRC16 校验低位	uchar
CRC16 校验高位	uchar
帧尾	uchar
*/

struct CHASSIS_AUTO
{
    HEADER mHeader;
    uint8_t mPlatformStatus;
    uint8_t mVelocity;
    uint8_t mYaw;
    uint8_t mWheelSpeedLeft;
    uint8_t mWheelSpeedRight;
    uint8_t mBatteryRemaining;
    uint8_t mSystemVol;
    uint8_t mPowerStatus;
    uint16_t mLoadStatus;
    uint8_t mCRC16Low;
    uint8_t mCRC16High;
    uint8_t mFrameEnd;
};
/*
帧头	uint16
帧类型	uchar
帧长	uchar
平台机动状态	uchar
车速	uchar
车横摆角速度	uchar
左轮速	uchar
右轮速	uchar
电池剩余电量	uchar
系统电压（电池状态电压 V）	uchar
用电设备通电状态	uchar
载荷状态	uchar
CRC16 校验低位	uchar
CRC16 校验高位	uchar
帧尾	uchar
*/

unsigned short crc_table[256] =
    {
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7, 0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6, 0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485, 0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4, 0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
        0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823, 0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
        0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12, 0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41, 0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70, 0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
        0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F, 0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E, 0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D, 0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C, 0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB, 0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
        0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A, 0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9, 0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
        0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8, 0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0};

uint16_t calculateCRC16(const uint8_t *data, size_t length)
{
    unsigned int loopn, CRC, tmp;
    CRC = 0;
    // printf("CRC: ");
    for (loopn = 0; loopn < length; loopn++)
    {
        // printf("%02X ", *data);
        tmp = ((CRC >> 8) & 0xff) ^ ((*data) & 0xff);
        CRC = ((CRC << 8) & 0xff00) ^ crc_table[tmp];
        data++;
    }
    // printf("\n");

    return (unsigned short)CRC;
}


int computeComplement(int number)
{
    if (number >= 0)
    {
        return number;
    }
    else
    {
        int absValue = -number;
        int negValue = ~absValue + 1;
        return negValue;
    }
}




CMD_CHASSIS mOcuChassis;
// mOcuChassis.mDriveGear = 0x0001;
void TankSend::TankCmdCallback(const tank_sdk::TankCMD::ConstPtr &msg) {
    mOcuChassis.mHeader.mFrameHeader = 0xaaff;
    mOcuChassis.mHeader.mFrameType = 0x0a;
    mOcuChassis.mHeader.mFrameLength = 0x10;
    mOcuChassis.mModeSwitch = 2; // 0 auto 2 joy
    mOcuChassis.mDriveGear = gear;
    // std::cout << "gear: " << mOcuChassis.mDriveGear << std::endl;
    if((abs(msg->velocity) < mid2low_v && abs(msg->angle_velocity) < mid2low_omega) && (mOcuChassis.mDriveGear == 0x0002 || mOcuChassis.mDriveGear == 0x0003 || mOcuChassis.mDriveGear == 0x0000)) {
        gear = 0x0001;
        k_v = k_low;
        ROS_INFO_STREAM("to low");
    }
    else if((abs(msg->velocity) < high2mid_v && abs(msg->angle_velocity) < high2mid_omega) && mOcuChassis.mDriveGear == 0x0003) {
        gear = 0x0002;
        k_v = k_mid;
        ROS_INFO_STREAM("to mid");
    }
    else if((abs(msg->velocity) > low2mid_v || abs(msg->angle_velocity) > low2mid_omega) && mOcuChassis.mDriveGear == 0x0001) {
        gear = 0x0002;
        k_v = k_mid;
        ROS_INFO_STREAM("to mid");
    }
    else if((abs(msg->velocity) > mid2high_v || abs(msg->angle_velocity) > mid2high_omega) && (mOcuChassis.mDriveGear == 0x0001 || mOcuChassis.mDriveGear == 0x0002)) {
        gear = 0x0003;
        k_v = k_high;
        ROS_INFO_STREAM("to high");
    }
    // ROS_INFO_STREAM("k_low: " << k_low);
    // ROS_INFO_STREAM("k_mid: " << k_mid);
    // ROS_INFO_STREAM("k_high: " << k_high);
    int cur = msg->angle_velocity / k_v * 100;
    int velocity = msg->velocity / k_v * 100;
    // ROS_INFO("before velocity:%i, cur:%i \n", velocity, cur);
    velocity = computeComplement(velocity);
    cur = computeComplement(cur);
    // ROS_INFO("after velocity:%i, cur:%i \n", velocity, cur);

    mOcuChassis.mAngleVelocity = cur;      // 负值为对应正向值的补码
    mOcuChassis.mVelocity = velocity; //
    mOcuChassis.mHorValLeftJoy = 0;
    mOcuChassis.mVerValLeftJoy = 0;
    mOcuChassis.mDriveGear = gear; // low:0x0001    mid:0x0002   high:0x0003
    mOcuChassis.mChassisControl = 0x0b;
    mOcuChassis.mEquipControl = 0x07;
    mOcuChassis.mLoadControl = 0x0;

    uint16_t mCrc16 = calculateCRC16(reinterpret_cast<uint8_t *>(&mOcuChassis), sizeof(mOcuChassis) - 4);

    mOcuChassis.mCRC16Low = mCrc16 & 0xff;
    mOcuChassis.mCRC16High = (mCrc16 >> 8) & 0xff;
    // printf(" %02X %02X\n", mOcuChassis.mCRC16Low, mOcuChassis.mCRC16High);
    // mOcuChassis.mCRC16Low = 0x9C;
    // mOcuChassis.mCRC16High = 0x4F;

    mOcuChassis.mFrameEnd = 0x0D;

    std::vector<uint8_t> byteData;
    byteData.resize(sizeof(mOcuChassis)-1);
    std::memcpy(byteData.data(), &mOcuChassis, sizeof(mOcuChassis));

    // for(int i =0; i < byteData.size(); i++) {
    //     printf("%02X ", byteData[i]);

    // }
    // std::cout << "\n";
    // std::cout<< byteData.size() << "\n";

    serial_.send(&byteData[0], byteData.size());

    // serialPort.write_some(boost::asio::buffer(byteData,sizeof(byteData)));
    // char sendBuffer[17] = {(char)0xff, (char)0xaa, (char)0x0a, (char)0x10, (char)0x02, (char)0x00, (char)0x1e, (char)0x00, (char)0x00, (char)0x02, (char)0x0b,
    //                          (char)0x07, (char)0x00, (char)0x00, (char)0x9c, (char)0x4f, (char)0x0d};
    // serialPort.write_some(boost::asio::buffer(sendBuffer,sizeof(sendBuffer)));
}

void TankSend::serialDataProc(uint8_t *data, unsigned int data_len)
{
    // std::cout << "data_len: " << data_len << std::endl;
    uint8_t *p = data;
    // printf("%02X %02X %02X\n", *p, *(p+1), *(p+2));

    CHASSIS_AUTO receivedData;
    memcpy(&receivedData, p, data_len);

    if(receivedData.mHeader.mFrameHeader==0xBBFF && receivedData.mFrameEnd == 0x0C) {

    }
    else return;

    /* Battery No Power Warning */
    // std::cout << "##############\n";
    // std::cout << "mFrameHeader:" << std::hex << (int)receivedData.mHeader.mFrameHeader << "\n";
    // std::cout << "mPlatformStatus:" << (int)receivedData.mPlatformStatus << "\n";
    // std::cout << "mVelocity:" << (int)(receivedData.mVelocity&0x7f)*0.1<< "\n";
    // std::cout << "mYaw:" << (int)(receivedData.mYaw&0x7f)*0.1 << "\n";
    // std::cout << "mWheelSpeedLeft:" << (int)(receivedData.mWheelSpeedLeft&0x7f)*0.1 << "\n";
    // std::cout << "mWheelSpeedRight:" << (int)(receivedData.mWheelSpeedRight&0x7f)*0.1 << "\n";
    // std::cout << "mBatteryRemaining:" << (int)receivedData.mBatteryRemaining*0.5  << "\n";
    // std::cout << "mPowerStatus:" << (int)receivedData.mPowerStatus << "\n";
    // std::cout << "mLoadStatus:" << (int)receivedData.mLoadStatus << "\n";
    // std::cout << "mCRC16Low:" << (int)receivedData.mCRC16Low << "\n";
    // std::cout << "mCRC16High:" << (int)receivedData.mCRC16High << "\n";
    // std::cout << "mFrameEnd:" << (int)receivedData.mFrameEnd << "\n";

    // if((int)receivedData.mBatteryRemaining*0.5 < 15) {
    //     ROS_WARN("No Power! Remaining of Battery is Less Than 15");
    // }

    tank_sdk::TankFeedback fb_msg;
    fb_msg.velocity = (int)(receivedData.mVelocity&0x7f)*0.1;
    fb_msg.yaw_velocity = (int)(receivedData.mYaw&0x7f)*0.1;
    fb_msg.wheel_speed_left = (int)(receivedData.mWheelSpeedLeft&0x7f)*0.1;
    fb_msg.wheel_speed_right = (int)(receivedData.mWheelSpeedRight&0x7f)*0.1;
    fb_msg.gear = k_v;
    feedback_pub_.publish(fb_msg);

}


TankSend::TankSend(ros::NodeHandle *nh) : nh_(*nh)
{
    std::string param_serial_port = "/dev/ttyTHS0";

    nh_.param<std::string>("serial_port", param_serial_port, "/dev/ttyTHS0");
    nh_.getParam(nh_.getNamespace()+"/low2mid_v", low2mid_v);
    nh_.getParam(nh_.getNamespace()+"/mid2low_v", mid2low_v);
    nh_.getParam(nh_.getNamespace()+"/low2mid_omega", low2mid_omega);
    nh_.getParam(nh_.getNamespace()+"/mid2low_omega", mid2low_omega);
    nh_.getParam(nh_.getNamespace()+"/mid2high_v", mid2high_v);
    nh_.getParam(nh_.getNamespace()+"/high2mid_v", high2mid_v);
    nh_.getParam(nh_.getNamespace()+"/mid2high_omega", mid2high_omega);
    nh_.getParam(nh_.getNamespace()+"/high2mid_omega", high2mid_omega);
    nh_.getParam(nh_.getNamespace()+"/k_low", k_low);
    nh_.getParam(nh_.getNamespace()+"/k_mid", k_mid);
    nh_.getParam(nh_.getNamespace()+"/k_high", k_high);

    ROS_INFO_STREAM("k_low: " << k_low);
    ROS_INFO_STREAM("k_mid: " << k_mid);
    ROS_INFO_STREAM("k_high: " << k_high);
    k_v = k_low;



    cmd_sub_ = nh_.subscribe("cmd", 5, &TankSend::TankCmdCallback, this);
    feedback_pub_ = nh_.advertise<tank_sdk::TankFeedback>("feed_back", 5);

    if (serial_.open(param_serial_port.c_str(), 115200, 0, 8, 1, 'N',
                     boost::bind(&TankSend::serialDataProc, this, _1, _2)) != true)
    {
        exit(-1);
    }
}
