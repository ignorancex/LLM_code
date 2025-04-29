/* Copyright (C) 2012-2022 Ultraleap Limited. All rights reserved.
 *
 * Use of this code is subject to the terms of the Ultraleap SDK agreement
 * available at https://central.leapmotion.com/agreements/SdkAgreement unless
 * Ultraleap has signed a separate license agreement with you or your
 * organisation.
 *
 */

#include <LeapC.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>
#include <iomanip>

#include "GL/glut.h"
#include "Linking/multiLMC_process.h"

#include <filesystem>
namespace fs = std::filesystem;	

#if defined(_MSC_VER)
  #include <Windows.h>
  #include <process.h>
  #define LockType CRITICAL_SECTION
  #define ThreadType HANDLE
  #define ThreadReturnType void
  #define LockMutex EnterCriticalSection
  #define UnlockMutex LeaveCriticalSection
  #include <winsock2.h>
  #pragma comment(lib, "ws2_32.lib")
#else
  #include <unistd.h>        
  #include <pthread.h>
  #define LockType pthread_mutex_t
  #define ThreadType pthread_t
  #define ThreadReturnType void*
  #define LockMutex pthread_mutex_lock
  #define UnlockMutex pthread_mutex_unlock
#endif

// To stop the service loop.
int stop;

static ThreadType pollingThread;
static LockType dataLock;

//Mode
bool mode_writefile = true;
bool mode_printterminal = false;
bool WriteFileDevice = false;
unsigned int numLMC = 4;
bool RobotComm = false;
bool RobotUDP = false;
bool ViconUDP = true;
bool DataCheckUDP = false;
//Visualization
//Drawing all LMCs
bool drawLMCs = true;
//Drawing one LMC
bool drawSingleLMC = false;
//Drawing cartesian line
bool drawCartes = true;
//Drawing origin dot 
bool drawOrigin = true;

struct HandDataTrack
{
  LEAP_TRACKING_EVENT* tracking_event;
  LEAP_HAND* hand; 
  uint32_t device_id;
};

//FILE variable
std::ofstream file;
uint32_t filecount=0;

#if defined(_MSC_VER)  
  std::string file_path;  
  std::filesystem::file_status fileAttributes;
#else  
  char file_path[1024];  
#endif

std::string project_path;  
std::ofstream fileDevice1, fileDevice2, fileDevice3, fileDevice4;

/*Tracking Mode*/
//Tracking status, enabled when enough LMCs detected. Initial always false.
bool TrackStat = false;
//To enable Bypass, opposite of sampling method which will stream data as fast it can be received.
bool BypassMode = true;
//Status of sampling data, used to adjust data so will be recorded in sequence by device number (1,2,3,4)
//and skip data which are not sequenced. Initial always false.
bool SamplingStat = false;
// variable to count data
int CountDev1=0, CountDev2=0, CountDev3=0, CountDev4=0, CountAll=0; 
// Initial sampling seq start from 1.
unsigned int SamplingSeq = 1;

//GLUT Variable
int window; // GLUT window handle
bool RefreshStat = false; // to refresh display

//HAND
int DrawColor = 1; // color of drawing

HandDataTrack* DataDevice1 = NULL;
HandDataTrack* DataDevice2 = NULL;
HandDataTrack* DataDevice3 = NULL;
HandDataTrack* DataDevice4 = NULL;
HandDataTrack* DataDeviceAll = NULL;

HandData DataHandLMC1, DataHandLMC2, DataHandLMC3, DataHandLMC4;
//Variable for all DataHand LMC
HandData DataHandAllLMC;
//Hand Data after fusion (interpolation, aligned, kalman filter fusion)
HandData DataHandFusion;

float robotangle[5];
float wristangle[2];
float elbow[2];

//Command Robot to Communication Protocol
int robot_cmd[9];
float sendUDP_angle[9];

//COMPORT Initilization
HANDLE hSerial;
DCB dcbSerialParams = {0};
COMMTIMEOUTS timeouts = {0};
unsigned char firstcalibrate[1] = {0x42};
unsigned char thumbAb[] = {0x44, 0x00, 0x00};
unsigned char thumb[] = {0x44, 0x01, 0x00};
unsigned char index[] = {0x44, 0x02, 0x00};
unsigned char middle[] = {0x44, 0x03, 0x00};
unsigned char ringlittle[] = {0x44, 0x04, 0x00};
DWORD bytes_written;

//UDP Robot Variable
float UDPData[9];
int UDPdataSize;
struct sockaddr_in serverAddr;
int UDP_port = 12345;
char serv_IPAddress [14] = "192.168.0.11"; //For UDP Communication to Robot
WSADATA wsaData;
SOCKET udpSocket;

//UDP Vicon variable
struct sockaddr_in Vicon_serverAddr;
int Vicon_port = 30;
char Vicon_IPAddress [20] ="192.168.8.2";//For UDP Communication to VICON
WSADATA Vicon_wsaData;
SOCKET Vicon_udpSocket;
std::string Vicon_path;
std::string Vicon_dir = "F:\\ViconData\\VICON\\";
std::string Vicon_subject = "Subject1";
std::string Vicon_session = "26_2_2024";
std::string Vicon_trialname;
std::string Vicon_name = "Trial1"; //or Trial2 or Trial1 or Static2 or Static1
std::string LMC_configname = "Naiv_"; //or Naiv_ or Opti_
int Vicon_stat_send;
int randomNumber;
// Seed the random number generator
std::mt19937 rng(std::time(nullptr));
// Generate a random number between 0 and 1000
std::uniform_int_distribution<int> dist(0, 1000);
//XML for send
std::string xmlString_start;
std::string xmlString_stop;
bool statSendVicon = false;

//UDP toChekData variable
struct sockaddr_in DataCheck_serverAddr;
int DataCheck_port = 12345;
char DataCheck_IPAddress [14] = "127.0.0.1";
WSADATA DataCheck_wsaData;
SOCKET DataCheck_Socket;


LEAP_CONNECTION connection;
LEAP_CONNECTION_CONFIG config;

/*--------------DISPLAY* --------------*/
std::vector<double> LMC1_q1 (3), LMC1_q2 (3), LMC1_q3 (3), LMC1_q4 (3);
std::vector<double> LMC2_q1 (3), LMC2_q2 (3), LMC2_q3 (3), LMC2_q4 (3);
std::vector<double> LMC3_q1 (3), LMC3_q2 (3), LMC3_q3 (3), LMC3_q4 (3);
std::vector<double> LMC4_q1 (3), LMC4_q2 (3), LMC4_q3 (3), LMC4_q4 (3);

/*
//.. Naiv Configuration ..//
std::vector<double> LMC1c = {-60,0,60};
std::vector<double> LMC2c = {60,0,60};
std::vector<double> LMC3c = {-60, 0, -60};
std::vector<double> LMC4c = {60, 0, -60};
*/

//.. Opti Configuration ..//
std::vector<double> LMC1c = {-120.37,0,-256.29};
std::vector<double> LMC2c = {-69.97,0,342.57};
std::vector<double> LMC3c = {-190.60, 0, 88.70};
std::vector<double> LMC4c = {178.88, 0, 10.90};


#define LEAPC_CHECK(func)                  \
  do                                       \
  {                                        \
    eLeapRS result = func;                 \
    if (result != eLeapRS_Success)         \
    {                                      \
      std::cout << "Fatal error in calling function: " << #func << ": " << std::hex << result << std::endl; \
      abort();                             \
    }                                      \
  } while (0)

static const char* devicePIDToString(eLeapDevicePID p)
{
  if (p == eLeapDevicePID_3Di)
  {
    return "3Di";
  }
  if (p == eLeapDevicePID_Peripheral)
  {
    return "Leap Motion Controller";
  }
  if (p == eLeapDevicePID_SIR170)
  {
    return "Stereo SIR170";
  }
  return "Unknown Tracking Device";
}

struct DeviceState
{
  LEAP_DEVICE device;
  uint32_t id;
  char serial_number[100];
  uint32_t PositionId;
};

int GetPositionID (int deviceIDread, DeviceState* devices){   
    uint32_t j;
    for(j = 0; j < 4; j++){
      if (devices[j].id == deviceIDread) break;
    }  
    return devices[j].PositionId;
  }

/// Function Initialization ///
void closing_procedure(void);

void display(void);
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);
void idle(void);

void drawBG(void);
void drawHand(HandDataTrack* HandDataTrack);

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

HandDataTrack* HandDataCopy(const LEAP_TRACKING_EVENT* src, uint32_t device_id);
HandData CopyHandPosition (HandDataTrack* CopiedData);
int HandDataSendUDP(SOCKET Socket,HandData HandData, struct sockaddr_in serverAddr);

static ThreadReturnType pollingServiceLoop(void* p)
{
  LEAP_CONNECTION* connection = (LEAP_CONNECTION*)p;

  // LEAP_DEVICE's can be passed to -Ex forms of the LeapC API
  DeviceState* devices = NULL;
  uint32_t deviceCount = 0;

  while (true)
  {
    LockMutex(&dataLock);
    if (stop)
    {
      UnlockMutex(&dataLock);
      break;
    }
    UnlockMutex(&dataLock);
  
    const uint32_t timeoutMilliseconds = 10;
    LEAP_CONNECTION_MESSAGE msg;        
 
      eLeapRS result = LeapPollConnection(*connection, timeoutMilliseconds, &msg);   
      if (result == eLeapRS_Timeout)
      {
        continue;
      }
    
    if (msg.type == eLeapEventType_Device)
    {
      DeviceState* newDevices = static_cast<DeviceState*>(realloc(devices, sizeof(DeviceState) * (deviceCount + 1)));
      if (!newDevices)
      {
        std::cout << "Failed to allocate memory to accommodate " << deviceCount + 1 << " devices" << std::endl;        
        abort();
      }
      else
      {
        devices = newDevices;
        LEAPC_CHECK(LeapOpenDevice(msg.device_event->device, &devices[deviceCount].device));

        devices[deviceCount].id = msg.device_event->device.id;

        LEAP_DEVICE_INFO deviceInfo;
        memset(&deviceInfo, 0, sizeof(deviceInfo));

        // Use stack memory to allocate for the serial number field.
        char serial[1000];
        memset(serial, 0, sizeof(serial));
        deviceInfo.serial = serial;
        deviceInfo.serial_length = sizeof(serial) - 1;
        deviceInfo.size = sizeof(deviceInfo);

        LEAPC_CHECK(LeapGetDeviceInfo(devices[deviceCount].device, &deviceInfo));

        std::cout << "Found a device:: Registered ID: " << msg.device_event->device.id
          << ", type: " << devicePIDToString(deviceInfo.pid)
          << ", SN: " << deviceInfo.serial;        
        strcpy(devices[deviceCount].serial_number,deviceInfo.serial);

        if((strcmp(deviceInfo.serial,"LP06159183280")==0)) {
          std::cout << " >> Dev.ID= 1" << std::endl;          
          devices[deviceCount].PositionId = 1;
        }
        if((strcmp(deviceInfo.serial,"LP32306109995")==0)) {
          std::cout << " >> Dev.ID= 2" << std::endl;          
          devices[deviceCount].PositionId = 2;
        }
        if((strcmp(deviceInfo.serial,"LP62335458489")==0)) {
          std::cout << " >> Dev.ID= 3" << std::endl;
          devices[deviceCount].PositionId = 3;
        }        
        if((strcmp(deviceInfo.serial,"LP76447179865")==0)) {
          std::cout << " >> Dev.ID= 4" << std::endl;
          devices[deviceCount].PositionId = 4;
        }
        
        // Unconditionally subscribe to the device:
        LEAPC_CHECK(LeapSubscribeEvents(*connection, devices[deviceCount].device));
        ++deviceCount;        
      }
    }

    if (msg.type == eLeapEventType_DeviceLost)
    {
      LEAP_DEVICE* deviceLost = NULL;
      for (int i = 0; i < deviceCount; ++i)
      {
        if (devices[i].id == msg.device_event->device.id)
        {
          deviceLost = &devices[i].device;
        }
      }

      if (deviceLost)
      {        
        std::cout << "Unsubscribing from device: " << msg.device_event->device.id << std::endl;
        LEAPC_CHECK(LeapUnsubscribeEvents(*connection, *deviceLost));
        LeapCloseDevice(*deviceLost);
      }
    }
        //Wait until 4 (numLMC) device detected (or any amount of devise/s)
    if (deviceCount==numLMC){              
      TrackStat = true; 
      if (BypassMode) SamplingStat = true; //to bypass sampling cue      
      if (ViconUDP && statSendVicon)
      // Send the XML string
      for (int i=0; i<3; ++i)
      Vicon_stat_send = sendto(Vicon_udpSocket, xmlString_start.c_str(), xmlString_start.size(), 0, (struct sockaddr*)&Vicon_serverAddr, sizeof(Vicon_serverAddr));    
      statSendVicon = false;
    }        

    if (msg.type == eLeapEventType_Tracking && TrackStat==true) // start when TrackStat enabled
    {          

      if ((GetPositionID(msg.device_id, devices)) == SamplingSeq){
        SamplingStat = true;   
        CountAll++;                              
      }
      
      if (SamplingStat)
      {                                 
        if (!BypassMode) SamplingStat = false; 
        SamplingSeq++;
        if (SamplingSeq == (numLMC+1)) SamplingSeq =1;

        if (GetPositionID(msg.device_id, devices) == 1) {CountDev1++; DrawColor=1;}
        if (GetPositionID(msg.device_id, devices) == 2) {CountDev2++; DrawColor=2;}
        if (GetPositionID(msg.device_id, devices) == 3) {CountDev3++; DrawColor=3;}
        if (GetPositionID(msg.device_id, devices) == 4) {CountDev4++; DrawColor=4;}
        
        //with Frame Rate and Time Stamp        
        std::cout << "Dev.ID: " << GetPositionID(msg.device_id, devices) << ", Frame: " << msg.tracking_event->info.frame_id
          << ", Rate: " << msg.tracking_event->framerate << ", Stamp: " << msg.tracking_event->info.timestamp
          << ", nHand: " << msg.tracking_event->nHands << std::endl;
        
        //For printing in terminal
        if (mode_printterminal)
        {        
          for(uint32_t h = 0; h < msg.tracking_event->nHands; h++){
            LEAP_HAND* hand = &msg.tracking_event->pHands[h];               
            std::cout << "    Hand id " << hand->id << " is a " 
                      << (hand->type == eLeapHandType_Left ? "left" : "right") << " hand with position ("
                      << hand->palm.position.x << ", " 
                      << hand->palm.position.y << ", " 
                      << hand->palm.position.z << ")." << std::endl;
            std::cout << "      Palm normal direction is (" << hand->palm.normal.x << ", " 
                      << hand->palm.normal.y << ", " << hand->palm.normal.z << ")." << std::endl;
            std::cout << "      Hand direction is (" << hand->palm.direction.x << ", " 
                      << hand->palm.direction.y << ", " << hand->palm.direction.z << ")." << std::endl;
            std::cout << "      Elbow position is (" << hand->arm.prev_joint.x << ", " 
                      << hand->arm.prev_joint.y << ", " << hand->arm.prev_joint.z << ")." << std::endl;
            std::cout << "      Wrist position is (" << hand->arm.next_joint.x << ", " 
                      << hand->arm.next_joint.y << ", " << hand->arm.next_joint.z << ")." << std::endl;

            for(uint32_t j = 0; j < 5; j++){                         
              std::cout << "        Finger " << j + 1 << "\tMetacarpals position is (" 
                        << hand->digits[j].bones[0].prev_joint.x << ", "
                        << hand->digits[j].bones[0].prev_joint.y << ", " 
                        << hand->digits[j].bones[0].prev_joint.z << ")." << std::endl;
              std::cout << "                 \tProximal phalanges position is (" 
                        << hand->digits[j].bones[0].next_joint.x << ", "
                        << hand->digits[j].bones[0].next_joint.y << ", "
                        << hand->digits[j].bones[0].next_joint.z << ")." << std::endl;
              std::cout << "                 \tIntermediate phalanges position is (" 
                        << hand->digits[j].bones[1].next_joint.x << ", "
                        << hand->digits[j].bones[1].next_joint.y << ", " 
                        << hand->digits[j].bones[1].next_joint.z << ")." << std::endl;
              std::cout << "                 \tDistal phalanges position is (" 
                        << hand->digits[j].bones[2].next_joint.x << ", "
                        << hand->digits[j].bones[2].next_joint.y << ", " 
                        << hand->digits[j].bones[2].next_joint.z << ")." << std::endl;
              std::cout << "                 \tEnd position is (" 
                        << hand->digits[j].bones[3].next_joint.x << ", "
                        << hand->digits[j].bones[3].next_joint.y << ", " 
                        << hand->digits[j].bones[3].next_joint.z << ")." << std::endl;
            }
          }
        }
                
        if (SamplingSeq == numLMC) RefreshStat = true; // refreshed every 4 (numLMC) data
        //Bypass Data
        if(msg.tracking_event->nHands>0)
        {      
          //RefreshStat = true; // for refreshing drawing          
          
          DataDeviceAll = HandDataCopy(msg.tracking_event, GetPositionID(msg.device_id, devices));
          DataHandAllLMC = CopyHandPosition(DataDeviceAll);

          if(DataCheckUDP){
            int statDataCheck = HandDataSendUDP(DataCheck_Socket, DataHandAllLMC, DataCheck_serverAddr);
          }

          switch(GetPositionID(msg.device_id, devices)){
            case 1:
              DataDevice1 = HandDataCopy(msg.tracking_event, GetPositionID(msg.device_id, devices));
              DataHandLMC1 = CopyHandPosition(DataDevice1);
              angle_calc(DataHandLMC1, robotangle, wristangle, elbow); 

              if(WriteFileDevice) SaveDataDevice(fileDevice1, DataHandLMC1);

              if (RobotComm){
                robot_cmd[0]=SendCmdThumbAb(robotangle[0]);
                robot_cmd[1]=SendCmdThumb(robotangle[1]);
                robot_cmd[2]=SendCmd(robotangle[2]);
                robot_cmd[3]=SendCmd(robotangle[3]);
                robot_cmd[4]=SendCmd(robotangle[4]);
                
                thumbAb[2] = robot_cmd[0];                 
                WriteFile(hSerial, thumbAb, sizeof(thumbAb), &bytes_written, NULL);
                thumb[2] = robot_cmd[1]; 
                WriteFile(hSerial, thumb, sizeof(thumb), &bytes_written, NULL);
                index[2] = robot_cmd[2];
                WriteFile(hSerial, index, sizeof(index), &bytes_written, NULL); 
                middle[2] = robot_cmd[3];
                WriteFile(hSerial, middle, sizeof(middle), &bytes_written, NULL); 
                ringlittle[2] = robot_cmd[4]; 
                WriteFile(hSerial, ringlittle, sizeof(ringlittle), &bytes_written, NULL);
              }
            
              if (RobotUDP){         
                ProcessUDP(sendUDP_angle, robotangle, wristangle, elbow);
                
                UDPData[0]= sendUDP_angle[0];
                UDPData[1]= sendUDP_angle[1]; 
                UDPData[2]= sendUDP_angle[2]; 
                UDPData[3]= sendUDP_angle[3];
                UDPData[4]= sendUDP_angle[4];
                UDPData[5]= sendUDP_angle[5];
                UDPData[6]= sendUDP_angle[6];
                UDPData[7]= sendUDP_angle[7];
                UDPData[8]= sendUDP_angle[8];

                UDPdataSize = sizeof(UDPData);

                int stat = sendto(udpSocket, (char*)UDPData, UDPdataSize, 0, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
                if (stat < 0) {
                    std::cerr << "Error sending data: " << std::strerror(errno) << std::endl;
                    closesocket(udpSocket);
                    WSACleanup();                    
                }
              }
              break;
            case 2:
              DataDevice2 = HandDataCopy(msg.tracking_event, GetPositionID(msg.device_id, devices));
              DataHandLMC2 = CopyHandPosition(DataDevice2);
              
              if(WriteFileDevice) SaveDataDevice(fileDevice2, DataHandLMC2);
              break;
            case 3:
              DataDevice3 = HandDataCopy(msg.tracking_event, GetPositionID(msg.device_id, devices));
              DataHandLMC3 = CopyHandPosition(DataDevice3);
              
              if(WriteFileDevice) SaveDataDevice(fileDevice3, DataHandLMC3);
              break;
            case 4:
              DataDevice4 = HandDataCopy(msg.tracking_event, GetPositionID(msg.device_id, devices));
              DataHandLMC4 = CopyHandPosition(DataDevice4);
              
              if(WriteFileDevice) SaveDataDevice(fileDevice4, DataHandLMC4);
              break;
            default:
              break;
          }
                          
        }
        
        //For printing in file
        if (mode_writefile)
        {
          file << GetPositionID(msg.device_id, devices) << "\t" << msg.tracking_event->info.frame_id << "\t"
              << msg.tracking_event->framerate << "\t" << msg.tracking_event->info.timestamp << "\t"; 
          if (msg.tracking_event->nHands == 0) 
            file << "\n";
          else {
            for (uint32_t h = 0; h < msg.tracking_event->nHands; h++) {
              LEAP_HAND* hand = &msg.tracking_event->pHands[h];            
              file  << hand->palm.position.x << "," << hand->palm.position.y << "," << hand->palm.position.z << "\t"
                    << hand->palm.normal.x << "," << hand->palm.normal.y << "," << hand->palm.normal.z << "\t"
                    << hand->palm.direction.x << "," << hand->palm.direction.y << "," << hand->palm.direction.z << "\t"
                    << hand->arm.prev_joint.x << "," << hand->arm.prev_joint.y << "," << hand->arm.prev_joint.z << "\t"
                    << hand->arm.next_joint.x << "," << hand->arm.next_joint.y << "," << hand->arm.next_joint.z << "\t";
              //Only send Palm comment between these
              for (uint32_t j = 0; j < 5; j++) {
                file << hand->digits[j].bones[0].prev_joint.x << "," << hand->digits[j].bones[0].prev_joint.y << "," << hand->digits[j].bones[0].prev_joint.z << "\t"
                     << hand->digits[j].bones[0].next_joint.x << "," << hand->digits[j].bones[0].next_joint.y << "," << hand->digits[j].bones[0].next_joint.z << "\t"
                     << hand->digits[j].bones[1].next_joint.x << "," << hand->digits[j].bones[1].next_joint.y << "," << hand->digits[j].bones[1].next_joint.z << "\t"
                     << hand->digits[j].bones[2].next_joint.x << "," << hand->digits[j].bones[2].next_joint.y << "," << hand->digits[j].bones[2].next_joint.z << "\t"
                     << hand->digits[j].bones[3].next_joint.x << "," << hand->digits[j].bones[3].next_joint.y << "," << hand->digits[j].bones[3].next_joint.z << "\t";
              } 
              //Only send Palm comment between these                  
            }
            file << "\n";
          }
        }                           
      }      
    }

  }
  free(devices);
}

int main(int argc, char *argv[])
{   
  // Set connection to multi-device aware
  memset(&config, 0, sizeof(config));
  config.size = sizeof(config);
  config.flags = eLeapConnectionConfig_MultiDeviceAware;

  // Open LeapC connection:
  LEAPC_CHECK(LeapCreateConnection(&config, &connection));
  LEAPC_CHECK(LeapOpenConnection(connection));

  //display//
  LMC1_q1[0] = LMC1c[0] + 40; LMC1_q1[1] = LMC1c[1] + 0; LMC1_q1[2] = LMC1c[2] + 15;
  LMC1_q2[0] = LMC1c[0] - 40; LMC1_q2[1] = LMC1c[1] + 0; LMC1_q2[2] = LMC1c[2] + 15;
  LMC1_q3[0] = LMC1c[0] - 40; LMC1_q3[1] = LMC1c[1] + 0; LMC1_q3[2] = LMC1c[2] - 15;
  LMC1_q4[0] = LMC1c[0] + 40; LMC1_q4[1] = LMC1c[1] + 0; LMC1_q4[2] = LMC1c[2] - 15;

  LMC2_q1[0] = LMC2c[0] + 40; LMC2_q1[1] = LMC2c[1] + 0; LMC2_q1[2] = LMC2c[2] + 15;
  LMC2_q2[0] = LMC2c[0] - 40; LMC2_q2[1] = LMC2c[1] + 0; LMC2_q2[2] = LMC2c[2] + 15;
  LMC2_q3[0] = LMC2c[0] - 40; LMC2_q3[1] = LMC2c[1] + 0; LMC2_q3[2] = LMC2c[2] - 15;
  LMC2_q4[0] = LMC2c[0] + 40; LMC2_q4[1] = LMC2c[1] + 0; LMC2_q4[2] = LMC2c[2] - 15;

  LMC3_q1[0] = LMC3c[0] + 40; LMC3_q1[1] = LMC3c[1] + 0; LMC3_q1[2] = LMC3c[2] + 15;
  LMC3_q2[0] = LMC3c[0] - 40; LMC3_q2[1] = LMC3c[1] + 0; LMC3_q2[2] = LMC3c[2] + 15;
  LMC3_q3[0] = LMC3c[0] - 40; LMC3_q3[1] = LMC3c[1] + 0; LMC3_q3[2] = LMC3c[2] - 15;
  LMC3_q4[0] = LMC3c[0] + 40; LMC3_q4[1] = LMC3c[1] + 0; LMC3_q4[2] = LMC3c[2] - 15;

  LMC4_q1[0] = LMC4c[0] + 40; LMC4_q1[1] = LMC4c[1] + 0; LMC4_q1[2] = LMC4c[2] + 15;
  LMC4_q2[0] = LMC4c[0] - 40; LMC4_q2[1] = LMC4c[1] + 0; LMC4_q2[2] = LMC4c[2] + 15;
  LMC4_q3[0] = LMC4c[0] - 40; LMC4_q3[1] = LMC4c[1] + 0; LMC4_q3[2] = LMC4c[2] - 15;
  LMC4_q4[0] = LMC4c[0] + 40; LMC4_q4[1] = LMC4c[1] + 0; LMC4_q4[2] = LMC4c[2] - 15;

  //obtain source path
  std::filesystem::current_path("..");
  project_path = std::filesystem::current_path().string();
  
  // Setting Directory for recording to text file
#if defined(_MSC_VER)
  do{
      if(ViconUDP){        
        std::string directoryPath = project_path + "\\data\\" + Vicon_subject;
        
        // Check if the directory exists
        if (!fs::exists(directoryPath)) {
          // Create the directory if it doesn't exist
          if (!fs::create_directory(directoryPath)) {
            std::cerr << "Error: Failed to create directory." << std::endl;
            return 1;
          }
          else{
          filecount++;
          std::stringstream ss;
          ss << std::setw(2) << std::setfill('0') << filecount;
          std::string filecount_str = ss.str();
          
          file_path = directoryPath + "\\LMC_" + LMC_configname + Vicon_name + filecount_str + ".txt";
          Vicon_trialname = LMC_configname + Vicon_name + filecount_str;
          fileAttributes = std::filesystem::status(file_path);      
        } 
        }
        else{
          filecount++;
          std::stringstream ss;
          ss << std::setw(2) << std::setfill('0') << filecount;
          std::string filecount_str = ss.str();
          
          file_path = directoryPath + "\\LMC_" + LMC_configname + Vicon_name + filecount_str + ".txt";
          Vicon_trialname = LMC_configname + Vicon_name + filecount_str;
          fileAttributes = std::filesystem::status(file_path);     
        }
      }
      else {
        filecount++;
        std::stringstream ss;
        ss << std::setw(2) << std::setfill('0') << filecount;
        std::string filecount_str = ss.str();

        file_path = project_path + "\\data\\DataLMC" + filecount_str + ".txt";
        fileAttributes = std::filesystem::status(file_path);
      }
      
  } while (fileAttributes.type() != std::filesystem::file_type::not_found && !std::filesystem::is_directory(fileAttributes));    
#else
  chdir("..");
  do{
      filecount++;   
      getcwd(file_path, sizeof(file_path));             
      sprintf(bufftext,"\\data\\DataLMC%d.txt",filecount);
      strcat(file_path,bufftext);        
  } while (access(file_path, F_OK) != -1);    
#endif  
  
  //For printing in file
  if (mode_writefile)
  {      
    file.open(file_path);  // Open file in write mode
    if (!file.is_open()) {
      std::cout << "Unable to open the file." << std::endl;
      return 1;
    }

    file << "Device ID\tFrame ID\tFrame Rate\tTime Stamp\t";
    file << "Palm Position.x,Palm Position.y,Palm Position.z\tPalm Normal.x,Palm Normal.y,Palm Normal.z\tPalm Direction.x,Palm Direction.y,Palm Direction.z\t";
    file << "Arm Proximal.x,Arm Proximal.y,Arm Proximal.z\tArm Distal.x,Arm Distal.y,Arm Distal.z\t";
    for (uint32_t j = 0; j < 5; j++) {
      file << "Finger" << j+1 << "_Meta.x,Finger" << j+1 << "_Meta.y,Finger" << j+1 << "_Meta.z\t";
      file << "Finger" << j+1 << "_Proxi.x,Finger" << j+1 << "_Proxi.y,Finger" << j+1 << "_Proxi.z\t";
      file << "Finger" << j+1 << "_Inter.x,Finger" << j+1 << "_Inter.y,Finger" << j+1 << "_Inter.z\t";
      file << "Finger" << j+1 << "_Dist.x,Finger" << j+1 << "_Dist.y,Finger" << j+1 << "_Dist.z\t";
      file << "Finger" << j+1 << "_End.x,Finger" << j+1 << "_End.y,Finger" << j+1 << "_End.z\t";
    }
    file << std::endl;
  }  

  if (WriteFileDevice)
  {
    fileDevice1 = initSaveFile(project_path, 1);
    fileDevice2 = initSaveFile(project_path, 2);
    fileDevice3 = initSaveFile(project_path, 3);
    fileDevice4 = initSaveFile(project_path, 4);
  }
  
  // Start to tracking loop
  std::cout << "Press Enter or Control-C to exit, tracking messages will follow:\n";
       
  // Plotting routine
  int screenWidth = GetSystemMetrics(SM_CXSCREEN);
  int screenHeight = GetSystemMetrics(SM_CYSCREEN);
  
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
  glutInitWindowSize(screenWidth*0.53, screenHeight*0.5);
  glutInitWindowPosition(1,1);
  window = glutCreateWindow("Hand Tracking Visualization");

  // GLUT callbacks
  glutIdleFunc(idle);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutDisplayFunc(display);
 
   // init GL
  glClearColor(1.0, 1.0, 1.0, 0.0);
  glColor3f(0.0, 0.0, 0.0);
  glLineWidth(3.0);
  
  if (RobotComm)
  {
    // Open the highest available serial port number
    hSerial = CreateFile(
                "\\\\.\\COM4", GENERIC_READ | GENERIC_WRITE, 0, NULL,
                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hSerial == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error while opening serial port!\n");
        return 1;
    }

    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
    if (GetCommState(hSerial, &dcbSerialParams) == 0) {
        fprintf(stderr, "Error getting device state\n");
        CloseHandle(hSerial);
        return 1;
    }

    dcbSerialParams.BaudRate = CBR_115200;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;
    if (SetCommState(hSerial, &dcbSerialParams) == 0) {
        fprintf(stderr, "Error setting device parameters\n");
        CloseHandle(hSerial);
        return 1;
    }

    // Set COM port timeout settings
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    if (SetCommTimeouts(hSerial, &timeouts) == 0) {
        fprintf(stderr, "Error setting timeouts\n");
        CloseHandle(hSerial);
        return 1;
    }
  }

  if (RobotUDP)
  {    
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }

    // Create a UDP socket
    udpSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (udpSocket == INVALID_SOCKET) {
        fprintf(stderr, "Failed to create socket\n");
        WSACleanup();
        return 1;
    }

    // Server information
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(UDP_port); // Port to send to    
    serverAddr.sin_addr.s_addr = inet_addr(serv_IPAddress); // IP address of the server
  }    

  if (ViconUDP)
  {    
    if (WSAStartup(MAKEWORD(2, 2), &Vicon_wsaData) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }

    // Create a UDP socket
    Vicon_udpSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (Vicon_udpSocket == INVALID_SOCKET) {
        fprintf(stderr, "Failed to create socket\n");
        WSACleanup();
        return 1;
    }

    // Server information
    Vicon_serverAddr.sin_family = AF_INET;
    Vicon_serverAddr.sin_port = htons(Vicon_port); // Port to send to    
    Vicon_serverAddr.sin_addr.s_addr = inet_addr(Vicon_IPAddress); // IP address of the server

    Vicon_path = Vicon_dir + "\\" + Vicon_subject + "\\" + Vicon_session  + "\\";
    randomNumber = dist(rng);
    // XML string to send (start)
    xmlString_start   = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"                      
                        "<CaptureStart>\n"
                        "  <Name VALUE=\""+Vicon_trialname+"\"/>\n"
                        "  <DatabasePath VALUE=\""+Vicon_path+"\"/>\n"
                        "  <Delay VALUE=\"0\"/>\n"
                        "  <PacketID VALUE=\""+std::to_string(randomNumber)+"\"/>\n"
                        "</CaptureStart>\n";        

    // Send the XML string in LMC received part (TrackStat)
    statSendVicon = true;
  }

  if (DataCheckUDP)
  {    
    if (WSAStartup(MAKEWORD(2, 2), &DataCheck_wsaData) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }

    // Create a UDP socket
    DataCheck_Socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (DataCheck_Socket == INVALID_SOCKET) {
        fprintf(stderr, "Failed to create socket\n");
        WSACleanup();
        return 1;
    }

    // Server information
    DataCheck_serverAddr.sin_family = AF_INET;
    DataCheck_serverAddr.sin_port = htons(DataCheck_port); // Port to send to    
    DataCheck_serverAddr.sin_addr.s_addr = inet_addr(DataCheck_IPAddress); // IP address of the server
  }
  
#if defined(_MSC_VER)
  InitializeCriticalSection(&dataLock);
  pollingThread = (ThreadType)_beginthread(pollingServiceLoop, 0, &connection);
#else
  pthread_mutex_init(&dataLock, NULL);
  pthread_create(&pollingThread, NULL, pollingServiceLoop, &connection);
#endif

  //Spot to try

  // Start GLUT loop
  glutMainLoop();

  // Waiting for enter to stop tracking loop
  (void)getchar();
  
  closing_procedure();

  return 0;
}

void closing_procedure(void)
{
  LockMutex(&dataLock);
  stop = 1;
  UnlockMutex(&dataLock);

#if defined(_MSC_VER)
  WaitForSingleObject(pollingThread, 0);
  CloseHandle(pollingThread);
#else
  pthread_join(pollingThread, NULL);
  pthread_mutex_destroy(&dataLock);
#endif
  
  if (ViconUDP) {
    // XML string to send (stop)
    randomNumber = dist(rng);
    xmlString_stop    = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"                      
                        "<CaptureStop>\n"
                        "  <Name VALUE=\""+Vicon_trialname+"\"/>\n"
                        "  <DatabasePath VALUE=\""+Vicon_path+"\"/>\n"
                        "  <Delay VALUE=\"0\"/>\n"
                        "  <PacketID VALUE=\""+std::to_string(randomNumber)+"\"/>\n"
                        "</CaptureStop>\n";

    for (int i=0; i<3; ++i)
    Vicon_stat_send = sendto(Vicon_udpSocket, xmlString_stop.c_str(), xmlString_stop.size(), 0, (struct sockaddr*)&Vicon_serverAddr, sizeof(Vicon_serverAddr));
    closesocket(Vicon_udpSocket);
    WSACleanup();
  }

    //For printing in file
  if (mode_writefile) file.close();

  if (WriteFileDevice) {
    fileDevice1.close();
    fileDevice2.close();
    fileDevice3.close();
    fileDevice4.close();
  }  

  if (RobotComm) CloseHandle(hSerial);
  if (RobotUDP) {
    closesocket(udpSocket);
    WSACleanup();
  }
  
  /*This part may be used to close the program.
  It is commented because sometimes make an error in debugging mode when program stopped*/
  //LeapCloseConnection(connection);
  //LeapDestroyConnection(connection);
  glutDestroyWindow(window);
    
}

void display(void)
{    
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  //glPushMatrix();
  //glTranslatef(0, -300, -500); //"Camera" viewpoint mode 1
  
  /*
  glLoadIdentity();
  gluLookAt(0.0, 400.0, 350.0,  // Camera position
              0.0, 200.0, -200.0,  // Look-at point
              0.0, 0.5, -5.0); // Up vector
    
  */
  
  glLoadIdentity();
  gluLookAt(0.0, 800.0, 0.0,  // Camera position mode 3 (from top)
              0.0, 0.0, 0.0,  // Look-at point
              0.0, 0, -1.0); // Up vector
  
  /*
  glLoadIdentity();
  gluLookAt(-500.0, 500.0, 0.0,  // Camera position mode 4 (from side)
              0.0, 250.0, 0.0,  // Look-at point
              0.0, 0, -1.0); // Up vector
  */

  drawBG();  

  if(DataDevice1)
  {
    glColor3f(1.0, 0.0, 0.0); // red
    drawHand(DataDevice1);
    free(DataDevice1);
    DataDevice1 = NULL;    
  }

  if(DataDevice2)
  {
    glColor3f(0.0, 0.0, 1.0); // blue
    drawHand(DataDevice2);
    free(DataDevice2);
    DataDevice2 = NULL;    
  }
  
  if(DataDevice3)
  {
    glColor3f(0.0, 1.0, 0.0); // green
    drawHand(DataDevice3);
    free(DataDevice3);
    DataDevice3 = NULL;    
  }

  if(DataDevice4)
  {
    glColor3f(0.0, 0.0, 0.0); // black
    drawHand(DataDevice4);
    free(DataDevice4);
    DataDevice4 = NULL;    
  }

  glFlush();
  glPopMatrix();

  glutSwapBuffers();   
}
 
void reshape(int w, int h)
{
  glViewport(0, 0, (GLsizei) w, (GLsizei) h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(100, 1920/800, 0.1, 1000); //for mode 2 (set 2000 for Edge Detection)
  //gluPerspective(90, 1920/800, 0.1, 800); //for mode 1 and 3
}
 
void keyboard(unsigned char key, int x, int y)
{
  switch((char)key) {
  case 'q':
  case 27:  // ESC    

    closing_procedure();

    exit(0);
  default:
    break;
  }
}

void idle(void){  
  if(RefreshStat == true){
    glutPostRedisplay();
    RefreshStat = false;
  }
}

void drawBG(void)
{
  
  if (drawCartes){
    //Cartesian Line
    glColor3f(1.0, 0.0, 0.0); // red (X Axis)
    glBegin(GL_LINES);
      glVertex3i(-400,0,0);
      glVertex3i(400,0,0);
    glEnd();
    
    glColor3f(0.0, 1.0, 0.0); // green (Y Axis)
    glBegin(GL_LINES);
      glVertex3i(0,0,0);
      glVertex3i(0,600,0);
    glEnd();
    
    glColor3f(0.0, 0.0, 1.0); // blue (Z Axis)
    glBegin(GL_LINES);
      glVertex3i(0,0,-400);
      glVertex3i(0,0,400);
    glEnd();
  }

  if (drawOrigin){
    glColor3f(1.0, 0.0, 0.0); // red
    glPushMatrix();
    glTranslatef(0, 0, 0);
    glutSolidSphere(4, 5, 5);
    glPopMatrix();
  }

  //Draw LMC
  if (drawLMCs){
    //// LMC 3
    glColor3f(0.0, 1.0, 0.0); // green
    glBegin(GL_LINES);
      glVertex3i(LMC3_q1[0],LMC3_q1[1],LMC3_q1[2]);
      glVertex3i(LMC3_q2[0],LMC3_q2[1],LMC3_q2[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC3_q2[0],LMC3_q2[1],LMC3_q2[2]);
      glVertex3i(LMC3_q3[0],LMC3_q3[1],LMC3_q3[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC3_q3[0],LMC3_q3[1],LMC3_q3[2]);
      glVertex3i(LMC3_q4[0],LMC3_q4[1],LMC3_q4[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC3_q4[0],LMC3_q4[1],LMC3_q4[2]);
      glVertex3i(LMC3_q1[0],LMC3_q1[1],LMC3_q1[2]);
    glEnd();
    //// LMC 4
    glColor3f(0.0, 0.0, 0.0); // black
    glBegin(GL_LINES);
      glVertex3i(LMC4_q1[0],LMC4_q1[1],LMC4_q1[2]);
      glVertex3i(LMC4_q2[0],LMC4_q2[1],LMC4_q2[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC4_q2[0],LMC4_q2[1],LMC4_q2[2]);
      glVertex3i(LMC4_q3[0],LMC4_q3[1],LMC4_q3[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC4_q3[0],LMC4_q3[1],LMC4_q3[2]);
      glVertex3i(LMC4_q4[0],LMC4_q4[1],LMC4_q4[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC4_q4[0],LMC4_q4[1],LMC4_q4[2]);
      glVertex3i(LMC4_q1[0],LMC4_q1[1],LMC4_q1[2]);
    glEnd();
    //// LMC 1
    glColor3f(1.0, 0.0, 0.0); // red
    glBegin(GL_LINES);
      glVertex3i(LMC1_q1[0],LMC1_q1[1],LMC1_q1[2]);
      glVertex3i(LMC1_q2[0],LMC1_q2[1],LMC1_q2[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC1_q2[0],LMC1_q2[1],LMC1_q2[2]);
      glVertex3i(LMC1_q3[0],LMC1_q3[1],LMC1_q3[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC1_q3[0],LMC1_q3[1],LMC1_q3[2]);
      glVertex3i(LMC1_q4[0],LMC1_q4[1],LMC1_q4[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC1_q4[0],LMC1_q4[1],LMC1_q4[2]);
      glVertex3i(LMC1_q1[0],LMC1_q1[1],LMC1_q1[2]);
    glEnd();
    //// LMC 2
    glColor3f(0.0, 0.0, 1.0); // blue
    glBegin(GL_LINES);
      glVertex3i(LMC2_q1[0],LMC2_q1[1],LMC2_q1[2]);
      glVertex3i(LMC2_q2[0],LMC2_q2[1],LMC2_q2[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC2_q2[0],LMC2_q2[1],LMC2_q2[2]);
      glVertex3i(LMC2_q3[0],LMC2_q3[1],LMC2_q3[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC2_q3[0],LMC2_q3[1],LMC2_q3[2]);
      glVertex3i(LMC2_q4[0],LMC2_q4[1],LMC2_q4[2]);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(LMC2_q4[0],LMC2_q4[1],LMC2_q4[2]);
      glVertex3i(LMC2_q1[0],LMC2_q1[1],LMC2_q1[2]);
    glEnd();
  }

  //LMC Center
  if (drawSingleLMC){    
    glColor3f(1.0, 0.0, 0.0); // red
    glBegin(GL_LINES);
      glVertex3i(-40,0,15);
      glVertex3i(-40,0,-15);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(-40,0,-15);
      glVertex3i(40,0,-15);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(40,0,-15);
      glVertex3i(40,0,15);
    glEnd();  
    glBegin(GL_LINES);
      glVertex3i(40,0,15);
      glVertex3i(-40,0,15);
    glEnd();  
 }
}

//this function to draw using custom pointer for each LMC independently
void drawHand(HandDataTrack* HandDataTrack)
{
  if (!HandDataTrack) return;
    
  if(HandDataTrack)
  {            
    for(uint32_t h = 0; h <HandDataTrack->tracking_event->nHands; h++)
    {
      // Draw the hand
      LEAP_HAND *hand = HandDataTrack->hand;            

      //palm position
      glPushMatrix();
      glTranslatef(hand->palm.position.x, hand->palm.position.y, hand->palm.position.z);
      glutSolidSphere(2.8, 5, 5);
      glPopMatrix();

      //elbow
      glPushMatrix();
      glTranslatef(hand->arm.prev_joint.x, hand->arm.prev_joint.y, hand->arm.prev_joint.z);
      glutSolidSphere(2.8, 5, 5);
      glPopMatrix();            

      //wrist
      glPushMatrix();
      glTranslatef(hand->arm.next_joint.x, hand->arm.next_joint.y, hand->arm.next_joint.z);
      glutSolidSphere(2.8, 5, 5);
      glPopMatrix();
      
      //arm bone
      glBegin(GL_LINES);
        glVertex3f(hand->arm.prev_joint.x, hand->arm.prev_joint.y, hand->arm.prev_joint.z);
        glVertex3f(hand->arm.next_joint.x, hand->arm.next_joint.y, hand->arm.next_joint.z);
      glEnd();
      
      //Distal ends of bones for each digit
      for(int f = 0; f < 5; f++){
        LEAP_DIGIT finger = hand->digits[f];
        for(int b = 0; b < 4; b++){
          LEAP_BONE bone = finger.bones[b];
          
          glPushMatrix();
          glTranslatef(bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z);
          glutSolidSphere(2.8, 5, 5);
          glPopMatrix();

          glPushMatrix();
          glTranslatef(bone.next_joint.x, bone.next_joint.y, bone.next_joint.z);
          glutSolidSphere(2.8, 5, 5);
          glPopMatrix();
          
          glBegin(GL_LINES);
            glVertex3f(bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z);
            glVertex3f(bone.next_joint.x, bone.next_joint.y, bone.next_joint.z);
          glEnd();          
        }
      }
      //Hand Frame
      //Bottom Part
      glBegin(GL_LINES); 
        glVertex3f(hand->digits[0].bones[1].prev_joint.x,hand->digits[0].bones[1].prev_joint.y,hand->digits[0].bones[1].prev_joint.z);
        glVertex3f(hand->digits[1].bones[0].prev_joint.x,hand->digits[1].bones[0].prev_joint.y,hand->digits[1].bones[0].prev_joint.z);        
      glEnd();
      glBegin(GL_LINES);
        glVertex3f(hand->digits[1].bones[0].prev_joint.x,hand->digits[1].bones[0].prev_joint.y,hand->digits[1].bones[0].prev_joint.z);        
        glVertex3f(hand->digits[2].bones[0].prev_joint.x,hand->digits[2].bones[0].prev_joint.y,hand->digits[2].bones[0].prev_joint.z);
      glEnd();
      glBegin(GL_LINES);
        glVertex3f(hand->digits[2].bones[0].prev_joint.x,hand->digits[2].bones[0].prev_joint.y,hand->digits[2].bones[0].prev_joint.z);        
        glVertex3f(hand->digits[3].bones[0].prev_joint.x,hand->digits[3].bones[0].prev_joint.y,hand->digits[3].bones[0].prev_joint.z);
      glEnd();
      glBegin(GL_LINES);
        glVertex3f(hand->digits[3].bones[0].prev_joint.x,hand->digits[3].bones[0].prev_joint.y,hand->digits[3].bones[0].prev_joint.z);        
        glVertex3f(hand->digits[4].bones[0].prev_joint.x,hand->digits[4].bones[0].prev_joint.y,hand->digits[4].bones[0].prev_joint.z);
      glEnd();
      //Top Part
      glBegin(GL_LINES); 
        glVertex3f(hand->digits[0].bones[1].next_joint.x,hand->digits[0].bones[1].next_joint.y,hand->digits[0].bones[1].next_joint.z);
        glVertex3f(hand->digits[1].bones[0].next_joint.x,hand->digits[1].bones[0].next_joint.y,hand->digits[1].bones[0].next_joint.z);        
      glEnd();
      glBegin(GL_LINES);
        glVertex3f(hand->digits[1].bones[0].next_joint.x,hand->digits[1].bones[0].next_joint.y,hand->digits[1].bones[0].next_joint.z);        
        glVertex3f(hand->digits[2].bones[0].next_joint.x,hand->digits[2].bones[0].next_joint.y,hand->digits[2].bones[0].next_joint.z);
      glEnd();
      glBegin(GL_LINES);
        glVertex3f(hand->digits[2].bones[0].next_joint.x,hand->digits[2].bones[0].next_joint.y,hand->digits[2].bones[0].next_joint.z);        
        glVertex3f(hand->digits[3].bones[0].next_joint.x,hand->digits[3].bones[0].next_joint.y,hand->digits[3].bones[0].next_joint.z);
      glEnd();
      glBegin(GL_LINES);
        glVertex3f(hand->digits[3].bones[0].next_joint.x,hand->digits[3].bones[0].next_joint.y,hand->digits[3].bones[0].next_joint.z);        
        glVertex3f(hand->digits[4].bones[0].next_joint.x,hand->digits[4].bones[0].next_joint.y,hand->digits[4].bones[0].next_joint.z);
      glEnd();
      // End of draw hand               
    } 
  }
}

//this function to copy hand data from LMC pointer to custom pointer
HandDataTrack* HandDataCopy(const LEAP_TRACKING_EVENT* src, uint32_t device_id)
{
  if (!src) return NULL;

  HandDataTrack* dest = (HandDataTrack*)malloc(sizeof(HandDataTrack));
    if (!dest) return NULL;
  dest->device_id=device_id;

    dest->tracking_event = (LEAP_TRACKING_EVENT*) malloc (sizeof(LEAP_TRACKING_EVENT));
    if (!dest->tracking_event) {
        free(dest);
        return NULL;
      }
    dest->tracking_event->framerate = src->framerate;
    dest->tracking_event->nHands = src->nHands;
    dest->tracking_event->info.frame_id = src->info.frame_id;
    dest->tracking_event->info.timestamp = src->info.timestamp;
    
    dest->hand = (LEAP_HAND*) malloc (sizeof(LEAP_HAND));
    if (!dest->hand) {
        free(dest);
        return NULL;
      }

    dest->hand = src->pHands;

  return dest;  
}

//this function to copy joint position from custom pointer and set it in custom sequence of arrays
HandData CopyHandPosition (HandDataTrack* CopiedData){
  HandData ResultData;
  ResultData.device_id = CopiedData->device_id;
  ResultData.frame_id = CopiedData->tracking_event->info.frame_id;
  ResultData.framerate = CopiedData->tracking_event->framerate;
  ResultData.timestamp = CopiedData->tracking_event->info.timestamp;
  ResultData.nHands = CopiedData->tracking_event->nHands;

  ResultData.PositionData[0].x = CopiedData->hand->palm.position.x;
  ResultData.PositionData[0].y = CopiedData->hand->palm.position.y;
  ResultData.PositionData[0].z = CopiedData->hand->palm.position.z;
  ResultData.PositionData[1].x = CopiedData->hand->palm.normal.x;
  ResultData.PositionData[1].y = CopiedData->hand->palm.normal.y;
  ResultData.PositionData[1].z = CopiedData->hand->palm.normal.z;
  ResultData.PositionData[2].x = CopiedData->hand->palm.direction.x;
  ResultData.PositionData[2].y = CopiedData->hand->palm.direction.y;
  ResultData.PositionData[2].z = CopiedData->hand->palm.direction.z;
  ResultData.PositionData[3].x = CopiedData->hand->arm.prev_joint.x;
  ResultData.PositionData[3].y = CopiedData->hand->arm.prev_joint.y;
  ResultData.PositionData[3].z = CopiedData->hand->arm.prev_joint.z;
  ResultData.PositionData[4].x = CopiedData->hand->arm.next_joint.x;
  ResultData.PositionData[4].y = CopiedData->hand->arm.next_joint.y;
  ResultData.PositionData[4].z = CopiedData->hand->arm.next_joint.z;

  int digit=0;
  for(int i=5; i<30; i=i+5){
    ResultData.PositionData[i].x = CopiedData->hand->digits[digit].bones[0].prev_joint.x;
    ResultData.PositionData[i].y = CopiedData->hand->digits[digit].bones[0].prev_joint.y;
    ResultData.PositionData[i].z = CopiedData->hand->digits[digit].bones[0].prev_joint.z;
    ResultData.PositionData[i+1].x = CopiedData->hand->digits[digit].bones[0].next_joint.x;
    ResultData.PositionData[i+1].y = CopiedData->hand->digits[digit].bones[0].next_joint.y;
    ResultData.PositionData[i+1].z = CopiedData->hand->digits[digit].bones[0].next_joint.z;
    ResultData.PositionData[i+2].x = CopiedData->hand->digits[digit].bones[1].next_joint.x;
    ResultData.PositionData[i+2].y = CopiedData->hand->digits[digit].bones[1].next_joint.y;
    ResultData.PositionData[i+2].z = CopiedData->hand->digits[digit].bones[1].next_joint.z;
    ResultData.PositionData[i+3].x = CopiedData->hand->digits[digit].bones[2].next_joint.x;
    ResultData.PositionData[i+3].y = CopiedData->hand->digits[digit].bones[2].next_joint.y;
    ResultData.PositionData[i+3].z = CopiedData->hand->digits[digit].bones[2].next_joint.z;
    ResultData.PositionData[i+4].x = CopiedData->hand->digits[digit].bones[3].next_joint.x;
    ResultData.PositionData[i+4].y = CopiedData->hand->digits[digit].bones[3].next_joint.y;
    ResultData.PositionData[i+4].z = CopiedData->hand->digits[digit].bones[3].next_joint.z;
    digit++;
  }
  return ResultData;
}

int HandDataSendUDP(SOCKET Socket,HandData HandData, struct sockaddr_in serverAddr){
  std::ostringstream buffer;

  buffer << HandData.device_id << "\t" << HandData.frame_id << "\t"
          << to_string_with_precision(HandData.framerate) << "\t" << HandData.timestamp << "\t";

  if (HandData.nHands == 0) {
      buffer << "\n";
  } else {
      for (int i = 0; i < 1; ++i) { //changed 1 to save only palm, it must be 5
          buffer << to_string_with_precision(HandData.PositionData[i].x) << ","
                  << to_string_with_precision(HandData.PositionData[i].y) << ","
                  << to_string_with_precision(HandData.PositionData[i].z) << "\t";
      }      
      //Only send Palm comment between these
      for (int i = 5; i < 30; i += 5) {
          for (int j = 0; j < 5; ++j) {
              buffer << to_string_with_precision(HandData.PositionData[i + j].x) << ","
                      << to_string_with_precision(HandData.PositionData[i + j].y) << ","
                      << to_string_with_precision(HandData.PositionData[i + j].z) << "\t";
          }
      }
      //Only send Palm comment between these
      buffer << "\n";
  }

  // Get the content of the buffer as an std::string
  std::string content = buffer.str();

  int stat = sendto(Socket, content.c_str(), content.size(), 0, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
  return stat;
}