#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include "multiLMC_process.h"

#include <fstream>
#include <iostream>
#include <string>
#include <windows.h>

//Initilization of writing data to file for each device, return filepath in std::str
std::ofstream initSaveFile(std::string project_path,  int numDevice){
    std::ofstream fileDevice;
    std::string file_pathDev;
    
    file_pathDev = project_path + "\\data\\DataDevice" + std::to_string(numDevice) + ".txt";

    fileDevice.open(file_pathDev);
    if (!fileDevice.is_open()) {
      std::cout << "Unable to open the file." << std::endl;
    } 
    else {
      fileDevice << "Device ID\tFrame ID\tFrame Rate\tTime Stamp\t";
      fileDevice << "Palm Position.x,Palm Position.y,Palm Position.z\tPalm Normal.x,Palm Normal.y,Palm Normal.z\tPalm Direction.x,Palm Direction.y,Palm Direction.z\t";
      fileDevice << "Arm Proximal.x,Arm Proximal.y,Arm Proximal.z\tArm Distal.x,Arm Distal.y,Arm Distal.z\t";
      for (uint32_t j = 0; j < 5; j++) {
        fileDevice << "Finger" << j + 1 << "_Meta.x,Finger" << j + 1 << "_Meta.y,Finger" << j + 1 << "_Meta.z\t";
        fileDevice << "Finger" << j + 1 << "_Proxi.x,Finger" << j + 1 << "_Proxi.y,Finger" << j + 1 << "_Proxi.z\t";
        fileDevice << "Finger" << j + 1 << "_Inter.x,Finger" << j + 1 << "_Inter.y,Finger" << j + 1 << "_Inter.z\t";
        fileDevice << "Finger" << j + 1 << "_Dist.x,Finger" << j + 1 << "_Dist.y,Finger" << j + 1 << "_Dist.z\t";
        fileDevice << "Finger" << j + 1 << "_End.x,Finger" << j + 1 << "_End.y,Finger" << j + 1 << "_End.z\t";
      }
      fileDevice << "\n";
    }    
    return fileDevice;
}

//Function to save data of each device.
void SaveDataDevice(std::ofstream &fileDevice, HandData HandData) {    
  fileDevice << HandData.device_id << "\t" << HandData.frame_id << "\t" << HandData.framerate << "\t" << HandData.timestamp << "\t";

  if (HandData.nHands == 0)
    fileDevice << "\n";
  else {
    fileDevice << HandData.PositionData[0].x << "," << HandData.PositionData[0].y << "," << HandData.PositionData[0].z << "\t";
    fileDevice << HandData.PositionData[1].x << "," << HandData.PositionData[1].y << "," << HandData.PositionData[1].z << "\t";
    fileDevice << HandData.PositionData[2].x << "," << HandData.PositionData[2].y << "," << HandData.PositionData[2].z << "\t";
    fileDevice << HandData.PositionData[3].x << "," << HandData.PositionData[3].y << "," << HandData.PositionData[3].z << "\t";
    fileDevice << HandData.PositionData[4].x << "," << HandData.PositionData[4].y << "," << HandData.PositionData[4].z << "\t";   
    int digit = 0;
    for (int i = 5; i < 30; i += 5) {
      fileDevice << HandData.PositionData[i].x << "," << HandData.PositionData[i].y << "," << HandData.PositionData[i].z << "\t";
      fileDevice << HandData.PositionData[i + 1].x << "," << HandData.PositionData[i + 1].y << "," << HandData.PositionData[i + 1].z << "\t";
      fileDevice << HandData.PositionData[i + 2].x << "," << HandData.PositionData[i + 2].y << "," << HandData.PositionData[i + 2].z << "\t";
      fileDevice << HandData.PositionData[i + 3].x << "," << HandData.PositionData[i + 3].y << "," << HandData.PositionData[i + 3].z << "\t";
      fileDevice << HandData.PositionData[i + 4].x << "," << HandData.PositionData[i + 4].y << "," << HandData.PositionData[i + 4].z << "\t";
      digit++;
    }
    fileDevice << "\n";
  }
}    
    
//calculate vector from coodinates (a, b : coordinate)
HAND_VECTOR VectCalc(HAND_VECTOR a, HAND_VECTOR b){
    HAND_VECTOR output;
    output.x = b.x-a.x;
    output.y = b.y-a.y;
    output.z = b.z-a.z;
    return output; 
}

//calculate result of dot product from vector (a, b : vector) / coordinates with base from center (0,0,0)
float VectDotProd(HAND_VECTOR a, HAND_VECTOR b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

HAND_VECTOR VectCrossProd(HAND_VECTOR a, HAND_VECTOR b){
    HAND_VECTOR r;
    r.x= a.y*b.z - a.z*b.y;
    r.y= a.z*b.x - a.x*b.z;
    r.z= a.x*b.y - a.y*b.x;
    return r;
}

//calculate magnitude of vector from coordinates (a, b : coordinate)
float MagVect_Coord(HAND_VECTOR a, HAND_VECTOR b){
    return sqrt((b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y) + (b.z-a.z)*(b.z-a.z));
}

//calculate degree from Coordinate (a, b, c : coordinate) 
float DegCalc_Coord(HAND_VECTOR a, HAND_VECTOR b, HAND_VECTOR c){
    float divider = MagVect_Coord(a, b) * MagVect_Coord(b, c); 
    if (divider == 0) return 0;
    else {
        return acos(VectDotProd(VectCalc(a,b), VectCalc(b,c)) / divider);
    }
}

//calculate degree from Coordinate 4 point (a, b, c, d : coordinate)
float DegCalc_Coord4point(HAND_VECTOR a, HAND_VECTOR b, HAND_VECTOR c, HAND_VECTOR d){
    float divider = MagVect_Coord(a, b) * MagVect_Coord(c, d); 
    if (divider == 0) return 0;
    else {        
        return acos(VectDotProd(VectCalc(a,b), VectCalc(c,d)) / divider);
    }
}

//calculate magnitude of vector (a : vector)
float MagVect(HAND_VECTOR a){
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

HAND_VECTOR VectUnit(HAND_VECTOR a){
    HAND_VECTOR r;
    float a_length;
    a_length = MagVect(a);
    r.x=a.x/a_length;
    r.y=a.y/a_length;
    r.z=a.z/a_length;
    return r;
}

//calculate degree from vector (a, b : vector)
float DegCalc_Vect(HAND_VECTOR a, HAND_VECTOR b){
    float divider = MagVect(a) * MagVect(b); 
    if (divider == 0) return 0;
    else {
        return acos(VectDotProd(a,b) / divider);
    }
}

void RadtoDeg (float radians[5], float degrees[5]){
    for (int i=0 ; i<5; i++){
        degrees[i] = radians[i]* 57.32484;
    }
}

float wristflex_angle(HAND_VECTOR a, HAND_VECTOR b, HAND_VECTOR c, HAND_VECTOR d){
    float divider = MagVect_Coord(a, b) * MagVect_Coord(c, d); 
    int sign=1;
    if (divider == 0) return 0;
    else {
        if (VectCalc(c,d).y  < VectCalc(a,b).y ) sign = 1;
        else sign = -1;        
        return acos(VectDotProd(VectCalc(a,b), VectCalc(c,d)) / divider) * sign;
    }
}

float wristflex_angle2(HAND_VECTOR a, HAND_VECTOR b, HAND_VECTOR c){
    float divider = MagVect_Coord(a, b) * MagVect(c); 
    int sign=1;
    if (divider == 0) return 0;
    else {
        if (c.y  < VectCalc(a,b).y ) sign = 1;
        else sign = -1;        
        return acos(VectDotProd(VectCalc(a,b), c) / divider) * sign;
    }
}

float wristdev_angle(HAND_VECTOR a, HAND_VECTOR b, HAND_VECTOR c, HAND_VECTOR d){
    float divider = MagVect_Coord(a, b) * MagVect_Coord(c, d); 
    if (divider == 0) return 0;
    else {        
        return acos(VectDotProd(VectCalc(a,b), VectCalc(c,d)) / divider) - 1.449 ; //minus degree
    }
}

float elbowpronation(HAND_VECTOR a, HAND_VECTOR b){
    HAND_VECTOR c,d;    
    c.x = 0; c.y = 500; c.z=0;
    d.x = 0; d.y = 0; d.z=0;
    
    float divider = MagVect_Coord(a, b) * MagVect_Coord(c, d);     
    if (divider == 0) return 0;
    else {                
        return acos(VectDotProd(VectCalc(a,b), VectCalc(c,d)) / divider) - 1.571; //minus degree
    }
}

float elbowflex(HAND_VECTOR a, HAND_VECTOR b){
    HAND_VECTOR c,d;    
    c.x = 0; c.y = 500; c.z=0;
    d.x = 0; d.y = 0; d.z=0;    

    float divider = MagVect_Coord(a, b) * MagVect_Coord(c, d);     
    if (divider == 0) return 0;
    else {                
        return acos(VectDotProd(VectCalc(a,b), VectCalc(c,d)) / divider);
    }
}


void angle_calc(HandData handdata, float angle[5], float wrist_angle[2], float elbow_angle[2]){    
    //thumb abd, thumb flex, finger flex, middle flex, ringlittle flex
    HAND_VECTOR n1,n2;
    n1 =  VectCrossProd(VectCalc(handdata.PositionData[11],handdata.PositionData[7]),VectCalc(handdata.PositionData[6],handdata.PositionData[7])); 
    n2 =  VectCrossProd(VectCalc(handdata.PositionData[6],handdata.PositionData[0]),VectCalc(handdata.PositionData[11],handdata.PositionData[0])); 
    
    angle[0]= DegCalc_Vect(n1,n2)*1.2; //Gain 1.2
    angle[1]=DegCalc_Coord4point(handdata.PositionData[6],handdata.PositionData[11],handdata.PositionData[6],handdata.PositionData[7]);
    //angle[1]=DegCalc_Coord4point(handdata.PositionData[7],handdata.PositionData[8],handdata.PositionData[8],handdata.PositionData[9]);    
    angle[2]=DegCalc_Coord(handdata.PositionData[10],handdata.PositionData[11],handdata.PositionData[12]);
    angle[3]=DegCalc_Coord(handdata.PositionData[15],handdata.PositionData[16],handdata.PositionData[17]);
    angle[4]=DegCalc_Coord(handdata.PositionData[25],handdata.PositionData[26],handdata.PositionData[27]);
    //printf("Angle: %.4f, %.4f, %.4f, %.4f, %.4f\n", angle[0]*57.2958, angle[1]*57.2958, angle[2]*57.2958, angle[3]*57.2958, angle[4]*57.2958);

    /*
    Wrist
    0 => flexion +38 to -40 extension
    1 => right/ulnar deviation +38 to -28 left/radial deviation 
    Elbow
    0 => pronation +13 (ccw) to -53 supination (cw)  
    1 => initial 90 degree to ....   
    */
    wrist_angle[0] = wristflex_angle(handdata.PositionData[3],handdata.PositionData[4],handdata.PositionData[4],handdata.PositionData[0]);
    wrist_angle[1] = wristdev_angle(handdata.PositionData[3],handdata.PositionData[4],handdata.PositionData[5],handdata.PositionData[25]);

    elbow_angle[0] = elbowpronation(handdata.PositionData[5],handdata.PositionData[25]);
    elbow_angle[1] = elbowflex(handdata.PositionData[3],handdata.PositionData[4]);
}


int SendCmd(float AngleData) {
    if (isnan(AngleData)) {
        return 1;
    } else {
        if (AngleData < 0.0) {
            return 1;
        }
        else if(AngleData > 1.5){
            return 254;
        }
        else {
            float result = AngleData * 183.45; // 183.45 = 255 (hex) / 80 (maximum degree)  162.42 =255 (90 degree)
            //float result = ((AngleData * 224.87) - 66.56); 
            if (result > 254) result = 254;
            else if (result < 0) result = 1;
            return (int)result;
        }
    }
}

int SendCmdThumb(float AngleData) {
    if (isnan(AngleData)) {
        return 1;
    } else {
        if (AngleData < 0.0) {
            return 1;
        }
        else if(AngleData > 1.5){
            return 254;
        } 
        else {
            float result = (510 - (AngleData * 584.46));
            //float result = (382.57 - (AngleData * 365.33));
            if (result > 254) result = 254;
            else if (result < 0) result = 1;
            return (int)result;
        }
    }
}

int SendCmdThumbAb(float AngleData) {
    if (isnan(AngleData)) {
        return 1;
    } else {
        if (AngleData < 0.0) {
            return 1;
        }
        else if(AngleData > 1.5){
            return 254;
        } 
        else {
            float result = AngleData * 183.45;
            //float result = ((AngleData * 180.47) - 28.33);
            if (result > 254) result = 254;
            else if (result < 0) result = 1;
            return (int)result;
        }
    }
}

void ProcessUDP(float UDP_angle[9], float angle[5], float wrist_angle[2], float elbow_angle[2]){
    /*
    //Send in ASCII
    UDP_angle[0]=SendCmdThumbAb(angle[0]);
    UDP_angle[1]=SendCmdThumb(angle[1]);
    UDP_angle[2]=SendCmd(angle[2]);
    UDP_angle[3]=SendCmd(angle[3]);
    UDP_angle[4]=SendCmd(angle[4]);
    */
    //Send in Rad
    UDP_angle[0]=SendCmdThumbAb(angle[0])*0.00616;
    UDP_angle[1]=SendCmdThumb(angle[1])*0.00616;
    UDP_angle[2]=SendCmd(angle[2])*0.00616;
    UDP_angle[3]=SendCmd(angle[3])*0.00616;
    UDP_angle[4]=SendCmd(angle[4])*0.00616;

    UDP_angle[5]=elbow_angle[0];
    UDP_angle[6]=elbow_angle[1];
    UDP_angle[7]=wrist_angle[0];
    UDP_angle[8]=wrist_angle[1];
}


