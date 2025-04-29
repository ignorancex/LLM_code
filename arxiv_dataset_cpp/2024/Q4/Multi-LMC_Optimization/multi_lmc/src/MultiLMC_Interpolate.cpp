#include "../ext_lib/alglib/header/stdafx.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "../ext_lib/alglib/header/interpolation.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>
#include "Linking/multiLMC_process.h"

#include <filesystem>

using namespace alglib;
using namespace std;
namespace fs = std::filesystem;

int ReadDataFile(std::string file_path, HandData datareading[]);
int ReadDataFileAll(std::string file_path, HandData datareading1[], HandData datareading2[], HandData datareading3[], HandData datareading4[]);
int Interpolate(HandData LMC[], HandData InterLMC[], int numDataLMC);
void SaveHandDataOffline (std::string filepath, HandData HandData[], int nHandData);
HandData fillstart(HandData basedata, int numdev);
std::vector<std::vector<double>> interpolation_process(std::vector<std::vector<double>> DataInput, int datalength, int tstartinterp);

HandData datareading[20000];
HandData LMC1[5000], LMC2[5000], LMC3[5000], LMC4[5000];
int ndataLMC1 = 0, ndataLMC2 = 0, ndataLMC3 = 0, ndataLMC4 = 0;
HandData InterLMC1[5000], InterLMC2[5000], InterLMC3[5000], InterLMC4[5000];
int nInterpLMC1 = 0, nInterpLMC2 = 0, nInterpLMC3 = 0, nInterpLMC4 = 0;
HandData InterLMC_all[20000];
int nInterpLMC_all;

bool HandDataisNaN(HandData datareading){
  bool result = false;
  bool isfirstdatathere = false;

  if (!std::isnan(datareading.PositionData[0].x)) isfirstdatathere = true;
  else isfirstdatathere = false;

  for (int k = 1; k < 30; ++k) {
    if(std::isnan(datareading.PositionData[k].x) && isfirstdatathere) {
      result = true;
      break;
    }
    if(std::isnan(datareading.PositionData[k].x) && isfirstdatathere) {
      result = true;
      break;
    }
    if(std::isnan(datareading.PositionData[k].z) && isfirstdatathere) {
      result = true;
      break;
    }
  }
  return result;
}

//use to mark start time when receiving data for first time
uint64_t tstart, tend;

int main(int argc, char **argv)
{
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " <folder_path>" << endl;
    return 1;
  }

  fs::path folderPath(argv[1]);

  if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
    cerr << "Invalid directory: " << folderPath << endl;
    return 1;
  }

  // Iterate over all files in the directory
  for (const auto& entry : fs::directory_iterator(folderPath)) {
    if (fs::is_regular_file(entry)) 
    {
      //Number of data in file read
      int numData;

      tstart = 0;
      tend = 0;
      
      std::string filename = entry.path().filename().string();
      std::string parentpath = entry.path().parent_path().string();

      std::string file_pathLMC = parentpath + "\\" + filename;
      
      ndataLMC1 = 0; ndataLMC2 = 0; ndataLMC3 = 0; ndataLMC4 = 0;
      numData = ReadDataFileAll(file_pathLMC, LMC1, LMC2, LMC3, LMC4);

      std::cout << "----- FINISH READING DATA "<< filename << " -----" << std::endl;
      std::cout << "ndataLMC1: " << ndataLMC1 << "  ndataLMC2: " << ndataLMC2 
                << "  ndataLMC3: " << ndataLMC3 << "  ndataLMC4: " << ndataLMC4 << std::endl;  
      std::cout << "Time Start: " << tstart << " ,Time End: " << tend << "\n" << std::endl;
      
      nInterpLMC1 = 0; nInterpLMC2 = 0; nInterpLMC3 = 0; nInterpLMC4 = 0;
      nInterpLMC1 = Interpolate(LMC1, InterLMC1, ndataLMC1);
      std::cout << "----- FINISH INTERPOLATING LMC1 -----" << std::endl;
      nInterpLMC2 = Interpolate(LMC2, InterLMC2, ndataLMC2);
      std::cout << "----- FINISH INTERPOLATING LMC2 -----" << std::endl;
      nInterpLMC3 = Interpolate(LMC3, InterLMC3, ndataLMC3);
      std::cout << "----- FINISH INTERPOLATING LMC3 -----" << std::endl;
      nInterpLMC4 = Interpolate(LMC4, InterLMC4, ndataLMC4);
      std::cout << "----- FINISH INTERPOLATING LMC4 -----" << std::endl;

      nInterpLMC_all = min(min(nInterpLMC1,nInterpLMC2),min(nInterpLMC3,nInterpLMC4));
      for (int i=0; i<nInterpLMC_all; i++){
        InterLMC_all [(i*4)] = InterLMC1[i];
        InterLMC_all [(i*4) + 1] = InterLMC2[i];
        InterLMC_all [(i*4) + 2] = InterLMC3[i];
        InterLMC_all [(i*4) + 3] = InterLMC4[i];
      };

      SaveHandDataOffline(parentpath + "\\Interp_" + filename, InterLMC_all, 4*nInterpLMC_all);

      std::cout << "----- FINISH INTERPOLATING "<< filename << " -----" << std::endl;
      std::cout << "nInterpolation_LMC1: " << nInterpLMC1 << "  nInterpolation_LMC2: " << nInterpLMC2 << std::endl;
      std::cout << "nInterpolation_LMC3: " << nInterpLMC3 << "  nInterpolation_LMC4: " << nInterpLMC4 << std::endl;
      std::cout << "nInterpolation_All: " << nInterpLMC_all << "\n" <<std::endl;
  
    }
  }

  return 0;
}


int Interpolate(HandData LMC[], HandData InterLMC[], int numData){
  //buffer of 3 row of handdata read
  HandData buff_handata [3]; 

  //for arranging the handdata to be interpolated
  HandData window_handdata [3];

  //iteration for data, Used in streaming to count how many
  //data read at current time
  int i_online=0;
  
  //iteration to track number of result of interpolated data
  int i_interp=0;

  //this looping not used in streaming
  while(i_online<numData){

    buff_handata  [i_online%3] = LMC[i_online];

    if(i_online%6==2) {
      window_handdata [0]= buff_handata [0];
      window_handdata [1]= buff_handata [1];
      window_handdata [2]= buff_handata [2];
    }

    if(i_online!=0 && i_online%2==0){
      
      if(i_online%6==4) {
        window_handdata [0]= window_handdata [2];
        window_handdata [1]= buff_handata [0];
        window_handdata [2]= buff_handata [1];        
      }
      if(i_online%6==0) {
        window_handdata [0]= window_handdata [2];
        window_handdata [1]= buff_handata [2];
        window_handdata [2]= buff_handata [0];        
      }
            
      int start_interp = i_interp;
      for (int k = 0; k < 30; ++k){
        std::vector<std::vector<double>> InputData;
        InputData.clear();
        for (int i = 0; i <= 2; ++i) {
          std::vector<double> rowtemp;  
          rowtemp.push_back((window_handdata[i].timestamp - tstart) / 1000);    
          rowtemp.push_back(window_handdata[i].PositionData[k].x);
          rowtemp.push_back(window_handdata[i].PositionData[k].y);
          rowtemp.push_back(window_handdata[i].PositionData[k].z);
          InputData.push_back(rowtemp);
          rowtemp.clear();
        }
        
        std::vector<std::vector<double>> ResultInterp;
        ResultInterp.clear();
        ResultInterp = interpolation_process(InputData,3,start_interp);
                  
        for(int j=0; j<ResultInterp.size()-1; j++){
          InterLMC[start_interp+j].device_id = window_handdata[0].device_id;
          InterLMC[start_interp+j].frame_id = 0;
          InterLMC[start_interp+j].framerate = 100;
          InterLMC[start_interp+j].timestamp = ResultInterp[j][0] * 1000 + tstart; 
          InterLMC[start_interp+j].PositionData[k].x = ResultInterp[j][1]; 
          InterLMC[start_interp+j].PositionData[k].y = ResultInterp[j][2]; 
          InterLMC[start_interp+j].PositionData[k].z = ResultInterp[j][3];            

          if (k==0) ++i_interp;          
        }                                 
      }
    }        
    ++i_online;
  }

  return i_interp;
}

std::vector<std::vector<double>> interpolation_process(std::vector<std::vector<double>> DataInput, int datalength, int tstartinterp){
  bool NA_stat = false;

  //vector result of interpolation
  //need to be cleared before used again
  std::vector<std::vector<double>> intp_result;
  intp_result.clear();
  
  //array data of interpolant
  //need to be cleared before used again
  real_1d_array intp_tstamp, intp_x, intp_y, intp_z;
  intp_tstamp.setlength(datalength);
  intp_x.setlength(datalength);
  intp_y.setlength(datalength);
  intp_z.setlength(datalength);

  //assigning interpolant temp data with data to be interpolated
  for (int i = 0; i < datalength; ++i) {
    intp_tstamp[i] =  DataInput[i][0];
    intp_x[i] = DataInput[i][1];
    intp_y[i] = DataInput[i][2];
    intp_z[i] = DataInput[i][3];
  }

  if (std::isnan(intp_x[0]) && std::isnan(intp_x[1]) && std::isnan(intp_x[2])) NA_stat = true;
  else if (!std::isnan(intp_x[0]) && std::isnan(intp_x[1]) && std::isnan(intp_x[2])){
    NA_stat = false;
    for (int i = 1; i < datalength; ++i) {    
    intp_x[i] = DataInput[0][1];
    intp_y[i] = DataInput[0][2];
    intp_z[i] = DataInput[0][3];
    }    
  }
  else if (!std::isnan(intp_x[0]) && !std::isnan(intp_x[1]) && std::isnan(intp_x[2])){
    NA_stat = false;
    intp_x[2] = DataInput[1][1];
    intp_y[2] = DataInput[1][2];
    intp_z[2] = DataInput[1][3];
  }
  else if (std::isnan(intp_x[0]) && !std::isnan(intp_x[1]) && std::isnan(intp_x[2])){
    NA_stat = false;
    intp_x[0] = DataInput[1][1];
    intp_y[0] = DataInput[1][2];
    intp_z[0] = DataInput[1][3];

    intp_x[2] = DataInput[1][1];
    intp_y[2] = DataInput[1][2];
    intp_z[2] = DataInput[1][3];    
  }
  else if (std::isnan(intp_x[0]) && !std::isnan(intp_x[1]) && !std::isnan(intp_x[2])){
    NA_stat = false;
    intp_x[0] = DataInput[1][1];
    intp_y[0] = DataInput[1][2];
    intp_z[0] = DataInput[1][3];
  }
  else if (std::isnan(intp_x[0]) && std::isnan(intp_x[1]) && !std::isnan(intp_x[2])){
    NA_stat = false;
    for (int i = 0; i < 2; ++i) {    
    intp_x[i] = DataInput[2][1];
    intp_y[i] = DataInput[2][2];
    intp_z[i] = DataInput[2][3];
    }    
  }
  
  //every 10ms or 100Hz
  double tstamp_start = tstartinterp * 10;
  
  spline1dinterpolant s_x,s_y,s_z;
  if (!NA_stat){
    spline1dbuildcubic(intp_tstamp, intp_x, s_x);
    spline1dbuildcubic(intp_tstamp, intp_y, s_y);
    spline1dbuildcubic(intp_tstamp, intp_z, s_z);
  }
  int i = 0;
  while(tstamp_start + (i*10) < DataInput[2][0])
  {
    std::vector<double> rowtemp;  
    rowtemp.push_back(tstamp_start + (i*10)); 

    if (!NA_stat){
      rowtemp.push_back(spline1dcalc(s_x, rowtemp[0]));
      rowtemp.push_back(spline1dcalc(s_y, rowtemp[0]));
      rowtemp.push_back(spline1dcalc(s_z, rowtemp[0]));
    }
    else {
      rowtemp.push_back(std::numeric_limits<double>::quiet_NaN());
      rowtemp.push_back(std::numeric_limits<double>::quiet_NaN());
      rowtemp.push_back(std::numeric_limits<double>::quiet_NaN());
    }  
    intp_result.push_back(rowtemp);        
    
    rowtemp.clear();
    ++i;
  }

  return intp_result;
}

int ReadDataFile(std::string file_path, HandData datareading[]) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file" << std::endl;
    return -1;
  }

  int iteration = 0;
  std::string line;

  // Read header line and discard
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::istringstream iss(line);

    std::string token;
    std::getline(iss, token, '\t');
    datareading[iteration].device_id = std::stoi(token);
    std::getline(iss, token, '\t');
    datareading[iteration].frame_id = std::stoll(token);
    std::getline(iss, token, '\t');
    datareading[iteration].framerate = std::stod(token);
    std::getline(iss, token, '\t');
    datareading[iteration].timestamp = std::stoll(token);

    for (int i = 0; i < 30; i++) {
        std::getline(iss, token, ',');
        datareading[iteration].PositionData[i].x = std::stod(token);
        std::getline(iss, token, ',');
        datareading[iteration].PositionData[i].y = std::stod(token);
        std::getline(iss, token, '\t');
        datareading[iteration].PositionData[i].z = std::stod(token);
    }
    iteration++;
    }
  file.close();

  return iteration;
}

int ReadDataFileAll(std::string file_path, HandData datareading1[], HandData datareading2[], HandData datareading3[], HandData datareading4[]){
  std::ifstream file(file_path);
  if (!file.is_open()) {
      std::cerr << "Failed to open file" << std::endl;
      return -1;
  }

  int iteration = 0;
  std::string line;

  int iter1 = 0, iter2 = 0, iter3 = 0, iter4 = 0;

  // Read header line and discard
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::istringstream iss(line);

    std::string token;
    std::getline(iss, token, '\t');
    datareading[iteration].device_id = std::stoi(token);
    std::getline(iss, token, '\t');
    datareading[iteration].frame_id = std::stoll(token);
    std::getline(iss, token, '\t');
    datareading[iteration].framerate = std::stod(token);
    std::getline(iss, token, '\t');
    datareading[iteration].timestamp = std::stoll(token);

    for (int i = 0; i < 30; i++) {
      std::getline(iss, token, ',');
      if (!token.empty()) {
        try { datareading[iteration].PositionData[i].x = std::stod(token); }
        catch (const std::invalid_argument& e) {
          iteration--;
          break;
        }
      }
      else datareading[iteration].PositionData[i].x = std::numeric_limits<double>::quiet_NaN();
      
      std::getline(iss, token, ',');
      if (!token.empty()) {
        try { datareading[iteration].PositionData[i].y = std::stod(token); }
        catch (const std::invalid_argument& e) {
          iteration--;
          break;
        }
      }      
      else datareading[iteration].PositionData[i].y = std::numeric_limits<double>::quiet_NaN();
      
      std::getline(iss, token, '\t');
      if (!token.empty()) {
        try { datareading[iteration].PositionData[i].z = std::stod(token); }
        catch (const std::invalid_argument& e) {
          iteration--;
          break;
        }
      }      
      else datareading[iteration].PositionData[i].z = std::numeric_limits<double>::quiet_NaN();
    }

    if (iteration==0) {
      tstart =  datareading[iteration].timestamp;            
      
      //Handling and augmenting first data each HandData variable
      if (datareading[iteration].device_id == 1) {
        datareading2[iter2] = fillstart(datareading[iteration],2);
        iter2++;
        datareading3[iter3] = fillstart(datareading[iteration],3);
        iter3++;
        datareading4[iter4] = fillstart(datareading[iteration],4);
        iter4++;
      }
      else if (datareading[iteration].device_id == 2) {
        datareading1[iter1] = fillstart(datareading[iteration],1);
        iter1++;
        datareading3[iter3] = fillstart(datareading[iteration],3);
        iter3++;
        datareading4[iter4] = fillstart(datareading[iteration],4);
        iter4++;
      }
      else if (datareading[iteration].device_id == 3) {
        datareading1[iter1] = fillstart(datareading[iteration],1);
        iter1++;
        datareading2[iter2] = fillstart(datareading[iteration],2);
        iter2++;
        datareading4[iter4] = fillstart(datareading[iteration],4);
        iter4++;
      }
      else if (datareading[iteration].device_id == 4) {
        datareading1[iter1] = fillstart(datareading[iteration],1);
        iter1++;
        datareading2[iter2] = fillstart(datareading[iteration],2);
        iter2++;
        datareading3[iter3] = fillstart(datareading[iteration],3);
        iter3++;
      }      
    }
    
    else if (iteration > 0 && datareading[iteration].timestamp > tstart) 
    {
      //Handling if the second data has very small timestamp
      if (datareading1[1].timestamp-datareading1[0].timestamp < 10000) 
        datareading1[1].timestamp = datareading1[0].timestamp + 10000;
      if (datareading2[1].timestamp-datareading2[0].timestamp < 10000) 
        datareading2[1].timestamp = datareading2[0].timestamp + 10000;
      if (datareading3[1].timestamp-datareading3[0].timestamp < 10000) 
        datareading3[1].timestamp = datareading3[0].timestamp + 10000;
      if (datareading4[1].timestamp-datareading4[0].timestamp < 10000) 
        datareading4[1].timestamp = datareading4[0].timestamp + 10000;
        
      //Move to respective HandData variable
      if (datareading[iteration].device_id == 1) {
          datareading1[iter1] = datareading[iteration];        
          if(datareading1[iter1].timestamp - datareading1[iter1-1].timestamp < 10000
            || HandDataisNaN(datareading1[iter1])) iter1--;
          iter1++;
      }
      if (datareading[iteration].device_id == 2) {
          datareading2[iter2] = datareading[iteration];
          if(datareading2[iter2].timestamp - datareading2[iter2-1].timestamp < 10000
            || HandDataisNaN(datareading2[iter2])) iter2--;
          iter2++;
      }
      if (datareading[iteration].device_id == 3) {
          datareading3[iter3] = datareading[iteration];
          if(datareading3[iter3].timestamp - datareading3[iter3-1].timestamp < 10000
            || HandDataisNaN(datareading3[iter3])) iter3--;
          iter3++;
      }
      if (datareading[iteration].device_id == 4) {
          datareading4[iter4] = datareading[iteration];
          if(datareading4[iter4].timestamp - datareading4[iter4-1].timestamp < 10000
            || HandDataisNaN(datareading4[iter4])) iter4--;
          iter4++;
      } 
    }                           

    iteration++;        
  }  
  int iterMax = max(max(iter1,iter2),max(iter3,iter4));
  int iterMin;
  if (iter1 < 3) iterMin = min(iter2,min(iter3,iter4));
  else if (iter2 < 3) iterMin = min(iter1,min(iter3,iter4));
  else if (iter3 < 3) iterMin = min(min(iter1,iter2),iter4);
  else if (iter4 < 3) iterMin = min(min(iter1,iter2),iter3);
  else iterMin = min(min(iter1,iter2),min(iter3,iter4));

  tend = datareading[iteration-1].timestamp;
 
  ndataLMC1 = iter1;
  ndataLMC2 = iter2;
  ndataLMC3 = iter3;
  ndataLMC4 = iter4;

  if(ndataLMC1 < 3) {
    ndataLMC1 = iterMin;
    for (int i = 1; i < ndataLMC1; i++){
      if(ndataLMC2==iterMin) datareading1[i] = fillstart(datareading2[i],1);
      if(ndataLMC3==iterMin) datareading1[i] = fillstart(datareading3[i],1);
      if(ndataLMC4==iterMin) datareading1[i] = fillstart(datareading4[i],1);      
    }
  }
  if(ndataLMC2 < 3) {
    ndataLMC2 = iterMin;
    for (int i = 1; i < ndataLMC2; i++){
      if(ndataLMC1==iterMin) datareading2[i] = fillstart(datareading1[i],2);
      if(ndataLMC3==iterMin) datareading2[i] = fillstart(datareading3[i],2);
      if(ndataLMC4==iterMin) datareading2[i] = fillstart(datareading4[i],2);      
    }
  }
    if(ndataLMC3 < 3) {
    ndataLMC3 = iterMin;
    for (int i = 1; i < ndataLMC3; i++){
      if(ndataLMC1==iterMin) datareading3[i] = fillstart(datareading1[i],3);
      if(ndataLMC2==iterMin) datareading3[i] = fillstart(datareading2[i],3);
      if(ndataLMC4==iterMin) datareading3[i] = fillstart(datareading4[i],3);      
    }
  }
    if(ndataLMC4 < 3) {
    ndataLMC4 = iterMin;
    for (int i = 1; i < ndataLMC4; i++){
      if(ndataLMC1==iterMin) datareading4[i] = fillstart(datareading1[i],4);
      if(ndataLMC2==iterMin) datareading4[i] = fillstart(datareading2[i],4);
      if(ndataLMC3==iterMin) datareading4[i] = fillstart(datareading3[i],4);      
    }
  }

  file.close();
  return iteration;
}

void SaveHandDataOffline (std::string filepath, HandData HandData[], int nHandData){
  std::ofstream fileDevice;
  fileDevice.open(filepath);

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
  
  for (int iter=0; iter < nHandData; ++iter){
    fileDevice << HandData[iter].device_id << "\t" << HandData[iter].frame_id << "\t" << HandData[iter].framerate << "\t" << 
    HandData[iter].timestamp << "\t";

    for (int k = 0; k < 30; ++k) {
      fileDevice << HandData[iter].PositionData[k].x << "," << HandData[iter].PositionData[k].y << "," << HandData[iter].PositionData[k].z << "\t";      
    }
    fileDevice << "\n";    
  } 
  fileDevice.close();
}

HandData fillstart(HandData basedata, int numdev){
  HandData filldata;

  filldata = basedata;
  filldata.device_id = numdev;
  for (int k = 0; k < 30; ++k) {
    filldata.PositionData[k].x = std::numeric_limits<double>::quiet_NaN();
    filldata.PositionData[k].y = std::numeric_limits<double>::quiet_NaN();
    filldata.PositionData[k].z = std::numeric_limits<double>::quiet_NaN();
  }

  return filldata;
}