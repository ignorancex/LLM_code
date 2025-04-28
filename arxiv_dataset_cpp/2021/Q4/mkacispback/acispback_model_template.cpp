#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <xsTypes.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <string>

int count, flux_num, warnflag=0;
double flux_temp[50000000], rmfestart, flux_temp2, rmfegrid=0.0;

extern "C" {
    void acispback(
        const Real* energy,
        int Nflux, 
        const Real* parameter, 
        int spectrum, 
        Real* flux, 
        Real* fluxError, 
        const char* init
    ){
		std::cout << std::setprecision(10);
		std::ifstream ifs("TEMP1");
		std::string str;
		if (ifs.fail()){
			std::cerr << "error opening the flux dat file !!" << std::endl;
			exit(1);
		}
		count=0;
		while (getline(ifs, str)){
			flux_temp[count]=std::stod(str);
			count++;
		}
		flux_num=count;
		std::ifstream ifs2("TEMP2");
		if (ifs2.fail()){
			std::cerr << "error opening the dumped dat file !!" << std::endl;
			exit(1);
		}
		count=0;
		while (getline(ifs2, str)){
			if(count==1){
				rmfegrid=std::stod(str)-rmfestart;
				break;
			}
			rmfestart=std::stod(str);
			count++;
		}
		count=0;
		while(rmfestart+count*0.001*rmfegrid < energy[0]) count++;
		for(size_t i=0;i<Nflux;i++){
			flux_temp2=0;
			while(rmfestart+count*0.001*rmfegrid < energy[i+1]){
				flux_temp2+=flux_temp[count];
				count++;
				if(count>flux_num){
					warnflag=1;
					break;
				}
			}
			flux[i]=flux_temp2;
		}
	}
}
