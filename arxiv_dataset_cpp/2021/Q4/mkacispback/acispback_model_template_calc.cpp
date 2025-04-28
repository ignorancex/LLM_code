#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <string>

int Nflux, count;
int grid = 1000;
double emin[100000], emax[100000];

int readdat(){
	std::ifstream ifs("emin.dat");
	std::string str;
	if (ifs.fail()){
		std::cerr << "error opening the dumped dat file !!" << std::endl;
		return -1;
	}
	count=0;
	while (getline(ifs, str)){
		emin[count]=std::stod(str);
		count++;
	}
	std::ifstream ifs2("emax.dat");
	if (ifs2.fail()){
		std::cerr << "error opening the dumped dat file !!" << std::endl;
		return -1;
	}
	count=0;
	while (getline(ifs2, str)){
		emax[count]=std::stod(str);
		count++;
	}
	Nflux = count;
	return 0;
}

int main(){
	std::cout << std::setprecision(10);
	readdat();
	std::ofstream ofs("flux.dat");
	for(size_t i=0;i<Nflux;i++){
		double elow=emin[i];
		double ehigh=emax[i];
		double flux_temp=0.0;
		double ewidth=1./(float (grid))*(ehigh-elow);
		for(int j=0;j<grid;j++){
			double egridL=elow + (float (j))/(float (grid))*(ehigh-elow);
			double egridH=elow + (float (j+1))/(float (grid))*(ehigh-elow);
// abrupt end here is not an error.
