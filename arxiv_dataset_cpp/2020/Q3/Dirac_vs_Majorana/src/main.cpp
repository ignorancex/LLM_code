#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <numeric>

#include "Linear_Algebra.hpp"
#include "Numerics_Functions.hpp"
#include "Dirac_vs_Majorana.hpp"
#include "Statistics.hpp"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[])
{
	auto Time_Start = steady_clock::now();
	std::cout<<"Dirac vs. Majorana v1.0 by Timon Emken (2020)"<<std::endl;
	//Initialize Random Number Generator:
	std::random_device rd;
	std::mt19937 PRNG(rd());

	////////////////////////////////////////////////////////////////////////////////////////
	// 1. Setup of the direct detection experiment.
	std::string target = "Xenon";
	double exposure = 1.0 * kg * day;
	int electron_threshold = 4;
	
	// Detector_Ionization detector(target, exposure); //Requires the tables of the atomic response functions.
	Detector_Ionization detector(exposure);
	detector.Set_Electron_Threshold(electron_threshold);

	//2. Benchmark hierarchies for the Dirac hypothesis
	std::vector<string> hierarchy_labels = {"D","A12","A13","A23","T2","T3","T1"};
	std::vector<std::vector<double>> hierarchies = {
		{1,		1,		1},		//D
		{1,		1,		1e-3},	//A12
		{1,		1e-3,	1},		//A13
		{1e-3,	1,		1},		//A23
		{1e-3,	1,		1e-3},	//T2
		{1e-3,	1e-3,	1},		//T3
		{1,		1e-3,	1e-3}	//T1
	};

	//3. Benchmark point
	double mDM = 100 * MeV;
	int h = 0;
	std::vector<double> hierarchy = hierarchies[h];

	//4. Compute the fiducial spectra.

	// Fiducial spectrum for mDM = 100 MeV for the travis test run. For the computation of the fiducial spectrum, the tables of the atomic response functions are required.
	// std::vector<std::vector<double>> fiducial_spectra = Fiducial_Spectra(detector, mDM);
	std::vector<std::vector<double>> fiducial_spectra = {
		{ 2.77307, 3.85075, 2.38784, 1.55659, 0.904172, 0.499408, 0.254584, 0.142955, 0.0766797, 0.040213, 0.0178609, 0.00777752},
		{3.92624e+08, 5.27263e+08, 2.65155e+08, 1.39245e+08, 6.8822e+07, 3.27614e+07, 1.41762e+07, 7.33713e+06, 3.50829e+06, 1.58081e+06, 600647, 224653},
		{3.08045e+10, 3.43227e+10, 1.39899e+10, 6.21276e+09, 2.68671e+09, 1.1302e+09, 4.21909e+08, 2.00007e+08, 8.623e+07, 3.42746e+07, 1.13205e+07, 3.73736e+06}
	};
	
	//5. Compute the p-value/Significance
	
	int observed_events = 100;
	bool terminal_output = false;
	int Sample_Size = 1000;
	double p = Compute_p_Value(Sample_Size, PRNG, detector, observed_events, mDM, hierarchy,fiducial_spectra,terminal_output);
	double Z = Significance(p);

	std::cout 	<<std::endl <<"DM mass:\t"<<In_Units(mDM,MeV)<<" MeV"<<std::endl
				<<"Hierarchy "<<hierarchy_labels[h] <<":\t"<<hierarchy[0]<<" : "<<hierarchy[1]<<" : "<<hierarchy[2]<<"\t(Anapole : Magn. Dipole : El. Dipole)"<<std::endl
				<<"Obs. signals:\t"<<observed_events<<std::endl
				<<"t-sample size:\t" <<Sample_Size <<std::endl<<std::endl
				<<"p = "<<p<<"\t=>\tZ = "<<Z<<std::endl;

////////////////////////////////////////////////////////////////////////////////////////
	//Ending time and computing time
	auto Time_End = steady_clock::now();
	double durationTotal =1e-6*duration_cast<microseconds>( Time_End - Time_Start ).count();
	cout <<"\nProcessing Time:\t"<< durationTotal<<"s ("<< floor(durationTotal/3600.0)<<":"<<floor(fmod(durationTotal/60.0,60.0))<<":"<<floor(fmod(durationTotal,60.0))<<":"<<floor(fmod(1000*durationTotal,1000.0))<<").\a"<<endl;

	return 0;
}
