#include "Direct_Detection.hpp"

#include <iostream>
#include <fstream>
#include <cmath>


//1. Detector base class
	Interpolation Detector::Spectrum_Base(const DM_Particle& DM,double Emin,double Emax,unsigned int points)
	{
		//1. Find maximum ER, i.e. the domain of the spectrum
			double ERmax = Maximum_Energy_Deposit(DM);
			ERmax = std::min(ERmax,Emax);
		//2. Tabulate the spectrum
			double dER = (ERmax-Emin)/(points-1.0);
			std::vector<std::vector<double>> interpol_list;
			for(unsigned int i = 0;i<points;i++)
			{
				double ER = Emin + i*dER;
				double dR = dRdE(ER,DM);
				interpol_list.push_back(std::vector<double> {ER,dR});
			}
		//3. if the maximum value is not Emax we append a last point
			if(ERmax<Emax) interpol_list.push_back(std::vector<double> {Emax,0.0});
		//4. Interpolate and return
			Interpolation spectrum(interpol_list);
			return spectrum;
	}

	void Detector::Print_Summary_Base() const
	{
		std::cout 	<<std::endl
					<<"----------------------------------------"<<std::endl
					<<"Experiment summary:\t"<<name<<std::endl
					<<"Target particles:\t" <<target_particles <<std::endl
					<<"Certainty level [\%]:\t" <<100.0*certainty_level<<std::endl
					<<"Exposure [kg day]:\t" <<In_Units(exposure,kg*day)<<std::endl
					<<"Flat efficiency [\%]:\t"<<Round(100.0*flat_efficiency)<<std::endl
					<<"Observed events:\t"<<observed_signals<<std::endl;
					// <<"Minimum DM mass [GeV]:\t" <<Round(Minimum_DM_Mass())<<std::endl;

	}

	void Detector::Set_Name(std::string n)
	{
		name=n;
	}
	void Detector::Set_Flat_Efficiency(double eff)
	{
		flat_efficiency = eff;
	}

	void Detector::Set_Observed_Signals(unsigned long int n)
	{
		observed_signals = n;
	}
	void Detector::Set_Certainty_Level(double cl)
	{
		certainty_level = cl;
	}

	std::vector<std::vector<double>> Detector::Limit_Curve(DM_Particle& DM,double mMin,double mMax, int points)
	{
		double mOriginal = DM.mass;
		std::cout <<"Compute lower bound on cross section by " <<name <<". (CL="<<100*certainty_level <<"\%) for mDM in (" <<mMin <<" GeV,"<<mMax<<" GeV) in " <<points <<" steps"<<std::endl;
		std::vector<std::vector<double>> limit;

		//Mass scan
		double dlogm = (points==1)? 0 : (log10(mMax)-log10(mMin))/(points-1);
		for(int i=0;i<points;i++)
		{
			double mDM = pow(10.0,log10(mMin)+i*dlogm);
			if(mDM < Minimum_DM_Mass(DM))
			{
				std::cerr<<"Warning in Detector::Limit_Curve(): mDM (= "<<mDM <<" GeV) < m_Min (= "<<Minimum_DM_Mass(DM) <<" GeV)"<<std::endl;
				continue;
			}
			DM.Set_Mass(mDM);
			double sigma = Upper_Bound(DM);
			limit.push_back(std::vector<double> {mDM,sigma});
			std::cout <<"\r                                                     \r" 
			<<Round(mDM) <<" GeV\t" <<Round(In_Units(sigma,cm*cm))<<"cm^2\t("<<floor(100.0*i/points)<<"\%)"<<std::flush;
		}

		std::cout <<"\rDone.                                                             "<<std::endl<<std::endl;
		DM.Set_Mass(mOriginal);
		return limit;
	}


