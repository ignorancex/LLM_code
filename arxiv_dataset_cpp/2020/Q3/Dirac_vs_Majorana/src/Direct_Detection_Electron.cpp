#include "Direct_Detection_Electron.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric> //for std::accumulate

#include <omp.h>

#include "Numerics_Functions.hpp"
#include "Physics_Functions.hpp"
#include "Statistics.hpp"

double vMinimal_e(double q,double Ee,double mDM)
{
	return (Ee/q+q/2.0/mDM);
}

double dR_dlnE_Ion(double E,const DM_Particle& DM, const Atomic_Electron& shell)
{
	return rhoDM/DM.mass /shell.Nucleus_Mass * DM.dSigmaV_dlnE(E,shell);
}

double dR_dlnE_Ion(unsigned int response,double E,const DM_Particle& DM, const Atomic_Electron& shell)
{
	return rhoDM/DM.mass /shell.Nucleus_Mass * DM.dSigmaV_dlnE(response,E,shell);
}

double dR_dlnE_Ion(double E,const DM_Particle& DM, const Atom& atom)
{
	double sum = 0.0;
	for(unsigned int i = 0; i < atom.electrons.size(); i++)
	{
		sum += dR_dlnE_Ion(E,DM,atom.electrons[i]);
	}
	return sum;
}
double dR_dlnE_Ion(unsigned int response,double E,const DM_Particle& DM, const Atom& atom)
{
	double sum = 0.0;
	for(unsigned int i = 0; i < atom.electrons.size(); i++)
	{
		sum += dR_dlnE_Ion(response,E,DM,atom.electrons[i]);
	}
	return sum;
}


double R_Total(const DM_Particle& DM,const Atom& atom)
{
	//Find the integration limits
	double kMin = atom.electrons[0].k_min;
	double E_min = kMin*kMin/2.0/mElectron;
	double E_max = DM.mass/2.0*pow(vEarth+vesc,2.0);
	//Integrand
	std::function<double(double)> integrand = [&DM,&atom] (double E)
	{
		return dR_dlnE_Ion(E,DM,atom) / E;
	};
	//Integrate
	double eps = Find_Epsilon(integrand,E_min,E_max,1e-3);
	double R = Integrate(integrand,E_min,E_max,eps);
	return R;
}


// Derived detector class
	Detector_Ionization::Detector_Ionization(std::string atom,double expo)
	:  Detector(expo,"Electrons") , target(Import_Electronic_Responses(atom)),  binned_data(false), Using_Electron_Bins(true) , Using_PE_Bins(false), ne_threshold (3)
	{
	}

	Detector_Ionization::Detector_Ionization(double expo)
	:  Detector(expo,"Electrons") ,  binned_data(false), Using_Electron_Bins(true) , Using_PE_Bins(false), ne_threshold (3)
	{
	}

	void Detector_Ionization::Import_Atomic_Response_Functions(std::string atom)
	{
		target = Import_Electronic_Responses(atom);
	}

	double Detector_Ionization::PDFne(unsigned int ne,double Ee,const Atomic_Electron& shell) const
	{
		double fR=0.0;
		double Nx_over_Ni = 0.2;
		double fe = (1.0-fR)/(1.0+Nx_over_Ni);
		double neMax = shell.ne_Secondary+floor(Ee/target.W);
		return PMF_Binomial(neMax,fe,ne-1);
	}

	double Detector_Ionization::Maximum_Energy_Deposit(const DM_Particle& DM) const
	{
		return DM.mass / 2.0 * pow((vesc+vEarth),2.0);
	}

	double Detector_Ionization::Minimum_DM_Mass(DM_Particle& DM) const
	{
		double mMin = 2.0 * target.Lowest_Binding_Energy() / pow((vesc+vEarth),2.0);
		if(Using_Electron_Bins) mMin *= ne_threshold;
		return mMin;
	}

	double Detector_Ionization::dRdE(double E, const DM_Particle& DM,double vDM)
	{
		return 1.0/E * flat_efficiency * dR_dlnE_Ion(E,DM,target);
	}

	Interpolation Detector_Ionization::Spectrum(const DM_Particle& DM,unsigned int points)
	{
		double Emin = 0.0;
		double Emax = keV;
		return Spectrum_Base(DM,Emin,Emax,points);
	}

	double Detector_Ionization::dRdne(unsigned int ne,const DM_Particle& DM,const Atomic_Electron& shell)
	{
		double sum=0.0;
		for(int ki=0;ki<shell.Nk;ki++)
		{
			double k = shell.k_Grid[ki];
			double Ee = k*k/2.0/mElectron;
			sum += 2.0 * shell.dlogk * log(10) * PDFne(ne,Ee,shell) * dR_dlnE_Ion(Ee,DM,shell);
		}
		return sum;
	}
	// double Detector_Ionization::dRdne_2(unsigned int ne,const DM_Particle& DM,const Atomic_Electron& shell)
	// {
	// 	//Find the integration limits
	// 	double kMin = shell.k_min;
	// 	double E_min = kMin*kMin/2.0/mElectron;
	// 	double E_max = DM.mass/2.0*pow(vEarth+vesc,2.0);
	// 	//Integrand
	// 	std::function<double(double)> integrand = [this,ne,&DM,&shell] (double E)
	// 	{
	// 		return dR_dlnE_Ion(E,DM,shell) / E * PDFne(ne,E,shell);
	// 	};
	// 	//Integrate
	// 	double eps = Find_Epsilon(integrand,E_min,E_max,1e-3);
	// 	double dRdn = Integrate(integrand,E_min,E_max,eps);
	// 	return dRdn;
	// }

	double Detector_Ionization::dRdne(unsigned int ne,const DM_Particle& DM)
	{
		double sum = 0.0;
		for(unsigned int i = 0 ; i < target.electrons.size() ; i++)
		{
			double drdn = dRdne(ne,DM,target.electrons[i]);
			if( Using_Electron_Bins && !(bin_efficiencies.empty()) ) drdn *= bin_efficiencies[ne-1];
			sum += drdn;
		}
		return flat_efficiency * sum;
	}

	void Detector_Ionization::Set_Electron_Threshold(unsigned int n)
	{
		Using_Electron_Bins = true;
		ne_threshold = n;
	}

	void Detector_Ionization::Set_Binned_Events(const std::vector<unsigned long int>& events,std::vector<double> eff)
	{

		binned_data = true;
		binned_events = events;
		if(eff.empty())
		{
			for(unsigned int i = 0 ; i < events.size() ; i++) bin_efficiencies.push_back(1.0);
		}
		else if (eff.size() == events.size()) bin_efficiencies = eff;
		else
		{
			std::cerr <<"Error in Detector_Ionization::Set_Binned_Events(): Events and efficiencies have different sizes."<<std::endl;
			std::exit(EXIT_FAILURE);
		}
		Set_Observed_Signals(std::accumulate(binned_events.begin(),binned_events.end(),0.0));
	}

	void Detector_Ionization::Set_PE_Distribution(double mu,double sigma, std::vector<int> binsizes)
	{
		Using_PE_Bins = true;
		Using_Electron_Bins = false;
		mu_PE = mu;
		sigma_PE = sigma;
		bins = binsizes;
	}

	void Detector_Ionization::Import_Trigger_Efficiency_PE(std::string filename)
	{
		Trigger_Efficiency_PE = Read_List(filename);
	}
	void Detector_Ionization::Import_Acceptance_Efficiency_PE(std::string filename)
	{
		Acceptance_Efficiency_PE = Read_List(filename);
	}

	double Detector_Ionization::dRdS2(unsigned int S2,const DM_Particle& DM)
	{
		double sum=0.0;
		for(int ne = 1; ne <= 15; ne++)
		{
				sum += PDF_Gauss(S2,mu_PE*ne,sqrt(ne)*sigma_PE) * dRdne(ne,DM);
		}
		if( !(Trigger_Efficiency_PE.empty()) ) sum *= Trigger_Efficiency_PE[S2-1];
		if( !(Acceptance_Efficiency_PE.empty()) ) sum *= Acceptance_Efficiency_PE[S2-1];
		return sum;
	}



	double Detector_Ionization::N_Signals(const DM_Particle& DM)
	{
		double N=0;
		if(Using_Electron_Bins)
		{
			for(int ne = ne_threshold ; ne < 16 ; ne++)
			{
				N += exposure * dRdne(ne,DM);
			}
		}
		else if(Using_PE_Bins)
		{
			for(int nPE = bins.front() ; nPE < bins.back() ; nPE++)
			{
				N += exposure * dRdS2(nPE,DM);
			}
		}
		return N;
	}

	double Detector_Ionization::Likelihood(const DM_Particle& DM)
	{
		double llh = 0.0;
		std::vector<double> llhs;
		if(Using_Electron_Bins)
		{
			if(binned_events.empty()) llh = CDF_Poisson( N_Signals(DM) ,observed_signals);
			else
			{
				//Compute likelihood in each bin.
				for(unsigned int ne = ne_threshold ; ne <= binned_events.size() ; ne++)
				{	
					double Nbin = exposure * dRdne(ne,DM);	
					double l=CDF_Poisson(Nbin,binned_events[ne-1]);
					llhs.push_back(l);
				
				}
				llh = *std::min_element(llhs.begin(),llhs.end());
			}	
		}
		else if(Using_PE_Bins)
		{
			//Compute likelihood in each bin.
			for(unsigned int bin = 0 ; bin < (bins.size()-1) ; bin++)
			{
				double Nbin=0;
				for(int nPE = bins[bin] ; nPE < bins[bin+1] ; nPE++)
				{
					Nbin += exposure * dRdS2(nPE,DM);
				}
				double l=CDF_Poisson(Nbin,binned_events[bin]);
				llhs.push_back(l);
			}
			llh = *std::min_element(llhs.begin(),llhs.end());

		}
		return llh;
	}

	double Detector_Ionization::Upper_Bound(DM_Particle& DM)
	{
		double effective_coupling = 1.0;
		std::vector<double> effective_couplings={};
		if(Using_Electron_Bins)
		{
			if(binned_events.empty())
			{
				double N_Limit = Inv_CDF_Poisson(observed_signals,1.0-certainty_level);
				effective_coupling = sqrt(N_Limit / N_Signals(DM));
			}
			else
			{
				//Compute likelihood in each bin.
				for(unsigned int ne = ne_threshold ; ne <= binned_events.size() ; ne++)
				{	
					double N_Limit = Inv_CDF_Poisson(binned_events[ne-1],1.0-certainty_level);
					double Nbin = exposure * dRdne(ne,DM);	
					effective_couplings.push_back( sqrt(N_Limit / Nbin) );
					// std::cout <<"\t"<<ne<<"\t"<<sqrt(N_Limit / Nbin)<<std::endl;
				}
				effective_coupling = *std::min_element(effective_couplings.begin(),effective_couplings.end());
			}	
		}
		else if(Using_PE_Bins)
		{
			//Compute likelihood in each bin.
			for(unsigned int bin = 0 ; bin < (bins.size()-1) ; bin++)
			{
				double N_Limit = Inv_CDF_Poisson(binned_events[bin],1.0-certainty_level);	
				double Nbin=0;
				for(int nPE = bins[bin] ; nPE < bins[bin+1] ; nPE++)
				{
					Nbin += bin_efficiencies[bin] * exposure * dRdS2(nPE,DM);
				}
				effective_couplings.push_back( sqrt(N_Limit / Nbin));
			}
			effective_coupling = *std::min_element(effective_couplings.begin(),effective_couplings.end());
		}
		return effective_coupling;
	}

	std::vector<std::vector<double>> Detector_Ionization::Limit_Curve(DM_Particle& DM,double mMin,double mMax, int points)
	{
		double mOriginal = DM.mass;
		std::cout <<"Compute lower bound on couplings/cross section by " <<name <<". (CL="<<100*certainty_level <<"\%) for mDM in (" <<mMin <<" GeV,"<<mMax<<" GeV) in " <<points <<" steps."<<std::endl<<std::endl;
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
			double coupling = Upper_Bound(DM);
			double sigma = Reduced_Mass(DM.mass,mElectron)*Reduced_Mass(DM.mass,mElectron)/16.0/M_PI/DM.mass/DM.mass/mElectron/mElectron*coupling*coupling;
			limit.push_back(std::vector<double> {mDM,coupling,sigma});
			std::cout <<"\r                                                                   \r" 
			<<i<<"/"<<points<<")\t"<<name<<"\t"<<Round(mDM) <<" GeV\tc="<<Round(coupling)<<"\t" <<Round(In_Units(sigma,cm*cm))<<"cm^2\t("<<floor(100.0*i/points)<<"\%)"<<std::flush;
		}

		std::cout <<"\rDone.             	                                                  "<<std::endl;
		DM.Set_Mass(mOriginal);
		return limit;
	}

	double Detector_Ionization::Minimum_Speed(const DM_Particle& DM) const
	{
		return sqrt(2.0*target.Lowest_Binding_Energy()/DM.mass);;
	}

	void Detector_Ionization::Print_Summary() const
	{
		Print_Summary_Base();
		std::cout 	<<std::endl<<"Electron scattering experiment."<<std::endl
					<<"Target:\t\t\t"	<<target.Name <<std::endl
					<<"Analysis:\t\t" <<(binned_data ? "Binned Poisson" : "Poisson") <<std::endl
					<<"PE spectrum:\t\t" <<(Using_PE_Bins? "[x]" : "[ ]") <<std::endl;
		if(Using_PE_Bins)
		{
			std::cout <<"\tmu_PE:\t" <<mu_PE<<std::endl;
			std::cout <<"\tsigma_PE:\t" <<sigma_PE<<std::endl;
		}
		if(binned_data)
		{
			std::cout <<"Bins:"<<std::endl;
						
			if(Using_Electron_Bins)
			{
				std::cout<<"n_e bin\tevents\tefficiency"<<std::endl;
				for(unsigned int bin = 0 ; bin < binned_events.size() ; bin++)
				{
					std::cout <<bin+1 <<"\t" <<binned_events[bin]<<"\t" <<bin_efficiencies[bin]<<std::endl;
				}
			}
			else if(Using_PE_Bins)
			{
				std::cout<<"PE bin\t\tevents\tefficiency"<<std::endl;

				for(unsigned int bin = 0 ; bin < binned_events.size()-1 ; bin++)
				{
					std::cout <<"["<<bins[bin]<<","<<bins[bin+1]<<")\t\t" <<binned_events[bin]<<"\t"<<bin_efficiencies[bin]<<std::endl;
				}
			}
		}
	 	std::cout<<"----------------------------------------"<<std::endl<<std::endl;
	}


