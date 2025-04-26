#include "DM_Particle.hpp"

#include <iostream>
#include <cmath>

#include "Numerics_Functions.hpp"
#include "Direct_Detection_Electron.hpp"

//Class for a DM particle with electron interactions (NREFT)
	//Constructors
	DM_Particle::DM_Particle()
	: mass(GeV), spin(0.5)
	{
		couplings_h = std::vector<double>(15,0.0);
		couplings_l = std::vector<double>(15,0.0);
	}

	DM_Particle::DM_Particle(double mass, double spin)
	: mass(mass), spin(spin)
	{
		couplings_h = std::vector<double>(15,0.0);
		couplings_l = std::vector<double>(15,0.0);
	}

	void DM_Particle::Set_Mass(double m)
	{
		mass=m;
	}

	//Set the couplings
	void DM_Particle::Set_Coupling(unsigned int index,double value_contact,double value_longrange)
	{
		if(index < 1 || index > 15)
		{
			std::cerr <<"Error in DM_Particle::Set_Coupling(): Coupling index out of range." <<std::endl;
			std::exit(EXIT_FAILURE);
		}
		couplings_h[index-1] = value_contact;
		couplings_l[index-1] = value_longrange;
	}
	void DM_Particle::Reset_Couplings()
	{
		couplings_h = std::vector<double>(15,0.0);
		couplings_l = std::vector<double>(15,0.0);
	}

	//DM Response functions
	double DM_Particle::DM_Response(unsigned int response,double q, double dE,double vDM) const
	{
		if(response < 1 || response > 4)
		{
			std::cerr <<"Error in DM_Particle::DM_Response(): Response index out of range." <<std::endl;
			std::exit(EXIT_FAILURE);
		}
		double R = 0.0;
		double mu_e = Reduced_Mass(mass,mElectron);
		// double vPerp2 = vDM*vDM - q*q/4.0/mu/mu - dE/mu;
		double vPerp2 = vDM*vDM + q*q/2.0/mu_e*(1.0/2.0/mu_e - 1.0/mass) - dE/mu_e;
		// std::cout <<vPerp2<<std::endl;
		double vPerp_dot_q = dE / mElectron - q*q/2.0/mElectron/mElectron;
		// double vDM_dot_q = dE + q*q/2.0/mu;
		//Effective couplings
		std::vector<double> C(15,0.0);
		for(unsigned int i = 0; i<C.size(); i++)
		{
			C[i] = couplings_h[i] + couplings_l[i] * pow(qRef/q,2.0);
		}
		switch(response)
		{
			case 1: 
			{
				double q_cross_vPerp_2 =  q*q/mElectron/mElectron * vPerp2 - vPerp_dot_q*vPerp_dot_q;
				R = C[0]*C[0] + C[2]*C[2]/4.0 * q_cross_vPerp_2 + C[6]*C[6]/4.0* vPerp2 + C[9]*C[9]/4.0 * q*q/mElectron/mElectron;
				R += spin*(spin+1.0)/12.0 * (3.0*C[3]*C[3] + (4.0*C[4]*C[4]-2.0*C[11]*C[14]) * q_cross_vPerp_2 + C[5]*C[5] * pow(q/mElectron,4) + (4.0*C[7]*C[7]+2.0*C[11]*C[11]) * vPerp2 + (2.0*C[8]*C[8]+4.0*C[10]*C[10]+2.0*C[3]*C[5]) * q*q/mElectron/mElectron + (C[12]*C[12]+C[13]*C[13]) * q*q/mElectron/mElectron * vPerp2 + C[14]*C[14] * q*q/mElectron/mElectron * q_cross_vPerp_2 + 2.0 * C[12]*C[13] * vPerp_dot_q*vPerp_dot_q);
				break;
			}
			case 2: 
			{
				R = vPerp_dot_q * (-C[6]*C[6]/2.0 * mElectron*mElectron/q/q - spin*(spin+1.0)/6.0 * ((4.0*C[7]*C[7]+2.0*C[11]*C[11] + (C[12]*C[12]+C[13]*C[13])*q*q/mElectron/mElectron) * mElectron*mElectron/q/q + 2.0 * C[12]*C[13]));
				break;
			}
			case 3: 
			{
				R = C[2]*C[2]/4.0 * q*q/mElectron/mElectron + C[6]*C[6]/4.0;
				R += spin*(spin+1.0)/12.0 * ((4.0*C[4]*C[4] + C[12]*C[12] + C[13]*C[13] - 2.0*C[11]*C[14]) * q*q/mElectron/mElectron + 4.0*C[7]*C[7] + 2.0*C[11]*C[11] + C[14]*C[14] * pow(q/mElectron,4));
				break;
			}
			case 4: 
			{
				R = -C[2]*C[2]/4.0 + spin*(spin+1.0)/12.0 * (-4.0*C[4]*C[4] - C[14]*C[14] * q*q/mElectron/mElectron + 2.0 * (C[11]*C[14] + C[12]*C[13]));
				break;
			}
		}
		return R;
	}

	double DM_Particle::dSigmaV_dlnE(double E,const Atomic_Electron& shell) const
	{
		double prefactor = 1 / 128.0 / M_PI / mass / mass / mElectron / mElectron;
		double Delta_E = shell.Binding_Energy + E;
		double vMax = vesc+vEarth;
		// Integral over q via summation of tabulated electro-responses
			double k = sqrt(2.0 * mElectron * E);
			int ki = std::round(log10(k/shell.k_min) / shell.dlogk);
			double integral = 0.0;
			if(ki<0 || ki >= shell.Nk)
			{
				integral = 0.0;
			}
			else
			{
				for(unsigned qi = 0; qi < shell.Nq ; qi++)
				{
					double q = shell.q_Grid[qi];
					// double q = pow(10.0, shell.logq_min + qi * shell.dlogq);
					double vMin = vMinimal_e(q,Delta_E,mass);
					if(vMin > vMax) continue;
					//Velocity Integral
					std::function<double(double)> integrand = [this,&shell,Delta_E,E,q] (double v)
					{
						double sum = 0.0;
						for(int response = 1; response < 5; response++)
						{
							double DM_response=DM_Response(response,q,Delta_E,v);
							if (DM_response != 0.0) sum += DM_response * shell.Electronic_Response(response,q,E);
							
						}

						return DM_Speed_Distribution(v,vEarth)/v * sum;
					};
					double eps = Find_Epsilon(integrand,vMin,vMax,1e-3);
					double velocity_integral = Integrate(integrand,vMin,vMax,eps);

					integral += log(10.0)*shell.dlogq*q*q * velocity_integral;
				}

			}
			return prefactor * integral;
	}
	double DM_Particle::dSigmaV_dlnE(unsigned int response, double E,const Atomic_Electron& shell) const
	{
		double prefactor = 1 / 128.0 / M_PI / mass / mass / mElectron / mElectron;
		double Delta_E = shell.Binding_Energy + E;
		double vMax = vesc+vEarth;
		// Integral over q via summation of tabulated electro-responses
			double k = sqrt(2.0 * mElectron * E);
			int ki = std::round(log10(k/shell.k_min) / shell.dlogk);
			double integral = 0.0;
			if(ki<0 || ki >= shell.Nk)
			{
				integral = 0.0;
			}
			else
			{
				for(unsigned qi = 0; qi < shell.Nq ; qi++)
				{
					double q = pow(10.0, shell.logq_min + qi * shell.dlogq);
					double vMin = vMinimal_e(q,Delta_E,mass);
					if(vMin > vMax) continue;
					//Velocity Integral
					std::function<double(double)> integrand = [this,response,&shell,Delta_E,E,q] (double v)
					{
						return DM_Speed_Distribution(v,vEarth)/v * DM_Response(response,q,Delta_E,v) * shell.Electronic_Response(response,q,E);
					};
					double eps = Find_Epsilon(integrand,vMin,vMax,1e-6);
					double velocity_integral = Integrate(integrand,vMin,vMax,eps);

					integral += log(10.0)*shell.dlogq*q*q * velocity_integral;
				}
			}
			return prefactor * integral;
	}

	void DM_Particle::Print_Summary() const 
	{
		std::cout 	<<std::endl
					<<"----------------------------------------"<<std::endl
					<<"DM particle summary:"<<std::endl;

		double massunit = (mass<keV)? eV: ( (mass<MeV)? keV : ((mass<GeV)? MeV : GeV) );
		std::string massunitstr = (mass<keV)? "eV": ( (mass<MeV)? "keV" : ((mass<GeV)? "MeV" : "GeV") );
		std::cout 	<<"Mass:\t\t\t" <<In_Units(mass,massunit)<<" "<<massunitstr<<std::endl
					<<"Spin:\t\t\t" <<spin<<std::endl;
		std::cout <<"DM-Electron NREFT couplings:" <<std::endl
			<<"-----------------------------------------------------------------"<<std::endl
			<<"Operator\tContact Coupling []\t Long Range Coupling []"<<std::endl
			<<"-----------------------------------------------------------------"<<std::endl
			;
		for(unsigned int i = 0; i < 15; i++)
		{
			if(couplings_h[i] != 0.0 || couplings_l[i] != 0.0)
			{
				std::cout <<i+1 <<"\t\t" <<couplings_h[i]<<"\t\t" <<couplings_l[i] <<std::endl;
			}
		}
		std::cout <<"-----------------------------------------------------------------"<<std::endl;
	}
