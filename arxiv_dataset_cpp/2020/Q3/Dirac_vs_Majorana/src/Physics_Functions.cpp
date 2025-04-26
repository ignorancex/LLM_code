#include "Physics_Functions.hpp"

#include <cmath>
#include <iostream>
#include <functional>
#include <cstdlib>

#include "Numerics_Functions.hpp"

//1. Units:
	//Energy
	const double GeV=1.0;
	const double eV=1.0E-9*GeV;
	const double keV=1.0E-6*GeV;
	const double MeV=1.0E-3*GeV;
	const double TeV=1.0E3*GeV;
	const double erg=gram*pow(cm/sec,2);
	const double Joule=kg*pow(meter/sec,2);
	//Mass
	const double gram=5.617977528089887E23*GeV;
	const double kg=1e3*gram;
	//Length
	const double cm=5.068e13/GeV;
	const double meter=100*cm;
	const double km=1000*meter;
	const double fm=1e-15*meter;
	const double pb=1e-36*pow(cm,2);
	const double parsec=3.0857e16*meter;
	const double kpc=1e3*parsec;
	const double Mpc=1e6*parsec;
	const double a0 = 5.29177e-11*meter; //Bohr radius
	//Time
	const double sec=299792458*meter;
	const double minute=60*sec;
	const double hour=60*minute;
	const double day=24*hour;
	const double year=365.24*day;
	//Temperature
	const double Kelvin=8.62E-14*GeV;
	//Angle
	const double deg=M_PI/180.0;

//2. Specific Parameters:
	//Masses
	const double mPlanck= 1.2209E19*GeV;
	const double mProton=0.938*GeV ;
	const double mElectron= 0.511*MeV;
	const double mNucleon = 0.932*GeV;
	//Coupling constants
	const double aEM = 1.0/137.035999139;
	const double GNewton=pow(mPlanck,-2);
	const double GFermi = 1.16637e-5/GeV/GeV;
	const double Elementary_Charge = sqrt(4.0 * M_PI * aEM);
	//Geographic Parameters
	const double mEarth=5.972E24*kg;
	const double rEarth=6371*km;
	const double rhoEarth=5.51*gram*pow(cm,-3);
	const double rhoCrust=2.7*gram*pow(cm,-3);
	//Solar Parameters
	const double mSun=1.989E30*kg;
	const double rSun=6.957E8*meter;
	//Dark Matter Halo Parameters
	double v0 = 220.0 * km/sec;
	double vesc = 544.0 * km/sec;
	double vEarth = 244 * km/sec;
	double rhoDM = 0.4 * GeV /cm/cm/cm;
	double Nesc = M_PI*v0*v0*(sqrt(M_PI)*v0*erf(vesc/v0)-2*vesc*exp(-vesc*vesc/v0/v0));

	//Reference momentum transfer
	const double qRef = aEM * mElectron;

//3. Unit Conversion
	double In_Units(double quantity, double dimension)
	{
		return quantity/dimension;
	}

//4. Simple Physics functions
	double Reduced_Mass(double m1,double m2)
	{
		return m1*m2/(m1+m2);
	}
	double DM_Speed_Distribution(double v,double vEarth)
	{
		return M_PI*v*v0*v0/Nesc/vEarth*(2*exp(-(v*v+vEarth*vEarth)/v0/v0)*sinh(2*v*vEarth/v0/v0)+(exp(-pow(v+vEarth,2.0)/v0/v0)-exp(-vesc*vesc/v0/v0))*StepFunction(fabs(v+vEarth)-vesc)-(exp(-pow(v-vEarth,2.0)/v0/v0)-exp(-vesc*vesc/v0/v0))*StepFunction(fabs(v-vEarth)-vesc) );
	}