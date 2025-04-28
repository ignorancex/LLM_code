//
// Make initial conditions of isolated binaries
//   Primary mass function is m1^(-alpha) [alpha=2.3]
//   Eccentricity function is proportinal to eccentricity
//   Semi-major axis function is logarithmically uniform
// Units are solar mass for mass, solar radius for radius, and day for period

//#define GenerateSimple
#define PopIandII

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <algorithm>

typedef float     F32;
typedef double    F64;
typedef int       S32;
typedef long long S64;
#include "empfuncs.hpp"

namespace Unit {
    const double GravityConstant = 6.674e-8;
    const double SolarRadius     = 6.955e10;
    const double SolarMass       = 1.989e33;
    const double Day             = 24. * 3600.;
}

double getRandomNumber() {
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.);
}

namespace RunParameter {

    class InitialMassFunction {
    private:
	int NumberOfAlpha;
	double * Mass;
	double * Alpha;
	double * Coeff;
	double * Prob;
	double MassRatioMinimum;
	double MassRatioAlpha; // q^-(MassRatioAlpha) 0:const
	double SecondaryMassMinimum;

	void setupInitialMassFunction() {
	    double tempcoeff1inv = 0.;
	    double tempcoeff     = 1.;
	    for(int i = 0; i < NumberOfAlpha; tempcoeff *= pow(Mass[i+1], Alpha[i+1]-Alpha[i]), i++) {
		assert(Alpha[i] != 1. && Alpha[i] != 2.);
		tempcoeff1inv += tempcoeff / (- Alpha[i] + 1.)
		    * (pow(Mass[i+1], -Alpha[i]+1.) - pow(Mass[i], -Alpha[i]+1.));
	    }
	    for(int i = 0; i < NumberOfAlpha; i++) {
		if(i == 0) {
		    Coeff[0] = 1. / tempcoeff1inv;
		} else {
		    Coeff[i] = Coeff[i-1] * pow(Mass[i], Alpha[i]-Alpha[i-1]);
		}
		Prob[i] = Coeff[i] / (- Alpha[i] + 1.)
		    * (pow(Mass[i+1], -Alpha[i]+1.) - pow(Mass[i], -Alpha[i]+1.));;
	    }
	}

    public:
	InitialMassFunction(const int _NumberOfAlpha,
		     const double * _Mass,
		     const double * _Alpha,
		     const double _MassRatioMinimum,
		     const double _MassRatioAlpha,
		     const double _SecondaryMassMinimum) {
	    NumberOfAlpha = _NumberOfAlpha;
	    Mass          = new double[NumberOfAlpha+1];
	    Alpha         = new double[NumberOfAlpha+1];
	    Coeff         = new double[NumberOfAlpha];
	    Prob          = new double[NumberOfAlpha];
	    for(int i=0; i<_NumberOfAlpha+1; i++) Mass[i]  = _Mass[i];
	    for(int i=0; i<_NumberOfAlpha;   i++) Alpha[i] = _Alpha[i];
	    MassRatioMinimum     = _MassRatioMinimum;
	    MassRatioAlpha       = _MassRatioAlpha;
	    SecondaryMassMinimum = _SecondaryMassMinimum; //Mass[0];
	    this->setupInitialMassFunction();
	}

	~InitialMassFunction() {
	    delete Mass;
	    delete Alpha;
	    delete Coeff;
	    delete Prob;
	}

	double getMinimumPrimaryMass() {
	    return Mass[0];
	}

	double getMinimumSecondaryMass() {
	    return SecondaryMassMinimum;
	}

	double getPrimaryMass() {
	    bool   flag = true;
	    int    ipow = 0;
	    double frac = getRandomNumber();
	    for(int i = 0; i < NumberOfAlpha; i++) {
		if(frac - Prob[i] <= 0.) {
		    flag = false;
		    ipow = i;
		    break;
		} else {
		    frac -= Prob[i];
		}
	    }
	    if(flag) {
		return Mass[NumberOfAlpha];
	    } else {
		double term1 = pow(Mass[ipow], -Alpha[ipow]+1.);
		double term2 = (-Alpha[ipow] + 1.) / Coeff[ipow] * frac;
		return pow(term1+term2, 1./(-Alpha[ipow] + 1.));
	    }
	}

	double getMassRatio(double m1) {
	    double frac = getRandomNumber();
	    double qmin = std::max(SecondaryMassMinimum / m1, MassRatioMinimum);
	    double qalf = MassRatioAlpha;
	    double q    = (1. - qmin) * pow(frac,1./(-qalf+1.)) + qmin;
	    return q;
	}

	double print(FILE * fp) {
	    fprintf(fp, "PrimaryMassPower:");
	    for(int i = 0; i < NumberOfAlpha; i++) {
		fprintf(fp, " %.1f", Alpha[i]);
	    }
	    fprintf(fp, "\n");
	    fprintf(fp, "CriticalMass:");
	    for(int i = 0; i < NumberOfAlpha+1; i++) {
		fprintf(fp, " %.2e", Mass[i]);
	    }
	    fprintf(fp, "\n");
	    fprintf(fp, "MassRatioMinimum      %e\n", MassRatioMinimum);
	    fprintf(fp, "MassRatioAlpha        %e\n", MassRatioAlpha);
	    fprintf(fp, "SecondaryMassMinimum: %e\n", SecondaryMassMinimum);
	}

    };

    namespace Primary {
	int      SimulatedNumberOfAlpha;
	double * SimulatedMass;
	double * SimulatedAlpha;
	int      IntrinsicNumberOfAlpha;
	double * IntrinsicMass;
	double * IntrinsicAlpha;
	double MassRatioMinimum;
	double MassRatioAlpha;
	double SimulatedSecondaryMassMinimum;
	double IntrinsicSecondaryMassMinimum;
    };
    bool   CutShortPeriodBinary;
    double CutBinaryDistance;  // enable only if CutShortPeriodBinary == true
    double BinaryFraction;
    double PericenterMinimum;
    bool   SemiMajorAxisLogFlat;
    double SemiMajorAxisMinimum;
    double SemiMajorAxisMaximum;
    double LogPeriodMinimum;  //0.15; // log10(p/day)
    double LogPeriodMaximum;  //5.5;  // log10(p/day)
    double LogPeriodAlpha;    //0.55; // (logP)^-(LogPeriodAlpha) 0:log uniform
    double EccentricityAlpha; //0.42; // e^-(EccentricityAlpha) -1:Thermal distribution
};

double calcRatioOfSeparationToRocheLobe(double q) {
    double p1 = pow(q, 1./3.);
    double p2 = p1 * p1;
    return (0.6 * p2 + log(1. + p1)) / (0.49 * p2);
}

double calcMinimumSeparation(double m1, double m2) {
    using namespace RunParameter;
    double rad1  = pow(10., getRadiusZAMSTime(&m1));
    double rad2  = pow(10., getRadiusZAMSTime(&m2));
    double rmin1 = rad1 * calcRatioOfSeparationToRocheLobe(m1/m2);
    double rmin2 = rad2 * calcRatioOfSeparationToRocheLobe(m2/m1);
    return std::max(rmin1, rmin2);
}

double getSemiMajorAxis(double m1, double m2) {
    using namespace RunParameter;
    double amin    = std::max(calcMinimumSeparation(m1, m2), SemiMajorAxisMinimum);
    if(SemiMajorAxisLogFlat) {
	double logamin = log(amin);
	double logamax = log(SemiMajorAxisMaximum);
	double frac    = getRandomNumber();
	double loga    = frac * (logamax - logamin) + logamin;
	return exp(loga);
    } else {
	double amax = SemiMajorAxisMaximum;
	double frac = getRandomNumber();
	double a    = frac * (amax - amin) + amin;
	return a;
    }
}

double calcPeriod(double m1, double m2, double ab) {
    using namespace Unit;
    double acgs   = ab * SolarRadius;
    double mcgs   = (m1 + m2) * SolarMass;
    double period = 2. * M_PI * sqrt(acgs * acgs * acgs / (GravityConstant * mcgs)) / Day;
    return period;
}

double getEccentricity(double m1, double m2, double ab) {
    using namespace RunParameter;
    double rpmin   = std::max(calcMinimumSeparation(m1, m2), PericenterMinimum);
    double eccmax = 1. - rpmin / ab;
    double frac   = getRandomNumber();
    double ealf   = EccentricityAlpha;
    double ecc    = eccmax * pow(frac,1./(-ealf+1.));
    return ecc;
}

double getPeriod() {
    using namespace RunParameter;
    double logpmin = LogPeriodMinimum;
    double logpmax = LogPeriodMaximum;
    double logpalf = LogPeriodAlpha;
    double frac = getRandomNumber();
    double logp = (logpmax - logpmin) * pow(frac,1./(-logpalf+1.)) + logpmin;
    return pow(10.,logp);
}

double calcSemiMajorAxis(double m1, double m2, double pd) {
    using namespace Unit;
    double pcgs = pd * Day;
    double mcgs = (m1 + m2) * SolarMass;
    double p2   = (pcgs * pcgs / (4. * M_PI * M_PI));
    double acgs = pow((GravityConstant * mcgs * p2), 1/3.);
    return acgs / SolarRadius;
}

int main(int argc,
         char ** argv) {

    if(argc != 2) {
	fprintf(stderr, "%s <inputfile>\n", argv[0]);
	exit(0);
    }

    int    nbin;
    double z, tmax;

    {
	using namespace RunParameter;
	using namespace Primary;
	FILE * fp = fopen(argv[1], "r");
	fscanf(fp, "%d%lf%lf", &nbin, &z, &tmax);
	fscanf(fp, "%d", &SimulatedNumberOfAlpha);
	SimulatedMass  = new double[SimulatedNumberOfAlpha+1];
	SimulatedAlpha = new double[SimulatedNumberOfAlpha];
	for(int i = 0; i < SimulatedNumberOfAlpha+1; i++) fscanf(fp, "%lf", &SimulatedMass[i]);
	for(int i = 0; i < SimulatedNumberOfAlpha;   i++) fscanf(fp, "%lf", &SimulatedAlpha[i]);
	fscanf(fp, "%d", &IntrinsicNumberOfAlpha);
	IntrinsicMass  = new double[IntrinsicNumberOfAlpha+1];
	IntrinsicAlpha = new double[IntrinsicNumberOfAlpha];
	for(int i = 0; i < IntrinsicNumberOfAlpha+1; i++) fscanf(fp, "%lf", &IntrinsicMass[i]);
	for(int i = 0; i < IntrinsicNumberOfAlpha;   i++) fscanf(fp, "%lf", &IntrinsicAlpha[i]);
	fscanf(fp, "%lf", &BinaryFraction);
	fscanf(fp, "%lf%lf", &MassRatioMinimum, &MassRatioAlpha);
	SimulatedSecondaryMassMinimum = SimulatedMass[0];
	IntrinsicSecondaryMassMinimum = IntrinsicMass[0];
	fscanf(fp, "%lf%lf%lf", &LogPeriodMinimum, &LogPeriodMaximum, &LogPeriodAlpha);
	fscanf(fp, "%lf", &EccentricityAlpha);
	fclose(fp);
    }

    if(z < 0.0001 || z == 2e-4) {
	double zeta = (int)(log10(z/0.02));
	setMetallicity(&zeta);
    }
    RunParameter::CutShortPeriodBinary = false;
    RunParameter::CutBinaryDistance    = 200.;
    RunParameter::PericenterMinimum    = 0.;
    RunParameter::SemiMajorAxisLogFlat = true;
    RunParameter::SemiMajorAxisMinimum = RunParameter::PericenterMinimum;
    RunParameter::SemiMajorAxisMaximum = 1e5;

#ifdef PopIandII
    fprintf(stderr, "Caution: PopIandII mode!\n");
#endif

    double SimulatedMass = 0.;
    RunParameter::InitialMassFunction SimulatedMassFunction(RunParameter::Primary::SimulatedNumberOfAlpha,
							    RunParameter::Primary::SimulatedMass,
							    RunParameter::Primary::SimulatedAlpha,
							    RunParameter::Primary::MassRatioMinimum,
							    RunParameter::Primary::MassRatioAlpha,
							    RunParameter::Primary::SimulatedSecondaryMassMinimum);
    FILE * fp;
    fp = fopen("binaries.in", "w");
    fprintf(fp, "%d\n", nbin);
    for(int i = 0; i < nbin; i++) {
        double m1 = SimulatedMassFunction.getPrimaryMass();
        double qq = SimulatedMassFunction.getMassRatio(m1);
        double m2 = m1 * qq;
	SimulatedMass += (m1+m2);
#ifdef PopIandII
	double pd, ab, ec;
	do {
	    pd = getPeriod();
	    ab = calcSemiMajorAxis(m1, m2, pd);
	    ec = getEccentricity(m1, m2, ab);
	} while(RunParameter::CutShortPeriodBinary && ab*(1.-ec) < RunParameter::CutBinaryDistance);
#else
        double ab = getSemiMajorAxis(m1, m2);
        double pd = calcPeriod(m1, m2, ab);
        double ec = getEccentricity(m1, m2, ab);
#endif
        fprintf(fp, "%+e %+e %+e %+e %+e %+e\n", m1, m2, pd, ec, tmax, 0.);
    }
    fclose(fp);

    double IntrinsicMass = 0.;
    double AdoptedMass   = 0.;
    RunParameter::InitialMassFunction IntrinsicMassFunction(RunParameter::Primary::IntrinsicNumberOfAlpha,
							    RunParameter::Primary::IntrinsicMass,
							    RunParameter::Primary::IntrinsicAlpha,
							    RunParameter::Primary::MassRatioMinimum,
							    RunParameter::Primary::MassRatioAlpha,
							    RunParameter::Primary::IntrinsicSecondaryMassMinimum);
    for(int i = 0; i < 10*nbin; i++) {
	double m1 = IntrinsicMassFunction.getPrimaryMass();
	double qq = IntrinsicMassFunction.getMassRatio(m1);
	double m2 = m1 * qq;
	IntrinsicMass += (m1+m2);
	if(m1 >= SimulatedMassFunction.getMinimumPrimaryMass()
	   && m2 >= SimulatedMassFunction.getMinimumSecondaryMass()) {
	    double SurvivingFactor = 1.;
	    /*
#ifdef PopIandII
	    {
		using namespace RunParameter;
		double pd = getPeriod();
		double ab = calcSemiMajorAxis(m1, m2, pd);
		double rpmin   = std::max(calcMinimumSeparation(m1, m2), PericenterMinimum);
		double eccmax = 1. - rpmin / ab;
		double ealf   = EccentricityAlpha;
		SurvivingFactor = pow(eccmax,-ealf+1.);
	    }
#else
#error PopIII is not yet supported for calculating intrinsic mass
#endif
	    */
	    AdoptedMass += (m1+m2)*SurvivingFactor;
	}
    }
    int nsin = (int)((1. - RunParameter::BinaryFraction) * nbin);
    for(int i = 0; i < 10*nbin; i++) {
	double m1 = IntrinsicMassFunction.getPrimaryMass();
	IntrinsicMass += m1;
    }    

    {
        using namespace RunParameter;
        fp = fopen("binaries.txt", "w");
	SimulatedMassFunction.print(fp);
        fprintf(fp, "PericenterMinimum:    %e\n", PericenterMinimum);
#ifdef PopIandII
	fprintf(fp, "LogPeriodMinimum:     %e\n", LogPeriodMinimum);
	fprintf(fp, "LogPeriodMaximum:     %e\n", LogPeriodMaximum);
	fprintf(fp, "LogPeriodAlpha:       %e\n", LogPeriodAlpha);
#else
	fprintf(fp, "SemiMajorAxisLogFlat  %d\n", SemiMajorAxisLogFlat);
        fprintf(fp, "SemiMajorAxisMinimum: %e\n", SemiMajorAxisMinimum);
        fprintf(fp, "SemiMajorAxisMaximum: %e\n", SemiMajorAxisMaximum);
#endif
        fprintf(fp, "EccentricityAlpha:    %e\n", EccentricityAlpha);
        fprintf(fp, "SimulatedMass:        %e\n", SimulatedMass);
        fprintf(fp, "IntrinsicMass:        %e\n", SimulatedMass*IntrinsicMass/AdoptedMass);
        fclose(fp);
    }

    return 0;
}
