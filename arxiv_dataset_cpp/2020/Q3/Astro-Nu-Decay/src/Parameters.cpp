#include <cmath>
#include <complex>

#include "Parameters.h"

// Oscillation parameters
// nu-fit v4.1, NO, w/o SK-atm
// http://www.nu-fit.org/sites/default/files/v41.tbl-parameters.pdf
double s12sq = 0.31;
double s13sq = 0.02241;
double s23sq = 0.558;
double Dmsq21 = 7.39e-5; // eV^2
double Dmsq31 = +2.523e-3; // eV^2
double delta = 222 * M_PI / 180;

double t12, t13, t23;
double Dmsq32;
double c12, s12, c12sq, s212, c212;
double c13, s13, c13sq, s213, c213;
double c23, s23, c23sq, s223, c223;
std::complex<double> eid;

std::complex<double> U[3][3];
double Ue1sq, Ue2sq, Ue3sq, Um1sq, Um2sq, Um3sq, Ut1sq, Ut2sq, Ut3sq;

// Anytime the oscillation parameters are changed, this needs to be called
// Only change the six main oscillation parameters: s12sq, s13sq, s23sq, Dmsq21, Dmsq31, and delta. Everything else is derived from them
void Recalc_Parameters()
{
	c12sq = 1 - s12sq;
	c13sq = 1 - s13sq;
	c23sq = 1 - s23sq;

	s12 = sqrt(s12sq);
	s13 = sqrt(s13sq);
	s23 = sqrt(s23sq);

	c12 = sqrt(c12sq);
	c13 = sqrt(c13sq);
	c23 = sqrt(c23sq);

	t12 = acos(c12);
	t13 = acos(c13);
	t23 = acos(c23);

	s212 = sin(2 * t12);
	s213 = sin(2 * t13);
	s223 = sin(2 * t23);

	c212 = cos(2 * t12);
	c213 = cos(2 * t13);
	c223 = cos(2 * t23);

	Dmsq32 = Dmsq31 - Dmsq21;

	eid = std::complex<double>(cos(delta), sin(delta));

	U[e][0] = c12 * c13;
	U[e][1] = s12 * c13;
	U[e][2] = s13 * std::conj(eid);
	U[m][0] = -s12 * c23 - c12 * s23 * s13 * eid;
	U[m][1] = c12 * c23 - s12 * s23 * s13 * eid;
	U[m][2] = s23 * c13;
	U[t][0] = s12 * s23 - c12 * c23 * s13 * eid;
	U[t][1] = -c12 * s23 - s12 * c23 * s13 * eid;
	U[t][2] = c23 * c13;

	Ue1sq = std::norm(U[e][0]);
	Ue2sq = std::norm(U[e][1]);
	Ue3sq = std::norm(U[e][2]);
	Um1sq = std::norm(U[m][0]);
	Um2sq = std::norm(U[m][1]);
	Um3sq = std::norm(U[m][2]);
	Ut1sq = std::norm(U[t][0]);
	Ut2sq = std::norm(U[t][1]);
	Ut3sq = std::norm(U[t][2]);
}

const double eVsqkm_to_GeV = 1e-9 / 1.97327e-7 * 1e3;
const double kmGeV = 1.97327e-19;

// Aghanim:2018eyx, 1807.06209
const double OmegaM = 0.315;
const double OmegaLambda = 1 - OmegaM;
const double H0 = 67.4; // km/s/Mpc

const double c = 299792458;
const double Mpc2m = 3.0857e16 * 1e6; // 1 pc is 3e16 m
const double LH = c / H0 / 1e3 * Mpc2m * 1e-3; // km. (1e3: km->m), (1e-3: m->km)

// Benchmark decay parameters
double g_bm = 1e-6;
decay_params dp_bm{0, g_bm, g_bm, g_bm, g_bm, g_bm, g_bm, 1e6, 2, 1, true, nu2nu_and_nu2nubar};

