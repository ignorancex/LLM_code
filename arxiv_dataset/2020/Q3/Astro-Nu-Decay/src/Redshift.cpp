#include <omp.h>

#include <gsl/gsl_integration.h>

#include <cmath>
#include <vector>

#include "Redshift.h"
#include "Interpolate.h"
#include "Parameters.h"
#include "SM.h"
#include "Depletion.h"
#include "Regeneration.h"

// Two redshift evolutions from the literature
double RSE(double z, void *p)
{
	bool ykbh = *(bool*)p;

	double p1, p2, p3, k, zc;

	if (ykbh)
	{
		// SFR from Beacom+ Yuksel:2008cu 0804.4008
		p1 = 3.4;
		p2 = -0.3;
		p3 = -3.5;
		k = -10.;
		return pow(pow(1 + z, p1 * k) + pow((1 + z) / 5000., p2 * k) + pow((1 + z) / 9., p3 * k), 1 / k);
	}
	else
	{
		// from 1302.5209
		zc = 1.5;
		z = std::min(z, zc);
		return pow(1 + z, +3);
	}
}

// The integral of the redshift evolution
double RSE_integral(double zmax, bool ykbh)
{
	double integral, error;
	gsl_function F;
	int gsl_limit = 1e3;

	F.function = &RSE;
	F.params = &ykbh;
	gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(gsl_limit);
	gsl_integration_qag(&F, 0, zmax, 0, 1e-4, gsl_limit, 6, workspace, &integral, &error);
	gsl_integration_workspace_free(workspace);

	return integral;
}

// Precalculate two redshift evolution integrals for the denominator
const double zmax = 5;
double RSE_integral_ykbh = RSE_integral(zmax, true);
double RSE_integral_hermes = RSE_integral(zmax, false);

// The integrand (and relevant struct) for the RSE integral
struct RSE_integral_helper_params {flavor a, b; decay_params dp; bool ykbh;};
double RSE_integral_helper(double z, void *pp)
{
	RSE_integral_helper_params p = *(RSE_integral_helper_params*)pp;
	p.dp.x = z;
	return RSE(z, &p.ykbh) * (PSM(p.a, p.b, p.dp) + Pdep(p.a, p.b, p.dp) + Preg(p.a, p.b, p.dp));
}

// The integral of the probability over RSE, divided by the integral of just the RSE
double RSE_integral(flavor a, flavor b, decay_params dp, bool ykbh)
{
	double integral, error;
	gsl_function F;
	int gsl_limit = 1e3;

	RSE_integral_helper_params p = {a, b, dp, ykbh};

	F.function = &RSE_integral_helper;
	F.params = &p;
	gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(gsl_limit);
	gsl_integration_qag(&F, 0, zmax, 0, 1e-3, gsl_limit, 1, workspace, &integral, &error);
	gsl_integration_workspace_free(workspace);

	if (ykbh)	return integral / RSE_integral_ykbh;
	else		return integral / RSE_integral_hermes;
}

// Cosmology distances
// To save time, some cosmology distance integrals are precalculated
bool I2s_calced = false; // have the I2s been precalculated or not
std::vector<std::pair<double, double> > I2s;

// The redshift evolution part of the Hubble parameter with no curvature
double h(double z)
{
	return sqrt(OmegaM * pow(1 + z, 3) + OmegaLambda); // take Omegak = 0
}

// The integrand relevant for neutrino decay
double wsqE_inv(double z, void *p)
{
	return 1. / (pow(1 + z, 2) * h(z));
}

// Calculate the integral from zi to zf
double I2_calc(double zi, double zf)
{
	double integral, error;
	gsl_function F;
	int gsl_limit = 1e3;

	F.function = &wsqE_inv;
	gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(gsl_limit);
	gsl_integration_qag(&F, zi, zf, 0, 1e-4, gsl_limit, 6, workspace, &integral, &error);
	gsl_integration_workspace_free(workspace);

	return integral; // dimensionless
}

// Precalc the integrals from 0 to various z's
void I2_precalc()
{
	if (not I2s_calced)
	{
		const int n = 1e3;
		double z, z_min, z_max, z_step;
		z_min = 0;
		z_max = 10;
		z_step = (z_max - z_min) / n;
		for (int i = 0; i <= n; i++)
		{
			z = z_min + i * z_step;
			I2s.push_back(std::make_pair(z, I2_calc(0, z)));
		} // i, n, z
		I2s_calced = true;
	} // need to pre calc
}

// Interpolate the previously precalced integrals
double I2(double z)
{
	if (z == 0) return 0;
	I2_precalc();

	return Interpolate(z, I2s);
}

// Same as I2 but with units km for [zi, zf]
double Lastro(double zi, double zf)
{
	if (zi == 0) return Lastro(zf);
	return I2_calc(zi, zf) * LH;
}

// same as I2 but with units km for [0,z]
double Lastro(double z)
{
	return I2(z) * LH;
}

