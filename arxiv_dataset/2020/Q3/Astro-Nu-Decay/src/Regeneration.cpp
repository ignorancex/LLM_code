#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_monte_vegas.h>

#include <cassert>
#include <stdexcept>

#include "Regeneration.h"
#include "Parameters.h"
#include "Redshift.h"
#include "Depletion.h"
#include "gsl2dint.h"

#define sq(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

// GammaW: the derivative of the width with respect to the final energy
// The resultant expression is dimensionless
double GijWij(int i, int j, double Ei, decay_params dp)
{
	assert(i == 3 or i == 2);
	if (i == 3) assert(j == 2 or j == 1);
	else		assert(j == 1);

	// Get the correct masses set up
	double m2, m3;
	m2 = sqrt(Dmsq21 + sq(dp.m1));
	m3 = sqrt(Dmsq31 + sq(dp.m1));

	double mi, mj = 0;
	if (i == 3) mi = m3;
	if (i == 2) mi = m2;
	if (j == 2) mj = m2;
	if (j == 1) mj = dp.m1;

	// Get the correct couplings
	double gij = 0;
	double gpij = 0;
	if (i == 2 and j == 1)
	{
		gij = dp.g21;
		gpij = dp.gp21;
	}
	if (i == 3 and j == 1)
	{
		gij = dp.g31;
		gpij = dp.gp31;
	}
	if (i == 3 and j == 2)
	{
		gij = dp.g32;
		gpij = dp.gp32;
	}

	// Handle if mj = 0 first; different nu/nubar configurations have different answers
	if (mj == 0)
	{
		switch (dp.nnb)
		{
			case nu2nu:
				return (sq(mi * 1e-9) * dp.Ef / (16 * M_PI * cube(Ei))) * (sq(gij) + sq(gpij)); // unitless
			case nu2nubar:
				return (sq(mi * 1e-9) / (16 * M_PI * sq(Ei))) * (sq(gij) + sq(gpij)) * (1 - dp.Ef / Ei); // unitless
			case nu2nu_and_nu2nubar:
				return (sq(mi * 1e-9) / (16 * M_PI * sq(Ei))) * (sq(gij) + sq(gpij)); // unitless
			default:
				throw std::domain_error("dp.nnb out of range in GijWij for mj = 0");
		} // switch dp.nnb
	} // mj == 0

	// Now handle mj > 0

	double xij, Aij;
	xij = mi / mj;
	Aij = Ei / (xij * dp.Ef) + xij * dp.Ef / Ei;

	switch (dp.nnb)
	{
		case nu2nu:
			return (mi * mj * 1e-18 / (16 * M_PI * sq(Ei))) * (sq(gij) * (Aij + 2) + sq(gpij) * (Aij - 2)); // unitless
		case nu2nubar:
			return (mi * mj * 1e-18 / (16 * M_PI * sq(Ei))) * (sq(gij) + sq(gpij)) * (1 / xij + xij - Aij); // unitless
		case nu2nu_and_nu2nubar:
			return (mi * mj * 1e-18 / (16 * M_PI * sq(Ei))) * (sq(gij) * (1 / xij + xij + 2) + sq(gpij) * (1 / xij + xij - 2)); // unitless
		default:
			throw std::domain_error("dp.nnb out of range in GijWij for mj > 0");
	} // switch dp.nnb
}

// The integrand for the regeneration probability
// Units are GeV-1
double Iij_integrand(int i, int j, double xp, double Ei, decay_params dp)
{
	assert(i == 3 or i == 2);
	if (i == 3)	assert(j == 2 or j == 1);
	else			assert(j == 1);

	// The widths are the sum of the partial widths
	double Gi, Gf;
	if (i == 2)	Gi = Gij(2, 1, Ei, dp);
	if (i == 3)	Gi = Gij(3, 1, Ei, dp) + Gij(3, 2, Ei, dp);
	if (j == 1)	Gf = 0;
	if (j == 2)	Gf = Gij(2, 1, Ei, dp);

	// Calculate the exponential
	double exponential = 1;
	if (Gi > 0)
	{
		if (dp.x_is_redshift)
			exponential = exp(-Gi * Lastro(xp, dp.x));
		else
			exponential = exp(-Gi * (dp.x - xp));
	}
	if (j == 2 and Gf > 0) // If the final state is unstable, include that
	{
		if (dp.x_is_redshift)
			exponential *= exp(-Gf * Lastro(xp));
		else
			exponential *= exp(-Gf * xp);
	} // 3->2

	if (dp.x_is_redshift)
		return GijWij(i, j, Ei, dp) * (LH * exponential / (sq(1 + xp) * h(xp))) * Spectrum(Ei, dp.gamma, dp.x) / kmGeV; // km -> GeV-1
	else
		return GijWij(i, j, Ei, dp) * exponential * Spectrum(Ei, dp.gamma, 0) / kmGeV; // km -> GeV-1
}

// The spectrum is a power law with the correct redshift dependence
double Spectrum(double Ei, double gamma, double z)
{
	return pow(Ei * (1 + z) / 100e3, -gamma);
}

// The 3->2->1 integrand is a 4D integral with non-trivial integration limits
// nu3 has energy Ei. nu3 decays to nu2 at z1. nu2 has energy Eint. nu2 decays to nu1 at z2. nu1 has energy Ef
// 3->2 decay happens at z1, 2->1 decay happens at z2. That is, 0<z2<z1<z.
// x[0] = Ei which is in the range Eint*(1+z1) to Eint*x32sq*(1+z1). Numerically we take it over Ef to Ef*x21sq*x32sq*(1+z)^2
// x[1] = Eint which is in the range Ef*(1+z2) to Ef*x21sq*(1+z2)
// x[2] = z1 which is in the range from z2 to z. Numerically we take it from 0 to z.
// x[3] = z2 which is in the range 0 to z
// Has units GeV-2
double I321_integrand(double x[], size_t dim, void *pp)
{
	decay_params dp = *(decay_params*)pp;

	// Define the parameters
	double Ei, Eint, Ef, z1, z2, z;
	Ei = x[0];
	Eint = x[1];
	Ef = dp.Ef;
	z1 = x[2];
	z2 = x[3];
	z = dp.x;
	assert(dp.x_is_redshift);

	// Get the correct masses set up
	double m2, m3, x32, x21;
	m2 = sqrt(Dmsq21 + sq(dp.m1));
	m3 = sqrt(Dmsq31 + sq(dp.m1));
	x32 = m3 / m2;
	if (dp.m1 == 0)	x21 = 1e2;
	else			x21 = m2 / dp.m1;

	// Check the edge cases of the integral and return zero if outside the allowed region
	if (z2 > z1) return 0;
	if (Ei < Eint * (1 + z1) or Ei > Eint * sq(x32) * (1 + z1)) return 0;
	if (Eint < Ef * (1 + z2) or Eint > Ef * sq(x21) * (1 + z2)) return 0;

	// The widths and GammaW functions including the correct intermediate redshifts and energies
	double G32W32, G21W21, G2, G3;
	// First we go from nu3 with Ei at z to nu2 with Eint at z1
	dp.Ef = Eint;
	G32W32 = GijWij(3, 2, Ei, dp);
	G3 = Gij(3, 1, Ei, dp) + Gij(3, 2, Ei, dp);

	// Next we go from nu2 with Eint at z1 to nu1 with Ef at z2
	dp.Ef = Ef;
	G21W21 = GijWij(2, 1, Eint, dp);
	G2 = Gij(2, 1, Eint, dp);

	// The exponent
	double exponent, denominator;
	exponent = -Lastro(z1, z) * G3 - Lastro(z2, z1) * G2;
	denominator = sq(1 + z1) * sq(1 + z2) * h(z1) * h(z2);

	if (exponent < -300) exponent = -1e100; // reduce numerical rounding problems in the integrator at the double precision threshold

	return G32W32 * G21W21 * sq(LH) * exp(exponent) * Spectrum(Ei, dp.gamma, z1) / denominator / sq(kmGeV); // km^2 -> GeV-2
}

// The 3->2->1 part of the probability integrated using VEGAS
double P321(decay_params dp)
{
	// Get the correct masses set up
	double m2, m3, x21, x32;
	m2 = sqrt(Dmsq21 + sq(dp.m1));
	m3 = sqrt(Dmsq31 + sq(dp.m1));
	x32 = m3 / m2;
	if (dp.m1 == 0)	x21 = 1e2;
	else			x21 = m2 / dp.m1;
	x32 = m3 / m2;

	// Set up the range of the integral
	assert(dp.x_is_redshift);
	double xl[4] = {dp.Ef,								dp.Ef,				0,		0};
	double xu[4] = {dp.Ef * sq(x21 * x32 * (1 + dp.x)),	dp.Ef * sq(x21),	dp.x,	dp.x};

	// Set up the integrator
	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_monte_function G = {&I321_integrand, 4, &dp};
	size_t calls = 1e7;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	double integral, error;
	gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(4);
	calls = 1e6;

	// Initialize the integration region
	gsl_monte_vegas_integrate(&G, xl, xu, 4, 0.1 * calls, r, s, &integral, &error);
	// Repeat the integral until reasonable convergence conditions are met
	do
	{
		gsl_monte_vegas_integrate(&G, xl, xu, 4, calls, r, s, &integral, &error);
	} while (fabs(gsl_monte_vegas_chisq(s) - 1) > 0.5 and error / integral > 1e-2 and integral / Spectrum(dp.Ef, dp.gamma, 0));

	// Clean up
	gsl_monte_vegas_free(s);
	gsl_rng_free(r);

	// Print warning if there is a problem
	if (error / integral > 1e-2 and integral / Spectrum(dp.Ef, dp.gamma, 0) > 1e-2) printf("Precision warning: integral = %g, error = %g, err/int = %g, probability = %g\n", integral, error, error / integral, integral / Spectrum(dp.Ef, dp.gamma, 0));

	return integral / Spectrum(dp.Ef, dp.gamma, 0);
}

// The regeneration probability in the mass basis where the integrals are performed and the 321 term is added if necessary
double Pregij(int i, int j, decay_params dp)
{
	assert(i == 3 or i == 2);
	if (i == 3) assert(j == 2 or j == 1);
	else		assert(j == 1);

	// Get the correct masses set up
	double m2, m3;
	m2 = sqrt(Dmsq21 + sq(dp.m1));
	m3 = sqrt(Dmsq31 + sq(dp.m1));

	double mi, mj = 0;
	if (i == 3) mi = m3;
	if (i == 2) mi = m2;
	if (j == 2) mj = m2;
	if (j == 1) mj = dp.m1;

	double xij;
	xij = mi / mj;
	if (mj == 0) xij = 1e2;

	// Set up the double integral
	int gsl_limit, key, status;
	double prec, integral, error, inner_integral, inner_error, Ei_min, Ei_max;
	gsl_limit = 1e5;
	key = 6;
	prec = 1e-2;
    IntegrationWorkspace w1(gsl_limit);
    IntegrationWorkspace w2(gsl_limit);

	// Turn off the error handler but print the messages later if there is a convergence problem.
	// Be sure to check the output for "error"
	gsl_set_error_handler_off();
	auto outer = make_gsl_function( [&](double zp)
	{
		auto inner = make_gsl_function( [&](double Ei) { return Iij_integrand(i, j, zp, Ei, dp); } );
		Ei_min = dp.Ef;
		Ei_max = dp.Ef * sq(xij);
		if (dp.x_is_redshift)
		{
			Ei_min *= (1 + zp);
			Ei_max *= (1 + zp);
		}
		status = gsl_integration_qag(inner, Ei_min, Ei_max, Spectrum(dp.Ef, dp.gamma, 0) * 1e-100, prec, gsl_limit, key, w1, &inner_integral, &inner_error);
		if (status != 0) printf("Ef integral has error. Error = %s. g21 = %g, m1 = %g, integral = %g, error on integral = %g, spectrum = %g\n", gsl_strerror(status), dp.g21, dp.m1, integral, error, Spectrum(dp.Ef, dp.gamma, 0));
		return inner_integral;
	} ); // outer integrand
	status = gsl_integration_qag(outer, 0, dp.x, 0, prec, gsl_limit, key, w2, &integral, &error);
	if (status != 0) printf("z integral has error. Error = %s. g21 = %g, m1 = %g, integral = %g, error on integral = %g, spectrum = %g\n", gsl_strerror(status), dp.g21, dp.m1, integral, error, Spectrum(dp.Ef, dp.gamma, 0));

	double preg;
	preg = integral / Spectrum(dp.Ef, dp.gamma, 0);

	// Add in the 3->2->1 second order decay, if necessary
	if (i == 3 and j == 1 and (dp.g32 != 0 or dp.gp32 != 0) and (dp.g21 != 0 or dp.gp21 != 0)) preg += P321(dp);

	return preg;
}

// The regeneration probability in the flavor basis
double Preg(flavor a, flavor b, decay_params dp)
{
	double p;

	p = 0;

	p += std::norm(U[a][1]) * std::norm(U[b][0]) * Pregij(2, 1, dp);
	p += std::norm(U[a][2]) * std::norm(U[b][0]) * Pregij(3, 1, dp);
	p += std::norm(U[a][2]) * std::norm(U[b][1]) * Pregij(3, 2, dp);

	return p;
}

