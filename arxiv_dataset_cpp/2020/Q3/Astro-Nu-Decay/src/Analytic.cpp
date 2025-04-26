#include <gsl/gsl_sf_gamma.h>

#include <cmath>
#include <cassert>

#include "Analytic.h"
#include "Parameters.h"

#define sq(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

// Calculate the regeneration term without cosmology without relying on integrals, except in the special function of the incomplete gamma function
// In some cases (integer spectral index) even that can be handled directly
// only works for one coupling (scalar and/or pseudo-scalar) on at a time
double Preg_anal(flavor a, flavor b, decay_params dp)
{
	assert(dp.gamma > 1);
	assert(not dp.x_is_redshift);

	if (dp.m1 == 0)
		dp.m1 = 1e-10;

	double m2, m3, mi, mj, x, gp, g, L, pvis;
	double A, B, C, D, E;
	int i, j;

	m2 = sqrt(Dmsq21 + sq(dp.m1));
	m3 = sqrt(Dmsq31 + sq(dp.m1));

	// fix i,j
	if (dp.g21 > 0 or dp.gp21 > 0)
	{
		assert(dp.g31 == 0 and dp.gp31 == 0 and dp.g32 == 0 and dp.gp32 == 0);
		x = m2 / dp.m1;
		i = 2;
		j = 1;
		mi = m2;
		mj = dp.m1;
		g = dp.g21;
		gp = dp.gp21;
	} // 21
	else if (dp.g31 > 0 or dp.gp31 > 0)
	{
		assert(dp.g21 == 0 and dp.gp21 == 0 and dp.g32 == 0 and dp.gp32 == 0);
		x = m3 / dp.m1;
		i = 3;
		j = 1;
		mi = m3;
		mj = dp.m1;
		g = dp.g31;
		gp = dp.gp31;
	} // 31
	else if (dp.g32 > 0 or dp.gp32 > 0)
	{
		assert(dp.g31 == 0 and dp.gp31 == 0 and dp.g21 == 0 and dp.gp21 == 0);
		x = m3 / m2;
		i = 3;
		j = 2;
		mi = m3;
		mj = m2;
		g = dp.g32;
		gp = dp.gp32;
	} // 32
	else
		return 0; // all the g's are zero

	// make everything in GeV or GeV-1
	mi *= 1e-9; // eV -> GeV
	mj *= 1e-9; // eV -> GeV
	L = dp.x / kmGeV; // km -> GeV-1

	// Helper functions
	double f, k, h, y, z, T;
	f = 0.5 * x + 2 + 2 * log(x) / x - 2 / sq(x) - 0.5 / cube(x);
	h = 0.5 * x - 2 + 2 * log(x) / x + 2 / sq(x) - 0.5 / cube(x);
	k = 0.5 * x - 2 * log(x) / x - 0.5 / cube(x);
	y = sq(g) * (f + k) + sq(gp) * (h + k);
	z = sq(g) * (1. / x + x + 2) + sq(gp) * (1. / x + x - 2);
	T = mi * mj * L * y / (16 * M_PI * dp.Ef);

	// The components of the expression
	A = z / (y * dp.gamma);
	B = 1 - pow(x, -2 * dp.gamma);
	C = dp.gamma * pow(T, - dp.gamma);

	// For integers the special functions can be skipped
	if (dp.gamma == 2)
	{
		D = exp(-T) * (1 + T);
		E = exp(-T / sq(x)) * (1 + T / sq(x));
	} // gamma=2
	else
	{
		D = gsl_sf_gamma_inc(dp.gamma, T);
		E = gsl_sf_gamma_inc(dp.gamma, T / sq(x));
	}

	// The i->j probability
	pvis = A * (B + C * (D - E));

	return std::norm(U[a][i - 1]) * std::norm(U[b][j - 1]) * pvis;
}

