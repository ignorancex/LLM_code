#include <gsl/gsl_integration.h>

#include <cmath>
#include <cassert>
#include <stdexcept>

#include "Depletion.h"
#include "Redshift.h"
#include "gsl2dint.h"

#define sq(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

// The partial width
double Gij(int i, int j, double Ei, decay_params dp)
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

	// Define the helper functions and calculate the width, first if mj = 0 ...
	double f, h, k;
	if (mj == 0)
	{
		return (sq(mi) / (16 * M_PI * Ei)) * (sq(gij) + sq(gpij)) * eVsqkm_to_GeV; // km-1
	} // mj == 0
	// ...then for mj > 0
	else
	{
		f = 0.5 * xij + 2 + 2 * log(xij) / xij - 2 / sq(xij) - 0.5 / cube(xij);
		h = 0.5 * xij - 2 + 2 * log(xij) / xij + 2 / sq(xij) - 0.5 / cube(xij);
		k = 0.5 * xij - 2 * log(xij) / xij - 0.5 / cube(xij);

		return (mi * mj / (16 * M_PI * Ei)) * (sq(gij) * (f + k) + sq(gpij) * (h + k)) * eVsqkm_to_GeV; // km-1
	} // mj > 0
}

// The depletion component of the probability in the mass basis for cosmology
double Pijdep(int i, int j, decay_params dp)
{
	assert(dp.x_is_redshift);

	if (i != j) return 0;
	if (i == 1) return 0;

	double Gi;
	if (i == 3)	Gi = Gij(3, 1, dp.Ef, dp) + Gij(3, 2, dp.Ef, dp);
	else		Gi = Gij(2, 1, dp.Ef, dp);

	// The first part corrects for the energy rescaling part of the definition of probability relevant with cosmology
	return -pow(1 + dp.x, -dp.gamma) * (1 - exp(-Gi * Lastro(dp.x)));
}

// The depletion component of the probability
double Pdep(flavor a, flavor b, decay_params dp)
{
	double G21, G31, G32, G2, G3, p, L;

	// The partial widths
	// Ei = Ef for invisible
	G21 = Gij(2, 1, dp.Ef, dp);
	G31 = Gij(3, 1, dp.Ef, dp);
	G32 = Gij(3, 2, dp.Ef, dp);

	// The full widths
	G2 = G21;
	G3 = G31 + G32;

	p = 0;

	if (dp.x_is_redshift)
	{
		p += std::norm(U[a][1]) * std::norm(U[b][1]) * Pijdep(2, 2, dp);
		p += std::norm(U[a][2]) * std::norm(U[b][2]) * Pijdep(3, 3, dp);
	}
	// Without cosmology the answer is trivial
	else
	{
		L = dp.x;
		p -= std::norm(U[a][1]) * std::norm(U[b][1]) * (1 - exp(-G2 * L));
		p -= std::norm(U[a][2]) * std::norm(U[b][2]) * (1 - exp(-G3 * L));
	}

	return p;
}

