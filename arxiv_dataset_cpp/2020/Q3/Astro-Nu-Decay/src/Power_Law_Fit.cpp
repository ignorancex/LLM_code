#include <gsl/gsl_fit.h>

#include <cmath>

#include "Power_Law_Fit.h"

// Fit data to a power law
// xs and ys are arrays of length n
// norm_x is the normalization of x (e.g. 1e5 for 100 TeV)
// A and B are the result of the fit, A is the normalization at norm_x while -B is the spectral index so E^-2 => B=2
void Power_Law_Fit(double *xs, double *ys, int n, double norm_x, double &A, double &B)
{
	double *weights = new double[n];
	double *x = new double[n];
	double *y = new double[n];
	double c0, c1, tmp;

	for (int i = 0; i < n; i++)
	{
		weights[i] = 1.;	// even weight across energy range
//		weights[i] = ys[i];	// take poisson fluctuations only, 1/sqrt(n)
		x[i] = log(xs[i]);
		y[i] = log(ys[i]);
	}

	gsl_fit_wlinear(x, 1, weights, 1, y, 1, n, &c0, &c1, &tmp, &tmp, &tmp, &tmp);

	delete[] weights;
	delete[] x;
	delete[] y;

	B = -c1; // want to report E^-2 as 2
	A = exp(c0) * pow(norm_x, c1); // renormalize to 100 TeV = 1e5 GeV or whatever norm_x is
}

