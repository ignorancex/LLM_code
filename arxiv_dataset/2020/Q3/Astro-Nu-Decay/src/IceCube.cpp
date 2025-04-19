#include <gsl/gsl_min.h>
#include <gsl/gsl_errno.h>

#include "IceCube.h"
#include "Parameters.h"
#include "SM.h"
#include "Depletion.h"
#include "Regeneration.h"
#include "Power_Law_Fit.h"

#define sq(x) ((x)*(x))

// Track to cascade ratio
double Rtc(decay_params dp, bool visible)
{
	// Calculate the regeneration part if necessary
	double P21, P31, P32;
	if (visible)
	{
		P21 = Pregij(2, 1, dp);
		P31 = Pregij(3, 1, dp);
		P32 = Pregij(3, 2, dp);
	}
	else
	{
		P21 = 0;
		P31 = 0;
		P32 = 0;
	}

	// Calculate the six relevant probabilities
	double Pee, Pem, Pet, Pme, Pmm, Pmt;
	Pee = PSM(e, e, dp) + Pdep(e, e, dp)
			+ Ue2sq * Ue1sq * P21
			+ Ue3sq * Ue1sq * P31
			+ Ue3sq * Ue2sq * P32;
	Pem = PSM(e, m, dp) + Pdep(e, m, dp)
			+ Ue2sq * Um1sq * P21
			+ Ue3sq * Um1sq * P31
			+ Ue3sq * Um2sq * P32;
	Pet = PSM(e, t, dp) + Pdep(e, t, dp)
			+ Ue2sq * Ut1sq * P21
			+ Ue3sq * Ut1sq * P31
			+ Ue3sq * Ut2sq * P32;
	Pme = PSM(m, e, dp) + Pdep(m, e, dp)
			+ Um2sq * Ue1sq * P21
			+ Um3sq * Ue1sq * P31
			+ Um3sq * Ue2sq * P32;
	Pmm	= PSM(m, m, dp) + Pdep(m, m, dp)
			+ Um2sq * Um1sq * P21
			+ Um3sq * Um1sq * P31
			+ Um3sq * Um2sq * P32;
	Pmt = PSM(m, t, dp) + Pdep(m, t, dp)
			+ Um2sq * Ut1sq * P21
			+ Um3sq * Ut1sq * P31
			+ Um3sq * Ut2sq * P32;

	double Nt, Nc;
	// Pion decay
	Nt = Pem + 2 * Pmm;
	Nc = Pee + 2 * Pme + Pet + 2 * Pmt;

	return Nt / Nc;
}

// Calculate the spectral index at the Earth over the range 100 TeV to 1 PeV assuming pion decay sources
void IC_gamma(decay_params dp, bool visible, double &gamma_track, double &gamma_cascade)
{
	double Efmin, Efmax, Efscale;
	const int n = 20; // n=10 => <0.5% error, n=20 => <0.3% error
	double Efs[n + 1], Phi_tracks[n + 1], Phi_cascades[n + 1];

	// The range that IceCube is sensitive to
	Efmin = 1e5;
	Efmax = 1e6;
	Efscale = pow(Efmax / Efmin, 1. / n);

	double p_track, p_cascade, p21, p31, p32, pee, pem, pet, pme, pmm, pmt;
	for (int i = 0; i <= n; i++)
	{
		dp.Ef = Efmin * pow(Efscale, i);
		Efs[i] = dp.Ef;

		// Track probability
		p_track = 0;
		p_track += PSM(e, m, dp) + Pdep(e, m, dp); // e->m
		p_track += 2 * (PSM(m, m, dp) + Pdep(m, m, dp)); // m->m

		// Cascade probability
		p_cascade = 0;
		p_cascade += PSM(e, e, dp) + Pdep(e, e, dp); // e->e
		p_cascade += PSM(e, t, dp) + Pdep(e, t, dp); // e->t
		p_cascade += 2 * (PSM(m, e, dp) + Pdep(m, e, dp)); // m->e
		p_cascade += 2 * (PSM(m, t, dp) + Pdep(m, t, dp)); // m->t

		// Include the regeneration terms if necessary
		if (visible)
		{
			// Calculate the three mass basis probabilities
			p21 = Pregij(2, 1, dp);
			p31 = Pregij(3, 1, dp);
			p32 = Pregij(3, 2, dp);

			// Calculate the six relevant flavor basis probabilities
			pee = Ue2sq * Ue1sq * p21 + Ue3sq * Ue1sq * p31 + Ue3sq * Ue2sq * p32;
			pem = Ue2sq * Um1sq * p21 + Ue3sq * Um1sq * p31 + Ue3sq * Um2sq * p32;
			pet = Ue2sq * Ut1sq * p21 + Ue3sq * Ut1sq * p31 + Ue3sq * Ut2sq * p32;
			pme = Um2sq * Ue1sq * p21 + Um3sq * Ue1sq * p31 + Um3sq * Ue2sq * p32;
			pmm = Um2sq * Um1sq * p21 + Um3sq * Um1sq * p31 + Um3sq * Um2sq * p32;
			pmt = Um2sq * Ut1sq * p21 + Um3sq * Ut1sq * p31 + Um3sq * Ut2sq * p32;

			p_track += pem + 2 * pmm; // track: e->m and m->m
			p_cascade += pee + pet + 2 * (pme + pmt); // cascade: e->e, e->t, m->e, m->t
		}

		// Multiply back by the initial spectra to get the observed spectrum
		Phi_tracks[i] = p_track * Spectrum(dp.Ef, dp.gamma, 0);
		Phi_cascades[i] = p_cascade * Spectrum(dp.Ef, dp.gamma, 0);
	} // i, n, Ef

	// Call the power law fitting function to get the exponent, discard the normalization (tmp)
	double tmp;
	Power_Law_Fit(Efs, Phi_tracks, n, 1e5, tmp, gamma_track);
	Power_Law_Fit(Efs, Phi_cascades, n, 1e5, tmp, gamma_cascade);
}

struct chisq_helper_params {decay_params dp; bool visible, m1min;};
double Chisq_helper(double m1, void *pp)
{
	chisq_helper_params p = *(chisq_helper_params*)pp;
	decay_params dp = p.dp;
	bool visible = p.visible;
	bool m1min = p.m1min;

	dp.m1 = m1;

	double gamma_track, gamma_cascade, chisq;

	// Get spectral indices from decay
	IC_gamma(dp, visible, gamma_track, gamma_cascade);

	chisq = 0;

	// 1607.08006
	chisq += sq((gamma_track - 2.13) / 0.13);
	// Niederhausen:2015svt
	chisq += sq((gamma_cascade - 2.62) / 0.07);

	if (m1min) // include a term from Planck
	{
		// Get the correct masses set up
		double m2, m3, summnu;
		m2 = sqrt(Dmsq21 + sq(dp.m1));
		m3 = sqrt(Dmsq31 + sq(dp.m1));
		summnu = dp.m1 + m2 + m3;

		// 1807.06209 PlanckTT,TE,EE+lowE+lensing+BAO
		// assume that it is linear in summnus and that 95% => 2 sigma and that things are gaussian
		// then 0.12 at 95% => the standard deviation is 0.06 and the data seems to prefer zero
		chisq += sq(summnu / 0.06);

		// subtract off the best case scenario (m1 = 0) since we don't want to pay a penalty for the fact that Planck prefers summnus = 0
		chisq -= sq((sqrt(Dmsq21) + sqrt(Dmsq31)) / 0.06);
	}

	return chisq;
}

// m1min: true: minimize over m1 weighted by cosmology, false: use the value of m1 in dp
double Chisq(decay_params dp, bool visible, bool m1min)
{
	chisq_helper_params p = {dp, visible, m1min};

	if (not m1min)
		return Chisq_helper(dp.m1, &p);

	const gsl_min_fminimizer_type *T;
	gsl_min_fminimizer *s;
	gsl_function F;
	F.function = &Chisq_helper;
	F.params = &p;

	T = gsl_min_fminimizer_brent;
	s = gsl_min_fminimizer_alloc(T);

	double chisqmin;

	// Turn off the error handler but print the messages later if there is a convergence problem.
	// Be sure to check the output for "error"
	gsl_set_error_handler_off();
	int status;
	status = gsl_min_fminimizer_set(s, &F, 1e-4, 0, 1);
	if (status == 4) // almost certainly means m1=0 is the minimum
	{
		chisqmin = gsl_min_fminimizer_f_lower(s);
		if (visible)	printf("Visible: For gamma = %g and g = %g the minimum is at m1 = %g with chisq = %g\n", dp.gamma, dp.g31, gsl_min_fminimizer_x_lower(s), chisqmin);
		else			printf("Invisible: For gamma = %g and g = %g the minimum is at m1 = %g with chisq = %g\n", dp.gamma, dp.g31, gsl_min_fminimizer_x_lower(s), chisqmin);
		gsl_min_fminimizer_free(s);
		return chisqmin;
	}
	printf("Status = %i. Error = %s\n", status, gsl_strerror(status));
	printf("%g %g %g %g %g %g\n", gsl_min_fminimizer_x_lower(s), gsl_min_fminimizer_x_upper(s), gsl_min_fminimizer_x_minimum(s), gsl_min_fminimizer_f_lower(s), gsl_min_fminimizer_f_upper(s), gsl_min_fminimizer_f_minimum(s));

	// Loop while minimizing
	double a, b;
	do
	{
		status = gsl_min_fminimizer_iterate(s);
		a = gsl_min_fminimizer_x_lower(s);
		b = gsl_min_fminimizer_x_upper(s);
		status = gsl_min_test_interval(a, b, 1e-3, 0.0);
	} while (status == GSL_CONTINUE);

	chisqmin = gsl_min_fminimizer_f_minimum(s);

	if (visible)	printf("Visible: For gamma = %g and g = %g the minimum is at m1 = %g with chisq = %g\n", dp.gamma, dp.g31, gsl_min_fminimizer_x_minimum(s), chisqmin);
	else			printf("Invisible: For gamma = %g and g = %g the minimum is at m1 = %g with chisq = %g\n", dp.gamma, dp.g31, gsl_min_fminimizer_x_minimum(s), chisqmin);

	gsl_min_fminimizer_free(s);

	return chisqmin;
}

