#include <stdio.h>
#include <cmath>

#include "Figures.h"
#include "SM.h"
#include "Depletion.h"
#include "Regeneration.h"
#include "IceCube.h"
#include "Analytic.h"
#include "Redshift.h"

// Calculates the flavor ratio at the Earth in the low energy (large coupling) limit
// Suggestions:
// 1. Turn on/off different couplins to see the effect of different couplings
// 2. Change the initial flavor ratio to see the effect of different production mechanisms
void Flavor_Ratio()
{
	double p[3][3];
	decay_params dp = dp_bm;

	dp.g21 = 1;
	dp.gp21 = 1;
	dp.g31 = 1;
	dp.gp31 = 1;
	dp.g32 = 1;
	dp.gp32 = 1;

	printf("SM\n");
	for (int i = 0; i < 3; i++)
	{
		for (int f = 0; f < 3; f++)
		{
			p[i][f] = PSM(flavor(i), flavor(f), dp);
			p[i][f] += Pdep(flavor(i), flavor(f), dp);
		} // f, 3
	} // i, 3

	// Initial flux, (1,2,0) is pion decay, (1,0,0) is neutron decay, etc.
	double phii[3] = {1, 2, 0};
	double phif[3];
	for (int f = 0; f < 3; f++)
		phif[f] = 0;
	
	for (int i = 0; i < 3; i++)
	{
		for (int f = 0; f < 3; f++)
			phif[f] += phii[i] * p[i][f];
	}
	printf("(1:2:0) -> (%.3g:%.3g:%.3g)\n", 1., phif[m] / phif[e], phif[t] / phif[e]);
}

// Invisible decay (no regeneration) as a function of final energy for different off-diagonal elements
void Invisible()
{
	printf("Invisible\n");

	double Efmin, Efmax, Efscale, g, psm, p21, p31, p32, pall;
	int n;
	flavor a, b;
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;
	g = 1e-6;

	FILE *data = fopen("data/Invisible.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%i %i %g %g %g\n", a, b, dp.m1, dp.x, g);
	for (int i = 0; i <= n; i++)
	{
		dp.Ef = Efmin * pow(Efscale, i);
		psm = PSM(a, b, dp);

		// (21)
		dp.g21 = g;
		dp.gp21 = g;
		dp.g31 = 0;
		dp.gp31 = 0;
		dp.g32 = 0;
		dp.gp32 = 0;
		p21 = psm + Pdep(a, b, dp);

		// (31)
		dp.g21 = 0;
		dp.gp21 = 0;
		dp.g31 = g;
		dp.gp31 = g;
		dp.g32 = 0;
		dp.gp32 = 0;
		p31 = psm + Pdep(a, b, dp);

		// (32)
		dp.g21 = 0;
		dp.gp21 = 0;
		dp.g31 = 0;
		dp.gp31 = 0;
		dp.g32 = g;
		dp.gp32 = g;
		p32 = psm + Pdep(a, b, dp);

		// All three
		dp.g21 = g;
		dp.gp21 = g;
		dp.g31 = g;
		dp.gp31 = g;
		dp.g32 = g;
		dp.gp32 = g;
		pall = psm + Pdep(a, b, dp);

		fprintf(data, "%g %g %g %g %g %g\n", dp.Ef, psm, p21, p31, p32, pall);
	} // i, Ef, n
	fclose(data);
}

// Visible decay as a function of final energy for different off-diagonal elements
void Visible_g()
{
	printf("Visible_g\n");

	double Efmin, Efmax, Efscale, g, psm, pdep, p21, p31, p32, pall;
	int n;
	flavor a, b;
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;
	g = 1e-6;

	FILE *data = fopen("data/Visible_g.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%i %i %g %g %g %g\n", a, b, g, dp.m1, dp.x, dp.gamma);
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);
		psm = PSM(a, b, dp);

		// (21)
		dp.g21 = g;
		dp.gp21 = g;
		dp.g31 = 0;
		dp.gp31 = 0;
		dp.g32 = 0;
		dp.gp32 = 0;
		pdep = Pdep(a, b, dp);
		p21 = Preg(a, b, dp) + pdep + psm;

		// (31)
		dp.g21 = 0;
		dp.gp21 = 0;
		dp.g31 = g;
		dp.gp31 = g;
		dp.g32 = 0;
		dp.gp32 = 0;
		pdep = Pdep(a, b, dp);
		p31 = Preg(a, b, dp) + pdep + psm;

		// (32)
		dp.g21 = 0;
		dp.gp21 = 0;
		dp.g31 = 0;
		dp.gp31 = 0;
		dp.g32 = g;
		dp.gp32 = g;
		pdep = Pdep(a, b, dp);
		p32 = Preg(a, b, dp) + pdep + psm;

		// All three
		dp.g21 = g;
		dp.gp21 = g;
		dp.g31 = g;
		dp.gp31 = g;
		dp.g32 = g;
		dp.gp32 = g;
		pdep = Pdep(a, b, dp);
		pall = Preg(a, b, dp) + pdep + psm;

		fprintf(data, "%g %g %g %g %g %g\n", dp.Ef, psm, p21, p31, p32, pall);
	} // i, Ef, n
	fclose(data);
}

// Visible decay as a function of lightest neutrino mass m1
void Visible_m1()
{
	printf("Visible_m1\n");

	double Efmin, Efmax, Efscale, g, p, psm;
	int n;
	flavor a, b;
	const int n_m1 = 3;
	double m1s[n_m1] = {0, 0.1, 1}; // the different masses to uses
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;

	g = 1e-6;

	FILE *data = fopen("data/Visible_m1.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%i %i %g %g %g %i ", a, b, g, dp.x, dp.gamma, n_m1);
	for (int i = 0; i < n_m1; i++)
		fprintf(data, "%g ", m1s[i]);
	fprintf(data, "\n");
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);
		psm = PSM(a, b, dp);
		fprintf(data, "%g %g ", dp.Ef, psm);

		for (int j = 0; j < n_m1; j++)
		{
			dp.m1 = m1s[j];
			p = Preg(a, b, dp) + Pdep(a, b, dp) + psm;
			fprintf(data, "%g ", p);
		} // j, n_m1
		fprintf(data, "\n");
	} // i, Ef, n
	fclose(data);
}

// Visible decay as a function of final energy for different spectral indices gamma
void Visible_gamma()
{
	printf("Visible_gamma\n");

	double Efmin, Efmax, Efscale, p;
	int n;
	flavor a, b;
	const int n_gamma = 4;
	double gammas[n_gamma] = {1, 2, 3, 4}; // the different spectral indices to use
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;

	FILE *data = fopen("data/Visible_gamma.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%i %i %g %g %g %i ", a, b, dp.g21, dp.x, dp.m1, n_gamma);
	for (int i = 0; i < n_gamma; i++)
		fprintf(data, "%g ", gammas[i]);
	fprintf(data, "\n");
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);

		for (int j = 0; j < n_gamma; j++)
		{
			if (j == 0) fprintf(data, "%g %g ", dp.Ef, PSM(a, b, dp) * pow(1 + dp.x, dp.gamma));

			dp.gamma = gammas[j];
			p = Preg(a, b, dp) + Pdep(a, b, dp) + PSM(a, b, dp);
			if (dp.x_is_redshift) p *= pow(1 + dp.x, dp.gamma);
			fprintf(data, "%g ", p);
		} // j, n_gamma
		fprintf(data, "\n");
	} // i, Ef, n
	fclose(data);
}

// Visible decay as a function of final energy for different redshift evolutions
void Visible_R()
{
	printf("Visible_R\n");

	double Ef, Efmin, Efmax, Efscale, psm;
	const int n = 1e2;
	flavor a, b;
	double p_sm[n + 1], p_delta5[n + 1], p_delta1[n + 1], p_delta15[n + 1], p_ykbh[n + 1], p_hermes[n + 1];

	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;

	# pragma omp parallel for schedule(dynamic)
	for (int i = 0; i <= n; i++)
	{
		decay_params dp = dp_bm;

		dp.Ef = Efmin * pow(Efscale, i);

		// delta function redshift evolutions
		dp.x = 0.5;
		psm = PSM(a, b, dp);
		p_delta5[i] = psm + Pdep(a, b, dp) + Preg(a, b, dp);
		dp.x = 1;
		p_sm[i] = PSM(a, b, dp);
		p_delta1[i] = p_sm[i] + Pdep(a, b, dp) + Preg(a, b, dp);
		dp.x = 1.5;
		psm = PSM(a, b, dp);
		p_delta15[i] = psm + Pdep(a, b, dp) + Preg(a, b, dp);

		// Two continuous red shift evolutions
		p_ykbh[i] = RSE_integral(a, b, dp, true);
		p_hermes[i] = RSE_integral(a, b, dp, false);
	} // i, Ef, n

	// Write to file
	FILE *data = fopen("data/Visible_R.txt", "w");
	fprintf(data, "%i %i %g %g %g\n", a, b, dp_bm.g21, dp_bm.m1, dp_bm.gamma);
	for (int i = 0; i <= n; i++)
	{
		Ef = Efmin * pow(Efscale, i);
		fprintf(data, "%g %g %g %g %g %g %g\n", Ef, p_sm[i], p_delta5[i], p_delta1[i], p_delta15[i], p_ykbh[i], p_hermes[i]);
	} // i, Ef, n
	fclose(data);
}

// Visible decay as a function of final energy for different flavor channels
void Visible_f()
{
	printf("Visible_f\n");

	double Efmin, Efmax, Efscale, p;
	int n;
	flavor a, b;
	const int n_fi = 2; // initial flavors
	const int n_ff = 3; // final flavors
	flavor fis[n_fi] = {e, m};
	flavor ffs[n_ff] = {e, m, t};
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	FILE *data = fopen("data/Visible_f.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%g %g %g %g %i %i ", dp.g21, dp.x, dp.m1, dp.gamma, n_fi, n_ff);
	for (int i = 0; i < n_fi; i++)
		fprintf(data, "%i ", fis[i]);
	for (int i = 0; i < n_ff; i++)
		fprintf(data, "%i ", ffs[i]);
	fprintf(data, "\n");
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);
		fprintf(data, "%g ", dp.Ef);

		for (int j = 0; j < n_fi; j++)
		{
			a = fis[j]; // initial flavor
			for (int k = 0; k < n_ff; k++)
			{
				b = ffs[k]; // final flavor

				p = PSM(a, b, dp);
				fprintf(data, "%g ", p);

				p += Preg(a, b, dp) + Pdep(a, b, dp);
				fprintf(data, "%g ", p);
			} // k, n_ff
		} // j, n_fi
		fprintf(data, "\n");
	} // i, Ef, n
	fclose(data);
}

// Track to cascade ratio
void Rtc()
{
	printf("Rtc\n");

	double Efmin, Efmax, Efscale, g, rtc_sm, rtc_inv, rtc_vis;
	int n;
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	g = 1e-6;

	FILE *data = fopen("data/Rtc.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%g %g %g %g\n", g, dp.m1, dp.x, dp.gamma);
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);

		// SM
		dp.g21 = 0;
		dp.gp21 = 0;
		dp.g31 = 0;
		dp.gp31 = 0;
		dp.g32 = 0;
		dp.gp32 = 0;
		rtc_sm = Rtc(dp, false); // do invisible only to avoid the expensive visible calculation

		dp.g21 = g;
		dp.gp21 = g;
		dp.g31 = g;
		dp.gp31 = g;
		dp.g32 = g;
		dp.gp32 = g;
		// Invisible
		rtc_inv = Rtc(dp, false);
		// Visible
		rtc_vis = Rtc(dp, true);

		fprintf(data, "%g %g %g %g\n", dp.Ef, rtc_sm, rtc_inv, rtc_vis);
	} // i, Ef, n
	fclose(data);
}

// Validates that the analytic expression as a function of final energy agrees with the numerical integral in the case of no cosmology
void Analytic_Validate()
{
	double Efmin, Efmax, Efscale, pvis_num, pvis_anal;
	int n;
	decay_params dp = dp_bm;
	flavor a, b;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;

	// Analytic expression only works for one coupling on at a time
	// Only g31 and gp31 are on
	dp.g21 = 0;
	dp.gp21 = 0;
	dp.g32 = 0;
	dp.gp32 = 0;

	dp.x = Lastro(.01);
	dp.x_is_redshift = false; // Analytic expression only works for no cosmology
	dp.m1 = 1e-1;

	FILE *data = fopen("data/Analytic_Validate.txt", "w");
	fprintf(data, "%i %i %g %g %g %g\n", a, b, dp.g31, dp.m1, dp.x, dp.gamma);
	for (int i = 0; i <= n; i++)
	{
		dp.Ef = Efmin * pow(Efscale, i);

		// Numerical integral
		pvis_num = Preg(a, b, dp);
		// Analytic expression
		pvis_anal = Preg_anal(a, b, dp);

		fprintf(data, "%g %g %g\n", dp.Ef, pvis_num, pvis_anal);
	} // i, Ef, n
	fclose(data);
}

// The spectral index measured by IceCube for visible/invisible and tracks/cascades as a function of coupling g
void IC_gamma()
{
	printf("IC_gamma\n");

	double gamma, m1, z;
	const int n = 1e2;
	double gamma_track_invs[n + 1], gamma_track_viss[n + 1], gamma_cascade_invs[n + 1], gamma_cascade_viss[n + 1];

	double gmin, gmax, gscale;
	gmin = 1e-8;
	gmax = 1e-5;
	gscale = pow(gmax / gmin, 1. / n);

	gamma = 2;
	m1 = 0;
	z = 1;

	int count = 0;
	# pragma omp parallel for schedule(dynamic)
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * count / n);

		decay_params dp = dp_bm;

		dp.gamma = gamma;
		dp.m1 = m1;
		dp.x = z;

		double g = gmin * pow(gscale, i);
		dp.g21 = g;
		dp.gp21 = g;
		dp.g31 = g;
		dp.gp31 = g;
		dp.g32 = g;
		dp.gp32 = g;

		// Invisible
		IC_gamma(dp, false, gamma_track_invs[i], gamma_cascade_invs[i]);
		// Visible
		IC_gamma(dp, true, gamma_track_viss[i], gamma_cascade_viss[i]);

		count++;
	} // i, g, n

	// Write to file
	FILE *data = fopen("data/IC_gamma.txt", "w");
	fprintf(data, "%g %g %g\n", m1, gamma, z);
	for (int i = 0; i <= n; i++)
	{
		double g = gmin * pow(gscale, i);
		fprintf(data, "%g ", g);
		fprintf(data, "%g %g ", gamma_track_invs[i], gamma_cascade_invs[i]);
		fprintf(data, "%g %g ", gamma_track_viss[i], gamma_cascade_viss[i]);
		fprintf(data, "\n");
	} // i, g, n
	fclose(data);
}

// The difference in track/cascade spectral indices at the Earth as a function of coupling g and lightest neutrino mass m1
void IC_gamma_2D()
{
	printf("IC_gamma_2D\n");

	const int n = 1e2;
	double Delta_gammas_inv[n + 1][n + 1], Delta_gammas_vis[n + 1][n + 1];

	double m1min, m1max, m1scale;
	m1min = 1e-4;
	m1max = 1e0;
	m1scale = pow(m1max / m1min, 1. / n);

	double g, gmin, gmax, gscale;
	gmin = 1e-8;
	gmax = 1e-5;
	gscale = pow(gmax / gmin, 1. / n);

	int count = 0;
	# pragma omp parallel for collapse(2) schedule(dynamic)
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= n; j++)
		{
			if(count % 5 == 0) printf("%.2g\n", 1.0 * count / ((n + 1) * (n + 1)));

			double gamma_track, gamma_cascade;
			decay_params dp = dp_bm;

			g = gmin * pow(gscale, i);
			dp.g21 = g;
			dp.gp21 = g;
			dp.g31 = g;
			dp.gp31 = g;
			dp.g32 = g;
			dp.gp32 = g;

			dp.m1 = m1min * pow(m1scale, j);

			// Invisible
			IC_gamma(dp, false, gamma_track, gamma_cascade);
			Delta_gammas_inv[i][j] = gamma_cascade - gamma_track;
			// Visible
			IC_gamma(dp, true, gamma_track, gamma_cascade);
			Delta_gammas_vis[i][j] = gamma_cascade - gamma_track;

			count++;
		} // j, n, m1
	} // i, n, g

	// Write to files
	FILE *data_inv = fopen("data/IC_gamma_2D_inv.txt", "w");
	FILE *data_vis = fopen("data/IC_gamma_2D_vis.txt", "w");

	fprintf(data_inv, "%g %g\n", dp_bm.gamma, dp_bm.x);
	fprintf(data_vis, "%g %g\n", dp_bm.gamma, dp_bm.x);

	// Define the dimensions of the grid
	for (int i = 0; i <= n; i++)
	{
		fprintf(data_inv, "%g ", gmin * pow(gscale, i));
		fprintf(data_vis, "%g ", gmin * pow(gscale, i));
	} // i, n, g
	fprintf(data_inv, "\n");
	fprintf(data_vis, "\n");

	for (int i = 0; i <= n; i++)
	{
		fprintf(data_inv, "%g ", m1min * pow(m1scale, i));
		fprintf(data_vis, "%g ", m1min * pow(m1scale, i));
	} // i, n, m1
	fprintf(data_inv, "\n");
	fprintf(data_vis, "\n");

	// Fill in the grid
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= n; j++)
		{
			fprintf(data_inv, "%g ", Delta_gammas_inv[i][j]);
			fprintf(data_vis, "%g ", Delta_gammas_vis[i][j]);
		} // j, n, m1
		fprintf(data_inv, "\n");
		fprintf(data_vis, "\n");
	} // i, n, g
	fclose(data_inv);
	fclose(data_vis);
}

// Visible decay as a function of scalar (g), pseudo-scalar (gp), or both
void Visible_SPS()
{
	printf("Visible_SPS\n");

	double Efmin, Efmax, Efscale, g, psm, pdep, ps, pps, psps;
	int n;
	flavor a, b;
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;
	g = 1e-6;

	FILE *data = fopen("data/Visible_SPS.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%i %i %g %g %g %g\n", a, b, g, dp.m1, dp.x, dp.gamma);
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);
		psm = PSM(a, b, dp);

		// Scalar only
		dp.g21 = g;
		dp.g31 = g;
		dp.g32 = g;
		dp.gp21 = 0;
		dp.gp31 = 0;
		dp.gp32 = 0;
		pdep = Pdep(a, b, dp);
		ps = Preg(a, b, dp) + pdep + psm;

		// Pseudo-scalar only
		dp.g21 = 0;
		dp.g31 = 0;
		dp.g32 = 0;
		dp.gp21 = g;
		dp.gp31 = g;
		dp.gp32 = g;
		pdep = Pdep(a, b, dp);
		pps = Preg(a, b, dp) + pdep + psm;

		// Both
		dp.g21 = g;
		dp.g31 = g;
		dp.g32 = g;
		dp.gp21 = g;
		dp.gp31 = g;
		dp.gp32 = g;
		pdep = Pdep(a, b, dp);
		psps = Preg(a, b, dp) + pdep + psm;

		fprintf(data, "%g %g %g %g %g\n", dp.Ef, psm, ps, pps, psps);
	} // i, Ef, n
	fclose(data);
}

// Visible decay as a function of final energy for different combinations of nu->nu, nu->nubar, and nu->nu + nu->nubar
void Visible_nnb()
{
	printf("Visible_nnb\n");

	double Efmin, Efmax, Efscale, psm, pdep, pnn, pnnb, pnn_nnb;
	int n;
	flavor a, b;
	decay_params dp = dp_bm;

	n = 1e2;
	Efmin = 1e3;
	Efmax = 1e9;
	Efscale = pow(Efmax / Efmin, 1. / n);

	a = m;
	b = e;

	FILE *data = fopen("data/Visible_nnb.txt", "w");
	setbuf(data, NULL);
	fprintf(data, "%i %i %g %g %g %g\n", a, b, dp.g31, dp.m1, dp.x, dp.gamma);
	for (int i = 0; i <= n; i++)
	{
		printf("%g\n", 1.0 * i / n);

		dp.Ef = Efmin * pow(Efscale, i);
		psm = PSM(a, b, dp);

		// nu->nu
		dp.nnb = nu2nu;
		pdep = Pdep(a, b, dp);
		pnn = Preg(a, b, dp) + pdep + psm;

		// nu->nubar
		dp.nnb = nu2nubar;
		pdep = Pdep(a, b, dp);
		pnnb = Preg(a, b, dp) + pdep + psm;

		// nu->nu + nu->nubar
		dp.nnb = nu2nu_and_nu2nubar;
		pdep = Pdep(a, b, dp);
		pnn_nnb = Preg(a, b, dp) + pdep + psm;

		fprintf(data, "%g %g %g %g %g\n", dp.Ef, psm, pnn, pnnb, pnn_nnb);
	} // i, Ef, n
	fclose(data);
}

void Chisq()
{
	printf("Chisq\n");

	const int n = 1e2;
	double Chisqs_inv[n + 1][n + 1], Chisqs_vis[n + 1][n + 1];

	double gammamin, gammamax, gammastep;
	gammamin = 2;
	gammamax = 3.5;
	gammastep = (gammamax - gammamin) / n;

	double g, gmin, gmax, gscale;
	gmin = 1e-9;
	gmax = 1e-5;
	gscale = pow(gmax / gmin, 1. / n);

	bool m1min = false;

	int count = 0;
	# pragma omp parallel for collapse(2) schedule(dynamic)
	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= n; j++)
		{
			if(count % 5 == 0) printf("%.2g\n", 1.0 * count / ((n + 1) * (n + 1)));

			decay_params dp = dp_bm;

			g = gmin * pow(gscale, i);
			dp.g21 = g;
			dp.gp21 = g;
			dp.g31 = g;
			dp.gp31 = g;
			dp.g32 = g;
			dp.gp32 = g;

			dp.gamma = gammamin + j * gammastep;

			// Invisible
			Chisqs_inv[i][j] = Chisq(dp, false, m1min);
			// Visible
			Chisqs_vis[i][j] = Chisq(dp, true, m1min);

			count++;
		} // j, n, gamma
	} // i, n, g

	// Write to files
	FILE *data_inv = fopen("data/Chisq_inv.txt", "w");
	FILE *data_vis = fopen("data/Chisq_vis.txt", "w");

	fprintf(data_inv, "%g\n", dp_bm.x);
	fprintf(data_vis, "%g\n", dp_bm.x);

	for (int i = 0; i <= n; i++)
	{
		fprintf(data_inv, "%g ", gmin * pow(gscale, i));
		fprintf(data_vis, "%g ", gmin * pow(gscale, i));
	} // i, n, g
	fprintf(data_inv, "\n");
	fprintf(data_vis, "\n");

	for (int i = 0; i <= n; i++)
	{
		fprintf(data_inv, "%g ", gammamin + i * gammastep);
		fprintf(data_vis, "%g ", gammamin + i * gammastep);
	} // i, n, gamma
	fprintf(data_inv, "\n");
	fprintf(data_vis, "\n");

	for (int i = 0; i <= n; i++)
	{
		for (int j = 0; j <= n; j++)
		{
			fprintf(data_inv, "%g ", Chisqs_inv[i][j]);
			fprintf(data_vis, "%g ", Chisqs_vis[i][j]);
		} // j, n, gamma
		fprintf(data_inv, "\n");
		fprintf(data_vis, "\n");
	} // i, n, g
	fclose(data_inv);
	fclose(data_vis);
}
