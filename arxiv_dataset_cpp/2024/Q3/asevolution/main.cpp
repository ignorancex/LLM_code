//////////////////////////
// Copyright (c) 2015-2019 Julian Adamek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESSED OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//////////////////////////

//////////////////////////
// main.cpp
//////////////////////////
//
// main control sequence of Geneva N-body code with evolution of metric perturbations (gevolution)
//
// Author: Julian Adamek (Université de Genève & Observatoire de Paris & Queen Mary University of London)
//
// Last modified: November 2019
//----------------------------------------------------------
// Then modified into asevolution by Øyvind Christiansen
// Last modified: December 2023 (applies to all files in folder)
//////////////////////////

#include <stdlib.h>
#include <set>
#include <vector>
#include <random>
#ifdef HAVE_CLASS
#include "class.h"
#undef MAX // due to macro collision this has to be done BEFORE including LATfield2 headers!
#undef MIN
#endif
#include "LATfield2.hpp"
#include "metadata.hpp"
#include "class_tools.hpp"
#include "tools.hpp"
#include "background.hpp"
#include "Particles_gevolution.hpp"
#include "gevolution.hpp"
#include "ic_basic.hpp"
#include "ic_read.hpp"
#ifdef ICGEN_PREVOLUTION
#include "ic_prevolution.hpp"
#endif
#ifdef ICGEN_FALCONIC
#include "fcn/togevolution.hpp"
#endif
#include "radiation.hpp"
#include "parser.hpp"
#include "output.hpp"
#include "interface_main.hpp"
#include "field_solvers.hpp"
#include "interface_fifth.hpp"

using namespace std;
using namespace LATfield2;

int main(int argc, char **argv)
{

	// ========== 1. Initialisations ===============
	int n = 0, m = 0;
	int io_size = 0;
	int io_group_size = 0;
	double start_time = MPI_Wtime();
	double run_time, achitime = 0, time, pcltime = 0, writetime = 0, ICtime = 0;
	double achiSEtime = 0;
	int i, j, cycle = 0, snapcount = 0, snapcount2 = 0, pkcount = 0, animcount = 0;
	int restartcount = 0, numparam = 0, numspecies;
	int numsteps_ncdm[MAX_PCL_SPECIES - 2];
	long numpts3d;
	int box[3];
	double avgsource;
	double eosparT;
	double dtau, dtau_old, dx, tau, a, fourpiG, tmp, tau2;
	double fracp = 0, fracm = 0;
	double achiB = 0, aqB = 0;
	int toBreak = 0;
	double maxvel[MAX_PCL_SPECIES];
	FILE *outfile;
	char filename[2 * PARAM_MAX_LENGTH + 24];
	string h5filename;
	char *settingsfile = NULL;
	char *precisionfile = NULL;
	parameter *params = NULL;
	metadata sim;
	cosmology cosmo;
	icsettings ic;
	double T00hom;
	double eospar;
	double T00As_hom;
	double Vav;
	double phik0;
	double achiav, achi2av, aqav, aq2av;
	double T00J, T00E;
	bool writeSnap, writeSnap2, writeSpec;
	int done_hij = 0;

#ifndef H5_DEBUG
	H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
#endif

	// Read command line
	doReadCommandLine(argv, argc, settingsfile, precisionfile,
					  n, m, io_size, io_group_size);

#ifndef EXTERNAL_IO
	parallel.initialize(n, m);
#else
	if (!io_size || !io_group_size)
	{
		cout << "invalid number of I/O tasks and group sizes for I/O server (-DEXTERNAL_IO)" << endl;
		exit(-1000);
	}
	parallel.initialize(n, m, io_size, io_group_size);
	if (parallel.isIO())
		ioserver.start();
#endif

	// Write title screen
	doTitleAndSettings(n, m, settingsfile, numparam, start_time,
					   filename, cosmo, sim, ic, params);
	doSetBackground(sim);
	int N_as = sim.nAs_numsteps;
	cosmo.Omega_sym = 0.; // pre symmetry breaking
	double phib = (cosmo.mu_as / sqrt(cosmo.lambda_as)) * sqrt(1 - pow(cosmo.astar, 3));
	cosmo.Omega_sym0 = (0.5 * cosmo.Omega_m * pow(phib / cosmo.M_as, 2) + (-0.5 * pow(cosmo.mu_as * phib, 2) + 0.25 * cosmo.lambda_as * pow(phib, 4)) * 2998. * 2998 / 3.) * (double)sim.As_source_gravity;
	numparam = 0;

	h5filename.reserve(2 * PARAM_MAX_LENGTH);
	h5filename.assign(sim.output_path);

	box[0] = sim.numpts;
	box[1] = sim.numpts;
	box[2] = sim.numpts;

	Lattice lat(3, box, 1);
	Lattice latFT;
	latFT.initializeRealFFT(lat, 0);

	Particles_gevolution<part_simple, part_simple_info, part_simple_dataType> pcls_cdm;
	Particles_gevolution<part_simple, part_simple_info, part_simple_dataType> pcls_b;
	Particles_gevolution<part_simple, part_simple_info, part_simple_dataType> pcls_ncdm[MAX_PCL_SPECIES - 2];
	int nfields;
	if (sim.As_5th_force > 0)
		nfields = 4;
	else
		nfields = 3;
	Field<Real> *update_cdm_fields[nfields];
	Field<Real> *update_b_fields[nfields];
	Field<Real> *update_ncdm_fields[nfields];
	double f_params[3];
	set<long> IDbacklog[MAX_PCL_SPECIES];

	Field<Real> phi;
	Field<Real> scalar;	 // multi-use
	Field<Real> scalar2; // for RK4
	Field<Real> scalar3; // for RK4
	Field<Real> scalar4; // for RK4
	Field<Real> scalar5; // for RK4
	Field<Real> chi;
	Field<Real> Bi;

	Field<Real> source;
	Field<Real> Sij;
	Field<Real> aq;		// q = a^3 dot achi.
	Field<Real> achi;	// achi=aphi/aphi_0
	Field<Real> nT_cdm; // negative of cdm SE tensor trace for use with asymmetron (first order equal to rho)
	Field<Cplx> scalarFT;
	Field<Cplx> SijFT;
	Field<Cplx> BiFT;

	source.initialize(lat, 1);
	phi.initialize(lat, 1);
	scalar.initialize(lat, 1);
	scalar2.initialize(lat, 1);
	scalar3.initialize(lat, 1);
	scalar4.initialize(lat, 1);
	scalar5.initialize(lat, 1);
	chi.initialize(lat, 1);
	Bi.initialize(lat, 3);
	Sij.initialize(lat, 3, 3, symmetric);
	achi.initialize(lat, 1);
	aq.initialize(lat, 1);
	nT_cdm.initialize(lat, 1);
	scalarFT.initialize(latFT, 1);
	BiFT.initialize(latFT, 3);
	SijFT.initialize(latFT, 3, 3, symmetric);

	PlanFFT<Cplx> plan_source(&source, &scalarFT);
	PlanFFT<Cplx> plan_phi(&phi, &scalarFT);
	PlanFFT<Cplx> plan_scalar(&scalar, &scalarFT);
	PlanFFT<Cplx> plan_scalar2(&scalar2, &scalarFT);
	PlanFFT<Cplx> plan_scalar3(&scalar3, &scalarFT);
	PlanFFT<Cplx> plan_scalar4(&scalar4, &scalarFT);
	PlanFFT<Cplx> plan_scalar5(&scalar5, &scalarFT);
	PlanFFT<Cplx> plan_chi(&chi, &scalarFT);
	PlanFFT<Cplx> plan_Sij(&Sij, &SijFT);
	//
	PlanFFT<Cplx> plan_Bi(&Bi, &BiFT);
	PlanFFT<Cplx> plan_aq(&aq, &scalarFT);
	PlanFFT<Cplx> plan_achi(&achi, &scalarFT);
	PlanFFT<Cplx> plan_Tcdm_As(&nT_cdm, &scalarFT);

	update_cdm_fields[0] = &phi;
	update_cdm_fields[1] = &chi;
	update_cdm_fields[2] = &Bi;

	update_b_fields[0] = &phi;
	update_b_fields[1] = &chi;
	update_b_fields[2] = &Bi;

	update_ncdm_fields[0] = &phi;
	update_ncdm_fields[1] = &chi;
	update_ncdm_fields[2] = &Bi;

	fourpiG = 1.5 * sim.boxsize * sim.boxsize / C_SPEED_OF_LIGHT / C_SPEED_OF_LIGHT;

	Site x(lat);
	rKSite kFT(latFT);

	dx = 1.0 / (double)sim.numpts;
	numpts3d = (long)sim.numpts * (long)sim.numpts * (long)sim.numpts;

	for (i = 0; i < 3; i++) // particles may never move farther than to the adjacent domain
	{
		if (lat.sizeLocal(i) - 1 < sim.movelimit)
			sim.movelimit = lat.sizeLocal(i) - 1;
	}
	parallel.min(sim.movelimit);
	int fieldExtraInd;
	if (sim.As_5th_force > 0)
		fieldExtraInd = 3;
	else if (sim.As_5th_force_Newton > 0)
		fieldExtraInd = 2;
	if (sim.As_5th_force > 0 || sim.As_5th_force_Newton > 0)
	{
		update_cdm_fields[fieldExtraInd] = &achi;
		update_b_fields[fieldExtraInd] = &achi;
		update_ncdm_fields[fieldExtraInd] = &achi;
		f_params[2] = cosmo.mu_as / sqrt(cosmo.lambda_as) / cosmo.M_as; // symmetron vev
		if (parallel.rank() == 0)
			cout << "Defined f_params[2] = vev/M = " << f_params[2] << endl;
	}

	a = 1. / (1. + sim.z_in);
	double a_old = a;

	avgsource = a * (((cosmo.Omega_cdm + cosmo.Omega_b + bg_ncdm(a, cosmo)) / a) + ((cosmo.Omega_Lambda - cosmo.Omega_asymmetron) * a * a) + (cosmo.Omega_rad / a / a) + (cosmo.Omega_fld * exp(3. * cosmo.wa_fld * (a - 1.)) / pow(a, 1. + 3. * (cosmo.w0_fld + cosmo.wa_fld))));

	tau = particleHorizon_old(a, fourpiG, cosmo, avgsource); // IC should be before SSB
	tau2 = tau;

	doSetNewTimestep(dtau, dx, a, N_as, sim, cosmo, avgsource, fourpiG, pkcount, snapcount, snapcount2, writeSnap, writeSnap2, writeSpec, cycle, 0.02);
	dtau_old = 0;
	//==================end initialisations==============================
	// Set initial conditions
	ICtime = MPI_Wtime();
	doICs(cosmo, a, sim, ic, achi, aq, avgsource, fourpiG, tau, dtau, dtau_old, h5filename,
		  &pcls_cdm, &pcls_b, pcls_ncdm, maxvel, phi, chi, Bi, Sij,
		  source, scalarFT, BiFT, SijFT, plan_phi, plan_chi, plan_Bi, plan_Sij,
		  plan_source, params, numparam, numspecies, cycle,
		  snapcount, pkcount, restartcount, IDbacklog,
		  plan_achi, plan_aq, nT_cdm);
	if (sim.fifth > 0)
	{
		if (strcmp(sim.As_ICopt, "relaxed") == 0)
			doConstructAchiSEtensor(
				source, Bi, Sij, dx, sim.boxsize, a, cosmo, achi, aq,
				sim, nT_cdm, scalarFT, plan_Tcdm_As,
				avgsource, eospar, eosparT, Vav, T00J,
				T00E, T00hom, T00As_hom, achiav, achi2av, aqav, aq2av,
				fracp, fracm,
				cycle, fourpiG); // without interaction
		doAchiICs(sim, ic, cosmo, achi, aq,
				  scalarFT, plan_achi, fourpiG, avgsource, a, achiB, aqB, numpts3d,
				  dtau, dx, cycle, nT_cdm);
	}
	ICtime = MPI_Wtime() - ICtime;
	time = ICtime;
	parallel.sum<double>(ICtime);
	parallel.max<double>(time);
	COUT << "Used " << ICtime << " seconds on ICs\n";
	COUT << "max time for proc " << time << " seconds on ICs\n";
#ifdef HAVE_HEALPIX
	COUT << "Have healpix\n";
#endif

	COUT << COLORTEXT_GREEN << " ICs complete. Starting loop." << COLORTEXT_RESET << endl
		 << endl;
	if (sim.breakafterzini > 0)
		return 0;
	double b = a;

	//=======================start main loop==============================
	while (true)
	{
		b = a_old;
		rungekutta4bg(b, fourpiG, dtau_old, cosmo, avgsource);
		if (sim.fifth > 0 && a > sim.aMG && cycle % CYCLE_INFO_INTERVAL == 0)
			COUT << 1 / a_old - 1 << ": Evolving achi to redshift z = " << 1 / b - 1
				 << ", N_as = " << N_as << endl;

		time = MPI_Wtime();
		// evolve scalar field
		if (sim.fifth > 0 && a > sim.aMG && cycle > 0 && sim.updateScalar > 0)
			doAchiFieldUpdate(dtau_old, N_as, a_old, fourpiG, sim, dx,
							  avgsource, cosmo, achi, aq, scalar, scalar2, scalar3, scalar4,
							  scalar5, nT_cdm, cycle, achiB);
		time = MPI_Wtime() - time;
		achitime += time;
		if (toBreak > 0)
			break;

		// find SE tensors // + misc
		if (sim.updatePtcls > 0 || sim.solveEinstein > 0)
			doConstructSEtensor(
				cosmo, a, sim, &pcls_cdm, &pcls_b, pcls_ncdm,
				phi, scalar, chi, achi, Bi, source, Sij);
		time = MPI_Wtime();
		if (sim.fifth > 0 && (sim.As_source_gravity > 0 || sim.As_compute_SE > 0))
			doConstructAchiSEtensor(
				source, Bi, Sij, dx, sim.boxsize, a, cosmo, achi, aq,
				sim, nT_cdm, scalarFT, plan_Tcdm_As,
				avgsource, eospar, eosparT, Vav, T00J,
				T00E, T00hom, T00As_hom, achiav, achi2av, aqav, aq2av,
				fracp, fracm, cycle, fourpiG); // without interaction
		else
			doFindBackgroundQuantities(
				source, Bi, Sij, dx, sim.boxsize, a, cosmo, achi, aq,
				sim, nT_cdm, avgsource, eospar, eosparT, Vav, T00J,
				T00E, T00hom, T00As_hom, achiav, achi2av, aqav, aq2av,
				fracp, fracm, cycle);

		time = MPI_Wtime() - time;
		achiSEtime += time;
		if (sim.solveEinstein > 0)
			doSolveEinsteinEquation(
				cosmo, a, sim, phi, chi, Bi, Sij, source, scalar2, T00hom, avgsource,
				scalarFT, BiFT, SijFT, plan_source, plan_scalar2, plan_phi, plan_chi,
				plan_Bi, plan_Sij, dx, fourpiG, dtau_old, cycle, phik0, done_hij);

		if (sim.fifth > 0 && sim.As_source_gravity > 0 && (strcmp(sim.bckopt, "sym") == 0 || strcmp(sim.bckopt, "symdyn") == 0))
			doUpdateBackgroundEnergy(cosmo, sim, a_old, achiB, aqB);
		// output background quantities
		doBackgroundOutput(
			phik0, cosmo, a, sim, filename, outfile, cycle, snapcount, tau,
			fourpiG, avgsource, T00hom, N_as, T00As_hom, eospar, eosparT, Vav, T00J, T00E, achiav, achi2av,
			aqav, aq2av, achiB, aqB, tau2,
			fracp, fracm,
			maxvel[0], pkcount);

		time = MPI_Wtime();
		// lightcone output
		if (sim.num_lightcone > 0)
		{
			writeLightcones(sim, cosmo, avgsource, 
							fourpiG, a, tau, dtau, dtau_old, maxvel[0],
							cycle, h5filename + sim.basename_lightcone,
							&pcls_cdm, &pcls_b, pcls_ncdm, &phi, &chi,
							&Bi, &Sij, &achi, &aq,
							IDbacklog,
							&plan_Sij, &SijFT, done_hij);
		}

		// snapshot output
		if (writeSnap)
		{
			COUT << COLORTEXT_CYAN << " writing snapshot" << COLORTEXT_RESET << " at z = " << ((1. / a) - 1.) << " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
			writeSnapshots(0, sim, cosmo, avgsource, fourpiG, a, dtau_old,
						   snapcount, h5filename + sim.basename_snapshot, &pcls_cdm,
						   &pcls_b, pcls_ncdm, &phi, &achi, &aq,
						   &chi, &Bi, &Sij,
						   &source, &scalar,
						   &plan_source, &plan_scalar, &plan_Sij,
						   &SijFT, done_hij

			);

			snapcount++;
		}
		if (writeSnap2)
		{
			COUT << COLORTEXT_CYAN << " writing secondary snapshot" << COLORTEXT_RESET << " at z = " << ((1. / a) - 1.) << " (cycle " << cycle << "), tau/boxsize = " << tau << endl;
			writeSnapshots(1, sim, cosmo, avgsource, fourpiG, a, dtau_old,
						   snapcount2, h5filename + sim.basename_snapshot, &pcls_cdm,
						   &pcls_b, pcls_ncdm, &phi, &achi, &aq,
						   &chi, &Bi, &Sij,
						   &source, &scalar,
						   &plan_source, &plan_scalar, &plan_Sij,
						   &SijFT, done_hij

			);

			snapcount2++;
		}
		// animation output
		if (sim.animation > 0 && sim.zanimationstart > 1. / a - 1 && sim.zanimationend < 1. / a - 1 && (cycle % sim.animationeveryn) == 0)
		{
			COUT << COLORTEXT_CYAN << " writing animation" << COLORTEXT_RESET << " at z = " << ((1. / a) - 1.) << endl;
			writeAnimation(sim, cosmo, avgsource, fourpiG, a, dtau_old,
						   animcount, h5filename + sim.basename_animation,
						   &phi, &achi, &aq, &chi, &Bi, &Sij,
						   &source, &scalar, &plan_source, &plan_scalar,
						   &plan_Sij, &SijFT, done_hij

			);

			animcount++;
		}

		// power spectra
		if (writeSpec)
		{
			doWriteSpectra(dtau_old, cycle, tau, sim, cosmo, avgsource, fourpiG,
						   a, pkcount, &pcls_cdm, &pcls_b, pcls_ncdm, &phi, &achi, &aq, &chi,
						   &Bi, &Sij, &source, &scalar,
						   &scalarFT, &BiFT, &SijFT, &plan_phi, &plan_achi, &plan_aq, &plan_chi,
						   &plan_Bi, &plan_source,
						   &plan_Sij, &plan_scalar, done_hij);
		}
		time = MPI_Wtime() - time;
		writetime += time;
		// finish simulation?
		if (pkcount >= sim.num_pk && snapcount >= sim.num_snapshot && snapcount2 >= sim.num_snapshot2 || a >= 1)
		{
			for (i = 0; i < sim.num_lightcone; i++)
			{
				if (sim.lightcone[i].z + 1. < 1. / a)
					i = sim.num_lightcone + 1;
			}
			if (i == sim.num_lightcone)
				break; // simulation complete
		}

		// print information to console
		doCoutCycleInfo(cycle, dtau_old, dx, a, fourpiG, cosmo, sim, numsteps_ncdm, maxvel, avgsource);

		a_old = a;
		time = MPI_Wtime();
		// update velocities, positions and background
		if (sim.As_compute_dynbck > 0)
			doUpdateAchiBackground(achiB, aqB, a, fourpiG, dtau_old, cosmo, sim, avgsource);
		if (sim.updatePtcls > 0)
			doUpdateParticlesAndBackground(sim, cosmo, avgsource, fourpiG, a,
										   pcls_cdm, pcls_b, pcls_ncdm, dtau, dtau_old, dx, maxvel,
										   f_params, ic, update_cdm_fields, update_b_fields, update_ncdm_fields,
										   nfields, numspecies, numsteps_ncdm);
		else
			rungekutta4bg(a, fourpiG, dtau, cosmo, avgsource);

		time = MPI_Wtime() - time;
		pcltime += time;
		if (sim.breakWhenNoDW > 0 && a > cosmo.astar * 1.2 && (fracp < 0.01 || fracm < 0.01))
			toBreak += 1;
		toBreak = toBreak + (int)(sim.breakafterz >= 1. / a_old - 1.);
		if (toBreak > 0)
			break;

		tau2 = particleHorizon(a, fourpiG, cosmo, avgsource); // LCDM, outputted at background for comparison
		if (strcmp(sim.bckopt, "lcdm"))
			tau = tau2;
		else
			tau += dtau;
		dtau_old = dtau;

		doSetNewTimestep(dtau, dx, a, N_as, sim, cosmo, avgsource, fourpiG, pkcount, snapcount, snapcount2, writeSnap, writeSnap2, writeSpec, cycle, maxvel[0]);
		cycle++;
	}

	COUT << COLORTEXT_GREEN << " simulation complete." << COLORTEXT_RESET << endl;
	COUT << COLORTEXT_WHITE << " --> Final sum Omega = " << avgsource << endl;
	COUT << COLORTEXT_WHITE << " -----> Set Omega_asymmetron = " << -(1 - avgsource - cosmo.Omega_asymmetron) << endl
		 << endl;
	COUT << COLORTEXT_WHITE << " --> Final tau(z=0) = " << tau + dtau * (a - 1) / (a_old - a) << endl
		 << " -----> The one used is " << sim.lightcone->tauobs << endl;

	run_time = MPI_Wtime() - start_time;
	parallel.sum(run_time);
	parallel.sum(pcltime);
	parallel.sum(writetime);
	parallel.sum(achitime);
	parallel.sum(achiSEtime);

	COUT << endl
		 << "BENCHMARK" << endl;
	COUT << "total execution time  : " << hourMinSec(run_time) << endl;
	COUT << "total number of cycles: " << cycle << endl;
	COUT << "\nTime spent (%) breakdown\n--------------\n";
	COUT << "IC time: " << ICtime / run_time * 100. << endl;
	COUT << "Particle update: " << pcltime / run_time * 100. << endl;
	COUT << "Achi update:     " << achitime / run_time * 100. << endl;
	COUT << "Achi SE finder   " << achiSEtime / run_time * 100. << endl;
	COUT << "Output write     " << writetime / run_time * 100. << endl;

	if (parallel.isRoot())
	{
		// print runtime to file
		snprintf(filename, SIZFILENAME, "%sruntime.dat", sim.output_path);
		FILE *outfile = fopen(filename, "w");
		if (outfile == NULL)
			cout << " error opening runtime output file!" << endl;
		else
		{
			fprintf(outfile, "# Result of runtime test\n");
			fprintf(outfile, "wtime = %.2f\n", run_time);
			fprintf(outfile, "icwtime = %.2f\n", ICtime);
			fprintf(outfile, "pclwtime = %.2f\n", pcltime);
			fprintf(outfile, "achiwtime = %.2f\n", achitime);
			fprintf(outfile, "achisewtime = %.2f\n", achiSEtime);
			fprintf(outfile, "outputwtime = %.2f\n", writetime);
			fclose(outfile);
		}
	}

#ifdef EXTERNAL_IO
	ioserver.stop();
#endif

	return 0;
}
