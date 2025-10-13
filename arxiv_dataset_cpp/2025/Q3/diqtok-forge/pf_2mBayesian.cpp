#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cmath>
#include "integrate.h"
#include "qtokentools.h"
#include <vector>
#include <algorithm>
#include <string>
#ifdef MPIusage
#include <mpi.h>
#endif

int main(int argc, char** argv)
{
	int Nint = 30;
	int Nscan = 20;
	int N = 30;
	int nT = 5;
	double p0b{ 0.029 }, p1b{ 0.927 };
	double p0f{ 0.029 }, p1f{ 0.927 };
	double pf;
	int nba, i, j, nf1, nf2;
	double phif, thetaf, thetaf2;
	int N1, N2;
	std::fstream file;
	int ithetaf2;
	int thetaf2Start{ 0 }, thetaf2End{ 100 }, thetaf2N{ 100 };
	std::vector<double> thetaWeights, thetaPoints, phiWeights, phiPoints;
	std::vector<double> probDist, probDistWeights, intWeights, pt_N1_nf1, pt_terms_probDist_weights;
	std::vector<double> p1probDist, p2probDist, pt_N2_nf2;
	std::string str, filename;
	double S0, S1, S2, pmax, pf2Ba;
	double U, Unf2, Uopt, thetaf2opt, pmarginal;

	auto t1 = std::chrono::high_resolution_clock::now();

#ifdef MPIusage
	int myrank, numranks;
	int ir;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	//std::cout << "myrank = " << myrank << " numranks = " << numranks << "\n";


	if (myrank == 0) {
		std::cout << "Using mpi, numranks: " << numranks << "\n";
#endif
		file.precision(16);
		file.open("control", std::ios::in);
		if (!file.is_open()) {
			std::cerr << "Error: File control cannot be opened!" << std::endl;
			return 1;
		}
		file >> str >> Nint;
		if (!file.good()) {
			std::cerr << "Error during reading Nint from control file.";
			return 1;
		}
		file >> str >> Nscan;
		if (!file.good()) {
			std::cerr << "Error during reading Nscan from control file.";
			return 1;
		}
		file >> str >> N;
		if (!file.good()) {
			std::cerr << "Error during reading N from control file.";
			return 1;
		}
		file >> str >> nT;
		if (!file.good()) {
			std::cerr << "Error during reading nT from control file.";
			return 1;
		}
		file >> str >> p0b;
		if (!file.good()) {
			std::cerr << "Error during reading p0b from control file.";
			return 1;
		}
		file >> str >> p1b;
		if (!file.good()) {
			std::cerr << "Error during reading p1b from control file.";
			return 1;
		}
		file >> str >> p0f;
		if (!file.good()) {
			std::cerr << "Error during reading p0f from control file.";
			return 1;
		}
		file >> str >> p1f;
		if (!file.good() && !file.eof()) {
			std::cerr << "Error during reading p1f from control file.";
			return 1;
		}
		file >> str >> thetaf2Start >> thetaf2End >> thetaf2N;
		if (!file.good() && !file.eof()) {
			std::cerr << "Error during reading thetaf2's from control file.";
			return 1;
		}
		file.close();
		std::cout << "Nint:  " << Nint << "\n";
		std::cout << "Nscan: " << Nscan << "\n";
		std::cout << "N:     " << N << "\n";
		std::cout << "nT:    " << nT << "\n";
		std::cout << "p0b:   " << p0b << "\n";
		std::cout << "p1b:   " << p1b << "\n";
		std::cout << "p0f:   " << p0f << "\n";
		std::cout << "p1f:   " << p1f << "\n";
		std::cout << "thetaf2: " << thetaf2Start << " " << thetaf2End << " " << thetaf2N << "\n";
		std::cout << "two measurement with Bayesian method\n";
#ifdef MPIusage
	}
	MPI_Bcast(&Nint, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Nscan, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nT, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&p0b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&p1b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&p0f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&p1f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&thetaf2Start, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&thetaf2End, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&thetaf2N, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

	std::cout.precision(16);

	//set integration points and weights for theta and phi for Gaußian quadrature
	thetaWeights.resize(Nint);
	thetaPoints.resize(Nint);
	phiWeights.resize(Nint);
	phiPoints.resize(Nint);
	get_weights_points(thetaWeights.data(), thetaPoints.data(), Nint, 0., M_PI);
	get_weights_points(phiWeights.data(), phiPoints.data(), Nint, 0., 2. * M_PI);


	//uniform distribution of thetaB,phiB, which is a priori probability distribution
	probDist.resize(Nint * Nint);
	probDistWeights.resize(Nint * Nint);
	intWeights.resize(Nint * Nint);
	S0 = 0; //Shanon entropy of uniform distribution
	for (i = 0; i < Nint; ++i) {
		for (j = 0; j < Nint; ++j) {
			probDist[i * Nint + j] = sin(thetaPoints[i]) / (4. * M_PI);
			probDistWeights[i * Nint + j] = probDist[i * Nint + j] * thetaWeights[i] * phiWeights[j];
			intWeights[i * Nint + j] = thetaWeights[i] * phiWeights[j];
			S0 += probDist[i * Nint + j] * log(probDist[i * Nint + j]) * intWeights[i * Nint + j];
		}
	}
	pt_N1_nf1.resize(Nint * Nint);
	pt_N2_nf2.resize(Nint * Nint);
	pt_terms_probDist_weights.resize(Nint * Nint);
	p1probDist.resize(Nint * Nint);
	p2probDist.resize(Nint * Nint);

	//first measurement is performed with thetaf1 = 0, phif1 = 0
	N1 = N / 2;
	N2 = N - N1;
	pf2Ba = 0.;
#ifdef MPIusage
	ir = -1;
#endif
	for (nf1 = 0; nf1 <= N1; ++nf1) {
#ifdef MPIusage
		++ir;
		if (ir % numranks == myrank) {
#endif
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					pt_N1_nf1[i * Nint + j] = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j]);
				}
			}

			//determine a posterior probability distribution after first measurement
			pf = 0.;
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					p1probDist[i * Nint + j] = pt_N1_nf1[i * Nint + j] * probDist[i * Nint + j];
					pf += p1probDist[i * Nint + j] * intWeights[i * Nint + j];
				}
			}
			//normalize the distribution and determine Shanon entropy
			pf = 1. / pf;
			S1 = 0.;
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					p1probDist[i * Nint + j] *= pf;
					S1 += p1probDist[i * Nint + j] * log(p1probDist[i * Nint + j]) * intWeights[i * Nint + j];
				}
			}

			//try to find optimal thetaf2 by considering average gain
			Uopt = 0.;
			thetaf2opt = 0.5 * M_PI;
			for (ithetaf2 = thetaf2Start; ithetaf2 <= thetaf2End; ++ithetaf2) {
				thetaf2 = ithetaf2 * M_PI / thetaf2N;
				U = 0.;
				for (nf2 = 0; nf2 <= N2; ++nf2) {
					//determine a posterior probability distribution after second measurement
					pf = 0.;
					for (i = 0; i < Nint; ++i) {
						for (j = 0; j < Nint; ++j) {
							p2probDist[i * Nint + j] = pt(N2, nf2, p0f, p1f, thetaf2, 0., thetaPoints[i], phiPoints[j]) * p1probDist[i * Nint + j];
							pf += p2probDist[i * Nint + j] * intWeights[i * Nint + j];
						}
					}
					//normalize the distribution and determine Shanon entropy
					pmarginal = pf;
					pf = 1. / pf;
					S2 = 0.;
					for (i = 0; i < Nint; ++i) {
						for (j = 0; j < Nint; ++j) {
							p2probDist[i * Nint + j] *= pf;
							S2 += p2probDist[i * Nint + j] * log(p2probDist[i * Nint + j]) * intWeights[i * Nint + j];
						}
					}
					Unf2 = S2 - S1;
					U += Unf2 * pmarginal;
				}
				if (U >= Uopt) {
					Uopt = U;
					thetaf2opt = thetaf2;
				}
				//std::cout << "thetaf2= " << std::left << std::setw(20) << thetaf2 << " U= " << std::left << std::setw(20) << U << "\n";
			}
			std::cout << "nf1= " << std::right << std::setw(3) << nf1 << " thetaf2opt= " << std::left << std::setw(20) << thetaf2opt << " U= " << std::left << std::setw(20) << Uopt << "\n";
			thetaf2 = thetaf2opt;


			for (nf2 = 0; nf2 <= N2; ++nf2) {
				for (i = 0; i < Nint; ++i) {
					for (j = 0; j < Nint; ++j) {
						pt_N2_nf2[i * Nint + j] = pt(N2, nf2, p0f, p1f, thetaf2, 0., thetaPoints[i], phiPoints[j]);
					}
				}

				//determine a posterior probability distribution after second measurement
				pf = 0.;
				for (i = 0; i < Nint; ++i) {
					for (j = 0; j < Nint; ++j) {
						p2probDist[i * Nint + j] = pt_N2_nf2[i * Nint + j] * p1probDist[i * Nint + j];
						pf += p2probDist[i * Nint + j] * intWeights[i * Nint + j];
					}
				}
				//normalize the distribution
				pf = 1. / pf;
				for (i = 0; i < Nint; ++i) {
					for (j = 0; j < Nint; ++j) {
						p2probDist[i * Nint + j] *= pf;
					}
				}

				//find arg maximum of the distribution
				thetaf = thetaPoints[0];
				phif = phiPoints[0];
				pmax = p2probDist[0];
				for (i = 0; i < Nint; ++i) {
					for (j = 0; j < Nint; ++j) {
						if (p2probDist[i * Nint + j] > pmax) {
							pmax = p2probDist[i * Nint + j];
							thetaf = thetaPoints[i];
							phif = phiPoints[j];
						}
					}
				}

				//calculate final value of pf
				pf = 0.;
				for (i = 0; i < Nint; ++i) {
					for (j = 0; j < Nint; ++j) {
						for (nba = 0; nba <= nT; ++nba) {
							pf += pt_N1_nf1[i * Nint + j]
								* pt_N2_nf2[i * Nint + j]
								* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j])
								* probDistWeights[i * Nint + j];
						}
					}
				}

				pf2Ba += pf;
			}
#ifdef MPIusage
		}
#endif
	}

#ifdef MPIusage
	pf = 0.;
	MPI_Reduce(&pf2Ba, &pf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (myrank == 0) {
		pf2Ba = pf;
#endif
		std::cout << "pf2Ba=" << std::setw(20) << std::left << pf2Ba << "\n";

		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001 << " s\n";
#ifdef MPIusage
	}
#endif

#ifdef MPIusage
	MPI_Finalize();
#endif

	return 0;
}