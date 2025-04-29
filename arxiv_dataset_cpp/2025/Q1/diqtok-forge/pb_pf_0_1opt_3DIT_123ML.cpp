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
	int Nint = 30; //number of points for Gaussian quadrature integration
	int Nscan = 20; //number of points for scan of maximum on Bloch sphere before the Newton iteration
	int N = 30; //number of qubits in the token
	int nT = 5; //acceptance threshold for number of qubits
	double p0b{ 0.029 }, p1b{ 0.927 }; //model parameters for bank
	double p0f{ 0.029 }, p1f{ 0.927 }; //model parameters for forger
	double pf, pftot;
	int nba, i, j, nf1, nf2, nf3;
	double phif, thetaf;
	double thetaf2, thetaf3, phif3;
	double tmp;
	int N1, N2, N3;
	std::fstream file, filep;
	int ithetaf, Nang;
	std::vector<double> thetaWeights, thetaPoints, phiWeights, phiPoints;
	std::vector<double> probDist, probDistWeights, pt_N1_nf1, pt_terms_probDist_weights;
	std::string str, filename;
	double normgrad;
	int step, maxstep = 100;
	double pfgrad[2];
	double pfHess[2][2], invHess[2][2];
	double detHess, idetHess, pftmp, thetaftmp, phiftmp;
	int myrank, numranks, ir;
	std::vector<double> optthetaf;
	std::vector<double> n_thetab;

	auto t1 = std::chrono::high_resolution_clock::now();

#ifdef MPIusage
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	//std::cout << "myrank = " << myrank << " numranks = " << numranks << "\n";

	if (myrank == 0) {
		std::cout << "Using mpi, numranks: " << numranks << "\n";
#endif

		//reading input from control file
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
		file.close();

		//output of input from control file
		std::cout << "Nint:  " << Nint << "\n";
		std::cout << "Nscan: " << Nscan << "\n";
		std::cout << "N:     " << N << "\n";
		std::cout << "nT:    " << nT << "\n";
		std::cout << "p0b:   " << p0b << "\n";
		std::cout << "p1b:   " << p1b << "\n";
		std::cout << "p0f:   " << p0f << "\n";
		std::cout << "p1f:   " << p1f << "\n";
		std::cout << "pb, pf0, pf1opt, pf3DIT, pf1ML, pf2ML, pf3ML \n";

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
#endif


	std::cout.precision(16);

	//set integration points and weights for theta and phi for Gaußian quadrature
	thetaWeights.resize(Nint);
	thetaPoints.resize(Nint);
	phiWeights.resize(Nint);
	phiPoints.resize(Nint);
	get_weights_points(thetaWeights.data(), thetaPoints.data(), Nint, 0., M_PI);
	get_weights_points(phiWeights.data(), phiPoints.data(), Nint, 0., 2. * M_PI);


	//uniform distribution of thetaB,phiB, which is the a priori probability distribution
	probDist.resize(Nint * Nint);
	probDistWeights.resize(Nint * Nint);
	for (i = 0; i < Nint; ++i) {
		for (j = 0; j < Nint; ++j) {
			probDist[i * Nint + j] = sin(thetaPoints[i]) / (4. * M_PI);
			probDistWeights[i * Nint + j] = probDist[i * Nint + j] * thetaWeights[i] * phiWeights[j];
		}
	}
	pt_terms_probDist_weights.resize(Nint * Nint);;
	pt_N1_nf1.resize(Nint * Nint);
	n_thetab.resize(Nint);

#ifdef MPIusage
	if (myrank == 0) {
#endif
		//probability of self acceptance of bank token
		pf = 0.;
		for (nba = 0; nba <= N; ++nba) {
			pftmp = 0.;
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					pftmp += probDistWeights[i * Nint + j]
						* pt(N, nba, p0b, p1b, thetaPoints[i], phiPoints[j], thetaPoints[i], phiPoints[j]);
				}
			}
			//std::cout << nba << " " << pftmp << "\n";
			if (nba <= nT) {
				pf += pftmp;
			}
		}
		std::cout << "pb=     " << std::setw(20) << std::fixed << std::left << pf << "\n";

		//probability of acceptance for just guessing the angles
		pf = 0.;
		thetaf = 0.;
		phif = 0.;
		for (i = 0; i < Nint; ++i) {
			for (j = 0; j < Nint; ++j) {
				for (nba = 0; nba <= nT; ++nba) {
					pf += probDistWeights[i * Nint + j]
						* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
				}
			}
		}
		std::cout << "pf0=    " << std::setw(20) << std::fixed << std::left << pf << "\n";
#ifdef MPIusage
	}
#endif


	//acceptance probability for optimal generation of forged token using one measurement
	optthetaf.resize(N+1, 0.);
	N1 = N;
	pftot = 0.;
	std::fill(n_thetab.begin(), n_thetab.end(), 0.);
	for (nf1 =0; nf1 <= N1; ++nf1) {
#ifdef MPIusage
		if (nf1 % numranks == myrank) {
#endif
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					pt_terms_probDist_weights[i * Nint + j] = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j]) * probDistWeights[i * Nint + j];
				}
			}

			normgrad = 1.;
			Nang = Nscan;
			while (normgrad >= 1.0E-15 && Nang <= 1000) {
				//search for the maximum thetaf,phif on a coarse grid
				pf = 0.;
				for (ithetaf = 0; ithetaf <= Nang; ++ithetaf) {
					thetaftmp = ithetaf * M_PI / Nang;
					phiftmp = 0;
					pftmp = 0;
					for (i = 0; i < Nint; ++i) {
						for (j = 0; j < Nint; ++j) {
							for (nba = 0; nba <= nT; ++nba) {
								pftmp += pt_terms_probDist_weights[i * Nint + j]
									* pt(N, nba, p0b, p1b, thetaftmp, phiftmp, thetaPoints[i], phiPoints[j]);
							}
						}
					}
					if (pftmp > pf) {
						pf = pftmp;
						thetaf = thetaftmp;
						phif = phiftmp;
					}

				}
				//searching for the maximum thetaf,phif with 2d Newton algorithm
				step = 0;
				while (step < maxstep) {
					//pf = 0;
					pfgrad[0] = 0.; // dpfdthetaf
					pfgrad[1] = 0.; // dpfdphif
					pfHess[0][0] = 0.; //d2pfdthetafdthetaf
					pfHess[0][1] = 0.; //d2pfdthetafdphif
					pfHess[1][0] = 0.; //d2pfdphifdthetaf
					pfHess[1][1] = 0.; //d2pfdphifdphif
					for (i = 0; i < Nint; ++i) {
						for (j = 0; j < Nint; ++j) {
							for (nba = 0; nba <= nT; ++nba) {
								//pf += pt_terms_probDist_weights[i * Nint + j]
								//	* pt(nq, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								pfgrad[0] += pt_terms_probDist_weights[i * Nint + j]
									* dptdtheta1(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								pfgrad[1] += pt_terms_probDist_weights[i * Nint + j]
									* dptdphi1(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								pfHess[0][0] += pt_terms_probDist_weights[i * Nint + j]
									* d2ptdtheta1dtheta1(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								pfHess[0][1] += pt_terms_probDist_weights[i * Nint + j]
									* d2ptdtheta1dphi1(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								pfHess[1][1] += pt_terms_probDist_weights[i * Nint + j]
									* d2ptdphi1dphi1(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
							}
						}
					}
					pfHess[1][0] = pfHess[0][1]; //using symmetry
					detHess = pfHess[0][0] * pfHess[1][1] - pfHess[1][0] * pfHess[0][1];
					normgrad = sqrt(pfgrad[0] * pfgrad[0] + pfgrad[1] * pfgrad[1]);
					++step;

					//std::cout << "thetaf= " << thetaf << " phif= " << phif << " normgrad=" << normgrad << "\n";// << " pf=" << pf << "\n";
					if (normgrad < 1E-15) {
						break;
					}

					if (std::abs(detHess) > 0.) {
						idetHess = 1.0 / detHess;
						invHess[0][0] = pfHess[1][1] * idetHess;
						invHess[1][0] = -pfHess[1][0] * idetHess;
						invHess[0][1] = -pfHess[0][1] * idetHess;
						invHess[1][1] = pfHess[0][0] * idetHess;

						thetaf -= invHess[0][0] * pfgrad[0] + invHess[0][1] * pfgrad[1];
						//phif -= invHess[1][0] * pfgrad[0] + invHess[1][1] * pfgrad[1];
						thetaf = fmod(thetaf, 2. * M_PI);
						//phif = fmod(phif, 2. * M_PI);
					}
					else {
						thetaf += pfgrad[0];
						//phif += pfgrad[1];
					}
				}
				if (normgrad >= 1E-15) {
					Nang += 5;
					std::cout << "WARNING:N1=" << N1 << " nf1=" << nf1 << " normgrad=" << normgrad << " increase Nang to " << Nang << "\n";
				}
			}
			if (normgrad >= 1E-15) {
				std::cout << "WARNING:N1=" << N1 << " nf1=" << nf1 << " normgrad=" << normgrad << " final not workink!\n";
			}

			//calculate final value of pf
			pf = 0.;
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					for (nba = 0; nba <= N; ++nba) {
						tmp = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j])
							* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
						n_thetab[i] += nba * tmp * phiWeights[j] / (2 * M_PI * N);
						if (nba <= nT) {
							pf += tmp * probDistWeights[i * Nint + j];
						}
					}
				}
			}

			pftot += pf;
			std::cout << " nf1= " << std::setw(3) << std::right << nf1;
			std::cout << " thetaf= " << std::setw(20) << std::left << std::fixed << thetaf;
			std::cout << " phif= " << std::setw(20) << std::left << phif;
			std::cout << " pf= " << std::setw(20) << std::left << pf;
			std::cout << " ng= " << std::setw(20) << std::left << std::scientific << normgrad;
			std::cout << " step= " << std::setw(3) << std::right << step << "\n";

			optthetaf[nf1] = thetaf;

			/*
			std::cout << thetaf;
			if (nf1 == nq) {
				std::cout << "]\n";
			}
			else {
				std::cout << ",\n";
			}
			*/

#ifdef MPIusage
		}
#endif
	}
#ifdef MPIusage
	if (myrank != 0) {
		MPI_Reduce(&pftot, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(optthetaf.data(), nullptr, N + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(n_thetab.data(), nullptr, Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	} else {
		MPI_Reduce(MPI_IN_PLACE, &pftot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, optthetaf.data(), N+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, n_thetab.data(), Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
		filep.precision(16);
		filep.open("set_optimal_thetaf_angle_1m_.py", std::ios::out);
		filep << "#one measurement at angles thetaf1 = 0, phif1 = 0 (choice of coordinate system is freedom of forger)\n";
		filep << "#we have N = "<< N <<" qubits in the quantum token and the bank accepts the token if at maximum nT = "<< nT <<" photons are detected in the measurement\n";
		filep << "#Using the given optimal thetaf angle(phif can be set to 0) the fake token acceptance is " << std::fixed << std::left << pftot*100 <<" %\n";
		filep << "#nf1 is the number of measured photons of the token(forger tries to make a dark measurement)\n";
		filep << "#output of function is optimal thetaf angle, the phif angle can be set to 0.\n";
		filep << "def set_optimal_thetaf_angle_1m_(nf1):\n";
		filep << "    optthetaf = np.zeros("<< N+1 << ")\n";
		for (nf1 = 0; nf1 <= N; ++nf1) {
			filep << "    optthetaf[" << std::setw(3) << std::right << nf1 << "]=" << std::setw(20) << std::left << optthetaf[nf1] << "\n";
		}
		filep << "    return optthetaf[nf1]\n";
		filep.close();
		std::cout << "pf1opt= " << std::setw(20) << std::fixed << std::left << pftot << "\n";
		std::cout << "n(thetab) for opt1: \n";
		for (i = 0; i < Nint; ++i) {
			std::cout << thetaPoints[i] << " " << std::setw(20) << std::fixed << std::left << n_thetab[i] << "\n";
		}
#ifdef MPIusage
	}
#endif


	//probability of acceptance for direct inversion tomography
	N1 = N / 3;
	N2 = N1;
	N3 = N - N1 - N2;
	thetaf2 = 0.5 * M_PI;
	thetaf3 = 0.5 * M_PI;
	phif3 = 0.5 * M_PI;
	pf = 0.;
	ir = -1;
	std::fill(n_thetab.begin(), n_thetab.end(), 0.);
	for (nf1 = 0; nf1 <= N1; ++nf1) {
		for (nf2 = 0; nf2 <= N2; ++nf2) {
			for (nf3 = 0; nf3 <= N3; ++nf3) {
#ifdef MPIusage
				++ir;
				if (ir % numranks == myrank) {
#endif
					direct_inversion_tomography(N1, N2, N3, nf1, nf2, nf3, thetaf, phif);

					for (i = 0; i < Nint; ++i) {
						for (j = 0; j < Nint; ++j) {
							for (nba = 0; nba <= N; ++nba) {
								tmp = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j])
									* pt(N2, nf2, p0f, p1f, thetaf2, 0., thetaPoints[i], phiPoints[j])
									* pt(N3, nf3, p0f, p1f, thetaf3, phif3, thetaPoints[i], phiPoints[j])
									* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								n_thetab[i] += nba * tmp * phiWeights[j] / (2 * M_PI * N);
								if (nba <= nT) {
									pf += tmp * probDistWeights[i * Nint + j];
								}
							}
						}
					}
#ifdef MPIusage
				}
#endif
			}
		}
	}
#ifdef MPIusage
	if (myrank != 0) {
		MPI_Reduce(&pf, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(n_thetab.data(), nullptr, Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, &pf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, n_thetab.data(), Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
		std::cout << "pf3DIT= " << std::setw(20) << std::fixed << std::left << pf << "\n";
		std::cout << "n(thetab) for DIT: \n";
		for (i = 0; i < Nint; ++i) {
			std::cout << thetaPoints[i] << " " << std::setw(20) << std::fixed << std::left << n_thetab[i] << "\n";
		}
#ifdef MPIusage
	}
#endif
	

	//probability of acceptance for maximum likelihood estimator using one measurement
	N1 = N;
	pf = 0.;
	std::fill(n_thetab.begin(), n_thetab.end(), 0.);
	for (nf1 = 0; nf1 <= N1; ++nf1) {
#ifdef MPIusage
		if (nf1 % numranks == myrank) {
#endif
			ml_estimator_1m(p0f, p1f, N1, nf1, thetaf);
			phif = 0.;

			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					for (nba = 0; nba <= N; ++nba) {
						tmp = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j])
							* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
						n_thetab[i] += nba * tmp * phiWeights[j] / (2 * M_PI * N);
						if (nba <= nT) {
							pf += tmp * probDistWeights[i * Nint + j];
						}
					}
				}
			}
#ifdef MPIusage
		}
#endif
	}
#ifdef MPIusage
	if (myrank != 0) {
		MPI_Reduce(&pf, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(n_thetab.data(), nullptr, Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, &pf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, n_thetab.data(), Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
	std::cout << "pf1ML=  " << std::setw(20) << std::fixed << std::left << pf << "\n";
	std::cout << "n(thetab) for ML1: \n";
	for (i = 0; i < Nint; ++i) {
		std::cout << thetaPoints[i] << " " << std::setw(20) << std::fixed << std::left << n_thetab[i] << "\n";
	}
#ifdef MPIusage
	}
#endif


	//probability of acceptance for maximum likelihood estimator using two measurements
	thetaf2 = 0.5 * M_PI;
	N1 = N / 2;
	N2 = N - N1;
	pf = 0.;
	ir = -1;
	std::fill(n_thetab.begin(), n_thetab.end(), 0.);
	for (nf1 = 0; nf1 <= N1; ++nf1) {
		for (nf2 = 0; nf2 <= N2; ++nf2) {
#ifdef MPIusage
			++ir;
			if (ir % numranks == myrank) {
#endif
				ml_estimator_2m(p0f, p1f, N1, N2, nf1, nf2, thetaf2, thetaf, phif);

				for (i = 0; i < Nint; ++i) {
					for (j = 0; j < Nint; ++j) {
						for (nba = 0; nba <= N; ++nba) {
							tmp = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j])
								* pt(N2, nf2, p0f, p1f, thetaf2, 0., thetaPoints[i], phiPoints[j])
								* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
							n_thetab[i] += nba * tmp * phiWeights[j] / (2 * M_PI * N);
							if (nba <= nT) {
								pf += tmp * probDistWeights[i * Nint + j];
							}
						}
					}
				}
#ifdef MPIusage
			}
#endif
		}
	}
#ifdef MPIusage
	if (myrank != 0) {
		MPI_Reduce(&pf, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(n_thetab.data(), nullptr, Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, &pf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, n_thetab.data(), Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
		std::cout << "pf2ML=  " << std::setw(20) << std::fixed << std::left << pf << "\n";
		std::cout << "n(thetab) for ML2: \n";
		for (i = 0; i < Nint; ++i) {
			std::cout << thetaPoints[i] << " " << std::setw(20) << std::fixed << std::left << n_thetab[i] << "\n";
		}
#ifdef MPIusage
	}
#endif


	//probability of acceptance for maximum likelihood estimator using three measurements
	thetaf2 = 0.5 * M_PI;
	thetaf3 = 0.5 * M_PI;
	phif3 = 0.5 * M_PI;
	N1 = N / 3;
	N2 = N1;
	N3 = N - N1 - N2;
	pf = 0.;
	ir = -1;
	std::fill(n_thetab.begin(), n_thetab.end(), 0.);
	for (nf1 = 0; nf1 <= N1; ++nf1) {
		for (nf2 = 0; nf2 <= N2; ++nf2) {
			for (nf3 = 0; nf3 <= N3; ++nf3) {
#ifdef MPIusage
				++ir;
				if (ir % numranks == myrank) {

#endif
					ml_estimator_3m(p0f, p1f, N1, N2, N3, nf1, nf2, nf3, thetaf2, thetaf3, phif3, thetaf, phif);

					for (i = 0; i < Nint; ++i) {
						for (j = 0; j < Nint; ++j) {
							for (nba = 0; nba <= N; ++nba) {
								tmp = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j])
									* pt(N2, nf2, p0f, p1f, thetaf2, 0., thetaPoints[i], phiPoints[j])
									* pt(N3, nf3, p0f, p1f, thetaf3, phif3, thetaPoints[i], phiPoints[j])
									* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								n_thetab[i] += nba * tmp * phiWeights[j] / (2 * M_PI * N);
								if (nba <= nT) {
									pf += tmp * probDistWeights[i * Nint + j];
								}
							}
						}
					}
#ifdef MPIusage
				}
#endif
			}
		}
	}
#ifdef MPIusage
	if (myrank != 0) {
		MPI_Reduce(&pf, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(n_thetab.data(), nullptr, Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Reduce(MPI_IN_PLACE, &pf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, n_thetab.data(), Nint, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
		std::cout << "pf3ML=  " << std::setw(20) << std::fixed << std::left << pf << "\n";
		std::cout << "n(thetab) for ML3: \n";
		for (i = 0; i < Nint; ++i) {
			std::cout << thetaPoints[i] << " " << std::setw(20) << std::fixed << std::left << n_thetab[i] << "\n";
		}

		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001 << " s\n";
#ifdef MPIusage
	}
	MPI_Finalize();
#endif

	return 0;
}