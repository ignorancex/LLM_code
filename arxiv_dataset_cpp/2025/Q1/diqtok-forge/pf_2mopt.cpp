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

struct Result2m{
	int N1;
	int nf1;
	double thetaf2;
	double pf;
};

void order_by_pf(std::vector<Result2m>& results) {
	std::sort(results.begin(), results.end(),
		[](const Result2m& a, const Result2m& b)
		{
			return a.pf > b.pf;
		}
		);
}

int main(int argc, char **argv)
{
	int Nint = 30;
	int Nscan = 20;
	int N = 30;
	int nT = 5;
	double p0b{ 0.029 }, p1b{ 0.927 };
	double p0f{ 0.029 }, p1f{ 0.927 };
	double pf, pfthetaf2;
	int nba, i, j, nf1, nf2;
	double phif,thetaf,thetaf2;
	int N1, N2;
	std::fstream file;
	std::vector<Result2m> results2mloc, results2m;
	std::vector<int> recvcounts; // Anzahl der Elemente pro Prozess
	std::vector<int> displs;    // Offsets im Empfangspuffer
	int ithetaf2, ir, irloc, Numr, Numrloc, ithetaf, iphif, Nang;
	int thetaf2Start{ 1 }, thetaf2End{ 99 }, thetaf2N{ 100 };
	int myrank, numranks;
	std::vector<double> thetaWeights, thetaPoints, phiWeights, phiPoints;
	std::vector<double> probDist, probDistWeights, pt_N1_nf1, pt_terms_probDist_weights;
	std::string str, filename;
	double normgrad;
	int step, maxstep = 100;
	double pfgrad[2];
	double pfHess[2][2], invHess[2][2];
	double detHess, idetHess, pftmp, thetaftmp, phiftmp;


	auto t1 = std::chrono::high_resolution_clock::now();


#ifdef MPIusage
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	//std::cout << "myrank = " << myrank << " numranks = " << numranks << "\n";

	//create MPI datatype for the Result2m struct 
	MPI_Datatype MPI_Result2m;
	int block_lengths[4] = { 1, 1, 1, 1 };
	MPI_Aint offsets[4];
	offsets[0] = offsetof(Result2m, N1);
	offsets[1] = offsetof(Result2m, nf1);
	offsets[2] = offsetof(Result2m, thetaf2);
	offsets[3] = offsetof(Result2m, pf);
	MPI_Datatype types[4] = { MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
	MPI_Type_create_struct(4, block_lengths, offsets, types, &MPI_Result2m);
	MPI_Type_commit(&MPI_Result2m);


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
		if (!file.good()) {
			std::cerr << "Error during reading thetaf2's from control file.";
			return 1;
		}
		file >> str >> filename;
		if (!file.good() && !file.eof()) {
			std::cerr << "Error during reading filename from control file.";
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
		std::cout << "filename: " << filename << "\n";
		std::cout << "optimal Measurement with constant N1 and thetaf2(nf1) \n";
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
	get_weights_points(phiWeights.data(),     phiPoints.data(), Nint, 0., 2. * M_PI);


	//uniform distribution of thetaB,phiB, which is a priori probability distribution
	probDist.resize(Nint * Nint);
	probDistWeights.resize(Nint * Nint);
	for (i = 0; i < Nint; ++i) {
		for (j = 0; j < Nint; ++j) {
			probDist[i*Nint+j] = sin(thetaPoints[i]) / (4. * M_PI);
			probDistWeights[i * Nint + j] = probDist[i * Nint + j] * thetaWeights[i] * phiWeights[j];
		}
	}
	pt_N1_nf1.resize(Nint * Nint);
	pt_terms_probDist_weights.resize(Nint * Nint);;

	//probability that bank accepts fake tokens generated from 2 measurements using angles
	//thetaF1 = 0, phiF1 = 0 in the 1. measurement
	//thetaF2 > 0, phiF2 = 0 in the 2. measurement

	Numr = 0;
	for (N1 = 1; N1 < N; ++N1) {
		for (nf1 = 0; nf1 <= N1; ++nf1) {
			for (ithetaf2 = thetaf2Start; ithetaf2 <= thetaf2End; ++ithetaf2) {
				++Numr;
			}
		}
	}
#ifdef MPIusage
	Numrloc = Numr / numranks + 1;
	if (myrank == 0) {
		std::cout << "Numr:    " << Numr << "\n";
		std::cout << "Numrloc: " << Numrloc << "\n";
	}
#else
	Numrloc = Numr;
#endif

	results2mloc.resize(Numrloc);
	ir = -1;
	irloc = -1;
	for (N1 = 1; N1 < N; ++N1) {
		N2 = N - N1;
		for (nf1 = 0; nf1 <= N1; ++nf1) {
			for (i = 0; i < Nint; ++i) {
				for (j = 0; j < Nint; ++j) {
					pt_N1_nf1[i * Nint + j] = pt(N1, nf1, p0f, p1f, 0., 0., thetaPoints[i], phiPoints[j]);
				}
			}
			for (ithetaf2 = thetaf2Start; ithetaf2 <= thetaf2End; ++ithetaf2) {
				++ir;
#ifdef MPIusage
				if (ir % numranks == myrank) {
#endif
					++irloc;
					thetaf2 = ithetaf2 * M_PI / thetaf2N;
					pfthetaf2 = 0.;
					for (nf2 = 0; nf2 <= N2; ++nf2) {
						for (i = 0; i < Nint; ++i) {
							for (j = 0; j < Nint; ++j) {
								pt_terms_probDist_weights[i * Nint + j] = pt_N1_nf1[i * Nint + j] * pt(N2, nf2, p0f, p1f, thetaf2, 0., thetaPoints[i], phiPoints[j]) * probDistWeights[i * Nint + j];
							}
						}
						
						normgrad = 1.;
						Nang = Nscan;
						while (normgrad >= 1.0E-15 && Nang <= 1000) {
							//search for the maximum thetaf,phif on a coarse grid
							pf = 0.;
							for (ithetaf = 0; ithetaf <= Nang; ++ithetaf) {
								thetaftmp = ithetaf * M_PI / Nang;
								for (iphif = -Nang; iphif < Nang; ++iphif) {
									if ((ithetaf == 0 || ithetaf == Nang) && iphif == 1) {
										break;
									}
									phiftmp = iphif * M_PI / Nang;
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
											//	* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
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
									phif -= invHess[1][0] * pfgrad[0] + invHess[1][1] * pfgrad[1];
									thetaf = fmod(thetaf, 2. * M_PI);
									phif = fmod(phif, 2. * M_PI);
								}
								else {
									thetaf += pfgrad[0];
									phif += pfgrad[1];
								}
							}
							if (normgrad >= 1E-15) {
								Nang += 5;
								std::cout << "WARNING:N1=" << N1 << " nf1=" << nf1 << " thetaf2=" << thetaf2 << " nf2=" << nf2 << " normgrad=" << normgrad << " increase Nang to " << Nang << "\n";
							}
						}
						if (normgrad >= 1E-15) {
							std::cout << "WARNING:nq1=" << N1 << " nf1=" << nf1 << " thetaf2=" << thetaf2 << " nf2=" << nf2 << " normgrad=" << normgrad << " final not workink!\n";
						}
						
						//calculate final value of pf
						pf = 0.;
						for (i = 0; i < Nint; ++i) {
							for (j = 0; j < Nint; ++j) {
								for (nba = 0; nba <= nT; ++nba) {
									pf += pt_terms_probDist_weights[i * Nint + j]
										* pt(N, nba, p0b, p1b, thetaf, phif, thetaPoints[i], phiPoints[j]);
								}
							}
						}
						
						pfthetaf2 += pf;
						//std::cout << "nf2=" << std::setw(3) << std::right << nf2;
						//std::cout << " thetaf=" << std::setw(22) << std::left << std::fixed << thetaf;
						//std::cout << " phif=" << std::setw(22) << std::left << phif;
						//std::cout << " pf=" << std::setw(22) << std::left << pf << "\n";
					}
					results2mloc[irloc].N1 = N1;
					results2mloc[irloc].nf1 = nf1;
					results2mloc[irloc].thetaf2 = thetaf2;
					results2mloc[irloc].pf = pfthetaf2;
					std::cout << " N1=" << std::setw(3) << std::right << N1;
					std::cout << " nf1=" << std::setw(3) << std::right << nf1;
					std::cout << " thetaf2=" << std::setw(20) << std::left << thetaf2;
					std::cout << " pf=" << std::setw(20) << std::left << pfthetaf2 << "\n";
#ifdef MPIusage
				}
#endif
			}
		}
	}

#ifdef MPIusage
	//prepare MPI_Gatherv
	if (myrank == 0) {
		//on rank 0 recvcounts and displs is definied
		results2m.resize(Numr);
		recvcounts.resize(numranks, 0);
		displs.resize(numranks);
		for (int ir = 0; ir < Numr; ++ir) {
			++recvcounts[ir % numranks];
		}
		displs[0] = 0;
		for (ir=1; ir<numranks; ++ir){
			displs[ir] = recvcounts[ir-1] + displs[ir-1];
		}
	}

	MPI_Gatherv(results2mloc.data(), irloc+1, MPI_Result2m,
		results2m.data(), recvcounts.data(), displs.data(), MPI_Result2m,
		0, MPI_COMM_WORLD);

	if (myrank == 0) {
#else
	results2m.swap(results2mloc);
#endif
		//std::cout << "\n";

		//order_by_pf(results2m);
		//for (ir = 0; ir < 10; ++ir) {
		//	std::cout << "pf2 = " << results2m[ir].pf << ", N1 = " << results2m[ir].N1 << " thetaf2 = " << results2m[ir].thetaf2 << "\n";
		//}

		file.precision(16);
		file.open(filename, std::ios::out);
		for (ir = 0; ir < Numr; ++ir) {
			file << results2m[ir].N1 << " " << results2m[ir].nf1 << " " << results2m[ir].thetaf2 << " " << results2m[ir].pf << "\n";
		}
		file.close();

		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001 << " s\n";
	
#ifdef MPIusage
    }
	MPI_Finalize();
#endif

	return 0;
}