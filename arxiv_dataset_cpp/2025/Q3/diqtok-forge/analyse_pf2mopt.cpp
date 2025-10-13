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


struct Result2m {
	int N1;
	int nf1;
	double thetaf2;
	double pf;
};

void partition_N1_nf1(std::vector<Result2m>& res, int const N1, int const nf1, int const start) {
	std::partition(std::begin(res) + start, std::end(res), [N1, nf1](Result2m const& r) {
		return r.N1 == N1 && r.nf1 == nf1;
		});
}

void sort_part_by_pf(std::vector<Result2m>& res, int const start, int const end) {
	std::sort(std::begin(res) + start, std::begin(res) + end, [](const Result2m& a, const Result2m& b)
		{
			return a.pf > b.pf;
		});
}


int main(int argc, char** argv)
{
	int Nint = 30;
	int N = 30;
	int nT = 5;
	double p0b{ 0.029 }, p1b{ 0.927 };
	double p0f{ 0.029 }, p1f{ 0.927 };
	double pf;
	int nf1;
	int N1, Numthetaf2, start, N1opt;
	double pfopt;
	std::fstream file;
	std::vector<Result2m> results2m;
	int ithetaf2, ir, Numr, Nscan;
	int thetaf2Start{ 1 }, thetaf2End{ 99 }, thetaf2N{ 100 };
	std::string str, filename;

	auto t1 = std::chrono::high_resolution_clock::now();

	std::cout.precision(16);
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

	Numthetaf2 = thetaf2End - thetaf2Start + 1;

	Numr = 0;
	for (N1 = 1; N1 < N; ++N1) {
		for (nf1 = 0; nf1 <= N1; ++nf1) {
			for (ithetaf2 = thetaf2Start; ithetaf2 <= thetaf2End; ++ithetaf2) {
				++Numr;
			}
		}
	}
	results2m.resize(Numr);

	
	file.open(filename, std::ios::in);
	for (ir = 0; ir < Numr; ++ir) {
		file >> results2m[ir].N1 >> results2m[ir].nf1 >> results2m[ir].thetaf2 >> results2m[ir].pf;
		if (!file.good()) {
			std::cerr << "Error during read " << ir << "from file " << filename << "\n";
			return 1;
		}
	}
	file.close();

	start = 0;
	for (N1 = 1; N1 < N; ++N1) {
		for (nf1 = 0; nf1 <= N1; ++nf1) {
			if (start < Numr) {
				partition_N1_nf1(results2m, N1, nf1, start);
				sort_part_by_pf(results2m, start, start + Numthetaf2);
				start += Numthetaf2;
			}
		}
	}

	std::cout << "\n";

	start = 0;
	N1opt = 0;
	pfopt = 0.;
	for (N1 = 1; N1 < N; ++N1) {
		pf = 0.;
		for (nf1 = 0; nf1 <= N1; ++nf1) {
			pf += results2m[start].pf;
			ir = start;
			//if (N1 == 16) {
				//std::cout << results2m[ir].N1 << " " << results2m[ir].nf1 << " " << results2m[ir].thetaf2 / M_PI * thetaf2N << " " << results2m[ir].pf << "\n";
			//}
			start += Numthetaf2;
		}
		std::cout << "N1= " << std::setw(3) << std::right << N1 << " pf= " << std::setw(22) << std::left << pf << "\n";
		if (pf > pfopt) {
			pfopt = pf;
			N1opt = N1;
		}
	}

	std::cout << "\n";
	std::cout << "opt N1= " << std::setw(3) << std::right << N1opt << " pf= " << std::setw(20) << std::left << pfopt << "\n";
	std::cout << "\n";

	start = 0;
	for (N1 = 1; N1 < N; ++N1) {
		pf = 0.;
		for (nf1 = 0; nf1 <= N1; ++nf1) {
			pf += results2m[start].pf;
			ir = start;
			if (N1 == N1opt) {
				std::cout << "nf1= " << std::setw(3) << std::right << results2m[ir].nf1;
				std::cout << " thetaf2= " << std::setw(20) << std::left << results2m[ir].thetaf2;
				std::cout << " pf= " << std::setw(20) << std::left << results2m[ir].pf << "\n";
			}
			start += Numthetaf2;
		}
	}
	
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001 << " s\n";

	return 0;
}