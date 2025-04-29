#include "quench_functions.h"

double r2MeanValue(int n, int nprim)
{
	double result = 0;
	double nconstant, power2;

	power2 = ((double)n + (double)nprim) / 2;
	nconstant = sqrt(factorial(n, 1) * factorial(nprim, 1)) * pow(2, -power2) * pow(PI, -1 / 2);

	// very ugly sums over lots of stuff
	for (int N = 0; N <= (int)std::min(n, nprim); N++)
	{
		for (int k = 0; k <= (int)floor(power2 - N); k++)
		{
			result += pow(2, N) * factorial(n + nprim - 2 * N, 1) / (factorial(n - N, 1) * factorial(nprim - N, 1) * factorial(N, 1)) * pow(-1, k) * pow(2, n + nprim - 2 * N - 2 * k - 1) * tgamma( ( (double)n + (double)nprim - 2 * (double)N + 3) / 2 - (double) k) / (factorial(k, 1) * factorial(n + nprim - 2 * N - 2 * k, 1));
		}
	}
	return nconstant * result;
}

double j_r2_jprim(int j, int jprim, pair <double*, double*>& eigenproblem, vector<int>& channels, int QN)
{
	double result = 0;
	unsigned int basis_size = basisSize(channels, QN);
	int n, nprim, J, Jprim, M, Mprim, j1, j2, j1prim, j2prim;

	for (unsigned int i = 0; i < basis_size; i++) // going over channels alpha
	{
		for (unsigned int iprim = 0; iprim < basis_size; iprim++) // going over channels alpha'
		{
			// read quantum numbers of the channel i and iprim
			J = channels[(int64_t)QN * i + 1];
			Jprim = channels[(int64_t)QN * iprim + 1];
			M = channels[(int64_t)QN * i + 2];
			Mprim = channels[(int64_t)QN * iprim + 2];
			j1 = channels[(int64_t)QN * i + 3];
			j2 = channels[(int64_t)QN * i + 4];
			j1prim = channels[(int64_t)QN * iprim + 3];
			j2prim = channels[(int64_t)QN * iprim + 4];

			if (J == Jprim && M == Mprim && j1 == j1prim && j2 == j2prim)
			{
				n = channels[(int64_t)QN * i];
				nprim = channels[(int64_t)QN * iprim];

				result += r2MeanValue(n, nprim) * eigenproblem.second[j + i * basis_size] * eigenproblem.second[jprim + iprim * basis_size];
			}
			else
			{
				continue;
			}
		}
	}
	return result;
}

double j_j2_jprim(int j, int jprim, pair <double*, double*>& eigenproblem, vector<int>& channels, int QN)
{
	double result = 0;
	unsigned int basis_size = basisSize(channels, QN);
	int j1;

	for (unsigned int i = 0; i < basis_size; i++) // going over channels alpha
	// j1 is totally diagonal, so no double loop is needed
	{
		// read quantum numbers of the channel i and iprim
		j1 = channels[(int64_t)QN * i + 3];

		result += (double)j1 * ((double)j1 + 1) * eigenproblem.second[j + i * basis_size] * eigenproblem.second[jprim + i * basis_size];
	}

	return result;
}

double j_Jtot2_jprim(int j, int jprim, pair <double*, double*>& eigenproblem, vector<int>& channels, int QN)
{
	double result = 0;
	unsigned int basis_size = basisSize(channels, QN);
	int J;

	for (unsigned int i = 0; i < basis_size; i++) // going over channels alpha
	// j1 is totally diagonal, so no double loop is needed
	{
		// read quantum numbers of the channel i and iprim
		J = channels[(int64_t)QN * i + 1];

		result += (double)J * ((double)J + 1) * eigenproblem.second[j + i * basis_size] * eigenproblem.second[jprim + i * basis_size];
	}

	return result;
}

double j_Mg_jprim(int j, int jprim, pair <double*, double*>& eigenproblem, vector<int>& channels, int QN)
{
	double result = 0;
	unsigned int basis_size = basisSize(channels, QN);
	int M;

	for (unsigned int i = 0; i < basis_size; i++) // going over channels alpha
	// Ms is totally diagonal, so no double loop is needed
	{
		// read quantum numbers of the channel i and iprim
		M = channels[(int64_t)QN * i + 6];

		result += (double)M * eigenproblem.second[j + i * basis_size] * eigenproblem.second[jprim + i * basis_size];
	}

	return result;
}

double j_Observable_jprim(int j, int jprim, pair <double*, double*>& eigenproblem, vector<int>& channels, int QN)
{
	if (QUENCH_R2 == true)
	{
		return j_r2_jprim(j, jprim, eigenproblem, channels, QN);
	}
	else if (QUENCH_J == true)
	{
		return j_j2_jprim(j, jprim, eigenproblem, channels, QN);
	}
	else if (QUENCH_JTOT == true)
	{
		return j_Jtot2_jprim(j, jprim, eigenproblem, channels, QN);
	}
	else if (QUENCH_MG == true)
	{
		return j_Mg_jprim(j, jprim, eigenproblem, channels, QN);
	}
	else
	{
		std::cout << "Error! Quench of whaat?" << std::endl;
		return 0;
	}
}

bool indexOccursInVector(unsigned int index, vector<unsigned int>& vector_of_indices)
{
	for (int i = 0; i < vector_of_indices.size(); i++)
	{
		if (index == vector_of_indices[i])
		{
			return true;
		}
		else
		{
			continue;
		}
	}

	return false;
}