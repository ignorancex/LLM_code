#include "ham_functions.h"

long double factorial(unsigned int liczba, unsigned int m)
{
	if (liczba == 0)
	{
		return 1;
	}

	long double result = 1;

	for (int i = liczba; i >= 1; i -= m)
	{
		result = result * (double) i;
	}

	return result;
}

double Hermite0(int n)
{
	if (n == 0)
	{
		return 1;
	}

	if (n % 2 != 0)
	{
		return 0;
	}

	double result = pow(-1, n / 2) * pow(2, n / 2) * factorial(n - 1, 2);

	return result;
}

long double ClebschGordanCoefficient(int j1, int m1, int j2, int m2, int Jtot, int Mtot)
{
	long double result, term;
	long double sum = 0.0;

	if ((Jtot < abs(j1 - j2)) || (Jtot > (j1 + j2)) || (abs(m1) > j1) || (abs(m2) > j2) || (abs(Mtot) > Jtot) || ((m1 + m2) != Mtot))
	{
		result = 0;
	}
	else
	{
		result = sqrt((double)2 * Jtot + 1) * sqrt((double)pow(factorial(j1 + j2 + Jtot + 1, 1), -1));
		result = result * sqrt((double)factorial(j1 + j2 - Jtot, 1) * factorial(Jtot - j1 + j2, 1) * factorial(Jtot + j1 - j2, 1));
		result = result * sqrt((double)factorial(j1 + m1, 1) * factorial(j1 - m1, 1) * factorial(j2 + m2, 1) * factorial(j2 - m2, 1) * factorial(Jtot + Mtot, 1) * factorial(Jtot - Mtot, 1));

		for (int k = 0; k < 100; k++)
		{
			int fact1, fact2, fact3, fact4, fact5;
			fact1 = j1 + j2 - Jtot - k;
			fact2 = Jtot - j1 - m2 + k;
			fact3 = Jtot - j2 + m1 + k;
			fact4 = j1 - m1 - k;
			fact5 = j2 + m2 - k;

			if (fact1 < 0 || fact2 < 0 || fact3 < 0 || fact4 < 0 || fact5 < 0)
			{
				continue;
			}
			else
			{
				term = factorial(fact1, 1) * factorial(fact2, 1) * factorial(fact3, 1) * factorial(fact4, 1) * factorial(fact5, 1) * factorial(k, 1);
				if (k % 2 == 1)
				{
					term = -term;
				}

				sum = sum + pow(term, -1);
			}
		}

		result = result * sum;
	}

	return result;
}

long double SpinClebschGordanCoefficient(double j1, double m1, double j2, double m2, int Jtot, int Mtot)
{
	long double result, term;
	long double sum = 0.0;

	if ((Jtot < abs(j1 - j2)) || (Jtot > (j1 + j2)) || (abs(m1) > j1) || (abs(m2) > j2) || (abs(Mtot) > Jtot) || ((m1 + m2) != Mtot))
	{
		result = 0;
	}
	else
	{
		result = sqrt((double)2 * Jtot + 1) * sqrt((double)pow(factorial(j1 + j2 + Jtot + 1, 1), -1));
		result = result * sqrt((double)factorial(j1 + j2 - Jtot, 1) * factorial(Jtot - j1 + j2, 1) * factorial(Jtot + j1 - j2, 1));
		result = result * sqrt((double)factorial(j1 + m1, 1) * factorial(j1 - m1, 1) * factorial(j2 + m2, 1) * factorial(j2 - m2, 1) * factorial(Jtot + Mtot, 1) * factorial(Jtot - Mtot, 1));

		for (int k = 0; k < 100; k++)
		{
			int fact1, fact2, fact3, fact4, fact5;
			fact1 = j1 + j2 - Jtot - k;
			fact2 = Jtot - j1 - m2 + k;
			fact3 = Jtot - j2 + m1 + k;
			fact4 = j1 - m1 - k;
			fact5 = j2 + m2 - k;

			if (fact1 < 0 || fact2 < 0 || fact3 < 0 || fact4 < 0 || fact5 < 0)
			{
				continue;
			}
			else
			{
				term = factorial(fact1, 1) * factorial(fact2, 1) * factorial(fact3, 1) * factorial(fact4, 1) * factorial(fact5, 1) * factorial(k, 1);
				if (k % 2 == 1)
				{
					term = -term;
				}

				sum = sum + pow(term, -1);
			}
		}

		result = result * sum;
	}

	return result;
}

double diagonalFunction(int n, int j1, int j2)
{
	double result;
	result = n + 0.5 + B1 * (double) j1 * ((double) j1 + 1) + B2 * j2 * ((double) j2 + 1);
	return result;
}

double electricFunction(double mu, double el_field, int j, int jprim, int m)
{
	double result;
	result = -mu * el_field * sqrt( (2 * (double) j + 1) / (2 * (double) jprim + 1) ) * ClebschGordanCoefficient(j, 0, 1, 0, jprim, 0) * ClebschGordanCoefficient(j, m, 1, 0, jprim, m);
	return result;
}

double magneticFunction(double mg_field, double sum_of_mss)
{
	return gs * sum_of_mss * mg_field;
}

double deltaFunction(double g, int n1, int n2)
{
	double result;
	result = pow(2, -0.5) * g * pow(2, -0.5 * n1) * pow(2, -0.5 * n2) * pow(factorial(n1, 1), -0.5) * pow(factorial(n2, 1), -0.5) * pow(PI, -0.5) * Hermite0(n1) * Hermite0(n2);
	return result;
}

double spinRotationDiagonalFunction(double spinrot_strength, int m1, int m2, double ms1, double ms2)
{
	return spinrot_strength * (m1 * ms1 + m2 * ms2);
}

bool differByNOnly(vector<int>& channels, int QN, int i, int j) // for isoInteraction
{
	return equal(channels.begin() + i * (int64_t)QN + 1, channels.begin() + (int64_t)i * (int64_t)QN + QN, channels.begin() + (int64_t)j * (int64_t)QN + 1);
}

bool differByNAndjsOnly(vector<int>& channels, int QN, int i, int j) // for anisoInteraction
{
	bool resultJM, resultspin;

	resultJM = equal(channels.begin() + (int64_t)i * (int64_t)QN + 1, channels.begin() + (int64_t)i * (int64_t)QN + 3, channels.begin() + (int64_t)j * (int64_t)QN + 1); // n J M j1 j2 | S MS s1 s2
	if (QN == QN_withspin)
	{
		resultspin = equal(channels.begin() + (int64_t)i * (int64_t)QN + 5, channels.begin() + (int64_t)i * (int64_t)QN + 9, channels.begin() + (int64_t)j * (int64_t)QN + 5); // n J M j1 j2 | S MS s1 s2
	}
	else
		resultspin = true;
	return resultJM * resultspin;
}

bool jsExchangeByOne(vector<int>& channels, int QN, int i, int j) // for anisoInteraction
{
	int j1 = channels[(int64_t)QN * (int64_t)i + 3];
	int j2 = channels[(int64_t)QN * (int64_t)i + 4];
	int j1prim = channels[(int64_t)QN * (int64_t)j + 3];
	int j2prim = channels[(int64_t)QN * (int64_t)j + 4];

	if (j1 != j1prim && j2 != j2prim && abs(j1 - j1prim) == 1)
	{
		return true;
	}
	else
		return false;
}

bool sameNAndSpin(vector<int>& channels, int QN, int i, int j) // for electricField
{
	bool resultN, resultspin;

	resultN = equal(channels.begin() + (int64_t)i * (int64_t)QN, channels.begin() + (int64_t)i * (int64_t)QN + 1, channels.begin() + (int64_t)j * (int64_t)QN); // n J M j1 j2 | S MS s1 s2
	if (QN == QN_withspin)
	{
		resultspin = equal(channels.begin() + (int64_t)i * (int64_t)QN + 5, channels.begin() + (int64_t)i * (int64_t)QN + 9, channels.begin() + (int64_t)j * (int64_t)QN + 5); // n J M j1 j2 | S MS s1 s2
	}
	else
		resultspin = true;
	return resultN * resultspin;
}

bool onejDifferByOne(vector<int>& channels, int QN, int i, int j) // for electricField
{
	int j1 = channels[(int64_t)QN * (int64_t)i + 3];
	int j2 = channels[(int64_t)QN * (int64_t)i + 4];
	int j1prim = channels[(int64_t)QN * (int64_t)j + 3];
	int j2prim = channels[(int64_t)QN * (int64_t)j + 4];

	if ((j1 == j1prim && abs(j2 - j2prim) == 1) || (abs(j1 - j1prim) == 1 && j2 == j2prim))
	{
		return true;
	}
	else
		return false;
}

bool differByJMjSAndMsOnly(vector<int>& channels, int i, int j) // for spinRotation
{
	bool resultN, resultjs;
	int QN = QN_withspin;

	resultN = equal(channels.begin() + (int64_t)i * (int64_t)QN, channels.begin() + (int64_t)i * (int64_t)QN + 1, channels.begin() + (int64_t)j * (int64_t)QN); // n J M j1 j2 | S MS s1 s2
	resultjs = equal(channels.begin() + (int64_t)i * (int64_t)QN + 3, channels.begin() + (int64_t)i * (int64_t)QN + 5, channels.begin() + (int64_t)j * (int64_t)QN + 3); // n J M j1 j2 | S MS s1 s2

	return resultN * resultjs;
}

bool msAndmssOffDiagonalSpinRotation(int m1, int m1prim, int m2, int m2prim, double ms1, double ms1prim, double ms2, double ms2prim) // for spinRotation
{
	if (ms1 == ms1prim + 1 && ms2 == ms2prim && m1 == m1prim - 1 && m2 == m2prim)
		return true;
	else if (ms1 == ms1prim && ms2 == ms2prim + 1 && m1 == m1prim && m2 == m2prim - 1)
		return true;
	else if (ms1 == ms1prim - 1 && ms2 == ms2prim && m1 == m1prim + 1 && m2 == m2prim)
		return true;
	else if (ms1 == ms1prim && ms2 == ms2prim - 1 && m1 == m1prim && m2 == m2prim + 1)
		return true;
	else return false;		
}

bool nonZeroCG(int m1, int m2, int M, int m1prim, int m2prim, int Mprim)
{
	if (m1 + m2 == M && m1prim + m2prim == Mprim)
		return true;
	else
		return false;
}

bool nonZeroSpinCG(double m1, double m2, double M, double m1prim, double m2prim, double Mprim)
{
	if (m1 + m2 == M && m1prim + m2prim == Mprim)
		return true;
	else
		return false;
}

bool momentumConserved(vector<int>& channels, int QN, int i, int j) // for electricField and anisoInteraction
{
	int sumJ = channels[(int64_t)QN * (int64_t)i + 3] + channels[(int64_t)QN * (int64_t)i + 4];
	int sumJprim = channels[(int64_t)QN * (int64_t)j + 3] + channels[(int64_t)QN * (int64_t)j + 4];

	if (sumJ == sumJprim)
	{
		return true;
	}
	else
		return false;
}