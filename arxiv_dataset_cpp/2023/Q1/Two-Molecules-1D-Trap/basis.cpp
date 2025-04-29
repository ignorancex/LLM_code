#include "basis.h"

// remember, you set the default parameter only in declaration in .h, not in definitions!
void generateChannelsNoFieldNoSpin(vector<int>& channels, int Jtot, bool print) // you have to specify the default values for the arguments only in the declaration but not in the definition
{
	// fun fact - how big the whole basis set would be?
	int sum = 0;
	for (int i = 0; i <= JMAX; i++)
	{
		sum += 2 * i + 1;
	}
	unsigned long int simplebasesize = sum * sum * (NMAX + 1);

	// generate channels, with Mtot = 0
	for (int n = 0; n <= NMAX; n = n + 2)
	{
		for (int j1 = 0; j1 <= JMAX; j1++)
		{
			for (int j2 = 0; j2 <= JMAX; j2++)
			{
				if ((abs(j1 - j2) > Jtot) || (j1 + j2) < Jtot)
				{
					continue;
				}
				else
				{
					channels.push_back(n);
					channels.push_back(Jtot);
					channels.push_back(0); // (zdegenerowane Mtoty)
					channels.push_back(j1);
					channels.push_back(j2);
				}
			}
		}
	}

	// more fun facts!
	cout << "Basis size: " << channels.size() / QN_nospin << endl;
	cout << "Simple basis size: " << simplebasesize / QN_nospin << endl;

	// print out the channels. You don't want to do that usually!
	if (print == true)
	{
		int j = 1;
		cout << "it\tn\tJ\tM\tj1\tj2" << endl;
		for (int i = 0; i < channels.size(); i++)
		{
			if ((i + 1) % QN_nospin == 1)
			{
				cout << j << "\t";
				j++;
			}

			cout << channels[i] << '\t';

			if ((i + 1) % QN_nospin == 0)
			{
				cout << endl;
			}
		}
		cout << endl << endl;
	}
}

void generateChannelsWithFieldNoSpin(vector<int>& channels, int M_J, bool print) // you have to specify the default values for the arguments only in the declaration but not in the definition
{
	// fun fact - how big the whole basis set would be?
	int sum = 0;
	for (int i = 0; i <= JMAX; i++)
	{
		sum += 2 * i + 1;
	}
	unsigned long int simplebasesize = sum * sum * (NMAX + 1);

	// generate channels, for a chosen Mtot
	for (int n = 0; n <= NMAX; n = n + 2)
	{
		for (int Jtot = abs(M_J); Jtot <= 2 * JMAX; Jtot++)
		{
			for (int j1 = 0; j1 <= JMAX; j1++)
			{
				for (int j2 = 0; j2 <= JMAX; j2++)
				{
					if ((abs(j1 - j2) > Jtot) || (j1 + j2) < Jtot)
					{
						continue;
					}
					else
					{
						channels.push_back(n);
						channels.push_back(Jtot);
						channels.push_back(M_J);
						channels.push_back(j1);
						channels.push_back(j2);
					}
				}
			}
		}
	}

	// more fun facts!
	cout << "Basis size: " << channels.size() / QN_nospin << endl;
	cout << "Simple basis size: " << simplebasesize / QN_nospin << endl;

	// print out the channels. You don't want to do that usually!
	if (print == true)
	{
		int j = 1;
		cout << "it\tn\tJ\tM\tj1\tj2" << endl;
		for (int i = 0; i < channels.size(); i++)
		{
			if ((i + 1) % QN_nospin == 1)
			{
				cout << j << "\t";
				j++;
			}

			cout << channels[i] << '\t';

			if ((i + 1) % QN_nospin == 0)
			{
				cout << endl;
			}
		}
		cout << endl << endl;
	}
}

void generateChannelsWithFieldWithSpin(vector<int>& channels, int Mtot, bool print) // you have to specify the default values for the arguments only in the declaration but not in the definition
{
	// generate channels, for a chosen Mtot = M_J + ms1 + ms2
	for (int n = 0; n <= NMAX; n = n + 2)
	{
		for (int Jtot = 0; Jtot <= 2 * JMAX; Jtot++)
		{
			for (int j1 = 0; j1 <= JMAX; j1++)
			{
				for (int j2 = 0; j2 <= JMAX; j2++)
				{
					if ((abs(j1 - j2) > Jtot) || (j1 + j2) < Jtot)
					{
						continue;
					}
					else
					{
						for (int Mtot_J = -Jtot; Mtot_J <= Jtot; Mtot_J++)
						{
							for (int Stot = 0; Stot <= S1; Stot++)
							{
								for (int M_S = -Stot; M_S <= Stot; M_S++)
								{
									if (Mtot_J + M_S == Mtot)
									{
										channels.push_back(n);
										channels.push_back(Jtot);
										channels.push_back(Mtot_J);
										channels.push_back(j1);
										channels.push_back(j2);
										channels.push_back(Stot);
										channels.push_back(M_S);
										channels.push_back(S1);
										channels.push_back(S2);
									}
									else
									{
										continue;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// more fun facts!
	cout << "Basis size: " << channels.size() / QN_withspin << endl;

	// print out the channels. You don't want to do that usually!
	if (print == true)
	{
		int j = 1;
		cout << "it\tn\tJ\tM\tj1\tj2\tS\tMS\ts1\ts2" << endl;
		for (int i = 0; i < channels.size(); i++)
		{
			if ((i + 1) % QN_withspin == 1)
			{
				cout << j << "\t";
				j++;
			}

			cout << channels[i] << '\t';

			if ((i + 1) % QN_withspin == 0)
			{
				cout << endl;
			}
		}
		cout << endl << endl;
	}
}

unsigned int basisSize(vector<int>& channels, unsigned int QN)
{
	unsigned int basesize = channels.size() / QN;
	return basesize;
}