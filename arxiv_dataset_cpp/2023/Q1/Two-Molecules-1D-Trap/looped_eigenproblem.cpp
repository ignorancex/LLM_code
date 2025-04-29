#include "looped_eigenproblem.h"

void loopedEigenproblem(Matrix& base_hamiltonian, vector<int>& channels, unsigned int QN, vector<string>& params, unsigned int how_many_eigenstates_to_save)
{
	unsigned int basis_size = basisSize(channels, QN);
	Matrix hamiltonian(basis_size, basis_size, 0);
	Matrix looping_hamiltonian(basis_size, basis_size, 0);
	Matrix iso_interaction(basis_size, basis_size, 0), aniso_interaction(basis_size, basis_size, 0), electric_field(basis_size, basis_size, 0), magnetic_field(basis_size, basis_size, 0), spin_rotation(basis_size, basis_size, 0);

	switch (vsWhat) // 0:vsgjj, 1:vsgjk, 2:vself, 3:vsmf, 4:vsA
	{
	case 0:

		if (ONLY_ENERGIES == true)
		{
			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				gjj = -3 + (double)iterator * step;
				params[vsWhat] = doubleToString(gjj);
				iso_interaction = isoInteraction(channels, QN, gjj);
				findEnergiesAndSave(gjj, iso_interaction, iterator, params, hamiltonian, base_hamiltonian, basis_size, how_many_eigenstates_to_save);
			}
		}
		else
		{
			vector <pair <double, int> > set_of_pairs;
			pair <double*, double*> eigenproblem;

			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				gjj = -3 + (double)iterator * step;
				params[vsWhat] = doubleToString(gjj);
				iso_interaction = isoInteraction(channels, QN, gjj);
				hamiltonian = base_hamiltonian + iso_interaction;
				
				// Diagonalize
				eigenproblem = MKLDiagonalization(hamiltonian, basis_size, true); // eigenvalues = eigenproblem.first, eigenvectors = eigenproblem.second

				// Create sorted vector of pairs: eigenenergies + their indices
				set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);

				if (J2 == true)
				{
					vector <double> expected_J = findExpectedJ2(set_of_pairs, eigenproblem, channels, QN);
					saveResults("expectedJ", expected_J, params, iterator, gjj, how_many_eigenstates_to_save);
				}
				else if (MAGNETIZATION == true)
				{
					vector <double> expected_M = findExpectedMg(set_of_pairs, eigenproblem, channels, QN);
					saveResults("magnetization", expected_M, params, iterator, gjj, how_many_eigenstates_to_save);
				}
				else if (PRINT_EIGENVECTORS == true)
				{
					printHeigenvectors(eigenproblem, channels, QN, params, iterator);
				}
				else
				{
					cout << "ERROR! Boolean troubles!" << endl;
					break;
				}

				set_of_pairs.clear();
			}
		}

		break;

	case 1:
		if (ONLY_ENERGIES == true)
		{
			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				gjk = (double)iterator * step;
				params[vsWhat] = doubleToString(gjk);
				aniso_interaction = anisoInteraction(channels, QN, gjk);
				findEnergiesAndSave(gjk, aniso_interaction, iterator, params, hamiltonian, base_hamiltonian, basis_size, how_many_eigenstates_to_save);
			}
		}
		else
		{
			vector <pair <double, int> > set_of_pairs;
			pair <double*, double*> eigenproblem;

			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				gjk = (double)iterator * step;
				params[vsWhat] = doubleToString(gjk);
				aniso_interaction = anisoInteraction(channels, QN, gjk);
				hamiltonian = base_hamiltonian + aniso_interaction;

				// Diagonalize
				eigenproblem = MKLDiagonalization(hamiltonian, basis_size, true); // eigenvalues = eigenproblem.first, eigenvectors = eigenproblem.second

				// Create sorted vector of pairs: eigenenergies + their indices
				set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);

				if (J2 == true)
				{
					vector <double> expected_J = findExpectedJ2(set_of_pairs, eigenproblem, channels, QN);
					saveResults("expectedJ", expected_J, params, iterator, gjk, how_many_eigenstates_to_save);
				}
				else if (MAGNETIZATION == true)
				{
					vector <double> expected_M = findExpectedMg(set_of_pairs, eigenproblem, channels, QN);
					saveResults("magnetization", expected_M, params, iterator, gjk, how_many_eigenstates_to_save);
				}
				else if (PRINT_EIGENVECTORS == true)
				{
					printHeigenvectors(eigenproblem, channels, QN, params, iterator);
				}
				else
				{
					cout << "ERROR! Boolean troubles!" << endl;
					break;
				}

				set_of_pairs.clear();
			}
		}	

		break;

	case 2:
		if (ONLY_ENERGIES == true)
		{
			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				E = (double)iterator * step;
				params[vsWhat] = doubleToString(E);
				electric_field = electricField(channels, QN, E);
				findEnergiesAndSave(E, electric_field, iterator, params, hamiltonian, base_hamiltonian, basis_size, how_many_eigenstates_to_save);
			}
		}
		else
		{
			vector <pair <double, int> > set_of_pairs;
			pair <double*, double*> eigenproblem;

			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				E = (double)iterator * step;
				params[vsWhat] = doubleToString(E);
				electric_field = electricField(channels, QN, E);
				hamiltonian = base_hamiltonian + electric_field;

				// Diagonalize
				eigenproblem = MKLDiagonalization(hamiltonian, basis_size, true); // eigenvalues = eigenproblem.first, eigenvectors = eigenproblem.second

				// Create sorted vector of pairs: eigenenergies + their indices
				set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);

				if (J2 == true)
				{
					vector <double> expected_J = findExpectedJ2(set_of_pairs, eigenproblem, channels, QN);
					saveResults("expectedJ", expected_J, params, iterator, E, how_many_eigenstates_to_save);
				}
				else if (MAGNETIZATION == true)
				{
					vector <double> expected_M = findExpectedMg(set_of_pairs, eigenproblem, channels, QN);
					saveResults("magnetization", expected_M, params, iterator, E, how_many_eigenstates_to_save);
				}
				else if (PRINT_EIGENVECTORS == true)
				{
					printHeigenvectors(eigenproblem, channels, QN, params, iterator);
				}
				else
				{
					cout << "ERROR! Boolean troubles!" << endl;
					break;
				}

				set_of_pairs.clear();
			}
		}

		break;

	case 3:
		if (ONLY_ENERGIES == true)
		{
			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				B = (double)iterator * step;
				params[vsWhat] = doubleToString(B);
				magnetic_field = magneticField(channels, B);
				findEnergiesAndSave(B, magnetic_field, iterator, params, hamiltonian, base_hamiltonian, basis_size, how_many_eigenstates_to_save);
			}
		}
		else
		{
			vector <pair <double, int> > set_of_pairs;
			pair <double*, double*> eigenproblem;

			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				B = (double)iterator * step;
				params[vsWhat] = doubleToString(B);
				magnetic_field = magneticField(channels, B);
				hamiltonian = base_hamiltonian + magnetic_field;

				// Diagonalize
				eigenproblem = MKLDiagonalization(hamiltonian, basis_size, true); // eigenvalues = eigenproblem.first, eigenvectors = eigenproblem.second

				// Create sorted vector of pairs: eigenenergies + their indices
				set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);

				if (J2 == true)
				{
					vector <double> expected_J = findExpectedJ2(set_of_pairs, eigenproblem, channels, QN);
					saveResults("expectedJ", expected_J, params, iterator, B, how_many_eigenstates_to_save);
				}
				else if (MAGNETIZATION == true)
				{
					vector <double> expected_M = findExpectedMg(set_of_pairs, eigenproblem, channels, QN);
					saveResults("magnetization", expected_M, params, iterator, B, how_many_eigenstates_to_save);
				}
				else if (PRINT_EIGENVECTORS == true)
				{
					printHeigenvectors(eigenproblem, channels, QN, params, iterator);
				}
				else
				{
					cout << "ERROR! Boolean troubles!" << endl;
					break;
				}

				set_of_pairs.clear();

			}
		}

		break;

	case 4:
		if (ONLY_ENERGIES == true)
		{
			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				A = (double)iterator * step;
				params[vsWhat] = doubleToString(A);
				spin_rotation = spinRotation(channels, A);
				findEnergiesAndSave(A, spin_rotation, iterator, params, hamiltonian, base_hamiltonian, basis_size, how_many_eigenstates_to_save);
			}
		}
		else
		{
			vector <pair <double, int> > set_of_pairs;
			pair <double*, double*> eigenproblem;

			for (int iterator = START_IT; iterator <= END_IT; iterator++)
			{
				A = (double)iterator * step;
				params[vsWhat] = doubleToString(A);
				spin_rotation = spinRotation(channels, A);
				hamiltonian = base_hamiltonian + spin_rotation;

				// Diagonalize
				eigenproblem = MKLDiagonalization(hamiltonian, basis_size, true); // eigenvalues = eigenproblem.first, eigenvectors = eigenproblem.second

				// Create sorted vector of pairs: eigenenergies + their indices
				set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);

				if (J2 == true)
				{
					vector <double> expected_J = findExpectedJ2(set_of_pairs, eigenproblem, channels, QN);
					saveResults("expectedJ", expected_J, params, iterator, A, how_many_eigenstates_to_save);
				}
				else if (MAGNETIZATION == true)
				{
					vector <double> expected_M = findExpectedMg(set_of_pairs, eigenproblem, channels, QN);
					saveResults("magnetization", expected_M, params, iterator, A, how_many_eigenstates_to_save);
				}
				else if (PRINT_EIGENVECTORS == true)
				{
					printHeigenvectors(eigenproblem, channels, QN, params, iterator);
				}
				else
				{
					cout << "ERROR! Boolean troubles!" << endl;
					break;
				}

				set_of_pairs.clear();

			}
		}

		break;
	}

	// If you need to know any coupling between basis states explicitly...
	/*int i = indexOfSpinBasisState(0, 0, 0, 0, 0, 1, 0, channels, QN);
	int j = indexOfSpinBasisState(0, 1, 1, 1, 0, 1, 0, channels, QN);

	PrintHamiltonianElement(i, j, channels, hamiltonian, QN);*/
}

void findEnergiesAndSave(double value, Matrix& matrix_to_add, int iterator, vector<string>& params, Matrix& hamiltonian, Matrix& base_hamiltonian, unsigned int basis_size, unsigned int how_many_eigenstates_to_save, bool print)
{
	hamiltonian = base_hamiltonian + matrix_to_add;

	pair <double*, double*> eigenproblem;
	eigenproblem = MKLDiagonalization(hamiltonian, basis_size);

	if (print == true)
	{
		printEigenvalues("Eigenvalues", basis_size, eigenproblem.first);
	}

	if (ENERGY_DIFFERENCES == true)
	{
		vector<double> energydiffstoprint, cutenergydiffstoprint;
		for (int j = 0; j < basis_size; j++)
		{
			for (int jprim = 0; jprim < basis_size; jprim++)
			{
				if (j > jprim && eigenproblem.first[j] < 25.0 && eigenproblem.first[jprim] < 25.0)
				{
					energydiffstoprint.push_back(eigenproblem.first[j] - eigenproblem.first[jprim]);
				}
			}
		}
		sort(energydiffstoprint.begin(), energydiffstoprint.end());

		saveResults("eigenvalue_differences", energydiffstoprint, params, iterator, value, energydiffstoprint.size());
	}
	else
	{
		saveResults("eigenvalues", eigenproblem.first, basis_size, params, iterator, value, how_many_eigenstates_to_save);
	}
}

vector <double> findExpectedJ2(vector <pair <double, int> >& set_of_pairs, pair <double*, double*>& eigenproblem, vector<int>& channels, unsigned int QN)
{
	vector <double> expected_J;
	double coefficient, coefficient_J;
	int index;
	unsigned int basis_size = basisSize(channels, QN);

	// Calculate expected J of the eigenvectors using indices from the vector of pairs and write them to expected_J vector
	for (unsigned int i = 0; i < basis_size; i++)
	{
		expected_J.push_back(0);
		index = set_of_pairs[i].second;
		for (unsigned int j = 0; j < basis_size; j++)
		{
			//cout << "We look at the eigenvector with index " << index << endl;
			coefficient = eigenproblem.second[index + j * basis_size];
			coefficient_J = (double)channels[(int64_t)QN * j + 1];
			//cout << "Coefficient " << coefficient << " with Jtot value " << coefficient_J << endl;

			if (coefficient != 0.0 && coefficient_J != 0.0)
			{
				expected_J[i] += coefficient * coefficient * coefficient_J * (coefficient_J + 1);
				//cout << "Result: " << coefficient * coefficient * coefficient_J * (coefficient_J + 1) << endl;
				//cout << "expected_J contains: " << expected_J[i] << endl;
			}
			else
			{
				continue;
			}
		}

		//cout << "J2 value for eigenvector with index " << index << " is: " << expected_J[i] << endl << endl;
	}

	return expected_J;
}

vector <double> findExpectedMg(vector <pair <double, int> >& set_of_pairs, pair <double*, double*>& eigenproblem, vector<int>& channels, unsigned int QN, bool print)
{
	vector <double> expected_M;
	double coefficient, coefficient_M;
	int index;
	unsigned int basis_size = basisSize(channels, QN);

	if (print == true)
	{
		printEigenvalues("Eigenvalues", basis_size, eigenproblem.first);
		printEigenvectors("Eigenvectors", basis_size, eigenproblem.second);
	}

	// Calculate expected magnetization of the eigenvectors using indices from the vector of pairs and write them to expected_M vector
	for (unsigned int i = 0; i < basis_size; i++)
	{
		expected_M.push_back(0);
		index = set_of_pairs[i].second;
		for (unsigned int j = 0; j < basis_size; j++)
		{
			//cout << "We look at the eigenvector with index " << index << endl;
			coefficient = eigenproblem.second[index + j * basis_size];
			coefficient_M = (double) (channels[(int64_t)QN * j + 6]); // quantum numbers: n, J, M_J, m1, m2, S, MS, s1, s2
			//cout << "Coefficient " << coefficient << " with Ms value " << coefficient_M << endl;

			if (coefficient != 0.0 && coefficient_M != 0.0)
			{
				expected_M[i] += coefficient * coefficient * coefficient_M;
				//cout << "Result: " << coefficient * coefficient * coefficient_M << endl;
				//cout << "expected_M contains: " << expected_M[i] << endl;
			}
			else
			{
				continue;
			}
		}

		if (print == true)
		{
			cout << "M value for eigenvector with index " << index << " is: " << expected_M[i] << endl << endl;
		}
	}

	return expected_M;
}

vector < pair <double, int> > pairAndSortEigenenergies(pair <double*, double*>& eigenproblem, unsigned int basis_size)
{
	vector <pair <double, int> > set_of_pairs;
	// Create vector of pairs: eigenenergies + their indices
	for (unsigned int i = 0; i < basis_size; i++)
	{
		set_of_pairs.push_back(make_pair(eigenproblem.first[i], i));
	}

	// Sort the vector of pairs (std::sort does it according to the first element)
	std::sort(set_of_pairs.begin(), set_of_pairs.end());

	return set_of_pairs;
}

void printHeigenvectors(pair <double*, double*>& eigenproblem, vector<int>& channels, unsigned int QN, vector<string>& params, int iterator)
{
	bool squared_coefficients = true;
	unsigned int basis_size = basisSize(channels, QN);

	double coefficient, energy;
	int ile_stanow = 3;
	int index;

	cout << "Here!" << endl;

	string quantity = "qnumbers";

	string filename = createFilename(params, iterator, quantity);

	std::ofstream file_with_results(filename.c_str());

	vector <pair <double, int> > set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);

	file_with_results << "\t" << "n" << "\t" << "J" << "\t" << "M" << "\t" << "j1" << "\t" << "j2" << "\t" << "S" << "\t" << "MS" << "\t" << "s1" << "\t" << "s2" << endl;

	// Find basis states that enter the three eigenvectors with lowest energies with |coefficients|^2 bigger than 0.05.
	for (int state = 0; state < ile_stanow; state++)
	{
		index = set_of_pairs[state].second;
		energy = set_of_pairs[state].first;

		switch (state)
		{
		case 0:
			file_with_results << "Ground state with energy " << energy << ": " << endl;
			break;
		case 1:
			file_with_results << "First excited state with energy " << energy << ": " << endl;
			break;
		case 2:
			file_with_results << "Second excited state with energy " << energy << ": " << endl;
			break;
		default:
			break;
		}

		for (unsigned int i = 0; i < basis_size; i++)
		{
			coefficient = eigenproblem.second[index + i * basis_size];

			if (coefficient * coefficient > 0.05)
			{
				switch (state)
				{
				case 0:
					if (squared_coefficients == false)
					{
						file_with_results << setprecision(4) << coefficient << "\t";
					}
					else
					{
						file_with_results << setprecision(4) << coefficient * coefficient << "\t";
					}
					
					for (unsigned int j = 0; j < QN; j++)
					{
						file_with_results << channels[(int64_t)QN * i + j] << "\t";
					}
					file_with_results << endl;
					break;
				case 1:
					if (squared_coefficients == false)
					{
						file_with_results << setprecision(4) << coefficient << "\t";
					}
					else
					{
						file_with_results << setprecision(4) << coefficient * coefficient << "\t";
					}

					for (unsigned int j = 0; j < QN; j++)
					{
						file_with_results << channels[(int64_t)QN * i + j] << "\t";
					}
					file_with_results << endl;
					break;
				case 2:
					if (squared_coefficients == false)
					{
						file_with_results << setprecision(4) << coefficient << "\t";
					}
					else
					{
						file_with_results << setprecision(4) << coefficient * coefficient << "\t";
					}

					for (unsigned int j = 0; j < QN; j++)
					{
						file_with_results << channels[(int64_t)QN * i + j] << "\t";
					}
					file_with_results << endl;
					break;
				default:
					break;
				}
			}
			else
			{
				continue;
			}
		}
	}

	file_with_results.close();
}

int indexOfSpinBasisState(int n, int J, int M, int j1, int j2, int Stot, int MS, vector<int>& channels, unsigned int QN)
{
	vector<int> basisState = {n, J, M, j1, j2, Stot, MS, S1, S2};
	auto it = search(channels.begin(), channels.end(), basisState.begin(), basisState.end());

	int index = distance(channels.begin(), it++) / QN;
	return index;
}

void printBasisState(int n, int J, int M, int j1, int j2, int Stot, int MS)
{
	cout << "|" << n << "," << J << "," << M << "," << j1 << "," << j2;
	
	if (Stot != 5 && MS != 5)
	{
		cout << "," << Stot << "," << MS << "," << S1 << "," << S2 << ">";  
	}
	else
	{
		cout << ">";
	}
		
}

void printBasisState(int index, vector<int>& channels, unsigned int QN)
{
	int n = channels[(int64_t)QN * (int64_t)index + 0];
	int J = channels[(int64_t)QN * (int64_t)index + 1];
	int M = channels[(int64_t)QN * (int64_t)index + 2];
	int j1 = channels[(int64_t)QN * (int64_t)index + 3];
	int j2 = channels[(int64_t)QN * (int64_t)index + 4];

	cout << "|" << n << "," << J << "," << M << "," << j1 << "," << j2;

	if (QN == QN_withspin)
	{
		int Stot = channels[(int64_t)QN * index + 5];
		int MS = channels[(int64_t)QN * index + 6];

		cout << "," << Stot << "," << MS << "," << S1 << "," << S2 << ">";
	}
	else
	{
		cout << ">";
	}
}

void printHamiltonianElement(int i, int j, vector<int>& channels, Matrix& hamiltonian, unsigned int QN)
{
	cout << "The coupling between the states ";
	printBasisState(i, channels, QN);
	cout << " and ";
	printBasisState(j, channels, QN);
	cout << " is: " << hamiltonian(i, j) << endl;
}

void printChosenEigenvector(int state, pair <double*, double*>& eigenproblem, vector<int>& channels, unsigned int QN, bool sorted, std::ostream& ostr)
{
	bool squared_coefficients = false;
	unsigned int basis_size = basisSize(channels, QN);

	double coefficient, energy;
	int index;
	vector <pair <double, int> > set_of_pairs;

	if (sorted == true)
	{
		set_of_pairs = pairAndSortEigenenergies(eigenproblem, basis_size);
	}
	else
	{
		// Create vector of pairs: eigenenergies + their indices
		for (unsigned int i = 0; i < basis_size; i++)
		{
			set_of_pairs.push_back(make_pair(eigenproblem.first[i], i));
		}
	}

	if (QN == QN_nospin)
	{
		ostr << "\t" << "n" << "\t" << "J" << "\t" << "M" << "\t" << "j1" << "\t" << "j2" << endl;
	}
	else
	{
		ostr << "\t" << "n" << "\t" << "J" << "\t" << "M" << "\t" << "j1" << "\t" << "j2" << "\t" << "S" << "\t" << "MS" << "\t" << "s1" << "\t" << "s2" << endl;
	}

	// Find basis states that enter the three eigenvectors with lowest energies with |coefficients|^2 bigger than 0.05.
	index = set_of_pairs[state].second;
	energy = set_of_pairs[state].first;

	ostr << "The state " << state << " energy is: " << energy << ": " << endl;

	for (unsigned int i = 0; i < basis_size; i++)
	{
		coefficient = eigenproblem.second[index + i * basis_size];

		if (coefficient * coefficient > 0.01)
		{				
			if (squared_coefficients == false)
			{
				ostr << setprecision(4) << coefficient << "\t";
			}
			else
			{
				ostr << setprecision(4) << coefficient * coefficient << "\t";
			}

			for (unsigned int j = 0; j < QN; j++)
			{
				ostr << channels[(int64_t)QN * i + j] << "\t";
			}
			ostr << endl;
		}
		else
		{
			continue;
		}
	}
	ostr << endl;
}