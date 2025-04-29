#include "quench_dynamics.h"

vector <double> chosenInitialStateBasisCoefficients(Matrix& hamiltonian_initial, vector<int>& channels, unsigned int QN, int which_state)
{
	unsigned int basis_size = basisSize(channels, QN);
	std::pair <double*, double*> eigenproblem_initial = MKLDiagonalization(hamiltonian_initial, basis_size, true); // first: eigenvalues, second: eigenvectors or dummy array
	printChosenEigenvector(0, eigenproblem_initial, channels, QN);
	//// save the coefficients C_{phi,alpha} of the ground state of the H_0 in the basis alpha

	// create sorted vector of pairs: eigenenergies + their indices
	vector <pair <double, int> > set_of_pairs_initial = pairAndSortEigenenergies(eigenproblem_initial, basis_size);

	int index = set_of_pairs_initial[which_state].second; // we're interested in the ground state for now

	vector<double> chosen_state_coefficients(basis_size);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		chosen_state_coefficients[i] = eigenproblem_initial.second[index + i * basis_size];
	}
	cout << "Coefficients C_{phi,alpha} of the ground state of the H_0 in the basis alpha are noted." << endl;

	set_of_pairs_initial.clear();
	return chosen_state_coefficients;
}

vector <double> calculatePhiPsiJOverlaps(pair <double*, double*>& eigenproblem_final, vector<double>& chosen_initial_state_coefficients, unsigned int basis_size, vector<string>& string_params, vector<string>& string_quenched_params, bool save_overlaps)
{
	vector <pair <double, int> > set_of_pairs_final, overlap_sorting_indices;
	vector <pair <int, int> > reverse_state_indices;
	vector<double> phi_j_overlap(basis_size);

	set_of_pairs_final = pairAndSortEigenenergies(eigenproblem_final, basis_size);

	// Create vector of pairs: indices of sorted eigenvectors + new indices that will sort reversely the eigenstates
	// Namely: psi_j -> to what sorted eigenfunction j corresponds?
	reverse_state_indices = pairAndSort(set_of_pairs_final); // set_of_pairs_final.second

	for (unsigned int j = 0; j < basis_size; j++) // iterating over phi_j
	{
		for (unsigned int i = 0; i < basis_size; i++) // iterating over basis states
		{
			phi_j_overlap[j] += eigenproblem_final.second[j + i * basis_size] * chosen_initial_state_coefficients[i];
		}
	}

	if (save_overlaps == true)
	{
		// Create vector of pairs: overlaps of phi with psi_j's + indices that sort them
		for (unsigned int i = 0; i < basis_size; i++)
		{
			overlap_sorting_indices.push_back(make_pair(phi_j_overlap[i], i));
		}

		// Sort the vector of pairs (std::sort does it according to the first element)
		sort(overlap_sorting_indices.begin(), overlap_sorting_indices.end());

		// save to file
		string filename_overlap = createQuenchFilename(string_params, string_quenched_params, "quench_overlap");

		std::ofstream overlap(filename_overlap.c_str());

		for (unsigned int i = 0; i < basis_size; i++)
		{
			overlap << setprecision(6) << "(" << reverse_state_indices[overlap_sorting_indices[i].second].second << "): " << overlap_sorting_indices[i].first << "\t";
		}

		overlap.close();
	}

	set_of_pairs_final.clear();
	return phi_j_overlap;

}

vector <double> calculateMatrixOfPsiOPsiOverlaps(pair <double*, double*>& eigenproblem_final, vector<int>& channels, unsigned int QN)
{
	unsigned int basis_size = basisSize(channels, QN);
	///// Calculate all <psi_j|O|psi_j'> and save for further use
	// I will use only j=j' and j < j', so half of the matrix can stay empty
	vector<double> meanvalues((int64_t)basis_size * basis_size);

	for (unsigned int j = 0; j < basis_size; j++)
	{
		for (unsigned int jprim = 0; jprim < basis_size; jprim++)
		{
			if (j <= jprim)
			{
				meanvalues[j * (int64_t)basis_size + jprim] = j_Observable_jprim(j, jprim, eigenproblem_final, channels, QN);
			}
			else
			{
				continue;
			}
		}

	}

	cout << "The matrix of Phi and all Psi_js overlaps is calculated and saved as well." << endl;
	return meanvalues;
}

vector<unsigned int> findConvergedPsiJIndices(pair<double*, double*>& eigenproblem_final, vector<int>& channels, unsigned int QN, bool print)
{
	vector<unsigned int> indices;
	unsigned int basis_size = basisSize(channels, QN);

	double coefficient;
	int n;

	if (print == true)
	{
		cout << "Indices of unconverged eigenstates are: ";
	}

	for (unsigned int j = 0; j < basis_size; j++) // iterating over eigenstates
	{
		coefficient = 0;

		for (unsigned int i = 0; i < basis_size; i++) // iterating over basis states
		{
			n = channels[(int64_t)QN * i];

			if (n == NMAX)
			{
				coefficient += pow(eigenproblem_final.second[j + i * basis_size], 2);
			}
		}

		if (coefficient < 0.1)
		{
			indices.push_back(j);

		}
		else
		{
			if (print == true)
			{
				cout << j << "(E=" << eigenproblem_final.first[j] << ") , ";
			}
			else
			{
				continue;
			}
		}
	}

	if (print == true)
	{
		cout << endl;
	}

	return indices;
}

void quenchDynamics(Matrix& hamiltonian_final, vector<int>& channels, unsigned int QN, vector<double>& chosen_initial_state_coefficients, vector<std::string>& string_params, vector<std::string>& string_quenched_params, bool save_couplings)
{
	pair <double*, double*> eigenproblem_final;
	unsigned int basis_size = basisSize(channels, QN);
	vector<double> phi_j_overlap(basis_size);
	vector<double> meanvalues((int64_t)basis_size * basis_size);
	vector<unsigned int> converged_psij_indices;
	unsigned int j, jprim;
	double difference;

	eigenproblem_final = MKLDiagonalization(hamiltonian_final, basis_size, true); // first: eigenvalues, second: eigenvectors or dummy array

	// Calculate all C_{phi,j} = <phi|psi_j>, where psi_j are all H_tot eigenfunctions (having all C_{phi,alpha} and C_{j,alpha})
	phi_j_overlap = calculatePhiPsiJOverlaps(eigenproblem_final, chosen_initial_state_coefficients, basis_size, string_params, string_quenched_params, false);

	// Calculate all <psi_j|O|psi_j'> and save for further use
	meanvalues = calculateMatrixOfPsiOPsiOverlaps(eigenproblem_final, channels, QN);

	// Find converged psi_js, i.e., eigenstates without the ones with the sum of squared contributions from basis states with n = nmax bigger than 20%
	converged_psij_indices = findConvergedPsiJIndices(eigenproblem_final, channels, QN, true);

	// Do time evolution of the observable <r^2>
	double time, observable, mean_value, amplitude;
	vector<double> observable_in_time, amplitudes;
	vector<int> relevant_js, relevant_jprims;
	double final_time = (double)END_IT * step;

	for (int iterator = START_IT; iterator <= END_IT; iterator++)
	{
		time = (double)iterator * step;
		observable = 0;

		if (iterator % 100 == 0)
		{
			cout << "t=" << time << "/" << final_time << endl;
		}

		for (unsigned int i = 0; i < converged_psij_indices.size(); i++) // going over psi_j
		{
			for (unsigned int iprim = 0; iprim < converged_psij_indices.size(); iprim++) // going over psi_j'
			{
				j = converged_psij_indices[i];
				jprim = converged_psij_indices[iprim];

				if (j == jprim)
				{
					mean_value = meanvalues[j * (int64_t)basis_size + j];
					observable += pow(phi_j_overlap[j], 2) * mean_value;
				}
				else if (j < jprim)
				{
					mean_value = meanvalues[j * (int64_t)basis_size + jprim];
					difference = eigenproblem_final.first[j] - eigenproblem_final.first[jprim];
					amplitude = 2 * phi_j_overlap[j] * phi_j_overlap[jprim] * mean_value;
					observable += amplitude * cos(difference * time);

					if ( (abs(amplitude) > 0.0005) && time == 0.0)
					{
						if (save_couplings == true)
						{
							relevant_js.push_back(j);
							relevant_jprims.push_back(jprim);
							amplitudes.push_back(abs(amplitude));
						}
						else
						{
							continue;
							//cout << "Energy difference: " << difference << ". States: " << j << " and " << jprim << ". Overlaps with phi(0): " << phi_j_overlap[j] * phi_j_overlap[jprim] << ". Mean value: " << meanvalues[j * (int64_t)basis_size + jprim] << ". Amplitude: " << abs(amplitude) << "." << endl;
							//printChosenEigenvector(j, eigenproblem_final, channels, QN, false);
							//printChosenEigenvector(jprim, eigenproblem_final, channels, QN, false);
							//cout << endl;
						}
					}
				}
				else
				{
					continue;
				}
			}
		}

		observable_in_time.push_back(observable);
		
	}

	// Save to file
	string filename = createQuenchFilename(string_params, string_quenched_params, "time_evolution_after_quench", true);
	
	std::ofstream timeEvolution(filename.c_str());

	for (int i = 0; i < observable_in_time.size(); i++)
	{
		timeEvolution << setprecision(6) << observable_in_time[i] << "\t";
	}

	timeEvolution.close();

	// Save couplings, if requested
	if (save_couplings == true)
	{
		vector < pair <double, int> > amps_and_indices;
		int index;
		string filename = createQuenchFilename(string_params, string_quenched_params, "quench_couplings", true);

		std::ofstream couplings(filename.c_str());

		amps_and_indices = pairAndSort(amplitudes);

		for (int i = 0; i < relevant_js.size(); i++)
		{
			index = amps_and_indices[i].second;
			j = relevant_js[index];
			jprim = relevant_jprims[index];

			couplings << "Energy difference: " << eigenproblem_final.first[j] - eigenproblem_final.first[jprim] << ". States: " << j << " and " << jprim << ". Overlaps with phi(0): " << phi_j_overlap[j] * phi_j_overlap[jprim] << ". Mean value: " << meanvalues[j * (int64_t)basis_size + jprim] << ". Amplitude: " << amps_and_indices[i].first << "." << endl;
			printChosenEigenvector(j, eigenproblem_final, channels, QN, false, couplings);
			printChosenEigenvector(jprim, eigenproblem_final, channels, QN, false, couplings);
			couplings << endl;
		}

		couplings.close();
	}
}