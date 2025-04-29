#include "utility.h"

double gjj, gjk, E, B, A;
double quenched_gjj, quenched_gjk, quenched_E, quenched_B, quenched_A;
int Jtot, M_J, Mtot, S;
int START_IT, END_IT;
double step;
int QNmode, vsWhat;
double PMode;

bool ONLY_ENERGIES, MAGNETIZATION, J2, PRINT_EIGENVECTORS;
bool QUENCH_DYNAMICS, QUENCH_R2, QUENCH_MG, QUENCH_JTOT, QUENCH_J;

void readArguments(int argc, char* argv[])
{
	// possible options encoded in argv[1]: 
	//// 0.X - only spectrum iterated over vsWhat = X
	//// 1.X - magnetization iterated over vsWhat = X
	//// 2.X - <J^2> iterated over vsWhat = X
	//// 3.X - quench of X
	//// 4.X - print eigenfunctions in a basis
	// possible options encoded in argv[2]: 0 - Jtot, 1 - M_J, 2 - Mtot, 3 - S & Mtot
	// -Gjj -Gjk -elf -mf -A -Jtot/M_J/Mtot -START_IT -END_IT -step
	
	PMode = atof(argv[1]);
	QNmode = atoi(argv[2]);

	START_IT = atoi(argv[9]);
	END_IT = atoi(argv[10]);
	step = atof(argv[11]);
	
	switch (QNmode) // choose an option
	{
		case 0:
			Jtot = atoi(argv[8]);
			break;
		case 1:
			M_J = atoi(argv[8]);
			break;
		case 2:
			Mtot = atoi(argv[8]);
			break;
	}

	double intpart, fractpart;
	fractpart = modf(PMode, &intpart);
	vsWhat = (int)round(fractpart * 10);

	if (intpart == 3)
	{
		QUENCH_DYNAMICS = true;
		// 0 - r2, 1 - j2, 2 - Jtot2, 3 - Mg
		if (vsWhat == 0)
		{
			QUENCH_R2 = true;
			cout << "***INITIALIZATION OF QUENCH DYNAMICS OF MEAN VALUE OF R^2***" << endl;
		}
		else if (vsWhat == 1)
		{
			QUENCH_J = true;
			cout << "***INITIALIZATION OF QUENCH DYNAMICS OF ROTATIONAL ANGULAR MOMENTUM OF ONE MOLECULE***" << endl;
		}
		else if (vsWhat == 2)
		{
			QUENCH_JTOT = true;
			cout << "***INITIALIZATION OF QUENCH DYNAMICS OF TOTAL ROTATIONAL ANGULAR MOMENTUM***" << endl;
		}
		else if (vsWhat == 3)
		{
			QUENCH_MG = true;
			cout << "***INITIALIZATION OF QUENCH DYNAMICS OF MAGNETIZATION***" << endl;
		}
		else
		{
			cout << "ERROR! Quench parameters read improperly!" << endl;
		}

		for (int i = 3; i <= 7; i++)
		{
			if (argv[i][0] == 'q')
			{
				string q_arg(argv[i]);
				q_arg.erase(remove(q_arg.begin(), q_arg.end(), 'q'), q_arg.end());

				switch (i)
				{
				case 3:
					quenched_gjj = stod(q_arg);
					gjj = 0;
					cout << "*Quench of isotropic interaction*" << endl;
					break;
				case 4:
					quenched_gjk = stod(q_arg);
					gjk = 0;
					cout << "*Quench of anisotropic interaction*" << endl;
					break;
				case 5:
					quenched_E = stod(q_arg);
					E = 0;
					cout << "*Quench of external electric field*" << endl;
					break;
				case 6:
					quenched_B = stod(q_arg);
					B = 0;
					cout << "*Quench of external magnetic field*" << endl;
					break;
				case 7:
					quenched_A = stod(q_arg);
					A = 0;
					cout << "*Quench of spin-rotation coupling*" << endl;
					break;
				}
				
			}
			else
			{
				switch (i)
				{
				case 3:
					gjj = atof(argv[3]);
					quenched_gjj = 0;
					break;
				case 4:
					gjk = atof(argv[4]);
					quenched_gjk = 0;
					break;
				case 5:
					E = atof(argv[5]);
					quenched_E = 0;
					break;
				case 6:
					B = atof(argv[6]);
					quenched_B = 0;
					break;
				case 7:
					A = atof(argv[7]);
					quenched_A = 0;
					break;
				}
			}
		}
	}
	else
	{
		gjj = atof(argv[3]);
		gjk = atof(argv[4]);
		E = atof(argv[5]);
		B = atof(argv[6]);
		A = atof(argv[7]);

		QUENCH_DYNAMICS = false;
		string str_vsWhat[5] = { "ISOTROPIC INTERACTION", "ANISOTROPIC INTERACTION", "ELECTRIC FIELD", "MAGNETIC FIELD", "SPIN-ROTATION COUPLING"};

		if (intpart == 2)
		{
			J2 = true;
			cout << "***INITIALIZATION OF OF CALCULATIONS OF J^2 MEAN VALUE AS A FUNCTION OF " << str_vsWhat[vsWhat] << "***" << endl;
		}
		else if (intpart == 1)
		{
			MAGNETIZATION = true;
			cout << "***INITIALIZATION OF CALCULATIONS OF MAGNETIZATION AS A FUNCTION OF " << str_vsWhat[vsWhat] << "***" << endl;
		}
		else if (intpart == 4)
		{
			PRINT_EIGENVECTORS = true;
			cout << "***INITIALIZATION OF CALCULATIONS OF SPECTRUM AND EIGENVECTORS EXPRESSED IN BASIS AS A FUNCTION OF " << str_vsWhat[vsWhat] << "***" << endl;
		}
		else
		{
			ONLY_ENERGIES = true;
			cout << "***INITIALIZATION OF CALCULATIONS OF SPECTRUM AS A FUNCTION OF " << str_vsWhat[vsWhat] << "***" << endl;
		}
	}
}

void saveResults(string what_quantity, double* w, unsigned int basis_size, vector<string>& params, int iterator, double value, unsigned int how_many)
{
	string filename = createFilename(params, iterator, what_quantity);
	std::sort(w, w + basis_size);

	std::ofstream results(filename.c_str());
	results << value << "\t";

	for (unsigned int i = 0; i < how_many; i++)
	{
		results << std::setprecision(6) << w[i] << "\t";
	}

	results.close();
}

void saveResults(string what_quantity, vector<double>& results, vector<string>& params, int iterator, double value, unsigned int how_many)
{
	string filename = createFilename(params, iterator, what_quantity);

	std::ofstream file_with_results(filename.c_str());
	file_with_results << value << "\t";

	for (unsigned int i = 0; i < how_many; i++)
	{
		file_with_results << std::setprecision(6) << results[i] << "\t";
	}

	file_with_results.close();
}

vector < pair <double, int> > pairAndSort(vector<double>& first_set)
{
	vector <pair <double, int> > set_of_pairs;
	// Create vector of pairs: first set elements and their indices
	for (int i = 0; i < first_set.size(); i++)
	{
		set_of_pairs.push_back(std::make_pair(first_set[i], i));
	}

	// Sort the vector of pairs (std::sort does it according to the first element)
	std::sort(set_of_pairs.begin(), set_of_pairs.end());

	return set_of_pairs;
}

vector < pair <int, int> > pairAndSort(vector <pair <double, int> >& set_of_pairs)
{
	vector <pair <int, int> > set_of_sorted_pairs;
	// Create vector of pairs: first set elements and their indices
	for (unsigned int i = 0; i < set_of_pairs.size(); i++)
	{
		set_of_sorted_pairs.push_back(std::make_pair(set_of_pairs[i].second, i));
	}

	// Sort the vector of pairs (std::sort does it according to the first element)
	std::sort(set_of_sorted_pairs.begin(), set_of_sorted_pairs.end());

	return set_of_sorted_pairs;
}

vector < pair <double, int> > pairAndSort(vector<double>& first_set, vector<int>& second_set)
{
	vector <pair <double, int> > set_of_pairs;
	if (first_set.size() != second_set.size())
	{
		cout << "Error! Two sets intended to be paired and sorted have different sizes! Be careful with the result." << endl;
	}

	// Create vector of pairs: first set elements and their indices
	for (int i = 0; i < first_set.size(); i++)
	{
		set_of_pairs.push_back(std::make_pair(first_set[i], second_set[i]));
	}

	// Sort the vector of pairs (std::sort does it according to the first element)
	std::sort(set_of_pairs.begin(), set_of_pairs.end());

	return set_of_pairs;
}

vector < pair <double, double> > pairAndSort(vector<double>& first_set, vector<double>& second_set)
{
	vector <pair <double, double> > set_of_pairs;
	if (first_set.size() != second_set.size())
	{
		cout << "Error! Two sets intended to be paired and sorted have different sizes! Be careful with the result." << endl;
	}

	// Create vector of pairs: first set elements and their indices
	for (int i = 0; i < first_set.size(); i++)
	{
		set_of_pairs.push_back(std::make_pair(first_set[i], second_set[i]));
	}

	// Sort the vector of pairs (std::sort does it according to the first element)
	std::sort(set_of_pairs.begin(), set_of_pairs.end());

	return set_of_pairs;
}

string createFilename(vector<string>& params, int iterator, string what_quantity)
{
	string filename;
	string string_iterator = std::to_string(iterator);

	// strings order: -Gjj -Gjk -elf -mf -A -Jtot/M_J/Mtot -NMAX -JMAX -B_rot -spin

	if (iterator < 10)
	{
		filename = "00" + string_iterator + "_" + what_quantity + "_n" + params[6] + "_j" + params[7] + "_B=" + params[8] + "_s=" + params[9] + "_g" + params[0] + "_gjk" + params[1] + "_E" + params[2] + "_mf=" + params[3] + "_A" + params[4] + "_" + params[10] + params[5] + "_even.txt";
	}
	else if (iterator < 100)
	{
		filename = "0" + string_iterator + "_" + what_quantity + "_n" + params[6] + "_j" + params[7] + "_B=" + params[8] + "_s=" + params[9] + "_g" + params[0] + "_gjk" + params[1] + "_E" + params[2] + "_mf=" + params[3] + "_A" + params[4] + "_" + params[10] + params[5] + "_even.txt";
	}
	else
	{
		filename = string_iterator + "_" + what_quantity + "_n" + params[6] + "_j" + params[7] + "_B=" + params[8] + "_s=" + params[9] + "_g" + params[0] + "_gjk" + params[1] + "_E" + params[2] + "_mf=" + params[3] + "_A" + params[4] + "_" + params[10] + params[5] + "_even.txt";
	}

	return filename;
}

string createQuenchFilename(vector<string>& params, vector<string>& quenched_params, string what_quantity, bool time)
{
	string filename;
	vector<string> parameters(5);

	for (int i = 0; i <= 4; i++)
	{
		if (quenched_params[i] == "0")
		{
			parameters[i] = params[i];
		}
		else
		{
			parameters[i] = "q" + quenched_params[i];
		}
	}

	// strings order: -Gjj -Gjk -elf -mf -A -Jtot/M_J/Mtot -NMAX -JMAX -B_rot -spin -q_num (Jtot/M_J/Mtot)
	if (time == false)
	{
		filename = quenched_params[5] + "_" + what_quantity + "_n" + params[6] + "_j" + params[7] + "_B=" + params[8] + "_s=" + params[9] + "_g" + parameters[0] + "_gjk" + parameters[1] + "_E" + parameters[2] + "_mf=" + parameters[3] + "_A" + parameters[4] + "_" + params[10] + params[5] + "_even.txt";
	}
	else
	{
		filename = quenched_params[5] + "_" + what_quantity + "_n" + params[6] + "_j" + params[7] + "_B=" + params[8] + "_s=" + params[9] + "_g" + parameters[0] + "_gjk" + parameters[1] + "_E" + parameters[2] + "_mf=" + parameters[3] + "_A" + parameters[4] + "_" + params[10] + params[5] + "_start" + to_string(START_IT) + "_end" + to_string(END_IT) + "_step" + doubleToString(step) + "_even.txt";
	}
	
	return filename;
}

vector <string> argToStrings(char* argv[])
{
	vector <string> params;
	for (int i = 3; i < 9; i++)
	{
		params.push_back(argv[i]);
	}

	params.push_back(to_string(NMAX));
	params.push_back(to_string(JMAX));

	params.push_back(doubleToString(B1));
	params.push_back(doubleToString(S1));

	return params;
}

vector <string> quenchedToString()
{
	vector <string> quenched_params;

	quenched_params.push_back(doubleToString(quenched_gjj));
	quenched_params.push_back(doubleToString(quenched_gjk));
	quenched_params.push_back(doubleToString(quenched_E));
	quenched_params.push_back(doubleToString(quenched_B));
	quenched_params.push_back(doubleToString(quenched_A));

	if (QUENCH_R2 == true)
	{
		quenched_params.push_back("r2");
	}
	else if (QUENCH_J == true)
	{
		quenched_params.push_back("j2");
	}
	else if (QUENCH_JTOT == true)
	{
		quenched_params.push_back("Jtot2");
	}
	else if (QUENCH_MG == true)
	{
		quenched_params.push_back("Mg");
	}
	else
	{
		std::cout << "Error! Quench of whaat?" << std::endl;
	}

	return quenched_params;
}

string doubleToString(double number)
{
	std::stringstream stream;
	stream << number;
	return stream.str();
}

void coutSquareMatrix(vector<double>& matrix, int dim, int precision)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		cout << std::setprecision(precision) << matrix[i] << "\t";
		if ((i + 1) % dim == 0)
		{
			cout << endl;
		}
	}

	cout << endl << endl;
}

void saveSquareMatrix(vector<double>& matrix, int dim, int precision, string filename)
{
	std::ofstream file_with_results(filename.c_str());
	for (int i = 0; i < matrix.size(); i++)
	{
		file_with_results << std::setprecision(precision) << matrix[i] << "\t";
		if ((i + 1) % dim == 0)
		{
			file_with_results << endl;
		}
	}

	file_with_results.close();
}

void saveVector(vector<double>& vector, int precision, string filename)
{
	std::ofstream file_with_results(filename.c_str());
	for (int i = 0; i < vector.size(); i++)
	{
		file_with_results << std::setprecision(precision) << vector[i] << "\t";
	}

	file_with_results.close();
}