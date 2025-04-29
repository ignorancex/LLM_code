#include <utility>

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#include "constants.h"
#include "parameters.h"
#include "utility.h"
#include "basis.h"
#include "build_hamiltonian.h"
#include "looped_eigenproblem.h"
#include "quench_dynamics.h"

using namespace std;

// argc - ARGument Count, including name of the program (=argv[0])
// argv[1] - integer part: 0 - only spectrum, 1 - magnetization, 2 - <J^2>, 3 - quench dynamics, 4 - print eigenvectors
//		   - fractional part: 0:vsgjj, 1:vsgjk, 2:vself, 3:vsmf, 4:vsA / or quench of 0:r2, 1:j2, 2:Jtot2, 3:Mg
// argv[2] - 0:Jtot/1:M_J/2:Mtot
// rest: -Gjj -Gjk -E -B -A -Jtot/M_J/Mtot -START_IT -END_IT -step
// if you choose to iterate over a param, set it to 0

int main(int argc, char* argv[])
{

	// START
	auto start = chrono::high_resolution_clock::now();

	// parameters that are declared in parameters.h and will be dynamically initialized within readArguments function:
	// -vsWhat -QNmode -Gjj -Gjk -elf -mf -A -Jtot/M_J/Mtot -START_IT -END_IT -step
	// as well as appropriate strings
	vector <string> string_params;	// strings order: -Gjj -Gjk -elf -mf -A -Jtot/M_J/Mtot -NMAX -JMAX -B_rot -spin -"Jtot/M_J/Mtot"
	readArguments(argc, argv);
	string_params = argToStrings(argv); 
 
	// switch: QNmode -> which channels to generate
	vector<int> channels;
	unsigned int QN = 0;
	switch (QNmode) // 0:Jtot / 1:M_J / 2:Mtot
	{ 
	case 0:
		generateChannelsNoFieldNoSpin(channels, Jtot, true);
		QN = QN_nospin;
		string_params.push_back("Jtot");
		break;
	case 1:
		generateChannelsWithFieldNoSpin(channels, M_J, false);
		QN = QN_nospin;
		string_params.push_back("M");
		break;
	case 2:
		generateChannelsWithFieldWithSpin(channels, Mtot, false);
		QN = QN_withspin;
		string_params.push_back("Mtot");
		break;
	}

	cout << "Basis allocated." << endl;

	unsigned int basis_size = basisSize(channels, QN);

	// simple diagonalization or quench dynamics?
	if (QUENCH_DYNAMICS == false)
	{
		// build hamiltonian with ifs: if param =/= 0, then... -> it'll skip also the part of hamiltonian that will be iterated over
		Matrix hamiltonian(basis_size, basis_size, 0);
		buildHamiltonian(hamiltonian, channels, QN, gjj, gjk, E, B, A);
		//hamiltonian.print("Final result");

		// the eigenproblem is solved when iterating over chosen parameter (vsWhat) with values between START_IT and END_IT with step
		loopedEigenproblem(hamiltonian, channels, QN, string_params, basis_size); // no. of eigenstates to save, max. basis_size
	}
	else
	{
		vector <string> string_quenched_params = quenchedToString(); // strings order: -qGjj -qGjk -qelf -qmf -qA -"r2/j2/Jtot2/Mg "
		vector <double> chosen_initial_state_coefficients(basis_size);

		// solve the eigenproblem for H_0 and save basis coefficients of the chosen initial state
		Matrix hamiltonian_initial(basis_size, basis_size, 0);
		buildHamiltonian(hamiltonian_initial, channels, QN, gjj, gjk, E, B, A);
		//hamiltonian_initial.print("Initial Hamiltonian");
		chosen_initial_state_coefficients = chosenInitialStateBasisCoefficients(hamiltonian_initial, channels, QN);
		
		// solve the eigenproblem for H_tot, find the overlaps between the initial state and new eigenstates, do the quench dynamics
		Matrix quenched_part_of_hamiltonian(basis_size, basis_size, 0), hamiltonian_final(basis_size, basis_size, 0);
		buildHamiltonian(quenched_part_of_hamiltonian, channels, QN, quenched_gjj, quenched_gjk, quenched_E, quenched_B, quenched_A, false);
		hamiltonian_final = hamiltonian_initial + quenched_part_of_hamiltonian;
		//hamiltonian_final.print("Final Hamiltonian");
		quenchDynamics(hamiltonian_final, channels, QN, chosen_initial_state_coefficients, string_params, string_quenched_params);
	}

	// END
	auto finish = chrono::high_resolution_clock::now();
	// print duration
	auto duration_min = chrono::duration_cast<chrono::minutes>(finish - start);
	auto duration_sec = chrono::duration_cast<chrono::seconds>(finish - start);
	cout << endl << "Run took " << duration_min.count() << " mins (" << duration_sec.count() << " s)." << endl;
}