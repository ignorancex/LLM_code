#include "Dirac_vs_Majorana.hpp"

#include <iostream>
#include <fstream>

#include "Statistics.hpp"

//1. Create a DM particle interacting with the photon via higher order operators (anapole, magnetic dipole, electric dipole)
DM_Particle Create_DM_Particle(double mDM, double C_1, double C_2, double C_3)
{
	DM_Particle DM(mDM);
	DM.Reset_Couplings();
	
	// 1 Anapole - Effective couplings
	double c8 = 8.0*Elementary_Charge*mElectron*mDM * C_1;
	double c9 = -8.0*Elementary_Charge*mElectron*mDM * C_1;

	// 2. Magnetic dipole - Effective couplings
	double c1 = 4.0*Elementary_Charge*mElectron * C_2;
	double c4 = 16.0*Elementary_Charge*mDM * C_2;
	double c5_long_range = 16.0*Elementary_Charge*mDM*mElectron*mElectron/qRef/qRef * C_2;
	double c6_long_range = -16.0*Elementary_Charge*mDM*mElectron*mElectron/qRef/qRef * C_2;

	// 3. Electric dipole - Effective couplings
	double c11_long_range = 16.0*Elementary_Charge*mDM*mElectron*mElectron/qRef/qRef * C_3;
	
	// 4. Set the couplings
	DM.Set_Coupling(8,c8);
	DM.Set_Coupling(9,c9);

	DM.Set_Coupling(1,c1);
	DM.Set_Coupling(4,c4);
	DM.Set_Coupling(5, 0.0, c5_long_range);
	DM.Set_Coupling(6, 0.0, c6_long_range);

	DM.Set_Coupling(11, 0.0, c11_long_range);

	return DM;
}

//2. Event spectrum in terms of n_e.
std::vector<double> Binned_Event_Spectrum(Detector_Ionization& detector, double mDM, double C_1, double C_2, double C_3)
{
	// 1. Define the DM particle instance.
	DM_Particle DM = Create_DM_Particle(mDM,C_1,C_2,C_3);

	std::vector<double> expectation_values = {};
	for(int ne = detector.ne_threshold; ne < 16; ne++)
	{
		double N_expected = detector.exposure * detector.dRdne(ne,DM);
		expectation_values.push_back(N_expected);
	}
	return expectation_values;
}

//3. Tabulate the three fiducial spectra.
std::vector<std::vector<double>> Fiducial_Spectra(Detector_Ionization& detector, double mDM)
{
	std::vector<std::vector<double>> fiducial_spectra = {};
	fiducial_spectra.push_back(Binned_Event_Spectrum(detector, mDM, 1.0, 0.0, 0.0));
	fiducial_spectra.push_back(Binned_Event_Spectrum(detector, mDM, 0.0, 1.0, 0.0));
	fiducial_spectra.push_back(Binned_Event_Spectrum(detector, mDM, 0.0, 0.0, 1.0));
	return fiducial_spectra;
}

//4. Fix couplings to a given number of events and hierarchy.
std::vector<double> Determine_Couplings(Detector_Ionization& detector, double mDM, double number_of_events, std::vector<double> operator_hierarchy, std::vector<std::vector<double>> fiducial_spectra)
{
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}

	std::vector<double> fiducial_total_events = {};
	fiducial_total_events.push_back( std::accumulate(fiducial_spectra[0].begin(), fiducial_spectra[0].end(), 0.0) );
	fiducial_total_events.push_back( std::accumulate(fiducial_spectra[1].begin(), fiducial_spectra[1].end(), 0.0) );
	fiducial_total_events.push_back( std::accumulate(fiducial_spectra[2].begin(), fiducial_spectra[2].end(), 0.0) );

	double normalization = std::accumulate(operator_hierarchy.begin(), operator_hierarchy.end(), 0.0);
	double C_1 = sqrt(operator_hierarchy[0]/normalization) * sqrt(number_of_events / fiducial_total_events[0]);
	double C_2 = sqrt(operator_hierarchy[1]/normalization) * sqrt(number_of_events / fiducial_total_events[1]);
	double C_3 = sqrt(operator_hierarchy[2]/normalization) * sqrt(number_of_events / fiducial_total_events[2]);
	return {C_1,C_2,C_3};
}

//5. Monte Carlo simulate an experiment generating number of events in a given number of bins between E_i, E_f
std::vector<std::vector<double>> Simulate_Experiment(std::mt19937& PRNG, Detector_Ionization& detector, double mDM, double C_1, double C_2, double C_3, std::vector<std::vector<double>> fiducial_spectra, bool terminal_output)
{
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}

	std::vector<std::vector<double>> data;

	// Bin size
	if(terminal_output)
	{
		std::cout 	<<"\nMonte Carlo simulation of a data run by a direct detection experiment."<<std::endl
					<<"n_e threshold:\t"<<detector.ne_threshold<<std::endl
					<<"Number of bins:\t"<<(16-detector.ne_threshold)<<std::endl
					<<"Benchmark couplings:"<<std::endl
					<<"\tC_1 = " <<C_1<<std::endl
					<<"\tC_2 = " <<C_2<<std::endl
					<<"\tC_3 = " <<C_3<<std::endl <<std::endl
					<<"n_e\tN\t<N>"<<std::endl;
	}
	
	double N_expected_tot = 0.0;
	int N_simulated_tot = 0;
	for(int ne = detector.ne_threshold; ne < 16; ne++)
	{
		double N_expected = 0.0;
		N_expected += C_1*C_1 * fiducial_spectra[0][ne-detector.ne_threshold];
		N_expected += C_2*C_2 * fiducial_spectra[1][ne-detector.ne_threshold];
		N_expected += C_3*C_3 * fiducial_spectra[2][ne-detector.ne_threshold];
		int N_simulated = Sample_Poisson(PRNG,N_expected);

		N_expected_tot += N_expected;
		N_simulated_tot += N_simulated;

		if(terminal_output) std::cout <<ne<<"\t"<<N_simulated <<"\t" <<Round(N_expected) <<std::endl;
		data.push_back({(double)ne,(double)N_simulated,N_expected});
	}
	if(terminal_output)
	{
		std::cout	<<"______________________"<<std::endl
					<<"\t"<<N_simulated_tot<<"\t"<<Round(N_expected_tot)<<std::endl<<std::endl;
	}
	return data;
}

std::vector<std::vector<double>> Import_MC_Data(std::string filename, bool terminal_output)
{
	std::vector<std::vector<double>> data;

	std::ifstream f;
	f.open(filename);
	if (f.good())
	{

        while (!f.eof())
        {
        	std::vector<double> aux;
        	double x1,x2,x3;
            f >>x1;
            f >>x2;
            f >>x3;

			aux.push_back(x1);	    
			aux.push_back(x2);	    
			aux.push_back(x3);	
	
            data.push_back(aux);
        }
        // Close the file.
        f.close();
	}
	else
	{
    	std::cerr << "Error in Import_MC_Data(" <<filename<<"): File does not exist."<<std::endl;
    	std::exit(EXIT_FAILURE);
	}
	if(terminal_output)
	{
		std::cout 	<<"\nImport of MC data from " <<filename<<"."<<std::endl
					<<"n_e threshold:\t"<<data.front()[0]<<std::endl
					<<"Number of bins:\t"<<data.size()<<std::endl <<std::endl
					<<"n_e\tN\t<N>"<<std::endl;
	}
	
	double N_expected_tot = 0.0;
	int N_simulated_tot = 0;
	for(unsigned int i = 0; i < data.size(); i++)
	{
		N_expected_tot += data[i][2];
		N_simulated_tot += data[i][1];
		if(terminal_output) std::cout <<data[i][0] <<"\t" <<data[i][1] <<"\t" <<Round(data[i][2]) <<std::endl;
	}
	if(terminal_output) std::cout	<<"______________________"<<std::endl
				<<"\t"<<N_simulated_tot<<"\t"<<Round(N_expected_tot)<<std::endl;
	return data;
}

//6. Statistics functions
double Test_Statistic_t(std::vector<std::vector<double>>& data, Detector_Ionization& detector,double mDM, std::vector<std::vector<double>> fiducial_spectra, bool terminal_output)
{
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}

	if(terminal_output) std::cout<<"Compute test statistic q for a given data set."<<std::endl;
	
	double ftol = 1.0e-12;
  	Minimization am(ftol);

  	std::vector<int> events = {};
	for(int i = 0; i < data.size(); i++) events.push_back(data[i][1]);

	//1. Hypothesis Majorana
	if(terminal_output)  std::cout<<"Majorana hypothesis: Maximize the L(data|C_1) likelihood" <<std::endl; 

	//1.1 Formulate the log likelihood as one-parameter function.
  	std::function<double(std::vector<double>)> logL_M = [&events, &fiducial_spectra] (std::vector<double> theta)
  	{
  		std::vector<double> expectation_values = {};
  		for(unsigned int i = 0; i<fiducial_spectra[0].size(); i++ ) expectation_values.push_back(theta[0]*theta[0] * fiducial_spectra[0][i]);
  		double minus_log_llh = (-1.0) * Log_Likelihood_Poisson_Binned(expectation_values, events);
  		return minus_log_llh;
  	};
  	//1.2 Maximize the log likelihood
  	std::vector<double> theta_start_M = {3e-3};
  	double delta = theta_start_M[0] / 3.0;
  	std::vector<double> theta_min_M = am.minimize(theta_start_M,delta,logL_M);
  	double log_likelihood_Majorana = (-1.0)*am.fmin;
  	if(terminal_output) std::cout<<"Maximum log likelihood at\n\tC1 = " <<theta_min_M[0] <<"\nlog L = " <<log_likelihood_Majorana<<std::endl<<std::endl;
	
	//2. Hypothesis Dirac
	if(terminal_output) std::cout<<"Dirac hypothesis: Maximize the L(data|C_1,C_2,C_3) likelihood." <<std::endl; 
	//2.1 Formulate the log likelihood as one-parameter function.
  	std::function<double(std::vector<double>)> logL_D = [&events, &fiducial_spectra] (std::vector<double> theta)
  	{
  		std::vector<double> expectation_values = {};
  		for(unsigned int i = 0; i<fiducial_spectra[0].size(); i++ ) 
		{
			double contribution_1 = theta[0]*theta[0] * fiducial_spectra[0][i];
			double contribution_2 = theta[1]*theta[1] * fiducial_spectra[1][i];
			double contribution_3 = theta[2]*theta[2] * fiducial_spectra[2][i];
  			expectation_values.push_back(contribution_1 + contribution_2 + contribution_3);
		}
  		double minus_log_llh = (-1.0) * Log_Likelihood_Poisson_Binned(expectation_values, events);
  		return minus_log_llh;
  	};

  	//2.2 Maximize the log likelihood
  	std::vector<double> theta_start_D = {3e-3,3e-7,3e-8};
  	std::vector<double> deltas = {theta_start_D[0] / 3.0,theta_start_D[1] / 3.0,theta_start_D[2] / 3.0};
  	std::vector<double> theta_min_D = am.minimize(theta_start_D,deltas,logL_D);
  	double log_likelihood_Dirac = (-1.0)*am.fmin;
  	if(terminal_output) 
  	{
  		std::cout 	<<"Maximum log likelihood at"<<std::endl
			  		<<"\tC1 = " <<theta_min_D[0]<<std::endl
			  		<<"\tC2 = " <<theta_min_D[1]<<std::endl
			  		<<"\tC3 = " <<theta_min_D[2]<<std::endl
			  		<<"log L = " <<log_likelihood_Dirac<<std::endl<<std::endl;
  	}
  	double t = -2.0 * (log_likelihood_Majorana - log_likelihood_Dirac);
  	if(std::fabs(t) < 1e-6) return 0.0;
	else return t;
}

Matrix Fisher_Matrix_Asimov(Detector_Ionization& detector, double mDM,  double C_1, double C_2, double C_3, std::vector<std::vector<double>>& fiducial_spectra)
{
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}
	
	Matrix output(3,3,0.0);
	for(unsigned int i = 0; i < output.Rows(); i++)
	{
		for(unsigned int j = 0; j < output.Columns(); j++)
		{
			for(unsigned int bin = 0; bin < fiducial_spectra[0].size(); bin++)
			{
				double N_i = C_1*C_1 * fiducial_spectra[0][bin] + C_2*C_2 * fiducial_spectra[1][bin] + C_3*C_3 * fiducial_spectra[2][bin];
				output[i][j] += 1.0/N_i * (1.0 * fiducial_spectra[i][bin]) * (1.0 * fiducial_spectra[j][bin]);
			}
		}
	}
	return output;
}

std::vector<double> Sample_t(unsigned int sample_size, std::mt19937& PRNG, Detector_Ionization& detector, double mDM, double C_1_true, double C_2_true, double C_3_true,  std::vector<std::vector<double>> fiducial_spectra, bool terminal_output)
{
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}

	std::vector<double> t_sample = {};
	for(int i = 0; i < sample_size; i++)
	{
	  	std::vector<std::vector<double>> simulated_data = Simulate_Experiment(PRNG, detector, mDM, C_1_true, C_2_true, C_3_true, fiducial_spectra, terminal_output);
	  	double t = Test_Statistic_t(simulated_data, detector,mDM, fiducial_spectra, terminal_output);
	  	t_sample.push_back(t);
		if(terminal_output) std::cout <<i+1<<")\tt =\t" <<t<<std::endl;
	}
	return t_sample;
}

std::vector<double> Asymptotic_Distribution_Weights(std::vector<double> &t_sample, int N_weights)
{
	if(N_weights == 1) return {1.0};
	else if(N_weights == 2) return {0.5,0.5};
	std::vector<double> weights(N_weights,0.0);
	unsigned int sample_size = t_sample.size();
	//1. N-2 reference values of q.
	std::vector<double> t_ref = {};
	for(int i = 0; i < (N_weights-2); i++) t_ref.push_back(1.0*i);

	//2. Compute the CDF values at the reference values by counting.
	std::vector<unsigned int> counters(N_weights-2,0);
	for(unsigned int i = 0; i < sample_size; i++)
	{
		if(std::fabs(t_sample[i]) < 1.0e-6) counters[0]++;
		for(int j = 1; j < (N_weights-2);j++)
		{
			if(t_sample[i] < t_ref[j]) counters[j]++;
		}
	}

	std::vector<double> cdf_ref(N_weights-2,0.0);
	for(unsigned int i = 0; i < cdf_ref.size(); i++) 
	{
		cdf_ref[i] = 1.0 * counters[i] / sample_size;
	}

	//3. Compute weights by solving a system of linear equations via matrix inversion M.w = rhs. 
	//3.1 Construct the right hand side vector.
	std::vector<double> rhs_comp(N_weights,0.0);
	for(int i = 0; i < N_weights; i++)
	{
		if(i == N_weights-2) rhs_comp[i] = 0.5;
		else if(i == N_weights-1) rhs_comp[i] = 1.0;
		else rhs_comp[i] = cdf_ref[i];
	}
	Vector rhs(rhs_comp);
	//3.2 Construct and inverse the matrix M
	std::vector<std::vector<double>> M_comp(N_weights, std::vector<double>(N_weights,0.0));
	for(unsigned int i = 0; i < N_weights; i++)
	{
		for(unsigned int j = 0; j < N_weights; j++)
		{
			if(i<N_weights-2)
			{
				M_comp[i][j] = CDF_Chi_Square(t_ref[i],j); 
			}
			else if( i == N_weights-1) M_comp[i][j] = 1.0;
			else if( N_weights%2 == 0) M_comp[i][j] = (j+1)%2;
			else M_comp[i][j] = j%2;
		} 
	}
	Matrix M(M_comp);
	Matrix M_inv = M.Inverse();
	Vector solution= M_inv*rhs;
	for(int i = 0; i < N_weights; i++) 
	{
		weights[i] = solution[i];
		if(weights[i]< -0.01) std::cout <<"Warning in Asymptotic_Distribution_Weights(std::vector<double>,int): w_"<<i<<"="<<weights[i]<<" < 0."<<std::endl;
	}
	return weights;
}

double Compute_p_Value(unsigned int sample_size, std::mt19937& PRNG, Detector_Ionization& detector, double number_of_signals, double mDM, std::vector<double> operator_hierarchy, std::vector<std::vector<double>> fiducial_spectra,  bool terminal_output)
{
	double p_value = 1.0;
	//1. Compute the fiducial spectra if necessary.
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}

	//2. Hypothesis Dirac:
	//2.1 Fix the couplings to the desired number of signals and hierarchy.
	std::vector<double> C_Dirac = Determine_Couplings(detector, mDM, number_of_signals, operator_hierarchy, fiducial_spectra);

	//2.2 Compute a sample of the test statistic q under H_D and its median value.
	int sample_size_D = 10000;
	std::vector<double> t_D_sample = Sample_t(sample_size_D, PRNG, detector, mDM, C_Dirac[0], C_Dirac[1], C_Dirac[2],fiducial_spectra, terminal_output);
	double t_D_median = Median(t_D_sample);

	//3. Hypothesis Majorana
	//3.1 Fix the couplings.
	std::vector<double> C_Majorana = Determine_Couplings(detector, mDM, number_of_signals, {1.0,0.0,0.0}, fiducial_spectra);

	//3.2 Compute a sample of the test statistic q under H_M.
	std::vector<double> t_M_sample = Sample_t(sample_size, PRNG, detector, mDM, C_Majorana[0], C_Majorana[1], C_Majorana[2], fiducial_spectra, terminal_output);

	//4. Compute p-value(s)
	//4.1 Monte Carlo
	int number_of_t_above_t_median = 0;
	for(unsigned int i = 0; i < sample_size; i++)
	{
		if(t_M_sample[i] > t_D_median) number_of_t_above_t_median++;
	}
	double p_value_MC= 1.0*number_of_t_above_t_median/sample_size;	
	if(terminal_output) std::cout <<std::endl<<"P value with MC:\t" <<p_value_MC<<std::endl;

	//4.2 Asymptotic q distribution
	if(number_of_signals > 10)
	{
		std::vector<double> chi_bar_weights = Asymptotic_Distribution_Weights(t_M_sample,4);
		double p_value_asymptotics = 1.0 - CDF_Chi_Bar_Square(t_D_median, chi_bar_weights);
		if(terminal_output)
		{
			std::cout <<"P value with Asympt.:\t" <<p_value_asymptotics <<std::endl;
			std::cout <<"\t(Weights:\t"<<Vector(chi_bar_weights)<<")."<<std::endl;
		}
		if(p_value_MC > 0.0 && number_of_t_above_t_median > 100) p_value = p_value_MC;
		else p_value = p_value_asymptotics;
	}
	else p_value = p_value_MC;
	return p_value;
}


double Significance(double p_value)
{
	return Quantile_Gauss(1.0-p_value, 0.0, 1.0);
}

double Significance_Threshold(double Z, unsigned int sample_size, std::mt19937& PRNG, Detector_Ionization& detector, double mDM, std::vector<double> operator_hierarchy, double N_min, double N_max, std::vector<std::vector<double>> fiducial_spectra,  bool terminal_output)
{
	if(fiducial_spectra.empty())
	{
		fiducial_spectra = Fiducial_Spectra(detector,mDM);
	}
	std::function<double(double)> fct = [Z, sample_size,&PRNG,&detector,mDM,operator_hierarchy,&fiducial_spectra,terminal_output] (double N)
	{
		double p = Compute_p_Value(sample_size,PRNG,detector, N, mDM, operator_hierarchy, fiducial_spectra);
		double z = Significance(p);
		if(terminal_output) std::cout<<"\tZ("<<N<<")="<<z<<std::endl;
		return z - Z;
	};
	double N_Z = Find_Root(fct,N_min,N_max, 1e-1);
	if(terminal_output) std::cout<<"A significance of Z="<<Z<<" can be expected from the observation of ~"<<N_Z<<" events."<<std::endl;
	return N_Z;
}





