#include "Statistics.hpp"

#include <iostream>
#include <numeric>
#include <algorithm>

#include "Numerics_Functions.hpp"

//1. Distributions
double PDF_Uniform(double x, double x_min, double x_max)
{
	return 1.0/(x_max-x_min);
}

double CDF_Uniform(double x, double x_min, double x_max)
{
	return (x - x_min) / (x_max - x_min);
}

double PDF_Gauss(double x, double mu, double sigma)
{
	return 1.0/sqrt(2.0*M_PI)/sigma*exp(-pow((x-mu)/sigma,2.0)/2.0);
}

double CDF_Gauss(double x, double mu, double sigma)
{
	return 0.5 * (1.0 + erf( (x-mu) / (sqrt(2)*sigma)));
}

double Quantile_Gauss(double p, double mu, double sigma)
{
	return mu + sqrt(2.0) * sigma * Inv_erf(2.0*p-1.0);
}


double PMF_Binomial(int mu, double p, int trials)
{
	return Binomial_Coefficient(mu,trials)*pow(p,trials)*pow(1.0-p,(mu-trials));;
}

double CDF_Binomial(int mu, double p, int trials)
{
	double cdf = 0.0;
	for(int trial = 0; trial <= trials; trial++) cdf += PMF_Binomial(mu,p,trial);
	return cdf;
}

double PMF_Poisson(double expected_events, unsigned int events)
{
	if(expected_events==0&&events==0) return 1.0;
	else if(expected_events==0&&events>0) return 0.0;
	else
	{
		double sum=events*log(expected_events) -expected_events;
		for(unsigned int i=2;i<=events;i++)sum-= log(i);
		return exp(sum);
	}
}

double CDF_Poisson(double expectation_value,unsigned int observed_events)
{
	double gq = GammaQ(expectation_value,observed_events+1);
	if (gq>=0) return gq;
	else return 0.0;
}

double Inv_CDF_Poisson(unsigned int observed_events, double cdf)
{
	if(observed_events==0) return (-1.0)*log(cdf);
	else return Inv_GammaQ(cdf,observed_events+1); 
}

double PDF_Chi_Square(double x, double dof)
{
	if(x < 0)
	{
		std::cerr <<"Error in PDF_Chi_Square(double, double): Negative x value: x = "<<x<<"."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	else return 1.0/ pow(2.0,dof/2.0) / Gamma(dof/2.0) * pow(x,dof/2.0-1.0) * exp(-x/2.0);
}

double CDF_Chi_Square(double x, double dof)
{
	if(x < 0)
	{
		std::cerr <<"Error in CDF_Chi_Square(double, double): Negative x value: x = "<<x<<"."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	else if(fabs(dof)<1e-6)  return 1.0;
	else return 1.0/Gamma(dof/2.0) * Lower_Incomplete_Gamma(x/2.0,dof/2.0);
}

double PDF_Chi_Bar_Square(double x, std::vector<double> weights)
{
	if(x < 0)
	{
		std::cerr <<"Error in PDF_Chi_Bar_Square(double, std::vector<double>): Negative x value: x = "<<x<<"."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	else
	{
		double pdf = 0.0;
		for(unsigned int dof = 1; dof < weights.size(); dof++) //start at 1 because of dof = 0
		{
			pdf += weights[dof] * PDF_Chi_Square(x,dof);
		}
		return pdf;
	}
}

double CDF_Chi_Bar_Square(double x, std::vector<double> weights)
{
	if(x < 0)
	{
		std::cerr <<"Error in CDF_Chi_Bar_Square(double, std::vector<double>): Negative x value: x = "<<x<<"."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	else if(x == 0) return 0.0;
	else
	{
		double cdf = weights[0];
		for(unsigned int dof = 1; dof < weights.size(); dof++) //start at 1 because of dof = 0
		{
			cdf += weights[dof] * CDF_Chi_Square(x,dof);
		}
		if(cdf > 1.0)
		{
			std::cerr<<"Warning in CDF_Chi_Bar_Square(double,std::vector<double>): 1-CDF = "<<(1.0 - cdf)<<"< 0. Return 1." <<std::endl;
			return 1.0;
		}
		else return cdf;
	}
}


//2. Sampling random numbers
// Real random number which is used as a seed for the PRNG:
	std::random_device rd;
//The PRNG:
  	std::mt19937 generator(rd());
//Propability:
  	std::uniform_real_distribution<double> distrXi(0,1);

double Sample_Uniform(std::mt19937& PRNG, double x_min, double x_max)
{
	return x_min + distrXi(PRNG) * (x_max - x_min);
}

int Sample_Poisson(std::mt19937& PRNG, double expectation_value) //Algorithm from https://en.wikipedia.org/wiki/Poisson_distribution
{
	double STEP = 500;
	double lambda_left = expectation_value;
	int k = 0;
	double p = 1.0;
	do
	{
		k++;
		double u = distrXi(PRNG);
		p = p*u;
		while(p < 1.0 && lambda_left > 0.0)
		{
			if(lambda_left > STEP)
			{
				p = p * exp(STEP);
				lambda_left -= STEP;
			}
			else
			{
				p = p * exp(lambda_left);
				lambda_left = 0.0;
			}	
		}
	} while (p > 1);
	return (k-1);
}

//3. Likelihoods
double Likelihood_Poisson_Binned(std::vector<double> expectation_values,std::vector<int> observed_events)
{
	int N_Bins = observed_events.size();
	if(N_Bins != expectation_values.size())
	{
		std::cerr <<"Error in Likelihood_Poisson_Binned(): Expectation values and observed events for bins are not of equal size."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	double likelihood = 1.0;
	for(int bin = 0; bin < N_Bins; bin++)
	{
		int N_Observed = observed_events[bin];
		double N_Expected = expectation_values[bin];
		likelihood *= pow(N_Expected,N_Observed) / Factorial(N_Observed) * exp(-1.0*N_Expected);
	}
	return likelihood;
}
double Log_Likelihood_Poisson_Binned(std::vector<double> expectation_values,std::vector<int> observed_events)
{
	int N_Bins = observed_events.size();
	if(N_Bins != expectation_values.size())
	{
		std::cerr <<"Error in Log_Likelihood_Poisson_Binned(): Expectation values and observed events for bins are not of equal size."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	double log_likelihood = 0.0;
	for(int bin = 0; bin < N_Bins; bin++)
	{
		int N_Observed = observed_events[bin];
		double N_Expected = expectation_values[bin];
		log_likelihood += N_Observed * log(N_Expected) - N_Expected;
		double aux = 0.0;
		for(int j=1; j<= N_Observed; j++) aux += log(j);
		log_likelihood -= aux;
	}
	return log_likelihood;
}

//4. Basic data analysis
double Arithmetic_Mean(const std::vector<double>& data)
{
	return 1.0 * std::accumulate(data.begin(),data.end(),0.0) / data.size();
}

double Median(std::vector<double>& data)
{
	if (data.size() % 2 == 0)
	{
	    const auto median_it1 = data.begin() + data.size() / 2 - 1;
	    const auto median_it2 = data.begin() + data.size() / 2;
	    std::nth_element(data.begin(), median_it1 , data.end());
	    const auto e1 = *median_it1;
	    std::nth_element(data.begin(), median_it2 , data.end());
	    const auto e2 = *median_it2;
	    return (e1 + e2) / 2;
	}
	else
	{
	    const auto median_it = data.begin() + data.size() / 2;
	    std::nth_element(data.begin(), median_it , data.end());
	    return *median_it;
	}
}

double Variance(const std::vector<double>& data)
{
	double mean = Arithmetic_Mean(data);
	double variance = 0.0;
	for(unsigned int i = 0; i < data.size(); i++)
	{
		variance += (data[i] - mean) * (data[i] - mean);
	}
	variance = 1.0 * variance / data.size();
	return variance;
}

double Standard_Deviation(const std::vector<double>& data)
{
	return sqrt(Variance(data));
}


