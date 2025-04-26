#include "Targets.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "Numerics_Functions.hpp"
#include "Physics_Functions.hpp"

std::string s_names[5] = {"s","p","d","f","g"};

std::vector<int> Count_Rows_Columns(std::string filename)
{
	int rows = 0, columns = 0;
	int first_columns = -1;
	std::ifstream f;
	f.open(filename);
	if(f.is_open())
	{
		std::string line;
		while(!f.eof()) 
	    {
	    	getline(f, line);
	    	if(line.length()>0)
	    	{
	    		rows++;
	    		columns = 0;
	    		std::istringstream iss(line);
				while(iss)
		    	{
		    		std::string subs;
	        		iss >> subs;
		    		if(subs.length()>0) columns++;
		    	}
		    	if(first_columns == -1) first_columns = columns;
		    	else if(first_columns != columns)
		    	{
	    			std::cerr <<"Error in Count_Rows_Columns(): The table in "<<filename <<" is not rectangular."<<std::endl;
					std::exit(EXIT_FAILURE);
		    	}
	    	}
	    }
	}
	else
	{
		std::cerr <<"Error in Count_Rows_Columns(): File "<<filename <<" not found."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
	f.close();
	return std::vector<int>{rows,columns};
}


Atomic_Electron::Atomic_Electron(std::string element,double A, int N, int L,double Ebinding, double kMin,double kMax, double qMin, double qMax,int neSecondary)
: n(N), l(L), k_min(kMin), k_max(kMax), q_min(qMin), q_max(qMax), logk_min(log10(kMin)) , logk_max(log10(kMax)), logq_min(log10(qMin)) , logq_max(log10(qMax)), Binding_Energy(Ebinding), Nucleus_Mass(A * mNucleon), ne_Secondary(neSecondary)
{
	Name = element + "_" + std::to_string(n) + s_names[l];
	//1. Count number of rows and columns.
		std::vector<int> rows_columns = Count_Rows_Columns("../data/atomic_response_1/"+Name+".txt");
		Nk = rows_columns[0];
		Nq = rows_columns[1];
		k_Grid = Log_Space(k_min,k_max,Nk);
		q_Grid = Log_Space(q_min,q_max,Nq);
		dlogk = log10(k_max/k_min) / (Nk-1.0);
		dlogq = log10(q_max/q_min) / (Nq-1.0);
	//2. Import electronic responses
		std::ifstream fin;
		for(int response=1; response<5; response++)
		{
			std::string filename ="../data/atomic_response_"+std::to_string(response)+"/"+Name+".txt";
			fin.open(filename);
			if(fin.is_open())
			{
				std::vector<std::vector<double>> temp;
				for(int ki=0; ki<Nk; ki++)
				{
					std::vector<double> q_line;
					for(int qi=0; qi<Nq; qi++)
					{
						double x;
						fin>>x;
						q_line.push_back(x);
					}
					temp.push_back(q_line);
				}
				Response_Tables.push_back(temp);
			}
			else
			{
				std::cerr <<"Error in Atomic_Electron::Atomic_Electron(): File "<<filename <<" not found."<<std::endl;
				std::exit(EXIT_FAILURE);
			}
			fin.close();	
		}
}



double Atomic_Electron::Electronic_Response(unsigned int response,double q, double E) const
{
	if(response<1 || response > 6)
	{
		std::cerr <<"Error in Atomic_Electron::Electronic_Response(): Electronic response index out of range." <<std::endl;
		std::exit(EXIT_FAILURE);
	}
	double k = sqrt(2.0*mElectron*E);
	int ki = std::floor(log10(k/k_min) / dlogk);
	int qi = std::floor(log10(q/q_min) / dlogq);
	if(ki== Nk-1) ki--;
	else if(ki==-1) ki++;
	//Bilinear interpolation between four points: (Source: https://en.wikipedia.org/wiki/Bilinear_interpolation)
		double x = k;
		double y = q;
		double x1 = k_Grid[ki];
		double x2 = k_Grid[ki+1];
		double y1 = q_Grid[qi];
		double y2 = q_Grid[qi+1];
		double f11 = Response_Tables[response-1][ki][qi];
		double f12 = Response_Tables[response-1][ki][qi+1];
		double f21 = Response_Tables[response-1][ki+1][qi];
		double f22 = Response_Tables[response-1][ki+1][qi+1];
		double f = 1.0 / (x2-x1) / (y2 - y1) * ( f11*(x2-x)*(y2-y) + f21*(x-x1)*(y2-y) + f12*(x2-x)*(y-y1) + f22*(x-x1)*(y-y1));
	return f;
	// int ki = std::round(log10(k/k_min) / dlogk);
	// int qi = std::round(log10(q/q_min) / dlogq);
	// return Response_Tables[response-1][ki][qi];
}


Atom::Atom(std::string element_name, int z, double a, std::vector<Atomic_Electron> shells)
: Name(element_name), Z(z), A(a), electrons(shells)
{
	mass = A * mNucleon;
	if(element_name == "Xenon" || element_name == "Xe")
	{
		W = 13.8*eV;
	}
	else if(element_name == "Argon" || element_name == "Ar")
	{
		W = 19.6*eV;
	}
	else
	{
		std::cerr<<"Error in Import_Electronic_Responses(string): Element "<<element_name <<" not recognized."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
}

double Atom::Lowest_Binding_Energy() const
{
	double Binding_Energy_Min = 1.0;
	for(unsigned int i = 0; i< electrons.size(); i++)
	{
		if( std::fabs(electrons[i].Binding_Energy) < Binding_Energy_Min) Binding_Energy_Min = std::fabs(electrons[i].Binding_Energy);
	}
	return Binding_Energy_Min;
}

Atomic_Electron Atom::Electron(int n, int l)
{
	for(unsigned int i = 0; i<electrons.size(); i++)
	{
		if(electrons[i].n == n && electrons[i].l == l) return electrons[i];
	}
	std::cerr <<"Error in Atom::Electron(): (n,l) = ("<<n<<","<<l<<") of " <<Name<<" does not exist."<<std::endl;
	std::exit(EXIT_FAILURE);
}

Atom Import_Electronic_Responses(std::string element)
{
	if(element == "Xenon" || element == "Xe")
	{
		//Xenon:
		// Atomic_Electron Xe_1s("Xe", 131.0, 1, 0, 33317.6*eV, 0.1*keV, 100.0*keV, 0.1*keV, 1000.0*keV);
		// Atomic_Electron Xe_2s("Xe", 131.0, 2, 0, 5149.21*eV, 0.1*keV, 100.0*keV, 0.1*keV, 1000.0*keV);
		// Atomic_Electron Xe_2p("Xe", 131.0, 2, 1, 4837.71*eV, 0.1*keV, 100.0*keV, 0.1*keV, 1000.0*keV);
		// Atomic_Electron Xe_3s("Xe", 131.0, 3, 0, 1093.24*eV, 0.1*keV, 100.0*keV, 0.1*keV, 1000.0*keV);
		// Atomic_Electron Xe_3p("Xe", 131.0, 3, 1, 958.43*eV, 0.1*keV, 100.0*keV, 0.1*keV, 1000.0*keV);
		// Atomic_Electron Xe_3d("Xe", 131.0, 3, 2, 710.73*eV, 0.1*keV, 100.0*keV, 0.1*keV, 1000.0*keV);
		Atomic_Electron Xe_4s("Xe", 131.0, 4, 0, 213.781*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,3);
		Atomic_Electron Xe_4p("Xe", 131.0, 4, 1, 163.495*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,6);
		Atomic_Electron Xe_4d("Xe", 131.0, 4, 2, 75.5897*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,4);
		Atomic_Electron Xe_5s("Xe", 131.0, 5, 0, 25.6986*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		Atomic_Electron Xe_5p("Xe", 131.0, 5, 1, 12.4433*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		return Atom("Xenon",54,131.0,{Xe_5p,Xe_5s,Xe_4d,Xe_4p,Xe_4s});
	}
	else if(element == "Argon" || element == "Ar")
	{
		//Argon:
		Atomic_Electron Ar_1s("Ar", 40.0, 1, 0, 3227.55*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		Atomic_Electron Ar_2s("Ar", 40.0, 2, 0, 335.303*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		Atomic_Electron Ar_2p("Ar", 40.0, 2, 1, 260.453*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		Atomic_Electron Ar_3s("Ar", 40.0, 3, 0, 34.7585*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		Atomic_Electron Ar_3p("Ar", 40.0, 3, 1, 16.0824*eV, 0.1*keV, 100.0*keV, 1.0*keV, 1000.0*keV,0);
		return Atom("Argon",18,40.0,{Ar_3p,Ar_3s,Ar_2p,Ar_2s,Ar_1s});
	}
	else
	{
		std::cerr<<"Error in Import_Electronic_Responses(string): Element "<<element <<" not recognized."<<std::endl;
		std::exit(EXIT_FAILURE);
	}
}


