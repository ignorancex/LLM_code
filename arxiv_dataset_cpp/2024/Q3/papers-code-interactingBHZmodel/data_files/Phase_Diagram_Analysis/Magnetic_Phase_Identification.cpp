#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
using namespace std;

typedef vector<double> Mat_1_doub;

int main (int argc, char** argv){

	string head1("Local_spin_observes_m");
	string head2("Sq_vs_U_observes_m");
	string head3("Local_charge_observes_m");
	string base(".txt");

	vector<double> m_in{0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5};
//	vector<double> m_in{0.0, 0.5};	
	double m_val;
	m_val=0.0;

	string file_NM_out_="NM_phase_points.txt";
	ofstream file_NM_out(file_NM_out_.c_str());

	string file_PM_out_="PM_phase_points.txt";
	ofstream file_PM_out(file_PM_out_.c_str());

	string file_AFM_out_="AFM_only_phase_points.txt";
	ofstream file_AFM_out(file_AFM_out_.c_str());

	string file_MI_out_="MI_AFM_phase_points.txt";
	ofstream file_MI_out(file_MI_out_.c_str());


	for(int a=1;a<=8;a++){
		file_PM_out<<0.0<<"	"<<0.25*a<<endl;
	}
	for(int a=1;a<=10;a++){
		file_NM_out<<0.0<<"	"<<2.0+0.25*a<<endl;
	}

	for(int m=0;m<m_in.size();m++){
		m_val= m_in[m];

		stringstream ss;
		if (floor(m_val) == m_val){
			ss << fixed << setprecision(1)<< m_val;
			//	cout<<m_val<<endl;
		}
		else {
			ss << m_val;
			//	cout<<m_val<<endl;
		}
		string m_val_str=ss.str();


		string file_S2_in_(head1 + m_val_str +base);
		ifstream file_S2_in(file_S2_in_.c_str());

		string line1,line2,line3,line4;
		double temp_U,temp_S2_a,temp_S2_b,temp_S2_tot;
		double temp_S00,temp_S0p,temp_Sp0,temp_Spp;

		Mat_1_doub U_val,S2a_val,S2b_val,S2tot_val;
		U_val.clear();		S2a_val.clear();
		S2b_val.clear();	S2tot_val.clear();

		while( getline(file_S2_in,line1) ){
			stringstream line_ss;
			line_ss << line1;
			line_ss>>temp_U>>temp_S2_a>>temp_S2_b>>temp_S2_tot;

			U_val.push_back(temp_U);
			S2a_val.push_back(temp_S2_a);
			S2b_val.push_back(temp_S2_b);
			S2tot_val.push_back(temp_S2_tot);
		}

		string file_Sq_in_(head2 + m_val_str +base);
		ifstream file_Sq_in(file_Sq_in_.c_str());

		Mat_1_doub S00_val,S0p_val,Sp0_val,Spp_val;
		S00_val.clear();	S0p_val.clear();
		Sp0_val.clear();	Spp_val.clear();

		while( getline(file_Sq_in,line2) ){
			stringstream line_ss;
			line_ss << line2;
			line_ss>>temp_U>>temp_S00>>temp_S0p>>temp_Sp0>>temp_Spp;

			S00_val.push_back(temp_S00);
			S0p_val.push_back(temp_S0p);
			Sp0_val.push_back(temp_Sp0);
			Spp_val.push_back(temp_Spp);
		}

		string file_charge_in_(head3 + m_val_str +base);
		ifstream file_charge_in(file_charge_in_.c_str());

		double temp_ns,temp_np;
		Mat_1_doub ns_val,np_val;
		ns_val.clear();		np_val.clear();

		while( getline(file_charge_in,line3) ){
			stringstream line_ss;
			line_ss << line3;
			line_ss>>temp_U>>temp_ns>>temp_np;

			ns_val.push_back(temp_ns);
			np_val.push_back(temp_np);
		}

		/*
		for(int i=0;i<U_val.size();i++){
			cout<<U_val[i]<<"	"<<ns_val[i]<<endl;
		}
		*/

		for(int i=0;i<U_val.size();i++){
			if( S2tot_val[i] <= 5e-2){
				file_NM_out<<U_val[i]<<"	"<<m_val<<endl;
			}
			else if( S2tot_val[i] > 5e-2 && (Spp_val[i]-Sp0_val[i]) > 0.04 && (S0p_val[i]-Sp0_val[i]) > 0.04){
				if(abs(ns_val[i]-1.0) <= 3e-2/2.0){
					file_MI_out<<U_val[i]<<"	"<<m_val<<endl;
				}
				else {
					file_AFM_out<<U_val[i]<<"	"<<m_val<<endl;
				}
			}
			else {
				file_PM_out<<U_val[i]<<"	"<<m_val<<endl;
			}
		}
	}


	return 0;
}
