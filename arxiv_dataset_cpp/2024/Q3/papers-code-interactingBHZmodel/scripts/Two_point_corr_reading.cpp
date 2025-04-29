#include "tensor.h"
#include <fstream>
#include <iostream>
//#include <iomanip>
#include <algorithm>
#define PI 3.14159265359
using namespace std;

complex<double> reading_pair(string pair_str){

	complex<double> value;
	double part_one, part_two;

	int pair_str_Length=pair_str.length()-1;
	int pos = pair_str.find(",");

	string one_str=pair_str.substr(1,pos-1);
	string two_str=pair_str.substr(pos+1,pair_str_Length-1-pos);

	stringstream onestream(one_str);
	stringstream twostream(two_str);
	onestream>>part_one;
	twostream>>part_two;

	value.real(part_one);
	value.imag(part_two);

	return value;
}


int main (int argc, char** argv){

	int Size=atoi(argv[1]);
	int Ly_=4;
	int Lx_=Size/Ly_;

	complex<double> Zero_Complex(0.0,0.0),Iota_Complex(0.0,1.0),One_Complex(1.0,0.0);

	//--------------------------------------------------------------------------//
	//Initializing:--->
	string Si_Sj_0_in_, Si_Sj_1_in_, Si_Sj_tot_in_;

	//SS files:----->
	Si_Sj_0_in_="Si_Sj_orb-0.txt";
	ifstream si_sj_0_in(Si_Sj_0_in_.c_str());

	Si_Sj_1_in_="Si_Sj_orb-1.txt";
	ifstream si_sj_1_in(Si_Sj_1_in_.c_str());

	Si_Sj_tot_in_="Si_Sj_tot.txt";
	ifstream si_sj_in(Si_Sj_tot_in_.c_str());

	//--------------------------------------------------------------------------//
	//SS files resizing:--->
	Mat_2_Complex_doub SS_orb0_mat, SS_orb1_mat, SS_tot_mat;
	SS_orb0_mat.resize(Size);
	SS_orb1_mat.resize(Size);
	SS_tot_mat.resize(Size);

	for(int len=0;len<Size;len++){
		SS_orb0_mat[len].resize(Size);
		SS_orb1_mat[len].resize(Size);
		SS_tot_mat[len].resize(Size);
	}

	//--------------------------------------------------------------------------//
	//Reading Spin-Spin Correlation matrices----->
	string orb0_si_sj, orb1_si_sj, total_si_sj;
	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			si_sj_0_in>>orb0_si_sj;
			si_sj_1_in>>orb1_si_sj;
			si_sj_in>>total_si_sj;

			SS_orb0_mat[i][j]=reading_pair(orb0_si_sj);
			SS_orb1_mat[i][j]=reading_pair(orb1_si_sj);
			SS_tot_mat[i][j]=reading_pair(total_si_sj);
		}
	}

	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			if(i>j){
				SS_orb0_mat[i][j] = conj(SS_orb0_mat[j][i]);
				SS_orb1_mat[i][j] = conj(SS_orb1_mat[j][i]);
				SS_tot_mat[i][j] = conj(SS_tot_mat[j][i]);
			}
		}
	}

	//--------------------------------------------------------------------------//
	//Local charge files:---->
	string n_0_up_in_,n_1_up_in_,n_0_dn_in_,n_1_dn_in_;
	string line1,line2,line3,line4;

	n_0_up_in_="n0_up.txt";
	ifstream n0up_in(n_0_up_in_.c_str());

	n_1_up_in_="n1_up.txt";
	ifstream n1up_in(n_1_up_in_.c_str());

	n_0_dn_in_="n0_dn.txt";
	ifstream n0dn_in(n_0_dn_in_.c_str());

	n_1_dn_in_="n1_dn.txt";
	ifstream n1dn_in(n_1_dn_in_.c_str());

	Mat_1_int i_val,j_val,k_val,l_val;
	i_val.clear();j_val.clear();k_val.clear();l_val.clear();
	int temp_i,temp_j,temp_k,temp_l;

	Mat_1_doub n_0up,n_1up,n_0dn,n_1dn;
	n_0up.clear();n_1up.clear();n_0dn.clear();n_1dn.clear();
	double temp_n0up,temp_n0dn,temp_n1up,temp_n1dn;
	double null;

	while( getline(n0up_in,line1) ){
		stringstream line_ss;
		line_ss << line1;
		line_ss>>temp_i>>temp_n0up>>null;

		i_val.push_back(temp_i);
		n_0up.push_back(temp_n0up);
	}
	while( getline(n1up_in,line2) ){
		stringstream line_ss;
		line_ss << line2;
		line_ss>>temp_j>>temp_n1up>>null;

		j_val.push_back(temp_j);
		n_1up.push_back(temp_n1up);
	}
	while( getline(n0dn_in,line3) ){
		stringstream line_ss;
		line_ss << line3;
		line_ss>>temp_k>>temp_n0dn>>null;

		k_val.push_back(temp_k);
		n_0dn.push_back(temp_n0dn);
	}
	while( getline(n1dn_in,line4) ){
		stringstream line_ss;
		line_ss << line4;
		line_ss>>temp_l>>temp_n1dn>>null;

		l_val.push_back(temp_l);
		n_1dn.push_back(temp_n1dn);
	}

	//--------------------------------------------------------------------------//
	string ni_1_up_nj_1_up_, ni_1_up_nj_0_up_, ni_1_up_nj_1_dn_, ni_1_up_nj_0_dn_;
	string ni_0_up_nj_0_up_, ni_0_up_nj_1_dn_, ni_0_up_nj_0_dn_;
	string ni_1_dn_nj_1_dn_, ni_1_dn_nj_0_dn_;
	string ni_0_dn_nj_0_dn_;

	//nn files:---->
	ni_1_up_nj_1_up_="ni1up_nj1up.txt";
	ifstream ni_1_up_nj_1_up_in(ni_1_up_nj_1_up_.c_str());

	ni_1_up_nj_0_up_="ni1up_nj0up.txt";
	ifstream ni_1_up_nj_0_up_in(ni_1_up_nj_0_up_.c_str());

	ni_1_up_nj_1_dn_="ni1up_nj1dn.txt";
	ifstream ni_1_up_nj_1_dn_in(ni_1_up_nj_1_dn_.c_str());

	ni_1_up_nj_0_dn_="ni1up_nj0dn.txt";
	ifstream ni_1_up_nj_0_dn_in(ni_1_up_nj_0_dn_.c_str());//ni_{1,up} done


	ni_0_up_nj_0_up_="ni0up_nj0up.txt";
	ifstream ni_0_up_nj_0_up_in(ni_0_up_nj_0_up_.c_str());

	ni_0_up_nj_1_dn_="ni0up_nj1dn.txt";
	ifstream ni_0_up_nj_1_dn_in(ni_0_up_nj_1_dn_.c_str());

	ni_0_up_nj_0_dn_="ni0up_nj0dn.txt";
	ifstream ni_0_up_nj_0_dn_in(ni_0_up_nj_0_dn_.c_str());//ni_{0,up} done


	ni_1_dn_nj_1_dn_="ni1dn_nj1dn.txt";
	ifstream ni_1_dn_nj_1_dn_in(ni_1_dn_nj_1_dn_.c_str());

	ni_1_dn_nj_0_dn_="ni1dn_nj0dn.txt";
	ifstream ni_1_dn_nj_0_dn_in(ni_1_dn_nj_0_dn_.c_str());//ni_{1,dn} done


	ni_0_dn_nj_0_dn_="ni0dn_nj0dn.txt";
	ifstream ni_0_dn_nj_0_dn_in(ni_0_dn_nj_0_dn_.c_str());//ni_{0,dn} done

	//nn files declaration done:----


	//--------------------------------------------------------------------------//
	//nn files resizing:---->
	Mat_2_Complex_doub nn_1up_1up, nn_1up_0up, nn_1up_1dn, nn_1up_0dn;
	Mat_2_Complex_doub nn_0up_0up, nn_0up_1dn, nn_0up_0dn;
	Mat_2_Complex_doub nn_1dn_1dn, nn_1dn_0dn;
	Mat_2_Complex_doub nn_0dn_0dn;
	nn_1up_1up.resize(Size); nn_1up_0up.resize(Size); nn_1up_1dn.resize(Size); nn_1up_0dn.resize(Size);
	nn_0up_0up.resize(Size); nn_0up_1dn.resize(Size); nn_0up_0dn.resize(Size);
	nn_1dn_1dn.resize(Size); nn_1dn_0dn.resize(Size);
	nn_0dn_0dn.resize(Size);

	for(int len=0;len<Size;len++){
		nn_1up_1up[len].resize(Size); nn_1up_0up[len].resize(Size); nn_1up_1dn[len].resize(Size); nn_1up_0dn[len].resize(Size);
		nn_0up_0up[len].resize(Size); nn_0up_1dn[len].resize(Size); nn_0up_0dn[len].resize(Size);
		nn_1dn_1dn[len].resize(Size); nn_1dn_0dn[len].resize(Size);
		nn_0dn_0dn[len].resize(Size);
	}

	//--------------------------------------------------------------------------//
	//Reading Charge-Charge Correlation matrices----->
	string dummy_1up_1up, dummy_1up_0up, dummy_1up_1dn, dummy_1up_0dn;
	string dummy_0up_0up, dummy_0up_1dn, dummy_0up_0dn;
	string dummy_1dn_1dn, dummy_1dn_0dn;
	string dummy_0dn_0dn;
	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			ni_1_up_nj_1_up_in>>dummy_1up_1up;      ni_1_up_nj_0_up_in>>dummy_1up_0up;
			ni_1_up_nj_1_dn_in>>dummy_1up_1dn;      ni_1_up_nj_0_dn_in>>dummy_1up_0dn;

			ni_0_up_nj_0_up_in>>dummy_0up_0up;
			ni_0_up_nj_1_dn_in>>dummy_0up_1dn;      ni_0_up_nj_0_dn_in>>dummy_0up_0dn;

			ni_1_dn_nj_1_dn_in>>dummy_1dn_1dn;      ni_1_dn_nj_0_dn_in>>dummy_1dn_0dn;

			ni_0_dn_nj_0_dn_in>>dummy_0dn_0dn;

			nn_1up_1up[i][j]=reading_pair(dummy_1up_1up);   nn_1up_0up[i][j]=reading_pair(dummy_1up_0up);
			nn_1up_1dn[i][j]=reading_pair(dummy_1up_1dn);   nn_1up_0dn[i][j]=reading_pair(dummy_1up_0dn);

			nn_0up_0up[i][j]=reading_pair(dummy_0up_0up);
			nn_0up_1dn[i][j]=reading_pair(dummy_0up_1dn);   nn_0up_0dn[i][j]=reading_pair(dummy_0up_0dn);

			nn_1dn_1dn[i][j]=reading_pair(dummy_1dn_1dn);   nn_1dn_0dn[i][j]=reading_pair(dummy_1dn_0dn);

			nn_0dn_0dn[i][j]=reading_pair(dummy_0dn_0dn);
		}
	}

	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			if(i>j){

				nn_1up_1up[i][j]=conj(nn_1up_1up[j][i]);        nn_1up_0up[i][j]=conj(nn_1up_0up[j][i]);
				nn_1up_1dn[i][j]=conj(nn_1up_1dn[j][i]);        nn_1up_0dn[i][j]=conj(nn_1up_0dn[j][i]);

				nn_0up_0up[i][j]=conj(nn_0up_0up[j][i]);
				nn_0up_1dn[i][j]=conj(nn_0up_1dn[j][i]);        nn_0up_0dn[i][j]=conj(nn_0up_0dn[j][i]);

				nn_1dn_1dn[i][j]=conj(nn_1dn_1dn[j][i]);        nn_1dn_0dn[i][j]=conj(nn_1dn_0dn[j][i]);

				nn_0dn_0dn[i][j]=conj(nn_0dn_0dn[j][i]);
			}
		}
	}

	//--------------------------------------------------------------------------//
	//Creating charge-charge correlations matrices for individual orbitals and for total system:---->
	Mat_2_Complex_doub nn_00_mat,nn_11_mat,nn_10_mat;
	Mat_2_Complex_doub nn_tot_mat,tztz_corr_mat;
	nn_00_mat.resize(Size);	nn_10_mat.resize(Size);	nn_11_mat.resize(Size);
	nn_tot_mat.resize(Size);	tztz_corr_mat.resize(Size);
	for(int i=0;i<Size;i++){
		nn_00_mat[i].resize(Size);	nn_11_mat[i].resize(Size);
		nn_10_mat[i].resize(Size);	
		nn_tot_mat[i].resize(Size);	tztz_corr_mat[i].resize(Size);
	}

	string file_NN0_mat="NN_orb-0_matrix.txt";
	ofstream file_NN_orb0_mat(file_NN0_mat.c_str());

	string file_NN1_mat="NN_orb-1_matrix.txt";
	ofstream file_NN_orb1_mat(file_NN1_mat.c_str());

	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			nn_00_mat[i][j]=nn_0up_0up[i][j] + nn_0up_0dn[i][j] + nn_0up_0dn[j][i] + nn_0dn_0dn[i][j]
				- 1.0*One_Complex*(n_0up[i]+n_0dn[i])*(n_0up[j]+n_0dn[j]);
			nn_11_mat[i][j]=nn_1up_1up[i][j] + nn_1up_1dn[i][j] + nn_1up_1dn[j][i] + nn_1dn_1dn[i][j]
				- 1.0*One_Complex*(n_1up[i]+n_1dn[i])*(n_1up[j]+n_1dn[j]);
			nn_10_mat[i][j]=nn_1up_0up[i][j] + nn_1up_0dn[i][j] + nn_0up_1dn[j][i] + nn_1dn_0dn[i][j]
				- 1.0*One_Complex*(n_1up[i]+n_1dn[i])*(n_0up[j]+n_0dn[j]);

			file_NN_orb0_mat<<nn_00_mat[i][j]<<" ";
			file_NN_orb1_mat<<nn_11_mat[i][j]<<" ";
		}
		file_NN_orb0_mat<<endl;
		file_NN_orb1_mat<<endl;
	}

	string file_NN_mat="NN_total_matrix.txt";
	ofstream file_NN_tot_mat(file_NN_mat.c_str());

	string file_TzTz_mat="TzTz_matrix.txt";
	ofstream file_TzTz_tot_mat(file_TzTz_mat.c_str());

	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			nn_tot_mat[i][j]=nn_00_mat[i][j] + nn_11_mat[i][j] + nn_10_mat[i][j] + nn_10_mat[j][i];
			tztz_corr_mat[i][j]=0.25*(nn_00_mat[i][j] + nn_11_mat[i][j] - nn_10_mat[i][j] - nn_10_mat[j][i]);

			file_NN_tot_mat<<nn_tot_mat[i][j]<<" ";
			file_TzTz_tot_mat<<tztz_corr_mat[i][j]<<" ";
		}
		file_NN_tot_mat<<endl;
		file_TzTz_tot_mat<<endl;
	}


	//--------------------------------------------------------------------------//
	//Generating charge-charge correlations and local charge order params data files:------->
	string file_nn_orb0_out="Nd_corr_vs_d_orb0.txt";
	ofstream file_nin0_orb0_out(file_nn_orb0_out.c_str());

	string file_nn_orb1_out="Nd_corr_vs_d_orb1.txt";
	ofstream file_nin0_orb1_out(file_nn_orb1_out.c_str());

	string file_nn_full_out="Nd_corr_vs_d_tot.txt";
	ofstream file_nin0_full_out(file_nn_full_out.c_str());

	string file_ttz_full_out="Tzd_corr_vs_d_tot.txt";
	ofstream file_tzitz0_full_out(file_ttz_full_out.c_str());


	double Avg_n_orb0, Avg_n_orb1;
	Avg_n_orb0 = 0.0;
	Avg_n_orb1 = 0.0;

	int max_dist=Lx_+Ly_-2;
	Mat_1_int Nlinks_at_dist;
	int ix,iy,jx,jy;
	int links;

	for(int d=0;d<=max_dist;d++){
		links=0;
		for(int i=0;i<Size;i++){
			ix=i/Ly_;	iy=i%Ly_;
			for(int j=i;j<Size;j++){
				jx=j/Ly_;	jy=j%Ly_;
				if(abs(ix-jx)+abs(iy-jy)==d){
					links++;
				}
			}
		}
		Nlinks_at_dist.push_back(links);
	}

	complex<double> nn_orb0_val,nn_orb1_val,nn_tot_val,tt_tot_val;
	for(int d=0;d<=max_dist;d++){
		nn_orb0_val.real(0.0);	nn_orb0_val.imag(0.0);
		nn_orb1_val.real(0.0);	nn_orb1_val.imag(0.0);
		nn_tot_val.real(0.0);	nn_tot_val.imag(0.0);
		tt_tot_val.real(0.0);	tt_tot_val.imag(0.0);
		for(int i=0;i<Size;i++){
			for(int j=i;j<Size;j++){
				if( (abs((i-j)%Ly_) + abs((i-j)/Ly_) )==d){
					nn_orb0_val += (1.0/(1.0*Nlinks_at_dist[d]))*nn_00_mat[i][j];
					nn_orb1_val += (1.0/(1.0*Nlinks_at_dist[d]))*nn_11_mat[i][j];
					nn_tot_val += (1.0/(1.0*Nlinks_at_dist[d]))*nn_tot_mat[i][j];
					tt_tot_val += (1.0/(1.0*Nlinks_at_dist[d]))*tztz_corr_mat[i][j];
				}
			}
		}
		file_nin0_orb0_out<<d<<"	"<<nn_orb0_val.real()<<endl;
		file_nin0_orb1_out<<d<<"	"<<nn_orb1_val.real()<<endl;
		file_nin0_full_out<<d<<"	"<<nn_tot_val.real()<<endl;
		file_tzitz0_full_out<<d<<"	"<<tt_tot_val.real()<<endl;
	}

	string file_ni_orb0_="ni_vs_i_orb0.txt";
	ofstream file_ni_orb0_out(file_ni_orb0_.c_str());

	string file_ni_orb1_="ni_vs_i_orb1.txt";
	ofstream file_ni_orb1_out(file_ni_orb1_.c_str());

	string file_nrx_orb0_="Avg_n_vs_rx_orb0.txt";
	ofstream file_nrx_orb0_out(file_nrx_orb0_.c_str());

	string file_nrx_orb1_="Avg_n_vs_rx_orb1.txt";
	ofstream file_nrx_orb1_out(file_nrx_orb1_.c_str());

	double avg_bulk_val0,avg_bulk_val1;	
	int rd;
	avg_bulk_val0=0.0;
	avg_bulk_val1=0.0;
	if(Size>16){
		for(int rx=Lx_/2-1;rx<=Lx_/2;rx++){
			for(int ry=0;ry<Ly_;ry++){
				rd=ry+Ly_*rx;
				avg_bulk_val0 += (1.0/8.0)*(n_0up[rd]+n_0dn[rd]);
				avg_bulk_val1 += (1.0/8.0)*(n_1up[rd]+n_1dn[rd]);
			}
		}
	}
	else{
		for(int rx=1;rx<Lx_-1;rx++){
			for(int ry=0;ry<Ly_;ry++){
				rd=ry+Ly_*rx;
				avg_bulk_val0 += (1.0/(1.0*(Size-8)))*(n_0up[rd]+n_0dn[rd]);
				avg_bulk_val1 += (1.0/(1.0*(Size-8)))*(n_1up[rd]+n_1dn[rd]);
			}
		}
	}

	double val0,val1;
	for(int rx=0;rx<Lx_;rx++){
		val0=0.0;	val1=0.0;
		for(int ry=0;ry<Ly_;ry++){
			rd=ry+Ly_*rx;
			val0 += (1.0/(1.0*Ly_))*(n_0up[rd]+n_0dn[rd]);
			val1 += (1.0/(1.0*Ly_))*(n_1up[rd]+n_1dn[rd]);
		}
		file_nrx_orb0_out<<rx<<"		"<<val0-avg_bulk_val0<<endl;
		file_nrx_orb1_out<<rx<<"		"<<val1-avg_bulk_val1<<endl;
	}

	string file_out_nAvg="Occupation_number_file.txt";
	ofstream file_nAvg_out(file_out_nAvg.c_str());

	for(int i=0;i<Size;i++){
		Avg_n_orb0 += (1.0/(1.0*Size))*(n_0up[i]+n_0dn[i]);
		Avg_n_orb1 += (1.0/(1.0*Size))*(n_1up[i]+n_1dn[i]);

		file_ni_orb0_out<<i<<"	"<<(n_0up[i]+n_0dn[i]) - avg_bulk_val0<<endl;
		file_ni_orb1_out<<i<<"	"<<(n_1up[i]+n_1dn[i]) - avg_bulk_val1<<endl;
	}

	file_nAvg_out<<"Avg_occ_for_orb-0= "<<Avg_n_orb0<<endl;
	file_nAvg_out<<"Avg_occ_for_orb-1= "<<Avg_n_orb1<<endl;

	double Avg_Charge_Fluc0,Avg_Charge_Fluc1,Avg_doub_occ0,Avg_doub_occ1;
	Avg_Charge_Fluc0=0.0;	Avg_Charge_Fluc1=0.0;
	Avg_doub_occ0=0.0;	Avg_doub_occ1=0.0;
	for(int i=0;i<Size;i++){
		Avg_Charge_Fluc0 += (1.0/(1.0*Size))*( nn_00_mat[i][i].real() );
		Avg_Charge_Fluc1 += (1.0/(1.0*Size))*( nn_11_mat[i][i].real() );

		Avg_doub_occ0 += (1.0/(1.0*Size))*( nn_0up_0dn[i][i].real() );
		Avg_doub_occ1 += (1.0/(1.0*Size))*( nn_1up_1dn[i][i].real() );
	}

	file_nAvg_out<<"Charge_fluctuations_for_orb-0= "<<Avg_Charge_Fluc0<<endl;
	file_nAvg_out<<"Charge_fluctuations_for_orb-1= "<<Avg_Charge_Fluc1<<endl;

	file_nAvg_out<<"Double_occupancy_for_orb-0= "<<Avg_doub_occ0<<endl;
	file_nAvg_out<<"Double_occupancy_for_orb-1= "<<Avg_doub_occ1<<endl;

	//--------------------------------------------------------------------------//
	//Generating Spin-Spin correlations data files:------->
	string file_SS_orb0_out_="Sd_corr_vs_d_orb0.txt";
	ofstream file_SS_orb0_out(file_SS_orb0_out_.c_str());

	string file_SS_orb1_out_="Sd_corr_vs_d_orb1.txt";
	ofstream file_SS_orb1_out(file_SS_orb1_out_.c_str());

	string file_SS_full_out_="Sd_corr_vs_d_tot.txt";
	ofstream file_SS_full_out(file_SS_full_out_.c_str());


	string file_S0Si_orb0_out_="S0Si_vs_i_orb0.txt";
	ofstream file_S0Si_orb0_out(file_S0Si_orb0_out_.c_str());

	string file_S0Si_orb1_out_="S0Si_vs_i_orb1.txt";
	ofstream file_S0Si_orb1_out(file_S0Si_orb1_out_.c_str());

	string file_S0Si_full_out_="S0Si_vs_i_tot.txt";
	ofstream file_S0Si_full_out(file_S0Si_full_out_.c_str());

	int bulk_site=4;
	for(int i=bulk_site;i<Size;i++){
		file_S0Si_orb0_out<<i-bulk_site<<"	"<<SS_orb0_mat[bulk_site][i].real()<<endl;
		file_S0Si_orb1_out<<i-bulk_site<<"	"<<SS_orb1_mat[bulk_site][i].real()<<endl;
		file_S0Si_full_out<<i-bulk_site<<"	"<<SS_tot_mat[bulk_site][i].real()<<endl;
	}



	complex<double> SS_orb0_val,SS_orb1_val,SS_tot_val;
	for(int d=0;d<=max_dist;d++){
		SS_orb0_val=Zero_Complex;	SS_orb1_val=Zero_Complex;	SS_tot_val=Zero_Complex;
		for(int i=0;i<Size;i++){
			for(int j=i;j<Size;j++){
				if( (abs((i-j)%Ly_) + abs((i-j)/Ly_) )==d){
					SS_orb0_val += (1.0/(1.0*Nlinks_at_dist[d]))*SS_orb0_mat[i][j];
					SS_orb1_val += (1.0/(1.0*Nlinks_at_dist[d]))*SS_orb1_mat[i][j];
					SS_tot_val += (1.0/(1.0*Nlinks_at_dist[d]))*SS_tot_mat[i][j];
				}
			}
		}
		file_SS_orb0_out<<d<<"	"<<SS_orb0_val.real()<<endl;
		file_SS_orb1_out<<d<<"	"<<SS_orb1_val.real()<<endl;
		file_SS_full_out<<d<<"	"<<SS_tot_val.real()<<endl;
	}

	complex<double> S2_orb0, S2_orb1, S2_tot, Local_S2_Avg_orb0, Local_S2_Avg_orb1, Local_S2_Avg_tot;
	S2_orb0 = Zero_Complex;	Local_S2_Avg_orb0=Zero_Complex;
	S2_orb1 = Zero_Complex;	Local_S2_Avg_orb1=Zero_Complex;
	S2_tot = Zero_Complex;	Local_S2_Avg_tot=Zero_Complex;

	complex<double> Local_S2_orb0_edge, Local_S2_orb1_edge, Local_S2_orb0_bulk, Local_S2_orb1_bulk;
	Local_S2_orb0_edge = Zero_Complex; Local_S2_orb1_edge = Zero_Complex;
	Local_S2_orb0_bulk = Zero_Complex; Local_S2_orb1_bulk = Zero_Complex;

	complex<double> Local_S2_tot_edge, Local_S2_tot_bulk;
	Local_S2_tot_edge=Zero_Complex;	Local_S2_tot_bulk=Zero_Complex;

	for(int i=0;i<Size;i++){
		for(int j=0;j<Size;j++){
			S2_orb0 += SS_orb0_mat[i][j];
			S2_orb1 += SS_orb1_mat[i][j];
			S2_tot  += SS_tot_mat[i][j];
		}
		Local_S2_Avg_orb0 += (1.0/(1.0*Size))*SS_orb0_mat[i][i];
		Local_S2_Avg_orb1 += (1.0/(1.0*Size))*SS_orb1_mat[i][i];
		Local_S2_Avg_tot  += (1.0/(1.0*Size))*SS_tot_mat[i][i];

		if(i<4 || i>=Size-4){
			Local_S2_tot_edge += (1.0/8.0)*SS_tot_mat[i][i];
			//			Local_S2_orb1_edge += (1.0/8.0)*SS_orb1_mat[i][i];
		}
		else {
			Local_S2_tot_bulk += (1.0/(1.0*(Size-8)))*SS_tot_mat[i][i];
			//			Local_S2_orb1_bulk += (1.0/(1.0*(Size-8)))*SS_orb1_mat[i][i];
		}
	}

	string file_out_S2="S2_file.txt";
	ofstream file_s2_out(file_out_S2.c_str());

	file_s2_out<<"Total_S^2_for_orb-0= "<<S2_orb0.real()<<endl;
	file_s2_out<<"Total_S^2_for_orb-1= "<<S2_orb1.real()<<endl;
	file_s2_out<<"Total_S^2_of_the_system= "<<S2_tot.real()<<endl;
	file_s2_out<<"Avg_local_S^2_for_orb-0= "<<Local_S2_Avg_orb0.real()<<endl;
	file_s2_out<<"Avg_local_S^2_for_orb-1= "<<Local_S2_Avg_orb1.real()<<endl;
	file_s2_out<<"Avg_local_S^2_of_the_system= "<<Local_S2_Avg_tot.real()<<endl;
	//	file_s2_out<<"Avg_local_S^2_tot_on_edge= "<<Local_S2_orb0_edge.real()+Local_S2_orb1_edge.real()<<endl;
	//	file_s2_out<<"Avg_local_S^2_tot_in_bulk= "<<Local_S2_orb0_bulk.real()+Local_S2_orb1_bulk.real()<<endl;
	file_s2_out<<"Avg_local_S^2_tot_on_edge= "<<Local_S2_tot_edge.real()<<endl;
	file_s2_out<<"Avg_local_S^2_tot_in_bulk= "<<Local_S2_tot_bulk.real()<<endl;

	//--------------------------------------------------------------------------//
	//Calculating the structure factor S(qx,qy):---->
	Mat_2_Complex_doub Sk_ee, Sk_bb, Sk_eb;
	Sk_ee.resize(Lx_); Sk_bb.resize(Lx_); Sk_eb.resize(Lx_);

	Mat_2_Complex_doub Sk_;
	Sk_.resize(Lx_);
	for(int ix=0;ix<Lx_;ix++){
		Sk_ee[ix].resize(Ly_);
		Sk_bb[ix].resize(Ly_);
		Sk_eb[ix].resize(Ly_);

		Sk_[ix].resize(Ly_);
		for(int iy=0;iy<Ly_;iy++){
			Sk_ee[ix][iy] = Zero_Complex;
			Sk_bb[ix][iy] = Zero_Complex;
			Sk_eb[ix][iy] = Zero_Complex;

			Sk_[ix][iy]=Zero_Complex;
		}
	}

	string file_out_Sq="Sq_heatmap_file.txt";
	ofstream file_sq_out(file_out_Sq.c_str());

	int Edge_Size,Bulk_Size;
	Edge_Size=8;
	Bulk_Size=Size-8;

	double kx,ky;
	int r,rp;
	for(int kx_ind=0;kx_ind<Lx_;kx_ind++){
		kx=(2.0*PI*kx_ind)/(1.0*Lx_);
		for(int ky_ind=0;ky_ind<Ly_;ky_ind++){
			ky=(2.0*PI*ky_ind)/(1.0*Ly_);

			for(int rx=0;rx<Lx_;rx++){
				for(int ry=0;ry<Ly_;ry++){
					r=ry+Ly_*rx;
					for(int rxp=0;rxp<Lx_;rxp++){
						for(int ryp=0;ryp<Ly_;ryp++){
							rp=ryp+Ly_*rxp;

							Sk_[kx_ind][ky_ind] += (1.0/(1.0*Size*Size))*exp( Iota_Complex*1.0*(kx*1.0*(rx-rxp)+ky*1.0*(ry-ryp)) )*SS_tot_mat[r][rp];
							if(rx==0 || rx==Lx_-1){
								if(rxp==0 || rxp==Lx_-1){
									Sk_ee[kx_ind][ky_ind] += (1.0/(Edge_Size*Edge_Size)*1.0)*
										exp( Iota_Complex*1.0*(kx*1.0*(rx-rxp)+ky*1.0*(ry-ryp)) )*SS_tot_mat[r][rp];
								}
								if(rxp!=0 && rxp!=Lx_-1){
									Sk_eb[kx_ind][ky_ind] += (1.0/(Edge_Size*Bulk_Size)*1.0)*
										exp( Iota_Complex*1.0*(kx*1.0*(rx-rxp)+ky*1.0*(ry-ryp)) )*SS_tot_mat[r][rp];
								}
							}
							else {
								if(rxp!=0 && rxp!=Lx_-1){
									Sk_bb[kx_ind][ky_ind] += (1.0/(Bulk_Size*Bulk_Size)*1.0)*
										exp( Iota_Complex*1.0*(kx*1.0*(rx-rxp)+ky*1.0*(ry-ryp)) )*SS_tot_mat[r][rp];
									//									cout<<"(rx,ry)= ("<<rx<<","<<ry<<") and (rxp,ryp)= ("<<rxp<<","<<ryp<<")"<<endl;
								}
							}
						}
					}
				}
			}
			file_sq_out<<kx<<"	"<<ky<<"	"<<Sk_[kx_ind][ky_ind].real()<<endl;
		}
		file_sq_out<<endl;
	}

	file_s2_out<<"Sk[0][0]= "<<Sk_[0][0].real()<<endl;
	file_s2_out<<"Sk[0][pi]= "<<Sk_[0][Ly_/2].real()<<endl;
	file_s2_out<<"Sk[pi][0]= "<<Sk_[Lx_/2][0].real()<<endl;
	file_s2_out<<"Sk[pi][pi]= "<<Sk_[Lx_/2][Ly_/2].real()<<endl;

	file_s2_out<<"Sk_ee[0][0]= "<<Sk_ee[0][0].real()<<endl;
	file_s2_out<<"Sk_ee[0][pi]= "<<Sk_ee[0][Ly_/2].real()<<endl;
	file_s2_out<<"Sk_ee[pi][0]= "<<Sk_ee[Lx_/2][0].real()<<endl;
	file_s2_out<<"Sk_ee[pi][pi]= "<<Sk_ee[Lx_/2][Ly_/2].real()<<endl;

	file_s2_out<<"Sk_bb[0][0]= "<<Sk_bb[0][0].real()<<endl;
	file_s2_out<<"Sk_bb[0][pi]= "<<Sk_bb[0][Ly_/2].real()<<endl;
	file_s2_out<<"Sk_bb[pi][0]= "<<Sk_bb[Lx_/2][0].real()<<endl;
	file_s2_out<<"Sk_bb[pi][pi]= "<<Sk_bb[Lx_/2][Ly_/2].real()<<endl;

	file_s2_out<<"Sk_eb[0][0]= "<<Sk_eb[0][0].real()<<endl;
	file_s2_out<<"Sk_eb[0][pi]= "<<Sk_eb[0][Ly_/2].real()<<endl;
	file_s2_out<<"Sk_eb[pi][0]= "<<Sk_eb[Lx_/2][0].real()<<endl;
	file_s2_out<<"Sk_eb[pi][pi]= "<<Sk_eb[Lx_/2][Ly_/2].real()<<endl;
	//--------------------------------------------------------------------------//

	return 0;
}
