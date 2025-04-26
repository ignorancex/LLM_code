#include "utility.h"

double interpolate2d(	boost::multi_array<double, 2> * A, 
					 	const int& ni, const int& nj, 
					 	const double& ri, const double& rj)
{
	double wi[2] = {1.-ri, ri}, wj[2] = {1.-rj, rj};
	double result = 0.;
	for (int i=0; i<2; i++){
		for (int j=0; j<2; j++){
			result += (*A)[ni+i][nj+j]*wi[i]*wj[j];
		}
	}
	return result;
}



double interpolate2d_YX(boost::multi_array<double, 3> * A, const int& index, 
					 	const int& ni, const int& nj, 
					 	const double& ri, const double& rj)
{
	double wi[2] = {1.-ri, ri}, wj[2] = {1.-rj, rj};
	double result = 0.;
	for (int i=0; i<2; i++){
		for (int j=0; j<2; j++){
			result += (*A)[index][ni+i][nj+j]*wi[i]*wj[j];
		}
	}
	return result;
}

double interpolate3d(	boost::multi_array<double, 3> * A, 
						const int& ni, const int& nj, const int& nk,
						const double& ri, const double& rj, const double& rk)
{
	double wi[2] = {1.-ri, ri}, wj[2] = {1.-rj, rj}, wk[2] = {1.-rk, rk};
	double result = 0.;
	for (int i=0; i<2; i++){
		for (int j=0; j<2; j++){
			for (int k=0; k<2; k++){
				result += (*A)[ni+i][nj+j][nk+k]*wi[i]*wj[j]*wk[k];
			}
		}
	}
	return result;
}

double interpolate4d(	boost::multi_array<double, 4> * A, 
						const int& ni, const int& nj, const int& nk, const int& nt, 
						const double& ri, const double& rj, const double& rk, const double& rt)
{
	double wi[2] = {1.-ri, ri}, wj[2] = {1.-rj, rj}, wk[2] = {1.-rk, rk}, wt[3] = {1.-rt, rt};
	double result = 0.;
	for (int i=0; i<2; i++){
		for (int j=0; j<2; j++){
			for (int k=0; k<2; k++){
				for (int t=0; t<2; t++){
					result += (*A)[ni+i][nj+j][nk+k][nt+t]*wi[i]*wj[j]*wk[k]*wt[t];
				}
			}
		}
	}
	return result;
}

