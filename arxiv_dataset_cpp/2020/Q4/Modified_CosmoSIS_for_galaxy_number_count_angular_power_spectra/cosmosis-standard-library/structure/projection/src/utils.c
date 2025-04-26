#include "utils.h"
#include <assert.h>
#include "stdio.h"
#include "string.h"
#include "cosmosis/datablock/c_datablock.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

gsl_spline * spline_from_arrays(int n, double * x, double *y)
{
	gsl_spline * output = gsl_spline_alloc(gsl_interp_akima, n);
	assert (output!=NULL);
	gsl_spline_init(output,x,y,n);
	return output;
}

DATABLOCK_STATUS save_spline(c_datablock * block, const char * section, 
	const char * n_name, const char * x_name, const char * y_name,
	gsl_spline * s)
{

	DATABLOCK_STATUS status = 0;
	if (strlen(n_name)>0){
		status |= c_datablock_put_int(block, section, n_name, s->size);
	}
	if (strlen(x_name)>0){
		status |= c_datablock_put_double_array_1d(block, section, x_name, 
			s->x, s->size);
	}
	status |= c_datablock_put_double_array_1d(block, section, y_name, 
		s->y, s->size);
	return status;
}

void reverse(double * x, int n)
{
	for (int i=0; i<n/2; i++){
		double tmp = x[i];
		x[i] = x[n-1-i];
		x[n-1-i] = tmp;
	}
}

static 
double identity_function(double x, double y, double z, void * a)
{
  return z;
}



Interpolator2D * 
load_interpolator_chi_function(c_datablock * block, gsl_spline * chi_of_z_spline, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	int nk, nz, nP;
	double *k=NULL, *z=NULL;
	double **P=NULL;
	int status = 0;

	status = c_datablock_get_double_grid(block, section, 
		k_name, &nk, &k, 
		z_name, &nz, &z, 
		P_name, &P);

	if (status){
		fprintf(stderr, "Could not load interpolator for P(k).  Error %d\n",status);
		return NULL;
	}

        //read only P(k,z) and not P(k,0) since the growth factor will be inside the kernels
	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			P[j][i] = function(k[j], z[0], P[j][0], args);
		}
	}


	// What we have now is P(k, z).
	// We can optionally convert to P(k, chi)
	// If so we loop, lookup, and replace
	if (chi_of_z_spline){
		for (int i=0; i<nz; i++){
			double zi = z[i];
			double chi_i = gsl_spline_eval(chi_of_z_spline, zi, NULL);
			z[i] = chi_i;
		}
	}

	if (status) return NULL;
	Interpolator2D * interp = init_interp_2d_akima_grid(k, z, P, nk, nz);
	deallocate_2d_double(&P, nk);
	free(k);
	free(z);
	return interp;
}


Interpolator2D * 
load_interpolator_chi(c_datablock * block, gsl_spline * chi_of_z_spline, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{

	return load_interpolator_chi_function(block, chi_of_z_spline, section, k_name, z_name, P_name, identity_function, NULL);
}


Interpolator2D * 
load_interpolator_function(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	return load_interpolator_chi_function(block, NULL, section, k_name, z_name, P_name, function, args);

}


Interpolator2D * 
load_interpolator(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{
	return load_interpolator_chi_function(block, NULL, section, k_name, z_name, P_name, identity_function, NULL);
}



// this is for the scale dependent growth rate from here----------------

Interpolator2D * 
load_interpolator_chi_function2(c_datablock * block, gsl_spline * chi_of_z_spline,  gsl_spline * z_of_chi_spline,
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	int nk, nz, nT;
	double *k=NULL, *z=NULL;
	double **P=NULL;
	int status = 0;

	status = c_datablock_get_double_grid(block, section, 
		k_name, &nk, &k, 
		z_name, &nz, &z, 
		P_name, &P);

	double a[nk][nz],*pp[nk],**dTdz=NULL;
        for(int j=0;j<nk;j++){
	pp[j]=&a[j][0];}//for the derivative
	dTdz=pp;

	double b[nk][nz],*ppp[nk],**f=NULL;
        for(int j=0;j<nk;j++){
	ppp[j]=&b[j][0];}//for f(k,z)
	f=ppp;

	double c[nk][nz],*pppp[nk],**T=NULL;
        for(int j=0;j<nk;j++){
	pppp[j]=&c[j][0];}//for the T(k,z)
	T=pppp;


	if (status){
		fprintf(stderr, "Could not load interpolator for T(k).  Error %d\n",status);
		return NULL;
	}

	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			P[j][i] = function(k[j], z[i], P[j][i], args);
		}
	}

	//this is to take T(k,z)
	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			T[j][i] = sqrt(P[j][i]/P[j][0]);
		}
	}



        //derivative dT/dz

	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){

   	        	
   	        //forward differences
   	        	
   		if(i==0){
   			dTdz[j][i] = (T[j][i+1]-T[j][i])/(z[i+1]-z[i]);
			}

		//central differences
	
		if(0<i && i<(nz-1)){
   			dTdz[j][i] = (T[j][i+1]-T[j][i-1])/(2.*(z[i+1]-z[i]));
			}	   
		//backward differences
					   
		if(i==(nz-1)){
   			dTdz[j][i] = (T[j][i]-T[j][i-1])/(z[i]-z[i-1]);
			}


		}
	}

	//printf("ok with derivative\n");

	//this is to take f(k,z)
	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			f[j][i] = -((z[i]+1.)/T[j][i])*dTdz[j][i];
		}
	}

	// What we have now is f(k, z).
	// We can optionally convert to f(k, z(chi))
	// If so we loop, lookup, and replace
	if (chi_of_z_spline){
		for (int i=0; i<nz; i++){
                        //int j=0; //my modification
			double zi = z[i]; //// j instead of i
			double chi_i = gsl_spline_eval(chi_of_z_spline, zi, NULL);
			//double z_i = gsl_spline_eval(z_of_chi_spline, chi_i, NULL);
			z[i] = chi_i;
		}
	}

	Interpolator2D * interp2 = init_interp_2d_akima_grid(k, z, f, nk, nz);
	return interp2;
}


Interpolator2D * 
load_interpolator_chi2(c_datablock * block, gsl_spline * chi_of_z_spline, gsl_spline * z_of_chi_spline,
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{

	return load_interpolator_chi_function2(block, chi_of_z_spline, z_of_chi_spline, section, k_name, z_name, P_name, identity_function, NULL);
}


Interpolator2D * 
load_interpolator_function2(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	return load_interpolator_chi_function2(block, NULL, NULL, section, k_name, z_name, P_name, function, args);

}


Interpolator2D * 
load_interpolator2(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{
	return load_interpolator_chi_function2(block, NULL, NULL, section, k_name, z_name, P_name, identity_function, NULL);
}

// to here---------------------------------------

// here write the scale dependent growth factor

Interpolator2D * 
load_interpolator_chi_function3(c_datablock * block, gsl_spline * chi_of_z_spline,  gsl_spline * z_of_chi_spline,
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	int nk, nz, nT;
	double *k=NULL, *z=NULL;
	double **P=NULL;
	int status = 0;

	status = c_datablock_get_double_grid(block, section, 
		k_name, &nk, &k, 
		z_name, &nz, &z, 
		P_name, &P);

	double b[nk][nz],*ppp[nk],**D=NULL;
        for(int j=0;j<nk;j++){
	ppp[j]=&b[j][0];}//for D(k,z)
	D=ppp;




	if (status){
		fprintf(stderr, "Could not load interpolator for P_nl(k).  Error %d\n",status);
		return NULL;
	}

	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			P[j][i] = function(k[j], z[i], P[j][i], args);
		}
	}

	//printf("ok with derivative\n");

	//this is to take D(k,z)
	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			D[j][i] = sqrt(P[j][i]/P[j][0]);
		}
	}




	// What we have now is D(k, z).
	// We can optionally convert to D(k, z(chi))
	// If so we loop, lookup, and replace
	if (chi_of_z_spline){
		for (int i=0; i<nz; i++){
                        //int j=0; //my modification
			double zi = z[i]; //// j instead of i
			double chi_i = gsl_spline_eval(chi_of_z_spline, zi, NULL);
			//double z_i = gsl_spline_eval(z_of_chi_spline, chi_i, NULL);
			z[i] = chi_i;
		}
	}

	//printf("ok with z_of_chi\n");
	Interpolator2D * interp3 = init_interp_2d_akima_grid(k, z, D, nk, nz);
	return interp3;
}


Interpolator2D * 
load_interpolator_chi3(c_datablock * block, gsl_spline * chi_of_z_spline, gsl_spline * z_of_chi_spline,
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{

	return load_interpolator_chi_function3(block, chi_of_z_spline, z_of_chi_spline, section, k_name, z_name, P_name, identity_function, NULL);
}


Interpolator2D * 
load_interpolator_function3(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	return load_interpolator_chi_function3(block, NULL, NULL, section, k_name, z_name, P_name, function, args);

}


Interpolator2D * 
load_interpolator3(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{
	return load_interpolator_chi_function3(block, NULL, NULL, section, k_name, z_name, P_name, identity_function, NULL);
}

//-------------------to here





//-----------------here is the scale dependent galaxy bias accounting for neutrinos


Interpolator2D * 
load_interpolator_chi_function4(c_datablock * block, gsl_spline * chi_of_z_spline,  gsl_spline * z_of_chi_spline,
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	int nk, nz, nT;
	double *k=NULL, *z=NULL;
	double **P=NULL;
	int status = 0;

	status = c_datablock_get_double_grid(block, section, 
		k_name, &nk, &k, 
		z_name, &nz, &z, 
		P_name, &P);

	double bb[nk][nz],*pppp[nk],**Bias=NULL;
        for(int j=0;j<nk;j++){
	pppp[j]=&bb[j][0];}//for b(k,z)
	Bias=pppp;


	double bbL[nk][nz],*ppppL[nk],**bL=NULL;
        for(int j=0;j<nk;j++){
	ppppL[j]=&bbL[j][0];}//for bL(k,z)
	bL=ppppL;

	double bbS[nk][nz],*ppppS[nk],**bS=NULL;
        for(int j=0;j<nk;j++){
	ppppS[j]=&bbS[j][0];}//for bS(k,z)
	bS=ppppS;


	if (status){
		fprintf(stderr, "Could not load interpolator for P_nl(k).  Error %d\n",status);
		return NULL;
	}

	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){
			P[j][i] = function(k[j], z[i], P[j][i], args);
		}
	}


  	int status7=0;
	double omega_m;
	status7 = c_datablock_get_double(block, COSMOLOGICAL_PARAMETERS_SECTION, "omega_m", &omega_m);
	double Om = omega_m;


  	int status8=0;
	double omega_vh2;
	status8 = c_datablock_get_double(block, COSMOLOGICAL_PARAMETERS_SECTION, "omnuh2", &omega_vh2);
	double Ovh2 = omega_vh2;


  	int status9=0;
	double hs;
	status9 = c_datablock_get_double(block, COSMOLOGICAL_PARAMETERS_SECTION, "h0", &hs);
	double hubble = hs;

        double fv = (Ovh2/(hubble * hubble))/Om;

        double mv = Ovh2 * 93.14; //this is Sum(mv)/1eV in h/Mpc but for one massive neutrino only Sum(mv)=mv


        // this the neutrino free streaming scale in h/Mpc
	//for (int i=0; i<nz; i++){
	//	kfs[0][i] = 0.82 * sqrt(1.0 - Om + Om * (1.0 + z[i]) * (1.0 + z[i]) * (1.0 + z[i]))/((1.0 + z[i]) * (1.0 + z[i])) * mv;
	//}  
        double knr;
        knr = 0.018 * sqrt(Om) * sqrt(mv);      

        double constant=1.0;
	//this is to take b(k,z)
	for (int j=0; j<nk; j++){
		for (int i=0; i<nz; i++){

                bL[j][i] =  constant/sqrt(P[0][i]/P[0][0]);


                bS[j][i] = constant/sqrt(P[0][i]/P[0][0]) - constant/sqrt(P[0][i]/P[0][0]) * fv;

                Bias[j][i] = bL[j][i] + (bS[j][i] - bL[j][i]) * 0.5 * (tanh(log(pow(k[j]/(knr),5.0)))+1.0) ;

		}
	}


	// What we have now is b(k, z).
	// We can optionally convert to b(k, z(chi))
	// If so we loop, lookup, and replace
	if (chi_of_z_spline){
		for (int i=0; i<nz; i++){
                        //int j=0; //my modification
			double zi = z[i]; //// j instead of i
			double chi_i = gsl_spline_eval(chi_of_z_spline, zi, NULL);
			//double z_i = gsl_spline_eval(z_of_chi_spline, chi_i, NULL);
			z[i] = chi_i;
		}
	}

	//printf("ok with z_of_chi\n");
	Interpolator2D * interp4 = init_interp_2d_akima_grid(k, z, Bias, nk, nz);
	return interp4;
}


Interpolator2D * 
load_interpolator_chi4(c_datablock * block, gsl_spline * chi_of_z_spline, gsl_spline * z_of_chi_spline,
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{

	return load_interpolator_chi_function4(block, chi_of_z_spline, z_of_chi_spline, section, k_name, z_name, P_name, identity_function, NULL);
}


Interpolator2D * 
load_interpolator_function4(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	interp2d_modifier_function function, void * args
	)
{
	return load_interpolator_chi_function4(block, NULL, NULL, section, k_name, z_name, P_name, function, args);

}


Interpolator2D * 
load_interpolator4(c_datablock * block, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name)
{
	return load_interpolator_chi_function4(block, NULL, NULL, section, k_name, z_name, P_name, identity_function, NULL);
}



//------------------to here








gsl_spline * load_spline(c_datablock * block, const char * section, 
	const char * x_name, const char * y_name)
{
	int status = 0 ;
	int nx, ny;
	double *x, *y;
	status |= c_datablock_get_double_array_1d(block, section, x_name, &x, &nx);
	if (status) return NULL;
	status |= c_datablock_get_double_array_1d(block, section, y_name, &y, &ny);
	if (status){
		free(x);
		return NULL;
	}

	gsl_spline * output = NULL;

	if (x[1]<x[0]) {reverse(x, nx); reverse(y, ny);}


	if (nx==ny) output=spline_from_arrays(nx, x, y);
	else {fprintf(stderr, "Non-matching array sizes %s=%d, %s=%d\n", x_name, nx, y_name, ny);}
	free(x);
	free(y);
	return output;
}

/*
gsl_spline2d *
load_gsl_interpolator_chi(c_datablock * block, gsl_spline * chi_of_z_spline, 
	const char * section,
	const char * k_name, const char * z_name, const char * P_name,
	)
{
	int nk, nz, nP;
	double *k=NULL, *z=NULL;
	double **P=NULL;
	int status = 0;

	status = c_datablock_get_double_grid(block, section, 
		k_name, &nk, &k, 
		z_name, &nz, &z, 
		P_name, &P);

	if (status){
		fprintf(stderr, "Could not load interpolator for P(k).  Error %d\n",status);
		return NULL;
	}

	// What we have now is P(k, z).
	// We can optionally convert to P(k, chi)
	// If so we loop, lookup, and replace
	if (chi_of_z_spline){
		for (int i=0; i<nz; i++){
			double zi = z[i];
			double chi_i = gsl_spline_eval(chi_of_z_spline, zi, NULL);
			z[i] = chi_i;
		}
	}

	if (status) return NULL;
        const gsl_interp2d_type *T = gsl_interp2d_bicubic;
	gsl_spline2d *spline = gsl_spline2d_alloc(T, nz, nk);
	gsl_interp_accel *xacc = gsl_interp_accel_alloc();
	gsl_interp_accel *yacc = gsl_interp_accel_alloc();
	gsl_spline_init

	Interpolator2D * interp = init_interp_2d_akima_grid(k, z, P, nk, nz);
	deallocate_2d_double(&P, nk);
	free(k);
	free(z);
	return interp;
}
*/
