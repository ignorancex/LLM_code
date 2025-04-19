#include "cosmosis/datablock/c_datablock.h"
//#include "gsl/gsl_spline.h"
//#include "galcl_galcl.h"
#include "utils.h"
#include <stdio.h>
#include "limber.h"
#include <math.h>
#include "assert.h"
#include "string.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_errno.h"

// Short-hand names for the sections we will
// be looking at
const char * wl_nz = WL_NUMBER_DENSITY_SECTION;
const char * dist = DISTANCES_SECTION;
const char * cosmo = COSMOLOGICAL_PARAMETERS_SECTION;
const char * ia = INTRINSIC_ALIGNMENT_PARAMETERS_SECTION;
const char * lum = GALAXY_LUMINOSITY_FUNCTION_SECTION;
const char * linear_cdm_transfer = LINEAR_CDM_TRANSFER_SECTION;
const char * growthparameters = GROWTH_PARAMETERS_SECTION;

//const char * growth_section = "GROWTH_PARAMETERS"

#define NGLT 2048

//----------
typedef enum spectrum_type_t {
	// Shape-shape
	galcl_galcl,
	galcl_intrinsic,
	intrinsic_intrinsic,

	// position-position
	matter_matter,
	matter_magnification,
	magnification_magnification,

	// position-shape
	matter_galcl,
	matter_intrinsic,
	magnification_intrinsic,
	magnification_galcl
} spectrum_type_t;

typedef struct galcl_spectrum_config {
	int n_ell;
	double ell_min;
	double ell_max;
	bool verbose;
	bool galcl_galcl;
	bool intrinsic_alignments;
	bool matter_spectra;
	bool ggl_spectra;
	bool gal_IA_cross_spectra;
	bool galaxy_magnification;
	bool magnification_magnification;
	bool magnification_intrinsic;
	bool magnification_galcl;
	
} galcl_spectrum_config;


//those are P(k,0), f(k,z) D(k,z) and b(k,z)

int destroy_and_null(Interpolator2D ** P){
	if (*P) destroy_interp_2d(*P);
	*P = NULL;
}


int destroy_and_null2(Interpolator2D ** f){
	if (*f) destroy_interp_2d(*f);
	*f = NULL;
}


int destroy_and_null3(Interpolator2D ** D){
	if (*D) destroy_interp_2d(*D);
	*D = NULL;
}


int destroy_and_null4(Interpolator2D ** BB){
	if (*BB) destroy_interp_2d(*BB);
	*BB = NULL;
}


void * setup(c_datablock * options){

	galcl_spectrum_config * config = malloc(sizeof(galcl_spectrum_config));
	int status = 0;
	status |= c_datablock_get_int(options, OPTION_SECTION, "n_ell", &(config->n_ell));
	status |= c_datablock_get_double(options, OPTION_SECTION, "ell_min", &(config->ell_min));
	status |= c_datablock_get_double(options, OPTION_SECTION, "ell_max", &(config->ell_max));
	// Selectors for different types of angular power spectra to compute
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "verbose", false,
		&(config->verbose));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "galcl_galcl", true,
		&(config->galcl_galcl));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "intrinsic_alignments", false,
		&(config->intrinsic_alignments));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "matter_spectra", false,
		&(config->matter_spectra));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "ggl_spectra", false,
		&(config->ggl_spectra));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "gal_IA_cross_spectra", false,
		&(config->gal_IA_cross_spectra));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "mag_gal_cross_spectra", false,
		&(config->galaxy_magnification));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "mag_mag", false,
			&(config->magnification_magnification));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "mag_IA", false,
			&(config->magnification_intrinsic));
	status |= c_datablock_get_bool_default(options, OPTION_SECTION, "mag_galcl", false,
			&(config->magnification_galcl));

	if (status){
		fprintf(stderr, "Please specify n_ell, ell_min, and ell_max in the galcl spectra module.\n");
		exit(status);
	}

	return config;
}

gsl_spline * linear_spline_from_arrays(int n, double * x, double *y)
{
	gsl_spline * output = gsl_spline_alloc(gsl_interp_linear, n);
	assert (output!=NULL);
	gsl_spline_init(output,x,y,n);
	return output;
}


#define NZ_FORMAT_HISTOGRAM 1
#define NZ_FORMAT_SAMPLE 2

void nz_string_error_message(){
	fprintf(stderr, "\nIf specifying n(z) format when loading, please use one of:\n");
	fprintf(stderr, "format=histogram\n");
	fprintf(stderr, "format=sample\n\n");	
}

// n(z) values come in two general forms.
// Most n(z) codes generate histograms in small z bins, with zlow, zhigh.
// This isn't a great model for n(z) of course - we don't really think that
// the number density is really blocky like that for real.  So we also support
// another mode where we take the z as sample values through which a spline should
// be drawn.
int detect_nz_format(c_datablock * block, const char * section)
{
	char *nz_format_string=NULL;
	int nz_format;
	int status = 0;
	bool has_format = c_datablock_has_value(block, section, "format");
	if (has_format){
		status |= c_datablock_get_string(block, section, "format", &nz_format_string);
		if (nz_format_string==NULL){
			nz_string_error_message();
			status = 1;
		}
		if (!strcmp(nz_format_string, "histogram")){
			nz_format = NZ_FORMAT_HISTOGRAM;
		}
		else if (!strcmp(nz_format_string, "sample")){
			nz_format = NZ_FORMAT_SAMPLE;
		}
		else{
			nz_string_error_message();
			status = 1;
		}

		free(nz_format_string);
		if (status){
			return -1;
		}
	}
	else{
		nz_format = NZ_FORMAT_SAMPLE;	
	}
	return nz_format;
}


//----------


typedef struct w_integrand_data{
  gsl_spline * n_of_z;
  gsl_spline * a_of_chi;
  //gsl_spline * b_of_z; //new
  //gsl_spline * D_of_z; //new2
  gsl_interp_accel * acc_chi;
  gsl_interp_accel * acc_z;
  double chi;
  double zmax;
  double chi_max;
} w_integrand_data;

static double n_of_chi(double chi, void *p)
{
  w_integrand_data * data = (w_integrand_data*) p;
  if (chi>data->chi_max) return 0.0;
  // Get a(chi), da/dchi(chi) and z(chi)
  double a = gsl_spline_eval(data->a_of_chi, chi, data->acc_chi);
  double da_dchi = gsl_spline_eval_deriv(data->a_of_chi, chi, data->acc_chi);
  double z = 1.0/a - 1.0;
  if (z>=data->zmax) {
    // short circuit this in future
    data->chi_max = chi;
    return 0.0;
  }
  // Get n(chi) from n(z)
  double nz = gsl_spline_eval(data->n_of_z, z, data->acc_z);
  return -(1+z)*(1+z)*nz*da_dchi;
}


static double b_of_chi(double chi, void *p)
{
	w_integrand_data * data = (w_integrand_data*) p;
	if (chi>data->chi_max) return 0.0;
	// Get a(chi) and z(chi)
	double a = gsl_spline_eval(data->a_of_chi, chi, data->acc_chi);
	double z = 1.0/a - 1.0;
	if (z>=data->zmax) {
		// short circuit this in future
		data->chi_max = chi;
		return 0.0;
	}
	// Get b(z(chi)) from b(z)
	//double bz = gsl_spline_eval(data->b_of_z, z, data->acc_z);
	return sqrt(1.0+z);
}

/*
static double bLRG_of_chi(double chi, void *p)
{
	w_integrand_data * data = (w_integrand_data*) p;
	if (chi>data->chi_max) return 0.0;
	// Get a(chi) and z(chi)
	double a = gsl_spline_eval(data->a_of_chi, chi, data->acc_chi);
	double z = 1.0/a - 1.0;
	if (z>=data->zmax) {
		// short circuit this in future
		data->chi_max = chi;
		return 0.0;
	}
	// Get D(z(chi)) from D(z)
	double dz = gsl_spline_eval(data->D_of_z, z, data->acc_z);
	return 1.7/dz;
}

static double bELG_of_chi(double chi, void *p)
{
	w_integrand_data * data = (w_integrand_data*) p;
	if (chi>data->chi_max) return 0.0;
	// Get a(chi) and z(chi)
	double a = gsl_spline_eval(data->a_of_chi, chi, data->acc_chi);
	double z = 1.0/a - 1.0;
	if (z>=data->zmax) {
		// short circuit this in future
		data->chi_max = chi;
		return 0.0;
	}
	// Get D(z(chi)) from D(z)
	double dz = gsl_spline_eval(data->D_of_z, z, data->acc_z);
	return 0.84/dz;
}
*/
/*
static double D_of_chi(double chi, void *p)
{
	w_integrand_data * data = (w_integrand_data*) p;
	if (chi>data->chi_max) return 0.0;
	// Get a(chi) and z(chi)
	double a = gsl_spline_eval(data->a_of_chi, chi, data->acc_chi);
	double z = 1.0/a - 1.0;
	if (z>=data->zmax) {
		// short circuit this in future
		data->chi_max = chi;
		return 0.0;
	}
	// Get D(z(chi)) from D(z)
	double dz = gsl_spline_eval(data->D_of_z, z, data->acc_z);
	return dz;
}
*/

static double w_of_chi_integrand(double chis, void *p)
{
  w_integrand_data * data = (w_integrand_data*) p;
  double nchi = n_of_chi(chis, data);
  return nchi*(chis - data->chi)/chis;
}


//this has only n(z) but will be used as the density kernel (to be mutiplied with D(k,z)*b(k,z) )
gsl_spline * galcl_galcl_kernel(c_datablock * block, double chi_max, gsl_spline * n_of_z,
	gsl_spline * a_of_chi, int bin)
{


	// Integrand data
	w_integrand_data data;
	data.n_of_z = n_of_z;
	data.a_of_chi = a_of_chi;
	//data.b_of_z = b_of_z;
	//data.D_of_z = D_of_z;
	data.acc_chi = gsl_interp_accel_alloc();
	data.acc_z = gsl_interp_accel_alloc();
	data.zmax = n_of_z->x[n_of_z->size-1];

	// Limit of our range.
	double chi_min = 0.0;
	data.chi_max = chi_max;


	// Integration workspace
	gsl_integration_glfixed_table *table = 
		gsl_integration_glfixed_table_alloc((size_t) NGLT);

	// First compute the normalization
	gsl_function F;
	F.params = &data;
	F.function = &n_of_chi;
	double norm = 1.0 / gsl_integration_glfixed(&F, chi_min, chi_max, table);

	// Now do the main integral, chi-by-chi.
	// First, space for the results
	int n_chi = NGLT;
	double Chi[n_chi];
	double W[n_chi];

	// The evaluation points separation, and set up the integrator
	double delta_chi = (chi_max-chi_min)/ (n_chi - 1);

	gsl_error_handler_t * old_error_handler = gsl_set_error_handler_off();
	int error_status=0;

  	int status6=0;
  	double galbias;
  	char name1[20];
  	snprintf(name1, 20, "bias%d", bin);
  	status6 |= c_datablock_get_double(block, cosmo, name1, &galbias);
  	double galb = galbias;

        int status7=0;
        int DEN;
        status7 |= c_datablock_get_int_default(block, cosmo, "DEN", 1, &DEN);

        double factor;
        if (DEN==1){
        factor=1.0;
        }
        if (DEN==0){
        factor=0.0;
        }
        if (DEN<0 || DEN>1){
        printf("INVALID INPUT VALUE!!! INSERT 1 OR 0 TO CALCULATE OR NOT THE DENSITY FLUCTUATIONS\n");
        exit(0);
        }

	// Loop through samples to be evenly spaced in chi
	for(int i=0; i<n_chi; i++)
	{
		// Get chi at this evaluation point
		double chi = delta_chi * i;
		Chi[i] = chi;
		// We need it during the integration, so save.
		data.chi = chi;
		// ingredient for the prefactor chi/a
		double a;
		int err = gsl_spline_eval_e(a_of_chi, chi, data.acc_chi, &a);
		if (err) {error_status=1; break;} 

                //choose the bias constant accroding to the bin (survey) choice
                if (bin<=5){
                W[i] = norm * 1.7 * galb * n_of_chi(chi, F.params) * factor;
                }
                else {
                W[i] = norm * 0.84 * galb * n_of_chi(chi, F.params) * factor;
                }

	}
	gsl_set_error_handler(old_error_handler);

	// Convert the static vectors into a spline and return
	gsl_spline * output;
	if (error_status) output = NULL;
	else output = spline_from_arrays(n_chi, Chi, W);

	// Tidy up
	gsl_integration_glfixed_table_free(table);
	gsl_interp_accel_free(data.acc_chi);
	gsl_interp_accel_free(data.acc_z);

	// Finish
	return output;
}

//this has only n(z) but will be used as the RSD kernel (to be mutiplied with D(k,z)*f(k,z) )
gsl_spline * galcl_galcl2_kernel(c_datablock * block, double chi_max, gsl_spline * n_of_z,
	gsl_spline * a_of_chi)
{


	// Integrand data
	w_integrand_data data;
	data.n_of_z = n_of_z;
	data.a_of_chi = a_of_chi;
	//data.D_of_z = D_of_z;
	//data.f_of_z = f_of_z;
	data.acc_chi = gsl_interp_accel_alloc();
	data.acc_z = gsl_interp_accel_alloc();
	data.zmax = n_of_z->x[n_of_z->size-1];

	// Limit of our range.
	double chi_min = 0.0;
        //chi_max = chi_max*1.4;
	data.chi_max = chi_max;


	// Integration workspace
	gsl_integration_glfixed_table *table = 
		gsl_integration_glfixed_table_alloc((size_t) NGLT);

	// First compute the normalization
	gsl_function F;
	F.params = &data;
	F.function = &n_of_chi;
	double norm = 1.0 / gsl_integration_glfixed(&F, chi_min, chi_max, table);

	// Now do the main integral, chi-by-chi.
	// First, space for the results
	int n_chi = NGLT;
	double Chi[n_chi];
	double W2[n_chi];

	// The evaluation points separation, and set up the integrator
	double delta_chi = (chi_max-chi_min)/ (n_chi - 1);
	//F.function = &w_of_chi_integrand;

        int status9=0;
        int RSD;
        status9 |= c_datablock_get_int_default(block, cosmo, "RSD", 1, &RSD);

        double factor;
        if (RSD==1){
        factor=1.0;
        }
        if (RSD==0){
        factor=0.0;
        }
        if (RSD<0 || RSD>1){
        printf("INVALID INPUT VALUE!!! INSERT 1 OR 0 TO CALCULATE OR NOT THE RSD\n");
        exit(0);
        } 


	gsl_error_handler_t * old_error_handler = gsl_set_error_handler_off();
	int error_status=0;
	// Loop through samples to be evenly spaced in chi
	for(int i=0; i<n_chi; i++)
	{
		// Get chi at this evaluation point
		double chi = delta_chi * i;
		Chi[i] = chi;
		// We need it during the integration, so save.
		data.chi = chi;
		// ingredient for the prefactor chi/a
		double a;
		int err = gsl_spline_eval_e(a_of_chi, chi, data.acc_chi, &a);
		if (err) {error_status=1; break;} 
		// and calculate the integral
                W2[i] = norm * n_of_chi(chi, F.params) * factor;
	}
	gsl_set_error_handler(old_error_handler);

	// Convert the static vectors into a spline and return
	gsl_spline * output;
	if (error_status) output = NULL;
	else output = spline_from_arrays(n_chi, Chi, W2);

	// Tidy up
	gsl_integration_glfixed_table_free(table);
	gsl_interp_accel_free(data.acc_chi);
	gsl_interp_accel_free(data.acc_z);

	// Finish
	return output;
}


//this has \nabla{n(z)} and prefactors; will be used as the magnification kernel (to be mutiplied with D(k,z) )
gsl_spline * galcl_galcl3_kernel(c_datablock * block, double chi_max, gsl_spline * n_of_z, gsl_spline * a_of_chi, int bin)
{


	// Integrand data
	w_integrand_data data;
	data.n_of_z = n_of_z;
	data.a_of_chi = a_of_chi;
	//data.b_of_z = b_of_z;
	//data.D_of_z = D_of_z;
	data.acc_chi = gsl_interp_accel_alloc();
	data.acc_z = gsl_interp_accel_alloc();
	data.zmax = n_of_z->x[n_of_z->size-1];

	// Limit of our range.
	double chi_min = 0.0;
	data.chi_max = chi_max;


	// Integration workspace
	gsl_integration_glfixed_table *table = 
		gsl_integration_glfixed_table_alloc((size_t) NGLT);

	// First compute the normalization
	gsl_function F;
	F.params = &data;
	F.function = &n_of_chi;
	double norm = 1.0 / gsl_integration_glfixed(&F, chi_min, chi_max, table);

	// Now do the main integral, chi-by-chi.
	// First, space for the results
	int n_chi = NGLT;
	double Chi[n_chi];
	double W3[n_chi];

	// The evaluation points separation, and set up the integrator
	double delta_chi = (chi_max-chi_min)/ (n_chi - 1);
	//F.function = &w_of_chi_integrand;
	gsl_function F1;
	F1.params = &data;
	F1.function = &w_of_chi_integrand;

	gsl_error_handler_t * old_error_handler = gsl_set_error_handler_off();
	int error_status=0;
	// Loop through samples to be evenly spaced in chi

	int status7=0;
  	double mag;
  	char name2[20];
  	snprintf(name2, 20, "alpha_m%d", bin);
  	status7 |= c_datablock_get_double(block, cosmo, name2, &mag);
  	double magi = 2.0 * (mag - 1.0);

  	int status8=0;
	double omega_m;
	status8 = c_datablock_get_double(block, cosmo, "omega_m", &omega_m);
	const double c_kms = 299792.4580; //km/s
	double pre = 1.5 * (100.0*100.0)/(c_kms*c_kms) * omega_m;


        int status9=0;
        int MAG;
        status9 |= c_datablock_get_int_default(block, cosmo, "MAG", 1, &MAG);

        double factor;
        if (MAG==1){
        factor=1.0;
        }
        if (MAG==0){
        factor=0.0;
        }
        if (MAG<0 || MAG>1){
        printf("INVALID INPUT VALUE!!! INSERT 1 OR 0 TO CALCULATE OR NOT THE MAGNIFICATION BIAS\n");
        exit(0);
        } 


	for(int i=0; i<n_chi; i++)
	{
		// Get chi at this evaluation point
		double chi = delta_chi * i;
		Chi[i] = chi;
		// We need it during the integration, so save.
		data.chi = chi;
		// ingredient for the prefactor chi/a
		double a;
		int err = gsl_spline_eval_e(a_of_chi, chi, data.acc_chi, &a);
		if (err) {error_status=1; break;} 
		// and calculate the integral
		//W[i] = norm * (chi/a) * gsl_integration_glfixed(&F, chi, chi_max, table);
                W3[i] = norm * magi * pre * (chi/a) * gsl_integration_glfixed(&F1, chi, chi_max, table) * factor;
	}
	gsl_set_error_handler(old_error_handler);

	// Convert the static vectors into a spline and return
	gsl_spline * output;
	if (error_status) output = NULL;
	else output = spline_from_arrays(n_chi, Chi, W3);

	// Tidy up
	gsl_integration_glfixed_table_free(table);
	gsl_interp_accel_free(data.acc_chi);
	gsl_interp_accel_free(data.acc_z);

	// Finish
	return output;
}



/*
  Read z, N(z) values from the datablock and initialize a spline to
  use in the Limber integral. 
*/
gsl_spline * get_named_nchi_spline(c_datablock * block, const char * section, int bin, double *z,
				   gsl_spline * a_of_chi, gsl_spline * chi_of_z)
{
  int status = 0;
  double * N;
  int n;
  char name[20];
  snprintf(name, 20, "bin_%d", bin);

  status |= c_datablock_get_double_array_1d(block, section, name, &N, &n);
  double Chi[n];

  for (int i=0; i<n; i++){
    double chi = gsl_spline_eval(chi_of_z, z[i], NULL);
    double da_dchi = gsl_spline_eval_deriv(a_of_chi, chi, NULL);
    Chi[i] = chi;
    N[i] *= -(1+z[i])*(1+z[i])*da_dchi;
  }

  gsl_spline * n_of_chi = spline_from_arrays(n, Chi, N);
  free(N);
  return n_of_chi;

}

gsl_spline * get_nchi_spline(c_datablock * block, int bin, double *z,
			     gsl_spline * a_of_chi, gsl_spline * chi_of_z)
{
  return get_named_nchi_spline(block, wl_nz, bin, z, a_of_chi, chi_of_z);
}


gsl_spline * get_named_w_spline(c_datablock * block, const char * section, int bin, double * z,
	double chi_max, gsl_spline * a_of_chi_spline)
{
	char name[20];
	double * n_of_z;
	//double * b_of_z;
	//double * D_of_z;
	int nz1;
	//int bz1;
        //int Dz1;
	int status1 = 0;
	//int status2 = 0;
	//int status3 = 0;


	// Load n(z), b(z) and D(z)
	snprintf(name, 20, "bin_%d", bin);
	status1 |= c_datablock_get_double_array_1d(block, section, name, &n_of_z,
		&nz1);
	//status2 |= c_datablock_get_double_array_1d(block, section, "b", &b_of_z,
	//	&bz1);
        //status3 |= c_datablock_get_double_array_1d(block, section, "d", &D_of_z,
	//	&Dz1);
        //status3 |= c_datablock_get_double_array_1d(block,growthparameters, "d_z", &D_of_z, 
        //        &Dz1);
        //status3 |= c_datablock_get_double_array_1d(block,"GROWTH_PARAMETERS", "d_z", &D_of_z, 
        //        &Dz1);

	// Make a spline from n(z), b(z) and D(z)
	gsl_spline * n_of_z_spline = spline_from_arrays(nz1, z, n_of_z);
	//gsl_spline * b_of_z_spline = spline_from_arrays(bz1, z, b_of_z);
	//gsl_spline * D_of_z_spline = spline_from_arrays(Dz1, z, D_of_z);

	// Do the main computation
	gsl_spline * W = galcl_galcl_kernel(block, chi_max, n_of_z_spline,
		a_of_chi_spline, bin);

	// tidy up bin-specific data
	gsl_spline_free(n_of_z_spline);
	//gsl_spline_free(b_of_z_spline);
	//gsl_spline_free(D_of_z_spline);
	free(n_of_z);
	//free(b_of_z);
	//free(D_of_z);
	return W;
}


gsl_spline * get_named_w2_spline(c_datablock * block, const char * section, int bin, double * z,
	double chi_max, gsl_spline * a_of_chi_spline)
{
	char name[20];
	double * n_of_z;
	//double * D_of_z;
	//double * f_of_z;
	int nz1;
	//int Dz1;
        //int fz1;
	int status4 = 0;
	//int status5 = 0;
	//int status6 = 0;

	// Load n(z), D(z) and f(z)
	snprintf(name, 20, "bin_%d", bin);
	status4 |= c_datablock_get_double_array_1d(block, section, name, &n_of_z,
		&nz1);
	//status5 |= c_datablock_get_double_array_1d(block, section, "d", &D_of_z,
	//	&Dz1);
        //status5 |= c_datablock_get_double_array_1d(block,growthparameters, "d_z", &D_of_z, 
        //        &Dz1);
        //status5 |= c_datablock_get_double_array_1d(block,"GROWTH_PARAMETERS", "d_z", &D_of_z, 
        //        &Dz1);
        //status6 |= c_datablock_get_double_array_1d(block, section, "f", &f_of_z,
	//	&fz1);

	// Make a spline from n(z), b(z) and f(z)
	gsl_spline * n_of_z_spline = spline_from_arrays(nz1, z, n_of_z);
	//gsl_spline * D_of_z_spline = spline_from_arrays(Dz1, z, D_of_z);
	//gsl_spline * f_of_z_spline = spline_from_arrays(fz1, z, f_of_z);

	// Do the main computation
	gsl_spline * W2 = galcl_galcl2_kernel(block, chi_max, n_of_z_spline,
		a_of_chi_spline);

	// tidy up bin-specific data
	gsl_spline_free(n_of_z_spline);
	//gsl_spline_free(D_of_z_spline);
	//gsl_spline_free(f_of_z_spline);
	free(n_of_z);
	//free(D_of_z);
	//free(f_of_z);
	return W2;
}


gsl_spline * get_named_w3_spline(c_datablock * block, const char * section, int bin, double * z,
	double chi_max, gsl_spline * a_of_chi_spline)
{
	char name[20];
	double * n_of_z;
	//double * b_of_z;
	//double * D_of_z;
	int nz1;
	//int bz1;
        //int Dz1;
	int status1 = 0;
	//int status2 = 0;
	//int status3 = 0;


	// Load n(z), b(z) and D(z)
	snprintf(name, 20, "bin_%d", bin);
	status1 |= c_datablock_get_double_array_1d(block, section, name, &n_of_z,
		&nz1);
	//status2 |= c_datablock_get_double_array_1d(block, growthparameters, "b", &b_of_z,
	//	&bz1);
        //status3 |= c_datablock_get_double_array_1d(block, section, "d", &D_of_z,
	//	&Dz1);
        //status3 |= c_datablock_get_double_array_1d(block,"GROWTH_PARAMETERS", "d_z", &D_of_z, 
        //        &Dz1);

	// Make a spline from n(z), b(z) and D(z)
	gsl_spline * n_of_z_spline = spline_from_arrays(nz1, z, n_of_z);
	//gsl_spline * b_of_z_spline = spline_from_arrays(bz1, z, b_of_z);
	//gsl_spline * D_of_z_spline = spline_from_arrays(Dz1, z, D_of_z);

	// Do the main computation
	gsl_spline * W3 = galcl_galcl3_kernel(block, chi_max, n_of_z_spline,
		a_of_chi_spline, bin);

	// tidy up bin-specific data
	gsl_spline_free(n_of_z_spline);
	//gsl_spline_free(b_of_z_spline);
	//gsl_spline_free(D_of_z_spline);
	free(n_of_z);
	//free(b_of_z);
	//free(D_of_z);
	return W3;
}



/*
	Read z, N(z), b(z), D(z) values from the datablock and initialize a spline
   	of the lensing kernel W to use in the Limber integral.
*/
gsl_spline * get_w_spline(c_datablock * block, int bin, double * z,
	double chi_max, gsl_spline * a_of_chi_spline)
{
	return get_named_w_spline(block, wl_nz, bin, z, chi_max, a_of_chi_spline);
}



gsl_spline * get_w2_spline(c_datablock * block, int bin, double * z,
	double chi_max, gsl_spline * a_of_chi_spline)
{
	return get_named_w2_spline(block, wl_nz, bin, z, chi_max, a_of_chi_spline);
}


gsl_spline * get_w3_spline(c_datablock * block, int bin, double * z,
	double chi_max, gsl_spline * a_of_chi_spline)
{
	return get_named_w3_spline(block, wl_nz, bin, z, chi_max, a_of_chi_spline);
}



int get_wchi_array(c_datablock * block, const char * nz_section,
	 int bin, double * z, double chi_max, gsl_spline * a_of_chi_spline, double * chi_out, 
	 int nchi_out, double * wchi_out)
{
	// Do the main computation
	gsl_spline * W = get_named_w_spline(block, nz_section, bin, z, chi_max, a_of_chi_spline);

	//Now interpolate to output chi
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	for (int i_chi = 0; i_chi<nchi_out; i_chi++){
		wchi_out[i_chi] = gsl_spline_eval(W, chi_out[i_chi], acc);
	}

	// tidy up bin-specific data
	gsl_spline_free(W);
}


int get_w2chi_array(c_datablock * block, const char * nz_section,
	 int bin, double * z, double chi_max, gsl_spline * a_of_chi_spline, double * chi_out, 
	 int nchi_out, double * w2chi_out)
{       
	// Do the main computation

	gsl_spline * W2 = get_named_w2_spline(block, nz_section, bin, z, chi_max, a_of_chi_spline);

	//Now interpolate to output chi
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	for (int i_chi = 0; i_chi<nchi_out; i_chi++){
		w2chi_out[i_chi] = gsl_spline_eval(W2, chi_out[i_chi], acc);

	}

	// tidy up bin-specific data

	gsl_spline_free(W2);

}


int get_w3chi_array(c_datablock * block, const char * nz_section,
	 int bin, double * z, double chi_max, gsl_spline * a_of_chi_spline, double * chi_out, 
	 int nchi_out, double * w3chi_out)
{       
	// Do the main computation

	gsl_spline * W3 = get_named_w3_spline(block, nz_section, bin, z, chi_max, a_of_chi_spline);

	//Now interpolate to output chi
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	for (int i_chi = 0; i_chi<nchi_out; i_chi++){
		w3chi_out[i_chi] = gsl_spline_eval(W3, chi_out[i_chi], acc);

	}

	// tidy up bin-specific data

	gsl_spline_free(W3);

}


gsl_spline * cmb_wl_kappa_kernel(double chi_max, double chi_star, gsl_spline * a_of_chi)
{

  int n_chi = NGLT;
  double chi_min = 0.0;
  int error_status = 0;
  // The evaluation points separation, and set up the integrator
  double delta_chi = (chi_max-chi_min)/ (n_chi - 1);

  double W[n_chi];
  double Chi[n_chi];

  // Loop through samples to be evenly spaced in chi
  for(int i=0; i<n_chi; i++)
    {
      // Get chi at this evaluation point
      double chi = delta_chi * i;
      Chi[i] = chi;
      double a;
      int err = gsl_spline_eval_e(a_of_chi, chi, NULL, &a);
      if (err) {error_status=1; break;} 
      // and calculate the integral
      W[i] = (chi/a) * (chi_star-chi)/chi_star;
    }

  // Convert the static vectors into a spline and return
  gsl_spline * output;
  if (error_status) output = NULL;
  else output = spline_from_arrays(n_chi, Chi, W);

  // Finish
  return output;
}


//-----------------------------------------------------------------------------------------------------


static
int save_c_ell(c_datablock * block, const char * section,
	int bin1, int bin2, double coeff, gsl_spline * s, limber_config * lc)
{
	char name[64];
	snprintf(name, 64, "bin_%d_%d",bin1,bin2);
	int n_ell = lc->n_ell;
	double c_ell[n_ell];
	for (int i=0; i<n_ell; i++){
		double ell = lc->ell[i];
		if (lc->xlog) ell = log(ell);
		c_ell[i] = gsl_spline_eval(s, ell, NULL);
		if (lc->ylog && (lc->status==LIMBER_STATUS_OK)){
			c_ell[i] = exp(c_ell[i]);
		}
		c_ell[i]*=coeff;
		
	}

	int status = c_datablock_put_double_array_1d(block, section, name, c_ell, n_ell);
	status |= c_datablock_put_metadata(block, section, name, "note", "no ell^2 prefactor");
	return status;

}


const char * choose_output_section(spectrum_type_t spectrum_type,
	galcl_spectrum_config * config)
{
	switch(spectrum_type){
		case matter_matter: // config->matter_spectra
			return "matter_cl";
			break;
		case galcl_intrinsic: // config->intrinsic_alignments
			return 	GALCL_CL_GI_SECTION;
			break;
		case intrinsic_intrinsic: // config->intrinsic_alignments
			return GALCL_CL_II_SECTION;
			break;
		case galcl_galcl: // config->GALCL_GALCL
			return (config->intrinsic_alignments ? GALCL_CL_GG_SECTION : GALCL_CL_SECTION);
			break;
		case matter_galcl: // config->ggl_spectra
			return "ggl_cl";
			break;
		case matter_intrinsic: // config->gal_IA_cross_spectra
			return "gal_IA_cross_cl";
			break;
		case matter_magnification: // config->galaxy_magnification
			return "galaxy_magnification_cl";
			break;
		case magnification_magnification: // config->magnification_magnification
			return "magnification_magnification_cl";
			break;
		case magnification_intrinsic: // config->magnification_intrinsic
			return "magnification_intrinsic_cl";
			break;
		case magnification_galcl: // config->magnification_galcl
			return "magnification_galcl_cl";
			break;
		default:
			return NULL;
	}
}


int choose_configuration(c_datablock * block, spectrum_type_t spectrum_type,
	limber_config * lc, galcl_spectrum_config * config)
{
	int status = 0;
	// First we need to decide whether to use logs in the splines.
	// This is more accurate, but we cannot use it for cross-spectra
	// since they can go negative.
  	// Also, can't use for intrinsic_intrinsic if A=0 (i.e. zero signal)
	if (spectrum_type==intrinsic_intrinsic) {
	        double A;
	        status = c_datablock_get_double(block, "intrinsic_alignment_parameters", "A", &A);
		double eps=1e-9;
		if ( A>-1*eps && A<eps){
		  lc->xlog=false;
		  lc->ylog=false;
		}
		else {
		  lc->xlog=true;
		  lc->ylog=true;
		}
	}
	else if (spectrum_type==galcl_galcl || spectrum_type==matter_matter ||
		       spectrum_type==magnification_magnification){
		lc->xlog=true;
		lc->ylog=true;

	}
	else{
		lc->xlog=false;
		lc->ylog=false;
	}

	// Now we need to set up the ell range we wish to compute.
	// We need the number of ell values and the ell min and max.
	lc->n_ell = config->n_ell;

	// We log space the actual ell values
	lc->ell = malloc(sizeof(double)*lc->n_ell);
	double alpha = (log(config->ell_max) - log(config->ell_min)) / (config->n_ell - 1);
	for (int i=0; i<lc->n_ell; i++) lc->ell[i] = config->ell_min*exp(alpha * i);

	//Scalings

	// This scaling value is the bit that goes in front
	// of the kernels. For n(chi) this is 1.0 and for
	// galcl W(chi) it's this more complex number.

	// galcl-galcl picks up two copies of this scaling,
	// galcl-intrinsic picks up one,
	// and the other spectra just have a unity prefactor.

	// Note that this scaling is redshift independent.
	// The magnification spectra also have a luminosity
	// term for each bin pairing, but this is dealt with
	// separately.  

	double omega_m;
	status = c_datablock_get_double(block, cosmo, "omega_m", &omega_m);
	// in (Mpc/h)^-2
	// We leave the factor of h in because our P(k)
	// should have it in.  That's why there is a 100 in here.
	const double c_kms = 299792.4580; //km/s
	double galcl_scaling = 1.5 * (100.0*100.0)/(c_kms*c_kms) * omega_m;

	switch(spectrum_type){
		case galcl_galcl:
			lc->prefactor = 1.0;
			break;
		case galcl_intrinsic:
			lc->prefactor = galcl_scaling;
			break;
		case intrinsic_intrinsic:
			lc->prefactor = 1.0;
			break;
		case matter_matter:
			lc->prefactor = 1.0;
			break;
		case matter_galcl:
			lc->prefactor = galcl_scaling;
			break;
		case matter_intrinsic:
			lc->prefactor = 1.0;
			break;
		case matter_magnification:
			lc->prefactor = galcl_scaling;
			break;
		case magnification_magnification:
			lc->prefactor = galcl_scaling*galcl_scaling;
			break;
		case magnification_intrinsic:
			lc->prefactor = galcl_scaling;
			break;
		case magnification_galcl:
			lc->prefactor = galcl_scaling*galcl_scaling;
			break;
		default:
			lc->prefactor = 1.0/0.0;
			return 1;
	}
	return status;
}


/*
	Select the combinations of line-of-sight kernels for the Limber integral.
*/
int choose_kernels(spectrum_type_t spectrum_type, int nbin, gsl_spline * W[nbin], gsl_spline * W2[nbin], gsl_spline * W3[nbin], gsl_spline * Nchi[nbin], gsl_spline ** K1, gsl_spline ** K2,  gsl_spline ** K3, gsl_spline ** K4, gsl_spline ** K5, gsl_spline ** K6) //changed
{
	// The different spectra use different kernels to integrate over:
	// Here we choose which to use (we do it like this so we can use
	// the same code for the different spectra)

	switch(spectrum_type){
		case galcl_galcl:
		case magnification_magnification:
		case magnification_galcl:
			for(int b=0; b<nbin; b++) {K1[b] = W[b]; K2[b] = W2[b] ; K3[b]= W3[b]; K4[b] = W[b]; K5[b] = W2[b]; K6[b]= W3[b];}; //changed
			break;
		case intrinsic_intrinsic:
		case matter_matter:
		case matter_intrinsic:
			for(int b=0; b<nbin; b++) {K1[b] = Nchi[b]; K2[b] = Nchi[b];};
			break;
		case galcl_intrinsic:
		// This is actually IG rather than GI
		case matter_galcl:
		case matter_magnification:
			for(int b=0; b<nbin; b++) {K1[b] = Nchi[b]; K2[b] = W[b];};
			break;
		case magnification_intrinsic:
			for(int b=0; b<nbin; b++) {K1[b] = W[b]; K2[b] = Nchi[b];};
			break;
		default:
			return 1;
	}
	return 0;

}


int choose_bin2_max(spectrum_type_t spectrum_type, int nbin, int bin1)
{
	if (spectrum_type==galcl_galcl || spectrum_type==matter_matter ||
		spectrum_type==intrinsic_intrinsic || spectrum_type==magnification_magnification){
		return bin1;
	}
	else{
		return nbin;
	}

}

double choose_limber_coefficient(spectrum_type_t spectrum_type, double alpha_1, double alpha_2)
{
	double coeff;

	switch(spectrum_type){
		case magnification_magnification:
			coeff = 4.0*(alpha_1-1.0)*(alpha_2-1.0);
			break;
		case magnification_intrinsic:
		case magnification_galcl:
			coeff = 2.0*(alpha_1-1.0);
			break;
		case matter_magnification:
			coeff = 2.0*(alpha_2-1.0);
			break;
		case galcl_galcl:
		case intrinsic_intrinsic:
		case matter_matter:
		case matter_intrinsic:
		case galcl_intrinsic:
		default:
			coeff = 1.0/0.0;
	}

	return coeff;
}


int compute_spectra(c_datablock * block, int nbin, spectrum_type_t spectrum_type, gsl_spline * W[nbin], gsl_spline * W2[nbin], gsl_spline * W3[nbin], gsl_spline * Nchi[nbin], Interpolator2D * PK, Interpolator2D * fK, Interpolator2D * DK, Interpolator2D * BBK, galcl_spectrum_config * config) //changed
{
	// This is a unified interface for generating different
	// spectra.

	// We need:
	//  - to choose a section to save the output into
	//  - to configure the ell ranges and other options
	//  - to choose the W/N kernels to integrate over.

	limber_config lc;
	lc.ell = NULL;
	gsl_spline * K1[nbin];
	gsl_spline * K2[nbin];
	gsl_spline * K3[nbin]; //new
	gsl_spline * K4[nbin]; //new
	gsl_spline * K5[nbin]; //new
	gsl_spline * K6[nbin]; //new

	const char * section = choose_output_section(spectrum_type, config);
	int status = choose_configuration(block, spectrum_type, &lc, config);
	//status |= choose_kernels(spectrum_type, nbin, W, W2, Nchi, &K1[0], &K2[0], &K3[0], &K4[0]); //changed
	status |= choose_kernels(spectrum_type, nbin, W, W2, W3, Nchi, &K1[0], &K2[0], &K3[0], &K4[0], &K5[0], &K6[0]); //changed

	// If any of these have gone wrong we quit
	// after cleaning up any allocated memory
	if (section==NULL) status = 1;
	if (status) {
		free(lc.ell);
		return status;
	}

	// First save the ell values and number of bins
	status |= c_datablock_put_double_array_1d(block, section, "ell", lc.ell, lc.n_ell);
	status |= c_datablock_put_int(block, section, "nbin", nbin);

	// In addition to the Limber kernel, the magnification 
	// spectra also require the slope of the luminosity 
	// function at the limiting magnitude of the survey
	double *alpha, coeff;
	int Nzbin;
	int is_mag =   spectrum_type==magnification_magnification 
	            || spectrum_type==matter_magnification 
	            || spectrum_type==magnification_galcl
	            || spectrum_type==magnification_intrinsic;

	if (is_mag){
		status |= c_datablock_get_double_array_1d(block, lum, "alpha_binned", &alpha, &Nzbin);
	}
	if (status) return status;

	for (int bin1=1; bin1<=nbin; bin1++){
		// We also need to choose the max value of bin 2.
		// For auto-spectra we don't compute both bin_i_j and bin_j_i,
		// whereas for cross-spectra we do
		int bin2_max = choose_bin2_max(spectrum_type, nbin, bin1);
		for (int bin2=1; bin2<=bin2_max; bin2++){
			// reset the status
			lc.status = LIMBER_STATUS_OK;
			gsl_spline * c_ell = limber_integral(&lc, K1[bin1-1], K2[bin1-1],  K3[bin1-1], K4[bin2-1], K5[bin2-1], K6[bin2-1], PK, fK, DK, BBK);
			if (is_mag) coeff = choose_limber_coefficient(spectrum_type, alpha[bin1-1], alpha[bin2-1]);
			else coeff=1;
			int status = save_c_ell(block, section, bin1, bin2, coeff, c_ell, &lc);
			gsl_spline_free(c_ell);
			if (status) return status;
		}
	}
	free(lc.ell);
	if (is_mag) free(alpha);
	return status;
}

#define printf_verbose(...) if (config->verbose) printf(__VA_ARGS__)


int execute(c_datablock * block, void * config_in)
{
	DATABLOCK_STATUS status=0;
	double * chi;
	double * a;
	double * z;
	int nz1, nz2;
	int nbin;

	galcl_spectrum_config * config = (galcl_spectrum_config*) config_in;

	// Load the number of bins we will use
	status |= c_datablock_get_int(block, wl_nz, "nbin", &nbin);

	// Load z
	status |= c_datablock_get_double_array_1d(block, wl_nz, "z", &z, &nz1);

	// Load another, different z spacing.
	// I am putting this in an array called "a"
	// because I am about to convert it
	status |= c_datablock_get_double_array_1d(block, dist, "z", &a, &nz2);

	// Load chi(z)
	status |= c_datablock_get_double_array_1d(block, dist, "d_m", &chi, &nz2);

	// Reverse ordering so a is increasing - that is what
	// gsl_spline wants
	reverse(a, nz2);
	reverse(chi, nz2);

	// Convert chi from Mpc to Mpc/h
	double h0=0.0;
	status |= c_datablock_get_double(block, cosmo, "h0", &h0);
	for (int i=0; i<nz2; i++) chi[i]*=h0;

	// At the moment "a" is still actually redshift z
	gsl_spline * chi_of_z_spline = spline_from_arrays(nz2, a, chi);
	gsl_spline * z_of_chi_spline = spline_from_arrays(nz2, chi, a);//I put it to have f(k,z(x))

	// Replace z->a
	for (int i=0; i<nz2; i++) a[i] = 1.0/(1+a[i]);

	double chi_max = chi[nz2-1];

	// Make spline of a(chi)
	gsl_spline * a_of_chi_spline = spline_from_arrays(nz2, chi, a);

	// Make the W(), W2()
	int error_status=0;
	gsl_spline * W_splines[nbin];
	gsl_spline * W2_splines[nbin];
	gsl_spline * W3_splines[nbin];
	//gsl_spline * W4_splines[nbin];
	gsl_spline * Nchi_splines[nbin];
	//gsl_spline * Bchi_splines[nbin];
	//gsl_spline * Dchi_splines[nbin];
	//gsl_spline * Fchi_splines[nbin];
	for (int bin=1; bin<=nbin; bin++){
		W_splines[bin-1] = get_w_spline(block, bin, z, chi_max, a_of_chi_spline);
		W2_splines[bin-1] = get_w2_spline(block, bin, z, chi_max, a_of_chi_spline);
		W3_splines[bin-1] = get_w3_spline(block, bin, z, chi_max, a_of_chi_spline);
		//W4_splines[bin-1] = get_w4_spline(block, bin, z, chi_max, a_of_chi_spline);
		Nchi_splines[bin-1] = get_nchi_spline(block, bin, z, a_of_chi_spline, chi_of_z_spline);
		//Bchi_splines[bin-1] = get_bchi_spline(block, z, a_of_chi_spline, chi_of_z_spline);
		//Dchi_splines[bin-1] = get_Dchi_spline(block, z, a_of_chi_spline, chi_of_z_spline);
		//Fchi_splines[bin-1] = get_fchi_spline(block, z, a_of_chi_spline, chi_of_z_spline);
		if (W_splines[bin-1]==NULL) error_status=1;
		if (W2_splines[bin-1]==NULL) error_status=1;
		if (W3_splines[bin-1]==NULL) error_status=1;
		//if (W4_splines[bin-1]==NULL) error_status=1;
		if (Nchi_splines[bin-1]==NULL) error_status=1;
		//if (Bchi_splines[bin-1]==NULL) error_status=1;
		//if (Dchi_splines[bin-1]==NULL) error_status=1;
		//if (Fchi_splines[bin-1]==NULL) error_status=1;

	}
	if (error_status){
		free(chi);
		free(a);
		free(z);
		gsl_spline_free(a_of_chi_spline);
		gsl_spline_free(chi_of_z_spline);
		gsl_spline_free(z_of_chi_spline); //FOR F(k,z)
		return 1;
	}

	// ------------
	// Get the P(k)s we need
	printf_verbose("Loading interpolation splines\n");
	/*				Matter power spectrum			 		*/


	Interpolator2D * PK = load_interpolator_chi(
		block, chi_of_z_spline, MATTER_POWER_LIN_SECTION, "k_h", "z", "P_k");


	Interpolator2D * fK = load_interpolator_chi2(
		block, chi_of_z_spline, z_of_chi_spline, MATTER_POWER_LIN_SECTION, "k_h", "z", "P_k");


	Interpolator2D * DK = load_interpolator_chi3(
		block, chi_of_z_spline, z_of_chi_spline, MATTER_POWER_LIN_SECTION, "k_h", "z", "P_k");

	Interpolator2D * BBK = load_interpolator_chi4(
		block, chi_of_z_spline, z_of_chi_spline, MATTER_POWER_LIN_SECTION, "k_h", "z", "P_k");


// ------------
	// Compute the angular power spectra
	if (config->galcl_galcl){
		status |= compute_spectra(block, nbin, galcl_galcl,
			W_splines, W2_splines, W3_splines, Nchi_splines, PK, fK, DK, BBK, config);
		printf_verbose("Saved galcl-galcl spectrum.\n");//changed all
	}


	// tidy up global data
	for (int bin=0; bin<nbin; bin++) {gsl_spline_free(W_splines[bin]);
        gsl_spline_free(W2_splines[bin]);
        gsl_spline_free(W3_splines[bin]);}
	gsl_spline_free(a_of_chi_spline);
	gsl_spline_free(chi_of_z_spline);
	gsl_spline_free(z_of_chi_spline);

	destroy_and_null(&PK);
	destroy_and_null2(&fK); //scale dep. growth rate
	destroy_and_null3(&DK); //scale dep.growthf actor
	destroy_and_null4(&BBK); //scale dep. galaxy bias
	free(chi);
	free(a);
	free(z);

  return status;

}


int cleanup(void * config_in)
{
	// Free the memory that we allocated in the
	// setup
	galcl_spectrum_config * config = (galcl_spectrum_config*) config_in;
	free(config);
}
