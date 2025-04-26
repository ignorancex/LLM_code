#include "stdio.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_errno.h"
#include "limber.h"
#include <gsl/gsl_interp2d.h>
//#include <gsl/gsl_spline2d.h>
//#include <gsl/gsl_spline.h>

// This is a workspace size for the gsl integrator
#define LIMBER_FIXED_TABLE_SIZE 4096


// data that is passed into the integrator
// This is everything we need to compute the
// integrand
typedef struct IntegrandData{
	double chimin;
	double chimax;
	double ell;
	gsl_spline * WbX; //K1
	gsl_spline * WfX; //K2
	gsl_spline * WmX; //K3
	gsl_spline * WbY; //K4
	gsl_spline * WfY; //K5
	gsl_spline * WmY; //K6
	Interpolator2D * P;
	Interpolator2D * f; //for the f(k,z(x))
	Interpolator2D * D; //for the D(k,z(x))
	Interpolator2D * BB; //for the b(k,z(x))
	gsl_interp_accel * accelerator_xb;
	gsl_interp_accel * accelerator_yb;
	gsl_interp_accel * accelerator_xf;
	gsl_interp_accel * accelerator_yf;
	gsl_interp_accel * accelerator_xm;
	gsl_interp_accel * accelerator_ym;
} IntegrandData;


// the integrand for all 
static double integrand(double chi, void * data_void)
{
	IntegrandData * data = (IntegrandData*) data_void;
	// Return 0 if outside range, for convenience.
	// Important to ensure that ranges are wide enough.
	if(chi < data->chimin || chi > data->chimax) return 0.0;

//exactly the same notation of paperIII

	double wx0 = gsl_spline_eval(data->WbX,chi,data->accelerator_xb);
	double wy0 = gsl_spline_eval(data->WbY,chi,data->accelerator_yb);

	double wx0m = gsl_spline_eval(data->WmX,chi,data->accelerator_xm);
	double wy0m = gsl_spline_eval(data->WmY,chi,data->accelerator_ym);

	double wx1 = gsl_spline_eval(data->WfX,chi,data->accelerator_xf);
	double wy1 = gsl_spline_eval(data->WfY,chi,data->accelerator_yf);

	double wx2 = gsl_spline_eval(data->WfX,chi*(2.0*data->ell-3.0)/(2.0*data->ell+1.0),data->accelerator_xf);
	double wy2 = gsl_spline_eval(data->WfY,chi*(2.0*data->ell-3.0)/(2.0*data->ell+1.0),data->accelerator_yf);

	double chi5 = chi*(2.0*data->ell+5.0)/(2.0*data->ell+1.0) < data->chimax ? chi*(2.0*data->ell+5.0)/(2.0*data->ell+1.0) : 0.0;
	double wx3 = gsl_spline_eval(data->WfX, chi5, data->accelerator_xf);
	double wy3 = gsl_spline_eval(data->WfY, chi5, data->accelerator_yf);

        double c1 = (2.*data->ell*data->ell+2.*data->ell-1.)/((2.*data->ell-1.)*(2.*data->ell+3.));   
        double c2 = -((data->ell*(data->ell-1.))/((2.*data->ell-1.)*sqrt((2.*data->ell+1.)*(2.*data->ell-3.))));
        double c3 = -(((data->ell+1.)*(data->ell+2.))/((2.*data->ell+3.)*sqrt((2.*data->ell+1.)*(2.*data->ell+5.))));

	// Get P(k,0) using k=ell/chi.
	// The interp_2d interpolator returns 0 if either 
	// parameter is outside its range
	double ka = (data->ell+0.5) / chi;
	double pa = interp_2d(ka, chi, data->P);

        //for the scale dependent bias b(k,z)
        double b0 = interp_2d(ka, chi, data->BB);

        //for the scale dependent growth factor D(k,z)
        double d0 = interp_2d(ka, chi, data->D);
        double d1 = d0;
	double d2 = interp_2d(ka, chi*(2.0*data->ell-3.0)/(2.0*data->ell+1.0), data->D);
	double d3 = interp_2d(ka, chi5, data->D);

        //for the scale dependent growth rate f(k,z)
        double f1 = interp_2d(ka, chi, data->f);
	double f2 = interp_2d(ka, chi*(2.0*data->ell-3.0)/(2.0*data->ell+1.0), data->f);
	double f3 = interp_2d(ka, chi5, data->f);

        double WWX= (wx0*b0*d0 + wx0m*d0 + c1*wx1*f1*d1 + c2*wx2*f2*d2 + c3*wx3*f3*d3);
        double WWY= (wy0*b0*d0 + wy0m*d0 + c1*wy1*f1*d1 + c2*wy2*f2*d2 + c3*wy3*f3*d3);

	double result = WWX * WWY * pa / chi / chi;

	return result;

}


// These two convenience functions
// peer into the internals of the gsl_spline.
// This is probably a bit naughty, since they could
// in theory change the internals.
static double inline limber_gsl_spline_min_x(gsl_spline * s)
{
	return s->x[0];
}
static double inline limber_gsl_spline_max_x(gsl_spline * s)
{
	return s->x[s->size-1];
}



double get_kernel_peak(gsl_spline * WbX, gsl_spline * WbY, int n_chi)
{
  double chimin_x = limber_gsl_spline_min_x(WbX);
  double chimin_y = limber_gsl_spline_min_x(WbY);
  double chimax_x = limber_gsl_spline_max_x(WbX);
  double chimax_y = limber_gsl_spline_max_x(WbY);
  double chimin = chimin_x>chimin_y ? chimin_x : chimin_y;
  double chimax = chimax_x<chimax_y ? chimax_x : chimax_y;
  double dchi = (chimax - chimin)/n_chi;
  double chi_peak = chimin;
  double chi;
  double kernel_val=0.;
  double kernel_peak=0.;
  for (int i_chi=0; i_chi<=n_chi; i_chi++){
    chi=chimin+i_chi*dchi;
    kernel_val = gsl_spline_eval(WbX,chi,NULL) * gsl_spline_eval(WbY,chi,NULL) / chi / chi;
    if (kernel_val>kernel_peak){
      kernel_peak = kernel_val;
      chi_peak = chi;
    }
  }
  // printf("chi_peak = %f\n",chi_peak);
  return chi_peak;
}



gsl_integration_workspace * W = NULL;
gsl_integration_glfixed_table *table = NULL;

void setup_integration_workspaces(){
	if (W==NULL){
		W = gsl_integration_workspace_alloc(LIMBER_FIXED_TABLE_SIZE);
	}
	if (table==NULL){
		table = gsl_integration_glfixed_table_alloc((size_t) LIMBER_FIXED_TABLE_SIZE);
	}
}


static
void limber_gsl_fallback_integrator(gsl_function * F, double chimin, double chimax, 
	double abstol, double reltol, double * c_ell, double * error){

	// Only one warning per process
	static int fallback_warning_given = 0;

	// Deactivate error handling - if this one fails we will fall back to a more reliable but slower integrator
	gsl_error_handler_t * old_handler = gsl_set_error_handler_off();

	// Try the fast but flaky integrator.
	int status = gsl_integration_qag(F, chimin, chimax, abstol, reltol, LIMBER_FIXED_TABLE_SIZE, GSL_INTEG_GAUSS61, W, c_ell, error);

	// Restore the old error handler
	gsl_set_error_handler(old_handler); 

	// If the fast integrator failed fall back to the old one.
	if (status){
		IntegrandData * data = (IntegrandData*) F->params;
		double ell = data->ell;
		if (fallback_warning_given==0){
			fprintf(stderr, "Falling back to the old integrator for ell=%lf (status=%d)\n", ell,status);
			fallback_warning_given=1;
		}
		*c_ell = gsl_integration_glfixed(F,chimin,chimax,table);
	}

}


// The only function in this little library callable from the outside
// world.  The limber_config structure is defined in limber.h but is fairly
// obvious.  The splines and the interpolator need to be functions of 
// chi NOT z.
gsl_spline * limber_integral(limber_config * config, gsl_spline * WbX,  gsl_spline * WfX,  gsl_spline * WmX, gsl_spline * WbY,
	                 gsl_spline * WfY, gsl_spline * WmY, Interpolator2D * P, Interpolator2D * f, Interpolator2D * D, Interpolator2D * BB)

{

    config->status = LIMBER_STATUS_ERROR;
    int any_parameter_error=0;
    if (WbX==NULL){
        fprintf(stderr, "NULL WbX parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (WbY==NULL){
        fprintf(stderr, "NULL WbY parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (WfX==NULL){
        fprintf(stderr, "NULL WfX parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (WfY==NULL){
        fprintf(stderr, "NULL WfY parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (WmX==NULL){
        fprintf(stderr, "NULL WmX parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (WmY==NULL){
        fprintf(stderr, "NULL WmY parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (P==NULL){
        fprintf(stderr, "NULL P parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (f==NULL){
        fprintf(stderr, "NULL f parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (D==NULL){
        fprintf(stderr, "NULL D parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (config->n_ell<0){
        fprintf(stderr, "Negative n_ell parameter in limber_integral\n");
        any_parameter_error = 1;
    }
    if (config->n_ell==0){
        fprintf(stderr, "Error: n_ell=0 in limber calculation.\n");
        any_parameter_error = 1;
    }
    if (any_parameter_error){
        return NULL;
    }

	config->status = LIMBER_STATUS_OK;


	// Get the appropriate ranges over which to integrate
	// It is assumed that (at least one of) the kernel
	// splines should go to zero in some kind of reasonable
	// place, so we just use the range they specify
	IntegrandData data;
	double chimin_xb = limber_gsl_spline_min_x(WbX);
	double chimin_yb = limber_gsl_spline_min_x(WbY);
	double chimax_xb = limber_gsl_spline_max_x(WbX);
	double chimax_yb = limber_gsl_spline_max_x(WbY);

	double chimin_xf = limber_gsl_spline_min_x(WfX);
	double chimin_yf = limber_gsl_spline_min_x(WfY);
	double chimax_xf = limber_gsl_spline_max_x(WfX);
	double chimax_yf = limber_gsl_spline_max_x(WfY);

	double chimin_xm = limber_gsl_spline_min_x(WmX);
	double chimin_ym = limber_gsl_spline_min_x(WmY);
	double chimax_xm = limber_gsl_spline_max_x(WmX);
	double chimax_ym = limber_gsl_spline_max_x(WmY);

        //my ifs here
        double chimin_x = chimin_xb > chimin_xf ? chimin_xb : chimin_xf;
        double chimin_y = chimin_yb > chimin_yf ? chimin_yb : chimin_yf;
        double chimax_x = chimax_xb < chimax_xf ? chimax_xb : chimax_xf;
        double chimax_y = chimax_yb < chimax_yf ? chimax_yb : chimax_yf;  
	double c_ell, error;

	// Workspaces for the main and falback integrators.
	// Static, so only allocated once as it has a fixed size.
	setup_integration_workspaces();

	double reltol = config->relative_tolerance;
	double abstol = config->absolute_tolerance;
	// double reltol = 0.001;
	// double abstol = 0.00001;
	// printf("TOLS: %le %le\n",reltol,abstol);


	// Take the smallest range since we want both the
	// splines to be valid there.
	// This range as well as all the data needed to compute
	// the integrand is put into a struct to be passed
	// through the integrator to the function above.
	data.chimin = chimin_x>chimin_y ? chimin_x : chimin_y;
	data.chimax = chimax_x<chimax_y ? chimax_x : chimax_y;
	data.WbX = WbX;
	data.WfX = WfX;
	data.WmX = WmX;
	data.WbY = WbY;
	data.WfY = WfY;
	data.WmY = WmY;
	data.P = P;
	data.f = f;//new
	data.D = D;//new
	data.BB = BB;//new
	data.accelerator_xb = gsl_interp_accel_alloc();
	data.accelerator_xf = gsl_interp_accel_alloc();
	data.accelerator_xm = gsl_interp_accel_alloc();
	data.accelerator_yb = gsl_interp_accel_alloc();
	data.accelerator_yf = gsl_interp_accel_alloc();
	data.accelerator_ym = gsl_interp_accel_alloc();

	// Set up the workspace and inputs to the integrator.
	// Not entirely sure what the table is.
	gsl_function F;
	F.function = integrand;
	F.params = &data;

	//gsl_integration_workspace * workspace = gsl_integration_workspace_alloc(2048);

	// results of the integration go into these arrays.
	double c_ell_vector[config->n_ell];
	double ell_vector[config->n_ell];

	// loop through ell values according to the input configuration
	for (int i_ell = 0; i_ell<config->n_ell; i_ell++){
		double ell = config->ell[i_ell];
		data.ell=ell;

		// Perform the main integration.
		// This particular function is used because that's what Matt Becker 
		// found to work best.
		//c_ell = gsl_integration_glfixed(&F,data.chimin,data.chimax,table);
		// New function still attributable to the legacy of Matt Becker's integrator wisdom.
		// gsl_integration_qag(&F, data.chimin, data.chimax, abstol, reltol, LIMBER_FIXED_TABLE_SIZE, GSL_INTEG_GAUSS61, W, table, &c_ell, &error);
		//printf("%d %f %f\n",i_ell,c_ell_old,c_ell);
		limber_gsl_fallback_integrator(&F, data.chimin, data.chimax, 
			abstol, reltol, &c_ell, &error);

		//Include the prefactor scaling
		c_ell *= config->prefactor;

		// Record the results into arrays
		c_ell_vector[i_ell] = c_ell;
		ell_vector[i_ell] = ell;
	}

		// It is often useful to interpolate into the logs of the functions
		// This is optional in the config. We move this outside the main loop
		// since we may have all zeros in the output
		if (config->xlog) {
			for (int i_ell = 0; i_ell<config->n_ell; i_ell++){
				ell_vector[i_ell] = log(ell_vector[i_ell]);
			}
		}
		if (config->ylog){
			for (int i_ell = 0; i_ell<config->n_ell; i_ell++){
				if (c_ell_vector[i_ell]<0){
					config->status = LIMBER_STATUS_NEGATIVE;
				}
					// negative is worse than zero so only set to zero it not already negative
				else if ((c_ell_vector[i_ell]==0) && (config->status<LIMBER_STATUS_ZERO)){
					config->status = LIMBER_STATUS_ZERO;
				}
			}
			// If none of the values are <= 0 then we are okay to go ahead and take the logs.
			if (config->status == LIMBER_STATUS_OK){
				for (int i_ell = 0; i_ell<config->n_ell; i_ell++) c_ell_vector[i_ell] = log(c_ell_vector[i_ell]);
			}

		}

	// Create a spline of the arrays as the output
	gsl_spline * output = gsl_spline_alloc(gsl_interp_akima, (size_t) config->n_ell);
	gsl_spline_init(output, ell_vector, c_ell_vector, (size_t) config->n_ell);

	// Tidy up
	gsl_interp_accel_free(data.accelerator_xb);
	gsl_interp_accel_free(data.accelerator_yb);
	gsl_interp_accel_free(data.accelerator_xf);
	gsl_interp_accel_free(data.accelerator_yf);
	gsl_interp_accel_free(data.accelerator_xm);
	gsl_interp_accel_free(data.accelerator_ym);

	// These two are not deallocated because they are static and only initialized once.
	// gsl_integration_glfixed_table_free(table);	
	// gsl_integration_workspace_free(W);

	// And that's it
	return output;
}
