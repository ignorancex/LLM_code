#include <cmath>
#include <vector>
#include <thread>
#include <fstream>
#include <string>

#include <boost/filesystem.hpp>

#include "utility.h"
#include "matrix_elements.h"
#include "rates.h"
#include "H5Cpp.h"
using std::placeholders::_1;
extern Debye_mass * t_channel_mD2;
//=============Thernalized Distribution funtion=================================
// xi = 1: Fermi Dirac; xi = -1 Bose Einsterin; xi = 0, Maxwell-Boltzmann
double inline f_0(double x, double xi){
    if (x<1e-9) x=1e-9;
    double result = 1./(std::exp(x)+xi);
    return result;
}

//=============function wrapper for GSL integration R22======================
double approx_R22(double * arg){
	//double E1 = arg[0];
	double T = arg[1];
	return std::pow(T,3)/t_channel_mD2->get_mD2(T);
}

double approx_R23(double * arg, double M){
	double E1 = arg[0];
	double T = arg[1];
	double dt = arg[2];
	double meff = std::sqrt(t_channel_mD2->get_mD2(T));
	double u = std::pow(dt*meff,2);
	return u/(1.+u)*T;
}

double approx_R32(double * arg){
	double E1 = arg[0];
	double T = arg[1];
	double dt = arg[2];
	double meff = std::sqrt(t_channel_mD2->get_mD2(T));
	double u = std::pow(dt*meff,2);
	return u/(1.+u)*T;
}

double fy_wrapper22(double y, void * params_){
	// unpack Vegas params
	integrate_params_2 * params = static_cast<integrate_params_2 *>(params_);
	double coeff = params->params[0];
	double Temp = params->params[1];
	double M2 = params->params[2];
	double v1 = params->params[3];
	double s = M2 + coeff*(1.-v1*y);
	double arg[2];
	arg[0] = s; arg[1] = Temp;
	double Xsection = params->f(arg);
	return (1.-v1*y)*Xsection;
}

double fx_wrapper22(double x, void * px_){
	integrate_params_2 * px = static_cast<integrate_params_2 *>(px_);
	double E1 = px->params[0];
	double v1 = px->params[1];
	double Temp = px->params[2];
	double M2 = px->params[3];
	double zeta = px->params[4];

	double result, error, ymin, ymax;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000);
	integrate_params_2 * py = new integrate_params_2;
	py->f = px->f;
	py->params = new double[4];
	py->params[0] = 2.*E1*x*Temp;
	py->params[1] = Temp;
	py->params[2] = M2;
	py->params[3] = v1;

        gsl_function F;
	F.function = fy_wrapper22;
	F.params = py;
	ymax = 1.;
	ymin = -1.;
	gsl_integration_qag(&F, ymin, ymax, 0, 1e-4, 10000, 6, w, &result, &error);
	delete [] py->params;
	delete py;
	gsl_integration_workspace_free(w);
	return x*x*f_0(x, zeta)*result;
}


//=============function wrapper for GSL integration R23======================
double fy_wrapper23(double y, void * params_){
	// unpack Vegas params
	integrate_params_2 * params = static_cast<integrate_params_2 *>(params_);
	double coeff = params->params[0];
	double Temp = params->params[1];
	double M2 = params->params[2];
	double v1 = params->params[3];
	double dt = params->params[4];// dts in the Cell Frame
	double E2 = params->params[5];// E2 in the Cell Frame
	double E1 = params->params[6];// E1 in the Cell Frame
	double s = M2 + coeff*(1.-v1*y);// s variable
	// transform time separation in to CoM frame
	double costheta2 = (M2 + 2.*E1*E2 - s)/(2.*v1*E1*E2);
	double vz_com = (v1*E1 + E2*costheta2)/(E1+E2);
	double gamma_com = (E1+E2)/std::sqrt(s);
	double dtp = gamma_com*(1. - vz_com*v1)*dt; // dt in the CoM Frame
	double * arg = new double[3];
	arg[0] = s; arg[1] = Temp; arg[2] = dtp; // dt in the CoM Frame
	double Xsection = params->f(arg);
	delete [] arg;
	return (1.-v1*y)*Xsection;
}

double fx_wrapper23(double x, void * px_){
	integrate_params_2 * px = static_cast<integrate_params_2 *>(px_);
	double E1 = px->params[0];
	double v1 = px->params[1];
	double Temp = px->params[2];
	double M2 = px->params[3];
	double zeta = px->params[4];
	double dt = px->params[5]; // dt in the Cell Frame

	double result, error, ymin, ymax;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000);
	integrate_params_2 * py = new integrate_params_2;
	py->f = px->f;
	py->params = new double[7];
	py->params[0] = 2.*E1*x*Temp;
	py->params[1] = Temp;
	py->params[2] = M2;
	py->params[3] = v1;
	py->params[4] = dt; // dt in the Cell Frame
	py->params[5] = x*Temp; // E2 in the Cell Frame
	py->params[6] = E1; // E1 in the Cell Frame

        gsl_function F;
	F.function = fy_wrapper23;
	F.params = py;
	ymax = 1.;
	ymin = -1.;
	gsl_integration_qag(&F, ymin, ymax, 0, 1e-3, 10000, 6, w, &result, &error);

	delete [] py->params;
	delete py;
	gsl_integration_workspace_free(w);
	return x*x*f_0(x, zeta)*result;
}

//=======================Rates abstract class==================================
rates::rates(std::string name_)
:	rd(), gen(rd()),
	dist_x(3.0, 1.0), dist_xcorr(5.0, 1.0), dist_norm_y(-1.0, 1.0), dist_reject(0.0, 1.0)
{
	std::cout << "#----------" << __func__ << " " << name_  << "----------" << std::endl;
}

//=======================Derived Scattering Rate class 2 to 2================================
rates_2to2::rates_2to2(Xsection_2to2 * Xprocess_, int degeneracy_, double eta_2_, std::string name_, bool refresh)
:	rates(name_), Xprocess(Xprocess_), M(Xprocess->get_M1()), degeneracy(degeneracy_),
	eta_2(eta_2_),
	NE1(100), NT(16), E1L(M*1.01), E1H(M*100), TL(0.13), TH(0.75),
	dE1((E1H-E1L)/(NE1-1.)), dT((TH-TL)/(NT-1.)),
	Rtab(boost::extents[NE1][NT])
{
	bool fileexist = boost::filesystem::exists(name_);
	if ( (!fileexist) || ( fileexist && refresh) ){
		std::cout << "# Populating table with new calculation" << std::endl;
		std::vector<std::thread> threads;
		size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
		size_t call_per_core = size_t(NT*1./Ncores);
		size_t call_for_last_core = NT - call_per_core*(Ncores-1);
		auto code = [this](size_t NTstart_, size_t dNT_) { this->tabulate_E1_T(NTstart_, dNT_); };
		for (size_t i=0; i< Ncores ; i++){
			size_t Nstart = i*call_per_core;
			size_t dN = (i==Ncores-1)? call_for_last_core : call_per_core;
			threads.push_back( std::thread(code, Nstart, dN) );
		}
		for (std::thread& t : threads)	t.join();

		save_to_file(name_, "Rates-tab");
	}
	else{
		std::cout << "# loading existing table" << std::endl;
		read_from_file(name_, "Rates-tab");
	}
	std::cout << std::endl;
}

void rates_2to2::save_to_file(std::string filename, std::string datasetname){
	H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);

	const size_t rank = 2;
	hsize_t dims[rank] = {NE1, NT};
	H5::DSetCreatPropList proplist{};
	proplist.setChunk(rank, dims);

	H5::DataSpace dataspace(rank, dims);
	auto datatype(H5::PredType::NATIVE_DOUBLE);
	H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
	dataset.write(Rtab.data(), datatype);

	// Attributes
	hdf5_add_scalar_attr(dataset, "E1_low", E1L);
	hdf5_add_scalar_attr(dataset, "E1_high", E1H);
	hdf5_add_scalar_attr(dataset, "N_E1", NE1);

	hdf5_add_scalar_attr(dataset, "T_low", TL);
	hdf5_add_scalar_attr(dataset, "T_high", TH);
	hdf5_add_scalar_attr(dataset, "N_T", NT);
	file.close();

}

void rates_2to2::read_from_file(std::string filename, std::string datasetname){
	H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
	const size_t rank = 2;

	H5::DataSet dataset = file.openDataSet(datasetname.c_str());
	hdf5_read_scalar_attr(dataset, "E1_low", E1L);
	hdf5_read_scalar_attr(dataset, "E1_high", E1H);
	hdf5_read_scalar_attr(dataset, "N_E1", NE1);
	dE1 = (E1H-E1L)/(NE1-1.);

	hdf5_read_scalar_attr(dataset, "T_low", TL);
	hdf5_read_scalar_attr(dataset, "T_high", TH);
	hdf5_read_scalar_attr(dataset, "N_T", NT);
	dT = (TH-TL)/(NT-1.);

	Rtab.resize(boost::extents[NE1][NT]);

	hsize_t dims_mem[rank];
  	dims_mem[0] = NE1;
  	dims_mem[1] = NT;
	H5::DataSpace mem_space(rank, dims_mem);

	H5::DataSpace data_space = dataset.getSpace();
	dataset.read(Rtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
	file.close();
}

void rates_2to2::tabulate_E1_T(size_t T_start, size_t dnT){
	double * arg = new double[2];
	for (size_t i=0; i<NE1; i++){
		arg[0] = E1L + i*dE1;
		for (size_t j=T_start; j<(T_start+dnT); j++){
			arg[1] = TL + j*dT;
			Rtab[i][j] = calculate(arg)/approx_R22(arg);
		}
	}
	delete [] arg;
}

double rates_2to2::interpR(double * arg){
	double E1 = arg[0], Temp = arg[1];
	if (Temp < TL) Temp = TL;
	if (Temp >= TH) Temp = TH-dT;
	if (E1 < E1L) E1 = E1L;
	if (E1 >= E1H) E1 = E1H-dE1;
	double xT, rT, xE1, rE1;
	size_t iT, iE1;
	xT = (Temp-TL)/dT;	iT = floor(xT); rT = xT - iT;
	xE1 = (E1 - E1L)/dE1; iE1 = floor(xE1); rE1 = xE1 - iE1;
	return interpolate2d(&Rtab, iE1, iT, rE1, rT)*approx_R22(arg);
}

double rates_2to2::calculate(double * arg)
{
	double E1 = arg[0], Temp = arg[1];
	double p1 = std::sqrt(E1*E1-M*M);
	double result, error, xmin, xmax;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(5000);
	integrate_params_2 * px = new integrate_params_2;
	px->f = std::bind( &Xsection_2to2::interpX, Xprocess, _1);
	px->params = new double[5];
	px->params[0] = E1;
	px->params[1] = p1/E1;
	px->params[2] = Temp;
	px->params[3] = M*M;
	px->params[4] = eta_2;

    gsl_function F;
	F.function = fx_wrapper22;
	F.params = px;
	xmax = 10.0;
	xmin = 0.0;
	gsl_integration_qag(&F, xmin, xmax, 0, 1e-3, 5000, 6, w, &result, &error);

	gsl_integration_workspace_free(w);
	delete [] px->params;
	delete px;
	return result*std::pow(Temp, 3)*4./c16pi2*degeneracy;
}

void rates_2to2::sample_initial(double * arg, std::vector< std::vector<double> > & IS){
	// this function samples x = E2/T and y = cos(theta2) from the distribution:
	// P(x, y) ~ x^2*exp(-x) * (1-v1*y) * sigma(M^2 + 2*E1*T*x - 2*p1*T*x*y, T)
	// We first generate X from gamma distribution Gamma(x; 3,1) ~ x^2*exp(-x) (cut off x < 20. )
	// and uniform sample y within (-1., 1.)
	// and finally rejected with P_rej(x,y) = (1-v1*y) * sigma(M^2 + 2*E1*T*x - 2*p1*T*x*y, T);
	// this function returns all initial state particles' four-vector in the order (p1, p2)
	double E1 = arg[0], Temp = arg[1];
	double * Xarg = new double[2];
	double M2 = M*M, x, y, max, smax;
	double v1 = std::sqrt(E1*E1 - M2)/E1;
	double intersection = M2, coeff1 = 2.*E1*Temp, coeff2 = -2.*E1*v1*Temp;
	smax = intersection + (coeff1 + (-1)*coeff2)*10.;
	if (smax < 2.*M2) smax = 2.*M2;
	Xarg[0] = smax; Xarg[1] = Temp;
	max = (1.+v1)*Xprocess->interpX(Xarg);
	do{
		do{x = dist_x(gen);}while(x>10.);
		y = dist_norm_y(gen);
		Xarg[0] = intersection + (coeff1 + coeff2*y)*x;
	}while( (1.-v1*y)*Xprocess->interpX(Xarg)/max < dist_reject(gen) );
	delete [] Xarg;
	double costheta2 = y, sintheta2 = std::sqrt(1. - y*y);
	double E2 = x*Temp, phi2 = (rand()*2.*M_PI)/RAND_MAX;
	double cosphi2 = std::cos(phi2), sinphi2 = std::sin(phi2);
	// Constructing initial states
	IS.resize(2); IS[0].resize(4); IS[1].resize(4);
	IS[0][0] = E1; IS[0][1] = 0.0;
	IS[0][2] = 0.0; IS[0][3] = v1*E1;

	IS[1][0] = E2; IS[1][1] = E2*sintheta2*cosphi2;
	IS[1][2] = E2*sintheta2*sinphi2; IS[1][3] = E2*costheta2;
}




//=======================Derived Scattering Rate class 2 to 3================================
rates_2to3::rates_2to3(Xsection_2to3 * Xprocess_, int degeneracy_, double eta_2_, std::string name_, bool refresh)
:	rates(name_), Xprocess(Xprocess_), M(Xprocess->get_M1()), degeneracy(degeneracy_),
	eta_2(eta_2_),
	NE1(100), NT(8), Ndt(20), E1L(M*1.01), E1H(M*100), TL(0.13), TH(0.75), dtL(0.1), dtH(10.0),
	dE1((E1H-E1L)/(NE1-1.)), dT((TH-TL)/(NT-1.)), ddt((dtH-dtL)/(Ndt-1.)),
	Rtab(boost::extents[NE1][NT][Ndt])
{
	bool fileexist = boost::filesystem::exists(name_);
	if ( (!fileexist) || ( fileexist && refresh) ){
		std::cout << "# Populating table with new calculation" << std::endl;
		std::vector<std::thread> threads;
		size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
		size_t call_per_core = size_t(NT*1./Ncores);
		size_t call_for_last_core = NT - call_per_core*(Ncores-1);
		auto code = [this](size_t NTstart_, size_t dNT_) { this->tabulate_E1_T(NTstart_, dNT_); };
		for (size_t i=0; i< Ncores ; i++){
			size_t Nstart = i*call_per_core;
			size_t dN = (i==Ncores-1)? call_for_last_core : call_per_core;
			threads.push_back( std::thread(code, Nstart, dN) );
		}
		for (std::thread& t : threads)	t.join();
		save_to_file(name_, "Rates-tab");
	}
	else{
		std::cout << "# loading existing table" << std::endl;
		read_from_file(name_, "Rates-tab");
	}
	std::cout << std::endl;
}

void rates_2to3::save_to_file(std::string filename, std::string datasetname){
	H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
	const size_t rank = 3;

	hsize_t dims[rank] = {NE1, NT, Ndt};
	H5::DSetCreatPropList proplist{};
	proplist.setChunk(rank, dims);

	H5::DataSpace dataspace(rank, dims);
	auto datatype(H5::PredType::NATIVE_DOUBLE);
	H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
	dataset.write(Rtab.data(), datatype);

	// Attributes
	hdf5_add_scalar_attr(dataset, "E1_low", E1L);
	hdf5_add_scalar_attr(dataset, "E1_high", E1H);
	hdf5_add_scalar_attr(dataset, "N_E1", NE1);

	hdf5_add_scalar_attr(dataset, "T_low", TL);
	hdf5_add_scalar_attr(dataset, "T_high", TH);
	hdf5_add_scalar_attr(dataset, "N_T", NT);

	hdf5_add_scalar_attr(dataset, "dt_low", dtL);
	hdf5_add_scalar_attr(dataset, "dt_high", dtH);
	hdf5_add_scalar_attr(dataset, "N_dt", Ndt);
	file.close();
}

void rates_2to3::read_from_file(std::string filename, std::string datasetname){
	H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
	const size_t rank = 3;

	H5::DataSet dataset = file.openDataSet(datasetname.c_str());
	hdf5_read_scalar_attr(dataset, "E1_low", E1L);
	hdf5_read_scalar_attr(dataset, "E1_high", E1H);
	hdf5_read_scalar_attr(dataset, "N_E1", NE1);
	dE1 = (E1H-E1L)/(NE1-1.);

	hdf5_read_scalar_attr(dataset, "T_low", TL);
	hdf5_read_scalar_attr(dataset, "T_high", TH);
	hdf5_read_scalar_attr(dataset, "N_T", NT);
	dT = (TH-TL)/(NT-1.);

	hdf5_read_scalar_attr(dataset, "dt_low", dtL);
	hdf5_read_scalar_attr(dataset, "dt_high", dtH);
	hdf5_read_scalar_attr(dataset, "N_dt", Ndt);
	ddt = (dtH-dtL)/(Ndt-1.);

	Rtab.resize(boost::extents[NE1][NT][Ndt]);
	hsize_t dims_mem[rank];
  	dims_mem[0] = NE1;
  	dims_mem[1] = NT;
	dims_mem[2] = Ndt;
	H5::DataSpace mem_space(rank, dims_mem);

	H5::DataSpace data_space = dataset.getSpace();
	dataset.read(Rtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
	file.close();
}

void rates_2to3::tabulate_E1_T(size_t T_start, size_t dnT){
	double * arg = new double[3];
	for (size_t i=0; i<NE1; i++){
		arg[0] = E1L + i*dE1;
		for (size_t j=T_start; j<(T_start+dnT); j++){
			arg[1] = TL + j*dT;
			for (size_t k=0; k<Ndt; k++){
				arg[2] = dtL + k*ddt;
				Rtab[i][j][k] = calculate(arg)/approx_R23(arg, M);
			}
		}
	}
	delete [] arg;
}

double rates_2to3::interpR(double * arg){
	double E1 = arg[0], Temp = arg[1], dt = arg[2];
	if (Temp < TL) Temp = TL;
	if (Temp >= TH) Temp = TH-dT;
	if (dt < dtL) dt = dtL;
	if (dt >= dtH) dt = dtH-ddt;
	if (E1 < E1L) E1 = E1L;
	if (E1 >= E1H) E1 = E1H-dE1;
	double xT, rT, xE1, rE1, xdt, rdt;
	size_t iT, iE1, idt;
	xT = (Temp-TL)/dT;	iT = floor(xT); rT = xT - iT;
	xdt = (dt-dtL)/ddt;	idt = floor(xdt); rdt = xdt - idt;
	xE1 = (E1 - E1L)/dE1; iE1 = floor(xE1); rE1 = xE1 - iE1;
	return interpolate3d(&Rtab, iE1, iT, idt, rE1, rT, rdt)*approx_R23(arg, M);
}

double rates_2to3::calculate(double * arg)
{
	double E1 = arg[0], Temp = arg[1], dt = arg[2]; // dt in the Cell Frame
	double p1 = std::sqrt(E1*E1-M*M);
	double result, error, xmin, xmax;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
	integrate_params_2 * px = new integrate_params_2;
	px->f = std::bind( &Xsection_2to3::interpX, Xprocess, _1);
	px->params = new double[6];
	px->params[0] = E1;
	px->params[1] = p1/E1;
	px->params[2] = Temp;
	px->params[3] = M*M;
	px->params[4] = eta_2;
	px->params[5] = dt; // dt in the Cell Frame

    gsl_function F;
	F.function = fx_wrapper23;
	F.params = px;
	xmax = 10.0;
	xmin = 0.0;
	gsl_integration_qag(&F, xmin, xmax, 0, 1e-2, 2000, 6, w, &result, &error);

	gsl_integration_workspace_free(w);
	delete [] px->params;
	delete px;
	return result*std::pow(Temp, 3)*4./c16pi2*degeneracy;
}

void rates_2to3::sample_initial(double * arg, std::vector< std::vector<double> > & IS){
	// this function samples x = E2/T and y = cos(theta2) from the distribution:
	// P(x, y) ~ x^2*exp(-x) * (1-v1*y) * sigma(M^2 + 2*E1*T*x - 2*p1*T*x*y, T)
	// We first generate X from gamma distribution Gamma(x; 3,1) ~ x^3*exp(-x) (cut off x < 20. )
	// and uniform sample y within (-1., 1.)
	// and finally rejected with P_rej(x,y) = (1-v1*y) * sigma(M^2 + 2*E1*T*x - 2*p1*T*x*y, T);
	double E1 = arg[0], Temp = arg[1], dt = arg[2];
	double * Xarg = new double[3]; Xarg[1] = Temp; Xarg[2] = dt; // dt in Cell Frame
	double M2 = M*M, x, y, max, smax, stemp;
	double v1 = std::sqrt(E1*E1 - M2)/E1;
	double intersection = M*M, coeff1 = 2.*E1*Temp, coeff2 = -2.*E1*Temp*v1;
	smax = M2 + (coeff1 +(-1)*coeff2)*10.;
	if (smax < 2.*M2) smax = 2.*M2;
	Xarg[0] = smax;
	max = (1.+v1)*Xprocess->interpX(Xarg);
	do{
		do{ x = dist_x(gen); }while(x>10.);
		y = dist_norm_y(gen);
		stemp = intersection + coeff1*x + coeff2*x*y;
		Xarg[0] = stemp;
	}while( (1.-v1*y)*Xprocess->interpX(Xarg) <= max*dist_reject(gen) );
	delete [] Xarg;
	double E2 = x*Temp;
	double costheta2 = y, sintheta2 = std::sqrt(1. - y*y);
	double phi2 = (rand()*2.*M_PI)/RAND_MAX;
	double cosphi2 = std::cos(phi2), sinphi2 = std::sin(phi2);
	// Constructing initial states
	IS.resize(2); IS[0].resize(4); IS[1].resize(4);
	IS[0][0] = E1; IS[0][1] = 0.0; IS[0][2] = 0.0; IS[0][3] = v1*E1;
	IS[1][0] = E2; IS[1][1] = E2*sintheta2*cosphi2; IS[1][2] = E2*sintheta2*sinphi2; IS[1][3] = E2*costheta2;
}

//=======================Derived Scattering Rate class 3 to 2================================
rates_3to2::rates_3to2(f_3to2 * Xprocess_, int degeneracy_, double eta_2_, double eta_k_, std::string name_, bool refresh)
:	rates(name_), Xprocess(Xprocess_), M(Xprocess->get_M1()), degeneracy(degeneracy_),
	eta_2(eta_2_), eta_k(eta_k_),
	NE1(50), NT(8), Ndt(10), E1L(M*1.01), E1H(M*50), TL(0.13), TH(0.75), dtL(0.1), dtH(10.0),
	dE1((E1H-E1L)/(NE1-1.)), dT((TH-TL)/(NT-1.)), ddt((dtH-dtL)/(Ndt-1.)),
	Rtab(boost::extents[NE1][NT][Ndt])
{
	bool fileexist = boost::filesystem::exists(name_);
	if ( (!fileexist) || ( fileexist && refresh) ){
		std::cout << "# Populating table with new calculation" << std::endl;
		std::vector<std::thread> threads;
		size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
		size_t call_per_core = size_t(NT*1./Ncores);
		size_t call_for_last_core = NT - call_per_core*(Ncores-1);
		auto code = [this](size_t NTstart_, size_t dNT_) { this->tabulate_E1_T(NTstart_, dNT_); };
		for (size_t i=0; i< Ncores ; i++){
			size_t Nstart = i*call_per_core;
			size_t dN = (i==Ncores-1)? call_for_last_core : call_per_core;
			threads.push_back( std::thread(code, Nstart, dN) );
		}
		for (std::thread& t : threads)	t.join();

		save_to_file(name_, "Rates-tab");
	}
	else{
		std::cout << "# loading existing table" << std::endl;
		read_from_file(name_, "Rates-tab");
	}
	std::cout << std::endl;
}

void rates_3to2::save_to_file(std::string filename, std::string datasetname){
	H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
	const size_t rank = 3;

	hsize_t dims[rank] = {NE1, NT, Ndt};
	H5::DSetCreatPropList proplist{};
	proplist.setChunk(rank, dims);

	H5::DataSpace dataspace(rank, dims);
	auto datatype(H5::PredType::NATIVE_DOUBLE);
	H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
	dataset.write(Rtab.data(), datatype);

	// Attributes
	hdf5_add_scalar_attr(dataset, "E1_low", E1L);
	hdf5_add_scalar_attr(dataset, "E1_high", E1H);
	hdf5_add_scalar_attr(dataset, "N_E1", NE1);

	hdf5_add_scalar_attr(dataset, "T_low", TL);
	hdf5_add_scalar_attr(dataset, "T_high", TH);
	hdf5_add_scalar_attr(dataset, "N_T", NT);

	hdf5_add_scalar_attr(dataset, "dt_low", dtL);
	hdf5_add_scalar_attr(dataset, "dt_high", dtH);
	hdf5_add_scalar_attr(dataset, "N_dt", Ndt);
	file.close();
}

void rates_3to2::read_from_file(std::string filename, std::string datasetname){
	H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
	const size_t rank = 3;

	H5::DataSet dataset = file.openDataSet(datasetname.c_str());
	hdf5_read_scalar_attr(dataset, "E1_low", E1L);
	hdf5_read_scalar_attr(dataset, "E1_high", E1H);
	hdf5_read_scalar_attr(dataset, "N_E1", NE1);
	dE1 = (E1H-E1L)/(NE1-1.);

	hdf5_read_scalar_attr(dataset, "T_low", TL);
	hdf5_read_scalar_attr(dataset, "T_high", TH);
	hdf5_read_scalar_attr(dataset, "N_T", NT);
	dT = (TH-TL)/(NT-1.);

	hdf5_read_scalar_attr(dataset, "dt_low", dtL);
	hdf5_read_scalar_attr(dataset, "dt_high", dtH);
	hdf5_read_scalar_attr(dataset, "N_dt", Ndt);
	ddt = (dtH-dtL)/(Ndt-1.);

	Rtab.resize(boost::extents[NE1][NT][Ndt]);
	hsize_t dims_mem[rank];
  	dims_mem[0] = NE1;
  	dims_mem[1] = NT;
	dims_mem[2] = Ndt;
	H5::DataSpace mem_space(rank, dims_mem);

	H5::DataSpace data_space = dataset.getSpace();
	dataset.read(Rtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
	file.close();
}

void rates_3to2::tabulate_E1_T(size_t T_start, size_t dnT){
	double * arg = new double[3];
	for (size_t i=0; i<NE1; i++){
		arg[0] = E1L + i*dE1;
		for (size_t j=T_start; j<(T_start+dnT); j++){
			arg[1] = TL + j*dT;
			for (size_t k=0; k<Ndt; k++){
				arg[2] = dtL + k*ddt;
				Rtab[i][j][k] = calculate(arg)/approx_R32(arg);
			}
		}
	}
	delete [] arg;
}

double rates_3to2::interpR(double * arg){
	double E1 = arg[0], Temp = arg[1], dt = arg[2];
	if (Temp < TL) Temp = TL;
	if (Temp >= TH) Temp = TH-dT;
	if (dt < dtL) dt = dtL;
	if (dt >= dtH) dt = dtH-ddt;
	if (E1 < E1L) E1 = E1L;
	if (E1 >= E1H) E1 = E1H-dE1;
	double xT, rT, xE1, rE1, xdt, rdt;
	size_t iT, iE1, idt;
	xT = (Temp-TL)/dT;	iT = floor(xT); rT = xT - iT;
	xdt = (dt-dtL)/ddt;	idt = floor(xdt); rdt = xdt - idt;
	xE1 = (E1 - E1L)/dE1; iE1 = floor(xE1); rE1 = xE1 - iE1;
	return interpolate3d(&Rtab, iE1, iT, idt, rE1, rT, rdt)*approx_R32(arg);;
}


//-------------3->2 wrapper function--------------------------
double dRdPS_wrapper(double * x_, size_t n_dims_, void * params_){
	integrate_params_2 * params = static_cast<integrate_params_2 *>(params_);
	double x2 = x_[0], costheta2 = x_[1],
		   xk = x_[2], costhetak = x_[3],
		   phik = x_[4];
	if (costheta2 <= -1. || costheta2 >= 1.) return 0.;
	if (costhetak <= -1. || costhetak >= 1. ) return 0.;
	if (phik <= 0. || phik >= 2.*M_PI) return 0.;
	if (x2 <= 0.0 || x2 >= 5.) return 0.;
	if (xk <= 0.0 || xk >= 5.) return 0.;

	double sintheta2 = std::sqrt(1. - costheta2*costheta2);
	double sinthetak = std::sqrt(1. - costhetak*costhetak);
	double cosphik = std::cos(phik), sinphik = std::sin(phik);

	double E1 = params->params[0],
		   Temp = params->params[1],
		   dt = params->params[2],
		   M = params->params[3],
		   p1 = params->params[4],
		   eta_2 = params->params[5],
		   eta_k = params->params[6];
	double E2 = x2*Temp, k = xk*Temp;

	double kx = k*sinthetak*cosphik, ky = k*sinthetak*sinphik, kz = k*costhetak;
	double p2x = sintheta2*E2, p2z = costheta2*E2;
	// given p1 = (E1, 0, 0, p1),
	//		 p2 = (E2, p2x, 0, p2z)
	// and   k  = (k, kx, ky, kz)
	double P0 = E1 + E2 + k, P1 = p2x + kx, P2 = ky, P3 = p1 + p2z + kz;
	double s = P0*P0 - P1*P1 - P2*P2 - P3*P3;
	double vx = P1/P0, vy = P2/P0, vz = P3/P0;
	double gamma = P0/std::sqrt(s);
	double kp  = gamma*(k - vx*kx -vy*ky -vz*kz),
		   E2p = gamma*(E2 - vx*p2x - vz*p2z),
		   E1p = gamma*(E1 - vz*p1);
	double p1p = std::sqrt(E1p*E1p - M*M);
	double p_tot = E2p+p1p+kp;
	double w2 = E2p/p_tot, wk = kp/p_tot;

	// Quantity in CoM frame of p1 and p2
	double * arg = new double[5];
	arg[0] = s; arg[1] = Temp; arg[2] =  w2 + wk; arg[3] = (w2 - wk)/(1. - w2 - wk); arg[4] = gamma*(1. - vz*p1/E1)*dt;
	double result = x2*f_0(x2, eta_2)*xk*f_0(xk, eta_k)*params->f(arg);
	delete [] arg;
	return result;
}

double rates_3to2::calculate(double * arg){
	double E1 = arg[0], Temp = arg[1], dt = arg[2]; // dt in the Cell Frame
	double result, error;
	integrate_params_2 * params = new integrate_params_2;
	params->f = std::bind(&f_3to2::interpX, Xprocess, _1);

	const gsl_rng_type * Tr = gsl_rng_default;
	gsl_rng * r = gsl_rng_alloc(Tr);

	params->params = new double[7];
	params->params[0] = E1;
	params->params[1] = Temp;
	params->params[2] = dt;
	params->params[3] = M;
	params->params[4] = std::sqrt(E1*E1-M*M);
	params->params[5] = eta_2;
	params->params[6] = eta_k;

	gsl_monte_function G;
	G.f = dRdPS_wrapper;
	G.dim = 5;
	G.params = params;

	// integration limits
	double xl[5], xu[5];
	xl[0] = 0.0; xu[0] = 5.;
	xl[1] = -1.; xu[1] = 1.;
	xl[2] = 0.0; xu[2] = 5.;
	xl[3] = -1.; xu[3] = 1.;
	xl[4] = 0.0; xu[4] = 2.0*M_PI;

	// Actuall integration, require the Xi-square to be close to 1,  (0.5, 1.5)
	gsl_monte_vegas_state * sv = gsl_monte_vegas_alloc(5);
	do{
		gsl_monte_vegas_integrate(&G, xl, xu, 5, 10000, r, sv, &result, &error);
	}while(std::abs(gsl_monte_vegas_chisq(sv)-1.0)>0.5);
	gsl_monte_vegas_free(sv);
	gsl_rng_free(r);
	delete [] params->params;
	delete params;

	return result/256./std::pow(M_PI, 5)/E1*std::pow(Temp, 4)*degeneracy;
}

void rates_3to2::sample_initial(double * arg, std::vector< std::vector<double> > & IS){
	double E1 = arg[0], Temp = arg[1], dt = arg[2];
	double p1 = std::sqrt(E1*E1-M*M);
	integrate_params_2 * params = new integrate_params_2;
	params->f = std::bind(&f_3to2::interpX, Xprocess, _1);
	params->params = new double[7];
	params->params[0] = E1;
	params->params[1] = Temp;
	params->params[2] = dt;
	params->params[3] = M;
	params->params[4] = p1;
	params->params[5] = eta_2;
	params->params[6] = eta_k;
	size_t n_dims = 5;
	double * guessl = new double[n_dims];
	double * guessh = new double[n_dims];
	guessl[0] = 0.9; guessl[1] = -0.1; guessl[2] = 0.9; guessl[3] = -0.1; guessl[4] = 0.9*M_PI;
	guessh[0] = 1.1; guessh[1] = 0.1; guessh[2] = 1.1; guessh[3] = 0.1; guessh[4] = 1.1*M_PI;
	std::vector<double> vec5 = sampler.sample(dRdPS_wrapper, n_dims, params, guessl, guessh);
	double x2 = vec5[0],
		   costheta2 = vec5[1],
		   xk = vec5[2],
	       costhetak = vec5[3],
		   phik = vec5[4];
	double phi2 = (rand()*2.*M_PI)/RAND_MAX;
	double E2 = x2*Temp, k = xk*Temp,
		   sintheta2 = std::sqrt(1. - costheta2*costheta2),
		   sinthetak = std::sqrt(1. - costhetak*costhetak);
	IS.resize(3); IS[0].resize(4); IS[1].resize(4); IS[2].resize(4); // HQ, light q or g, absorbed g
	IS[0][0] = E1; IS[0][1] = 0.; IS[0][2] = 0.; IS[0][3] = p1;
	IS[1][0] = E2; IS[1][1] = E2*sintheta2*std::cos(phi2); IS[1][2] = E2*sintheta2*std::sin(phi2); IS[1][3] = E2*costheta2;
	IS[2][0] = k; IS[2][1] = k*sinthetak*std::cos(phi2+phik); IS[2][2] = k*sinthetak*std::sin(phi2+phik); IS[2][3] = k*costhetak;
	delete [] params->params;
	delete params;
	delete [] guessl;
	delete [] guessh;
}
