#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <vector>
#include <string>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>

#include <boost/filesystem.hpp>


#include "utility.h"
#include "matrix_elements.h"
#include "Xsection.h"
#include "H5Cpp.h"

extern Debye_mass * t_channel_mD2;

double approx_X22(double * arg, double M){
	double s = arg[0], T = arg[1];
	return 1.0/std::pow(1.0 - M*M/s, 2)/t_channel_mD2->get_mD2(T);
}

double approx_X23(double * arg, double M){
	double s = arg[0], T = arg[1], dt = arg[2];
	double meff = std::sqrt(t_channel_mD2->get_mD2(T));
	double u = std::pow(dt*meff, 2);
	return u/(1.+u)*std::log(s/M/M)/meff/meff;
}

double approx_X32(double * arg, double M){
	double s = arg[0], Temp = arg[1], a1 = arg[2], a2 = arg[3];
	double sqrts = std::sqrt(s);
	double xk = 0.5*(a1*a2 + a1 - a2);
	double x2 = 0.5*(-a1*a2 + a1 + a2);
	double M2 = M*M;
	double A = (2.*xk/x2 - 1.), B = -2.*sqrts*(1. + xk/x2), C = s - M2;
	double E2 = (-B - std::sqrt(B*B-4.*A*C))/2./A;
	double k = xk/x2*E2;
	double p1 = (1. - x2 - xk)/x2*E2;
	double E1 = std::sqrt(p1*p1 + M2);
	double cosk = (E2*E2-k*k-p1*p1)/2./p1/k;
	double kz = k*cosk;
	double kt2 = k*k - kz*kz;
	double frac = (k + kz)/(E1 + p1 + k + kz);
	double fracbar = (k + std::abs(kz))/(E1 + p1 + k + std::abs(kz));
	double x2M2 = frac*frac*M2;
	double mD2t = t_channel_mD2->get_mD2(Temp);
	double coeff_mg2 = (1.-fracbar)*mD2t/2.;
	double D1 = kt2 + x2M2 + coeff_mg2;
	double D2 = D1 + mD2t*0.2;
	double prop2 = kt2/D1/D1 + mD2t*0.2/D2/D2;
	return (s - M*M)/mD2t/x2*prop2;
}

double gsl_1dfunc_wrapper(double x, void * params_){
	Mygsl_integration_params * p = static_cast<Mygsl_integration_params*>(params_);
	return p->f(&x, 1, p->params);
}

//=============Xsection base class===================================================
// this is the base class for 2->2 and 2->3 cross-sections
Xsection::Xsection(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh)
: dXdPS(dXdPS_), M1(M1_)
{
	std::cout << "#----------" << __func__ << " " << name_  << "----------" << std::endl;
}


//============Derived 2->2 Xsection class===================================
Xsection_2to2::Xsection_2to2(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh)
:	Xsection(dXdPS_, M1_, name_, refresh), rd(), gen(rd()), dist_phi3(0.0, 2.0*M_PI),
	Nsqrts(200), NT(32),
	sqrtsL(M1_*1.01), sqrtsH(M1_*30.),
	dsqrts((sqrtsH-sqrtsL)/(Nsqrts-1.)),
	TL(0.12), TH(0.8), dT((TH-TL)/(NT-1.)), Xtab(boost::extents[Nsqrts][NT])
{
	bool fileexist = boost::filesystem::exists(name_);
	if ( (!fileexist) || ( fileexist && refresh) ){
		std::cout << "# Populating table with new calculation" << std::endl;
		std::vector<std::thread> threads;
		size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
		size_t call_per_core = size_t(NT*1./Ncores);
		size_t call_for_last_core = NT - call_per_core*(Ncores-1);
		auto code = [this](size_t NTstart_, size_t dNT_) { this->tabulate(NTstart_, dNT_); };
		for (size_t i=0; i< Ncores ; i++){
			size_t Nstart = i*call_per_core;
			size_t dN = (i==Ncores-1)? call_for_last_core : call_per_core;
			threads.push_back( std::thread(code, Nstart, dN) );
		}
		for (std::thread& t : threads)	t.join();
		save_to_file(name_, "Xsection-tab");
	}
	else{
		std::cout << "# loading existing table" << std::endl;
		read_from_file(name_, "Xsection-tab");
	}
	std::cout << std::endl;
}

void Xsection_2to2::save_to_file(std::string filename, std::string datasetname){
	const size_t rank = 2;

	H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
	hsize_t dims[rank] = {Nsqrts, NT};
	H5::DSetCreatPropList proplist{};
	proplist.setChunk(rank, dims);

	H5::DataSpace dataspace(rank, dims);
	auto datatype(H5::PredType::NATIVE_DOUBLE);
	H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
	dataset.write(Xtab.data(), datatype);

	// Attributes
	hdf5_add_scalar_attr(dataset, "sqrts_low", sqrtsL);
	hdf5_add_scalar_attr(dataset, "sqrts_high", sqrtsH);
	hdf5_add_scalar_attr(dataset, "N_sqrt", Nsqrts);

	hdf5_add_scalar_attr(dataset, "T_low", TL);
	hdf5_add_scalar_attr(dataset, "T_high", TH);
	hdf5_add_scalar_attr(dataset, "N_T", NT);
	file.close();
}

void Xsection_2to2::read_from_file(std::string filename, std::string datasetname){
	const size_t rank = 2;

	H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = file.openDataSet(datasetname.c_str());
	hdf5_read_scalar_attr(dataset, "sqrts_low", sqrtsL);
	hdf5_read_scalar_attr(dataset, "sqrts_high", sqrtsH);
	hdf5_read_scalar_attr(dataset, "N_sqrt", Nsqrts);
	dsqrts = (sqrtsH-sqrtsL)/(Nsqrts-1.);

	hdf5_read_scalar_attr(dataset, "T_low", TL);
	hdf5_read_scalar_attr(dataset, "T_high", TH);
	hdf5_read_scalar_attr(dataset, "N_T", NT);
	dT = (TH-TL)/(NT-1.);

	Xtab.resize(boost::extents[Nsqrts][NT]);
	hsize_t dims_mem[rank];
  	dims_mem[0] = Nsqrts;
  	dims_mem[1] = NT;
	H5::DataSpace mem_space(rank, dims_mem);

	H5::DataSpace data_space = dataset.getSpace();
	dataset.read(Xtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
	file.close();
}

void Xsection_2to2::tabulate(size_t T_start, size_t dnT){
	double arg[2];
	for (size_t i=0; i<Nsqrts; ++i) {
		arg[0] = std::pow(sqrtsL + i*dsqrts, 2);
		for (size_t j=T_start; j<(T_start+dnT); j++) {
			arg[1] = TL + j*dT;
			Xtab[i][j] = calculate(arg)/approx_X22(arg, M1);
		}
	}
}

double Xsection_2to2::interpX(double * arg){
	double sqrts = std::sqrt(arg[0]), Temp = arg[1];
	if (Temp < TL) Temp = TL;
	if (Temp >= TH) Temp = TH-dT;
	if (sqrts < sqrtsL) sqrts = sqrtsL;
	if (sqrts >= sqrtsH) sqrts = sqrtsH-dsqrts;
	double xT, rT, xsqrts, rsqrts;
	size_t iT, isqrts;
	xT = (Temp-TL)/dT;	iT = floor(xT); rT = xT - iT;
	xsqrts = (sqrts - sqrtsL)/dsqrts; isqrts = floor(xsqrts); rsqrts = xsqrts - isqrts;
	return approx_X22(arg, M1)*interpolate2d(&Xtab, isqrts, iT, rsqrts, rT);
}


double Xsection_2to2::calculate(double * arg){
	double s = arg[0], Temp = arg[1];
	double result, error, tmin, tmax;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
	Mygsl_integration_params * params = new Mygsl_integration_params;
	params->f = dXdPS;
	params->params = new double[3];
	params->params[0] = s;
	params->params[1] = Temp;
	params->params[2] = M1;

        gsl_function F;
	F.function = gsl_1dfunc_wrapper;
	F.params = params;
	tmax = 0.0;
	tmin = -std::pow(s-M1*M1, 2)/s;
	gsl_integration_qag(&F, tmin, tmax, 0, 1e-4, 1000, 6, w, &result, &error);
        delete [] params->params;
	delete params;
	gsl_integration_workspace_free(w);

    return result;
}

void Xsection_2to2::sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS){
	double s = arg[0], Temp = arg[1];
	double * p = new double[3]; //s, T, M
	p[0] = s; p[1] = Temp; p[2] = M1;
	double M2 = M1*M1;
	double sqrts = std::sqrt(s);
	double pQ = (s-M2)/2./sqrts;
	double EQ = sqrts - pQ;
	double t = sampler1d.sample(dXdPS, -std::pow(s-M2, 2)/s, 0.0, p);
	double costheta3 = 1. + t/pQ/pQ/2.;
	double sintheta3 = std::sqrt(1. - costheta3*costheta3);
	double phi3 = dist_phi3(gen);
	double cosphi3 = std::cos(phi3), sinphi3 = std::sin(phi3);
	FS.resize(2);
	FS[0].resize(4);
	FS[0][0] = EQ; FS[0][1] = pQ*sintheta3*cosphi3;
	FS[0][2] = pQ*sintheta3*sinphi3; FS[0][3] = pQ*costheta3;
	FS[1].resize(4);
	FS[1][0] = pQ; FS[1][1] = -FS[0][1];
	FS[1][2] = -FS[0][2]; FS[1][3] = -FS[0][3];
	delete [] p;
}

//============Derived 2->3 Xsection class===================================
Xsection_2to3::Xsection_2to3(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh)
:	Xsection(dXdPS_, M1_, name_, refresh), rd(), gen(rd()), dist_phi4(0.0, 2.0*M_PI),
	Nsqrts(50), NT(16), Ndt(10),
	sqrtsL(M1_*1.01), sqrtsH(M1_*30.), dsqrts((sqrtsH-sqrtsL)/(Nsqrts-1.)),
	TL(0.12), TH(0.8), dT((TH-TL)/(NT-1.)),
	dtL(0.1), dtH(5.0), ddt((dtH-dtL)/(Ndt-1.)), Xtab(boost::extents[Nsqrts][NT][Ndt])
{

	bool fileexist = boost::filesystem::exists(name_);
	if ( (!fileexist) || ( fileexist && refresh) ){
		std::cout << "# Populating table with new calculation" << std::endl;
		std::vector<std::thread> threads;
		size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
		size_t call_per_core = size_t(NT*1./Ncores);
		size_t call_for_last_core = NT - call_per_core*(Ncores-1);
		auto code = [this](size_t NTstart_, size_t dNT_) { this->tabulate(NTstart_, dNT_); };
		for (size_t i=0; i< Ncores ; i++){
			size_t Nstart = i*call_per_core;
			size_t dN = (i==Ncores-1)? call_for_last_core : call_per_core;
			threads.push_back( std::thread(code, Nstart, dN) );
		}
		for (std::thread& t : threads)	t.join();
		save_to_file(name_, "Xsection-tab");
	}
	else{
		std::cout << "# loading existing table" << std::endl;
		read_from_file(name_, "Xsection-tab");
	}
	std::cout << std::endl;
}

void Xsection_2to3::save_to_file(std::string filename, std::string datasetname){
	const size_t rank = 3;

	H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
	hsize_t dims[rank] = {Nsqrts, NT, Ndt};
	H5::DSetCreatPropList proplist{};
	proplist.setChunk(rank, dims);

	H5::DataSpace dataspace(rank, dims);
	auto datatype(H5::PredType::NATIVE_DOUBLE);
	H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
	dataset.write(Xtab.data(), datatype);

	// Attributes
	hdf5_add_scalar_attr(dataset, "sqrts_low", sqrtsL);
	hdf5_add_scalar_attr(dataset, "sqrts_high", sqrtsH);
	hdf5_add_scalar_attr(dataset, "N_sqrt_half", Nsqrts);

	hdf5_add_scalar_attr(dataset, "T_low", TL);
	hdf5_add_scalar_attr(dataset, "T_high", TH);
	hdf5_add_scalar_attr(dataset, "N_T", NT);

	hdf5_add_scalar_attr(dataset, "dt_low", dtL);
	hdf5_add_scalar_attr(dataset, "dt_high", dtH);
	hdf5_add_scalar_attr(dataset, "N_dt", Ndt);
	file.close();
}

void Xsection_2to3::read_from_file(std::string filename, std::string datasetname){
	const size_t rank = 3;

	H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = file.openDataSet(datasetname.c_str());

	hdf5_read_scalar_attr(dataset, "sqrts_low", sqrtsL);
	hdf5_read_scalar_attr(dataset, "sqrts_high", sqrtsH);
	hdf5_read_scalar_attr(dataset, "N_sqrt_half", Nsqrts);
	dsqrts = (sqrtsH-sqrtsL)/(Nsqrts-1.);

	hdf5_read_scalar_attr(dataset, "T_low", TL);
	hdf5_read_scalar_attr(dataset, "T_high", TH);
	hdf5_read_scalar_attr(dataset, "N_T", NT);
	dT = (TH-TL)/(NT-1.);

	hdf5_read_scalar_attr(dataset, "dt_low", dtL);
	hdf5_read_scalar_attr(dataset, "dt_high", dtH);
	hdf5_read_scalar_attr(dataset, "N_dt", Ndt);
	ddt = (dtH-dtL)/(Ndt-1.);

	Xtab.resize(boost::extents[Nsqrts][NT][Ndt]);
	hsize_t dims_mem[rank];
  	dims_mem[0] = Nsqrts;
  	dims_mem[1] = NT;
	dims_mem[2] = Ndt;
	H5::DataSpace mem_space(rank, dims_mem);

	H5::DataSpace data_space = dataset.getSpace();
	dataset.read(Xtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
	file.close();
}

void Xsection_2to3::tabulate(size_t T_start, size_t dnT){
	double * arg = new double[3]; // s, T, dt
	for (size_t i=0; i<Nsqrts; i++) {
		arg[0] = std::pow(sqrtsL + i*dsqrts, 2);
		for (size_t j=T_start; j<(T_start+dnT); j++) {
			arg[1] = TL + j*dT;
			for (size_t k=0; k<Ndt; k++) {
				arg[2] = dtL + k*ddt;
				Xtab[i][j][k] = calculate(arg)/approx_X23(arg, M1);
			}
		}
	}
	delete [] arg;
}

double Xsection_2to3::interpX(double * arg){
	double sqrts = std::sqrt(arg[0]), Temp = arg[1], dt = arg[2];
	if (sqrts < sqrtsL) sqrts = sqrtsL;
	if (sqrts >= sqrtsH) sqrts = sqrtsH-dsqrts;
	if (Temp < TL) Temp = TL;
	if (Temp >= TH) Temp = TH-dT;
	if (dt < dtL) dt = dtL;
	if (dt >= dtH) dt = dtH-ddt;
	double xsqrts, rsqrts,
		   xT, rT,
		   xdt, rdt;
	size_t isqrts, iT, idt;
	xsqrts = (sqrts-sqrtsL)/dsqrts;	isqrts = floor(xsqrts); rsqrts = xsqrts - isqrts;
	xT = (Temp-TL)/dT;	iT = floor(xT); rT = xT - iT;
	xdt = (dt-dtL)/ddt;	idt = floor(xdt); rdt = xdt - idt;
	return approx_X23(arg, M1)*interpolate3d(&Xtab, isqrts, iT, idt, rsqrts, rT, rdt);
}

double Xsection_2to3::calculate(double * arg){
	double s = arg[0], Temp = arg[1], dt = arg[2];
	double result, error;

	const gsl_rng_type * Tr = gsl_rng_default;
	gsl_rng * r = gsl_rng_alloc(Tr);

	double * params = new double[4];
	params[0] = s; params[1] = Temp; params[2] = M1; params[3] = dt;

	gsl_monte_function G;
	G.f = dXdPS;
	G.dim = 4; // k, p4, phi4k, cos4
	G.params = params;

	// limits of the integration
	double sqrts = std::sqrt(s), M2 = M1*M1;
	double xl[4], xu[4];
	xl[0] = -15.; xu[0] = std::log((sqrts-M1)/(sqrts+M1));
	xl[1] = -std::log(1.-M2/s); xu[1] = 15.;
	xl[2] = -10.0; xu[2] = 0.;
	xl[3] = -M_PI; xu[3] = M_PI;

	// Actuall integration, require the Xi-square to be close to 1,  (0.5, 1.5)
	gsl_monte_vegas_state * sv = gsl_monte_vegas_alloc(4);
	do{
		gsl_monte_vegas_integrate(&G, xl, xu, 4, 4000, r, sv, &result, &error);
	}while(std::abs(gsl_monte_vegas_chisq(sv)-1.0)>1.);
	gsl_monte_vegas_free(sv);
	gsl_rng_free(r);
	delete [] params;
	return result/c256pi4/(s-M2);
}

void Xsection_2to3::sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS){
	// for 2->3, dXdPS is a 5-dimensional distribution,
	// In center of mass frame:
	// there is an overall azimuthal symmetry which allows a flat sampling
	// the rese 4 variables are sampled from Affine-invariant MCMC procedure,
	// since the distribution scale of each variable could vary a lot.
	// returns heavy quark 4-momentum and radiated gluon 4-momentum
	double s = arg[0], Temp = arg[1], dt = arg[2];
	double * p = new double[4]; //s, T, dt, M
	double sqrts = std::sqrt(s);
	double M2 = M1*M1;
	p[0] = s; p[1] = Temp; p[2] = M1; p[3] = dt; // dt in CoM frame
	size_t n_dims = 4;
	double * guessl = new double[n_dims];
	double * guessh = new double[n_dims];
	double xlim0 = std::log((sqrts-M1)/(sqrts+M1));
	double xlim1 = -std::log(1.-M2/s);
	double pmax = 0.5*(s-M2)/sqrts;
	double g = pmax/Temp+1.;
	double eta0 = -std::log(g+std::sqrt(g*g-1.));
	guessl[0] = xlim0-3.;
	guessl[1] = xlim1+2.;
	guessl[2] = eta0-1.;
	guessl[3] = -M_PI/2.;

	guessh[0] = xlim0-2.;
	guessh[1] = xlim1+3.;
	guessh[2] = eta0+1.;
	guessh[3] = M_PI/2.;
	std::vector<double> x_ = sampler.sample(dXdPS, n_dims, p, guessl, guessh);
	double expx1 = std::exp(x_[0]);
	double expmx2 = std::exp(-x_[1]);
	double k = pmax*(expx1+expmx2)/(1.-M2/s);
	double p4 = pmax - pmax*(expx1*M2/s+expmx2)/(1.-M2/s);
	double cos4 = std::tanh(x_[2]);
	double phi4k = x_[3];
	double cos_star = ((s-M2)-2.*sqrts*(p4+k))/(2.*p4*k) +1.;
	double sin_star = std::sqrt(1. - cos_star*cos_star), sin4 = std::sqrt(1. - cos4*cos4);
	double cos_4k = std::cos(phi4k), sin_4k = std::sin(phi4k);
	// k-vec
	double kxp = k*(sin_star*cos_4k*cos4 + sin4*cos_star),
		   kyp = k*sin_star*sin_4k,
		   kz = k*(-sin_star*cos_4k*sin4 + cos4*cos_star);
	// HQ-vec
	double HQxp = -kxp - p4*sin4,
		   HQyp = -kyp,
		   HQz = -kz - p4*cos4,
		   EQ = std::sqrt(HQxp*HQxp+HQyp*HQyp+HQz*HQz+M2);
	// --- randomize the azimuthal angle phi4----
	double phi4 = dist_phi4(gen);
	double cos_phi4 = std::cos(phi4), sin_phi4 = std::sin(phi4);
	double kx = kxp*cos_phi4 + kyp*sin_phi4, ky = -kxp*sin_phi4 + kyp*cos_phi4;
	double HQx = HQxp*cos_phi4 + HQyp*sin_phi4, HQy = -HQxp*sin_phi4 + HQyp*cos_phi4;
	FS.resize(3);
	FS[0].resize(4);
	FS[0][0] = EQ; FS[0][1] = HQx;
	FS[0][2] = HQy; FS[0][3] = HQz;

	FS[1].resize(4);
	FS[1][0] = sqrts - EQ - k; FS[1][1] = -HQx-kx;
	FS[1][2] = -HQy-ky; FS[1][3] = -HQz-kz;

	FS[2].resize(4);
	FS[2][0] = k; FS[2][1] = kx;
	FS[2][2] = ky; FS[2][3] = kz;
	delete [] p;
	delete [] guessl;
	delete [] guessh;
}


//============Derived 3->2 Xsection class===================================
// Go to the center of mass frame of p1 + p2 + k
// Tabulate variables:
// 1. the center of mass energy sqrts = sqrt((p1 + p2 + k)*(p1 + p2 + k))
// 2. T
// 3. a1 = x2 + xk in (0.5, 1.0)
// 4. a2 = (x2 - xk)/(1 - x2 - xk) in (-1, 1)
// where p2 momentum fraction x2 = |p2|/(|p1| + |p2| + |k|)
// and k momentum fraction xk = |k|/(|p1| + |p2| + |k|)

f_3to2::f_3to2(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh)
:	Xsection(dXdPS_, M1_, name_, refresh), rd(), gen(rd()), dist_phi4(0.0, 2.0*M_PI),
	Nsqrts(40), NT(8), Na1(10), Na2(10),
	sqrtsL(M1_*1.01), sqrtsH(M1_*30.), dsqrts((sqrtsH-sqrtsL)/(Nsqrts-1.)),
	TL(0.12), TH(0.8), dT((TH-TL)/(NT-1.)),
	a1L(0.501), a1H(0.999), da1((a1H-a1L)/(Na1-1.)),
	a2L(-0.999), a2H(0.999), da2((a2H-a2L)/(Na2-1.)),
	Xtab(boost::extents[Nsqrts][NT][Na1][Na2])
{

	bool fileexist = boost::filesystem::exists(name_);
	if ( (!fileexist) || ( fileexist && refresh) ){
		std::cout << "# Populating table with new calculation" << std::endl;
		std::vector<std::thread> threads;
		size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
		size_t call_per_core = size_t(NT*1./Ncores);
		size_t call_for_last_core = NT - call_per_core*(Ncores-1);
		auto code = [this](size_t NTstart_, size_t dNT_) { this->tabulate(NTstart_, dNT_); };
		for (size_t i=0; i< Ncores ; i++){
			size_t Nstart = i*call_per_core;
			size_t dN = (i==Ncores-1)? call_for_last_core : call_per_core;
			threads.push_back( std::thread(code, Nstart, dN) );
		}
		for (std::thread& t : threads)	t.join();
		save_to_file(name_, "Xsection-tab");
	}
	else{
		std::cout << "# loading existing table" << std::endl;
		read_from_file(name_, "Xsection-tab");
	}
	std::cout << std::endl;
}

void f_3to2::save_to_file(std::string filename, std::string datasetname){
	const size_t rank = 4;

	H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
	hsize_t dims[rank] = {Nsqrts, NT, Na1, Na2};
	H5::DSetCreatPropList proplist{};
	proplist.setChunk(rank, dims);

	H5::DataSpace dataspace(rank, dims);
	auto datatype(H5::PredType::NATIVE_DOUBLE);
	H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
	dataset.write(Xtab.data(), datatype);

	// Attributes
	hdf5_add_scalar_attr(dataset, "sqrts_low", sqrtsL);
	hdf5_add_scalar_attr(dataset, "sqrts_high", sqrtsH);
	hdf5_add_scalar_attr(dataset, "N_sqrt_half", Nsqrts);

	hdf5_add_scalar_attr(dataset, "T_low", TL);
	hdf5_add_scalar_attr(dataset, "T_high", TH);
	hdf5_add_scalar_attr(dataset, "N_T", NT);

	hdf5_add_scalar_attr(dataset, "a1_low", a1L);
	hdf5_add_scalar_attr(dataset, "a1_high", a1H);
	hdf5_add_scalar_attr(dataset, "N_a1", Na1);

	hdf5_add_scalar_attr(dataset, "a2_low", a2L);
	hdf5_add_scalar_attr(dataset, "a2_high", a2H);
	hdf5_add_scalar_attr(dataset, "N_a2", Na2);
	file.close();
}

void f_3to2::read_from_file(std::string filename, std::string datasetname){
	const size_t rank = 4;

	H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = file.openDataSet(datasetname.c_str());

	hdf5_read_scalar_attr(dataset, "sqrts_low", sqrtsL);
	hdf5_read_scalar_attr(dataset, "sqrts_high", sqrtsH);
	hdf5_read_scalar_attr(dataset, "N_sqrt_half", Nsqrts);
	dsqrts = (sqrtsH-sqrtsL)/(Nsqrts-1.);

	hdf5_read_scalar_attr(dataset, "T_low", TL);
	hdf5_read_scalar_attr(dataset, "T_high", TH);
	hdf5_read_scalar_attr(dataset, "N_T", NT);
	dT = (TH-TL)/(NT-1.);

	hdf5_read_scalar_attr(dataset, "a1_low", a1L);
	hdf5_read_scalar_attr(dataset, "a1_high", a1H);
	hdf5_read_scalar_attr(dataset, "N_a1", Na1);
	da1 = (a1H-a1L)/(Na1-1.);

	hdf5_read_scalar_attr(dataset, "a2_low", a2L);
	hdf5_read_scalar_attr(dataset, "a2_high", a2H);
	hdf5_read_scalar_attr(dataset, "N_a2", Na2);
	da1 = (a1H-a1L)/(Na1-1.);

	Xtab.resize(boost::extents[Nsqrts][NT][Na1][Na2]);
	hsize_t dims_mem[rank];
  	dims_mem[0] = Nsqrts;
  	dims_mem[1] = NT;
	dims_mem[2] = Na1;
	dims_mem[3] = Na2;
	H5::DataSpace mem_space(rank, dims_mem);

	H5::DataSpace data_space = dataset.getSpace();
	dataset.read(Xtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
	file.close();
}

void f_3to2::tabulate(size_t T_start, size_t dnT){
	double * arg = new double[4];
	for (size_t i=0; i<Nsqrts; i++) { arg[0] = std::pow(sqrtsL + i*dsqrts, 2);
		for (size_t j=0; j<NT; j++) { arg[1] = TL + j*dT;
			for (size_t k=0; k<Na1; k++) { arg[2] = a1L + k*da1;
				for (size_t t=0; t<Na2; t++) { arg[3] = a2L + t*da2;
					Xtab[i][j][k][t] = calculate(arg)/approx_X32(arg, M1);
				}
			}
		}
	}
	delete [] arg;
}

double f_3to2::interpX(double * arg){
	double s = arg[0];
	double sqrts = std::sqrt(s), Temp = arg[1], a1 = arg[2], a2 = arg[3], dt = arg[4];
	if (sqrts < sqrtsL) sqrts = sqrtsL;
	if (sqrts >= sqrtsH) sqrts = sqrtsH-dsqrts;
	if (Temp < TL) Temp = TL;
	if (Temp >= TH) Temp = TH-dT;
	if (a1 < a1L) a1 = a1L;
	if (a1 >= a1H) a1 = a1H-da1;
	if (a2 < a2L) a2 = a2L;
	if (a2 >= a2H) a2 = a2H-da2;

	double xT, rT,
		   xsqrts, rsqrts,
		   xa1, ra1,
		   xa2, ra2;
	size_t isqrts, iT, ia1, ia2;
	xsqrts = (sqrts-sqrtsL)/dsqrts;	isqrts = floor(xsqrts); rsqrts = xsqrts - isqrts;
	xT = (Temp-TL)/dT;	iT = floor(xT); rT = xT - iT;
	xa1 = (a1-a1L)/da1;	ia1 = floor(xa1); ra1 = xa1 - ia1;
	xa2 = (a2-a2L)/da2;	ia2 = floor(xa2); ra2 = xa2 - ia2;

	double raw_result = interpolate4d(&Xtab, isqrts, iT, ia1, ia2, rsqrts, rT, ra1, ra2)*approx_X32(arg, M1);

	double xk = 0.5*(a1*a2 + a1 - a2);
	double x2 = 0.5*(-a1*a2 + a1 + a2);
	double M2 = M1*M1;
	double A = (2.*xk/x2 - 1.), B = -2.*sqrts*(1. + xk/x2), C = s - M2;
	double E2 = (-B - std::sqrt(B*B-4.*A*C))/2./A;
	double k = xk/x2*E2;
	double p1 = (1. - x2 - xk)/x2*E2;
	double E1 = std::sqrt(p1*p1 + M2);
	double cosk = (E2*E2-k*k-p1*p1)/2./p1/k;
	double kz = k*cosk;
	double kt2 = k*k - kz*kz;
	double frac = (k + kz)/(E1 + p1+k+kz);
	double fracbar = (k + std::abs(kz))/(E1 + p1+k + std::abs(kz));
	double x2M2 = frac*frac*M2;
	double coeff_mg2 = (1.-fracbar)*t_channel_mD2->get_mD2(Temp)/2.;
	double tauk = 2.*k*(1.-fracbar)/(kt2+x2M2+coeff_mg2);
	// here u is the ratio of the mean-free-path over the formation length
	// mean-free-path \sim mean-free-time*v_HQ,
	// v_HQ = p1/E1
	// formation length = tau_k*v_k = tau_k
	double u = dt/tauk*p1/E1;
	double alpha_rad = alpha_s(kt2);

	return 1.5/M_PI*(1. - M1*M1/s) * alpha_rad * f_LPM(u) * std::pow(1./(1. + 0*fracbar), 2) * raw_result;
}

//------Integration function-------------------

double df_dcostheta42_dphi42(double phi42, void * params_){
	Mygsl_integration_params * params = static_cast<Mygsl_integration_params *>(params_);
	double arg[2] = {params->params[10], phi42};
	return params->f(arg, 2, params->params);
}

double df_dcostheta42(double costheta42, void * params_){
	Mygsl_integration_params * params = static_cast<Mygsl_integration_params *>(params_);
	params->params[10] = costheta42;

	double result, error;
	double phi42min = 0., phi42max = M_PI;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(500);

    gsl_function F;
	F.function = df_dcostheta42_dphi42;
	F.params = params;
	gsl_integration_qag(&F, phi42min, phi42max, 0, 1e-3, 500, 3, w, &result, &error);
	gsl_integration_workspace_free(w);
	return 2.*result;
}

double f_3to2::calculate(double * arg){
	double s = arg[0], Temp = arg[1], a1 = arg[2], a2 = arg[3];
	double sqrts = std::sqrt(s);
	double xk = 0.5*(a1*a2 + a1 - a2);
	double x2 = 0.5*(-a1*a2 + a1 + a2);
	double M2 = M1*M1;
	double A = (2.*xk/x2 - 1.), B = -2.*sqrts*(1. + xk/x2), C = s - M2;
	double E2 = (-B - std::sqrt(B*B-4.*A*C))/2./A, E4 = (s-M2)/2./sqrts;

	double k = xk/x2*E2;
	double p1 = (1. - x2 - xk)/x2*E2;
	double E1 = std::sqrt(p1*p1 + M2);
	double cosk = (E2*E2-k*k-p1*p1)/2./p1/k;
	double cos2 = (-E2*E2+k*k-p1*p1)/2./p1/E2;
	double kz = k*cosk;
	double kt2 = k*k - kz*kz;
	double frac = (k+kz)/(E1+p1+k+kz);
	double fracbar = (k+std::abs(kz))/(E1+p1+k+std::abs(kz));
	double x2M2 = frac*frac*M2;
	double coeff_mg2 = (1.-fracbar)*t_channel_mD2->get_mD2(Temp)/2.;


	// Integration for (1)p4 and (2)phi4
	double result, error;
	double costheta42min = -1., costheta42max = 1.;
	gsl_integration_workspace *w = gsl_integration_workspace_alloc(200);
	Mygsl_integration_params * params_df = new Mygsl_integration_params;
	params_df->f = dXdPS;
	params_df->params = new double[11];
	params_df->params[0] = s;
	params_df->params[1] = Temp;
	params_df->params[2] = M1;
	params_df->params[3] = E2;
	params_df->params[4] = E4;
	params_df->params[5] = k;
	params_df->params[6] = cos2;
	params_df->params[7] = cosk;
	params_df->params[8] = x2M2;
	params_df->params[9] = coeff_mg2;
	params_df->params[10] = 0.0;

	gsl_function F;
	F.function = df_dcostheta42;
	F.params = params_df;
	gsl_integration_qag(&F, costheta42min, costheta42max, 0, 1e-2, 200, 3, w, &result, &error);

	gsl_integration_workspace_free(w);
	delete [] params_df->params;
	delete params_df;
	return result;
}

void f_3to2::sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS){
	double s = arg[0], Temp = arg[1], a1 = arg[2], a2 = arg[3];
	double sqrts = std::sqrt(s);
	double xk = 0.5*(a1*a2 + a1 - a2);
	double x2 = 0.5*(-a1*a2 + a1 + a2);
	double M2 = M1*M1;
	double A = (2.*xk/x2 - 1.), B = -2.*sqrts*(1. + xk/x2), C = s - M2;
	double E2 = (-B - std::sqrt(B*B-4.*A*C))/2./A, E4 = (s-M2)/2./sqrts;
	double k = xk/x2*E2, p1 = (1. - x2 - xk)/x2*E2;
	double E1 = std::sqrt(p1*p1 + M2);
	double cos21 = (k*k-E2*E2-p1*p1)/2./p1/E2;
	double sin21 = std::sqrt(1. - cos21*cos21);
	double cosk = (E2*E2-k*k-p1*p1)/2./p1/k;
	double kz = k*cosk;
	double kt2 = k*k - kz*kz;
	double frac = (k+kz)/(E1+p1+k+kz);
	double fracbar = (k+std::abs(kz))/(E1+p1+k+std::abs(kz));
	double x2M2 = frac*frac*M2;
	double coeff_mg2 = (1.-fracbar)*t_channel_mD2->get_mD2(Temp)/2.;
	double * params = new double[10];
	params[0] = s; params[1] = Temp; params[2] = M1; params[3] = E2; params[4] = E4;
	params[5] = k; params[6] = cos21; params[7] = cosk; params[8] = x2M2; params[9] = coeff_mg2;
	// sample costheta_24, phi_24
	double * guessl = new double[2];
	double * guessh = new double[2];
	guessl[0] = -0.1; guessl[1] = M_PI*0.9;
	guessh[0] = 0.1; guessh[1] = M_PI*1.1;
	std::vector<double> result = sampler.sample(dXdPS, 2, params, guessl, guessh);
	double costheta_24 = result[0], phi_24 = result[1];
	double sintheta_24 = std::sqrt(1. - costheta_24*costheta_24);
	double cosphi_24 = std::cos(phi_24), sinphi_24 = std::sin(phi_24);

	// transform to final states E3(Q), E4(q, g)
	FS.resize(2);
	FS[0].resize(4);
	FS[0][0] = sqrts - E4;
	FS[0][1] = -E4*(cos21*sintheta_24*cosphi_24 + sin21*costheta_24);
	FS[0][2] = -E4*sintheta_24*sinphi_24;
	FS[0][3] = -E4*(-sin21*sintheta_24*cosphi_24 + cos21*costheta_24);

	FS[1].resize(4);
	FS[1][0] = E4;
	FS[1][1] = -FS[0][1];
	FS[1][2] = -FS[0][2];
	FS[1][3] = -FS[0][3];

	delete[] params;
	delete[] guessl;
	delete[] guessh;
}
