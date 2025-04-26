#include <cmath>
#include <vector>
#include <thread>
#include <fstream>
#include <string>

#include <boost/filesystem.hpp>
#include <gsl/gsl_errno.h>

#include "utility.h"
#include "qhat.h"
#include "TLorentz.h"
#include "H5Cpp.h"

using std::placeholders::_1;
using std::vector;

vector<double> transform_from_CoM_array2(double *args)
// args ={sqrts, M2, E1, E2, cos_theta12}
// return Boost*Rotation([1][1], [1][3], [3][1], [3][3])
{
        double sqrts = args[0];
        double M2 = args[1];
        double E1 = args[2];
        double E2 = args[3];
        double Etotal = E1 + E2;
        double cos_theta12 = args[4];
        double sin_theta12 = std::sqrt(1. - cos_theta12*cos_theta12);
        double betax = E2 * sin_theta12 / Etotal;
        double pz_cell = std::sqrt(E1*E1 - M2);

        double betaz = (pz_cell + E2 * cos_theta12) / Etotal;
        double gamma = Etotal/sqrts;

        //double lambda = (0.5*(M2/sqrts + sqrts) + E1) / (E1 + E2 + sqrts);
        //double pz_prime = pz_cell - lambda * (pz_cell + E2 * cos_theta12);
        //double px_prime = -lambda*(E2*sin_theta12);

        double gamma2 = gamma*gamma/(1. + gamma);
        double pz_prime = -gamma*betaz* E1 + (1 + gamma2*betaz*betaz) * pz_cell;

        double pz_com = 0.5*(sqrts - M2/sqrts);

        double cosTheta13 = pz_prime / pz_com;
        //std::cout << cosTheta13 << std::endl;
        double sinTheta13 = std::sqrt(1. - cosTheta13 * cosTheta13);

        vector<double> botMatrix(4);
        double gammaBetaxx = gamma2*betax*betax;
        double gammaBetaxz = gamma2*betax*betaz;
        double gammaBetazz = gamma2*betaz*betaz;

        //double check = gamma*gamma*(1-betax*betax - betaz*betaz) - 1.0;
        //std::cout << "transformation: " << check << std::endl;
        botMatrix[0] = (1. + gammaBetaxx) * cosTheta13 + gammaBetaxz * sinTheta13 ;
        botMatrix[1] = -(1. + gammaBetaxx) * sinTheta13 + gammaBetaxz * cosTheta13 ;
        botMatrix[2] = gammaBetaxz *cosTheta13 + (1. + gammaBetazz) * sinTheta13 ;
        botMatrix[3] = - gammaBetaxz*sinTheta13 + (1. + gammaBetazz) * cosTheta13 ;
        //std::cout << "transform: " << botMatrix[0] << " " << botMatrix[1] << " " << botMatrix[2] << " " << botMatrix[3] << std::endl;
        return botMatrix;
}




//==== Thermal distribution function
double inline f0(double x, double xi)
{
        if (x < 1e-9) x = 1e-9;
        double result = 1./(std::exp(x) + xi);
        return result;
}



// == function wrapper for GSL integration Raa ===
double viscosfn(double costheta, int n)
{
        if (n==0) return 1.;
        else return costheta*costheta;
}

double viscosgn(double costheta, int n)
{
        if (n==0) return 1.;
        if (n==1) return -0.5;
        else return 1.5*costheta*costheta;
}



double fy_wrapper22_YX(double y, void *params_)
{
        integrate_params_2_YX *params = static_cast<integrate_params_2_YX *>(params_);
        double coeff = params->params[0]; // coeff = 2*E1*E2;
        double Temp = params->params[1];
        double M2 = params->params[2];
        double v1 = params->params[3];
        double E1 = params->params[4];
        int qidx = int(params->params[5] + 0.5);

        double s = M2 + coeff * (1. - v1*y);
        double *args = new double[3];
        args[0] = s; args[1] = Temp;
        //args[2] = 0; double QhatXsection_q1 = params->f(args);
        args[2] = 1; double QhatXsection_q3 = params->f(args);
        args[2] = 2; double QhatXsection_q11 = params->f(args);
        args[2] = 3; double QhatXsection_q33 = params->f(args);
        //args[2] = 4; double QhatXsection_q13 = params->f(args);
        delete [] args;

        double E2 = coeff/(2.*E1);
        double* args_ = new double[5];
        args_[0] = std::sqrt(s); args_[1] = M2; args_[2] = E1; args_[3] = E2; args_[4] = y;
        vector<double> botMatrix = transform_from_CoM_array2(args_); // [1][1], [1][3], [3][1], [3][3]
        delete[] args_;

        /*
        double Qhat_q1cell = botMatrix[0]*QhatXsection_q1 + botMatrix[1]*QhatXsection_q3;
        double Qhat_q3cell = botMatrix[2]*QhatXsection_q1 + botMatrix[3]*QhatXsection_q3;
        double Qhat_q11cell = pow(botMatrix[0], 2)*QhatXsection_q11 + pow(botMatrix[1],2)*QhatXsection_q33 + 2*botMatrix[0]*botMatrix[1]*QhatXsection_q13;
        double Qhat_q33cell = pow(botMatrix[2], 2)*QhatXsection_q11 + pow(botMatrix[3],2)*QhatXsection_q33 + 2*botMatrix[2]*botMatrix[3]*QhatXsection_q13;
        */

        double Qhat_q1cell = botMatrix[1]*QhatXsection_q3;
        double Qhat_q3cell = botMatrix[3]*QhatXsection_q3;
        double Qhat_q11cell = 0.5 * pow(botMatrix[0], 2)*QhatXsection_q11 + pow(botMatrix[1],2)*QhatXsection_q33;
        double Qhat_q33cell = 0.5 * pow(botMatrix[2], 2)*QhatXsection_q11 + pow(botMatrix[3],2)*QhatXsection_q33;


        //std::cout << "wrapper_fy: " << botMatrix[0] << " " << botMatrix[1] << " " << botMatrix[2] << " " << botMatrix[3] << std::endl;

        double Qhat[] = {Qhat_q1cell, Qhat_q3cell, Qhat_q11cell, Qhat_q33cell};
        //double Qhat[] = {QhatXsection_q1, QhatXsection_q3, QhatXsection_q11, QhatXsection_q33};
        
        return Qhat[qidx] * (1. - v1*y);

}




double fx_wrapper22_YX(double x, void *px_)
{
        integrate_params_2_YX * px = static_cast<integrate_params_2_YX *>(px_);
        double E1 = px->params[0];
        double v1 = px->params[1];
        double Temp = px->params[2];
        double M2 = px->params[3];
        double zeta = px->params[4];

        int qidx = int(px->params[5]);


        {
        double result, error, ymin, ymax;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(10000);
        gsl_error_handler_t * old_handler = gsl_set_error_handler_off();
        integrate_params_2_YX* py = new integrate_params_2_YX;
        py->f = px->f;
        py->params = new double[7];
        py->params[0] = 2.*E1*x*Temp;
        py->params[1] = Temp;
        py->params[2] = M2;
        py->params[3] = v1;
        py->params[4] = E1;
        py->params[5] = qidx;

        gsl_function F;
        F.function = fy_wrapper22_YX;
        F.params = py;
        ymax = 1.;
        ymin = -1.;
        int status = 1, nloop = 0;

        while(status && nloop < 5)
        {
                status = gsl_integration_qag(&F, ymin, ymax, 0, 1e-3, 10000,6, w, &result, &error);
                nloop += 1;
        }


        delete [] py->params;
        delete py;
       
        gsl_set_error_handler(old_handler);
        gsl_integration_workspace_free(w);
       
        return x*x*f0(x, zeta)*result;
       }

}





// ======== qhat abstract class
Qhat::Qhat(std::string name_)
{
        std::cout << "-------" << __func__ << "  " << name_ << "--------" << std::endl;
}



// ======= derived scattering rate 2-> 2 class
Qhat_2to2::Qhat_2to2(QhatXsection_2to2 * Xprocess_, int degeneracy_, double eta_2_, std::string name_, bool refresh)
:  Qhat(name_), Xprocess(Xprocess_), M(Xprocess->get_M1()),
   degeneracy(degeneracy_), eta_2(eta_2_),
   NE(50), NT(10), E1L(M*1.01), E1M(M*20), E1H(M*100), TL(0.15), TH(0.60),
   dE1((E1M - E1L)/(NE -1.)), dE2((E1H - E1M)/(NE -1.)),
   dT((TH - TL)/(NT -1.)),
   QhatTab(boost::extents[4][2*NE][NT])
{
        bool fileexist = boost::filesystem::exists(name_);
        if ((!fileexist) || (fileexist && refresh))
        {
                std::cout << "Populating table with new calculation" << std::endl;
                std::vector<std::thread> threads;
                size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
                    
                size_t call_per_core = int(NT*1./Ncores);
                size_t call_for_last_core = NT - call_per_core*(Ncores -1);
                //std::cout << Ncores << " " << call_per_core <<  " " << call_for_last_core << std::endl;
                for (size_t i=0; i< Ncores; i++)
                {
                        size_t Nstart = i*call_per_core;
                        size_t dN = (i==Ncores-1) ? call_for_last_core : call_per_core;
                        auto code = [this](size_t NTstart_, size_t dNT_) {this->tabulate_E1_T(NTstart_, dNT_);};
                        threads.push_back(std::thread(code, Nstart, dN));
                }

                for (std::thread& t: threads) t.join();

                
                save_to_file(name_, "Qhat-tab");
        }
        else
        {
                std::cout << "loading existing table " << std::endl;
                read_from_file(name_, "Qhat-tab");
        }
        std::cout << std::endl;
}



void Qhat_2to2::save_to_file(std::string filename, std::string datasetname)
{
		H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
        const size_t rank=3;
        hsize_t dims[rank] = {3, 2*NE, NT};
        H5::DSetCreatPropList proplist{};
        proplist.setChunk(rank, dims);

        H5::DataSpace dataspace(rank, dims);
        auto datatype(H5::PredType::NATIVE_DOUBLE);
        H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
        dataset.write(QhatTab.data(), datatype);

        hdf5_add_scalar_attr(dataset, "E1_low", E1L);
        hdf5_add_scalar_attr(dataset, "E1_high", E1H);
        hdf5_add_scalar_attr(dataset, "E1_mid", E1M);
        hdf5_add_scalar_attr(dataset, "N_E1_half", NE);

        hdf5_add_scalar_attr(dataset, "T_low", TL);
        hdf5_add_scalar_attr(dataset, "T_high", TH);
        hdf5_add_scalar_attr(dataset, "N_T", NT);
		file.close();
}



void Qhat_2to2::read_from_file(std::string  filename, std::string datasetname)
{
		H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
        const size_t rank=3;
        H5::DataSet dataset = file.openDataSet(datasetname.c_str());
        hdf5_read_scalar_attr(dataset, "E1_low", E1L);
        hdf5_read_scalar_attr(dataset, "E1_high", E1H);
        hdf5_read_scalar_attr(dataset, "E1_mid", E1M);
        hdf5_read_scalar_attr(dataset, "N_E1_half", NE);
        dE1 = (E1M - E1L) / (NE -1.);
	dE2 = (E1H - E1M) / (NE - 1.);

        hdf5_read_scalar_attr(dataset, "T_low", TL);
        hdf5_read_scalar_attr(dataset, "T_high", TH);
        hdf5_read_scalar_attr(dataset, "N_T", NT);
        dT = (TH - TL)/(NT -1.);

        QhatTab.resize(boost::extents[3][2*NE][NT]);

        hsize_t dims_mem[rank];
        dims_mem[0] = 3;
        dims_mem[1] = 2*NE;
        dims_mem[2] = NT;

        H5::DataSpace mem_space(rank, dims_mem);

        H5::DataSpace data_space = dataset.getSpace();
        dataset.read(QhatTab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
		file.close();
}


void Qhat_2to2::tabulate_E1_T(size_t T_start, size_t dnT)
{
        double *args = new double[3];
        for (size_t i=0; i < 2*NE; ++i)
        {
        	if (i < NE) args[0] = E1L + i * dE1;
		else args[0] = E1M + (i-NE)*dE2;
                for (size_t j = T_start; j < (T_start + dnT) ; ++j)
                {
                        args[1] = TL + j * dT;
                        args[2] = 1; double drag = calculate(args); // dpz/dt
                        args[2] = 2; double kperp = calculate(args); // dpx^2/dt
                        args[2] = 3; double kpara = calculate(args); // dpz^2/dt
						args[2] = 5; double R = calculate(args); // 1/dt
                        QhatTab[0][i][j] = drag;
                        QhatTab[1][i][j] = kperp;
                        QhatTab[2][i][j] = kpara - drag*drag/R;
                        //if (i == 0) std::cout << i <<  " " << j <<" " <<  TL << " " << dT << " " << args[0] << " " << args[1] << " " << QhatTab[0][i][j] << " "<< QhatTab[1][i][j] << " " << QhatTab[2][i][j] << std::endl;
                           
                }
        }
        
        delete [] args;
}




double Qhat_2to2::interpQ(double * args)
{
        double E1 = args[0], Temp = args[1];
        int qidx = int(args[2]);

        if (Temp < TL) Temp = TL;
        if (Temp >= TH) Temp = TH - dT;
        if (E1 < E1L) E1 = E1L;
        if (E1 > E1H) E1  = E1H - dE1;

        double xT, rT, xE, rE, dE, Emin;
        size_t iT, iE, Noffset;
	if (E1 < E1M)
	{dE=dE1; Emin = E1L; Noffset = 0;}
	else
	{dE=dE2; Emin = E1M; Noffset =NE;}

        xT = (Temp - TL)/dT;    iT = floor(xT);     rT = xT - iT;
        xE = (E1 - Emin)/dE;    iE = floor(xE);     rE = xE - iE; iE += Noffset;

        return interpolate2d_YX(&QhatTab, qidx, iE, iT, rE, rT); 
}




double Qhat_2to2::calculate(double *args)
{
        double E1 = args[0], Temp = args[1];
        int qidx = int(args[2]);
        double p1 = std::sqrt(E1*E1 - M*M);
        double result, error, xmin, xmax;

        gsl_error_handler_t * old_handler = gsl_set_error_handler_off();

        gsl_integration_workspace * w = gsl_integration_workspace_alloc(5000);
        integrate_params_2_YX * px = new integrate_params_2_YX;
        px->f = std::bind(&QhatXsection_2to2::interpX, Xprocess, _1);
        px->params = new double[6];
        px->params[0] = E1;
        px->params[1] = p1/E1;
        px->params[2] = Temp;
        px->params[3] = M*M;
        px->params[4] = eta_2;
        px->params[5] = qidx;

        gsl_function F;
        F.function = fx_wrapper22_YX;
        F.params=px;
        xmax = 10.0;
        xmin = 0.0;
        int status = 1, nloop = 0;
        while(status && nloop < 5)
        {
        status = gsl_integration_qag(&F, xmin, xmax, 0, 1e-3, 5000, 6, w, &result, &error);
        nloop += 1;
        }

        gsl_set_error_handler(old_handler);
        gsl_integration_workspace_free(w);
        delete [] px->params;
        delete px;
        //if ((E1-1.313)<0.001 && (Temp-0.13)<0.001 && iweight==0) std::cout << "qhat calculate: " << E1 << " " << Temp << " " <<result << std::endl;
        return result*std::pow(Temp, 3)*4 / c16pi2 * degeneracy;
}
