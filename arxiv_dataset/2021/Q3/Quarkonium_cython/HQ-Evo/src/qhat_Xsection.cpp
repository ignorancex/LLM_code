#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <vector>
#include <string>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include <boost/filesystem.hpp>

#include "utility.h"
#include "qhat_Xsection.h"


double gsl_1dfunc_wrapper_YX(double x, void *params_)
{
        YXgsl_integration_params *p = static_cast<YXgsl_integration_params*>(params_);
        return p->f(&x, 1, p->params);
}


double approx_QhatX22(double * arg, double M){	
	double s = arg[0], T = arg[1];
	return 1.0;///T/T;/std::pow(1.0 - M*M/s, 2)
}

// ===== QhatXsection base class
QhatXsection::QhatXsection(double (*dXdPS_)(double* , size_t, void*), double M1_, std::string name_, bool refresh)
:  dXdPS(dXdPS_), M1(M1_)
{
        std::cout << "--------" << __func__ << "  " << name_ << "--------" << std::endl;
}



// ==== derived 2->2 QhatXsection class ========
QhatXsection_2to2::QhatXsection_2to2(double (*dXdPS_)(double*, size_t, void*), double M1_, std::string name_, bool refresh)
:    QhatXsection(dXdPS_, M1_, name_, refresh),
     Nsqrts(50), NT(32),
     sqrtsL(M1_*1.01), sqrtsM(M1_*5.), sqrtsH(M1_*40.),
     dsqrts1((sqrtsM-sqrtsL)/(Nsqrts-1.)), dsqrts2((sqrtsH - sqrtsM)/(Nsqrts - 1.)),
     TL(0.12), TH(0.8), dT((TH-TL)/(NT-1.)),
     QhatXtab(boost::extents[6][Nsqrts*2][NT])
{
        bool fileexist = boost::filesystem::exists(name_);
        if ( (!fileexist) || (fileexist && refresh))
        {
                std::cout << "Populating table with new calculation" << std::endl;
                std::vector<std::thread> threads;
                size_t Ncores = std::min(size_t(std::thread::hardware_concurrency()), NT);
                size_t call_per_core = size_t(NT*1./Ncores);
                size_t call_for_last_core = NT - call_per_core * (Ncores- 1);
                auto code = [this](size_t NTstart_, size_t dNT_) {this->tabulate(NTstart_, dNT_);};
		for (size_t i=0; i < Ncores; i++)
                {
                        size_t Nstart = i * call_per_core;
                        size_t dN = (i==Ncores -1) ? call_for_last_core : call_per_core;
                        threads.push_back(std::thread(code, Nstart, dN));
                }
                for (std::thread& t: threads) t.join();
                save_to_file(name_, "QhatXsection-tab");
        }
        else
        {
                std::cout << "loading existing table" << std::endl;
                read_from_file(name_, "QhatXsection-tab");
        }
        std::cout << std::endl;
}





void QhatXsection_2to2::save_to_file(std::string filename, std::string datasetname)
{
        const size_t rank = 3;
        H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
        hsize_t dims[rank] = {6,Nsqrts*2, NT};
        H5::DSetCreatPropList proplist{};
        proplist.setChunk(rank, dims);

        H5::DataSpace dataspace(rank, dims);
        auto datatype(H5::PredType::NATIVE_DOUBLE);
        H5::DataSet dataset = file.createDataSet(datasetname.c_str(), datatype, dataspace, proplist);
        dataset.write(QhatXtab.data(), datatype);


        hdf5_add_scalar_attr(dataset, "sqrts_low", sqrtsL);
        hdf5_add_scalar_attr(dataset, "sqrts_mid", sqrtsM);
        hdf5_add_scalar_attr(dataset, "sqrts_high", sqrtsH);
        hdf5_add_scalar_attr(dataset, "N_sqrt_half", Nsqrts);

        hdf5_add_scalar_attr(dataset, "T_low", TL);
        hdf5_add_scalar_attr(dataset, "T_high", TH);
        hdf5_add_scalar_attr(dataset, "N_T", NT);
}



void QhatXsection_2to2::read_from_file(std::string filename, std::string datasetname)
{
        const size_t rank=3;
        H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetname.c_str());
        hdf5_read_scalar_attr(dataset, "sqrts_low", sqrtsL);
        hdf5_read_scalar_attr(dataset, "sqrts_mid", sqrtsM);
        hdf5_read_scalar_attr(dataset, "sqrts_high", sqrtsH);
        hdf5_read_scalar_attr(dataset, "N_sqrt_half", Nsqrts);
        dsqrts1 = (sqrtsM - sqrtsL)/(Nsqrts - 1.);
        dsqrts2 = (sqrtsH - sqrtsM)/(Nsqrts - 1.);

        hdf5_read_scalar_attr(dataset, "T_low", TL);
        hdf5_read_scalar_attr(dataset, "T_high", TH);
        hdf5_read_scalar_attr(dataset, "N_T", NT);
        dT = (TH - TL)/ (NT-1.);

        QhatXtab.resize(boost::extents[6][Nsqrts*2][NT]);
        hsize_t dims_mem[rank];
        dims_mem[0] = 6;
        dims_mem[1] = 2*Nsqrts;
        dims_mem[2] = NT;
        H5::DataSpace mem_space(rank, dims_mem);

        H5::DataSpace data_space = dataset.getSpace();
        dataset.read(QhatXtab.data(), H5::PredType::NATIVE_DOUBLE, mem_space, data_space);
        //std::cout << "Read in QhatXtab successfully :)" << std::endl;
}

void QhatXsection_2to2::tabulate(size_t T_start, size_t dnT)
{
        double *args = new double[3];
        for (size_t index = 0; index < 6; ++index)
        {
                args[2] = index;
                for (size_t i=0; i<2*Nsqrts; ++i)
                {
                        if (i < Nsqrts) args[0] = std::pow(sqrtsL + i*dsqrts1, 2);
                        else args[0] = std::pow(sqrtsM + (i - Nsqrts) * dsqrts2, 2);
                        for (size_t j = T_start; j < (T_start + dnT); j++)
                        {
                                args[1] = TL + j*dT;
                                QhatXtab[index][i][j] = calculate(args)/approx_QhatX22(args, M1);
                                //QhatXtab[index][i][j] = calculate(args);
                        }
                }
        }
        delete [] args;
}





double QhatXsection_2to2::interpX(double* args)
{
        double sqrts = std::sqrt(args[0]), Temp = args[1];
        int index = static_cast<int>(args[2]); //floor double into integer

        if (Temp < TL) Temp = TL;
        if (Temp >= TH) Temp = TH - dT;
        if (sqrts < sqrtsL) sqrts = sqrtsL;
        if (sqrts >= sqrtsH) sqrts = sqrtsH - dsqrts2;

        double xT, rT, xsqrts, rsqrts, dsqrts, sqrtsmin;
        size_t iT, isqrts, Noffsets;
        if (sqrts < sqrtsM)
        {dsqrts = dsqrts1; sqrtsmin = sqrtsL; Noffsets = 0;}
        else
        {dsqrts = dsqrts2; sqrtsmin = sqrtsM; Noffsets = Nsqrts;}

        xsqrts = (sqrts - sqrtsmin)/dsqrts; isqrts = floor(xsqrts);  rsqrts = xsqrts-isqrts; isqrts += Noffsets;
        xT = (Temp - TL)/dT; iT = floor(xT);  rT = xT-iT;

//        std::cout << "interpX: "<< iT << " " << isqrts <<" " << interpolate2d_YX(&QhatXtab, index, isqrts, iT, rsqrts, rT) <<std::endl;
        return approx_QhatX22(args, M1) * interpolate2d_YX(&QhatXtab, index, isqrts, iT, rsqrts, rT);
//        return interpolate2d_YX(&QhatXtab, index, isqrts, iT, rsqrts, rT);
}


double QhatXsection_2to2::calculate(double* args)
{
        double s = args[0], Temp = args[1];
        int index = static_cast<int>(args[2]);  // floor double into integer

        double result, error, tmin, tmax;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(5000);
        YXgsl_integration_params * params = new YXgsl_integration_params;
        params->f = dXdPS;
        double* p = new double[4];
        p[0] = s;
        p[1] = Temp;
        p[2] = M1;
        p[3] = index;

        params->params = p;

        gsl_function F;
        F.function = gsl_1dfunc_wrapper_YX;
        F.params = params;
        tmax = 0.0;
        tmin = -pow(s-M1*M1,2)/s;
        gsl_integration_qag(&F, tmin, tmax, 0, 1e-4, 5000, 6, w, &result, &error);
        
        delete [] p;
        delete params;
        gsl_integration_workspace_free(w);

        return result;

}


