#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <boost/multi_array.hpp>
#include <H5Cpp.h>

#include "matrix_elements.h"
#include "Xsection.h"
#include "rates.h"
//#include "qhat_Xsection.h"
#include "qhat.h"
#include "Langevin.h"


using std::vector;

int main()
{
        double M = 1.3;

        double temp = 0.3;

        //Xsection_2to2 xQq2Qq(&dX_Qq2Qq_dPS, &approx_XQq2Qq, M, "XQq2Qq.hdf5", false);
        //Xsection_2to2 xQg2Qg(&dX_Qg2Qg_dPS, &approx_XQg2Qg, M, "XQg2Qg.hdf5", false);
        //rates_2to2 rQq2Qq(&xQq2Qq, 36, 0., "rQq2Qq.hdf5", false);
        //rates_2to2 rQg2Qg(&xQg2Qg, 16, 0., "rQg2Qg.hdf5", false);

        bool refresh = true;
        QhatXsection_2to2 qhat_xQq2Qq(&dqhat_Qq2Qq_dPS, &approx_XQq2Qq, M, "qhat_XQq2Qq.hdf5", refresh);
        QhatXsection_2to2 qhat_xQg2Qg(&dqhat_Qg2Qg_dPS, &approx_XQg2Qg, M, "qhat_XQg2Qg.hdf5", refresh);
        Qhat_2to2 qhatQq2Qq(&qhat_xQq2Qq, 36, 0., "qhat_Qq2Qq.hdf5", refresh);
        Qhat_2to2 qhatQg2Qg(&qhat_xQg2Qg, 16, 0., "qhat_Qg2Qg.hdf5", refresh);

/*

        std::cout << "read in table successfully :) " << std::endl;

        particle testHQ;
    
        double E1 = 10.;
    
        testHQ.p={E1, 0., 0, std::sqrt(E1*E1 - M*M)};
        testHQ.x={0., 0., 0.};

        double drag_Qq, drag_Qg, drag;
        double kperp_Qq, kperp_Qg, kperp;
        double kpara_Qq, kpara_Qg, kpara;

        int npart = 20000;
        vector<particle> events(npart, testHQ);

        int ntime = 10000;
        double deltat = 0.01;

        int NTau = int(ntime*deltat);


        typedef boost::multi_array<double, 3> array3D;
        array3D part_info(boost::extents[NTau][npart][4]);

*/
        // test 
        /*
        particle HQ;
        HQ.p = testHQ.p;
        HQ.x = testHQ.x;

        double* args = new double[4];
        args[0] = HQ.p[0];
        args[1] = temp;
        args[2] = 0.0;
 
        args[3] = 1;  drag_Qq = qhatQq2Qq.interpQ(args); drag_Qg = qhatQg2Qg.interpQ(args);
        args[3] = 2;  kperp_Qq = qhatQq2Qq.interpQ(args); kperp_Qg = qhatQg2Qg.interpQ(args);
        args[3] = 3;  kpara_Qq = qhatQq2Qq.interpQ(args); kpara_Qg = qhatQg2Qg.interpQ(args);
        delete [] args;
        drag = (drag_Qq + drag_Qg) * 5.068;
        kperp = (kperp_Qq + kperp_Qg) * 5.068;
        kpara = ((kpara_Qq - drag_Qq*drag_Qq) + (kpara_Qg - drag_Qg*drag_Qg) ) * 5.068;

        std::cout << "main: " << HQ.p[0] <<" " <<  drag << " " << kperp << " " << kpara << std::endl;


        //update_by_Langevin_test(HQ, &qhatQq2Qq, &qhatQg2Qg, temp, deltat, true);
        update_by_Langevin(HQ, temp, drag, kpara, kperp, deltat, false);
        std::cout << "main: " << HQ.p[0] <<" " <<  HQ.p[1]<< " " << HQ.p[2] << " " << HQ.p[3] << std::endl;
        */

/*
        for (size_t ipart = 0; ipart < npart; ++ipart)
        {
                particle HQ;
                HQ.p = testHQ.p;
                HQ.x = testHQ.x;

                for (size_t itime=0; itime<ntime; ++itime)
                {
                        update_by_Langevin_test(HQ, &qhatQq2Qq, &qhatQg2Qg, temp, deltat, false);
                        if (itime % 100 == 0)
                        {
                                int idx_t = itime/100;
                                part_info[idx_t][ipart][0] = HQ.p[0];
                                part_info[idx_t][ipart][1] = HQ.p[1];
                                part_info[idx_t][ipart][2] = HQ.p[2];
                                part_info[idx_t][ipart][3] = HQ.p[3];
                        }
                }

        }
      
*/
 
/*
        for (int ipart = 0; ipart < npart; ++ipart)
        {

        particle HQ;
        HQ.p = testHQ.p;
        HQ.x = testHQ.x;

        for (int itime = 0; itime < ntime; ++itime)
        {
                double *args = new double[4];
                args[0] = HQ.p[0];
                args[1] = temp;
                args[2] = 0.0;
                args[3] = 1;  drag_Qq = qhatQq2Qq.interpQ(args); drag_Qg = qhatQg2Qg.interpQ(args);
                args[3] = 2;  kperp_Qq = qhatQq2Qq.interpQ(args); kperp_Qg = qhatQg2Qg.interpQ(args);
                args[3] = 3;  kpara_Qq = qhatQq2Qq.interpQ(args); kpara_Qg = qhatQg2Qg.interpQ(args);

                drag = (drag_Qq + drag_Qg)*5.068;
                kperp = (kperp_Qq + kperp_Qg) * 5.068;
                kpara = ((kpara_Qq - drag_Qq*drag_Qq) + (kpara_Qg - drag_Qg*drag_Qg)) * 5.068;

                delete [] args;

                update_by_Langevin(HQ, temp, drag, kpara, kperp, deltat, false);

                if (itime % 100 == 0)
                {
                    int idx_t = itime/100;
                    part_info[idx_t][ipart][0] = HQ.p[0];
                    part_info[idx_t][ipart][1] = HQ.p[1];
                    part_info[idx_t][ipart][2] = HQ.p[2];
                    part_info[idx_t][ipart][3] = HQ.p[3];
                }
        }
        }
*/


/*

        H5::H5File* file = new H5::H5File("Event_static_EinRFalse_preIto.hdf5", H5F_ACC_TRUNC);
        std::string datasetname("Langevin-events");
        const size_t rank=3;
        hsize_t dims[rank] = {NTau, npart, 4};
        H5::DSetCreatPropList proplist{};
        proplist.setChunk(rank, dims);

        H5::DataSpace dataspace(rank, dims);
        auto datatype(H5::PredType::NATIVE_DOUBLE);
        H5::DataSet* dataset = new H5::DataSet( file->createDataSet(datasetname, datatype, dataspace, proplist) );
        dataset->write(part_info.data(), datatype);

        delete dataset;
        delete file;
*/

        return 0;
}
