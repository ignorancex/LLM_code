#include <iostream>
#include <cmath>

using namespace std;




double* rot(double theta, double phi, double* p)
{
    double p0=p[0], p1=p[1],p2=p[2],p3=p[3];
    if ((theta*theta + phi*phi) > 1e-10)
    {
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);
        p1 = cos_theta*cos_phi*p[1] - sin_phi*p[2] + sin_theta*cos_phi*p[3];
        p2 = cos_theta*sin_phi*p[1] + cos_phi*p[2] + sin_theta*sin_phi*p[3];
        p3 = -sin_theta*p[1] + cos_theta*p[3];
    }

    double* p_new = new double[4];
    p_new[0] = p0;
    p_new[1] = p1;
    p_new[2] = p2;
    p_new[3] = p3;
    return p_new;
}

double* rot2(double cos_theta, double sin_theta, double cos_phi, double sin_phi, double *p)
{
    double p0=p[0], p1=p[1], p2=p[2], p3=p[3];
    p1 = cos_theta*cos_phi*p[1] - sin_phi*p[2] + sin_theta*cos_phi*p[3];
    p2 = cos_theta*sin_phi*p[1] + cos_phi*p[2] + sin_theta*sin_phi*p[3];
    p3 = -sin_theta*p[1] + cos_theta*p[3];
    double* p_new = new double[4];
    p_new[0] = p0;
    p_new[1] = p1;
    p_new[2] = p2;
    p_new[3] = p3;
    return p_new;
}   


double* bos(double* beta, double *p)
{
    double p0=p[0], p1=p[1], p2=p[2], p3=p[3];

    double beta2 = beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2];
    if (beta2 > 1e-6)
    {
        double gamma =  1.0/sqrt(1. - beta2);
        double betax = beta[0];
        double betay = beta[1];
        double betaz = beta[2];
        double beta_p = betax*p[1] + betay*p[2] + betaz*p[3];
        double gamma_beta = gamma * (gamma * beta_p/(1. + gamma) + p[0]);

        p0 = gamma*(p0 + beta_p);
        p1 = p1 + gamma_beta*betax;
        p2 = p2 + gamma_beta*betay;
        p3 = p3 + gamma_beta*betaz;
        //cout << "bos: " << p0 << "   " << p1 << "   " << p2 << "   " << p3 << endl;
    }
 
    double* p_new = new double[4];
    p_new[0] = p0;
    p_new[1] = p1;
    p_new[2] = p2;
    p_new[3] = p3;
    return p_new;

}



double* transform_to_CoM(double* p1, double* p2, double* vec)
// boost and rotate the frame to center of mass frame, such that p1 in CoM align in +z direction
// p1, p2 are four vectors in cell frame, vec is the four vector in cell frame
// return vec in CoM frame
{
    static double minus_beta[3];
    double E_total = p1[0] + p2[0];
    minus_beta[0] =-1.* (p1[1] + p2[1])/E_total;
    minus_beta[1] =-1.* (p1[2] + p2[2])/E_total;
    minus_beta[2] =-1.* (p1[3] + p2[3])/E_total;


    //cout << "p1: " << p1[0] << "  " << p1[1] << "   " << p1[2] << "   " << p1[3] << endl;
    //cout << "p2: " << p2[0] << "  " << p2[1] << "   " << p2[2] << "   " << p2[3] << endl;


    double* p1_bos;
    p1_bos= bos(minus_beta, p1);
    double* p2_bos;
    p2_bos= bos(minus_beta, p2);

    //cout << "p1_bos: " << p1_bos[0] << "   " << p1_bos[1] << "   " << p1_bos[2] << "    " << p1_bos[3] << endl;
    //cout << "p2_bos: " << p2_bos[0] << "   " << p2_bos[2] << "    " << p2_bos[2] << "    " << p2_bos[3] << endl;
    double* vec_bos = bos(minus_beta, vec);

    double p1_mole = std::sqrt(p1_bos[1]*p1_bos[1] + p1_bos[2]*p1_bos[2] + p1_bos[3]*p1_bos[3]);

    double cos_theta = p1_bos[3]/p1_mole;
    double sin_theta = std::sqrt(1. - cos_theta*cos_theta);
    double cos_phi = p1_bos[1]/p1_mole/sin_theta;
    double sin_phi = p1_bos[2]/p1_mole/sin_theta;

    //cout <<"p1_mole: " << p1_mole << "   " <<  cos_theta << "  " << sin_theta << "  " << cos_phi << "   " << sin_phi << endl;
    double* vec_bos_rot = rot2(cos_theta, sin_theta, cos_phi, sin_phi, vec_bos);
    double* p1_bos_rot = rot2(cos_theta, sin_theta, cos_phi, sin_phi, p1_bos);
    double* p2_bos_rot = rot2(cos_theta, sin_theta, cos_phi, sin_phi, p2_bos);
   
    //cout << "p1_bos_rot: " << p1_bos_rot[0] << "   " << p1_bos_rot[1] << "   " << p1_bos_rot[2] << "    " << p1_bos_rot[3] << endl;
    //cout << "p2_bos_rot: " << p2_bos_rot[0] << "    " << p2_bos_rot[1] << "    " << p2_bos_rot[2] << "   "  << p2_bos_rot[3] << endl;
    //cout << "vec_bos_rot: " << vec_bos_rot[0] << "   " << vec_bos_rot[1] << "   " << vec_bos_rot[2] << "   " << vec_bos_rot[3] << endl;
    //cout << endl;
    //cout << p1_bos_rot[0] << "  " << p1_bos_rot[1] << "  " << p1_bos_rot[2] << "  " << p1_bos_rot[3] << endl;

    delete [] p1_bos;
    delete [] p2_bos;
    delete [] p1_bos_rot;
    delete [] p2_bos_rot;

    return vec_bos_rot;
}



double* transform_from_CoM(double* p1, double* p2, double *vec)
// giving p1, p2 in cell frame, transform the vector vec (in CoM frame) back to cell frame
{
    static double beta[3];
    static double minus_beta[3];
    double E_total = p1[0] + p2[0];
    beta[0] = (p1[1] + p2[1])/E_total;
    beta[1] = (p1[2] + p2[2])/E_total;
    beta[2] = (p1[3] + p2[3])/E_total;


    minus_beta[0] = -(p1[1] + p2[1])/E_total;
    minus_beta[1] = -(p1[2] + p2[2])/E_total;
    minus_beta[2] = -(p1[3] + p2[3])/E_total;


    double beta2 = minus_beta[0]*minus_beta[0] + minus_beta[1]*minus_beta[1] + minus_beta[2]*minus_beta[2];
    double gamma =1./ std::sqrt(1-beta2);
    cout << "gamma: " << gamma << endl;

    double* p1_bos = bos(minus_beta, p1);
    double* p2_bos = bos(minus_beta, p2);

    double p1_mole = std::sqrt(p1_bos[1]*p1_bos[1] + p1_bos[2]*p1_bos[2] + p1_bos[3]*p1_bos[3]);
    double cos_theta = p1_bos[3]/p1_mole;
    double sin_theta = std::sqrt(1. - cos_theta*cos_theta);
    double cos_phi = p1_bos[1]/p1_mole/sin_theta;
    double sin_phi = p1_bos[2]/p1_mole/sin_theta;

    double* vec_rot = rot2(cos_theta, sin_theta, cos_phi, sin_phi, vec);
    
    double* vec_rot_bos = bos(beta, vec_rot);

    delete [] p1_bos;
    delete [] p2_bos;
    delete [] vec_rot;
    return vec_rot_bos;

}



double** transform_from_CoM_array(double* p1, double* p2)
// giving any p1 and p2 in cell frame, return a matrix that will transform any vec into the vec in CoM frame of (p1, p2)
{
    double E_total = p1[0] + p2[0];

    static double minus_beta[3];
    
    double betax = (p1[1] + p2[1])/E_total;
    double betay  = (p1[2] + p2[2])/E_total;
    double betaz  = (p1[3] + p2[3])/E_total;
   
    minus_beta[0] = -betax;
    minus_beta[1] = -betay;
    minus_beta[2] = -betaz;

    double* p1_bos = bos(minus_beta, p1);
    double p1_mole = std::sqrt(p1_bos[1]*p1_bos[1] + p1_bos[2]*p1_bos[2] + p1_bos[3]*p1_bos[3]);
    double cos_theta = p1_bos[3]/p1_mole;
    double sin_theta = std::sqrt(1. - cos_theta*cos_theta);
    double cos_phi = p1_bos[1]/p1_mole/sin_theta;
    double sin_phi = p1_bos[2]/p1_mole/sin_theta;


    delete [] p1_bos;

    double** rotation = new double*[4];
    int i;
    for (i=0; i<4; ++i)
        rotation[i] = new double [4];

    rotation[0][0] = 1;
    rotation[0][1] = 0;
    rotation[0][2] = 0;
    rotation[0][3] = 0;
    rotation[1][0] = 0;
    rotation[1][1] = cos_theta * cos_phi;
    rotation[1][2] = - sin_phi;
    rotation[1][3] = sin_theta * cos_phi;
    rotation[2][0] = 0;
    rotation[2][1] = cos_theta*sin_phi;
    rotation[2][2] = cos_phi;
    rotation[2][3] = sin_theta*sin_phi;
    rotation[3][0] = 0;
    rotation[3][1] = -sin_theta;
    rotation[3][2] = 0;
    rotation[3][3] = cos_theta;

    double beta2 = betax*betax + betay*betay + betaz*betaz;                                    

    double **boost_matrix = new double* [4];
    for (i=0; i< 4; ++i)
        boost_matrix[i] = new double [4];

    int mu, nu;

    if (beta2 > 1e-10)
    {
        double gamma = 1.0/std::sqrt(1. - beta2);
        double gamma2 = gamma*gamma/(1. + gamma);
        boost_matrix[0][0] = gamma;
        boost_matrix[0][1] = gamma*betax;
        boost_matrix[0][2] = gamma*betay;
        boost_matrix[0][3] = gamma*betaz;
        boost_matrix[1][0] = gamma*betax;
        boost_matrix[1][1] = 1. + gamma2*betax*betax;
        boost_matrix[1][2] = gamma2*betax*betay;
        boost_matrix[1][3] = gamma2*betax*betaz;
        boost_matrix[2][0] = gamma*betay;
        boost_matrix[2][1] = gamma2*betax*betay;
        boost_matrix[2][2] = 1. + gamma2*betay*betay;
        boost_matrix[2][3] = gamma2 * betay*betaz;
        boost_matrix[3][0] = gamma*betaz;
        boost_matrix[3][1] = gamma2 * betax*betaz;
        boost_matrix[3][2] = gamma2 * betay*betaz;
        boost_matrix[3][3] = 1. + gamma2*betaz*betaz;
        //cout << gamma << endl;
    }
    else
    {
        for (mu=0; mu<4; ++mu)
        {
            for (nu=0; nu<4; nu++)
            {
                if (mu == nu)
                        boost_matrix[mu][nu] = 1;
                else
                        boost_matrix[mu][nu] = 0;
            }


        }
    }

    double **From_CoM = new double* [4];
    for (i=0; i<4; ++i)
        From_CoM[i] = new double [4];

    int k;

    for (mu=0; mu < 4; ++mu)
    {
        for (nu = 0; nu < 4; ++nu)
        {
            double dum = 0;
            for (k = 0; k < 4; ++k)

               dum += boost_matrix[mu][k]*rotation[k][nu];

            From_CoM[mu][nu] = dum;
        }
    }

    for (int i=0; i<4; ++i)
    {
        delete [] rotation[i];
        delete [] boost_matrix[i];
    }

    delete [] rotation;
    delete [] boost_matrix;
    return From_CoM;
}
