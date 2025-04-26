#include "rk_routines.hpp"

/**
    evaluate stability function for given Runge-Kutta method
*/
void stability_function(const int method,       ///< Runge-Kutta method, see constants.hpp
                        arma::Col<double> *z,   ///< z = dt*eta, with time step size dt and spatial eigenvalue eta
                        arma::Col<double> *rz   ///< on return, the value of the stability function
                        ){
    // get Butcher tableau for requested method
    arma::mat           A;
    arma::Col<double>   b;
    arma::Col<double>   c;
    get_butcher_tableau(method, A, b, c);
    for(int evalIdx = 0; evalIdx < (*z).n_elem; evalIdx++){
        (*rz)(evalIdx) = arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A + (*z)(evalIdx) * arma::ones(b.n_elem, 1) * b.t())
                            / arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A);
    }
}

/**
    evaluate stability function for given Runge-Kutta method (complex version)
*/
void stability_function(const int method,               ///< Runge-Kutta method, see constants.hpp
                        arma::Col<arma::cx_double> *z,  ///< z = dt*eta, with time step size dt and spatial eigenvalue eta
                        arma::Col<arma::cx_double> *rz  ///< on return, the value of the stability function
                        ){
    // get Butcher tableau for requested method
    arma::mat           A;
    arma::Col<double>   b;
    arma::Col<double>   c;
    get_butcher_tableau(method, A, b, c);
    for(int evalIdx = 0; evalIdx < (*z).n_elem; evalIdx++){
        (*rz)(evalIdx) = arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A + (*z)(evalIdx) * arma::ones(b.n_elem, 1) * b.t())
                            / arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A);
    }
}

/**
    evaluate stability function for Runge-Kutta method with provided Butcher tableau
*/
void stability_function(arma::mat A,            ///< Butcher coefficients, matrix A
                        arma::Col<double> b,    ///< Butcher coefficients, vector b
                        arma::Col<double> c,    ///< Butcher coefficients, vector c
                        arma::Col<double> *z,   ///< z = dt*eta, with time step size dt and spatial eigenvalue eta
                        arma::Col<double> *rz   ///< on return, the value of the stability function
                        ){
    for(int evalIdx = 0; evalIdx < (*z).n_elem; evalIdx++){
        (*rz)(evalIdx) = arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A + (*z)(evalIdx) * arma::ones(b.n_elem, 1) * b.t())
                            / arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A);
    }
}

/**
    evaluate stability function for Runge-Kutta method with provided Butcher tableau (complex version)
*/
void stability_function(arma::mat A,                    ///< Butcher coefficients, matrix A
                        arma::Col<double> b,            ///< Butcher coefficients, vector b
                        arma::Col<double> c,            ///< Butcher coefficients, vector c
                        arma::Col<arma::cx_double> *z,  ///< z = dt*eta, with time step size dt and complex spatial eigenvalue eta
                        arma::Col<arma::cx_double> *rz  ///< on return, the value of the stability function
                        ){
    for(int evalIdx = 0; evalIdx < (*z).n_elem; evalIdx++){
        (*rz)(evalIdx) = arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A + (*z)(evalIdx) * arma::ones(b.n_elem, 1) * b.t())
                            / arma::det(arma::eye(b.n_elem,b.n_elem) - (*z)(evalIdx) * A);
    }
}

/** 
    sets the coefficients of a given Runge-Kutta method corresponding to Butcher tableau of form
        <table>
            <tr><td>c<td>A
            <tr><td> <td>b^T
        </table>
*/
void get_butcher_tableau(const int method,      ///< Runge-Kutta method, see constants.hpp
                         arma::mat &A,          ///< matrix A of Butcher tableau
                         arma::Col<double> &b,  ///< vector b of Butcher tableau
                         arma::Col<double> &c   ///< vector c of Butcher tableau
                         ){
    switch(method){
        case rkconst::A_stable_SDIRK2:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{0.25,   0.0},
                       {0.5,    0.25}};
            b       = {0.5,     0.5};
            c       = {0.25,    0.75};
            break;
        }
        case rkconst::A_stable_SDIRK3:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            double gamma = (3.0 + sqrt(3.0)) / 6.0;
            A       = {{gamma,          0.0},
                       {1.0-2.0*gamma,  gamma}};
            b       = {0.5,             0.5};
            c       = {gamma,           1.0-gamma};
            break;
        }
        case rkconst::A_stable_SDIRK4:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            double  q = 1.0 / sqrt(3.0) * cos(constants::pi / 18.0) + 0.5;
            double  r = 1.0 / (6.0 *(2.0 * q - 1.0) * (2.0 * q - 1.0));
            A       = {{q,      0.0,        0.0},
                       {0.5-q,  q,          0.0},
                       {2.0*q,  1.0-4.0*q,  q}};
            b       = {r,       1.0-2.0*r,  r};
            c       = {q,       0.5,        1.0-q};
            break;
        }
        case rkconst::L_stable_SDIRK1:
        {
            A.set_size(1,1);
            b.set_size(1);
            c.set_size(1);
            A(0,0)  = 1.0;
            b(0)    = 1.0;
            c(0)    = 1.0;
            break;
        }
        case rkconst::L_stable_SDIRK2:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            double  gamma = 1.0 / sqrt(2.0);
            A       = {{1.0-gamma,      0.0},
                       {2.0*gamma-1.0,  1.0-gamma}};
            b       = {0.5,             0.5};
            c       = {1.0-gamma,       gamma};
            break;
        }
        case rkconst::L_stable_SDIRK3:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            double  q   = 0.435866521508458999416019;
            double  r   = 1.20849664917601007033648;
            double  s   = 0.717933260754229499708010;
            A       = {{q,      0.0,        0.0},
                       {s-q,    q,          0.0},
                       {r,      1.0-q-r,    q}};
            b       = {r,       1.0-q-r,    q};
            c       = {q,       s,          1.0};
            break;
        }
        case rkconst::L_stable_SDIRK4:
        {
            A.set_size(5,5);
            b.set_size(5);
            c.set_size(5);
            A       = {{0.25,           0.0,            0.0,        0.0,        0.0},
                       {0.5,            0.25,           0.0,        0.0,        0.0},
                       {17.0/50.0,      -1.0/25.0,      0.25,       0.0,        0.0},
                       {371.0/1360.0,   -137.0/2720.0,  15.0/544.0, 0.25,       0.0},
                       {25.0/24.0,      -49.0/48.0,     125.0/16.0, -85.0/12.0, 0.25}};
            b       = {25.0/24.0,       -49.0/48.0,     125.0/16.0, -85.0/12.0, 0.25};
            c       = {0.25,            0.75,           11.0/20.0,  0.5,        1.0};
            break;
        }
        case rkconst::L_stable_SDIRK5:
        {
            double dval = 4024571134387.0 / 14474071345096.0;
            A.set_size(5,5);
            b.set_size(5);
            c.set_size(5);
            A       = {{dval,                               0.0,                                0.0,                                0.0,                                0.0},
                       {9365021263232.0/12572342979331.0,   dval,                               0.0,                                0.0,                                0.0},
                       {2144716224527.0/9320917548702.0,    -397905335951.0/4008788611757.0,    dval,                               0.0,                                0.0},
                       {-291541413000.0/6267936762551.0,    226761949132.0/4473940808273.0,     -1282248297070.0/9697416712681.0,   dval,                               0.0},
                       {-2481679516057.0/4626464057815.0,   -197112422687.0/6604378783090.0,    3952887910906.0/9713059315593.0,    4906835613583.0/8134926921134.0,    4024571134387.0/14474071345096.0}};
            b       = {-2522702558582.0/12162329469185.0,   1018267903655.0/12907234417901,     4542392826351.0/13702606430957.0,   5001116467727.0/12224457745473.0,   1509636094297.0/3891594770934.0};
            c       = {dval,                                5555633399575.0/5431021154178.0,    5255299487392.0/12852514622453.0,   3.0/20.0,                           10449500210709.0/14474071345096.0};
            break;
        }
        case rkconst::L_stable_RadauIIA_order1:
        {
            A.set_size(1,1);
            b.set_size(1);
            c.set_size(1);
            A(0,0)  = 1.0;
            b(0)    = 1.0;
            c(0)    = 1.0;
            break;
        }
        case rkconst::L_stable_RadauIIA_order3:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{5.0/12.0,   -1.0/12.0},
                       {0.75,       0.25}};
            b       = {0.75,        0.25};
            c       = {1.0/3.0,     1.0};
            break;
        }
        case rkconst::L_stable_RadauIIA_order5:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            A       = {{(88.0-7.0*sqrt(6.0))/360.0,     (296.0-169.0*sqrt(6.0))/1800.0,     (-2.0+3.0*sqrt(6.0))/225.0},
                       {(296.0+169.0*sqrt(6.0))/1800.0, (88.0+7.0*sqrt(6.0))/360.0,         (-2.0-3.0*sqrt(6.0))/225.0},
                       {(16.0-sqrt(6.0))/36.0,          (16.0+sqrt(6.0))/36.0,              1.0/9.0}};
            b       = {(16.0-sqrt(6.0))/36.0,           (16.0+sqrt(6.0))/36.0,              1.0/9.0};
            c       = {(4.0-sqrt(6.0))/10.0,            (4.0+sqrt(6.0))/10.0,               1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIA_order2:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{0.0,    0.0},
                       {0.5,    0.5}};
            b       = {0.5,     0.5};
            c       = {0.0,     1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIA_order4:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            A       = {{0.0,        0.0,        0.0},
                       {5.0/24.0,   1.0/3.0,    -1.0/24.0},
                       {1.0/6.0,    2.0/3.0,    1.0/6.0}};
            b       = {1.0/6.0,     2.0/3.0,    1.0/6.0};
            c       = {0.0,     0.5,    1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIA_order6:
        {
            A.set_size(4,4);
            b.set_size(4);
            c.set_size(4);
            A       = {{0.0,                    0.0,                            0.0,                            0.0},
                       {(11.0+sqrt(5.0))/120.0, (25.0-sqrt(5.0))/120.0,         (25.0-13.0*sqrt(5.0))/120.0,    (-1.0+sqrt(5.0))/120.0},
                       {(11.0-sqrt(5.0))/120.0, (25.0+13.0*sqrt(5.0))/120.0,    (25.0+sqrt(5.0))/120.0,         (-1.0-sqrt(5.0))/120.0},
                       {1.0/12.0,               5.0/12.0,                       5.0/12.0,                       1.0/12.0}};
            b       = {1.0/12.0,                5.0/12.0,                       5.0/12.0,                       1.0/12.0};
            c       = {0.0,                     (5.0-sqrt(5.0))/10.0,           (5.0-sqrt(5.0))/10.0,           1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIB_order2:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{0.5,    0.0},
                       {0.5,    0.0}};
            b       = {0.5,     0.5};
            c       = {0.0, 1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIB_order4:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            A       = {{1.0/6.0,    1.0/6.0,    0.0},
                       {1.0/6.0,    1.0/3.0,    0.0},
                       {1.0/6.0,    5.0/6.0,    0.0}};
            b       = {1.0/6.0,     2.0/3.0,    1.0/6.0};
            c       = {0.0,     0.5,        1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIB_order6:
        {
            A.set_size(4,4);
            b.set_size(4);
            c.set_size(4);
            A       = {{1.0/12.0,   (-1.0-sqrt(5.0))/24.0,          (-1.0+sqrt(5.0))/24.0,          0.0},
                       {1.0/12.0,   (25.0+sqrt(5.0))/120.0,         (25.0-13.0*sqrt(5.0))/120.0,    0.0},
                       {1.0/12.0,   (25.0+13.0*sqrt(5.0))/120.0,    (25.0-sqrt(5.0))/120.0,         0.0},
                       {1.0/12.0,   (11.0-sqrt(5.0))/24.0,          (11.0+sqrt(5.0))/24.0,          0.0}};
            b       = {1.0/12.0,    5.0/12.0,                       5.0/12.0,                       1.0/12.0};
            c       = {0.0,         (5.0-sqrt(5.0))/10.0,           (5.0+sqrt(5.0))/10.0,           1.0};
            break;
        }
        case rkconst::A_stable_LobattoIIIB_order8:
        {
            A.set_size(5,5);
            b.set_size(5);
            c.set_size(5);
            A       = {{1.0/20.0,   (-7.0-sqrt(21.0))/120.0,        1.0/15.0,                       (-7.0+sqrt(21))/120.0,          0.0},
                       {1.0/20.0,   (343.0+9.0*sqrt(21.0))/2520.0,  (56.0-15.0*sqrt(21.0))/315.0,   (343.0-69.0*sqrt(21.0))/2520.0, 0.0},
                       {1.0/20.0,   (49.0+12.0*sqrt(21.0))/360.0,   8.0/45.0,                       (49.0-12.0*sqrt(21.0))/360.0,   0.0},
                       {1.0/20.0,   (343.0+69.0*sqrt(21.0))/2520.0, (56.0+15.0*sqrt(21.0))/315.0,   (343.0-9.0*sqrt(21.0))/2520.0,  0.0},
                       {1.0/20.0,   (119.0-3.0*sqrt(21.0))/360.0,   13.0/45.0,                      (119.0+3.0*sqrt(21.0))/360.0,   0.0}};
            b       = {1.0/20.0,    49.0/180.0,                     16.0/45.0,                      49.0/180.0,                     1.0/20.0};
            c       = {0.0,         (7.0-sqrt(21))/14.0,            0.5,                            (7.0+sqrt(21.0))/14.0,          1.0};
            break;
        }
        case rkconst::L_stable_LobattoIIIC_order2:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{0.5,    -0.5},
                       {0.5,    0.5}};
            b       = {0.5,     0.5};
            c       = {0.0,     1.0};
            break;
        }
        case rkconst::L_stable_LobattoIIIC_order4:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            A       = {{1.0/6.0,    -1.0/3.0,   1.0/6.0},
                       {1.0/6.0,    5.0/12.0,   -1.0/12.0},
                       {1.0/6.0,    2.0/3.0,    1.0/6.0}};
            b       = {1.0/6.0,     2.0/3.0,    1.0/6.0};
            c       = {0.0,         0.5,        1.0};
            break;
        }
        case rkconst::L_stable_LobattoIIIC_order6:
        {
            A.set_size(4,4);
            b.set_size(4);
            c.set_size(4);
            A       = {{1.0/12.0,   -sqrt(5.0)/12.0,            sqrt(5.0)/12.0,             -1.0/12.0},
                       {1.0/12.0,   0.25,                       (10.0-7.0*sqrt(5.0))/60.0,  sqrt(5.0)/60.0},
                       {1.0/12.0,   (10.0+7.0*sqrt(5.0))/60.0,  0.25,                       -sqrt(5.0)/60.0},
                       {1.0/12.0,   5.0/12.0,                   5.0/12.0,                   1.0/12.0}};
            b       = {1.0/12.0,    5.0/12.0,                   5.0/12.0,                   1.0/12.0};
            c       = {0.0,         (5.0-sqrt(5.0))/10.0,       (5.0+sqrt(5.0))/10.0,       1.0};
            break;
        }
        case rkconst::L_stable_LobattoIIIC_order8:
        {
            A.set_size(5,5);
            b.set_size(5);
            c.set_size(5);
            A       = {{1.0/20.0,   -7.0/60.0,                          2.0/15.0,                       -7.0/60.0,                      1.0/20.0},
                       {1.0/20.0,   29.0/180.0,                         (47.0-15.0*sqrt(21.0))/315.0,   (203.0-30.0*sqrt(21.0))/1260.0, -3.0/140.0},
                       {1.0/20.0,   (329.0+105.0*sqrt(21.0))/2880.0,    73.0/360.0,                     (329.0-105.0*sqrt(21))/2880.0,  3.0/160.0},
                       {1.0/20.0,   (203.0+30.0*sqrt(21))/1260.0,       (47.0+15.0*sqrt(21))/315.0,     29.0/180.0,                     -3.0/140.0},
                       {1.0/20.0,   49.0/180.0,                         16.0/45.0,                      49.0/180.0,                     1.0/20.0}};
            b       = {1.0/20.0,    49.0/180.0,                         16.0/45.0,                      49.0/180.0,                     1.0/20.0};
            c       = {0.0,         (7.0-sqrt(21.0))/14.0,              0.5,                            (7.0+sqrt(21.0))/14.0,          1.0};
            break;
        }
        case rkconst::LobattoIIICast_order2:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{0.0,    0.0},
                       {1.0,    0.0}};
            b       = {0.5,     0.5};
            c       = {0.0,     1.0};
            break;
        }
        case rkconst::LobattoIIICast_order4:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            A       = {{0.0,    0.0,        0.0},
                       {0.25,   0.25,       0.0},
                       {0.0,    1.0,        0.0}};
            b       = {1.0/6.0, 2.0/3.0,    1.0/6.0};
            c       = {0.0,     0.5,        1.0};
            break;
        }
        case rkconst::LobattoIIICast_order6:
        {
            A.set_size(4,4);
            b.set_size(4);
            c.set_size(4);
            A       = {{0.0,                    0.0,                        0.0,                        0.0},
                       {(5.0+sqrt(5.0))/60.0,   1.0/6.0,                    (15.0-7.0*sqrt(5.0))/60.0,  0.0},
                       {(5.0-sqrt(5.0))/60.0,   (15.0+7.0*sqrt(5.0))/60.0,  1.0/6.0,                    0.0},
                       {1.0/6.0,                (5.0-sqrt(5.0))/12.0,       (5.0+sqrt(5.0))/12.0,       0.0}};
            b       = {1.0/12.0,                5.0/12.0,                   5.0/12.0,                   1.0/12.0};
            c       = {0.0,                     (5.0-sqrt(5.0))/10.0,       (5.0+sqrt(5.0))/10.0,       1.0};
            break;
        }
        case rkconst::LobattoIIICast_order8:
        {
            A.set_size(5,5);
            b.set_size(5);
            c.set_size(5);
            A       = {{0.0,            0.0,                            0.0,                        0.0,                            0.0},
                       {1.0/14.0,       1.0/9.0,                        (13.0-3.0*sqrt(21.0))/63.0, (14.0-3.0*sqrt(21.0))/126.0,    0.0},
                       {1.0/32.0,       (91.0+21.0*sqrt(21.0))/576.0,   11.0/72.0,                  (91.0-21.0*sqrt(21.0))/576.0,   0.0},
                       {1.0/14.0,       (14.0+3.0*sqrt(21.0))/126.0,    (13.0+3.0*sqrt(21.0))/63.0, 1.0/9.0,                        0.0},
                       {0.0,            7.0/18.0,                       2.0/9.0,                    7.0/18.0,                       0.0}};
            b       = {1.0/20.0,        49.0/180.0,                     16.0/45.0,                  49.0/180.0,                     1.0/20.0};
            c       = {0.0,             (7.0-sqrt(21.0))/14.0,          0.5,                        (7.0+sqrt(21.0))/14.0,          1.0};
            break;
        }
        case rkconst::A_stable_Gauss_order2:
        {
            A.set_size(1,1);
            b.set_size(1);
            c.set_size(1);
            A(0,0)  = 0.5;
            b(0)    = 1.0;
            c(0)    = 0.5;
            break;
        }
        case rkconst::A_stable_Gauss_order4:
        {
            A.set_size(2,2);
            b.set_size(2);
            c.set_size(2);
            A       = {{0.25,                       (6.0-4.0*sqrt(3.0))/24.0},
                       {(6.0+4.0*sqrt(3.0))/24.0,   0.25}};
            b       = {0.5,                         0.5};
            c       = {(3.0-sqrt(3.0))/6.0,         (3.0+sqrt(3.0))/6.0};
            break;
        }
        case rkconst::A_stable_Gauss_order6:
        {
            A.set_size(3,3);
            b.set_size(3);
            c.set_size(3);
            A       = {{5.0/36.0,                   2.0/9.0-1.0/sqrt(15.0),     5.0/36.0-sqrt(15.0)/30.0},
                       {5.0/36.0+sqrt(15.0)/24.0,   2.0/9.0,                    5.0/36.0-sqrt(15.0)/24.0},
                       {5.0/36.0+sqrt(15.0)/30.0,   2.0/9.0+1.0/sqrt(15.0),     5.0/36.0}};
            b       = {5.0/18.0,                    4.0/9.0,                    5.0/18.0};
            c       = {(5.0-sqrt(15.0))/10.0,       0.5,                        (5.0+sqrt(15.0))/10.0};
            break;
        }
        default:
        {
            std::cerr << "Runge-Kutta method " << method << " not implemented." << std::endl;
            throw;
        }
    }
}