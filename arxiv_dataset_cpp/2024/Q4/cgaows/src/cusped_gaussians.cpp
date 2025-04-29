#include <iostream>
#include <cmath>
#include "cusped_gaussians.hpp"
#include <vector>

// non-orthogonalized s-type GTO
double CuspedGaussians::GTO_s(double r, int ao_ind) {

  const double pi = 3.14159265358979323846;
 
  // evaluate s-type GTO
  double orb_total = 0.0;

  // loop over number of primitives  
  for (int k = 0; k < m_ng; k++) {
    
    // exponents
    double a = m_bf_exp[k * m_no + ao_ind];
    
    // coefficients 
    double d = m_bf_coeff[k * m_no + ao_ind];

    // add to running sum
    orb_total += d * pow(((2.0 * a) / pi), 0.75) * std::exp(-a * r * r);
  }

  return orb_total;
}

// orthogonalized s-type GTO
double CuspedGaussians::GTO_s(double r, int ao_ind, int core_ao_ind, double proj) {

  // orthogonalize against core
  double orb_total = 0.0;
  orb_total = GTO_s(r, ao_ind) - proj * GTO_s(r, core_ao_ind);
  return orb_total;
}

void CuspedGaussians::d1_GTO_s(const double r, int ao_ind, const double (&delta)[3], double (&d1_vec)[3]) {

  const double pi = 3.14159265358979323846;
  
  // sum over primitives
  for (int k = 0; k < m_ng; k++) {

    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind];
    d1_vec[0] += -delta[0] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
    d1_vec[1] += -delta[1] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
    d1_vec[2] += -delta[2] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);

  } 

}

// for orthogonalized s-type GTO
void CuspedGaussians::d1_GTO_s(const double r, int ao_ind, int core_ao_ind, double proj, const double (&delta)[3], double (&d1_vec)[3]) {
	
  double d1_current_orb[3] = {0.0, 0.0, 0.0}; 
  double d1_core_orb[3] = {0.0, 0.0, 0.0}; 
  d1_GTO_s(r, ao_ind, delta, d1_current_orb);
  d1_GTO_s(r, core_ao_ind, delta, d1_core_orb);

  // orthogonalize against core
  for (int j = 0; j < 3; j++)
    d1_vec[j] = d1_current_orb[j] - proj * d1_core_orb[j]; 

}

void CuspedGaussians::d2_GTO_s(const double r, int ao_ind, const double (&delta)[3], double (&d2_vec)[3]) {
  
  const double pi = 3.14159265358979323846;

  // sum over primitives
  for (int k = 0; k < m_ng; k++) {
    
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind];
    double Ns = pow(((2.0 * a) / pi), 0.75);
    d2_vec[0] += -2.0 * a * d * Ns * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[0] * delta[0]);
    d2_vec[1] += -2.0 * a * d * Ns * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[1] * delta[1]);
    d2_vec[2] += -2.0 * a * d * Ns * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[2] * delta[2]);
	
   }

}

// for orthogonalized s-type orbital
void CuspedGaussians::d2_GTO_s(const double r, int ao_ind, int core_ao_ind, double proj, const double (&delta)[3], double (&d2_vec)[3]) {
	
  double d2_current_orb[3] = {0.0, 0.0, 0.0};
  double d2_core_orb[3] = {0.0, 0.0, 0.0}; 
  d2_GTO_s(r, ao_ind, delta, d2_current_orb);
  d2_GTO_s(r, core_ao_ind, delta, d2_core_orb);

  // orthogonalized against core
  for (int j = 0; j < 3; j++)
    d2_vec[j] = d2_current_orb[j] - proj * d2_core_orb[j]; 

}

double CuspedGaussians::GTO_p(double r, double (&dist_xi)[3], int ao_ind, int ao_type) {

  const double pi = 3.14159265358979323846;
  
  double orb_total = 0.0;
  
  // sum over primitives
  for (int k = 0; k < m_ng; k++) {
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind];
    double Ns = d * pow((2.0 * a / pi), 0.75);
    orb_total += Ns * 2.0 * std::sqrt(a) * dist_xi[ao_type-2] * std::exp(-a * r * r);
  } 
 
  return orb_total;
}

void CuspedGaussians::d1_GTO_p(const double r, int ao_ind, const double (&delta)[3], int ao_type, double (&d1_vec)[3]) {

  const double pi = 3.14159265358979323846;
  
  // sum over primitives
  for (int k = 0; k < m_ng; k++) {
			
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind];
    double Ns = pow(((2.0 * a) / pi), 0.75);
    double Np = Ns * 2.0 * std::sqrt(a);

    // if px
    if (ao_type == 2) {
      d1_vec[0] += d * Np * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[0] * delta[0]);
      d1_vec[1] += -2.0 * a * d * Np * delta[0] * delta[1] * std::exp(-a * r * r);
      d1_vec[2] += -2.0 * a * d * Np * delta[0] * delta[2] * std::exp(-a * r * r);
    } // px
    
    // if py
    else if (ao_type == 3) {
      d1_vec[0] += -2.0 * a * d * Np * delta[1] * delta[0] * std::exp(-a * r * r);
      d1_vec[1] += d * Np * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[1] * delta[1]);
      d1_vec[2] += -2.0 * a * d * Np * delta[1] * delta[2] * std::exp(-a * r * r);
    } // py

    // if pz
    else if (ao_type == 4) {
      d1_vec[0] += -2.0 * a * d * Np * delta[2] * delta[0] * std::exp(-a * r * r);
      d1_vec[1] += -2.0 * a * d * Np * delta[2] * delta[1] * std::exp(-a * r * r);
      d1_vec[2] += d * Np * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[2] * delta[2]);
    } // pz
  } // end sum over primitives
	
}

void CuspedGaussians::d2_GTO_p(const double r, int ao_ind, const double (&delta)[3], int ao_type, double (&d2_vec)[3]) {

  const double pi = 3.14159265358979323846;
  
  // sum over primitives
  for (int k = 0; k < m_ng; k++) {
	
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind];
    double Ns = pow(((2.0 * a) / pi), 0.75);
    double Np = Ns * 2.0 * std::sqrt(a);

    // if px
    if (ao_type == 2) {
      d2_vec[0] += d * Np * std::exp(-a * r * r) * (-6.0 * a * delta[0] + 4.0 * a * a * delta[0] * delta[0] * delta[0]);
      d2_vec[1] += 2.0 * a * d * Np * delta[0] * std::exp(-a * r * r) * (2.0 * a * delta[1] * delta[1] - 1.0);
      d2_vec[2] += 2.0 * a * d * Np * delta[0] * std::exp(-a * r * r) * (2.0 * a * delta[2] * delta[2] - 1.0);
    } // px

    // if py
    else if (ao_type == 3) {
      d2_vec[0] += 2.0 * a * d * Np * delta[1] * std::exp(-a * r * r) * (2.0 * a * delta[0] * delta[0] - 1.0);
      d2_vec[1] += d * Np * std::exp(-a * r * r) * (-6.0 * a * delta[1] + 4.0 * a * a * delta[1] * delta[1] * delta[1]); 
      d2_vec[2] += 2.0 * a * d * Np * delta[1] * std::exp(-a * r * r) * (2.0 * a * delta[2] * delta[2] - 1.0);
    } // py

    // if pz
    else if (ao_type == 4) {
      d2_vec[0] += 2.0 * a * d * Np * delta[2] * std::exp(-a * r * r) * (2.0 * a * delta[0] * delta[0] - 1.0);
      d2_vec[1] += 2.0 * a * d * Np * delta[2] * std::exp(-a * r * r) * (2.0 * a * delta[1] * delta[1] - 1.0);
      d2_vec[2] += d * Np * std::exp(-a * r * r) * (-6.0 * a * delta[2] + 4.0 * a * a * delta[2] * delta[2] * delta[2]);
    } // pz
  } // end loop over primitives

}

double CuspedGaussians::GTO_d(double r, double (&dist_xi)[3], int ao_ind, int ao_type) {
  
  const double pi = 3.14159265358979323846;
  
  // intialize running sum
  double orb_total = 0.0;

  // sum over primitives
  for (int k = 0; k < m_ng; k++) {

    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind]; 
    double N = d * pow(2048.0 * pow(a,7.0) / pow(pi, 3.0), 0.25);
    
    // 3dxy
    if (ao_type == 5) {
      orb_total += N * dist_xi[0] * dist_xi[1] * std::exp(-a * r * r);
    }
    // 3dyz
    else if (ao_type == 6) {
      orb_total += N * dist_xi[1] * dist_xi[2] * std::exp(-a * r * r);
    }
    // 3dz^2
    else if (ao_type == 7) {
      double M = pow(144.0, -0.25);
      orb_total += M * N * (3.0 * pow(dist_xi[2], 2.0) - pow(r, 2.0)) * std::exp(-a * r * r);
    }
    // 3dxz
    else if (ao_type == 8) {
      orb_total += N * dist_xi[0] * dist_xi[2] * std::exp(-a * r * r);
    }
    // 3dx^2-y^2
    else {
      double M = 0.5;
      orb_total += M * N * (pow(dist_xi[0], 2.0) - pow(dist_xi[1], 2.0)) * std::exp(-a * r * r);
    }
  } 
  
  return orb_total;
}

void CuspedGaussians::d1_GTO_d(const double dist, double (&delta)[3], int ao_ind, int ao_type, double (&d1_vec)[3]) {

  const double pi = 3.14159265358979323846;

  // sum over primitives
  for (int k = 0; k < m_ng; k++) {
    
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind]; 
    double N = d * pow(2048.0 * pow(a,7.0) / pow(pi, 3.0), 0.25);
    double x = delta[0];
    double y = delta[1];
    double z = delta[2];
    double r = dist;
    double exp = std::exp(-a * dist * dist);

    // 3dxy
    if (ao_type == 5) {
      d1_vec[0] += N * y * (1.0 - 2.0 * a * x * x ) * exp;  
      d1_vec[1] += N * x * (1.0 - 2.0 * a * y * y ) * exp; 
      d1_vec[2] += -2.0 * N * a * x * y * z * exp;	  
    }
    // 3dyz
    else if (ao_type == 6) {
      d1_vec[0] += -2.0 * N * a * x * y * z * exp; 	  
      d1_vec[1] += N * z * (1.0 - 2.0 * a * y * y ) * exp;
      d1_vec[2] += N * y * (1.0 - 2.0 * a * z * z ) * exp;
    }
    // 3dz^2
    else if (ao_type == 7) {
      double M = pow(144.0, -0.25);
      d1_vec[0] += 2.0 * M * N * x * (a * r * r - 3.0 * a * z * z - 1.0) * exp;	
      d1_vec[1] += 2.0 * M * N * y * (a * r * r - 3.0 * a * z * z - 1.0) * exp;	
      d1_vec[2] += -2.0 * M * N * z * (2.0 * a * z * z - a * y * y - a * x * x - 2.0) * exp;
    }
    // 3dxz
    else if (ao_type == 8) {
      d1_vec[0] += N * z * (1.0 - 2.0 * a * x * x ) * exp; 
      d1_vec[1] += -2.0 * N * a * x * y * z * exp;	  
      d1_vec[2] += N * x * (1.0 - 2.0 * a * z * z ) * exp; 
    }
    // 3dx^2-y^2
    else {
      double M = 0.5; 
      d1_vec[0] += -2.0 * M * N * x * (a * x * x - a * y * y - 1.0) * exp; 
      d1_vec[1] += 2.0 * M * N * y * (a * y * y - a * x * x - 1.0) * exp; 
      d1_vec[2] += -2.0 * M * N * a * z * (x * x - y * y) * exp;         
    }
  } // end sum over primitives

} 
    
void CuspedGaussians::d2_GTO_d(const double dist, double (&delta)[3], int ao_ind, int ao_type, double (&d2_vec)[3]) {

  const double pi = 3.14159265358979323846;
  
  // sum over primitives
  for (int k = 0; k < m_ng; k++) {
 
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind]; 
    double N = d * pow(2048.0 * pow(a,7.0) / pow(pi, 3.0), 0.25);
    double x = delta[0];
    double y = delta[1];
    double z = delta[2];
    double r = dist;
    double a2 = a * a;
    double x2 = x * x;
    double y2 = y * y;
    double z2 = z * z;
    double exp = std::exp(-a * dist * dist);

    // 3dxy
    if (ao_type == 5) {
      d2_vec[0] += 2.0 * N * a * x * y * (2.0 * a * x * x - 3.0) * exp; 
      d2_vec[1] += 2.0 * N * a * x * y * (2.0 * a * y * y - 3.0) * exp;
      d2_vec[2] += 2.0 * N * a * x * y * (2.0 * a * z * z - 1.0) * exp; 
    }

    // 3dyz
    else if (ao_type == 6) {
      d2_vec[0] += 2.0 * N * a * y * z * (2.0 * a * x * x - 1.0) * exp; 
      d2_vec[1] += 2.0 * N * a * y * z * (2.0 * a * y * y - 3.0) * exp;
      d2_vec[2] += 2.0 * N * a * y * z * (2.0 * a * z * z - 3.0) * exp; 
    }

    // 3dz^2
    else if (ao_type == 7) {
      double M = pow(144.0, -0.25);
      d2_vec[0] += -2.0 * M * N * (2.0 * a * a * x * x * x * x + a * x * x * (2.0 * a * y * y - 4.0 * a * z * z - 5.0) - a * y * y + 2.0 * a * z * z + 1.0) * exp;
      d2_vec[1] += -2.0 * M * N * (2.0 * a * a * y * y * y * y + a * y * y * (2.0 * a * x * x - 4.0 * a * z * z - 5.0) - a * x * x + 2.0 * a * z * z + 1.0) * exp;
      d2_vec[2] += -2.0 * M * N * ((2.0 * a2 * z2 - a) * (x2 + y2) - 4.0 * a2 * z2 * z2 + 10.0 * a * z2 - 2.0) * exp; 						  
    }

    // 3dxz
    else if (ao_type == 8) {
      d2_vec[0] += 2.0 * N * a * x * z * (2.0 * a * x * x - 3.0) * exp;
      d2_vec[1] += 2.0 * N * a * x * z * (2.0 * a * y * y - 1.0) * exp;
      d2_vec[2] += 2.0 * N * a * x * z * (2.0 * a * z * z - 3.0) * exp; 
    }

    // 3dx^2-y^2
    else {
      double M = 0.5;
      d2_vec[0] += 2.0 * M * N * (2.0 * a * a * x * x * x * x - a * x * x * (2.0 * a * y * y + 5.0) + a * y * y + 1.0) * exp;  
      d2_vec[1] += 2.0 * M * N * (x2 * (2.0 * a2 * y2 - a) - 2.0 * a2 * y2 * y2 + 5.0 * a * y2 - 1.0) * exp; 		
      d2_vec[2] += 2.0 * M * N * a * (2.0 * a * z * z - 1.0) * (x * x - y * y) * exp;				
    }
  } // end sum over primitives

}
 
double CuspedGaussians::b_func(double rc, double nuc_dist) {
  
  // 5th order polynomial switching function parameters
  const double c1 = (-6.0 / pow(rc, 5.0));
  const double c2 = (15.0 / pow(rc, 4.0));
  const double c3 = (-10.0 / pow(rc, 3.0));
  const double c4 = 0.0;
  const double c5 = 0.0;
  const double c6 = 1.0;

  const double b_val = c1 * pow(nuc_dist, 5.0) + c2 * pow(nuc_dist, 4.0) + c3 * pow(nuc_dist, 3.0) + c4 * pow(nuc_dist, 2.0) + c5 * nuc_dist + c6;

  return b_val;
}

void CuspedGaussians::d1_b_func(double rc, double nuc_dist, int i, int n, double (&diff)[3], double (&d1_b)[3]) {
	
  // 5th order polynomial math
  const double c1 = (-6.0 / pow(rc, 5));
  const double c2 = (15.0 / pow(rc, 4));
  const double c3 = (-10.0 / pow(rc, 3));
  const double c4 = 0.0;
  const double c5 = 0.0;
  const double c6 = 1.0;

  // loop through xyz to calc derivative terms
  for (int l = 0; l < 3; l++) {

    d1_b[l] = diff[l] * (5 * c1 * pow(nuc_dist, 3) + 4 * c2 * pow(nuc_dist, 2) + 3 * c3 * nuc_dist + 2 * c4 + c5 / nuc_dist); 
  }

}

void CuspedGaussians::d2_b_func(double rc, double nuc_dist, int i, int n, double (&diff)[3], double (&d2_b)[3]) {
	
  // 5th order polynomial math
  const double c1 = (-6.0 / pow(rc, 5));
  const double c2 = (15.0 / pow(rc, 4));
  const double c3 = (-10.0 / pow(rc, 3));
  const double c4 = 0.0;
  const double c5 = 0.0;
  const double c6 = 1.0;

  // loop through xyz to calc derivative terms
  for (int l = 0; l < 3; l++) {
    
    d2_b[l] = c1 * (5 * pow(nuc_dist, 3) + 15 * pow(diff[l], 2) * nuc_dist) + c2 * (4 * pow(nuc_dist, 2) + 8 * pow(diff[l], 2)) + 3 * c3 * (nuc_dist + pow(diff[l], 2) / nuc_dist) + 2 * c4 + c5 * (1 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3));
  }

}

double CuspedGaussians::slater_func(double a0, double zeta, double r) {
  
  // evaluate f = a0 * exp(-zeta * |r-R|)
  double Q_fn = a0 * std::exp(-zeta * r);
  
  return Q_fn;
}

void CuspedGaussians::d1_slater_func(double a0, double zeta, double r, int i, int n, double (&diff)[3], double (&d1_Q_vec)[3]) {

  // sum over x, y, z
  for (int l = 0; l < 3; l++) {
  
    d1_Q_vec[l] = -(a0 * zeta * diff[l] * std::exp(-zeta * r)) / r;
  }
}

void CuspedGaussians::d2_slater_func(double a0, double zeta, double r, int i, int n, double (&diff)[3], double (&d2_Q_vec)[3]) {

  // sum over x, y, z
  for (int l = 0; l < 3; l++) {
  
    d2_Q_vec[l] = a0 * zeta * std::exp(-zeta * r) * ((zeta * pow(diff[l], 2.0)) / pow(r, 2.0) + pow(diff[l], 2.0) / pow(r, 3.0) - 1.0 / r);
  }

}

double CuspedGaussians::Pn_func(double b, double Q, double orb_total, double nuc_dist, double n, double rc) {

  // evaluate higher-order Slater functions
  return b * Q * pow(nuc_dist / rc, n);
}

void CuspedGaussians::d1_Pn_func(double &orb_total, double &b_val, double &Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double &nuc_dist, double (&diff)[3], double &n, double &qn, double (&d1_P_vec)[3], double &rc) {
	
  for (int l = 0; l < 3; l++) {

    d1_P_vec[l] += qn * ( b_val * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + n * Q_fn * pow(nuc_dist / rc, (n-1.0)) * diff[l] / (nuc_dist * rc)) + d1_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n) )); 
  }

}

void CuspedGaussians::d2_Pn_func(double &orb_total, double &b_val, double &Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double (&der2_total)[3], double (&d2_b_func)[3], double (&d2_slater_s_cusp)[3], double &nuc_dist, double (&diff)[3], double &n, double &qn, double (&d2_P_vec)[3], double &rc) {

  for (int l = 0; l < 3; l++) {

    d2_P_vec[l] += qn * ( b_val * ( d2_slater_s_cusp[l] * pow(nuc_dist / rc, n) + 2.0 * d1_slater_s_cusp[l] * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist + Q_fn * n * (n-1.0) * pow(nuc_dist / rc, (n-2.0)) * pow(diff[l], 2) / pow(nuc_dist, 2) / pow(rc, 2) + n / rc * Q_fn * pow(nuc_dist / rc, (n-1.0)) * (1.0 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3)) ) + 2.0 * d1_b_func[l] * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + Q_fn * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist) + d2_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n)));
  }
}   

void CuspedGaussians::d1_one_minus_b(double &orb_total, double &b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&d1_Pn_vec)[3]) {

  for (int l = 0; l < 3; l++) {

    d1_Pn_vec[l] += (1.0 - b_val) * der1_total[l] - d1_b_vec[l] * orb_total;  
  }

}

void CuspedGaussians::d2_one_minus_b(double &orb_total, double &b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&der2_total)[3], double (&d2_b_vec)[3], double (&d2_Pn_vec)[3]) {

  for (int l = 0; l < 3; l++) {

    d2_Pn_vec[l] += (1.0 - b_val) * der2_total[l] - 2.0 * d1_b_vec[l] * der1_total[l] - d2_b_vec[l] * orb_total;  
  }
}

void CuspedGaussians::evaluate_orbs(std::vector<double> &e_pos, std::vector<double> &n_pos, std::vector<double> &AO_mat) {

  // get number of electrons (can vary if moving single electron at a time)
  const int ne = e_pos.size() / 3;
  
  // get number of nuclei
  const int nn = n_pos.size() / 3;

  // loop over the AOs
  for (int p = 0; p < m_no; p++) {

    // loop over number of electrons
    for (int i = 0; i < ne; i++) {
      
      // initialize orbital total
      double orb_total;
      
      // get distances between electron and basis function center
      double dist_xi[3];
      dist_xi[0] = e_pos[i * 3 + 0] - n_pos[m_bf_cen[p] * 3 + 0];
      dist_xi[1] = e_pos[i * 3 + 1] - n_pos[m_bf_cen[p] * 3 + 1];
      dist_xi[2] = e_pos[i * 3 + 2] - n_pos[m_bf_cen[p] * 3 + 2];
      const double dist = std::sqrt( dist_xi[0] * dist_xi[0] + dist_xi[1] * dist_xi[1] + dist_xi[2] * dist_xi[2] );

      // if AO is s-type
      if (m_bf_type[p] == 0 || m_bf_type[p] == 1) {
     
        // if it's a core-bital
        if (m_orth_orb[p] == 0) {
	
          // evaluate uncusped orbital
          orb_total = GTO_s(dist, p);
	}
	
        // if it's a valence s orbital -- orthogonalize against core 
        else {
 
          // get the index of core-bital to orthogonalize against
	  int core_orb_ind = m_orth_orb[p] - 1;
	  
          // get projection of orb onto core from python
	  double proj = m_proj_mat[core_orb_ind * m_no + p];
	  
          // evaluate uncusped orbital
          orb_total = GTO_s(dist, p, core_orb_ind, proj);

	}

      } // end s orbital  
     
      // if AO is p-type
      else if (m_bf_type[p] == 2 || m_bf_type[p] == 3 || m_bf_type[p] == 4) {

        // evaluate uncusped orbital
        orb_total = GTO_p(dist, dist_xi, p, m_bf_type[p]);

      } // end p orbital

      // if AO is d-type
      else {
         
        // evaluate uncusped orbital
        orb_total = GTO_d(dist, dist_xi, p, m_bf_type[p]);
      }

      // initialize terms when within cusp radius
 
      // value of switching function
      double b_val = 0.0;
  
      // value of Slater function a0 * exp(-Z * |r-R|)
      double Q_fn = 0.0;

      // sum of Slater expansion
      double Pn_val = 0.0;

      // counter to check we are only within one nucleus at a time
      int counter = 0;
     
      // check if within cusp radius of a nuc
      for (int n = 0; n < nn; n++) {
      
        // check if we are cusping 
        if (m_cusp_a0_mat[p * nn + n] != 0.0) {
							
          // get electron-nuclear distance
          double dx = e_pos[i * 3 + 0] - n_pos[n * 3 + 0];
          double dy = e_pos[i * 3 + 1] - n_pos[n * 3 + 1];
          double dz = e_pos[i * 3 + 2] - n_pos[n * 3 + 2];
          const double nuc_dist = std::sqrt( dx * dx + dy * dy + dz * dz );
 
          // get cusp radius for this nuc
          double rc = m_cusp_radii_mat[p * nn + n];

          // if we're within the cusping radius
	  if (nuc_dist < rc) {
	
            // evaluate switching function
	    b_val = b_func(rc, nuc_dist);
		
            // get charge of nucleus to satisfy cusp condition
            double zeta = m_Z[n];

	    // evaluate slater function
	    Q_fn = slater_func(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist);
 
            // get number of Slater functions in expansion
            int npbf = m_n_vec.size();
		
            // sum over expansion
            for (int k = 0; k < npbf; k++) {
              
              Pn_val += m_cusp_coeff_mat[n * npbf + p * nn * npbf + k] * Pn_func(b_val, Q_fn, orb_total, nuc_dist, m_n_vec[k], rc);
	    }

            // add to counter 
            counter++;
				
          }  // if electron within cusp radius
	
        }  // if orbital over given nuc has a cusp, a0 != 0.0
		
      } // nuclei centers

      if (counter > 1)
        std::cerr << "CuspedGaussians::evaluate_orbs(), electron may be within only one nuclear center.\n";
      
      // add one minus b term
      Pn_val += (1.0 - b_val) * orb_total;
      
      // write orbital value to matrix
      AO_mat[i + p * ne] = Pn_val; 
 
    } // end loop over electrons

  } // end loop over orbitals

}
  

void CuspedGaussians::evaluate_derivs(std::vector<double> &e_pos, std::vector<double> &n_pos, double **der1, double **der2) {

  // get number of electrons
  const int ne = e_pos.size() / 3;

  // get number of nuclei
  const int nn = n_pos.size() / 3;

  // loop over atomic orbitals
  for (int p = 0; p < m_no; p++) {
				
    // loop over electrons
    for (int i = 0; i < ne; i++) {

      // get distances between electron and basis function center
      double dist_xi[3];
      dist_xi[0] = e_pos[i * 3 + 0] - n_pos[m_bf_cen[p] * 3 + 0];
      dist_xi[1] = e_pos[i * 3 + 1] - n_pos[m_bf_cen[p] * 3 + 1];
      dist_xi[2] = e_pos[i * 3 + 2] - n_pos[m_bf_cen[p] * 3 + 2];
      const double dist = std::sqrt( dist_xi[0] * dist_xi[0] + dist_xi[1] * dist_xi[1] + dist_xi[2] * dist_xi[2] );
    
      // initialize derivative totals

      // GTO orbital value
      double orb_total = 0.0;
 
      // 1st derivative x, y, z
      double der1_total[3] = {0.0, 0.0, 0.0};
      
      // 2nd derivative x, y, z
      double der2_total[3] = {0.0, 0.0, 0.0};
   
      // if s-type orbital
      if (m_bf_type[p] == 0 || m_bf_type[p] == 1) {

        // if 1s core electron (no orthogonalization)
        if (m_orth_orb[p] == 0) {
				
	  orb_total = GTO_s(dist, p);
	  d1_GTO_s(dist, p, dist_xi, der1_total);
          d2_GTO_s(dist, p, dist_xi, der2_total);
	}
	
        // if orthogonalized s-type orbital
        else {
							
          // index of core-bital to orthogonalize against
          int core_orb_ind = m_orth_orb[p] - 1;         
	  
          // get projection of orb onto core
          double proj = m_proj_mat[core_orb_ind * m_no + p];

	  orb_total = GTO_s(dist, p, core_orb_ind, proj);
	  d1_GTO_s(dist, p, core_orb_ind, proj, dist_xi, der1_total);
	  d2_GTO_s(dist, p, core_orb_ind, proj, dist_xi, der2_total);
	}
	
      } // end s orbital

      // if p-type orbital
      else if (m_bf_type[p] == 2 || m_bf_type[p] == 3 || m_bf_type[p] == 4) {  
						
        orb_total = GTO_p(dist, dist_xi, p, m_bf_type[p]);
        d1_GTO_p(dist, p, dist_xi, m_bf_type[p], der1_total);
	d2_GTO_p(dist, p, dist_xi, m_bf_type[p], der2_total);
      
      } // end p-orbital

      // if d-type orbital
      else {
        orb_total = GTO_d(dist, dist_xi, p, m_bf_type[p]);
        d1_GTO_d(dist, dist_xi, p, m_bf_type[p], der1_total);
        d2_GTO_d(dist, dist_xi, p, m_bf_type[p], der2_total);
      }  

      // initialize vectors for derivatives of cusp pieces
      double b_val = 0.0;
      double d1_b_vec[3] = {0.0, 0.0, 0.0};
      double d2_b_vec[3] = {0.0, 0.0, 0.0};

      double Q_fn = 0.0;
      double d1_Q_vec[3] = {0.0, 0.0, 0.0};
      double d2_Q_vec[3] = {0.0, 0.0, 0.0};

      double d1_Pn_vec[3] = {0.0, 0.0, 0.0};
      double d2_Pn_vec[3] = {0.0, 0.0, 0.0};

      int counter = 0;

      // loop through nuclei to cusp if within radius
      for (int n = 0; n < nn; n++) {
        
        // if cusping this orbital
        if (m_cusp_a0_mat[p * nn + n] != 0.0) {
 
          // get electron-nuclear distance
          double diff[3];
          diff[0] = e_pos[i * 3 + 0] - n_pos[n * 3 + 0];
          diff[1] = e_pos[i * 3 + 1] - n_pos[n * 3 + 1];
          diff[2] = e_pos[i * 3 + 2] - n_pos[n * 3 + 2];
          double nuc_dist = std::sqrt( diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] );

	  // get cusp radius for this nuc
          double rc = m_cusp_radii_mat[p * nn + n];

	  // if within cusp radius (can only be true for one nucleus) 
	  if (nuc_dist < rc) {
	
            // add to counter
            counter++;

	    // evaluate switching function and derivs
	    b_val = b_func(rc, nuc_dist);
	    d1_b_func(rc, nuc_dist, i, n, diff, d1_b_vec);
	    d2_b_func(rc, nuc_dist, i, n, diff, d2_b_vec);

            // get charge of nuc to satisfy cusp condition
 	    double zeta = m_Z[n];

	    // evaluate slater function and derivs
	    Q_fn = slater_func(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist);
	    d1_slater_func(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist, i, n, diff, d1_Q_vec);
	    d2_slater_func(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist, i, n, diff, d2_Q_vec);

	    // get number of higher-order Slater functions for sum
            int npbf = m_n_vec.size();

            // loop over sum of Slater functions and get derivatives
	    for (int k = 0; k < npbf; k++) {  
	    
              d1_Pn_func(orb_total, b_val, Q_fn, der1_total, d1_b_vec, d1_Q_vec, nuc_dist, diff, m_n_vec[k], m_cusp_coeff_mat[n * npbf + p * nn * npbf + k], d1_Pn_vec, rc);
	      d2_Pn_func(orb_total, b_val, Q_fn, der1_total, d1_b_vec, d1_Q_vec, der2_total, d2_b_vec, d2_Q_vec, nuc_dist, diff, m_n_vec[k], m_cusp_coeff_mat[n * npbf + p * nn * npbf + k], d2_Pn_vec, rc);
	    }

	  }  // if electron in cusping region
	
        } // if orbital over current nucleus is cusped, a0 != 0.0
					
      }	// end loop over nuclear centers

      if (counter > 1)
        std::cerr << "CuspedGaussians::evaluate_derivs(), electron may be within only one nuclear center.\n";

      // add derivative contributions from one minus b term to d_Pn_vec
      d1_one_minus_b(orb_total, b_val, der1_total, d1_b_vec, d1_Pn_vec); 
      d2_one_minus_b(orb_total, b_val, der1_total, d1_b_vec, der2_total, d2_b_vec, d2_Pn_vec);

      // fill derivative values
      der1[0][i + p * ne] = d1_Pn_vec[0];
      der1[1][i + p * ne] = d1_Pn_vec[1];
      der1[2][i + p * ne] = d1_Pn_vec[2];

      der2[0][i + p * ne] = d2_Pn_vec[0];
      der2[1][i + p * ne] = d2_Pn_vec[1];
      der2[2][i + p * ne] = d2_Pn_vec[2];
 
    } // end loop over electrons

  } // end loop over atomic orbitals

}

