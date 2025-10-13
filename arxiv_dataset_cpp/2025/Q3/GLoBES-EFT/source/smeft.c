  // ----------------------------------------------------------------------------
// Probability engine for simulating sterile neutrinos + non-standard
// interactions
// ----------------------------------------------------------------------------
// Author: Joachim Kopp (jkopp@fnal.gov)
// ----------------------------------------------------------------------------
// GLoBES -- General LOng Baseline Experiment Simulator
// (C) 2002 - 2010,  The GLoBES Team
//
// GLoBES as well as this add-on are mainly intended for academic purposes.
// Proper credit must be given if you use GLoBES or parts of it. Please
// read the section 'Credit' in the README file.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// ----------------------------------------------------------------------------
// Note: This file is written in C99. To compile in gcc use -std=c99 or
// -std=gnu99
// ----------------------------------------------------------------------------
// ChangeLog:
//   2011-01-14: - Implemented filter feature for n_leptonflavors > 3
//               - New function smeft_probability_matrix_all returns
//                 oscillation probabilities to/from sterile flavors
//                 (the standard smeft_probability_matrix returns only
//                 probabilities for oscillations among the 3 standard flavors
//                 for compatibility with GLoBES.)
// ----------------------------------------------------------------------------
// Citation information:
//
//      @Article{Kopp:2006wp,
//        author    = "Kopp, Joachim",
//        title     = "{Efficient numerical diagonalization of hermitian
//                     $3 \times 3$ matrices}",
//        journal   = "Int. J. Mod. Phys.",
//        volume    = "C19",
//        year      = "2008",
//        pages     = "523-548",
//        eprint    = "physics/0610206",
//        archivePrefix = "arXiv",
//        doi       = "10.1142/S0129183108012303",
//        SLACcitation  = "%%CITATION = PHYSICS/0610206;%%",
//        note      = "Erratum ibid.\ {\bf C19} (2008) 845",
//        memo      = "Algorithms for fast diagonalization of 3x3 matrices
//          (used for <= 3 neutrino flavors)"
//      }
//
//      @Article{Kopp:2007ne,
//        author    = "Kopp, Joachim and Lindner, Manfred and Ota,
//                     Toshihiko and Sato, Joe",
//        title     = "{Non-standard neutrino interactions in reactor and
//                     superbeam experiments}",
//        journal   = "Phys. Rev.",
//        volume    = "D77",
//        year      = "2008",
//        pages     = "013007",
//        eprint    = "0708.0152",
//        archivePrefix = "arXiv",
//       primaryClass  =  "hep-ph",
//       doi       = "10.1103/PhysRevD.77.013007",
//       SLACcitation  = "%%CITATION = 0708.0152;%%",
//       memo      = "This is the first paper in which an early version
//         of the present NSI engine was used.",
//     }
// ----------------------------------------------------------------------------
#if HAVE_CONFIG_H   /* config.h should come before any other includes */
#  include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include "globes/globes.h"
#include "glb_error.h"
#include "smeft.h"
#include "glb_multiex.h"
#include "Running_and_matching.h"

// Constants
#define GLB_V_FACTOR        7.5e-14    // Conversion factor for matter potentials
#define GLB_Ne_MANTLE       0.5        // Effective electron numbers for calculation
#define GLB_Ne_CORE         0.468      //   of MSW potentials
#define RHO_THRESHOLD       0.001      // The minimum matter density below which
                                       // vacuum algorithms are used
#define M_SQRT3  1.73205080756887729352744634151     // sqrt(3)


// Macros
#define SQR(x)      ((x)*(x))                        // x^2
#define SQR_ABS(x)  (SQR(creal(x)) + SQR(cimag(x)))  // |x|^2
#define POW10(x)    (exp(M_LN10*(x)))                // 10^x
#define MIN(X,Y)    ( ((X) < (Y)) ? (X) : (Y) )
#define MAX(X,Y)    ( ((X) > (Y)) ? (X) : (Y) )
#define SIGN(a,b)   ( (b) > 0.0 ? (fabs(a)) : (-fabs(a)) )

// Maximum number of neutrino species
/*#define MAX_PARAMS    (SQR(MAX_FLAVORS)-MAX_FLAVORS)+(2 * 5 * SQR(MAX_FLAVORS) * 6)+(SQR(MAX_FLAVORS)*2*3)*/
#define MAX_PARAMS 2405
                              //SQR(MAX_FLAVORS) * (3 * MAX_INTERACTIONS+1) - MAX_FLAVORS
#define MAX_ANGLES    ((MAX_FLAVORS * (MAX_FLAVORS-1))/2)
#define MAX_PHASES    (((MAX_FLAVORS-1)*(MAX_FLAVORS-2))/2)

// Fundamental oscillation parameters
int n_leptonflavors = 3;
int n_quarks = 0;
int n_params  = 0;
int n_angles  = 0;
int n_phases  = 0;
static double th[MAX_FLAVORS+1][MAX_FLAVORS+1];// Mixing angles
static double delta[MAX_PHASES];            // Dirac CP phase
static double dmsq[MAX_FLAVORS-1];         // Mass squared differences
static int has_run = 0;
static double running_smeft[3][3] = {0}; // Initialize with zeros
static double running_weft_mb[5][5] = {0}; // Empty initialization
static double running_weft_low_scale[5][5] = {0}; // Empty initialization
double SMEFT_FLAG;
struct indices
  {
    int flux_id;
    int cross_id;
    int d1;
    int d2;
    int d3;
    int d4;
  };




struct WEFT weft;
struct SMEFT smeft;


// Names of NSI parameters
char smeft_param_strings[MAX_PARAMS][64];

// Internal temporary variables
static gsl_matrix_complex *U=NULL; // The vacuum mixing matrix
static gsl_matrix_complex *H=NULL; // Neutrino Hamiltonian
static gsl_matrix_complex *Q=NULL; // Eigenvectors of Hamiltonian (= eff. mixing matrix)
static gsl_vector *lambda=NULL;    // Eigenvalues of Hamiltonian
static gsl_matrix_complex *S=NULL; // The neutrino S-matrix

static gsl_matrix_complex *H0_template=NULL;  // Used in the construction of the vac. Hamiltonian
static gsl_matrix_complex *S1=NULL, *T0=NULL; // Temporary matrix storage
static gsl_matrix_complex *Q1=NULL, *Q2=NULL; // More temporary storage

static gsl_eigen_hermv_workspace *w=NULL;     // Workspace for eigenvector algorithm
complex double QTil[MAX_INTERACTIONS][2][3][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino

extern int density_corr[];

// The order in which the rotation matrices corresponding to the different
// mixing angles are multiplied together (numbers are indices to th[][]
static int rotation_order[MAX_ANGLES][2];

// Which rotation matrices contain the complex phases? Indices are to
// delta[], -1 indicates no complex phase in a particular matrix;
// phase_order[0] corresponds to the leftmost rotation matrix
static int phase_order[MAX_ANGLES];


// ----------------------------------------------------------------------------
//     3 x 3   E I G E N S Y S T E M   F U N C T I O N S  (physics/0610206)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
static void zhetrd3(double complex A[3][3], double complex Q[3][3],
                    double d[3], double e[2])
// ----------------------------------------------------------------------------
// Reduces a hermitian 3x3 matrix to real tridiagonal form by applying
// (unitary) Householder transformations:
//            [ d[0]  e[0]       ]
//    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//            [       e[1]  d[2] ]
// The function accesses only the diagonal and upper triangular parts of
// A. The access is read-only.
// ---------------------------------------------------------------------------
{
  const int n = 3;
  double complex u[n], q[n];
  double complex omega, f;
  double K, h, g;

  // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }
#endif

  // Bring first row and column to the desired form
  h = SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
  if (creal(A[0][1]) > 0)
    g = -sqrt(h);
  else
    g = sqrt(h);
  e[0] = g;
  f    = g * A[0][1];
  u[1] = conj(A[0][1]) - g;
  u[2] = conj(A[0][2]);

  omega = h - f;
  if (creal(omega) > 0.0)
  {
    omega = 0.5 * (1.0 + conj(omega)/omega) / creal(omega);
    K = 0.0;
    for (int i=1; i < n; i++)
    {
      f    = conj(A[1][i]) * u[1] + A[i][2] * u[2];
      q[i] = omega * f;                  // p
      K   += creal(conj(u[i]) * f);      // u* A u
    }
    K *= 0.5 * SQR_ABS(omega);

    for (int i=1; i < n; i++)
      q[i] = q[i] - K * u[i];

    d[0] = creal(A[0][0]);
    d[1] = creal(A[1][1]) - 2.0*creal(q[1]*conj(u[1]));
    d[2] = creal(A[2][2]) - 2.0*creal(q[2]*conj(u[2]));

    // Store inverse Householder transformation in Q
#ifndef EVALS_ONLY
    for (int j=1; j < n; j++)
    {
      f = omega * conj(u[j]);
      for (int i=1; i < n; i++)
        Q[i][j] = Q[i][j] - f*u[i];
    }
#endif

    // Calculate updated A[1][2] and store it in f
    f = A[1][2] - q[1]*conj(u[2]) - u[1]*conj(q[2]);
  }
  else
  {
    for (int i=0; i < n; i++)
      d[i] = creal(A[i][i]);
    f = A[1][2];
  }

  // Make (23) element real
  e[1] = cabs(f);
#ifndef EVALS_ONLY
  if (e[1] != 0.0)
  {
    f = conj(f) / e[1];
    for (int i=1; i < n; i++)
      Q[i][n-1] = Q[i][n-1] * f;
  }
#endif
}


// ----------------------------------------------------------------------------
static int zheevc3(double complex A[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a hermitian 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  double m, c1, c0;

  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  double complex de = A[0][1] * A[1][2];                            // d * e
  double dd = SQR_ABS(A[0][1]);                                  // d * conj(d)
  double ee = SQR_ABS(A[1][2]);                                  // e * conj(e)
  double ff = SQR_ABS(A[0][2]);                                  // f * conj(f)
  m  = creal(A[0][0]) + creal(A[1][1]) + creal(A[2][2]);
  c1 = (creal(A[0][0])*creal(A[1][1])  // a*b + a*c + b*c - d*conj(d) - e*conj(e) - f*conj(f)
          + creal(A[0][0])*creal(A[2][2])
          + creal(A[1][1])*creal(A[2][2]))
          - (dd + ee + ff);
  c0 = creal(A[2][2])*dd + creal(A[0][0])*ee + creal(A[1][1])*ff
            - creal(A[0][0])*creal(A[1][1])*creal(A[2][2])
            - 2.0 * (creal(A[0][2])*creal(de) + cimag(A[0][2])*cimag(de));
                             // c*d*conj(d) + a*e*conj(e) + b*f*conj(f) - a*b*c - 2*Re(conj(f)*d*e)

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);

  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

  return 0;
}


// ----------------------------------------------------------------------------
static int zheevq3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using the QL algorithm with implicit shifts, preceded by a
// Householder reduction to real tridiagonal form.
// The function accesses only the diagonal and upper triangular parts of A.
// The access is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
// ----------------------------------------------------------------------------
// Dependencies:
//   zhetrd3()
// ----------------------------------------------------------------------------
{
  const int n = 3;
  double e[3];                 // The third element is used only as temporary workspace
  double g, r, p, f, b, s, c;  // Intermediate storage
  double complex t;
  int nIter;
  int m;

  // Transform A to real tridiagonal form by the Householder method
  zhetrd3(A, Q, w, e);

  // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  // with the QL method
  //
  // Loop over all off-diagonal elements
  for (int l=0; l < n-1; l++)
  {
    nIter = 0;
    while (1)
    {
      // Check for convergence and exit iteration loop if off-diagonal
      // element e(l) is zero
      for (m=l; m <= n-2; m++)
      {
        g = fabs(w[m])+fabs(w[m+1]);
        if (fabs(e[m]) + g == g)
          break;
      }
      if (m == l)
        break;

      if (nIter++ >= 30)
        return -1;

      // Calculate g = d_m - k
      g = (w[l+1] - w[l]) / (e[l] + e[l]);
      r = sqrt(SQR(g) + 1.0);
      if (g > 0)
        g = w[m] - w[l] + e[l]/(g + r);
      else
        g = w[m] - w[l] + e[l]/(g - r);

      s = c = 1.0;
      p = 0.0;
      for (int i=m-1; i >= l; i--)
      {
        f = s * e[i];
        b = c * e[i];
        if (fabs(f) > fabs(g))
        {
          c      = g / f;
          r      = sqrt(SQR(c) + 1.0);
          e[i+1] = f * r;
          c     *= (s = 1.0/r);
        }
        else
        {
          s      = f / g;
          r      = sqrt(SQR(s) + 1.0);
          e[i+1] = g * r;
          s     *= (c = 1.0/r);
        }

        g = w[i+1] - p;
        r = (w[i] - g)*s + 2.0*c*b;
        p = s * r;
        w[i+1] = g + p;
        g = c*r - b;

        // Form eigenvectors
#ifndef EVALS_ONLY
        for (int k=0; k < n; k++)
        {
          t = Q[k][i+1];
          Q[k][i+1] = s*Q[k][i] + c*t;
          Q[k][i]   = c*Q[k][i] - s*t;
        }
#endif
      }
      w[l] -= p;
      e[l]  = g;
      e[m]  = 0.0;
    }
  }

  return 0;
}


// ----------------------------------------------------------------------------
static int zheevh3(double complex A[3][3], double complex Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors. However,
// if conditions are such that a large error in the results is to be
// expected, the routine falls back to using the slower, but more
// accurate QL algorithm. Only the diagonal and upper triangular parts of A need
// to contain meaningful values. Access to A is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The hermitian input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   zheevc3(), zhetrd3(), zheevq3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1: Simplified fallback condition --> speed-up
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  double norm;          // Squared norm or inverse norm of current eigenvector
//  double n0, n1;        // Norm of first and second columns of A
  double error;         // Estimated maximum roundoff error
  double t, u;          // Intermediate storage
  int j;                // Loop counter
#endif

  // Calculate eigenvalues
  zheevc3(A, w);

#ifndef EVALS_ONLY
//  n0 = SQR(creal(A[0][0])) + SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
//  n1 = SQR_ABS(A[0][1]) + SQR(creal(A[1][1])) + SQR_ABS(A[1][2]);

  t = fabs(w[0]);
  if ((u=fabs(w[1])) > t)
    t = u;
  if ((u=fabs(w[2])) > t)
    t = u;
  if (t < 1.0)
    u = t;
  else
    u = SQR(t);
  error = 256.0 * DBL_EPSILON * SQR(u);
//  error = 256.0 * DBL_EPSILON * (n0 + u) * (n1 + u);

  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*creal(A[1][1]);
  Q[1][1] = A[0][2]*conj(A[0][1]) - A[1][2]*creal(A[0][0]);
  Q[2][1] = SQR_ABS(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = conj( (A - w[0]).e1 x (A - w[0]).e2 )
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = (creal(A[0][0]) - w[0]) * (creal(A[1][1]) - w[0]) - Q[2][1];
  norm    = SQR_ABS(Q[0][0]) + SQR_ABS(Q[1][0]) + SQR(creal(Q[2][0]));

  // If vectors are nearly linearly dependent, or if there might have
  // been large cancellations in the calculation of A(I,I) - W(1), fall
  // back to QL algorithm
  // Note that this simultaneously ensures that multiple eigenvalues do
  // not cause problems: If W(1) = W(2), then A - W(1) * I has rank 1,
  // i.e. all columns of A - W(1) * I are linearly dependent.
  if (norm <= error)
    return zheevq3(A, Q, w);
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }

  // Calculate second eigenvector by the formula
  //   v[1] = conj( (A - w[1]).e1 x (A - w[1]).e2 )
  Q[0][1]  = Q[0][1] + A[0][2]*w[1];
  Q[1][1]  = Q[1][1] + A[1][2]*w[1];
  Q[2][1]  = (creal(A[0][0]) - w[1]) * (creal(A[1][1]) - w[1]) - creal(Q[2][1]);
  norm     = SQR_ABS(Q[0][1]) + SQR_ABS(Q[1][1]) + SQR(creal(Q[2][1]));
  if (norm <= error)
    return zheevq3(A, Q, w);
  else
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][1] = Q[j][1] * norm;
  }

  // Calculate third eigenvector according to
  //   v[2] = conj(v[0] x v[1])
  Q[0][2] = conj(Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1]);
  Q[1][2] = conj(Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1]);
  Q[2][2] = conj(Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1]);
#endif

  return 0;
}

// ----------------------------------------------------------------------------
//                    I N T E R N A L   F U N C T I O N S
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
int smeft_print_gsl_matrix_complex(gsl_matrix_complex *A)
// ----------------------------------------------------------------------------
// Print entries of a complex GSL matrix in human-readable form
// ----------------------------------------------------------------------------
{
  int i, j;
  for (i=0; i < A->size1; i++)
  {
    for (j=0; j < A->size2; j++)
    {
    gsl_complex w = gsl_matrix_complex_get(A, i, j);
    printf("%.4f +%.4f*I   ", GSL_REAL(w),GSL_IMAG(w));
    }
    printf("\n");
  }

  return 0;
}


// ----------------------------------------------------------------------------
int smeft_init_probability_engine_3()
// ----------------------------------------------------------------------------
// Initialize probability engine for the 3-flavor case with NSI (no steriles)
// ----------------------------------------------------------------------------
{
  int rotation_order[][2] = { {2,3}, {1,3}, {1,2} };
  int phase_order[] = { -1, 0, -1 };
  return smeft_init_probability_engine(3, rotation_order, phase_order);
}


// ----------------------------------------------------------------------------
int smeft_init_probability_engine(int _n_leptonflavors, int _rotation_order[][2], int _phase_order[])
// ----------------------------------------------------------------------------
// Allocates internal data structures for the probability engine.
// ----------------------------------------------------------------------------
{
  if (_n_leptonflavors < 3 || _n_leptonflavors > MAX_FLAVORS)
  {
    fprintf(stderr, "smeft_init_probability_engine: Too many or too few neutrino flavors (%d).\n",
            _n_leptonflavors);
    return -1;
  }



  // Number of oscillation parameters:
  // For "n" lepton number, we have:
  // 1)  n (n-1)/2 mixing angles, (n-1)(n-2)/2 phases, n-1 mass squared differences. Total= n(n-1)

  // 2)  For each ``ij'' quark index (6 in total: ud, us, ub, cd, cs, cb), and each interaction ``X'' (5 in total) there are
  //   6*5*n^2 |\eps^ij_X| and 6*5*n^2 n_phases. The total is =60*n^2

  // 3) For the NC case, for each fermion ``f'' (3 in total: e, u, d) and interactions ``X'' (only L and R here, so 2 in total) there are
  //   3*2* n (n+1)/2 |\eps^m|, 3*2* n(n-1)/2 \phi^m
  //   which gives 6*n^2 parameters

  // The total number of parameters is = n(n-1)+2 * 6 * 5 * n^2+ 2 * 3 * n^2 = 67 * n^2-n
  n_leptonflavors = _n_leptonflavors;
  //n_params  = (SQR(n_leptonflavors)-n_leptonflavors)+(2 * 5 * SQR(n_leptonflavors) * 6)+(SQR(n_leptonflavors)*2*3);
  n_params = MAX_PARAMS;
  n_angles  = (n_leptonflavors * (n_leptonflavors-1))/2;
  n_phases  = ((n_leptonflavors-1)*(n_leptonflavors-2))/2;

  smeft_free_probability_engine();

  U = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  H = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  Q = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  lambda = gsl_vector_alloc(n_leptonflavors);
  S = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);

  H0_template = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  S1 = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  T0 = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  Q1 = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);
  Q2 = gsl_matrix_complex_calloc(n_leptonflavors, n_leptonflavors);


  w  = gsl_eigen_hermv_alloc(n_leptonflavors);

  for (int i=0; i < n_angles; i++)
  {
    if (_rotation_order[i][0] < 1 || _rotation_order[i][0] > n_angles ||
        _rotation_order[i][1] < 1 || _rotation_order[i][1] > n_angles)
    {
      fprintf(stderr, "smeft_init_probability_engine: Incorrect rotation order specification.\n");
      return -2;
    }
    if (_phase_order[i] >= n_phases)
    {
      fprintf(stderr, "smeft_init_probability_engine: Incorrect phase order specification.\n");
      return -3;
    }
    rotation_order[i][0] = _rotation_order[i][0];
    rotation_order[i][1] = _rotation_order[i][1];
    phase_order[i]       = _phase_order[i];
  }


  // Define names of oscillation parameters
  sprintf(smeft_param_strings[0], "%s", "TH12");    // Standard oscillation parameters
  sprintf(smeft_param_strings[1], "%s", "TH13");
  sprintf(smeft_param_strings[2], "%s", "TH23");
  sprintf(smeft_param_strings[3], "%s", "DELTA_0");
  sprintf(smeft_param_strings[4], "%s", "DM21");
  sprintf(smeft_param_strings[5], "%s", "DM31");

  int k = 6;
  for (int i=4; i <= n_leptonflavors; i++)            // Mass squared differences
    sprintf(smeft_param_strings[k++], "DM%d1", i);

  for (int i=1; i <= n_leptonflavors; i++)            // Sterile mixing angles
    for (int j=MAX(i+1,4); j <= n_leptonflavors; j++)
      sprintf(smeft_param_strings[k++], "TH%d%d", i, j);

  for (int i=1; i <= n_phases-1; i++)
    sprintf(smeft_param_strings[k++], "DELTA_%d", i); // Sterile phases

  const char *laptonflavors[] = { "E", "MU", "TAU", "S1", "S2", "S3", "S4", "S5", "S6" };
  const char *leptonflavors[] = { "E", "MU", "TAU", "S1", "S2", "S3", "S4", "S5", "S6" };
  const char *upquarks[] = { "u", "c","t" };
  const char *downquarks[] = { "d", "s", "b" };
  const char *matterfermions[] = { "E", "u", "d" };
  const char *interactions[] = { "L", "R", "S", "P", "T" };
  const char *quarkfamilies[] = { "Q1", "Q2", "Q3" };


  for (int x=0; x < 5; x++)                   // Production/Detection NSI: CC WEFT
    for (int l=0; l < 2; l++)
      for (int m=0; m < 3; m++)
          for (int i=0; i < n_leptonflavors; i++)
              for (int j=0; j < n_leptonflavors; j++)
    {
      sprintf(smeft_param_strings[k++], "ABS_EPS_CC_%s_%s%s_%s%s", interactions[x], upquarks[l], downquarks[m], laptonflavors[i], laptonflavors[j]);
      sprintf(smeft_param_strings[k++], "ARG_EPS_CC_%s_%s%s_%s%s", interactions[x], upquarks[l], downquarks[m], laptonflavors[i], laptonflavors[j]);
    }


    for (int x=0; x < 2; x++)     // Propagation NSI: NC WEFT
      for (int f=0; f < 3; f++)
        for (int i=0; i < n_leptonflavors; i++)
  {
    sprintf(smeft_param_strings[k++], "EPS_NC_%s_%s%s_%s%s", interactions[x], matterfermions[f], matterfermions[f], laptonflavors[i], laptonflavors[i]);
    for (int j=i+1; j < n_leptonflavors; j++)
    {
      sprintf(smeft_param_strings[k++], "ABS_EPS_NC_%s_%s%s_%s%s", interactions[x], matterfermions[f], matterfermions[f], laptonflavors[i], laptonflavors[j]);
      sprintf(smeft_param_strings[k++], "ARG_EPS_NC_%s_%s%s_%s%s", interactions[x], matterfermions[f], matterfermions[f], laptonflavors[i], laptonflavors[j]);
    }
  }
  
// Define names of SMEFT parameters



// Define modulus and argument for each SMEFT parameter

// Define parameters for c3_phi_l[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C3_PHI_L_%s%s", leptonflavors[p], leptonflavors[r]);
        sprintf(smeft_param_strings[k++], "ARG_C3_PHI_L_%s%s", leptonflavors[p], leptonflavors[r]);
    }
}

// Define parameters for c_phi_u[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C_PHI_U_%s%s", upquarks[p], upquarks[r]);
        sprintf(smeft_param_strings[k++], "ARG_C_PHI_U_%s%s", upquarks[p], upquarks[r]);
    }
}

// Define parameters for c_phi_d[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C_PHI_D_%s%s", downquarks[p], downquarks[r]);
        sprintf(smeft_param_strings[k++], "ARG_C_PHI_D_%s%s", downquarks[p], downquarks[r]);
    }
}

// Define parameters for c_phi_e[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C_PHI_E_%s%s", leptonflavors[p], leptonflavors[r]);
        sprintf(smeft_param_strings[k++], "ARG_C_PHI_E_%s%s", leptonflavors[p], leptonflavors[r]);
    }
}


for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C_PHI_Q_%s%s", quarkfamilies[p], quarkfamilies[r]);
        sprintf(smeft_param_strings[k++], "ARG_C_PHI_Q_%s%s", quarkfamilies[p], quarkfamilies[r]);
    }
}

// Define parameters for c_phi_l[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C_PHI_L_%s%s", leptonflavors[p], leptonflavors[r]);
        sprintf(smeft_param_strings[k++], "ARG_C_PHI_L_%s%s", leptonflavors[p], leptonflavors[r]);
    }
}



// Define parameters for c3_phi_q[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C3_PHI_Q_%s%s", quarkfamilies[p], quarkfamilies[r]);
        sprintf(smeft_param_strings[k++], "ARG_C3_PHI_Q_%s%s", quarkfamilies[p], quarkfamilies[r]);
    }
}

// Define parameters for c3_phi_e[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C3_PHI_E_%s%s", leptonflavors[p], leptonflavors[r]);
        sprintf(smeft_param_strings[k++], "ARG_C3_PHI_E_%s%s", leptonflavors[p], leptonflavors[r]);
    }
}

// Define parameters for c3_phi_le[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C3_PHI_LE_%s%s", leptonflavors[p], leptonflavors[r]);
        sprintf(smeft_param_strings[k++], "ARG_C3_PHI_LE_%s%s", leptonflavors[p], leptonflavors[r]);
    }
}

// Define parameters for c_phi_ud[3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        sprintf(smeft_param_strings[k++], "ABS_C_PHI_UD_%s%s", upquarks[p], downquarks[r]);
        sprintf(smeft_param_strings[k++], "ARG_C_PHI_UD_%s%s", upquarks[p], downquarks[r]);
    }
}

// Define parameters for c_ll[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LL_%s%s_%s%s", leptonflavors[p], leptonflavors[r], leptonflavors[s], leptonflavors[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LL_%s%s_%s%s", leptonflavors[p], leptonflavors[r], leptonflavors[s], leptonflavors[t]);
            }
        }
    }
}

// Define parameters for c3_lq[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C3_LQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], quarkfamilies[t]);
                sprintf(smeft_param_strings[k++], "ARG_C3_LQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], quarkfamilies[t]);
            }
        }
    }
}

// Define parameters for c_lq[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], quarkfamilies[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], quarkfamilies[t]);
            }
        }
    }
}

// Define parameters for c_ld[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LD_%s%s_%s%s", leptonflavors[p], leptonflavors[r], downquarks[s], downquarks[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LD_%s%s_%s%s", leptonflavors[p], leptonflavors[r], downquarks[s], downquarks[t]);
            }
        }
    }
}

// Define parameters for c_ld[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LU_%s%s_%s%s", leptonflavors[p], leptonflavors[r], upquarks[s], upquarks[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LU_%s%s_%s%s", leptonflavors[p], leptonflavors[r], upquarks[s], upquarks[t]);
            }
        }
    }
}

// Define parameters for c_le[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LE_%s%s_%s%s", leptonflavors[p], leptonflavors[r], leptonflavors[s], leptonflavors[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LE_%s%s_%s%s", leptonflavors[p], leptonflavors[r], leptonflavors[s], leptonflavors[t]);
            }
        }
    }
}

// Define parameters for c_ledq[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LEDQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], downquarks[s], quarkfamilies[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LEDQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], downquarks[s], quarkfamilies[t]);
            }
        }
    }
}

// Define parameters for c_lequ[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C_LEQU_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], upquarks[t]);
                sprintf(smeft_param_strings[k++], "ARG_C_LEQU_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], upquarks[t]);
            }
        }
    }
}

// Define parameters for c3_lequ[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C3_LEQU_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], upquarks[t]);
                sprintf(smeft_param_strings[k++], "ARG_C3_LEQU_%s%s_%s%s", leptonflavors[p], leptonflavors[r], quarkfamilies[s], upquarks[t]);
            }
        }
    }
}

// Define parameters for c3_ledq[3][3][3][3]
for (int p = 0; p < 3; p++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                sprintf(smeft_param_strings[k++], "ABS_C3_LEDQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], downquarks[s], quarkfamilies[t]);
                sprintf(smeft_param_strings[k++], "ARG_C3_LEDQ_%s%s_%s%s", leptonflavors[p], leptonflavors[r], downquarks[s], quarkfamilies[t]);
            }
        }
    }
}

// Define parameters for c_HWB
sprintf(smeft_param_strings[k++], "ABS_C_HWB");
sprintf(smeft_param_strings[k++], "ARG_C_HWB");

// Define parameters for c_HD
sprintf(smeft_param_strings[k++], "ABS_C_HD");
sprintf(smeft_param_strings[k++], "ARG_C_HD");

sprintf(smeft_param_strings[k++], "SMEFT_FLAG");


  if (k != n_params)
  {
    fprintf(stderr, "smeft_init_probability_engine: n_params has an incorrect value (%d).\n",
            n_params);
    return -2;
  }


  return 0;
}


// ----------------------------------------------------------------------------
int smeft_free_probability_engine()
// ----------------------------------------------------------------------------
// Destroys internal data structures of the probability engine.
// ----------------------------------------------------------------------------
{
  if (w !=NULL)     { gsl_eigen_hermv_free(w);      w  = NULL; }

  if (Q2!=NULL)     { gsl_matrix_complex_free(Q2);  Q2 = NULL; }
  if (Q1!=NULL)     { gsl_matrix_complex_free(Q1);  Q1 = NULL; }
  if (T0!=NULL)     { gsl_matrix_complex_free(T0);  T0 = NULL; }
  if (S1!=NULL)     { gsl_matrix_complex_free(S1);  S1 = NULL; }
  if (H0_template!=NULL) { gsl_matrix_complex_free(H0_template);  H0_template = NULL; }
  if (H!=NULL) { gsl_matrix_complex_free(H);  H = NULL; }

  if (S!=NULL)      { gsl_matrix_complex_free(S);   S = NULL; }
  if (lambda!=NULL) { gsl_vector_free(lambda);      lambda = NULL; }
  if (Q!=NULL)      { gsl_matrix_complex_free(Q);   Q = NULL; }
  if (H!=NULL)      { gsl_matrix_complex_free(H);   H = NULL; }
  if (U!=NULL)      { gsl_matrix_complex_free(U);   U = NULL; }

  return 0;
}

// ----------------------------------------------------------------------------
int smeft_set_oscillation_parameters(glb_params p, void *user_data)
// ----------------------------------------------------------------------------
// Sets the fundamental oscillation parameters and precomputes the mixing
// matrix and part of the Hamiltonian.
// ----------------------------------------------------------------------------
{
  gsl_matrix_complex *R = gsl_matrix_complex_alloc(n_leptonflavors, n_leptonflavors);
  gsl_matrix_complex *T = gsl_matrix_complex_alloc(n_leptonflavors, n_leptonflavors);
  double complex (*_R)[n_leptonflavors]
    = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(R, 0, 0);
  int i, j, k;

  // Implement correlations between density parameters. This requires that under
  // all circumstances the scaling of the matter density is performed _after_
  // calling set_oscillation_parameters! At present, this works only with
  // the hybrid minimizer (GLB_MIN_POWELL)!
//  for (j=0; j < glb_num_of_exps; j++)
//    if (density_corr[j] != j)
//      glbSetDensityParams(p, glbGetDensityParams(p, density_corr[j]), j);

  // Copy oscillation parameters
  th[1][2] = glbGetOscParams(p, GLB_THETA_12);    // Standard parameters
  th[1][3] = glbGetOscParams(p, GLB_THETA_13);
  th[2][3] = glbGetOscParams(p, GLB_THETA_23);
  delta[0] = glbGetOscParams(p, GLB_DELTA_CP);
  dmsq[0]  = glbGetOscParams(p, GLB_DM_21);
  dmsq[1]  = glbGetOscParams(p, GLB_DM_31);

  k = 6;
  for (i=4; i <= n_leptonflavors; i++)                // Mass squared differences
    dmsq[i-2] = glbGetOscParams(p, k++);

  for (i=1; i <= n_leptonflavors; i++)                // Sterile mixing angles
    for (j=MAX(i+1,4); j <= n_leptonflavors; j++)
      th[i][j] = glbGetOscParams(p, k++);

  for (i=1; i <= n_phases-1; i++)               // Sterile phases
    delta[i] = glbGetOscParams(p, k++);


  for (int x=0; x < 5; x++)                     // Production/Detection NSI: CC WEFT
  {
    for (int l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            weft.epsilon_CC[x][l][m][i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
            k += 2;
          }
        }
      }
    }
  }

  for (int x=0; x < 2; x++)           // Propagation NSI: NC WEFT
  {
    for (int f=0; f < 3; f++)
    {
      for (int i=0; i < n_leptonflavors; i++)
      {
        weft.epsilon_NC[x][f][f][i][i] = glbGetOscParams(p,k);
        k++;
        for (j=i+1; j < n_leptonflavors; j++)
        {
          weft.epsilon_NC[x][f][f][i][j] = glbGetOscParams(p,k) * cexp(I*glbGetOscParams(p,k+1));
          weft.epsilon_NC[x][f][f][j][i] = conj(weft.epsilon_NC[x][f][f][i][j]);
          k += 2;
        }
      }
    }
  }



// c3_phi_l[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c3_phi_l[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_phi_u[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c_phi_u[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_phi_d[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c_phi_d[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_phi_e[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c_phi_e[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_phi_q[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c_phi_q[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_phi_l[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c_phi_l[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c3_phi_q[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c3_phi_q[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c3_phi_e[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c3_phi_e[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c3_phi_le[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c3_phi_le[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_phi_ud[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        smeft.c_phi_ud[f][r] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
        k += 2;
    }
}

// c_ll[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_ll[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c3_lq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c3_lq[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c_lq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_lq[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c_ld[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_ld[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c_lu[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_lu[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}


// c_le[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_le[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c_ledq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_ledq[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c_lequ[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c_lequ[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c3_lequ[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c3_lequ[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c3_ledq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                smeft.c3_ledq[f][r][s][t] = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
                k += 2;
            }
        }
    }
}

// c_HWB
smeft.c_HWB = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
k += 2;

// c_HD
smeft.c_HD = glbGetOscParams(p, k) * cexp(I * glbGetOscParams(p, k + 1));
k += 2;

SMEFT_FLAG = glbGetOscParams(p, k);
k += 1;



  // Multiply rotation matrices
  gsl_matrix_complex_set_identity(U);
  for (i=0; i < n_angles; i++)
  {
    int u = rotation_order[i][0] - 1;
    int v = rotation_order[i][1] - 1;
    double complex c = cos(th[u+1][v+1]);
    double complex s = sin(th[u+1][v+1]);
    if (phase_order[i] >= 0)
      s *= cexp(-I * delta[phase_order[i]]);

    gsl_matrix_complex_set_identity(R);
    _R[u][u] = c;
    _R[v][v] = c;
    _R[u][v] = s;
    _R[v][u] = -conj(s);

//    printf("Multiplying in R[%d][%d], phase %d\n", u+1, v+1, phase_order[i]);
//    gsl_matrix_complex_fprintf(stdout, R, "%g");

    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, U, R,           // T = U.R
                   GSL_COMPLEX_ZERO, T);
    gsl_matrix_complex_memcpy(U, T);                                            // U = T
  }


  /* Calculate energy independent matrix H0 * E */
  gsl_matrix_complex_set_zero(H0_template);
  gsl_matrix_complex_set_zero(H);
  for (i=1; i < n_leptonflavors; i++)
    gsl_matrix_complex_set(H0_template, i, i, gsl_complex_rect(0.5*dmsq[i-1], 0.0));

  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, GSL_COMPLEX_ONE, H0_template, U, // T=H0.U^\dagger
                 GSL_COMPLEX_ZERO, T);
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, U, T,             // H0=U.T
                 GSL_COMPLEX_ZERO, H0_template);


  gsl_matrix_complex_free(T);
  gsl_matrix_complex_free(R);

  return 0;
}


// ----------------------------------------------------------------------------
int smeft_get_oscillation_parameters(glb_params p, void *user_data)
// ----------------------------------------------------------------------------
// Returns the current set of oscillation parameters.
// ----------------------------------------------------------------------------
{
  int i, j, k;
  int x, l, m;
  glbDefineParams(p, th[1][2], th[1][3], th[2][3], delta[0], dmsq[0], dmsq[1]);

  k = 6;
  for (i=4; i <= n_leptonflavors; i++)                // Mass squared differences
    glbSetOscParams(p, dmsq[i-2], k++);

  for (i=1; i <= n_leptonflavors; i++)                // Sterile mixing angles
    for (j=MAX(i+1,4); j <= n_leptonflavors; j++)
      glbSetOscParams(p, th[i][j], k++);

  for (i=1; i <= n_phases-1; i++)                // Sterile phases
    glbSetOscParams(p, delta[i], k++);


for (int x=0; x < 5; x++)                     // Production/Detection NSI: CC WEFT
{
  for (int l=0; l < 2; l++)
  {
    for (int m=0; m < 3; m++)
    {
      for (int i=0; i < n_leptonflavors; i++)
      {
        for (int j=0; j < n_leptonflavors; j++)
        {
          glbSetOscParams(p, cabs(weft.epsilon_CC[x][l][m][i][j]), k);
          glbSetOscParams(p, carg(weft.epsilon_CC[x][l][m][i][j]), k+1);
          k += 2;
        }
      }
    }
  }
}

for (int x=0; x < 2; x++)           // Propagation NSI: NC WEFT
{
  for (int f=0; f < 3; f++)
  {
    for (int i=0; i < n_leptonflavors; i++)
    {
        glbSetOscParams(p, weft.epsilon_NC[x][f][f][i][i], k);
      k++;
      for (j=i+1; j < n_leptonflavors; j++)
      {
        glbSetOscParams(p, cabs(weft.epsilon_NC[x][f][f][i][j]), k);
        glbSetOscParams(p, carg(weft.epsilon_NC[x][f][f][i][j]), k+1);
        k += 2;
      }
    }
  }
}


// c3_phi_l[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c3_phi_l[f][r]), k);
        glbSetOscParams(p, carg(smeft.c3_phi_l[f][r]), k + 1);
        k += 2;
    }
}

// c_phi_u[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c_phi_u[f][r]), k);
        glbSetOscParams(p, carg(smeft.c_phi_u[f][r]), k + 1);
        k += 2;
    }
}

// c_phi_d[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c_phi_d[f][r]), k);
        glbSetOscParams(p, carg(smeft.c_phi_d[f][r]), k + 1);
        k += 2;
    }
}

// c_phi_e[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c_phi_e[f][r]), k);
        glbSetOscParams(p, carg(smeft.c_phi_e[f][r]), k + 1);
        k += 2;
    }
}

// c_phi_q[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c_phi_q[f][r]), k);
        glbSetOscParams(p, carg(smeft.c_phi_q[f][r]), k + 1);
        k += 2;
    }
}

// c_phi_l[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c_phi_l[f][r]), k);
        glbSetOscParams(p, carg(smeft.c_phi_l[f][r]), k + 1);
        k += 2;
    }
}

// c3_phi_q[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c3_phi_q[f][r]), k);
        glbSetOscParams(p, carg(smeft.c3_phi_q[f][r]), k + 1);
        k += 2;
    }
}

// c3_phi_e[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c3_phi_e[f][r]), k);
        glbSetOscParams(p, carg(smeft.c3_phi_e[f][r]), k + 1);
        k += 2;
    }
}

// c3_phi_le[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c3_phi_le[f][r]), k);
        glbSetOscParams(p, carg(smeft.c3_phi_le[f][r]), k + 1);
        k += 2;
    }
}

// c_phi_ud[3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        glbSetOscParams(p, cabs(smeft.c_phi_ud[f][r]), k);
        glbSetOscParams(p, carg(smeft.c_phi_ud[f][r]), k + 1);
        k += 2;
    }
}

// c_ll[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_ll[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_ll[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c3_lq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c3_lq[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c3_lq[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_lq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_lq[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_lq[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_ld[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_ld[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_ld[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_lu[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_lu[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_lu[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_le[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_le[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_le[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_ledq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_ledq[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_ledq[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_lequ[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c_lequ[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c_lequ[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c3_lequ[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c3_lequ[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c3_lequ[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c3_ledq[3][3][3][3]
for (int f = 0; f < 3; f++) {
    for (int r = 0; r < 3; r++) {
        for (int s = 0; s < 3; s++) {
            for (int t = 0; t < 3; t++) {
                glbSetOscParams(p, cabs(smeft.c3_ledq[f][r][s][t]), k);
                glbSetOscParams(p, carg(smeft.c3_ledq[f][r][s][t]), k + 1);
                k += 2;
            }
        }
    }
}

// c_HWB
glbSetOscParams(p, cabs(smeft.c_HWB), k);
glbSetOscParams(p, carg(smeft.c_HWB), k + 1);
k += 2;

// c_HD
glbSetOscParams(p, cabs(smeft.c_HD), k);
glbSetOscParams(p, carg(smeft.c_HD), k + 1);
k += 2;

glbSetOscParams(p, SMEFT_FLAG, k);
k += 1;
  return 0;
}


// ----------------------------------------------------------------------------
int smeft_hamiltonian_cd(double E, double rho, int cp_sign)
// ----------------------------------------------------------------------------
// Calculates the Hamiltonian for neutrinos (cp_sign=1) or antineutrinos
// (cp_sign=-1) with energy E, propagating in matter of density rho
// (> 0 even for antineutrinos) and stores the result in H.
// ----------------------------------------------------------------------------
{
  double inv_E = 1.0 / E;
  double Ve = cp_sign * rho * (GLB_V_FACTOR * GLB_Ne_MANTLE); // Matter potential
  double Vn = cp_sign * rho * (GLB_V_FACTOR * (1.0 - GLB_Ne_MANTLE) / 2.0);
 
  double Ne = 1;
  double Nu = 3 * Ne; // The number 3 is for the u/d number density on the
  double Nd = 3 * Ne; // earth. It should change to 2 for the sun.
  double Nf[3]={Ne, Nu, Nd};
  

  double complex (*_H)[n_leptonflavors]
    = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(H, 0, 0);
  double complex (*_H0_template)[n_leptonflavors]
    = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(H0_template, 0, 0);

  int i, j;

  if (cp_sign > 0)
  {
    for (int i = 0; i < n_leptonflavors; ++i) {
      for (int j = 0; j < n_leptonflavors; ++j) {
          double complex HamResult = 0;
         
         HamResult = _H0_template[i][j] * inv_E;

          for (int f=0; f < 3; f++) {
              for (int x=0; x < 2; x++) {
                  // Compute the value directly inside the nested loops
                  HamResult +=  Ve*Nf[f] * weft.epsilon_NC[x][f][f][i][j];
                 
              }
          }

          _H[i][j] = HamResult;
      }
  }

  }
  else
  {
    for (int i = 0; i < n_leptonflavors; ++i) {
      for (int j = 0; j < n_leptonflavors; ++j) {
          double complex HamResult = 0;
          
          HamResult = conj(_H0_template[i][j] * inv_E);

          for (int f=0; f < 3; f++) {
              for (int x=0; x < 2; x++) {
                  HamResult += conj(Ve*Nf[f] * weft.epsilon_NC[x][f][f][i][j]);
                  
                
              }
          }

          // Assign the computed value to the matrix element
          _H[i][j] = HamResult;
      }
  }
                                                // delta_CP -> -delta_CP
  }

  // Add standard matter potential \sqrt{2} G_F (N_e - N_n/2) for \nu_e and
  // - \sqrt{2} G_F N_n / 2 for \nu_\mu and \nu_\tau
  _H[0][0] = _H[0][0] + Ve - Vn;
  _H[1][1] = _H[1][1] - Vn;
  _H[2][2] = _H[2][2] - Vn;


  return 0;
}


// ----------------------------------------------------------------------------
int smeft_S_matrix_cd(double E, double L, double rho, int cp_sign)
// ----------------------------------------------------------------------------
// Calculates the S matrix for neutrino oscillations in matter of constant
// density.
// ----------------------------------------------------------------------------
// Parameters:
//   E: Neutrino energy
//   L: Baseline
//   rho: Matter density (must be > 0 even for antineutrinos)
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_S)[n_leptonflavors] =(double complex (*)[n_leptonflavors])gsl_matrix_complex_ptr(S,0,0);
  double complex (*_Q)[n_leptonflavors] =(double complex (*)[n_leptonflavors])gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_leptonflavors]=(double complex (*)[n_leptonflavors])gsl_matrix_complex_ptr(T0,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k;

  if (fabs(rho) < RHO_THRESHOLD)                   // Vacuum
  {
    // Use vacuum mixing angles and masses
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_leptonflavors; i++)
      _lambda[i] = dmsq[i-1] * inv_E;

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      double complex (*_U)[n_leptonflavors]
        = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(U,0,0);
      for (i=0; i < n_leptonflavors; i++)
        for (j=0; j < n_leptonflavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }
  else                                             // Matter
  {
    // Calculate neutrino Hamiltonian
    if ((status=smeft_hamiltonian_cd(E, rho, cp_sign)) != 0)
      return status;

    // Calculate eigenvalues of Hamiltonian
    if (n_leptonflavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }

  // Calculate S-Matrix in mass basis in matter ...
  double phase;
  gsl_matrix_complex_set_zero(S);
  for (i=0; i < n_leptonflavors; i++)
  {
    phase    = -L * _lambda[i];
    _S[i][i] = (cos(phase) + I*sin(phase));
  }

  gsl_matrix_complex_set_identity(S1);                                 // S1 = 1

// Note that we are NOT going to flavor basis here! For the EFT probabilities we
// need the matter effects to be in the matter-mass basis. We go to flavor basis
// later by adding the production and detection coefficients.

  return 0;
}
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
double glb_eft_get_flux_coeff(int X, int Y, int alpha, double E, const glb_flux *flux);
double glb_eft_get_xsec_coeff(int X, int Y, int alpha,int cp_sign, double E, const glb_xsec *xs);
double glbEFTFluxCoeff(int experiment, int flux_ident, int X, int Y, int alpha, double energy);
int *glbEFTFluxQuarkFlavors(int experiment, int flux_ident);
double glbEFTXSecCoeff(int experiment, int xsec_ident, int X, int Y, int alpha,int cp_sign, double energy);
int *glbEFTXSecQuarkFlavors(int experiment, int xsec_ident);




// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
int smeft_filtered_probability_matrix_cd(double P[MAX_FLAVORS][MAX_FLAVORS],
        double E, double L, double rho, double sigma, int cp_sign, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the probability matrix for neutrino oscillations in matter
// of constant density, including a low pass filter to suppress aliasing
// due to very fast oscillations.
// ----------------------------------------------------------------------------
// Parameters:
//   P: Storage buffer for the probability matrix
//   E: Neutrino energy
//   L: Baseline
//   rho: Matter density (must be > 0 even for antineutrinos)
//   sigma: Width of Gaussian filter
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
// ----------------------------------------------------------------------------
{
  // Introduce some abbreviations
  double complex (*_Q)[n_leptonflavors]  = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(Q,0,0);
  double complex (*_T0)[n_leptonflavors] = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(T0,0,0);
  double complex (*_Q1)[n_leptonflavors] = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(Q1,0,0);
  double complex (*_Q2)[n_leptonflavors] = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(Q2,0,0);
  double *_lambda = gsl_vector_ptr(lambda,0);
  int status;
  int i, j, k, l, x, y, r, s, m;
  struct indices* structPtr = (struct indices*)user_data;
  
  int flux_id;
  int xsec_id;
  
  flux_id=structPtr->flux_id;
  xsec_id=structPtr->cross_id;
  
  
  // Vacuum: Use vacuum mixing angles and masses
  if (fabs(rho) < RHO_THRESHOLD)
  {
    double inv_E = 0.5/E;
    _lambda[0] = 0.0;
    for (i=1; i < n_leptonflavors; i++)
      _lambda[i] = dmsq[i-1] * inv_E;

    if (cp_sign > 0)
      gsl_matrix_complex_memcpy(Q, U);
    else
    {
      double complex (*_U)[n_leptonflavors]
        = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(U,0,0);
      for (i=0; i < n_leptonflavors; i++)
        for (j=0; j < n_leptonflavors; j++)
          _Q[i][j] = conj(_U[i][j]);
    }
  }

  // Matter: Rediagonalize Hamiltonian
  else
  {
    // Calculate neutrino Hamiltonian
    if ((status=smeft_hamiltonian_cd(E, rho, cp_sign)) != 0)
      return status;

    // Calculate eigenvalues and eigenvectors of Hamiltonian
    if (n_leptonflavors == 3)
    {
      double complex (*_H)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(H,0,0);
      double complex (*_Q)[3] = (double complex (*)[3]) gsl_matrix_complex_ptr(Q,0,0);
      double *_lambda = gsl_vector_ptr(lambda,0);
      if ((status=zheevh3(_H, _Q, _lambda)) != 0)
        return status;
    }
    else
    {
      if ((status=gsl_eigen_hermv(H, lambda, Q, w)) != GSL_SUCCESS)
        return status;
    }
  }



   

    struct glb_experiment *in;
    int experiment = glb_current_experiment;
    
    
    in = (struct glb_experiment *) glb_experiment_list[experiment];

    //int flux_ident;
    int *qFlux2;
    qFlux2  = (in->fluxes[flux_id])->q;

    
    

  //  int xsec_ident;
    int *qXsec2;
    qXsec2 = (in->xsecs[xsec_id])->q;


  double complex UTil[MAX_INTERACTIONS][2][3][MAX_FLAVORS][MAX_FLAVORS];
  // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino

  for (int x=0; x < 5; x++)
  {
    for (l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            double complex t;
            t=0.0;
            for (int k=0; k < n_leptonflavors; k++)
            {
              t+=(weft.epsilon_CC[x][l][m][i][k])*_Q[k][j] ;
            }
            UTil[x][l][m][i][j] =t;
          }
        }
      }
    }
  }


  double complex ProdLin[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
  double complex ProdQuad[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
  double complex DetLin[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
  double complex DetQuad[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
  double complex prob[MAX_FLAVORS][MAX_FLAVORS];

  for (l=0; l < 2; l++)
  {
    for (int m=0; m < 3; m++)
    {
      for (int i=0; i < n_leptonflavors; i++)
      {
        for (int j=0; j < n_leptonflavors; j++)
        {
          for (int k=0; k < n_leptonflavors; k++)
          {
             double complex t;
             t=0.0;
             for (int x=0; x < MAX_INTERACTIONS; x++)
             {
               
               t+=glb_eft_get_flux_coeff(x, 0, i, E/1.0e9, in->fluxes[flux_id])*conj(UTil[x][l][m][i][j])*_Q[i][k];
             }
             ProdLin[l][m][i][j][k] =t;
          }
        }
      }
    }
  }



  for (l=0; l < 2; l++)
  {
    for (int m=0; m < 3; m++)
    {
      for (int i=0; i < n_leptonflavors; i++)
      {
        for (int j=0; j < n_leptonflavors; j++)
        {
          for (int k=0; k < n_leptonflavors; k++)
            {
              double complex t;
              t=0.0;
              for (int x=0; x < MAX_INTERACTIONS; x++)
              for (int y=0; y < x+1; y++)
              {
                
                 t+=glb_eft_get_flux_coeff(x, y, i, E/1.0e9, in->fluxes[flux_id])*conj(UTil[x][l][m][i][j])*UTil[y][l][m][i][k];
              }
              ProdQuad[l][m][i][j][k] =t;
            }
        }
      }
    }
  }


  for (l=0; l < 2; l++)
  {
    for (int m=0; m < 3; m++)
    {
      for (int i=0; i < n_leptonflavors; i++)
      {
        for (int j=0; j < n_leptonflavors; j++)
        {
          for (int k=0; k < n_leptonflavors; k++)
          {
            double complex t;
            t=0.0;
            for (int x=0; x < MAX_INTERACTIONS; x++)
            {
              
              t+=glb_eft_get_xsec_coeff(x, 0, i,cp_sign, E/1.0e9, in->xsecs[xsec_id])*UTil[x][l][m][i][j]*conj(_Q[i][k]);
            }
            DetLin[l][m][i][j][k] =t;
          }
        }
      }
    }
  }


  for (l=0; l < 2; l++)
  {
    for (int m=0; m < 3; m++)
    {
      for (int i=0; i < n_leptonflavors; i++)
      {
        for (int j=0; j < n_leptonflavors; j++)
        {
          for (int k=0; k < n_leptonflavors; k++)
          {
            double complex t;
            t=0.0;
            for (int x=0; x < MAX_INTERACTIONS; x++)
            for (int y=0; y < x+1; y++)
            {
              
              t+=glb_eft_get_xsec_coeff(x, y, i,cp_sign, E/1.0e9, in->xsecs[xsec_id])*UTil[x][l][m][i][j]*conj(UTil[y][l][m][i][k]);
            }
            DetQuad[l][m][i][j][k] =t;
          }
        }
      }
    }
  }



  // Calculate probability matrix (see GLoBES manual for a discussion of the algorithm)
  double phase, filter_factor;
  double t = -0.5/1.0e-18 * SQR(sigma) / SQR(E);
  gsl_matrix_complex_set_zero(T0);
  for (i=0; i < n_leptonflavors; i++)
  {
    for (j=0; j < n_leptonflavors; j++)
    {
      phase         = -L * (_lambda[i] - _lambda[j]);
      filter_factor = exp(t * SQR(phase));
      _T0[i][j]     = filter_factor * (cos(phase) + I*sin(phase));
    }
  }
  double m_pi = 0.13957;
  double m_mu = 0.10566;
  double m_k= 0.493677;
  double m_u= 2.16e-3;
  double m_d = 4.7e-3;
  double Delta_kpi= 2*creal(weft.epsilon_CC[0][0][1][1][1]-weft.epsilon_CC[1][0][1][1][1]-weft.epsilon_CC[0][0][0][1][1]+weft.epsilon_CC[1][0][0][1][1])-2/(m_mu)*(pow(m_k,2)*creal(weft.epsilon_CC[3][0][1][1][1])/(m_u+m_d)-pow(m_pi,2)*creal(weft.epsilon_CC[3][0][0][1][1])/(m_u+m_d));
  
  double deltaV_CKM[3][3];
  double V_CKM[3][3];
  
  V_CKM[0][0] = 0.97428;
  V_CKM[0][1] = 0.225;
  V_CKM[0][2] = 0.00189;
  V_CKM[1][0] = -0.225;
  V_CKM[1][1] = 0.97340;
  V_CKM[1][2] = 0.0421;
  V_CKM[2][0] = 0.00764;
  V_CKM[2][1] = -0.0414;
  V_CKM[2][2] = 0.999;
  
  double delta_lambda = 0.107*Delta_kpi;
  double lambda_hat =0.225;
  
  deltaV_CKM[0][0] = -delta_lambda*(lambda_hat+0.5*pow(lambda_hat,3));
  deltaV_CKM[0][1] = delta_lambda;
  deltaV_CKM[0][2] = 0;
  deltaV_CKM[1][0] = -delta_lambda;
  deltaV_CKM[1][1] = deltaV_CKM[0][0];
  deltaV_CKM[1][2] = 0;
  deltaV_CKM[2][0] = 0;
  deltaV_CKM[2][1] = 0;
  deltaV_CKM[2][2] = 0;
  
  
  double cf=1.0;


  for (k = 0; k < n_leptonflavors; k++) {
      for (l = 0; l < n_leptonflavors; l++) {
          prob[k][l] = 0.0;

          for (i = 0; i < n_leptonflavors; i++) {
              for (j = 0; j < n_leptonflavors; j++) {

                  int s1, s2, s3, s4;
                  s1 = qFlux2[0];
                  s2 = qFlux2[1];
                  s3 = qXsec2[0];
                  s4 = qXsec2[1];
                  
                  
                 
                  
                  
                 
            

                  
                  cf = cabs(1/(1-2*deltaV_CKM[s1][s2]/V_CKM[s1][s2]));
                  
                  prob[k][l] += cf*(_T0[i][j])
                              * (conj(_Q[k][i]) * _Q[k][j]
                                 + ProdLin[s1][s2][k][i][j]
                                 + conj(ProdLin[s1][s2][k][j][i])
                                 + ProdQuad[s1][s2][k][i][j]
                               )
                              * (_Q[l][i] * conj(_Q[l][j])
                                 + DetLin[s3][s4][l][i][j]
                                 + conj(DetLin[s3][s4][l][j][i])
                                 + DetQuad[s3][s4][l][i][j]
                               );
              }
          }


          P[k][l] = creal(prob[k][l]);
      }
  }



  return 0;
}


// ----------------------------------------------------------------------------
int smeft_probability_matrix(double _P[3][3], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix for use by GLoBES.
// The problem is that GLoBES expects P to be a 3x3 matrix, so we compute the
// full matrix and then extract the upper left 3x3 submatrix.
// ----------------------------------------------------------------------------
{
  double P[MAX_FLAVORS][MAX_FLAVORS];
  int status;
  int i, j;

  status = smeft_probability_matrix_all(P, cp_sign, E, psteps, length, density,
                                      filter_sigma, user_data);
  for (i=0; i < 3; i++)
  {
    for (j=0; j < 3; j++)
    {
      _P[j][i] = P[j][i];
    }
  }


  return status;
}


// ----------------------------------------------------------------------------
int smeft_probability_matrix_all(double P[MAX_FLAVORS][MAX_FLAVORS], int cp_sign, double E,
    int psteps, const double *length, const double *density,
    double filter_sigma, void *user_data)
// ----------------------------------------------------------------------------
// Calculates the neutrino oscillation probability matrix.
// ----------------------------------------------------------------------------
// Parameters:
//   P:       Buffer for the storage of the matrix
//   cp_sign: +1 for neutrinos, -1 for antineutrinos
//   E:       Neutrino energy (in GeV)
//   psteps:  Number of layers in the matter density profile
//   length:  Lengths of the layers in the matter density profile in km
//   density: The matter densities in g/cm^3
//   filter_sigma: Width of low-pass filter or <0 for no filter
//   user_data: Unused here, should be NULL
// ----------------------------------------------------------------------------
{
  int status;
  int i, j;
  int k, l, x, y, r, s;
  
  struct indices* structPtr = (struct indices*)user_data;
  //int flux_id=structPtr->flux_id;
  //int xsec_id=structPtr->cross_id;
  int flux_id;
  int xsec_id;
  
  flux_id=structPtr->flux_id;
  xsec_id=structPtr->cross_id;
  
  
  
  
    //it will only run once)
    
  
    if(SMEFT_FLAG > 0.5){
    
    //printf("You are running in SMEFT mode \n");
    
   if(has_run < 0.5){
    running_filling(&running_smeft, &running_weft_mb, &running_weft_low_scale);
    
    has_run = 1;
    }
    
    
    Running_and_matching(running_smeft, running_weft_mb, running_weft_low_scale, smeft, &weft);
    }else{
    
    if(has_run < 0.5){
     //printf("You are running in WEFT mode \n");
    }
    
    
    
    
    }
    



  // Convert energy to eV
  E *= 1.0e9;
  

  if (filter_sigma > 0.0)                     // With low-pass filter
  {
    if (psteps == 1)
      smeft_filtered_probability_matrix_cd(P, E, GLB_KM_TO_EV(length[0]),
                                         density[0], filter_sigma, cp_sign, user_data);
    else
      return -1;
  }
  else                                        // Without low-pass filter
  {
    if (psteps > 1)
    {
      gsl_matrix_complex_set_identity(S1);                                 // S1 = 1
      for (i=0; i < psteps; i++)
      {
        status = smeft_S_matrix_cd(E, GLB_KM_TO_EV(length[i]), density[i], cp_sign);
        if (status != 0)
          return status;
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, S, S1, // T0 = S.S1
                       GSL_COMPLEX_ZERO, T0);
        gsl_matrix_complex_memcpy(S1, T0);                                 // S1 = T0
      }
      gsl_matrix_complex_memcpy(S, S1);                                    // S = S1
    }
    else
    {
      status = smeft_S_matrix_cd(E, GLB_KM_TO_EV(length[0]), density[0], cp_sign);
      if (status != 0)
        return status;
    }

    double complex (*_S)[n_leptonflavors]
      = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(S,0,0);

     double complex (*_Q)[n_leptonflavors]  = (double complex (*)[n_leptonflavors]) gsl_matrix_complex_ptr(Q,0,0);


     

    struct glb_experiment *in;
    int experiment = glb_current_experiment;
    
    
    in = (struct glb_experiment *) glb_experiment_list[experiment];

    //int flux_ident;
    int *qFlux2;
    qFlux2  = (in->fluxes[flux_id])->q;

    //double coeff = glb_eft_get_flux_coeff(3, 3, 0, 2, in->fluxes[flux_id]);
    //printf("coeff = %f \n",coeff);

  //  int xsec_ident;
    int *qXsec2;
    qXsec2 = (in->xsecs[xsec_id])->q;



  double complex UTil[MAX_INTERACTIONS][2][3][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino



  for (int x=0; x < 5; x++)
  {
    for (l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            double complex t;
            t=0.0;
            for (int k=0; k < n_leptonflavors; k++)
            {
              t+=(weft.epsilon_CC[x][l][m][i][k])*_Q[k][j] ;
            }
            UTil[x][l][m][i][j] =t;
          }
        }
      }
    }
  }

    double complex ProdLin[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
    double complex ProdQuad[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
    double complex DetLin[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
    double complex DetQuad[2][3][MAX_FLAVORS][MAX_FLAVORS][MAX_FLAVORS]; // Interaction index("L", "R", "S", "P", "T"),up-like index, down-like index,  charged lepton, neutrino
    double complex prob[MAX_FLAVORS][MAX_FLAVORS];


    for (l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            for (int k=0; k < n_leptonflavors; k++)
            {
               double complex t;
               t=0.0;
               for (int x=0; x < MAX_INTERACTIONS; x++)
               {
                
                 t+=0*glb_eft_get_flux_coeff(x, 0, i, E/1.0e9, in->fluxes[flux_id])*conj(UTil[x][l][m][i][j])*_Q[i][k];
               }
               ProdLin[l][m][i][j][k] =t;
            }
          }
        }
      }
    }


    for (l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            for (int k=0; k < n_leptonflavors; k++)
              {
                double complex t;
                t=0.0;
                for (int x=0; x < MAX_INTERACTIONS; x++)
                for (int y=0; y < x+1; y++)
                {
                  
                  t+=glb_eft_get_flux_coeff(x, y, i, E/1.0e9, in->fluxes[flux_id])*conj(UTil[x][l][m][i][j])*UTil[y][l][m][i][k];
                  
                }
                ProdQuad[l][m][i][j][k] =t;
              }
          }
        }
      }
    }


    for (l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            for (int k=0; k < n_leptonflavors; k++)
            {
              double complex t;
              t=0.0;
              for (int x=0; x < MAX_INTERACTIONS; x++)
              {
                
                t+=glb_eft_get_xsec_coeff(x, 0, i,cp_sign, E/1.0e9, in->xsecs[xsec_id])*UTil[x][l][m][i][j]*conj(_Q[i][k]);
                
              }
              DetLin[l][m][i][j][k] =t;
            }
          }
        }
      }
    }


    for (l=0; l < 2; l++)
    {
      for (int m=0; m < 3; m++)
      {
        for (int i=0; i < n_leptonflavors; i++)
        {
          for (int j=0; j < n_leptonflavors; j++)
          {
            for (int k=0; k < n_leptonflavors; k++)
            {
              double complex t;
              t=0.0;
              for (int x=0; x < MAX_INTERACTIONS; x++)
              for (int y=0; y < x+1; y++)
              {
                
                t+=glb_eft_get_xsec_coeff(x, y, i,cp_sign, E/1.0e9, in->xsecs[xsec_id])*UTil[x][l][m][i][j]*conj(UTil[y][l][m][i][k]);
                
               
                
                
              }
              DetQuad[l][m][i][j][k] =t;
              
            }
          }
        }
      }
    }
    
    double m_pi = 0.13957;
  double m_mu = 0.10566;
  double m_k= 0.493677;
  double m_u= 2.16e-3;
  double m_d = 4.7e-3;
  double Delta_kpi= 2*creal(weft.epsilon_CC[0][0][1][1][1]-weft.epsilon_CC[1][0][1][1][1]-weft.epsilon_CC[0][0][0][1][1]+weft.epsilon_CC[1][0][0][1][1])-2/(m_mu)*(pow(m_k,2)*creal(weft.epsilon_CC[3][0][1][1][1])/(m_u+m_d)-pow(m_pi,2)*creal(weft.epsilon_CC[3][0][0][1][1])/(m_u+m_d));
  
  double deltaV_CKM[3][3];
  double V_CKM[3][3];
  
  V_CKM[0][0] = 0.97428;
  V_CKM[0][1] = 0.225;
  V_CKM[0][2] = 0.00189;
  V_CKM[1][0] = -0.225;
  V_CKM[1][1] = 0.97340;
  V_CKM[1][2] = 0.0421;
  V_CKM[2][0] = 0.00764;
  V_CKM[2][1] = -0.0414;
  V_CKM[2][2] = 0.999;
  
  
  double delta_lambda = 0.107*Delta_kpi;
  double lambda_hat =0.225;
  
  deltaV_CKM[0][0] = -delta_lambda*(lambda_hat+0.5*pow(lambda_hat,3));
  deltaV_CKM[0][1] = delta_lambda;
  deltaV_CKM[0][2] = 0;
  deltaV_CKM[1][0] = -delta_lambda;
  deltaV_CKM[1][1] = deltaV_CKM[0][0];
  deltaV_CKM[1][2] = 0;
  deltaV_CKM[2][0] = 0;
  deltaV_CKM[2][1] = 0;
  deltaV_CKM[2][2] = 0;
  
  double cf=1.0;


    for (k=0; k < n_leptonflavors; k++)
    {
      for (l=0; l < n_leptonflavors; l++)
      {
        prob[k][l] = 0.0;
        for (i=0; i < n_leptonflavors; i++)
        {
          for (j=0; j < n_leptonflavors; j++)
          {
            int s1, s2, s3, s4;
            s1 = qFlux2[0];
            s2 = qFlux2[1];
            s3 = qXsec2[0];
            s4 = qXsec2[1];
            
             
           cf = cabs(1/(1-2*deltaV_CKM[s1][s2]/(V_CKM[s1][s2])));
          
            prob[k][l]+= cf*(_S[i][i] * conj(_S[j][j]))
                        *(conj(_Q[k][i]) * _Q[k][j]
                        +ProdLin[s1][s2][k][i][j]
                        +conj(ProdLin[s1][s2][k][j][i])
                        +ProdQuad[s1][s2][k][i][j]
                        )
                        *(_Q[l][i] * conj(_Q[l][j])
                        +DetLin[s3][s4][l][i][j]
                        +conj(DetLin[s3][s4][l][j][i])
                        +DetQuad[s3][s4][l][i][j]
                         );
                         
          }
        }
        P[k][l] = creal(prob[k][l]);
      }
    }

}
  return 0;
}
