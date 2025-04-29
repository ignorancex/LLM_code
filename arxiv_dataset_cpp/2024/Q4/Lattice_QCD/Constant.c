/**
  * This file is part of Lattice_QCD.
  *
  * Lattice_QCD is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * Lattice_QCD is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with Lattice_QCD.  If not, see <https://www.gnu.org/licenses/>.
  */
/**
 *  @file Constant.c
 *  @brief The global variables and constants used throughout the code
 *  @author Bradley V Smith
 *  @version 1
 */
 #include <math.h>
/**
 *  the number of data points generated
 */
const int points = 10;
/**
 * the starting point for the data
 */
double start = 0;
/**
 *  the ending point for the data
 */
double end = 0;
/**
 *some important constants
 */
const double mlambdab = 5.61951;
const double mlambdac = 2.28646;
const double GF = 1.1663787 * pow(10, -5);  // Fermi constant
const double hbar = 6.582119514 * pow(10,-25);
const double mtau = 1.77682;
const double mmu = 0.105658;
const double PI = 3.14159265359;  // Pi
const double vbc = 0.0414;  // Cabibbo-Kobayashi-Maskawa (CKM) matrix element
const double vcb = 0.0414;
const double m1 = 5.6195;
const double m2 = 2.28646;
const double mb = 4.18;
const double mc = 1.257;
const double eps = 1/pow(10, 8);
const double epsneg = -1/pow(10,8);
/**
 * pole locations
 */
const double mb0minus = 6.276;
const double mb1minus = 6.332;
const double mb0plus = 6.725;
const double mb1plus = 6.768;
/**
 * the number of parameters for the nominal data set
 */
const int set1 = 18;
/**
 *  the number of parameters for the higher order data set
 */
const int set2 = 28;
/**
 *  the number of form factors
 */
const int numfactors = 10;
    //t0 parameter of z expansion
double tminus = 0;
double t0 = 0;
double qsqrmax = 0;

/**
 *  the names of the parameters for the nominal form factors
 */
const char *nominalparam[] = {"a0_fplus", "a1_fplus", "a0_fperp", "a1_fperp", "a0_f0", "a1_f0",
  "a0_gpp", "a1_gplus", "a1_gperp", "a0_g0", "a1_g0", "a0_hplus",
  "a1_hplus", "a0_hperp", "a1_hperp", "a0_htildepp", "a1_htildeplus",
  "a1_htildeperp"};
/**
 *  the names of the parameters for the higher order form factors
 */
const char *hoparam[] = {"a0_fplus", "a1_fplus", "a2_fplus", "a0_fperp", "a1_fperp",
  "a2_fperp", "a0_f0", "a1_f0", "a2_f0", "a0_gpp", "a1_gplus",
  "a2_gplus", "a1_gperp", "a2_gperp", "a0_g0", "a1_g0", "a2_g0",
  "a0_hplus", "a1_hplus", "a2_hplus", "a0_hperp", "a1_hperp",
  "a2_hperp", "a0_htildepp", "a1_htildeplus", "a2_htildeplus",
  "a1_htildeperp", "a2_htildeperp"};
/**
 *  the maximum length for file names
 */
int file_name_length = 300;
/**
 *  the number of parameters for the current data set
 */
int param_num = 0;
/**
 *  the precision, number of digits, to take in for the parameters
 */
int pressision = 30;
/**
 * the factors used
 */
double gs = 0;
double gp = 0;
double gL = 0;
double gR = 0;
double gT = 0;
double ml = mtau;
