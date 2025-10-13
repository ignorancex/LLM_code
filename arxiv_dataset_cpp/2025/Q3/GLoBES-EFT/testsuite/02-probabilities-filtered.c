/* GLoBES -- General LOng Baseline Experiment Simulator
 * (C) 2002 - 2007,  The GLoBES Team
 *
 * GLoBES is mainly intended for academic purposes. Proper
 * credit must be given if you use GLoBES or parts of it. Please
 * read the section 'Credit' in the README file.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
 
// ============================================================================
// GLoBES Testsuite
// Part 02: Calculation of filtered oscillation probabilities
// ============================================================================

#include <iostream>
#include <iomanip>
#include <math.h>
#include <globes/globes.h>
using namespace std;

int main(int argc, char *argv[])
{ 
  // "True" oscillation parameter ranges
  const double p_ranges[6][3] = { 
    { 20.0 * M_PI/180.0, 45.0 * M_PI/180.0,  8.0 * M_PI/180.0 },              // theta-12
    { 0.0,               10.0 * M_PI/180.0,  2.5 * M_PI/180.0 },              // theta-13
    { 30.0 * M_PI/180.0, 60.0 * M_PI/180.0, 10.0 * M_PI/180.0 },              // theta-23
    { 0.0,             2*M_PI,               M_PI/4.0         },              // delta_CP
    { 6e-5,              9e-5,               1.0e-5           },              // dm21
    { 1e-3,              4e-3,               1.0e-3           } };            // dm31
  double params[6];
  long nLine = 0;
  char *GLB_FILE = argv[argc-1];

  // Initialize GLoBES
  glbInit(argv[0]);

  // Loop over all sub-tests
  glbClearExperimentList();
  glbInitExperiment(GLB_FILE, &glb_experiment_list[0], &glb_num_of_exps); 
  glb_params true_values = glbAllocParams();

  double *lengths = new double[10];
  double *densities = new double[10];
  size_t psteps;
  glbGetProfileDataInExperiment(0, &psteps, &lengths, &densities);
  delete[] lengths;
  delete[] densities;
  
  for (int j=0; j < glbGetNumOfOscParams(); j++)
    params[j] = p_ranges[j][0];
  
  int k;
  do
  {
    glbDefineParams(true_values, params[0], params[1], params[2], params[3], params[4], params[5]);
    glbSetOscillationParameters(true_values);
   
    double probs[3][3]; 
    for (double E=0.1; E < 10.0; E+=2.0)
    {
      for (int cp_sign=+1; cp_sign >= -1; cp_sign-=2)
      {
        for (int fi=1; fi <= 3; fi++)
          for (int ff=1; ff <= 3; ff++)
          {
            cout << "<" << ++nLine <<": E=" << E << ", CP=" << cp_sign << ", Osc.params = {";
            for (int k=0; k < glbGetNumOfOscParams(); k++)
              cout << params[k] << ", ";
            cout << "\b\b}, Channel " << fi << " -> " << ff << ">  ";
            probs[fi-1][ff-1] = glbFilteredConstantDensityProbability(0, fi, ff, cp_sign, E);
            cout << probs[fi-1][ff-1] << endl;
          }

        for (int k=0; k < 3; k++)
        {
          double tmp;
          tmp = fabs(1.0 - probs[k][0] - probs[k][1] - probs[k][2]);
          if (tmp > 1e-14)
            cout << "Unitarity mismatch. Badness " << tmp << endl;
          
          tmp = fabs(1.0 - probs[0][k] - probs[1][k] - probs[2][k]);
          if (tmp > 1e-14)
            cout << "Unitarity mismatch. Badness " << tmp << endl;
        }
      }
    }
    
    // Increment parameter values
    for (k=glbGetNumOfOscParams()-1; k >= 0; k--)
      if ((params[k] += p_ranges[k][2]) <= p_ranges[k][1])
        break;
      else
        params[k] = p_ranges[k][0];
  }
  while (k >= 0);

  glbFreeParams(true_values);

  return 0;
}



