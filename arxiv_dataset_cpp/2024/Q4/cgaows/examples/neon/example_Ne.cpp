#include <iostream>
#include "../../src/cusped_gaussians.hpp"
#include <vector>
#include <typeinfo>
#include <iomanip> 

// Here is an example file that demonstrates how to call the CuspedGaussian class functions for the case of the Ne atom (6-31G* basis)

int main() {
 
  std::string file_path = __FILE__; 
  std::string directory_path = file_path.substr(0, file_path.find_last_of("/\\"));
  std::cout << "Current working directory: " << directory_path << std::endl;

  // initialize position of nucleus  
  std::vector<double> npos;
  npos.assign(1*3, 0.0);

  // Create example electron positions (only for 5 electrons since each spin has it's own determinant)
  //std::vector<double> epos = {-0.23912265523, 0.598452450611, -0.001932951286, 
//	        0.08019104159, -0.178974374215, 0.004223309868,  
//	        0.006472402593, 0.520026488075, 0.119271758977,  
//		0.366762342113, 0.276272924761, -0.289578120812,
//		0.074348579879, -0.337904045004, 0.277494159034};
  std::vector<double> epos = {-0.239122655230,       0.598452450611,      -0.001932951286,
      			       0.080191041590,      -0.178974374215,       0.004223309868,
      			      -0.059690768873,       0.427209898873,      -0.405137371731,
      			       0.366762342113,       0.276272924761,      -0.289578120812,
      			      -0.186605581357,      -0.566435670875,      -0.105924159454};


  // initialize matrix to store orbital evaluations
  std::vector<double> xmat;
  xmat.assign(14*5, 0.0);
  
  // make cusped orbitals evaluator object
  // will look for cusp parameter AO files in current cpp directory
  CuspedGaussians AO(directory_path); 

  // evaluate orbitals at current electron configuration and populate xmat
  AO.evaluate_orbs(epos, npos, xmat);

  // print populated xmat
  std::cout << std::endl;
  std::cout << "Populated AO matrix:" << std::endl;
  for (int i = 0 ; i < 5; i++) {
    for (int j = 0; j < 14; j++) {
      std::cout << std::setw(10) << std::setprecision(6) << std::fixed << xmat[i + 5 * j] << "  ";
    }
    std::cout << std::endl;
  } 

  // create workspace to store 1st and 2nd derivatives of the AOs wrt each electron coordinate 
  std::vector<double> m_workspace;
  const int space_needed = 6 * 5 * 14;
  m_workspace.assign(space_needed, 0.0);

  // get pointers to where we will put derivatives
  double * der1[3];
  double * der2[3];
  der1[0] = &m_workspace.at(0 * 5 * 14); // x 1st derivatives of the atomic orbitals
  der1[1] = &m_workspace.at(1 * 5 * 14); // y 1st derivatives of the atomic orbitals
  der1[2] = &m_workspace.at(2 * 5 * 14); // z 1st derivatives of the atomic orbitals
  der2[0] = &m_workspace.at(3 * 5 * 14); // x 2nd derivatives of the atomic orbitals
  der2[1] = &m_workspace.at(4 * 5 * 14); // y 2nd derivatives of the atomic orbitals
  der2[2] = &m_workspace.at(5 * 5 * 14); // z 2nd derivatives of the atomic orbitals

  // populate workspace with derivatives
  AO.evaluate_derivs(epos, npos, der1, der2);

  // print workspace
  std::cout << std::endl;
  std::cout << "Workspace:" << std::endl;
  for (int k = 0; k < 6; k++) {
    for (int i = 0 ; i < 5; i++) {
      for (int j = 0; j < 14; j++) {
        std::cout << std::setw(10) << std::setprecision(6) << std::fixed << m_workspace[ k * 5 * 14 + i + 5 * j] << "  ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  } 

  return 0;
}
