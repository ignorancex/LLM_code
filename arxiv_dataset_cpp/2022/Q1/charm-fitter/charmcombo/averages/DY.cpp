/**
 * Charm Combination
 * Author: tommaso.pajero@cern.ch
 * Date: October 2022
 *
 * Code to perform the average of DeltaY (a.k.a. -A_Gamma) measurements
 *performed to date. Values are expressed in units of 10^-4.
 *
 * The code is based on the BLUE package (https://blue.hepforge.org/).
 * Run it from a root interactive session through:
 *
 *    gSystem->Load("libBlue.so")
 *    .L DY.cpp
 *    DY(<n-combination>)
 *
 * The scale factor to account for dilution in the mu-tagged measurement by LHCb (Run2_mu) is taken from
 * LHCb-ANA-2019-021, and gets contributions from:
 *  - Fig. 11, where the average from the deviation from unity of the slope is -0.0554
 *  - Fig. 12, where the average from the deviation from unity of the slope is -0.0521
 * The scale factor is thus equal to 1 / (1 - 0.0554) / (1 - 0.0521) = 1.117.
 **/

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Blue.h"

void print_header(const std::string avg_title) {
  const int line_length = 80;
  std::cout << std::setw(line_length) << std::setfill('-') << "\n"
            << avg_title << "\n"
            << std::setw(line_length) << std::setfill('-') << "\n";
}

/**
 * Perform the combination of a given set of measurements.
 *
 * @param flag Steers which set of measurements should be employed (according to the values of "combinations" below).
 */
void DY(int flag = 0) {

  const std::map<int, std::pair<std::string, std::vector<int>>> combinations = {
      {0, {"LHCb average 2024", {4, 5, 6, 7, 8, 9, 10, 11, 12}}},
      {1, {"LHCb average 2024 (K+ K-)", {4, 6, 8, 10}}},
      {2, {"LHCb average 2024 (pi+ pi-)", {5, 7, 9, 11}}},
      {100, {"World average 2024", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
      {101, {"World average 2024 (K+ K-)", {1, 4, 6, 8, 10}}},
      {102, {"World average 2024 (pi+ pi-)", {2, 5, 7, 9, 11}}},
  };

  // number of estimates, uncertainties, observables
  const std::vector<TString> names = {
      "BaBar         ",  //  0
      "CDF_KK        ",  //  1
      "CDF_PP        ",  //  2
      "Belle         ",  //  3
      "Run1_mu_KK    ",  //  4
      "Run1_mu_PP    ",  //  5
      "Run1_prompt_KK",  //  6
      "Run1_prompt_PP",  //  7
      "Run2_mu_KK    ",  //  8
      "Run2_mu_PP    ",  //  9
      "Run2_prompt_KK",  // 10
      "Run2_prompt_PP",  // 11
      "Run2_pipipi0",    // 12
  };
  const std::vector<TString> names_obs = {"     DY"};
  const auto num_est = names.size();

  // Array of estimates and uncertainties
  const std::vector<TString> names_unc = {"  Stat", "m(hhh)", "   Sec", " m(KK)", " m(PP)",
                                          "Weight", "TimRes", "Mistag", "Run1Mu", " Other"};

  // Inverse of Fp_pipipi0 = 0.942554 +/- 0.00404575
  constexpr auto pipipi0_corr = 1.061;

  const std::vector<double> x_est = {
      // clang-format off
      //                     0                    1      2     3     4     5      6       7      8      9
      // Val                Stat                  m(hhh) Sec   m(KK) m(PP) Weight TimeRes Mistag Run1Mu Other
      -8.8                , 25.5                , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  5.8                ,  // BaBar
      19.                 , 15.                 , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  0.                 ,  // CDF KK
       1.                 , 18.                 , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  0.                 ,  // CDF PP
       3.                 , 20.                 , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  7.                 ,  // Belle
      13.40               ,  7.70               , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.49,   0.  ,  2.55,  0.                 ,  // Run 1 mu KK
       9.2                , 14.5                , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.42,   0.  ,  2.48,  0.                 ,  // Run 1 mu PP
       3.0                ,  3.2                , 0.1 ,  0.8 , 0.5 , 0.  , 0.2 ,  0.  ,   0.  ,  0.  ,  0.                 ,  // Run 1 prompt KK
      -4.6                ,  5.8                , 0.1 ,  1.2 , 0.  , 0.  , 0.2 ,  0.  ,   0.  ,  0.  ,  0.                 ,  // Run 1 prompt PP
   //  4.3                ,  3.6                , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.3 ,   0.3 ,  0.  ,  0.3                ,  // Run 2 mu KK
   // -2.2                ,  7.0                , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.4 ,   0.6 ,  0.  ,  0.3                ,  // Run 2 mu PP
       4.8                ,  4.0                , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  0.3                ,  // Run 2 mu KK  (scale factor of 1.12 to account for dilution)
      -2.5                ,  7.8                , 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  0.3                ,  // Run 2 mu PP  (scale factor of 1.12 to account for dilution)
      -2.321              ,  1.524              , 0.24,  0.13, 0.06, 0.  , 0.05,  0.  ,   0.  ,  0.  ,  0.                 ,  // Run 2 prompt KK
      -4.014              ,  2.814              , 0.34,  0.13, 0.  , 0.03, 0.05,  0.  ,   0.  ,  0.  ,  0.                 ,  // Run 2 prompt PP
      -1.21 * pipipi0_corr,  5.97 * pipipi0_corr, 0.  ,  0.  , 0.  , 0.  , 0.  ,  0.  ,   0.  ,  0.  ,  2.00 * pipipi0_corr,  // Run 2 pi pi pi0 (remove sys. unc. from time binning)
      // clang-format on
  };
  const auto num_unc = names_unc.size();
  if (x_est.size() != num_est * (num_unc + 1)) {
    std::cout << "The size of the uncertainty matrix is inconsistent with the number of estimates and the number of "
                 "uncertainties per estimate"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto inp_est = std::make_unique<const TMatrixD>(num_est, num_unc + 1, &x_est[0]);

  // Statistical uncertainties are not correlated
  // All systematic uncertainties except "Other" are 100% correlated
  const std::vector<double> rho_val = {0., 1., 1., 1., 1., 1., 1., 1., 1., 0.};
  if (rho_val.size() == num_unc + 1) {
    std::cout << "The size of correlation vector among uncertainties is inconsistent with the number of uncertainties"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Statistical precision in systematic uncertainties
  const std::vector<double> s_unc(num_est * num_unc, 0.);
  auto inp_sta = std::make_unique<const TMatrixD>(num_est, num_unc, &s_unc[0]);

  // Format for the output
  const TString for_val = "%2.2f";
  const TString for_unc = "%2.2f";
  const TString for_wei = "%2.2f";
  const TString for_rho = "%2.2f";
  const TString for_pul = "%2.2f";
  const TString for_chi = "%2.2f";
  const TString for_uni = "";

  // Initialise the combiner
  auto my_blue = std::make_unique<Blue>(num_est, num_unc);
  my_blue->SetFormat(for_val, for_unc, for_wei, for_rho, for_pul, for_chi, for_uni);
  my_blue->FillNamEst(&names[0]);
  my_blue->FillNamUnc(&names_unc[0]);
  my_blue->FillNamObs(&names_obs[0]);
  my_blue->FillEst(inp_est.get());
  my_blue->FillSta(inp_sta.get());
  for (int i = 0; i < num_unc; ++i) my_blue->FillCor(i, rho_val[i]);

  // Perform the combination
  const auto [title, inputs] = combinations.at(flag);
  print_header(title);
  for (auto i = 0; i < num_est; ++i) my_blue->SetInActiveEst(i);
  for (auto i : inputs) my_blue->SetActiveEst(i);
  my_blue->FixInp();
  my_blue->PrintEst();
  my_blue->Solve();
  my_blue->PrintResult();

  return;
}
