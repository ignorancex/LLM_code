/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_BinFlip.h>
#include <ParametersCharmCombo.h>

PDF_BinFlip::PDF_BinFlip(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{4}, th_cfg{th_cfg} {
  name = "BinFlip_" + measurement_id;
  TString label;
  if (measurement_id.EqualTo("LHCb_Run1"))
    label = "LHCb Binflip Run 1";
  else if (measurement_id.EqualTo("LHCb_Run2_prompt"))
    label = "LHCb Binflip Run 2 (prompt)";
  else if (measurement_id.EqualTo("LHCb_Run2_sl"))
    label = "LHCb Binflip Run 2 (muon-tagged)";
  else if (measurement_id.EqualTo("LHCb_Run2"))
    label = "LHCb Binflip Run 2";
  initParameters();
  initRelations();
  initObservables(label);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_BinFlip::~PDF_BinFlip() {}

void PDF_BinFlip::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  switch (th_cfg) {
  case theory_config::phenomenological:
    parameters->add(*(p.get("x")));
    parameters->add(*(p.get("y")));
    parameters->add(*(p.get("qop")));
    parameters->add(*(p.get("phi")));
    break;
  case theory_config::theoretical:
    parameters->add(*(p.get("phiG")));
  case theory_config::superweak:
    parameters->add(*(p.get("x12")));
    parameters->add(*(p.get("y12")));
    parameters->add(*(p.get("phiM")));
    break;
  default:
    std::cout << "PDF_BinFlip::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_BinFlip::initRelations() {
  theory = new RooArgList("theory");  ///< the order of this list must match that of the COR matrix!
  switch (th_cfg) {
  case theory_config::phenomenological:
    theory->add(*(Utils::makeTheoryVar("x_th", "x_th",
                                       "0.5*(  x*cos(phi)*(qop+1 + 1/(qop+1))"
                                       "     + y*sin(phi)*(qop+1 - 1/(qop+1)))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th",
                                       "0.5*(  y*cos(phi)*(qop+1 + 1./(qop+1))"
                                       "     - x*sin(phi)*(qop+1 - 1./(qop+1)))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("dx_th", "dx_th",
                                       "0.5*(  x*cos(phi)*(qop+1 - 1./(qop+1))"
                                       "     + y*sin(phi)*(qop+1 + 1./(qop+1)))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("dy_th", "dy_th",
                                       "0.5*(  y*cos(phi)*(qop+1 - 1./(qop+1))"
                                       "     - x*sin(phi)*(qop+1 + 1./(qop+1)))",
                                       parameters)));
    break;
  case theory_config::theoretical:
    theory->add(*(Utils::makeTheoryVar("x_th", "x_th", " x12*cos(phiM)", parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th", " y12*cos(phiG)", parameters)));
    theory->add(*(Utils::makeTheoryVar("dx_th", "dx_th", "-y12*sin(phiG)", parameters)));
    theory->add(*(Utils::makeTheoryVar("dy_th", "dy_th", " x12*sin(phiM)", parameters)));
    break;
  case theory_config::superweak:
    theory->add(*(Utils::makeTheoryVar("x_th", "x_th", "x12*cos(phiM)", parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th", "y12", parameters)));
    theory->add(*(Utils::makeTheoryVar("dx_th", "dx_th", "0", parameters)));
    theory->add(*(Utils::makeTheoryVar("dy_th", "dy_th", "x12*sin(phiM)", parameters)));
    break;
  default:
    std::cout << "PDF_BinFlip::initRelations : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_BinFlip::initObservables(const TString& setName) {
  observables = new RooArgList("observables");  ///< the order of this list must match that of the COR matrix!
  observables->add(*(new RooRealVar("x_obs", setName + "   #it{x_{CP}}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("y_obs", setName + "   #it{y_{CP}}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("dx_obs", setName + "   #it{#Deltax}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("dy_obs", setName + "   #it{#Deltay}", 0., -1e4, 1e4)));
}

void PDF_BinFlip::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("LHCb_Run1")) {
    obsValSource = "https://inspirehep.net/literature/1724179";
    setObservable("x_obs", 0.27);
    setObservable("y_obs", 0.74);
    setObservable("dx_obs", -0.053);
    setObservable("dy_obs", 0.06);
  } else if (c.EqualTo("LHCb_Run2_prompt")) {
    obsValSource = "https://inspirehep.net/literature/1867376";
    setObservable("x_obs", 0.3973);
    setObservable("y_obs", 0.4589);
    setObservable("dx_obs", -0.0271);
    setObservable("dy_obs", 0.0203);
  } else if (c.EqualTo("LHCb_Run2_sl")) {
    obsValSource = "https://inspirehep.net/literature/2135966";
    setObservable("x_obs", 0.429);
    setObservable("y_obs", 1.261);
    setObservable("dx_obs", -0.077);
    setObservable("dy_obs", 0.301);
  } else if (c.EqualTo("LHCb_Run2")) {
    obsValSource = "https://inspirehep.net/literature/2135966";
    setObservable("x_obs", 0.400);
    setObservable("y_obs", 0.551);
    setObservable("dx_obs", -0.029);
    setObservable("dy_obs", 0.031);
  } else {
    std::cout << "PDF_BinFlip::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_BinFlip::setUncertainties(TString c) {
  if (c.EqualTo("LHCb_Run1")) {
    obsErrSource = "https://inspirehep.net/literature/1724179";
    StatErr[0] = 0.16;   // x
    StatErr[1] = 0.36;   // y
    StatErr[2] = 0.07;   // dx
    StatErr[3] = 0.16;   // dy
    SystErr[0] = 0.04;   // x
    SystErr[1] = 0.11;   // y
    SystErr[2] = 0.022;  // dx
    SystErr[3] = 0.03;   // dy
  } else if (c.EqualTo("LHCb_Run2_prompt")) {
    obsErrSource = "https://inspirehep.net/literature/1867376";
    StatErr[0] = pow(pow(0.0459, 2) + pow(0.029, 2), 0.5);  // x
    StatErr[1] = pow(pow(0.1198, 2) + pow(0.085, 2), 0.5);  // y
    StatErr[2] = pow(pow(0.0182, 2) + pow(0.001, 2), 0.5);  // dx
    StatErr[3] = pow(pow(0.0365, 2) + pow(0.011, 2), 0.5);  // dy
    SystErr[0] = 0.;                                        // x
    SystErr[1] = 0.;                                        // y
    SystErr[2] = 0.;                                        // dx
    SystErr[3] = 0.;                                        // dy
  } else if (c.EqualTo("LHCb_Run2_sl")) {
    obsErrSource = "https://inspirehep.net/literature/2135966";
    StatErr[0] = 0.148;  // x
    StatErr[1] = 0.312;  // y
    StatErr[2] = 0.093;  // dx
    StatErr[3] = 0.192;  // dy
    SystErr[0] = 0.026;  // x
    SystErr[1] = 0.083;  // y
    SystErr[2] = 0.028;  // dx
    SystErr[3] = 0.026;  // dy
  } else if (c.EqualTo("LHCb_Run2")) {
    obsErrSource = "https://inspirehep.net/literature/2135966";
    StatErr[0] = 0.045;   // x
    StatErr[1] = 0.116;   // y
    StatErr[2] = 0.018;   // dx
    StatErr[3] = 0.035;   // dy
    SystErr[0] = 0.0195;  // x
    SystErr[1] = 0.0594;  // y
    SystErr[2] = 0.0013;  // dx
    SystErr[3] = 0.0128;  // dy
  } else {
    std::cout << "PDF_BinFlip::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_BinFlip::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("LHCb_Run1")) {
    corSource = "https://inspirehep.net/literature/1724179";
    std::vector<double> dataStat = {
        // clang-format off
        1., -0.17, 0.04, -0.02,  // x
             1.,  -0.03,  0.01,  // y
                   1.,   -0.13,  // dx
                          1.     // dy
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
    std::vector<double> dataSyst = {
        // clang-format off
        1., 0.15,  0.01, -0.02,  // x
            1.,   -0.05, -0.03,  // y
                   1.,    0.14,  // dx
                          1.     // dy
        // clang-format on
    };
    corSystMatrix = Utils::buildCorMatrix(nObs, dataSyst);
  } else if (c.EqualTo("LHCb_Run2_prompt")) {
    corSource = "https://inspirehep.net/literature/1867376";
    std::vector<double> dataStat = {
        // clang-format off
        1., 0.111,  -0.017, -0.010,  // x
            1.,     -0.011, -0.051,  // y
                     1.,     0.077,  // dx
                             1.      // dy
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else if (c.EqualTo("LHCb_Run2_sl")) {
    corSource = "https://inspirehep.net/literature/2135966";
    std::vector<double> dataStat = {
        // clang-format off
        1., 0.085, -0.011, -0.009,  // x
            1.,    -0.001, -0.050,  // y
                    1.,     0.070,  // dx
                            1.      // dy
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
    std::vector<double> dataSyst = {
        // clang-format off
        1., 0.11,  -0.25, -0.02,  // x
            1.,    -0.05, -0.20,  // y
                    1.,    0.11,  // dx
                           1.     // dy
        // clang-format on
    };
    corSystMatrix = Utils::buildCorMatrix(nObs, dataSyst);
  } else if (c.EqualTo("LHCb_Run2")) {
    corSource = "https://inspirehep.net/literature/2135966";
    std::vector<double> dataStat = {
        // clang-format off
        1., 0.121,  -0.018, -0.016,  // x
            1.,     -0.012, -0.058,  // y
                     1.,     0.069,  // dx
                             1.      // dy
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
    std::vector<double> dataSyst = {
        // clang-format off
        1., 0.08,  0.,   -0.01,  // x
            1.,   -0.02, -0.04,  // y
                   1.,    0.33,  // dx
                          1.     // dy
        // clang-format on
    };
    corSystMatrix = Utils::buildCorMatrix(nObs, dataSyst);
  } else {
    std::cout << "PDF_BinFlip::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_BinFlip::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
