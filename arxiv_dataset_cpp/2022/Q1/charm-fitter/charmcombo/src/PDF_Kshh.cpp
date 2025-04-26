/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <vector>

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_Kshh.h>
#include <ParametersCharmCombo.h>

PDF_Kshh::PDF_Kshh(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{4}, th_cfg{th_cfg} {
  name = "Kshh_" + measurement_id;
  initParameters();
  initRelations();

  TString label = measurement_id;
  if (measurement_id == "Belle") label = "Belle #it{K}_{S}^{0}#it{#pi^{+}#pi^{#minus}}";

  initObservables(label);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_Kshh::~PDF_Kshh() {}

void PDF_Kshh::initParameters() {
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
    std::cout << "PDF_Kshh::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_Kshh::initRelations() {
  theory = new RooArgList("theory");  ///< the order of this list must match that of the COR matrix!
  switch (th_cfg) {
  case theory_config::phenomenological:
    theory->add(*(Utils::makeTheoryVar("x_th", "x_th", "x", parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th", "y", parameters)));
    theory->add(*(Utils::makeTheoryVar("qop_th", "qop_th", "qop+1", parameters)));
    theory->add(*(Utils::makeTheoryVar("phi_th", "phi_th", "phi", parameters)));
    break;
  case theory_config::theoretical:
    theory->add(*(Utils::makeTheoryVar("x_th", "x_th",
                                       "pow(2,-0.5) * pow( "
                                       "    pow(x12,2) - pow(y12,2) + pow( "
                                       "       + pow(pow(x12,2) + pow(y12,2),2)"
                                       "       - pow(2 * x12 * y12 * sin(phiM - phiG),2),"
                                       "    0.5)"
                                       ",0.5) * TMath::Sign(1., cos(phiM - phiG))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th",
                                       "pow(2,-0.5) * pow( "
                                       "    pow(y12,2) - pow(x12,2) + pow( "
                                       "       + pow(pow(x12,2) + pow(y12,2),2)"
                                       "       - pow(2 * x12 * y12 * sin(phiM - phiG),2),"
                                       "    0.5)"
                                       ",0.5)",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("qop_th", "qop_th",
                                       "pow(  (pow(x12,2) + pow(y12,2) + 2 * x12 * y12 * sin(phiM - phiG))"
                                       "    /  pow(  pow(pow(x12,2) + pow(y12,2),2)                       "
                                       "           - pow(2 * x12 * y12 * sin(phiM - phiG),2), 0.5), 0.5)  ",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("phi_th", "phi_th",
                                       "-0.5 * TMath::ATan("
                                       "      (pow(x12,2) * sin(2*phiM) + pow(y12,2) * sin(2*phiG))"
                                       "    / (pow(x12,2) * cos(2*phiM) + pow(y12,2) * cos(2*phiG)))",
                                       parameters)));
    break;
  case theory_config::superweak:
    theory->add(*(Utils::makeTheoryVar("x_th", "x_th",
                                       "pow(2,-0.5) * pow( "
                                       "    pow(x12,2) - pow(y12,2) + pow( "
                                       "       + pow(pow(x12,2) + pow(y12,2),2)"
                                       "       - pow(2 * x12 * y12 * sin(phiM),2),"
                                       "    0.5)"
                                       ",0.5) * TMath::Sign(1., cos(phiM))",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("y_th", "y_th",
                                       "pow(2,-0.5) * pow( "
                                       "    pow(y12,2) - pow(x12,2) + pow( "
                                       "       + pow(pow(x12,2) + pow(y12,2),2)"
                                       "       - pow(2 * x12 * y12 * sin(phiM),2),"
                                       "    0.5)"
                                       ",0.5)",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("qop_th", "qop_th",
                                       "pow(  (pow(x12,2) + pow(y12,2) + 2 * x12 * y12 * sin(phiM))"
                                       "    / pow(                                                      "
                                       "        pow(pow(x12,2) + pow(y12,2),2)                      "
                                       "      - pow(2 * x12 * y12 * sin(phiM),2), 0.5), 0.5)",
                                       parameters)));
    theory->add(*(Utils::makeTheoryVar("phi_th", "phi_th",
                                       "-0.5 * TMath::ATan("
                                       "       pow(x12,2) * sin(2*phiM)"
                                       "    / (pow(x12,2) * cos(2*phiM) + pow(y12,2)))",
                                       parameters)));
    break;
  default:
    std::cout << "PDF_Kshh::initRelations : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_Kshh::initObservables(const TString& setName) {
  observables = new RooArgList("observables");  ///< the order of this list must match that of the COR matrix!
  observables->add(*(new RooRealVar("x_obs", setName + "   #it{x}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("y_obs", setName + "   #it{y}", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("qop_obs", setName + "   |#it{q}/#it{p}|", 0., -1e4, 1e4)));
  observables->add(*(new RooRealVar("phi_obs", setName + "   #it{#phi}_{2}", 0., -1e4, 1e4)));
}

void PDF_Kshh::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("Belle")) {
    obsValSource = "https://inspirehep.net/literature/1289224";
    setObservable("x_obs", 0.56);
    setObservable("y_obs", 0.30);
    setObservable("qop_obs", 0.90);
    setObservable("phi_obs", Utils::DegToRad(-6.));
  } else {
    std::cout << "PDF_Kshh::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Kshh::setUncertainties(TString c) {
  if (c.EqualTo("Belle")) {
    obsErrSource = "https://inspirehep.net/literature/1289224";
    StatErr[0] = sqrt(pow(0.19, 2) + pow(0.093, 2));               // x
    StatErr[1] = sqrt(pow(0.15, 2) + pow(0.068, 2));               // y
    StatErr[2] = sqrt(pow(0.155, 2) + pow(0.071, 2));              // qop
    StatErr[3] = Utils::DegToRad(sqrt(pow(11, 2) + pow(4.6, 2)));  // phi
    SystErr[0] = 0;                                                // x
    SystErr[1] = 0;                                                // y
    SystErr[2] = 0;                                                // qop
    SystErr[3] = 0;                                                // phi
  } else {
    std::cout << "PDF_Kshh::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Kshh::setCorrelations(TString c) {
  resetCorrelations();
  if (c.EqualTo("Belle")) {
    corSource = "hflav";
    std::vector<double> dataStat = {
        // clang-format off
        1., 0.054, -0.074, -0.031,  // x
            1.,     0.034, -0.019,  // y
                    1.,     0.044,  // qop
                            1.      // phi
        // clang-format on
    };
    corStatMatrix = Utils::buildCorMatrix(nObs, dataStat);
  } else {
    std::cout << "PDF_Kshh::setCorrelations() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Kshh::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
