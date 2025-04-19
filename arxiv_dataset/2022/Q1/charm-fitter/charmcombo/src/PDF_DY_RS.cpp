/**
 * Charm Combination
 * Author: Tommaso Pajero, tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_DY_RS.h>
#include <ParametersCharmCombo.h>

PDF_DY_RS::PDF_DY_RS(TString measurement_id, const theory_config& th_cfg) : PDF_Abs{1}, th_cfg{th_cfg} {
  name = "DY_RS_" + measurement_id;
  initParameters();
  initRelations();
  initObservables(measurement_id);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_DY_RS::~PDF_DY_RS() {}

void PDF_DY_RS::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  parameters->add(*(p.get("R_Kpi")));
  if (th_cfg != theory_config::superweak) parameters->add(*(p.get("Acp_KP")));
  parameters->add(*(p.get("Delta_Kpi")));

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
    std::cout << "PDF_DY_RS::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_DY_RS::initRelations() {
  theory = new RooArgList("theory");  ///< the order of this list must match that of the COR matrix!
  switch (th_cfg) {
  case theory_config::phenomenological:
    theory->add(
        *(Utils::makeTheoryVar("DY_RS_th", "DY_RS_th",
                               "0.5 * pow(R_Kpi/100, 0.5) * "
                               "(  (y*cos(Delta_Kpi) - x*sin(Delta_Kpi))*((qop+1) - 1/(qop+1) - Acp_KP/100)*cos(phi)"
                               " - (x*cos(Delta_Kpi) + y*sin(Delta_Kpi))*((qop+1) + 1/(qop+1)             )*sin(phi))",
                               parameters)));
    break;
  case theory_config::theoretical:
    theory->add(
        *(Utils::makeTheoryVar("DY_RS_th", "DY_RS_th",
                               "pow(R_Kpi/100, 0.5) * "
                               "(  (-y12*cos(Delta_Kpi)*cos(phiG) + x12*sin(Delta_Kpi)*cos(phiM))*(Acp_KP/100)*0.5"
                               " + ( y12*sin(Delta_Kpi)*sin(phiG) + x12*cos(Delta_Kpi)*sin(phiM))                 )",
                               parameters)));
    break;
  case theory_config::superweak:
    theory->add(*(Utils::makeTheoryVar("DY_RS_th", "DY_RS_th", "pow(R_Kpi/100, 0.5) * x12 * cos(Delta_Kpi) * sin(phiM)",
                                       parameters)));
    break;
  default:
    std::cout << "PDF_DY_RS::initRelations : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
}

void PDF_DY_RS::initObservables(const TString& setName) {
  observables = new RooArgList("observables");
  observables->add(*(new RooRealVar("DY_RS_obs", setName + "   #it{A}_{#Gamma}^{#it{K#pi}}", 0, -1e4, 1e4)));
}

void PDF_DY_RS::setObservables(TString c) {
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("LHCb2021")) {
    obsValSource = "https://inspirehep.net/literature/1864385";
    setObservable("DY_RS_obs", -0.36e-2);
  } else {
    std::cout << "PDF_DY_RS::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_DY_RS::setUncertainties(TString c) {
  if (c.EqualTo("LHCb2021")) {
    obsErrSource = "https://inspirehep.net/literature/1864385";
    StatErr[0] = 0.50e-2;
    SystErr[0] = 0.23e-2;
  } else {
    std::cout << "PDF_DY_RS::setUncertainties() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_DY_RS::setCorrelations(TString c) {
  resetCorrelations();
  corSource = "No correlations for one observable";
}

void PDF_DY_RS::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
