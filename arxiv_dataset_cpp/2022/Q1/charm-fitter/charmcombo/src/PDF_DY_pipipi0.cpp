/**
 * Charm Combination
 * Author: tommaso.pajero@cern.ch
 * Date: June 2024
 **/

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_DY_pipipi0.h>
#include <ParametersCharmCombo.h>

PDF_DY_pipipi0::PDF_DY_pipipi0(TString measurement_id, const theory_config th_cfg) : PDF_Abs{1}, th_cfg{th_cfg} {
  name = "DY_pipipi0_" + measurement_id;
  initParameters();
  initRelations();
  initObservables(measurement_id);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  build();
}

PDF_DY_pipipi0::~PDF_DY_pipipi0() {}

void PDF_DY_pipipi0::initParameters() {
  std::vector<std::string> param_names = {"F_pipipi0"};
  switch (th_cfg) {
  case theory_config::phenomenological:
    param_names.insert(param_names.end(), {"x", "y", "qop", "phi"});
    break;
  case theory_config::theoretical:
  case theory_config::superweak:
    param_names.insert(param_names.end(), {"x12", "y12", "phiM"});
    break;
  default:
    std::cout << "PDF_DY_pipipi0::initParameters : ERROR : theory_config not supported.\n";
    exit(1);
  }
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  for (const auto& par : param_names) parameters->add(*(p.get(par)));
}

void PDF_DY_pipipi0::initRelations() {
  theory = new RooArgList("theory");
  theory->add(
      *(Utils::makeTheoryVar("DY_pipipi0_th", "DY_pipipi0_th",
                             "(2 * F_pipipi0 - 1) * (" + CharmUtils::get_dy_expression(th_cfg) + ")", parameters)));
}

void PDF_DY_pipipi0::initObservables(const TString& setName) {
  observables = new RooArgList("observables");
  observables->add(
      *(new RooRealVar("DY_pipipi0_obs", setName + "   #Delta#it{Y}(#pi^{+}#pi^{#minus}#pi^{0})", 0, -1e4, 1e4)));
}

void PDF_DY_pipipi0::setObservables(TString c) {
  obsValSource = "https://arxiv.org/abs/2405.06556";
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (c.EqualTo("LHCb-R2"))
    setObservable("DY_pipipi0_obs", -1.21e-2);
  else {
    std::cout << "PDF_DY_pipipi0::setObservables() : ERROR : config " << c << " not found for " << nObs
              << " DY_pipipi0 observables." << std::endl;
    exit(1);
  }
}

void PDF_DY_pipipi0::setUncertainties(TString c) {
  obsErrSource = "https://arxiv.org/abs/2405.06556";
  if (c.EqualTo("LHCb-R2")) {
    StatErr[0] = 5.97e-2;
    SystErr[0] = 2.01e-2;  // Removed the sys. unc. for the time binning
  } else {
    std::cout << "PDF_DY_pipipi0::setUncertainties() : ERROR : config " << c << " not found for " << nObs
              << " DY_pipipi0 observables." << std::endl;
    exit(1);
  }
}

void PDF_DY_pipipi0::setCorrelations(TString c) {
  resetCorrelations();
  corSource = "No correlations for one observable";
}

void PDF_DY_pipipi0::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
