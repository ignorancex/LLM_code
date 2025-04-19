/**
 * Charm Combination
 * Author: tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <RooMultiVarGaussian.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_DY.h>
#include <ParametersCharmCombo.h>

PDF_DY::PDF_DY(TString measurement_id, const theory_config th_cfg, const FSC fsc)
    : PDF_Abs{fsc == FSC::none ? 1 : 2}, th_cfg{th_cfg}, fsc{fsc} {
  name = "DY_" + measurement_id;
  initParameters();
  initRelations();
  initObservables(measurement_id);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_DY::~PDF_DY() {}

void PDF_DY::initParameters() {
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
  case theory_config::superweak:
    parameters->add(*(p.get("x12")));
    parameters->add(*(p.get("y12")));
    parameters->add(*(p.get("phiM")));
    break;
  default:
    std::cout << "PDF_DY::initParameters : ERROR : "
                 "theory_config not supported."
              << std::endl;
    exit(1);
  }
  switch (fsc) {
  case FSC::none:
    break;
  case FSC::partial:
    parameters->add(*(p.get("Acp_KK")));
    parameters->add(*(p.get("Acp_PP")));
    break;
  case FSC::full:
    parameters->add(*(p.get("Acp_KK")));
    parameters->add(*(p.get("Acp_PP")));
    parameters->add(*(p.get("cot_delta_KK")));
    parameters->add(*(p.get("cot_delta_PP")));
    break;
  }
}

void PDF_DY::initRelations() {
  theory = new RooArgList("theory");
  if (nObs == 1) {
    theory->add(*(Utils::makeTheoryVar("DY_th", "DY_th", CharmUtils::get_dy_expression(th_cfg), parameters)));
  } else if (nObs == 2) {
    theory->add(
        *(Utils::makeTheoryVar("DY_KK_th", "DY_KK_th", CharmUtils::get_dy_expression(th_cfg, fsc, "KK"), parameters)));
    theory->add(
        *(Utils::makeTheoryVar("DY_PP_th", "DY_PP_th", CharmUtils::get_dy_expression(th_cfg, fsc, "PP"), parameters)));
  }
}

void PDF_DY::initObservables(const TString& setName) {
  observables = new RooArgList("observables");
  if (nObs == 1) {
    observables->add(*(new RooRealVar("DY_obs", setName + "   #Delta#it{Y}", 0, -1e4, 1e4)));
  } else if (nObs == 2) {
    observables->add(
        *(new RooRealVar("DY_KK_obs", setName + "   #Delta#it{Y}_{#it{K}^{+}#it{K}^{#minus}}", 0, -1e4, 1e4)));
    observables->add(
        *(new RooRealVar("DY_PP_obs", setName + "   #Delta#it{Y}_{#it{#pi}^{+}#it{#pi}^{#minus}}", 0, -1e4, 1e4)));
  }
}

void PDF_DY::setObservables(TString c) {
  obsValSource = "https://github.com/tpajero/charm-fitter/tree/master/charmcombo/blue/DY.cpp";
  if (c.EqualTo("truth"))
    setObservablesTruth();
  else if (c.EqualTo("toy"))
    setObservablesToy();
  else if (nObs == 1 && c.EqualTo("WA2019"))
    setObservable("DY_obs", 3.2e-2);
  else if (nObs == 1 && c.EqualTo("WA2020"))
    setObservable("DY_obs", 3.1e-2);
  else if (nObs == 1 && c.EqualTo("WA2021"))
    setObservable("DY_obs", -0.92e-2);
  else if (nObs == 1 && c.EqualTo("Belle&BaBar"))
    setObservable("DY_obs", -1.68e-2);
  else if (nObs == 2 && c.EqualTo("WA2020")) {
    setObservable("DY_KK_obs", 4.99e-2);
    setObservable("DY_PP_obs", -2.40e-2);
  } else if (nObs == 2 && c.EqualTo("WA2021")) {
    setObservable("DY_KK_obs", -0.20e-2);
    setObservable("DY_PP_obs", -3.53e-2);
  } else {
    std::cout << "PDF_DY::setObservables() : ERROR : config " << c << " not found for " << nObs << " DY observables."
              << std::endl;
    exit(1);
  }
}

void PDF_DY::setUncertainties(TString c) {
  obsErrSource = "https://github.com/tpajero/charm-fitter/tree/master/charmcombo/blue/DY.cpp";
  if (nObs == 1 && c.EqualTo("WA2019")) {
    StatErr[0] = 2.6e-2;
    SystErr[0] = 0.;
  } else if (nObs == 1 && c.EqualTo("WA2020")) {
    StatErr[0] = 2.0e-2;
    SystErr[0] = 0.;
  } else if (nObs == 1 && c.EqualTo("WA2021")) {
    StatErr[0] = 1.11e-2;
    SystErr[0] = 0.33e-2;
  } else if (nObs == 1 && c.EqualTo("Belle&BaBar")) {
    StatErr[0] = 15.75e-2;
    SystErr[0] = 4.81e-2;
  } else if (nObs == 2 && c.EqualTo("WA2020")) {
    StatErr[0] = 2.35e-2;
    SystErr[0] = 0.57e-2;
    StatErr[1] = 4.30e-2;
    SystErr[1] = 0.70e-2;
  } else if (nObs == 2 && c.EqualTo("WA2021")) {
    StatErr[0] = 1.28e-2;
    SystErr[0] = 0.32e-2;
    StatErr[1] = 2.36e-2;
    SystErr[1] = 0.39e-2;
  } else {
    std::cout << "PDF_DY::setUncertainties() : ERROR : config " << c << " not found for " << nObs << " DY observables."
              << std::endl;
    exit(1);
  }
}

void PDF_DY::setCorrelations(TString c) {
  resetCorrelations();
  corSource = "https://github.com/tpajero/charm-fitter/tree/master/charmcombo/blue/DY.cpp";
  if (nObs == 1)
    corSource = "No correlations for one observable";
  else if (nObs == 2 && c.EqualTo("WA2020"))
    corSystMatrix[0][1] = 0.63;  // np.sum(np.square([0.05, 0.42, 0.10, 0.04, 0.23, 0.09])) / 0.57 / 0.70
  else if (nObs == 2 && c.EqualTo("WA2021"))
    corSystMatrix[0][1] = 0.68;  // np.sum(np.square([0.18, 0.21, 0.06, 0.01, 0.07])) / 0.32 / 0.39
  else {
    std::cout << "PDF_DY::setCorrelations() : ERROR : config " << c << " not found for " << nObs << " DY observables."
              << std::endl;
    exit(1);
  }
}

void PDF_DY::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
