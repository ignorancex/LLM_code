/**
 * Charm Combination
 * Author: tommaso.pajero@cern.ch
 * Date: October 2021
 **/

#include <RooMultiVarGaussian.h>

#include <TString.h>

#include <Utils.h>

#include <CharmUtils.h>
#include <PDF_Fp_pipipi0.h>
#include <ParametersCharmCombo.h>

PDF_Fp_pipipi0::PDF_Fp_pipipi0(TString measurement_id) : PDF_Abs{1} {
  name = "Fp-pipipi0" + measurement_id;
  initParameters();
  initRelations();
  initObservables(measurement_id);
  setObservables(measurement_id);
  setUncertainties(measurement_id);
  setCorrelations(measurement_id);
  buildCov();
  buildPdf();
}

PDF_Fp_pipipi0::~PDF_Fp_pipipi0() {}

void PDF_Fp_pipipi0::initParameters() {
  ParametersCharmCombo p;
  parameters = new RooArgList("parameters");
  parameters->add(*(p.get("F_pipipi0")));
}

void PDF_Fp_pipipi0::initRelations() {
  theory = new RooArgList("theory");
  theory->add(*(Utils::makeTheoryVar("F_pipipi0_th", "F_pipipi0_th", "F_pipipi0", parameters)));
}

void PDF_Fp_pipipi0::initObservables(TString measurement_id) {
  observables = new RooArgList("observables");
  observables->add(*(new RooRealVar("F_pipipi0_obs", "F_{#pi#pi#pi^{0}} " + measurement_id, 0, -1e4, 1e4)));
}

void PDF_Fp_pipipi0::setObservables(TString c) {
  if (c.EqualTo("Cleo-c")) {
    obsValSource = "https://inspirehep.net/literature/2139827";
    setObservable("F_pipipi0_obs", 0.973);
  } else if (c.EqualTo("BESIII")) {
    obsValSource = "https://inspirehep.net/literature/2827201";
    setObservable("F_pipipi0_obs", 0.9406);
  } else {
    std::cout << "PDF_Fp_pipipi0::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Fp_pipipi0::setUncertainties(TString c) {
  if (c.EqualTo("Cleo-c")) {
    obsErrSource = "https://inspirehep.net/literature/2139827";
    StatErr[0] = 0.017;
    SystErr[0] = 0.;
  } else if (c.EqualTo("BESIII")) {
    obsErrSource = "https://inspirehep.net/literature/2827201";
    StatErr[0] = 0.0036;
    SystErr[0] = 0.0021;
  } else {
    std::cout << "PDF_Fp_pipipi0::setObservables() : ERROR : config " + c + " not found." << std::endl;
    exit(1);
  }
}

void PDF_Fp_pipipi0::setCorrelations(TString c) {
  resetCorrelations();
  corSource = "No correlations for one observable";
}

void PDF_Fp_pipipi0::buildPdf() {
  pdf = new RooMultiVarGaussian("pdf_" + name, "pdf_" + name, *(RooArgSet*)observables, *(RooArgSet*)theory, covMatrix);
}
