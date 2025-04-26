/**
 * Combination of all WS/RS D0 -> K pi measurements.
 *
 * Needed to plot the y' vs. x'2 contours plot.
 */

#include <string>
#include <vector>

#include <GammaComboEngine.h>

#include <CharmUtils.h>
#include <PDF_WS.h>
#include <PDF_WS_NoCPV.h>

namespace {
  /**
   * Clone all combinations to enable the comparison of fits with and without allowing for CPV in DCS decays.
   */
  void add_no_dcs_cpv_combiner(GammaComboEngine& gc, const int icomb) {
    const auto combiner = gc.getCombiner(icomb);
    const auto name = combiner->getName() + "NoDcsCPV";
    const auto title = combiner->getTitle() + " (A_{CP}(K#pi) = 0)";
    gc.cloneCombiner(1000 + icomb, icomb, name, title);
    return;
  }

  std::vector<int> get_lhcb_pdfs(const std::string run, const FSC fsc) {
    if (run == "run12") {
      std::vector<int> list = {
          38,  // Prompt Run 1+2
          41,  // DT Run 1+2
      };
      return list;
    } else {
      std::cerr << "The list of the LHCb results from the period `" << run << "` is not supported. Exiting..."
                << std::endl;
      exit(1);
    }
  }
}  // namespace

int main(int argc, char* argv[]) {

  GammaComboEngine gc("D0ToKPi", argc, &argv[0]);

  // Define the PDFs
  const auto th_cfg = theory_config::d0_to_kpi;
  // clang-format off
  gc.addPdf(30, new PDF_WS_NoCPV("CDF",   th_cfg),                                         "WS/RS        CDF                            ");
  gc.addPdf(31, new PDF_WS_NoCPV("BaBar", th_cfg),                                         "WS/RS        BaBar    no CPV                ");
  gc.addPdf(32, new PDF_WS_NoCPV("Belle", th_cfg),                                         "WS/RS        Belle    no CPV                ");
  gc.addPdf(33, new PDF_WS_NoCPV("BaBar", th_cfg),                                         "WS/RS        BaBar                          ");
  gc.addPdf(34, new PDF_WS_NoCPV("Belle", th_cfg),                                         "WS/RS        Belle                          ");
  gc.addPdf(35, new PDF_WS("LHCb_DT_Run1",           th_cfg),                              "WS/RS        LHCb     Run 1    [B -> D* mu] ");
  gc.addPdf(36, new PDF_WS("LHCb_Run1",              th_cfg),                              "WS/RS        LHCb     Run 1                 ");
  gc.addPdf(37, new PDF_WS("LHCb_Prompt_2011_2016",  th_cfg),                              "WS/RS        LHCb     2011-6   [D* -> D0 pi]");
  gc.addPdf(38, new PDF_WS("LHCb_Prompt_Run12_sec9", th_cfg, WS_parametrisation::ccprime), "WS/RS        LHCb     Run 1-2  [D* -> D0 pi]");
  gc.addPdf(40, new PDF_WS("LHCb_DT_Run2", th_cfg),                                        "WS/RS        LHCb     Run 2    [B -> D* mu] ");
  gc.addPdf(41, new PDF_WS("LHCb_DT_Run12", th_cfg),                                       "WS/RS        LHCb     Run 1-2  [B -> D* mu] ");
  // clang-format on

  // Define the combinations -------------------------------------------------------------------------------------------

  gc.newCombiner(0, "emtpy", "empty");

  // Run 1 prompt + doubly tagged, Run 2 prompt
  gc.newCombiner(1, "LHCbSum2024", "LHCb Run 1 + LHCb Run 2 prompt", 35, 38);

  // Run 1+2 prompt+doubly tagged
  gc.cloneCombiner(2, 1, "LHCbAut2024", "LHCb Run 1+2");
  gc.getCombiner(2)->delPdf(gc[35]);  // LHCb DT Run 1
  gc.getCombiner(2)->addPdf(gc[41]);  // LHCb DT Run 1+2

  // Clone all combiners for no DCS CPV hypothesis ---------------------------------------------------------------------

  for (const auto id : gc.getCombinersIds()) add_no_dcs_cpv_combiner(gc, id);

  // Execute the combination
  gc.run();

  return 0;
}
