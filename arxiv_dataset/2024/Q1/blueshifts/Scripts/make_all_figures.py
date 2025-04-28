import fig2_heuristic
import fig4_Inclination
import fig5_Impacts
import fig6_BlueshiftSkew
import fig7_WindStructure
import fig10_StepPar
import fig11_LineFormation
import fig12_SedPlot
import sys 
import blueshift_util

tex = "True"
if len(sys.argv) > 1:
    if sys.argv[1] == "--notex":
        tex = "False"

# set defaults
blueshift_util.set_plot_defaults(tex=tex)

# make plots 
fig2_heuristic.make_figure()
fig4_Inclination.make_figure()
fig5_Impacts.make_figure()
fig6_BlueshiftSkew.make_figure()
fig7_WindStructure.make_figure()
fig10_StepPar.make_figure()
fig11_LineFormation.make_figure()
fig12_SedPlot.make_figure()
print("All Done.")

# TO DO
# Get all figure plots working
# minimal virtual environment and test.
