import sys
import sensitivity
import variants
import flows

todo = sys.argv[1]
if todo == 'phi-base':
  variants.gen_phi_base()
if todo == 'compare-turnover':
  variants.simple_turnover()
if todo == 'fit':
  variants.run_fit()
if todo == 'compare-tpaf':
  variants.exp_tpaf()
if todo == 'sensitivity-run':
  sensitivity.run_sims(sys.argv[2])
if todo == 'sensitivity-plot':
  sensitivity.gen_plots()
if todo == 'flows':
  flows.run_sims()
if todo == 'debug':
  sensitivity.run_sims('debug')
