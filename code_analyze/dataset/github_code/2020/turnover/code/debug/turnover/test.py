# double check X is constant through plotting
# can equivalently check if err(b/A) = 0
model = system.get_model()
model.params.collapse(['ii'])
sim = system.get_simulation(model,outputs=['X'])
print(sim.params['zeta'])
sim.solve()
sim.plot(outputs=['X'],selectors=[model.select[name] for name in ['S','I','T']])
import turnover
print(model.params['zeta'].update(0.2,ki='M',ii='H',ip='M'))
print(model.params['zeta'].update(0.2,ki='M',ii='H',ip='L'))
model.params['zeta'].update(turnover.turnover(
  nu   = model.params['nu'],
  mu   = model.params['mu'],
  px   = model.params['px'].islice(ki='M'),
  pe   = model.params['pe'].islice(ki='M'),
  zeta = model.params['zeta'].islice(ki='M'),
  dur  = model.params['dur'].islice(ki='M'),
))
sim = system.get_simulation(model,outputs=['X'])
sim.solve()
sim.plot(outputs=['X'],selectors=[model.select[name] for name in ['WH','MH','WM','MM','WL','ML']])
