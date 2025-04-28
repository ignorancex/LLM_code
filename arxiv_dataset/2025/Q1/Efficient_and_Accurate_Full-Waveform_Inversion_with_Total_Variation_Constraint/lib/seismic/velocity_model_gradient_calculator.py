from typing import List, Tuple

import numpy as np
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, solve
from numpy.typing import NDArray

from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


class VelocityModelGradientCalculator:
    def __init__(self, true_model: SeismicModel, geometry: AcquisitionGeometry, simulator: AcousticWaveSolver):
        self.true_model = true_model
        self.geometry = geometry
        self.simulator = simulator
        self.grad_operator = self._create_grad_op(true_model, geometry, simulator)
        pass

    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver):
        grad = Function(name="grad", grid=true_model.grid)
        u = TimeFunction(name="u", grid=true_model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
        v = TimeFunction(name="v", grid=true_model.grid, save=None, time_order=2, space_order=solver.space_order)

        eqns = [Eq(v.backward, solve(true_model.m * v.dt2 - v.laplace + true_model.damp * v.dt.T, v.backward))]
        rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * true_model.grid.stepping_dim.spacing**2 / true_model.m)
        gradient_update = Inc(grad, -u.dt2 * v * true_model.m**1.5)

        return Operator(eqns + rec_term + [gradient_update], subs=true_model.spacing_map, name="Gradient")

    def _calc_true_observed_waveforms(self, source_locations: NDArray[np.float64]) -> List[NDArray[np.float64]]:
        observed_waveform = Receiver(name="d_obs", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        true_observed_waveforms = []
        n_shots = source_locations.shape[0]
        for i in range(n_shots):
            self.geometry.src_positions[0, :] = source_locations[i, :]
            self.simulator.forward(vp=self.true_model.vp, save=True, rec=observed_waveform)
            true_observed_waveforms.append(observed_waveform.data[:].copy())
        return true_observed_waveforms

    def calc_grad(self, current_velocity_model: Function, source_locations: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
        grad = Function(name="grad", grid=self.true_model.grid)
        residual = Receiver(name="residual", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        calculated_waveform = Receiver(name="d_syn", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)

        observed_waveforms = self._calc_true_observed_waveforms(source_locations)

        objective = 0.0
        n_shots = source_locations.shape[0]
        for i in range(n_shots):
            self.geometry.src_positions[0, :] = source_locations[i, :]
            _, calculated_wave_field, _ = self.simulator.forward(vp=current_velocity_model, save=True, rec=calculated_waveform)

            residual.data[:] = calculated_waveform.data[:] - observed_waveforms[i]

            objective += 0.5 * norm(residual) ** 2
            self.grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=self.simulator.dt, vp=current_velocity_model)

        return objective, -grad.data
