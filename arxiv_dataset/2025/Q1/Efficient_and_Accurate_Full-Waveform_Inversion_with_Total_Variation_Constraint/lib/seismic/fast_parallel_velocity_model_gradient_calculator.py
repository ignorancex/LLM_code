from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from devito import Eq, Function, Inc, Operator, TimeFunction, norm, solve
from numpy.typing import NDArray

from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


class FastParallelVelocityModelGradientCalculatorProps(NamedTuple):
    true_velocity_model: npt.NDArray
    initial_velocity_model: npt.NDArray
    shape: Tuple[int, int]
    spacing: Tuple[float, float]
    damping_cell_thickness: int
    start_time: float
    end_time: float
    source_frequency: float
    source_locations: npt.NDArray
    receiver_locations: npt.NDArray
    noise_sigma: float
    n_jobs: int


def get_time_length(props: FastParallelVelocityModelGradientCalculatorProps):
    true_model = SeismicModel(
        space_order=2, vp=props.true_velocity_model, origin=(0, 0), shape=props.shape, dtype=np.float32, spacing=props.spacing, nbl=props.damping_cell_thickness, bcs="damp", fs=False
    )
    geometry = AcquisitionGeometry(true_model, props.receiver_locations, np.array([[0, 0]]), props.start_time, props.end_time, f0=props.source_frequency, src_type="Ricker")
    return geometry.nt


class FastParallelVelocityModelGradientCalculator:
    def __init__(self, props: FastParallelVelocityModelGradientCalculatorProps):
        self.n_shots = len(props.source_locations)
        self.n_jobs = props.n_jobs

        a = get_time_length(props)

        n_receivers = len(props.receiver_locations)
        dsize = props.damping_cell_thickness
        vm_shape = (props.shape[0] + dsize * 2, props.shape[1] + dsize * 2)
        time_length = get_time_length(props)

        self.velocity_model_shared_memory = SharedMemory(create=True, size=np.prod(vm_shape) * np.dtype(np.float32).itemsize)
        self.velocity_model = np.ndarray(vm_shape, dtype=np.float32, buffer=self.velocity_model_shared_memory.buf)

        self.true_observed_waveforms_memory = SharedMemory(create=True, size=self.n_shots * time_length * n_receivers * np.dtype(np.float32).itemsize)
        self.residual_norm_shared_memory = SharedMemory(create=True, size=self.n_shots * np.dtype(np.float32).itemsize)
        self.vm_grad_shared_memory = SharedMemory(create=True, size=self.n_shots * np.prod(vm_shape) * np.dtype(np.float32).itemsize)
        # self.true_observed_waveform = np.ndarray((self.n_shots, time_length, n_receivers), dtype=np.float32, buffer=self.residual_norm_shared_memory.buf)
        self.residual_norms = np.ndarray(self.n_shots, dtype=np.float32, buffer=self.residual_norm_shared_memory.buf)
        self.vm_grads = np.ndarray((self.n_shots, vm_shape[0], vm_shape[1]), dtype=np.float32, buffer=self.vm_grad_shared_memory.buf)

        self.input_queue = Queue()
        self.output_queue = Queue()

        print("process initializing...")
        self.processes = [
            Process(
                target=calc_grad_worker,
                args=(
                    props,
                    self.velocity_model_shared_memory.name,
                    self.true_observed_waveforms_memory.name,
                    self.vm_grad_shared_memory.name,
                    self.residual_norm_shared_memory.name,
                    self.input_queue,
                    self.output_queue,
                ),
            )
            for _ in range(self.n_jobs)
        ]
        for p in self.processes:
            p.start()

        # wait for initialization
        for _ in range(self.n_jobs):
            self.output_queue.get()

        for i in range(self.n_shots):
            self.input_queue.put(-i - 1)

        for _ in range(self.n_shots):
            self.output_queue.get()

        print("process initialized!")

    def __del__(self):
        for _ in range(self.n_jobs):
            self.input_queue.put(None)

        for p in self.processes:
            p.join()

        self.velocity_model_shared_memory.close()
        self.velocity_model_shared_memory.unlink()
        self.true_observed_waveforms_memory.close()
        self.true_observed_waveforms_memory.unlink()
        self.residual_norm_shared_memory.close()
        self.residual_norm_shared_memory.unlink()
        self.vm_grad_shared_memory.close()
        self.vm_grad_shared_memory.unlink()

    def calc_grad(self, current_velocity_model: npt.NDArray) -> Tuple[float, NDArray[np.float32]]:
        self.velocity_model[:] = current_velocity_model

        for i in range(self.n_shots):
            self.input_queue.put(i)

        for _ in range(self.n_shots):
            self.output_queue.get()

        objective = 0
        grad_value = np.zeros_like(current_velocity_model)
        for i in range(self.n_shots):
            objective += self.residual_norms[i]
            grad_value += self.vm_grads[i]

        return objective, grad_value


def calc_grad_worker(
    props: FastParallelVelocityModelGradientCalculatorProps,
    velocity_model_shared_memory_name: str,
    true_observed_waveforms_shared_memory_name: str,
    vm_grad_shared_memory_name: str,
    residual_norm_shared_memory_name: str,
    input_queue: Queue,
    output_queue: Queue,
):
    n_shots = len(props.source_locations)
    n_receivers = len(props.receiver_locations)
    vm_shape = (props.shape[0] + props.damping_cell_thickness * 2, props.shape[1] + props.damping_cell_thickness * 2)
    time_length = get_time_length(props)

    velocity_model_shared_memory = SharedMemory(name=velocity_model_shared_memory_name)
    velocity_model = np.ndarray(vm_shape, dtype=np.float32, buffer=velocity_model_shared_memory.buf)

    true_observed_waveforms_shared_memory = SharedMemory(name=true_observed_waveforms_shared_memory_name)
    true_observed_waveforms = np.ndarray((n_shots, time_length, n_receivers), dtype=np.float32, buffer=true_observed_waveforms_shared_memory.buf)

    vm_grad_shared_memory = SharedMemory(name=vm_grad_shared_memory_name)
    vm_grad = np.ndarray((n_shots, vm_shape[0], vm_shape[1]), dtype=np.float32, buffer=vm_grad_shared_memory.buf)

    residual_norm_shared_memory = SharedMemory(name=residual_norm_shared_memory_name)
    residual_norm = np.ndarray(n_shots, dtype=np.float32, buffer=residual_norm_shared_memory.buf)

    grad_calculator = FastParallelVelocityModelGradientCalculatorHelper(props, true_observed_waveforms)

    velocity_model[:] = grad_calculator.current_model.vp.data

    output_queue.put(0)

    while True:
        idx = input_queue.get()
        if idx is None:
            break
        if idx < 0:
            # calculate true observed waveform
            true_observed_waveforms[-idx - 1] = grad_calculator.calc_true_observed_waveform(-idx - 1)
        else:
            # calculate grad
            residual_norm[idx], vm_grad[idx] = grad_calculator.calc_grad(velocity_model, idx)

        output_queue.put(0)

    velocity_model_shared_memory.close()
    vm_grad_shared_memory.close()


class FastParallelVelocityModelGradientCalculatorHelper:
    def __init__(self, props: FastParallelVelocityModelGradientCalculatorProps, true_observed_waveforms: npt.NDArray):
        self.true_model = SeismicModel(
            space_order=2, vp=props.true_velocity_model, origin=(0, 0), shape=props.shape, dtype=np.float32, spacing=props.spacing, nbl=props.damping_cell_thickness, bcs="damp", fs=False
        )
        self.current_model = SeismicModel(
            space_order=2, vp=props.initial_velocity_model, origin=(0, 0), shape=props.shape, dtype=np.float32, spacing=props.spacing, nbl=props.damping_cell_thickness, bcs="damp", fs=False
        )

        self.geometry = AcquisitionGeometry(self.true_model, props.receiver_locations, np.array([[0, 0]]), props.start_time, props.end_time, f0=props.source_frequency, src_type="Ricker")
        # self.geometry.src_positions[0][:] = props.source_locations[idx]

        self.simulator = AcousticWaveSolver(self.true_model, self.geometry, space_order=4)

        self.grad_operator = self._create_grad_op(self.true_model, self.geometry, self.simulator)
        # self.observed_waveform = self._calc_true_observed_waveform(self.true_model, self.geometry, self.simulator, props.noise_sigma)

        self.true_observed_waveforms = true_observed_waveforms
        self.source_locations = props.source_locations
        self.noise_sigma = props.noise_sigma

    @classmethod
    def _add_gauss_noise(cls, observed_waveform_data: npt.NDArray[np.float32], sigma: float) -> npt.NDArray[np.float32]:
        noise = np.random.normal(0, sigma, observed_waveform_data.shape)
        return observed_waveform_data + noise

    @classmethod
    def _create_grad_op(cls, true_model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver) -> Operator:
        grad = Function(name="grad", grid=true_model.grid)
        u = TimeFunction(name="u", grid=true_model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
        v = TimeFunction(name="v", grid=true_model.grid, save=None, time_order=2, space_order=solver.space_order)

        eqns = [Eq(v.backward, solve(true_model.m * v.dt2 - v.laplace + true_model.damp * v.dt.T, v.backward))]
        rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * true_model.grid.stepping_dim.spacing**2 / true_model.m)
        gradient_update = Inc(grad, -u.dt2 * v * true_model.m**1.5)

        return Operator(eqns + rec_term + [gradient_update], subs=true_model.spacing_map, name="Gradient")

    def calc_true_observed_waveform(self, idx: int) -> NDArray:
        self.geometry.src_positions[0][:] = self.source_locations[idx]
        observed_waveform = Receiver(name="d_obs", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        self.simulator.forward(vp=self.true_model.vp, rec=observed_waveform)
        return FastParallelVelocityModelGradientCalculatorHelper._add_gauss_noise(observed_waveform.data.copy(), self.noise_sigma)

    def calc_grad(self, current_velocity_model: npt.NDArray, idx: int) -> Tuple[float, NDArray[np.float32]]:
        self.geometry.src_positions[0][:] = self.source_locations[idx]

        grad = Function(name="grad", grid=self.true_model.grid)
        residual = Receiver(name="residual", grid=self.current_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        calculated_waveform = Receiver(name="d_syn", grid=self.true_model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)

        self.current_model.vp.data[:] = current_velocity_model

        _, calculated_wave_field, _ = self.simulator.forward(vp=self.current_model.vp, save=True, rec=calculated_waveform)
        calculated_waveform.data[:] = calculated_waveform.data

        residual.data[:] = calculated_waveform.data - self.true_observed_waveforms[idx]

        # Proper objective calculation
        # objective = 0.5 * norm(residual) ** 2

        # Rough objective calculation
        objective = 0.5 * np.sum(np.abs(residual.data**2))

        self.grad_operator.apply(rec=residual, grad=grad, u=calculated_wave_field, dt=self.simulator.dt, vp=self.current_model.vp)

        return objective, -grad.data
