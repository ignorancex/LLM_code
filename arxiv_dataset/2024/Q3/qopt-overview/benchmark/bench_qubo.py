"""Contains core QPU-interaction code.

Implements the abstract interface :py:mod:`bench_qubo.QUBOSolver`, and the
related "QPU solver classes" :py:class:`bench_qubo.QUBOSolverDwave` and
:py:class:`bench_qubo.QUBOSolverIBM`, along with technical data preprocessing
and access code in :py:class:`bench_qubo.DataBaseQUBO`. Most notably, each
device-specific class implements ``_run`` method that is used to submit a job to
an actual device.

Relies on abstact descriptions from :py:class:`bench.DataBase` and
:py:class:`Solver`.

"""
import hashlib
import os
import time
from abc import abstractmethod
from datetime import datetime
import json

import numpy as np
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler
from dwave.system import FixedEmbeddingComposite
from minorminer import find_embedding
from qiskit.algorithms.optimizers import COBYLA, SPSA, BOBYQA
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Estimator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy import optimize

from bench import DataBase, Solver

__max_bound__ = 9999


class DataBaseQUBO(DataBase):
    """Implements QUBO instance storage."""
    def __init__(self, directory, file_ext='.json', filter_fun=None, sort_fun=None, directory_embeddings=None):
        super().__init__(directory, file_ext, filter_fun, sort_fun)
        self.directory_embeddings = directory_embeddings

    def _preprocess_qubo_data(self, Q, P, a, b):
        if a is None or b is None:
            return Q, P
        else:
            assert a == 1 and b == 1
            qubo_matrix = Q / 2 + np.diag(P)
            scaling_factor = a / np.max(np.abs(qubo_matrix)) ** b
            Q_scaled = Q * scaling_factor
            P_scaled = P * scaling_factor
            return Q_scaled, P_scaled

    def get_qubo_data(self, instance_id, a, b):
        qubo_data = self.db[instance_id]
        # get_data, Const is ignored
        Q = np.array(qubo_data['Q'])
        P = np.array(qubo_data['P'])
        return self._preprocess_qubo_data(Q, P, a, b)

    def get_embedding_data(self, instance_id, object_hook=None):
        if self.directory_embeddings is not None:
            emb_filename = os.path.join(self.directory_embeddings, instance_id + '.emb.json')
            with open(emb_filename, 'r') as embfile:
                emb_precalc = json.load(embfile, object_hook=object_hook)
            return emb_precalc
        else:
            raise ValueError


class QUBOSolver(Solver):
    """Describes an abstract QPU solver interface."""
    def __init__(self, name, db):
        super().__init__(f'QUBO{name}', db)

    @abstractmethod
    def _run(self, instance_id, *args):
        raise NotImplementedError

    def get_data(self, instance_id, a, b):
        return self.db.get_qubo_data(instance_id, a, b)

    def get_embedding_data(self, instance_id, object_hook=None):
        return self.db.get_embedding_data(instance_id, object_hook)


class QUBOSolverDwave(QUBOSolver):
    """Implements D-Wave specific solver code."""
    def __init__(self, db, token):
        super().__init__('DwaveSolver', db)
        self.token = token

    def _load_embedding(self, instance_id):
        """Loads precomputed D-Wave embeddings from a file."""
        # load embedding from file
        def fix_ints(obj):
            if isinstance(obj, dict):
                return {int(key): val for key, val in obj.items()}
            return obj
        emb_precalc = self.get_embedding_data(instance_id, object_hook=fix_ints)
        assert len(emb_precalc) > 0
        emb_precalc_dt = -1
        if self.logger is not None:
            self.logger.debug(f'embedding: loaded precalculated embedding.')
        return emb_precalc, emb_precalc_dt

    def _calculate_embedding(self, Q_dict ,target_edgelist):
        """Computes D-Wave embedding locally (and tracks the respective time for that)."""
        start = time.time()
        emb_precalc = find_embedding(Q_dict, target_edgelist, verbose=1)
        end = time.time()
        emb_precalc_dt = end - start
        if self.logger is not None:
            self.logger.debug(f'embedding: calculated embedding.')
        return emb_precalc, emb_precalc_dt

    def _run(self, instance_id, num_reads, a, b):
        Q, P = self.get_data(instance_id, a, b)
        job_label = f'{hashlib.sha1(Q.view(np.uint8)).hexdigest()}_{hashlib.sha1(P.view(np.uint8)).hexdigest()}'

        # setup model
        bqm = BinaryQuadraticModel(P, Q / 2, vartype={0, 1})

        # setup sampler
        sampler = DWaveSampler(token=self.token)

        # find embedding
        Q_dict, Q_const = bqm.to_qubo()
        __, target_edgelist, target_adjacency = sampler.structure
        try:
            emb_precalc, emb_precalc_dt = self._load_embedding(instance_id)
        except:
            emb_precalc, emb_precalc_dt = self._calculate_embedding(Q_dict ,target_edgelist)

        # recall embedding
        emb = FixedEmbeddingComposite(sampler,
                                      emb_precalc)  # https://support.dwavesys.com/hc/en-us/community/posts/15574739345047-EmbeddingComposite-get-Time-and-Mapping

        # run on dwave
        if self.logger is not None:
            self.logger.debug(f'qpu: start.')
        sample_start = time.time()
        response = emb.sample(bqm, num_reads=num_reads, label=job_label)
        sample_end = time.time()
        if self.logger is not None:
            self.logger.debug(f'qpu: finished.')
        sample_dt = sample_end - sample_start

        # compile results
        options = dict(num_reads=num_reads, a=a, b=b)
        outcome = dict(job_label=job_label, emb_precalc=emb_precalc, emb_precalc_dt=emb_precalc_dt, sample_dt=sample_dt,
                       Q_dict=Q_dict, Q_const=Q_const, target_edgelist=target_edgelist,
                       target_adjacency=target_adjacency, emb_properties=emb.properties,
                       response=response.to_serializable(),
                       samples=[dict(sample=s.sample, energy=s.energy, num_occurrences=s.num_occurrences,
                                     chain_break_fraction=s.chain_break_fraction) for s in response.data()])
        return dict(options=options, outcome=outcome)


class QUBOSolverIBM(QUBOSolver):
    """Implements IBM-specific solver code."""
    def __init__(self, db):
        super().__init__('IBMSolver', db)

    def _run_qpu_estimator(self, session, estimator_options, optimizer_name, optimizer_kwargs, optimizer_obj,
                           optimizer_obj_kwargs, cost_func, ansatz, qubitOp, x0):
        estimator = Estimator(session=session, options=estimator_options)
        if optimizer_name == 'SPSA':
            res = optimizer_obj.minimize(fun=lambda params: cost_func(params, ansatz, qubitOp, estimator), x0=x0,
                                         **optimizer_obj_kwargs)
        elif optimizer_name == 'BOBYQA':
            res = optimizer_obj.minimize(fun=lambda params: cost_func(params, ansatz, qubitOp, estimator), x0=x0,
                                         **optimizer_obj_kwargs)
        else:
            res = optimize.minimize(cost_func, x0, args=(ansatz, qubitOp, estimator), method=optimizer_name,
                                    options=optimizer_kwargs)
        return res

    def _run_qpu_sampler(self, session, sampler_options, ansatz, res):
        sampler = Sampler(session=session, options=sampler_options)
        qc = ansatz.assign_parameters(res.x)
        qc.measure_all()
        samp_dist = sampler.run(qc).result().quasi_dists[0]
        return samp_dist

    def _run(self, instance_id, optimizer_kwargs, optimizer_name, sampler_optimization_level, sampler_resilience_level,
             estimator_optimization_level, estimator_resilience_level, reps, sampler_shots,
             estimator_shots, backend_name, service_channel, service_instance, max_time, recover_on_exception, a, b):
        # optimizer ~ COBYLA, SPSA
        # reps
        # optimization level, resilience level
        Q, P = self.get_data(instance_id, a, b)

        # build ising qubo
        qp = QuadraticProgram()
        qp.binary_var_list(len(P))
        qp.minimize(constant=0.0, linear=P, quadratic=Q / 2)
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        qubitOp, qubitOffset = qubo.to_ising()

        # fixed parameters
        qiskit_runtime_service_channel = service_channel
        qiskit_runtime_service_instance = service_instance

        # setup quantum
        sampler_options = dict(optimization_level=sampler_optimization_level, resilience_level=sampler_resilience_level,
                               shots=sampler_shots)
        estimator_options = dict(optimization_level=estimator_optimization_level,
                                 resilience_level=estimator_resilience_level, shots=estimator_shots)
        solver_history = dict(params=[], cost=[], metadata=[], timestamp=[])

        def cost_func(params, ansatz, hamiltonian, estimator):
            estimator_result = estimator.run(ansatz, hamiltonian, parameter_values=params).result()
            cost = estimator_result.values[0]
            metadata = estimator_result.metadata[0]
            solver_history['params'].append(params)
            solver_history['cost'].append(cost)
            solver_history['metadata'].append(metadata)
            solver_history['timestamp'].append(datetime.now().timestamp())
            if self.logger is not None:
                self.logger.debug(f'solver: cost={cost}, params={params}, index={len(solver_history["params"]) - 1}.')
            return cost

        ansatz = QAOAAnsatz(qubitOp, reps=reps)
        x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)

        service = QiskitRuntimeService(channel=qiskit_runtime_service_channel, instance=qiskit_runtime_service_instance)
        backend = service.backend(backend_name)

        if optimizer_name == 'COBYLA':
            optimizer_obj = COBYLA(**optimizer_kwargs)
            optimizer_obj_kwargs = dict()
        elif optimizer_name == 'SPSA':
            optimizer_obj = SPSA(**optimizer_kwargs)
            optimizer_obj_kwargs = dict()
        elif optimizer_name == 'BOBYQA':
            optimizer_obj = BOBYQA(**optimizer_kwargs)
            optimizer_obj_kwargs = dict(jac=None, bounds=[
                (float(l) if l is not None else -__max_bound__, float(u) if u is not None else __max_bound__) for l, u
                in
                ansatz.parameter_bounds])

            print('optimizer_obj_kwargs\n\t', optimizer_obj_kwargs)

        else:
            optimizer_obj = None
            optimizer_obj_kwargs = None

        # solve quantum
        if self.logger is not None:
            self.logger.debug(f'session start.')
        with Session(service=service, backend=backend, max_time=max_time) as session:
            if self.logger is not None:
                self.logger.debug(f'estimator start.')
            optimization_loop_finished = False
            while not optimization_loop_finished:
                try:
                    res = self._run_qpu_estimator(session, estimator_options, optimizer_name, optimizer_kwargs,
                                                  optimizer_obj, optimizer_obj_kwargs, cost_func, ansatz, qubitOp, x0)
                except Exception as e:
                    if not recover_on_exception:
                        raise e
                    if self.logger is not None:
                        self.logger.debug(f'estimator exception: {e}, try to recover from history.')
                    x0 = solver_history['params'][-1]
                optimization_loop_finished = True
            if self.logger is not None:
                self.logger.debug(f'sampler start.')
            sampling_finished = False
            while not sampling_finished:
                try:
                    samp_dist = self._run_qpu_sampler(session, sampler_options, ansatz, res)
                except Exception as e:
                    if not recover_on_exception:
                        raise e
                    if self.logger is not None:
                        self.logger.debug(f'sampler exception: {e}, try to repeat.')
                sampling_finished = True
        if self.logger is not None:
            self.logger.debug(f'session end.')

        # retrieve result
        result = dict(res=res, samp_dist=samp_dist)
        session_data = dict(session_id=session.session_id, job_data=dict())
        for job in session.service.jobs():
            job_result = job.result()
            # filename = 'tmp_circuits.qpy'
            # circuits = job.inputs['circuits']
            # with open(filename, 'wb') as fh:
            #    qpy.dump(circuits, fh)
            # with open(filename, 'rb') as fh:
            #    file_contents = fh.readlines()
            # circuit_strings = [str(circuit) for circuit in job.inputs['circuits']]
            # num_circuits = len(circuits)
            try:
                quasi_dists = job_result.quasi_dists,
            except:
                quasi_dists = None
            try:
                values = job_result.values,
            except:
                values = None
            session_data['job_data'][job.job_id()] = dict(
                # circuits=dict(n=num_circuits, qpy=file_contents),
                quasi_dists=quasi_dists, values=values,
                metadata=job_result.metadata, metrics=job.metrics(),
                timestamp=job.creation_date.timestamp(),
                # logs=job.logs()
            )

        # query backend
        backend_properties = backend.properties()
        backend_properties = backend_properties.to_dict() if backend_properties is not None else {}

        # compile results
        options = dict(optimizer_kwargs=optimizer_kwargs, optimizer_name=optimizer_name,
                       sampler_optimization_level=sampler_optimization_level,
                       sampler_resilience_level=sampler_resilience_level,
                       estimator_optimization_level=estimator_optimization_level,
                       estimator_resilience_level=estimator_resilience_level,
                       reps=reps, sampler_shots=sampler_shots, estimator_shots=estimator_shots,
                       backend_name=backend_name,
                       qiskit_runtime_service_channel=qiskit_runtime_service_channel,
                       qiskit_runtime_service_instance=qiskit_runtime_service_instance,
                       max_time=max_time, recover_on_exception=recover_on_exception,
                       a=a, b=b)
        outcome = dict(result=result, session_data=session_data, x0=x0.tolist(),
                       solver_history=solver_history, backend_properties=backend_properties,
                       op=dict(qubitOp=repr(qubitOp), qubitOffset=qubitOffset))
        return dict(options=options, outcome=outcome)
