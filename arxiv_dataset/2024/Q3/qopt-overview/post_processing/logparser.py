"""Implements JSON pasers, summarizing the raw QPU output files.

USAGE:
    | python -m post_processing.logparser summarize_dwave ./run_logs/dwave   ./run_logs/summaries/dwave_summary.csv   ./run_logs/summaries/dwave_stats.csv dwave
    |
    | python -m post_processing.logparser summarize_ibm   ./run_logs/ibm-qpu ./run_logs/summaries/ibm-qpu_summary.csv ./run_logs/summaries/ibm-qpu_stats.csv "IBM-QPU"
    |
    | python -m post_processing.logparser summarize_quera ./run_logs/quera   ./run_logs/summaries/quera_summary.csv   ./run_logs/summaries/quera_stats.csv
    |
    | python -m post_processing.logparser extract_samples dwave ./run_logs/dwave/<instance>.json ./run_logs/dwave/samples-csv/<instance>.sample.csv ./instances/QUBO ./instances/orig

The module implements the abstract parser class befining a universal interface
:py:class:`post_processing.logparser.QPULogParser` and the three derived classes
for device-specific code. Each of these most notably implements the
method ``extract_samples`` used to extract the data regarding individual QPU
shots, which allows to recover a solution and is then used to calculate the
objective values with the helper functions :py:func:`calculate_QUBO_objective`
and :py:func:`calculate_orig_objective`.

When run from the command line as presented above, the function specified in the
first argument is run. Therefore, log parser interface is wrapped into several
functions accessible from the command line.

"""
import json
import pandas as pd
from glob import glob
import os
import numpy as np

from sys import argv
from abc import abstractmethod
from ast import literal_eval as make_tuple

from TSP_inst import load_instance_by_ID as load_TSP_by_ID
from TSP_inst import unpack_tour_QUBO_bitstring, obj_val
from TSP_inst import is_feasible as TSP_is_feasible

from MWC_inst import load_instance_by_ID as load_MWC_by_ID
from MWC_inst import get_objective_from_sol as get_MWC_objective

from UDMWIS_inst import load_instance_by_ID as load_UDMWIS_by_ID
from UDMWIS_inst import get_objective_from_sol as get_UDMWIS_objective

from MIS_inst import is_IS, extract_G_from_json
from MIS_inst import load_instance_by_ID as load_UDMIS_by_ID

from qubo_utils import get_instance_size_by_ID, get_QUBO_by_ID, load_QUBO
from qubo_utils import instance_present_in_folder

######################################################################
# Helper functions

def get_inst_type(inst_id):
    """Helper: Extracts the instance type from ``inst_id``."""
    if inst_id[:3] == "TSP":
        return "TSP"
    elif inst_id[:3] == "MWC":
        return "MWC"
    elif inst_id[:6] == "UDMWIS":
       return "UDMWIS"
    elif inst_id[:5] == "UDMIS":
        return "UDMIS"
    else:
        raise ValueError(f"{inst_id}: wrong ID; unexpected instance type marker (TSP/MWC/UDMWIS expected).")

def calculate_QUBO_objective(inst_id, top_samples,
                       orig_dir="./instances/orig",
                       qubo_dir="./instances/QUBO"):
    """Helper: Returns the QUBO objective and the feasibility flag from a collection of samples.

    Calculates the objective values using the QUBO machinery
    and assesses the feasibility of each respective solution.

    Args:
        inst_id(str): instance id,
        top_samples(list): best samples to choose from (list of bitstrings).
        orig_dir(str): directory with original instance JSONs,
        qubo_dir(str): directory with QUBO instance JSONs,

    Returns:
        A tuple of ``(obj, feas)``, where ``obj`` is the best (minimum)
        feasible QUBO objective value (just minimum if no feasible solutions
        found among ``top_samples``) and ``feas`` = ``True`` if it is feasible,
        and ``False`` otherwise.

    Notes:
        - ``top_samples`` must contain a list of bitstrings, where higher
            ranking qubits are on the left: e.g., ``[b3, b2, b1, b0]``.
    """
    filename = get_QUBO_by_ID(inst_id, folder=qubo_dir)
    Q, P, C, qubojs = load_QUBO(filename)
    int_samples = [np.array([int(s) for s in reversed(x)]) for x in top_samples]
    QUBO_objectives = [(0.5 * x @ Q @ x + P @ x + C) for x in int_samples]

    match get_inst_type(inst_id):
        case "TSP":
            D, _ = load_TSP_by_ID(inst_id, directory=orig_dir)
            tours = [unpack_tour_QUBO_bitstring([s for s in reversed(sol)], len(D))
                     for sol in top_samples]
            feasible = [TSP_is_feasible(tour, D) for tour in tours]
        case "MWC":
            G = load_MWC_by_ID(inst_id, directory=orig_dir)
            feasible = [True for _ in top_samples]

        case "UDMIS":
            G, js = load_UDMIS_by_ID(inst_id, directory=orig_dir)
            # Note that this order of nodes is correct
            # because ``int_samples`` is already reversed above
            # (as compared to ``top_samples``)
            feasible = [is_IS(G, sol) for sol in int_samples]
        case other:
            raise ValueError(f"{inst_id}: wrong instance ID (get_inst_type returned type {get_inst_type(inst_id)})")


    feasible_objs = [QUBO_objectives[i] for i in range(len(QUBO_objectives))
                     if feasible[i]]
    if len(feasible_objs)==0:
        return np.min(QUBO_objectives), False
    else:
        return np.min(feasible_objs), True


def calculate_orig_objective(inst_id, top_samples, orig_dir="./instances/orig"):
    """Helper: Calculates the objective given the instance ID and the collection of bitstrings.

    Uses the logic of the original problem, not the universal QUBO code.

    Args:
        inst_id(str): instance id,
        top_samples(list): best samples to choose from (bitstrings).

    Notes:
        - ``top_samples`` must contain a list of bitstrings, where higher
            ranking qubits are on the left: e.g., ``[b3, b2, b1, b0]``.
    """

    match get_inst_type(inst_id):
        case "TSP":
            D, _ = load_TSP_by_ID(inst_id, directory=orig_dir)
            objs = [obj_val(D, unpack_tour_QUBO_bitstring(
                [s for s in reversed(sol)],
                len(D)))
                        for sol in top_samples if sol is not None]
            objs = [obj for obj in objs if obj is not None]
            if len(objs)>0:
                return min(objs)
            else:
                return None

        case "MWC":
            G = load_MWC_by_ID(inst_id, directory=orig_dir)
            return max([get_MWC_objective(G, [s for s in reversed(sol)])
                           for sol in top_samples])
        case "UDMIS":
            G, _ = load_UDMIS_by_ID(inst_id, directory=orig_dir)
            int_sols = [[int(s) for s in reversed(sol)]
                        for sol in top_samples]
            objs = [sum(int_sol) for int_sol in int_sols
                        if is_IS(G, int_sol)]
            if len(objs)>0:
                return max(objs)
            else:
                return None
        case other:
            raise ValueError(f"{inst_id}: unknown instance type '{get_inst_type(inst_id)}'.")


######################################################################
# Base log parser class
class QPULogParser:
    """Implements the basic JSON log parser interface.

    Attributes:
        files(list): list of processed files (filenames)
        df(pd.DataFrame): the data accumulated
    """

    def __init__(self, files=None, log=True, orig_dir="./instances/orig", qubo_dir="./instances/QUBO"):
        self.files = files
        self.log = log
        self.inst_stats = dict()
        self.logtype="UNDEFINED"
        self.orig_dir = orig_dir
        self.qubo_dir = qubo_dir

    @abstractmethod
    def _extract_successful_line(self, js, filename):
        """Extracts a single successful instance run (from a single JSON)."""
        raise NotImplementedError

    @abstractmethod
    def _extract_failed_line(self, js, filename, statsfile=None):
        """Extracts a single failed instance run (from a single JSON)."""
        raise NotImplementedError

    def save(self, outfile, statsfile=None):
        """Helper: saves the statistics on the processed files.

        This generates ``*_stats.csv`` type of files, which summarize the
        number of successful and failed runs per instance.
        """
        self.df.to_csv(outfile, index=False)
        if self.log:
            print(f"Saved {len(self.df)} entries to {outfile}.")

        if statsfile is not None:
            stats = pd.DataFrame({
                "logtype": [self.logtype for _ in self.inst_stats],
                "instance_id" : [inst_id for inst_id in self.inst_stats],
                "instance_type": [get_inst_type(inst_id) for inst_id in self.inst_stats],
                "qubo_vars" : [get_instance_size_by_ID(inst_id) for inst_id in self.inst_stats],
                "success_runs": [stats[0] for (_, stats) in self.inst_stats.items()],
                "failed_runs": [stats[1] for (_, stats) in self.inst_stats.items()],
            })
            stats.to_csv(statsfile, index=False)
            if self.log:
                print(f"Saved {len(stats)} unique instances stats to {statsfile}.")

    def process_files(self, filenames=None, outfile=None, statsfile=None):
        """Processes the list of JSON raw log files (in ``filenames``).

        A universal high-level procedure, which relies on
        `_extract_successful_line` and `_extract_failed_line` methods
        implemented for each respective log file type (device type).
        """
        if filenames is None:
            filenames = self.files

        failed_instances = []
        notfound_instances = []

        for filename in filenames:
            if self.log:
                print(f"Adding: {filename}", flush=True)


            with open(filename, 'r') as infile:
                js = json.load(infile)

            if 'instance_id' in js:
                inst_id = js['instance_id']
            elif 'problem' in js:
                # this is a QuEra logfile
                # it has slightly different format ( :-/ )
                inst_name = js['problem']["instance_name"]
                with open(self.orig_dir + "/" + inst_name + ".json",'r') as ofile:
                    origjs = json.load(ofile)

                inst_id = origjs['description']['instance_id']
            else:
                raise ValueError(f"{filename}: instance ID not found.")

            if not instance_present_in_folder(inst_id):
                notfound_instances += [inst_id]
                print(f"{inst_id} (file {filename}): not found in 'instances/', skipping.")
                continue
            if inst_id not in self.inst_stats:
                self.inst_stats[inst_id] = [0, 0]

            if 'success' in js:
                if js['success']:
                    self.df.loc[len(self.df)] = self._extract_successful_line(js, filename)
                    self.inst_stats[js['instance_id']][0] += 1
                else:
                    self.df.loc[len(self.df)] = self._extract_failed_line(js, filename)
                    failed_instances.append(filename)
                    self.inst_stats[js['instance_id']][1] += 1
            else:
                # this is a QuEra log file (hopefully)
                if ('qpu_result' in js) and (len(js['qpu_result']['qpu_counts'])>0):
                    self.df.loc[len(self.df)] = self._extract_successful_line(js, filename)
                    self.inst_stats[inst_id][0] += 1
                else:
                    self.df.loc[len(self.df)] = self._extract_failed_line(js, filename)
                    failed_instances.append(filename)
                    self.inst_stats[inst_id][1] += 1

        if self.log:
            print(f"✅ Processed {len(filenames)} json files.")
            if len(failed_instances)>0:
                print(f"❌ Including the following {len(failed_instances)} failed runs:\n" +
                      "\n".join(failed_instances))
            else:
                print("(No failed instances in the list.)")

            if len(notfound_instances)>0:
                print(f"❌ Skipped the following {len(failed_instances)} instances (no data found in 'instances/'):\n" +
                      "\n".join(notfound_instances))
            else:
                print("(All instances in the list were found in 'instances/'.)")

        if outfile is not None:
            self.save(outfile, statsfile)

######################################################################
# QPU-specific parsers

class DWaveLogParser(QPULogParser):
    """Implements the D-Wave specific log parsing code."""

    def __init__(self, files=None, log=True):
        super().__init__(files, log)
        self.logtype="DWave"
        self.df = pd.DataFrame(columns = ["filename", "start_timestamp",
                                          "end_timestamp", "instance_id",
                                          "instance_type", "qubo_vars", "sol_time",
                                          "obj_from_QPU_sol", "success",
                                          "solver_name", "chip_id", "chip_type",
                                          "topology", "num_reads",
                                          "binary_vars", "emb_qubits",
                                          "embedding_time"])

    def _extract_successful_line(self, js, filename):
        """Extracts a summary for a single DWave's log.

        This function extracts a "successful" summary line: the one corresponding
        to an experiment that yielded some solutions (feasible or not).
        """
        n_log_qubits = 0
        n_act_qubits = 0
        embedding = js['solver']['outcome']['emb_properties']['embedding']
        for qubit in embedding:
            n_log_qubits += 1
            n_act_qubits += len(embedding[qubit])

        samples = DWaveLogParser.extract_samples(js)
        assert samples is not None

        best_sols = samples["solution"]  # choose the best from *all* samples

        return [filename,
                js['timestamps']['start'],
                js['timestamps']['end'],
                js['instance_id'],
                get_inst_type(js['instance_id']),
                get_instance_size_by_ID(js['instance_id']),
                (js['timestamps']['end'] - js['timestamps']['start']),
                calculate_orig_objective(js["instance_id"], best_sols, orig_dir=self.orig_dir),
                js["success"],
                js['solver_name'],
                js["solver"]["outcome"]["emb_properties"]["child_properties"]["chip_id"],
                js["solver"]["outcome"]["emb_properties"]["child_properties"]["category"],
                js["solver"]["outcome"]["emb_properties"]["child_properties"]["topology"]["type"],
                js['solver']['options']['num_reads'],
                n_log_qubits,
                n_act_qubits,
                js['solver']['outcome']['emb_precalc_dt']]

    def _extract_failed_line(self, js, filename):
        """Extracts a summary for a single D-Wave's log.

        This function extracts a "failed" summary line: the one corresponding
        to an experiment that yielded no solutions. (A separate function
        is needed as some fields might be absent from such logfile,
        as compared to a "successful" one.)
        """
        return [filename,
                js['timestamps']['start'],
                js['timestamps']['end'],
                js['instance_id'],
                get_inst_type(js['instance_id']),
                get_instance_size_by_ID(js['instance_id']),
                (js['timestamps']['end'] - js['timestamps']['start']),
                pd.NA,
                False,
                js['solver_name'],
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA]

    @staticmethod
    def extract_samples(js, logfile=None, qubo_dir="./instances/QUBO", orig_dir="./instances/orig"):
        """Extracts the sample data from JSON (DWave).

        Args:
            js: loaded JSON,
            logfile: log file name (to save into the output)
            qubo_dir, orig_dir(str): respective JSON files directories
                    (for objective calculations)

        Notes:
            - resulting solution bitstrings are in **reverse** order:
              e.g., ``[b3, b2, b1, b0]``.

        Returns:
            a ``pd.DataFrame`` with the samples, or ``None`` if corresponding
            JSON files are not found in either of ``qubo_dir`` or ``orig_dir``.
        """
        sols = []
        energies = []
        true_objs = []  # values calculated as QUBO values (incl. penalties)
        num_occs = []
        chain_breaks = []
        true_objs = []
        feasibility=[]
        orig_objs = []  # values calculated using the original instance-specific code

        inst_id = js["instance_id"]

        if not instance_present_in_folder(inst_id):
            print(f"{inst_id} ({logfile}): not found in '{qubo_dir}' or '{orig_dir}', skipping.")
            return None

        for sample in js["solver"]["outcome"]["samples"]:
            s = sample['sample']  # sample data (solution)
            assert [str(x) for x in s] == [str(j) for j in range(len(s))]
            # if sample['chain_break_fraction']>0:
            #     continue
            bitstring = "".join([str(s[str(x)]) for x in range(len(s)-1, -1,-1)])
            sols.append(bitstring)
            energies.append(sample["energy"])
            num_occs.append(int(sample["num_occurrences"]))
            chain_breaks.append(float(sample["chain_break_fraction"]))

            obj, feasible = calculate_QUBO_objective(inst_id, [bitstring],
                                                     orig_dir=orig_dir,
                                                     qubo_dir=qubo_dir)
            true_objs.append(obj)
            feasibility.append(feasible)

            if feasible:
                orig_obj = calculate_orig_objective(inst_id, [bitstring],
                                                    orig_dir=orig_dir)
            else:
                orig_obj = None

            orig_objs.append(orig_obj)


        inst_ids = [inst_id for _ in range(len(sols))]
        inst_type = [get_inst_type(inst_id) for _ in range(len(sols))]
        inst_size = [get_instance_size_by_ID(inst_id, folder=qubo_dir) for _ in range(len(sols))]
        logfiles = [logfile for _ in range(len(sols))]
        return pd.DataFrame(zip(logfiles, inst_ids, inst_type, inst_size, sols,
                                feasibility, energies, num_occs, chain_breaks,
                                true_objs, orig_objs),
                            columns=['logfile', 'inst_id', 'inst_type',
                                     'inst_size','solution', 'sol_feasible',
                                     'energy', 'no_occ', "chain_break_frac",
                                     "objective", "orig_obj"])


class IBMLogParser (QPULogParser):
    """Implements the IBM specific log parsing code."""
    def __init__(self, files=None, log=True, logtype="IBM"):
        super().__init__(files, log)
        self.logtype=logtype
        self.df = pd.DataFrame(columns = ["logfile",
                                          "start_timestamp",
                                          "end_timestamp",
                                          "instance_id",
                                          "instance_type",
                                          "qubo_vars",
                                          "success",
                                          "sol_time",
                                          "obj_from_QPU_sol",
                                          "classic_solver",
                                          "classic_solver_params",
                                          "backend_name",
                                          "sampler_shots",
                                          "est_shots"])

    def _extract_successful_line(self, js, filename):
        """Extracts a single instance run, parsing a single IBM's log.

        This function extract a "successful" line: the one corresponding
        to an experiment that yielded some solutions (feasible or not).
        """
        samples = IBMLogParser.extract_samples(js)
        if samples is not None:
            best_sols = samples["solution"]  # choose the best from *all* samples

            return [filename,
                    js['timestamps']['start'],
                    js['timestamps']['end'],
                    js['instance_id'],
                    get_inst_type(js['instance_id']),
                    get_instance_size_by_ID(js['instance_id']),
                    js["success"],
                    (js['timestamps']['end'] - js['timestamps']['start']),
                    calculate_orig_objective(js["instance_id"], best_sols, orig_dir=self.orig_dir),
                    js['args'][1],
                    js['solver']['options']['optimizer_kwargs'],
                    js['solver']['options']['backend_name'],
                    js['solver']['options']['sampler_shots'],
                    js['solver']['options']['estimator_shots']]
        else:
            # not quite successful run:
            # This is a case when the sample contains no data for some
            # reason. Still, some more information is available,
            # as compared to the case processed by
            # :py:func:`IBMLogParser._extract_failed_line`
            return [filename,
                    js['timestamps']['start'],
                    js['timestamps']['end'],
                    js['instance_id'],
                    get_inst_type(js['instance_id']),
                    get_instance_size_by_ID(js['instance_id']),
                    "NO_SOLS_LOGGED",
                    (js['timestamps']['end'] - js['timestamps']['start']),
                    pd.NA,
                    js['args'][1],
                    js['solver']['options']['optimizer_kwargs'],
                    js['solver']['options']['backend_name'],
                    js['solver']['options']['sampler_shots'],
                    js['solver']['options']['estimator_shots']]


    def _extract_failed_line(self, js, filename):
        """Extracts a single instance run, parsing a single IBM log.

        This function extract a "failed" summary line: the one corresponding to
        an experiment that yielded no solutions. (A separate function is needed
        as some fields might be absent from such logfile, as compared to a
        "successful" one.)
        """
        return [filename,
                js['timestamps']['start'],
                js['timestamps']['end'],
                js['instance_id'],
                get_inst_type(js['instance_id']),
                get_instance_size_by_ID(js['instance_id']),
                False,
                (js['timestamps']['end'] - js['timestamps']['start']),
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA]

    @staticmethod
    def extract_samples(js, logfile=None, qubo_dir="./instances/QUBO",
                        orig_dir="./instances/orig"):
        """Extracts the sample data from JSON (IBM).

        Args:
            js: loaded JSON,
            logfile: log file name (to save into the output)
            logfile(str): analyzed logfile name,
            qubo_dir, orig_dir(str): respective JSON files directories
                    (for objective calculations)

        Returns:
            a ``pd.DataFrame`` with the samples.
        """
        if not js["success"]:
            return None

        jobs = js["solver"]["outcome"]["session_data"]["job_data"]
        quasi_dists_jobs = 0

        samples = dict()
        for jobname, job in jobs.items():
            if job["quasi_dists"] is not None:
                assert len(job["quasi_dists"]) == 1
                assert len(job["quasi_dists"][0]) == 1
                nshots = job["metadata"][0]["shots"]
                for sol, freq in job["quasi_dists"][0][0].items():
                    if sol not in samples:
                        samples[sol] = 0

                    samples[sol] += freq*nshots


        if len(samples) == 0:
            return None  # couldn't find a solution in the log

        sols = []
        no_occ = []
        true_objs = []
        orig_objs = []
        feasibility = []
        inst_id = js["instance_id"]
        N = get_instance_size_by_ID(inst_id, folder=qubo_dir)
        for sample in samples:
            bitstring = ("{:0" + str(N) + "b}").format(int(sample))
            sols.append(bitstring)
            no_occ.append(samples[sample])

            obj, feasible = calculate_QUBO_objective(inst_id, [bitstring],
                                               orig_dir=orig_dir,
                                               qubo_dir=qubo_dir)
            true_objs.append(obj)
            feasibility.append(feasible)

            if feasible:
                orig_obj = calculate_orig_objective(inst_id, [bitstring],
                                               orig_dir=orig_dir)
            else:
                orig_obj = None

            orig_objs.append(orig_obj)

        IDs = [js["instance_id"] for _ in range(len(orig_objs))]

        return pd.DataFrame(zip(IDs, sols, feasibility, no_occ, true_objs,
                                orig_objs),
                            columns=['inst_id', 'solution', 'sol_feasible',
                                     'no_occ', 'objective', 'orig_obj'])

    @staticmethod
    def extract_convergence_data(js, logfile=None):
        """Extracts the outer loop convergence data from an IBM log.

        Args:
            js: loaded JSON,
            logfile: log file name (to save into the output)

        Returns:
            a ``pd.DataFrame`` with the samples.
        """
        # number of classic iterations
        if not js['success']:
            return None

        history = js['solver']['outcome']['solver_history']
        K = len(history['params'])

        params = []
        costs = []
        variances = []
        shots = []
        timestamps = []

        convdf = pd.DataFrame(columns=["logfile", "instance_id", "iteration",
                                       "variable", "value"])

        for k in range(K):
            row = [logfile, js["instance_id"], k]
            convdf.loc[len(convdf)] = row + ["cost", history['cost'][k]]
            convdf.loc[len(convdf)] = row + ["timestamp", history['timestamp'][k]]

            params = history['params'][k]
            for i,param in enumerate([float(x) for x in params[1:-1].split()]):
                convdf.loc[len(convdf)] = row + [f"param{i}", param]

            if len(history['metadata'][k]) > 1:
                for mdvar in history['metadata'][k]:
                    convdf.loc[len(convdf)] = row + [mdvar, history['metadata'][k][mdvar]]

        return convdf


class QuEraLogParser (QPULogParser):
    """Implements the QuEra specific log parsing code."""
    def __init__(self, files=None, log=True):
        super().__init__(files, log)
        self.logtype="QuEra"
        self.df = pd.DataFrame(columns = ["logfile",
                                          "start_timestamp",
                                          "end_timestamp",
                                          "instance_id",
                                          "instance_type",
                                          "qubo_vars",
                                          "R",
                                          "version",
                                          "success",
                                          "sol_time",
                                          "obj_from_QPU_sol"])

    def _extract_successful_line(self, js, filename):
        """Extracts a single instance run, parsing a single QuEra log.

        """
        samples = QuEraLogParser.extract_samples(js)
        inst_name = js['problem']["instance_name"]
        with open(self.orig_dir + "/" + inst_name + ".json",'r') as ofile:
            origjs = json.load(ofile)
        if samples is not None:
            best_sols = samples["solution"]  # choose the best from *all* samples
            inst_id = origjs['description']['instance_id']
            return [filename,
                    js['meta']['timestamp_start'],
                    js['meta']['timestamp_end'],
                    inst_id,
                    get_inst_type(inst_id),
                    get_instance_size_by_ID(inst_id),
                    js['settings']['R'],
                    js['meta']['version'],
                    True,
                    (js['meta']['timestamp_end'] - js['meta']['timestamp_start']),
                    calculate_orig_objective(inst_id, best_sols, orig_dir=self.orig_dir)]
        else:
            # no samples recorder (not really successful run)
            return [filename,
                    js['meta']['timestamp_start'],
                    js['meta']['timestamp_end'],
                    inst_id,
                    get_inst_type(inst_id),
                    get_instance_size_by_ID(inst_id),
                    js['settings']['R'],
                    "NO_SOLS_LOGGED",
                    (js['meta']['timestamp_end'] - js['meta']['timestamp_start']),
                    pd.NA]

    def _extract_failed_line(self, js, filename):
        """This function is not implemented, we never received such result."""
        # We expect it to never happen for QuEra
        raise NotImplementedError("QuEra: found an unsuccessful item / something went wrong before.")

    @staticmethod
    def extract_samples(js, logfile=None, qubo_dir="./instances/QUBO/",
                        orig_dir="./instances/orig/"):
        """Extracts the sample data from JSON (QuEra logfile).

        Args:
            js: loaded JSON,
            logfile: log file name (to save into the output)
            qubo_dir, orig_dir(str): respective JSON files directories
                    (for objective calculations)

        Note:
            The method assumes (UD)MIS instances only!

        Returns:
            a ``pd.DataFrame`` with the samples.
        """
        sols = []
        counts = []
        feasibility = []
        objs = []
        orig_objs = []
        inst_ids = []

        inst_name = js['problem']["instance_name"]
        with open(orig_dir + "/" + inst_name + ".json",'r') as ofile:
            origjs = json.load(ofile)

        inst_id = origjs['description']['instance_id']
        Q, P, C, qubojs = load_QUBO(get_QUBO_by_ID(inst_id))
        G = extract_G_from_json(origjs)

        for sample in js['qpu_result']['qpu_counts']:
            bs = {'g':'0', 'r':'1'}
            bitstring = [bs[s] for s in reversed(sample)]
            QUBO_obj, fflag = calculate_QUBO_objective(inst_id, [bitstring])

            inst_ids.append(inst_id)
            objs.append(QUBO_obj)
            feasibility.append(fflag)
            sols.append(bitstring)
            counts.append(js['qpu_result']['qpu_counts'][sample])
            orig_objs.append(calculate_orig_objective(inst_id, [bitstring]))


        return pd.DataFrame(zip(inst_ids, sols, feasibility, counts,
                                objs, orig_objs),
                            columns = ['inst_id', 'solution', 'sol_feasible',
                                    'no_occ', 'objective', 'orig_obj'])


######################################################################
# Script commands (to be run from the command line, see USAGE)

def summarize_dwave(directory, outfile, statsfile):
    """Processes all the DWave logs in the given directory."""
    parser = DWaveLogParser(files=glob(directory + "/*.json"))
    parser.process_files(outfile=outfile, statsfile=statsfile)

def summarize_ibm(directory, outfile, statsfile, logtype="IBM"):
    """Processes all the IBM logs in the given directory."""
    parser = IBMLogParser(files=glob(directory + "/*.json"), logtype=logtype)
    parser.process_files(outfile=outfile, statsfile=statsfile)

def summarize_quera(directory, outfile, statsfile):
    """Processes all the QuEra logs in the given directory."""
    parser = QuEraLogParser(files=glob(directory + "/*.json"))
    parser.process_files(outfile=outfile, statsfile=statsfile)


def extract_samples(logtype, infile, outfile,
                    qubo_dir="./instances/QUBO", orig_dir="./instances/orig"):
    """Extracts the last sample from the JSON log."""
    with open(infile, 'r') as infilehandle:
        js = json.load(infilehandle)

    ParserClass = {"dwave": DWaveLogParser, "ibm": IBMLogParser,
                   "quera": QuEraLogParser}
    parser = ParserClass[logtype.lower()](files = [infile])
    samples = parser.extract_samples(js, logfile=str(infile), qubo_dir=qubo_dir, orig_dir=orig_dir)
    if samples is None:
        print(f"{infile}: unsuccessful run, skipping.")
        exit(1)
    else:
        samples.to_csv(outfile)

def extract_all_samples(logtype, jsondir, outdir):
    """Extracts the last sample from the JSONs in `jsondir`."""
    for infile in glob(jsondir + "/benchmark_*.json"):
        print(f"Processing {infile}...", end="", flush=True)

        if not instance_present_in_folder(inst_id):
            notfound_instances += [inst_id]
            print(f"{inst_id} (file {filename}): not found in 'instances/', skipping.")
            continue

        with open(infile, 'r') as infilehandle:
            js = json.load(infilehandle)

        if not js["success"]:
            print(f"skipped, success={js['success']}", flush=True)
            continue

        outfile = outdir + "/" + os.path.basename(os.path.normpath(infile)) + \
            '.sample.csv'
        ParserClass = {"dwave": DWaveLogParser, "ibm": IBMLogParser,
                    "quantinuum": QuantinuumLogParser}
        parser = ParserClass[logtype.lower()](files = [infile])
        samples = parser.extract_samples(js, logfile=str(infile))
        samples.to_csv(outfile)
        print(f"done", flush=True)


if __name__ == '__main__':
    # treats command line arguments as function name and further args
    args = argv
    globals()[args[1]](*args[2:])
