# V1 no safety guard
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Tuple
from soar.llm_utils import extract_transform

# import scipy
Result = Tuple[str, List[bool]]

import os

from multiprocessing import Pool
import numpy as np

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

Timeout_sandbox = 3 #s
Memory_limit = 1 #GB
import builtins
import importlib


try:
    import resource
    RESOURCE_MODULE_AVAILABLE = True
except ImportError:
    RESOURCE_MODULE_AVAILABLE = False


def limit_memory(max_memory):
    if RESOURCE_MODULE_AVAILABLE:
        # Add some buffer for scipy's internal operations
        actual_limit = max_memory + (512 * 1024 * 1024)  # Add 256MB buffer
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (actual_limit, hard))

class Sandbox:
    """A sandbox for executing code with limited imports and memory usage."""
    def __init__(self, banned_modules=None, timeout=5, max_memory=1 * 1000 * 1024 * 1024):
        self.banned_modules = banned_modules or ['os', 'sys', 'subprocess', 'shutil', 'socket']
        self.timeout = timeout
        self.max_memory = max_memory
        self.globals = {'__builtins__': {}}
        try:
            import scipy
            import scipy.ndimage
            self.globals['scipy'] = scipy
            self.globals['ndimage'] = scipy.ndimage
        except ImportError:
            pass



        for name in dir(builtins):
            if name not in ['eval', 'exec', 'open']:
                self.globals['__builtins__'][name] = getattr(builtins, name)
        self.globals['__import__'] = self.safe_import

    def safe_import(self, name, *args, **kwargs):
        if name in self.banned_modules:
            raise ImportError(f"Import of '{name}' is not allowed for security reasons")
        if name == 'scipy.ndimage':
            import scipy.ndimage
            return scipy.ndimage
        elif name == 'scipy':
            import scipy

        return importlib.import_module(name)
        
    @staticmethod
    def _initialize_worker():
        """Initialize the worker process with required modules"""
        try:
            import numpy as np
            globals()['np'] = np
            
            import scipy
            globals()['scipy'] = scipy
            
            # Pre-import other commonly needed modules
            import math
            globals()['math'] = math
            
        except ImportError as e:
            print(f"Warning: Failed to initialize some modules: {e}")

    @staticmethod
    def _run_with_memory_limit(code, dict_io_train_test, max_memory, entry_point='transform'):
        limit_memory(max_memory)

        error_message = "Unknown error"
        try:
            exec(code, globals())
            fn = globals()[entry_point]

            for key, values in dict_io_train_test.items():
                list_input_matrix = values["inputs"]
                for id_input, input_matrix in enumerate(list_input_matrix):
                    try:
                        result = fn(input_matrix)
                        if isinstance(result, np.ndarray):
                            result = result.tolist()
                        dict_io_train_test[key]["prediction"][id_input] = result
                        
                        try:
                            if isinstance(dict_io_train_test[key]["outputs"][id_input], np.ndarray):
                                dict_io_train_test[key]["outputs"][id_input] = dict_io_train_test[key]["outputs"][id_input].tolist()
                            if result == dict_io_train_test[key]["outputs"][id_input]:
                                dict_io_train_test[key]["list_correct"][id_input] = True
                        except Exception as e:
                            pass
                    except Exception as e:
                        error_message = f"ERROR while executing the code, Error: {str(e)}"
                        dict_io_train_test[key]["prediction"][id_input] = "ERROR: while executing the code, Error: " + str(e)

        except MemoryError:
            error_message = f"Error: Memory limit of {max_memory / (1024 * 1024):.2f} MB exceeded"
        except Exception as e:
            error_message = f"Error: {str(e)}"

        for key, values in dict_io_train_test.items():
            list_input_matrix = values["inputs"]
            for id_input, input_matrix in enumerate(list_input_matrix):
                if dict_io_train_test[key]["prediction"][id_input] is None:
                    dict_io_train_test[key]["prediction"][id_input] = error_message

        return dict_io_train_test

    def run(self, code, dict_io_train_test, timeout=None, entry_point='transform'):
        if timeout is None:
            timeout = self.timeout

        with Pool(processes=1, initializer=self._initialize_worker) as pool:
            try:
                result = pool.apply_async(self._run_with_memory_limit, 
                                        (code, dict_io_train_test, self.max_memory, entry_point))
                res = result.get(timeout=timeout)
                return res
            except TimeoutError:
                pass
                # print(f"Error: Code execution timed out after {timeout} seconds")
            except Exception as e:
                pass
                # print(f"Error during code execution: {str(e)}")
            
            # If we reach here, an error occurred
            return {key: {"prediction": [f"Error: Code execution failed"] * len(values["inputs"]),
                        "list_correct": [False] * len(values["inputs"])}
                    for key, values in dict_io_train_test.items()}
        



def check_solutions(dict_response, dataset,max_workers=None,timeout=5, max_memory= 4, keep_only_correct_results=False):
    """ check if code generate the correct output for the 'train' input and 'test' input"""
    list_keys = list(dict_response.keys())
    if max_workers is None:
        max_workers = min(os.cpu_count(),int(2*os.cpu_count()/5)) 
    results = []
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        for key in list_keys:
            data_train_input = dataset[key]["train"]
            list_train_input = [data["input"] for data in data_train_input]
            #TODO: do something if output is not available
            list_train_output = [data["output"] for data in data_train_input]

            train_data_test_input = dataset[key]["test"]
            # print(len(train_data_test_input))
            list_test_input = [data["input"] for data in train_data_test_input]
            #TODO: do something if output is not available
            list_test_output = [data["output"] for data in train_data_test_input]
            
            for j in range(len(dict_response[key])):
                # clean_code = keep_functions_and_imports(code)
                if "code" in dict_response[key][j]:
                    code = dict_response[key][j]["code"]
                else:
                    code = extract_transform(dict_response[key][j]["transform"])

                futures.append(executor.submit(
                    check_solution_batch, 
                    list_train_input, list_train_output, 
                    list_test_input, list_test_output, 
                    code, key, j
                ))

        for future in as_completed(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except TimeoutError:
                pass
                # print(f"Task timed out after {timeout} seconds")
            except Exception as e:
                pass
                # print(f"Task raised an exception: {str(e)}")

    for result in results:
        key, j, train_results, test_results = result
        if keep_only_correct_results:
            train_results = {k: v for k, v in train_results.items() if k.startswith("correct")}
            test_results = {k: v for k, v in test_results.items() if k.startswith("correct")}
        dict_response[key][j].update(train_results)
        dict_response[key][j].update(test_results)

    return dict_response



def check_solution_batch(train_inputs, train_outputs, test_inputs, test_outputs, code, key, j):
    dict_io_train_test = {"train": {"inputs":train_inputs, "outputs":train_outputs, "prediction": [None for _ in range(len(train_inputs))], "list_correct": [False for _ in range(len(train_inputs))]},
                           "test": {"inputs": test_inputs, "outputs": test_outputs, "prediction": [None for _ in range(len(test_outputs))], "list_correct": [False for _ in range(len(test_outputs))] }
                           }
    train_results, test_results = check_solution_set(dict_io_train_test, code)
    # test_results = check_solution_set(test_inputs, test_outputs, code, train=False)

    return key, j, train_results, test_results

def check_solution_batch_repair(train_inputs, train_outputs, test_inputs, test_outputs, code, key, j, idx_repair):
    dict_io_train_test = {"train": {"inputs":train_inputs, "outputs":train_outputs, "prediction": [None for _ in range(len(train_inputs))], "list_correct": [False for _ in range(len(train_inputs))]},
                           "test": {"inputs": test_inputs, "outputs": test_outputs, "prediction": [None for _ in range(len(test_outputs))], "list_correct": [False for _ in range(len(test_outputs))] }
                           }
    train_results, test_results = check_solution_set(dict_io_train_test, code)
    # test_results = check_solution_set(test_inputs, test_outputs, code, train=False)

    return key, j, train_results, test_results, idx_repair


def check_solution_set(dict_io_train_test, code):
    """
    results for train or test set
    results: 
    results: dict = {"correct_train_input": list[bool], "predicted_train_output": list[str or list] }
    """
    # key_correct_input = "correct_train_input" if train else "correct_test_input"
    # key_predicted_output = "predicted_train_output" if train else "predicted_test_output"
    # results = {
    #     key_correct_input: [],
    #     key_predicted_output: []
    # }
    count_matrix = 0
    for _, value in dict_io_train_test.items():
        count_matrix += len(value["inputs"])
    timeout = 1 * count_matrix # 2 seconds per matrix
    timeout = max(2, timeout)
    timeout = min(20, timeout)
    sandbox = Sandbox(timeout=Timeout_sandbox, max_memory= Memory_limit * 1000 * 1024 * 1024)

    dict_io_train_test_results = sandbox.run(code, dict_io_train_test, timeout=timeout)
    # print(dict_io_train_test_results)
    (correct_train_input, predicted_train_output) = process_result_set(dict_io_train_test_results["train"])
    (correct_test_input, predicted_test_output) = process_result_set(dict_io_train_test_results["test"])
    train_results = {
        "correct_train_input": correct_train_input,
        "predicted_train_output": remove_circular_refs(predicted_train_output)
    }
    test_results = {
        "correct_test_input": correct_test_input,
        "predicted_test_output": remove_circular_refs(predicted_test_output)
    }
    del sandbox
    return train_results, test_results

def remove_circular_refs(ob, _seen=None):
    """weird stuff happening sometimes with circular references in predicted outptut so need to get rid of that"""
    from builtins import id

    if _seen is None:
        _seen = set()
    if id(ob) in _seen:
        # circular reference, remove it.
        return None
    _seen.add(id(ob))
    res = ob
    if isinstance(ob, dict):
        res = {
            remove_circular_refs(k, _seen): remove_circular_refs(v, _seen)
            for k, v in ob.items()}
    elif isinstance(ob, (list, tuple, set, frozenset)):
        res = type(ob)(remove_circular_refs(v, _seen) for v in ob)
    # remove id again; only *nested* references count
    _seen.remove(id(ob))
    return res

def process_result_set(dict_io_train_test_results):
    correct_input = dict_io_train_test_results["list_correct"]
    predicted_output = dict_io_train_test_results["prediction"]
    return correct_input, predicted_output



