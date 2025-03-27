import datetime
import os
import warnings
from pathlib import Path
from typing import NoReturn

_project_root_folder_name: str = 'chewing_sensor_study'
_global_experiment_name: str = 'untitled_experiment_'
_untitled_experiment_name: str = 'untitled_experiment_'


def _get_experiment_name() -> str:
    global _global_experiment_name

    if _global_experiment_name is _untitled_experiment_name:
        _global_experiment_name = _global_experiment_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        warnings.warn(
            'Module has no experiment name configured.\nAssigning the following name: ' + _global_experiment_name)

    return _global_experiment_name


def set_experiment_name(experiment_name: str, is_file_name: bool = False) -> NoReturn:
    assert isinstance(experiment_name, str)
    assert isinstance(is_file_name, bool)

    global _global_experiment_name
    if is_file_name:
        file_name = Path(experiment_name)
        _global_experiment_name = file_name.name[0:-len(file_name.suffix)]
    else:
        _global_experiment_name = experiment_name


def get_root_path() -> Path:
    cwd: Path = Path(os.getcwd())

    while cwd.name is not _project_root_folder_name:
        cwd = cwd.parent

    return cwd


def get_generated_path() -> Path:
    return get_root_path() / 'generated'


def get_results_path() -> Path:
    x = get_root_path() / 'results' / _get_experiment_name()
    x.mkdir(parents=True, exist_ok=True)

    return x


def get_models_path() -> Path:
    x = get_results_path() / 'models'
    x.mkdir(parents=True, exist_ok=True)

    return x


def get_best_model_path(lopo_idx: int = -1) -> Path:
    models_path: Path = get_models_path()
    if lopo_idx == -1:
        model_name = "best.h5"
    else:
        model_name = "best_" + str(lopo_idx) + ".h5"

    return models_path / model_name


def get_tensorboard_logdir(lopo_idx: int = -1) -> Path:
    assert isinstance(lopo_idx, int)

    x: Path = get_results_path() / "tensorboard"
    if lopo_idx != -1:
        x = x / ("LOPO_" + str(lopo_idx))
    x.mkdir(parents=True, exist_ok=True)

    return x


def set_tensorflow_log_level(n: int = 3) -> NoReturn:
    assert isinstance(n, int)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(n)


def enable_tensorflow_eager_execution(enabled: bool = True) -> NoReturn:
    assert isinstance(enabled, bool)

    from tensorflow.python.framework.ops import enable_eager_execution, disable_eager_execution
    if enabled:
        enable_eager_execution()
    else:
        disable_eager_execution()


def get_res_main() -> Path:
    return get_root_path() / 'res' / 'main'


def get_res_test() -> Path:
    return get_root_path() / 'res' / 'test'


def get_src_main() -> Path:
    return get_root_path() / 'src' / 'main'


def get_src_test() -> Path:
    return get_root_path() / 'src' / 'test'


def get_wu2_path(machine_name: str = "") -> Path:
    assert isinstance(machine_name, str)

    if machine_name == "mahakam":
        return Path("~/workspace/Datasets/chewing_sensor_study/wu2/Collected signals.min/")
    else:
        return Path("~/workspace/Datasets/chewing_sensor_study/wu2/Collected signals.min/")


def print_summary() -> NoReturn:
    print('root       : ' + str(get_root_path()))
    print('results    : ' + str(get_results_path()))
    print('models     : ' + str(get_models_path()))
    print('tensorboard: ' + str(get_tensorboard_logdir()))
    print('res/main   : ' + str(get_res_main()))
    print('res/test   : ' + str(get_res_test()))
    print('src/main   : ' + str(get_src_main()))
    print('src/test   : ' + str(get_src_test()))


if __name__ == '__main__':
    enable_tensorflow_eager_execution(True)
    print("globalconfig: enabled TensorFlow eager execution mode")

    print_summary()
