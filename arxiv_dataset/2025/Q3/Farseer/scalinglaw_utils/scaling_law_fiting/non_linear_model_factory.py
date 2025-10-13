
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Type
from scalinglaw_utils.scaling_law_fiting import non_linear_model
from scalinglaw_utils.scaling_law_fiting.non_linear_model import NonLinearFit

class NonLinearModelFactory():
    '\n    Factory for creating instances of nonlinear fitting models. Provides\n    both single instance creation and batch creation from a DataFrame.\n\n    The default and returned instances are of type NonLinearFit.\n    '

    def __init__(self, model_module: Optional[Any]=non_linear_model):
        "\n        :param model_module: The module that contains the model classes.\n                             This should be a module object (obtained via\n                             an import statement), not a filename.\n                             If None, defaults to the current module's globals().\n        "
        if (model_module is None):
            self.model_module = globals()
        else:
            self.model_module = model_module

    def _get_model_class(self, model_class_name: str) -> Optional[Type[NonLinearFit]]:
        '\n        Retrieve the model class based on its name using reflection.\n\n        :param model_class_name: Name of the model class (string)\n        :return: The model class if found; otherwise, None.\n        '
        if isinstance(self.model_module, dict):
            return self.model_module.get(model_class_name)
        else:
            return getattr(self.model_module, model_class_name, None)

    def create_instance(self, model_class_name: str, params_dict: dict, decimal_mode: bool=False) -> Optional[NonLinearFit]:
        '\n        Create a model instance based on the model class name and a dictionary of parameters.\n\n        :param model_class_name: Name of the model class (string)\n        :param params_dict: A dictionary of parameters required for the model.\n        :return: An instance of NonLinearFit if successful; otherwise, None.\n        '
        model_class = self._get_model_class(model_class_name)
        if (model_class is None):
            print(f"Warning: Could not find model class '{model_class_name}'.")
            return None
        try:
            instance = model_class(params_dict=params_dict, decimal_mode=decimal_mode)
        except Exception as e:
            print(f"Error: Exception occurred while instantiating model '{model_class_name}': {e}")
            return None
        if (not isinstance(instance, NonLinearFit)):
            print('Warning: The created instance is not a NonLinearFit.')
            return None
        return instance

    def create_instances_from_dataframe(self, df: pd.DataFrame, model_class_col: str='model_class', model_name_col: str='model_name', required_params: Optional[list]=None, decimal_mode=False) -> Dict[(str, NonLinearFit)]:
        '\n        Batch create model instances from a DataFrame and return a dictionary.\n        The keys are taken from the DataFrame\'s model_name column and the values\n        are the corresponding model instances.\n\n        The DataFrame must contain at least:\n          - A column for the model class name (default \'model_class\')\n          - A column for the model name (default \'model_name\')\n          - Other columns corresponding to the parameters required by the model.\n\n        :param df: DataFrame containing model information.\n        :param model_class_col: Column name for the model class name (default "model_class").\n        :param model_name_col: Column name for the model name (default "model_name").\n        :param required_params: List of required parameter names. If None, defaults to\n                                ["E", "s", "q", "S", "B", "b", "Q", "A", "a"].\n        :return: Dictionary where key is the model name and value is an instance of NonLinearFit.\n        '
        models: Dict[(str, NonLinearFit)] = {}
        if (required_params is None):
            required_params = ['E', 's', 'q', 'S', 'B', 'b', 'Q', 'A', 'a']
        if (model_name_col not in df.columns):
            raise ValueError(f"DataFrame must contain column '{model_name_col}'.")
        if (model_class_col not in df.columns):
            raise ValueError(f"DataFrame must contain column '{model_class_col}'.")
        for (index, row) in df.iterrows():
            model_class_name = row[model_class_col]
            model_name = row[model_name_col]
            params = {}
            missing_params = []
            for param in required_params:
                if ((param in row) and (not pd.isna(row[param]))):
                    params[param] = row[param]
                else:
                    missing_params.append(param)
            if missing_params:
                print(f"Warning: Model '{model_name}' is missing required parameters {missing_params}, skipping.")
                continue
            instance = self.create_instance(model_class_name, params, decimal_mode)
            if (instance is not None):
                models[model_name] = instance
        return models

def use_case_single():
    factory = NonLinearModelFactory()
    single_model = factory.create_instance('ExpExpPowFit', params_dict={'E': 0, 'q': 0.034, 'Q': (- 1.221218470265441), 's': (- 1.7048291345782254), 'S': 2.649367272335889, 'B': 255.5363422599814, 'b': (- 0.18030000000001317), 'A': (- 17.405595115941324), 'a': (- 0.19485496750920472)})
if (__name__ == '__main__'):
    use_case_single()
